#!/usr/bin/env python3
"""Real-model recurrent-vs-DPLR MLX prefill parity benchmark."""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.run_qwen35_apple_baseline import append_jsonl, device_info, make_prompt, parse_csv, parse_int_csv

AXIS = "mlx_dplr_model_prefill"


def _state_arrays(state: Any) -> list[Any]:
    return [state.v_first, *state.recurrent_state, *state.attn_x_prev, *state.ffn_x_prev]


def _max_abs(mx: Any, left: Any, right: Any) -> float:
    return float(mx.max(mx.abs(left.astype(mx.float32) - right.astype(mx.float32))))


def _state_max_abs(mx: Any, reference: Any, candidate: Any) -> float:
    return max(
        _max_abs(mx, left, right)
        for left, right in zip(_state_arrays(reference), _state_arrays(candidate), strict=True)
    )


def _decode_tokens(mx: Any, model: Any, logits: Any, state: Any, count: int) -> list[int]:
    generated: list[int] = []
    for _ in range(int(count)):
        token = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)
        mx.eval(token)
        generated.extend(int(value) for value in token.tolist())
        logits, state = model.decode_step(token, state)
        mx.eval(logits)
    return generated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True)
    parser.add_argument("--prompt-target-chars", type=int, default=512)
    parser.add_argument(
        "--prompt-seed",
        default="User: Explain recurrent language-model prefill, state caching, and chunk parallelism. Assistant: ",
    )
    parser.add_argument("--dplr-chunk-sizes", default="64")
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--decode-tokens", type=int, default=8)
    parser.add_argument("--prefill-eval-interval", type=int, default=2)
    parser.add_argument("--dtype", default="fp16", choices=["keep", "fp32", "fp16", "bf16"])
    parser.add_argument("--quantization", default="none", choices=["none", "mm8", "mm4"])
    parser.add_argument("--quant-min-params", type=int, default=4_000_000)
    parser.add_argument("--quant-backend", default="auto", choices=["reference", "affine", "metal", "auto"])
    parser.add_argument("--wkv-backend", default="metal", choices=["reference", "metal", "auto"])
    parser.add_argument("--logits-atol", type=float, default=0.25)
    parser.add_argument("--state-atol", type=float, default=0.15)
    parser.add_argument("--results", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    models = parse_csv(args.models)
    chunk_sizes = parse_int_csv(args.dplr_chunk_sizes)
    if not models:
        raise ValueError("--models must contain at least one path")
    if any(size > 64 for size in chunk_sizes):
        raise ValueError("DPLR chunk sizes must be <=64")
    if args.prompt_target_chars <= 0 or args.repeat <= 0 or args.warmup < 0:
        raise ValueError("prompt/repeat must be positive and warmup non-negative")
    if args.decode_tokens < 0 or args.prefill_eval_interval <= 0:
        raise ValueError("decode tokens must be non-negative and eval interval positive")
    if args.logits_atol < 0 or args.state_atol < 0:
        raise ValueError("parity tolerances must be non-negative")

    prompt = make_prompt(args.prompt_seed, args.prompt_target_chars)
    env = {
        "axis": AXIS + "_env",
        "status": "plan" if args.dry_run else "info",
        "models": models,
        "prompt_chars": len(prompt),
        "dplr_chunk_sizes": chunk_sizes,
        "repeat": int(args.repeat),
        "warmup": int(args.warmup),
        "decode_tokens": int(args.decode_tokens),
        "dtype": args.dtype,
        "quantization": args.quantization,
        "logits_atol": float(args.logits_atol),
        "state_atol": float(args.state_atol),
        **device_info(),
    }
    print(json.dumps(env, ensure_ascii=False))
    append_jsonl(args.results, env)
    if args.dry_run:
        return 0

    import mlx.core as mx
    from transformers import AutoTokenizer

    from rwkv7_hf.mlx_bridge import mlx_memory_telemetry, reset_mlx_peak_memory
    from rwkv7_hf.mlx_model import load_mlx_rwkv7_model

    all_rows: list[dict[str, Any]] = []
    for model_path in models:
        model = load_mlx_rwkv7_model(
            model_path,
            dtype=args.dtype,
            quantization=args.quantization,
            quant_min_params=args.quant_min_params,
            quant_backend=args.quant_backend,
            wkv_backend=args.wkv_backend,
        )
        model.prefill_eval_interval = int(args.prefill_eval_interval)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        prompt_ids = [int(token) for token in tokenizer(prompt, add_special_tokens=False).input_ids]
        if not prompt_ids:
            raise ValueError(f"{model_path}: tokenizer produced no prompt tokens")

        model.prefill_backend = "recurrent"
        reference_logits, reference_state = model.prefill([prompt_ids])
        mx.eval(reference_logits, *_state_arrays(reference_state))
        reference_tokens = _decode_tokens(
            mx,
            model,
            reference_logits,
            reference_state.clone(),
            args.decode_tokens,
        )
        cases = [("recurrent", 0), *[("dplr_metal", size) for size in chunk_sizes]]
        rows: list[dict[str, Any]] = []
        # Recurrent rows run first so the opt-in DPLR cache-release policy
        # cannot contaminate their allocator warm state. DPLR rows then run in
        # the same loaded model and compare against the retained reference.
        for order_index, (backend, chunk_size) in enumerate(cases, 1):
            model.prefill_backend = backend
            if chunk_size:
                model.dplr_chunk_size = int(chunk_size)
            for _ in range(args.warmup):
                logits, state = model.prefill([prompt_ids])
                mx.eval(logits, *_state_arrays(state))
            for repeat_index in range(args.repeat):
                model.prefill_backend = backend
                if chunk_size:
                    model.dplr_chunk_size = int(chunk_size)
                reset_mlx_peak_memory()
                started = time.perf_counter()
                logits, state = model.prefill([prompt_ids])
                mx.eval(logits, *_state_arrays(state))
                elapsed_s = time.perf_counter() - started
                logits_diff = _max_abs(mx, reference_logits, logits)
                state_diff = _state_max_abs(mx, reference_state, state)
                generated = _decode_tokens(mx, model, logits, state, args.decode_tokens)
                tokens_match = generated == reference_tokens
                status = (
                    "pass"
                    if logits_diff <= args.logits_atol and state_diff <= args.state_atol and tokens_match
                    else "fail"
                )
                row = {
                    "axis": AXIS,
                    "status": status,
                    "model": Path(model_path).name,
                    "model_path": model_path,
                    "dtype": args.dtype,
                    "quantization": args.quantization,
                    "prefill_backend": backend,
                    "dplr_chunk_size": int(chunk_size),
                    "prompt_chars": len(prompt),
                    "prompt_tokens": len(prompt_ids),
                    "repeat_index": repeat_index + 1,
                    "order_index": order_index,
                    "prefill_s": round(elapsed_s, 6),
                    "prefill_tok_s": round(len(prompt_ids) / elapsed_s, 6),
                    "logits_max_abs": logits_diff,
                    "state_max_abs": state_diff,
                    "logits_atol": float(args.logits_atol),
                    "state_atol": float(args.state_atol),
                    "generated_tokens_match": tokens_match,
                    "generated_token_ids": generated,
                    **mlx_memory_telemetry(),
                }
                print(json.dumps(row, ensure_ascii=False))
                append_jsonl(args.results, row)
                rows.append(row)
                all_rows.append(row)

        for backend, chunk_size in cases:
            selected = [
                row
                for row in rows
                if row["prefill_backend"] == backend and row["dplr_chunk_size"] == int(chunk_size)
            ]
            rates = [float(row["prefill_tok_s"]) for row in selected]
            summary = {
                "axis": AXIS + "_summary",
                "status": "pass" if all(row["status"] == "pass" for row in selected) else "fail",
                "model": Path(model_path).name,
                "dtype": args.dtype,
                "quantization": args.quantization,
                "prefill_backend": backend,
                "dplr_chunk_size": int(chunk_size),
                "repeats": len(selected),
                "min_prefill_tok_s": round(min(rates), 6),
                "median_prefill_tok_s": round(statistics.median(rates), 6),
                "max_prefill_tok_s": round(max(rates), 6),
            }
            print(json.dumps(summary, ensure_ascii=False))
            append_jsonl(args.results, summary)
            all_rows.append(summary)
        model = None
        tokenizer = None
        gc.collect()
        mx.clear_cache()
    return 1 if any(row["status"] == "fail" for row in all_rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
