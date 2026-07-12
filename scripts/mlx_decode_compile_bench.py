#!/usr/bin/env python3
"""Benchmark eager versus prepared ``mx.compile`` MLX decode graphs."""
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

from bench.run_qwen35_apple_baseline import append_jsonl, device_info, make_prompt, parse_csv

AXIS = "mlx_decode_compile"


def _state_arrays(state: Any) -> list[Any]:
    return [state.v_first, *state.recurrent_state, *state.attn_x_prev, *state.ffn_x_prev]


def _max_abs(mx: Any, left: Any, right: Any) -> float:
    return float(mx.max(mx.abs(left.astype(mx.float32) - right.astype(mx.float32))))


def _state_max_abs(mx: Any, left: Any, right: Any) -> float:
    return max(
        _max_abs(mx, a, b)
        for a, b in zip(_state_arrays(left), _state_arrays(right), strict=True)
    )


def _decode(mx: Any, model: Any, logits: Any, state: Any, count: int, backend: str):
    model.decode_backend = backend
    generated: list[int] = []
    step_s: list[float] = []
    for _ in range(int(count)):
        token = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.int32)
        mx.eval(token)
        generated.extend(int(value) for value in token.tolist())
        started = time.perf_counter()
        logits, state = model.decode_step(token, state)
        mx.eval(logits)
        step_s.append(time.perf_counter() - started)
    mx.eval(logits, *_state_arrays(state))
    return logits, state, generated, step_s


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True)
    parser.add_argument("--prompt-target-chars", type=int, default=512)
    parser.add_argument(
        "--prompt-seed",
        default="User: Compare eager and compiled recurrent decode on Apple Silicon. Assistant: ",
    )
    parser.add_argument("--decode-tokens", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--validation-tokens", type=int, default=64)
    parser.add_argument("--include-unguarded", action="store_true")
    parser.add_argument("--decode-norm-backend", default="reference", choices=["reference", "fast"])
    parser.add_argument("--dtype", default="fp16", choices=["keep", "fp32", "fp16", "bf16"])
    parser.add_argument("--quantization", default="none", choices=["none", "mm8", "mm4"])
    parser.add_argument("--quant-min-params", type=int, default=8_000_000)
    parser.add_argument("--quant-backend", default="auto", choices=["reference", "affine", "metal", "auto", "groupwise"])
    parser.add_argument("--quant-profile", default="uniform", choices=["uniform", "q4_k_m"])
    parser.add_argument("--quant-group-size", type=int, default=64, choices=[32, 64, 128])
    parser.add_argument("--wkv-backend", default="metal", choices=["reference", "metal", "auto"])
    parser.add_argument("--logits-atol", type=float, default=1e-5)
    parser.add_argument("--state-atol", type=float, default=1e-5)
    parser.add_argument("--reference-logits-atol", type=float, default=0.25)
    parser.add_argument("--reference-state-atol", type=float, default=0.5)
    parser.add_argument("--results", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    models = parse_csv(args.models)
    if not models:
        raise ValueError("--models must contain at least one path")
    if (
        args.prompt_target_chars <= 0
        or args.decode_tokens <= 0
        or args.repeat <= 0
        or args.validation_tokens <= 0
    ):
        raise ValueError("prompt, decode tokens, and repeat must be positive")
    if min(
        args.warmup,
        args.logits_atol,
        args.state_atol,
        args.reference_logits_atol,
        args.reference_state_atol,
    ) < 0:
        raise ValueError("warmup and tolerances must be non-negative")
    env = {
        "axis": AXIS + "_env",
        "status": "plan" if args.dry_run else "info",
        "models": models,
        "prompt_target_chars": int(args.prompt_target_chars),
        "decode_tokens": int(args.decode_tokens),
        "repeat": int(args.repeat),
        "warmup": int(args.warmup),
        "validation_tokens": int(args.validation_tokens),
        "include_unguarded": bool(args.include_unguarded),
        "decode_norm_backend": args.decode_norm_backend,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "quant_min_params": int(args.quant_min_params),
        "quant_backend": args.quant_backend,
        "quant_profile": args.quant_profile,
        "quant_group_size": int(args.quant_group_size),
        "wkv_backend": args.wkv_backend,
        "logits_atol": float(args.logits_atol),
        "state_atol": float(args.state_atol),
        "reference_logits_atol": float(args.reference_logits_atol),
        "reference_state_atol": float(args.reference_state_atol),
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

    failed = False
    prompt = make_prompt(args.prompt_seed, args.prompt_target_chars)
    for model_path in models:
        model = load_mlx_rwkv7_model(
            model_path,
            dtype=args.dtype,
            quantization=args.quantization,
            quant_min_params=args.quant_min_params,
            quant_backend=args.quant_backend,
            quant_profile=args.quant_profile,
            quant_group_size=args.quant_group_size,
            wkv_backend=args.wkv_backend,
        )
        model.prefill_backend = "auto"
        model.decode_norm_backend = args.decode_norm_backend
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        prompt_ids = [int(token) for token in tokenizer(prompt, add_special_tokens=False).input_ids]
        logits, base_state = model.prefill([prompt_ids])
        mx.eval(logits, *_state_arrays(base_state))
        compile_s = model.prepare_compiled_decode(batch_size=1)
        validation = model.validate_compiled_decode(
            logits,
            base_state,
            steps=int(args.validation_tokens),
            logits_atol=float(args.logits_atol),
            state_atol=float(args.state_atol),
            reference_logits_atol=float(args.reference_logits_atol),
            reference_state_atol=float(args.reference_state_atol),
        )

        reference_logits, reference_state, reference_tokens, _ = _decode(
            mx,
            model,
            logits,
            base_state.clone(),
            args.decode_tokens,
            "eager",
        )
        backends = ("eager", "auto", "compiled") if args.include_unguarded else ("eager", "auto")
        for backend in backends:
            for _ in range(args.warmup):
                _decode(mx, model, logits, base_state.clone(), args.decode_tokens, backend)

        rows: list[dict[str, Any]] = []
        for repeat_index in range(args.repeat):
            offset = repeat_index % len(backends)
            order = backends[offset:] + backends[:offset]
            for order_index, backend in enumerate(order, 1):
                reset_mlx_peak_memory()
                got_logits, got_state, generated, step_s = _decode(
                    mx,
                    model,
                    logits,
                    base_state.clone(),
                    args.decode_tokens,
                    backend,
                )
                elapsed_s = sum(step_s)
                logits_diff = _max_abs(mx, reference_logits, got_logits)
                state_diff = _state_max_abs(mx, reference_state, got_state)
                tokens_match = generated == reference_tokens
                status = (
                    "pass"
                    if tokens_match and logits_diff <= args.logits_atol and state_diff <= args.state_atol
                    else "fail"
                )
                failed = failed or status == "fail"
                row = {
                    "axis": AXIS,
                    "status": status,
                    "model": Path(model_path).name,
                    "model_path": model_path,
                    "dtype": args.dtype,
                    "quantization": args.quantization,
                    "quant_min_params": int(args.quant_min_params),
                    "quant_backend": args.quant_backend,
                    "quant_profile": args.quant_profile,
                    "quant_group_size": int(args.quant_group_size),
                    "wkv_backend": args.wkv_backend,
                    "decode_backend": backend,
                    "decode_backend_used": model.decode_backend_last,
                    "decode_norm_backend": args.decode_norm_backend,
                    "compile_s": round(float(compile_s), 6),
                    "compiled_validation": validation,
                    "prompt_tokens": len(prompt_ids),
                    "decode_tokens": int(args.decode_tokens),
                    "repeat_index": repeat_index + 1,
                    "order_index": order_index,
                    "decode_s": round(elapsed_s, 6),
                    "decode_tok_s": round(args.decode_tokens / elapsed_s, 6),
                    "median_step_ms": round(statistics.median(step_s) * 1000.0, 6),
                    "generated_tokens_match": tokens_match,
                    "logits_max_abs": logits_diff,
                    "state_max_abs": state_diff,
                    **mlx_memory_telemetry(),
                }
                print(json.dumps(row, ensure_ascii=False))
                append_jsonl(args.results, row)
                rows.append(row)

        for backend in backends:
            selected = [row for row in rows if row["decode_backend"] == backend]
            rates = [float(row["decode_tok_s"]) for row in selected]
            summary = {
                "axis": AXIS + "_summary",
                "status": "pass" if all(row["status"] == "pass" for row in selected) else "fail",
                "model": Path(model_path).name,
                "dtype": args.dtype,
                "quantization": args.quantization,
                "quant_min_params": int(args.quant_min_params),
                "quant_backend": args.quant_backend,
                "quant_profile": args.quant_profile,
                "quant_group_size": int(args.quant_group_size),
                "wkv_backend": args.wkv_backend,
                "decode_backend": backend,
                "decode_backend_used": selected[-1]["decode_backend_used"],
                "decode_norm_backend": args.decode_norm_backend,
                "compile_s": round(float(compile_s), 6),
                "compiled_validation": validation,
                "repeats": len(selected),
                "min_decode_tok_s": round(min(rates), 6),
                "median_decode_tok_s": round(statistics.median(rates), 6),
                "max_decode_tok_s": round(max(rates), 6),
            }
            print(json.dumps(summary, ensure_ascii=False))
            append_jsonl(args.results, summary)
        model = None
        tokenizer = None
        gc.collect()
        mx.clear_cache()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
