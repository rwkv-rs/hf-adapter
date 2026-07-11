#!/usr/bin/env python3
"""Interleaved MLX prefill graph-evaluation benchmark and parity gate.

MLX executes lazily. The original recurrent reference evaluated every state
tensor after every prompt token, which made prefill pay one host/device
synchronization per token. This tool measures less frequent evaluation in a
single process, rotates the interval order between repeats, and compares every
result with an interval-1 snapshot.
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.run_qwen35_apple_baseline import append_jsonl, device_info, make_prompt, parse_csv, parse_int_csv

AXIS = "mlx_prefill_eval_interval"


def _state_arrays(state: Any) -> Iterable[Any]:
    if state.v_first is not None:
        yield state.v_first
    yield from state.recurrent_state
    yield from state.attn_x_prev
    yield from state.ffn_x_prev


def _snapshot(mx: Any, logits: Any, state: Any) -> dict[str, Any]:
    arrays = [logits, *_state_arrays(state)]
    mx.eval(*arrays)
    return {
        "logits": np.asarray(logits.astype(mx.float32)),
        "state": [np.asarray(value.astype(mx.float32)) for value in _state_arrays(state)],
        "seen_tokens": int(state.seen_tokens),
        "next_token": int(mx.argmax(logits[:, -1, :], axis=-1).item()),
    }


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    if left.size == 0:
        return 0.0
    return float(np.max(np.abs(left - right)))


def _compare_snapshot(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    state_diffs = [
        _max_abs(left, right)
        for left, right in zip(reference["state"], candidate["state"], strict=True)
    ]
    return {
        "logits_max_abs": _max_abs(reference["logits"], candidate["logits"]),
        "state_max_abs": max(state_diffs, default=0.0),
        "next_token_match": candidate["next_token"] == reference["next_token"],
        "seen_tokens_match": candidate["seen_tokens"] == reference["seen_tokens"],
    }


def _release_mlx(mx: Any) -> None:
    gc.collect()
    clear_cache = getattr(mx, "clear_cache", None)
    if callable(clear_cache):
        clear_cache()


def run_model(
    *,
    model_path: str,
    prompt: str,
    intervals: list[int],
    repeats: int,
    warmup: int,
    dtype: str,
    quantization: str,
    quant_min_params: int,
    quant_backend: str,
    wkv_backend: str,
    atol: float,
    results: str,
) -> list[dict[str, Any]]:
    import mlx.core as mx
    from transformers import AutoTokenizer

    from rwkv7_hf.mlx_bridge import mlx_memory_telemetry, reset_mlx_peak_memory
    from rwkv7_hf.mlx_model import load_mlx_rwkv7_model

    model = load_mlx_rwkv7_model(
        model_path,
        dtype=dtype,
        quantization=quantization,
        quant_min_params=int(quant_min_params),
        quant_backend=quant_backend,
        wkv_backend=wkv_backend,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompt_ids = [int(token) for token in tokenizer(prompt, add_special_tokens=False).input_ids]
    if not prompt_ids:
        raise ValueError(f"{model_path}: tokenizer produced zero prompt tokens")

    model.prefill_eval_interval = 1
    reference_logits, reference_state = model.prefill([prompt_ids])
    reference = _snapshot(mx, reference_logits, reference_state)

    for interval in intervals:
        model.prefill_eval_interval = int(interval)
        for _ in range(int(warmup)):
            logits, state = model.prefill([prompt_ids])
            mx.eval(logits, *_state_arrays(state))

    rows: list[dict[str, Any]] = []
    for repeat_index in range(int(repeats)):
        offset = repeat_index % len(intervals)
        order = intervals[offset:] + intervals[:offset]
        for order_index, interval in enumerate(order):
            model.prefill_eval_interval = int(interval)
            reset_mlx_peak_memory()
            started = time.perf_counter()
            logits, state = model.prefill([prompt_ids])
            mx.eval(logits, *_state_arrays(state))
            elapsed_s = time.perf_counter() - started
            candidate = _snapshot(mx, logits, state)
            parity = _compare_snapshot(reference, candidate)
            parity_pass = (
                parity["logits_max_abs"] <= float(atol)
                and parity["state_max_abs"] <= float(atol)
                and parity["next_token_match"]
                and parity["seen_tokens_match"]
            )
            row = {
                "axis": AXIS,
                "status": "pass" if parity_pass else "fail",
                "model": Path(model_path).name,
                "model_path": model_path,
                "dtype": dtype,
                "quantization": quantization,
                "quant_min_params": int(quant_min_params),
                "quant_backend": quant_backend,
                "wkv_backend": wkv_backend,
                "prompt_chars": len(prompt),
                "prompt_tokens": len(prompt_ids),
                "prefill_eval_interval": int(interval),
                "repeat_index": repeat_index + 1,
                "order_index": order_index + 1,
                "prefill_s": round(float(elapsed_s), 6),
                "prefill_tok_s": round(float(len(prompt_ids) / elapsed_s), 6),
                "atol": float(atol),
                **parity,
                **mlx_memory_telemetry(),
            }
            print(json.dumps(row, ensure_ascii=False))
            append_jsonl(results, row)
            rows.append(row)

    for interval in intervals:
        selected = [row for row in rows if row["prefill_eval_interval"] == int(interval)]
        rates = [float(row["prefill_tok_s"]) for row in selected]
        summary = {
            "axis": AXIS + "_summary",
            "status": "pass" if all(row["status"] == "pass" for row in selected) else "fail",
            "model": Path(model_path).name,
            "dtype": dtype,
            "quantization": quantization,
            "prompt_chars": len(prompt),
            "prompt_tokens": len(prompt_ids),
            "prefill_eval_interval": int(interval),
            "repeats": len(rates),
            "min_prefill_tok_s": round(min(rates), 6),
            "median_prefill_tok_s": round(statistics.median(rates), 6),
            "max_prefill_tok_s": round(max(rates), 6),
        }
        print(json.dumps(summary, ensure_ascii=False))
        append_jsonl(results, summary)
        rows.append(summary)

    model = None
    tokenizer = None
    reference_logits = None
    reference_state = None
    logits = None
    state = None
    _release_mlx(mx)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True, help="Comma-separated converted RWKV-7 HF model directories.")
    parser.add_argument("--intervals", default="1,2,4")
    parser.add_argument("--prompt-target-chars", type=int, default=512)
    parser.add_argument(
        "--prompt-seed",
        default="User: Explain recurrent language-model prefill and state caching. Assistant: ",
    )
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--dtype", default="fp16", choices=["keep", "fp32", "fp16", "bf16"])
    parser.add_argument("--quantization", default="none", choices=["none", "mm8", "mm4"])
    parser.add_argument("--quant-min-params", type=int, default=4_000_000)
    parser.add_argument("--quant-backend", default="auto", choices=["affine", "reference", "metal", "auto", "groupwise"])
    parser.add_argument("--wkv-backend", default="metal", choices=["reference", "metal", "auto"])
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--results", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    models = parse_csv(args.models)
    intervals = parse_int_csv(args.intervals)
    if not models:
        raise ValueError("--models must contain at least one path")
    if args.repeat <= 0 or args.warmup < 0:
        raise ValueError("--repeat must be positive and --warmup must be non-negative")
    if args.prompt_target_chars <= 0:
        raise ValueError("--prompt-target-chars must be positive")
    if args.atol < 0:
        raise ValueError("--atol must be non-negative")

    prompt = make_prompt(args.prompt_seed, args.prompt_target_chars)
    env = {
        "axis": AXIS + "_env",
        "status": "plan" if args.dry_run else "info",
        "models": models,
        "intervals": intervals,
        "prompt_chars": len(prompt),
        "repeat": int(args.repeat),
        "warmup": int(args.warmup),
        "dtype": args.dtype,
        "quantization": args.quantization,
        "atol": float(args.atol),
        **device_info(),
    }
    print(json.dumps(env, ensure_ascii=False))
    append_jsonl(args.results, env)
    if args.dry_run:
        return 0

    rows: list[dict[str, Any]] = []
    for model_path in models:
        rows.extend(
            run_model(
                model_path=model_path,
                prompt=prompt,
                intervals=intervals,
                repeats=int(args.repeat),
                warmup=int(args.warmup),
                dtype=args.dtype,
                quantization=args.quantization,
                quant_min_params=int(args.quant_min_params),
                quant_backend=args.quant_backend,
                wkv_backend=args.wkv_backend,
                atol=float(args.atol),
                results=args.results,
            )
        )
    return 1 if any(row["status"] == "fail" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
