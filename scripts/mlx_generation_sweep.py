#!/usr/bin/env python3
# coding=utf-8
"""Prompt/decode sweep for the optional MLX RWKV-7 reference backend.

The single-prompt and session smokes prove API shape.  This harness records a
small matrix of prompt lengths and decode lengths with MLX memory telemetry so
Apple Silicon validation can track longer-context pressure before fused
MLX/Metal kernels are available.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

from rwkv7_hf.mlx_bridge import mlx_available, mlx_memory_telemetry, reset_mlx_peak_memory
from rwkv7_hf.mlx_model import load_mlx_rwkv7_model


def append_result(path: str, row: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_ints(raw: str, *, name: str) -> list[int]:
    values = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError(f"--{name} must contain at least one integer")
    if any(x <= 0 for x in values):
        raise ValueError(f"--{name} values must be positive: {values}")
    return values


def max_abs(a: Any, b: Any) -> float:
    import mlx.core as mx

    return float(mx.max(mx.abs(a - b)))


def model_quant_runtime_telemetry(model: Any) -> dict[str, Any]:
    """Return quant runtime fields that can change after each prefill/decode row."""

    telemetry = model.telemetry()
    return {
        "step_eval_interval": telemetry.get("step_eval_interval"),
        "quantized_linear_last_backend_counts": telemetry.get("quantized_linear_last_backend_counts"),
        "group_rkv_quant_projection": telemetry.get("group_rkv_quant_projection"),
        "group_rkv_quant_projection_mode": telemetry.get("group_rkv_quant_projection_mode"),
        "group_rkv_quant_projection_counts": telemetry.get("group_rkv_quant_projection_counts"),
    }


def make_prompt_ids(tokenizer: Any, target_tokens: int, seed_text: str) -> list[int]:
    seed_ids = [int(x) for x in tokenizer(seed_text, add_special_tokens=False).input_ids]
    if not seed_ids:
        raise ValueError("--seed-text tokenizes to an empty list of tokens")
    repeats = (int(target_tokens) + len(seed_ids) - 1) // len(seed_ids)
    return (seed_ids * repeats)[: int(target_tokens)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="Converted RWKV-7 HF model directory.")
    ap.add_argument("--prompt-lengths", default="16,64", help="Comma-separated prompt token lengths.")
    ap.add_argument("--decode-lengths", default="2,4", help="Comma-separated generated-token counts.")
    ap.add_argument("--seed-text", default="User: Apple Silicon RWKV generation sweep. Assistant: ")
    ap.add_argument("--dtype", default="fp16", choices=["keep", "fp32", "fp16", "bf16"])
    ap.add_argument("--chunk-size", type=int, default=0, help="If >0, compare chunked prefill final logits.")
    ap.add_argument("--chunk-tolerance", type=float, default=0.2)
    ap.add_argument("--repeat", type=int, default=1, help="Repeat each prompt/decode point for pressure/stability telemetry.")
    ap.add_argument("--quantization", default="none", choices=["none", "mm8", "mm4"], help="Optional MLX packed W8/W4 projection path.")
    ap.add_argument("--quant-min-params", type=int, default=8_000_000, help="Minimum dense weight params to replace when MLX quantization is enabled.")
    ap.add_argument(
        "--quant-rkv-min-params",
        type=int,
        default=-1,
        help="Separate min-params threshold for attention r/k/v projection quantization; -1 preserves --quant-min-params.",
    )
    ap.add_argument("--quant-backend", default="affine", choices=["affine", "reference", "metal", "auto"], help="MLX quantized matmul backend.")
    ap.add_argument("--wkv-backend", default="reference", choices=["reference", "metal", "auto"], help="MLX recurrent WKV update backend.")
    ap.add_argument("--require-mlx", action="store_true")
    ap.add_argument("--json-only", action="store_true")
    ap.add_argument("--results", default="", help="Optional JSONL file to append sweep rows.")
    args = ap.parse_args()

    prompt_lengths = parse_ints(args.prompt_lengths, name="prompt-lengths")
    decode_lengths = parse_ints(args.decode_lengths, name="decode-lengths")
    if int(args.chunk_size) < 0:
        raise ValueError("--chunk-size must be >= 0")
    if int(args.repeat) <= 0:
        raise ValueError("--repeat must be positive")

    if not mlx_available():
        row = {
            "axis": "mlx_generation_sweep",
            "status": "skip",
            "reason": "mlx not installed",
            "platform": platform.platform(),
            "machine": platform.machine(),
            "model": Path(args.model).name,
            "prompt_lengths": prompt_lengths,
            "decode_lengths": decode_lengths,
            "repeat": int(args.repeat),
            "quantization": args.quantization,
            "quant_min_params": int(args.quant_min_params),
            "quant_rkv_min_params": None if int(args.quant_rkv_min_params) < 0 else int(args.quant_rkv_min_params),
            "quant_backend": args.quant_backend,
            "wkv_backend": args.wkv_backend,
        }
        print(json.dumps(row, ensure_ascii=False))
        append_result(args.results, row)
        return 2 if args.require_mlx else 0

    from transformers import AutoTokenizer
    import mlx.core as mx

    t_load = time.perf_counter()
    model = load_mlx_rwkv7_model(
        args.model,
        dtype=args.dtype,
        quantization=args.quantization,
        quant_min_params=int(args.quant_min_params),
        quant_rkv_min_params=None if int(args.quant_rkv_min_params) < 0 else int(args.quant_rkv_min_params),
        quant_backend=args.quant_backend,
        wkv_backend=args.wkv_backend,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    load_s = time.perf_counter() - t_load
    header = {
        "axis": "mlx_generation_sweep_env",
        "status": "info",
        "model": Path(args.model).name,
        "dtype": args.dtype,
        "prompt_lengths": prompt_lengths,
        "decode_lengths": decode_lengths,
        "chunk_size": int(args.chunk_size),
        "repeat": int(args.repeat),
        "quantization": args.quantization,
        "quant_min_params": int(args.quant_min_params),
        "quant_rkv_min_params": None if int(args.quant_rkv_min_params) < 0 else int(args.quant_rkv_min_params),
        "quant_backend": args.quant_backend,
        "wkv_backend": args.wkv_backend,
        "load_s": round(float(load_s), 6),
        **model.telemetry(),
        **mlx_memory_telemetry(),
    }
    print(json.dumps(header, ensure_ascii=False))
    append_result(args.results, header)

    rows: list[dict[str, Any]] = []
    for prompt_tokens in prompt_lengths:
        prompt_ids = make_prompt_ids(tokenizer, prompt_tokens, args.seed_text)
        ids = [prompt_ids]
        for decode_tokens in decode_lengths:
            for repeat_index in range(1, int(args.repeat) + 1):
                reset_mlx_peak_memory()
                t_prefill = time.perf_counter()
                logits, state = model.prefill(ids)
                mx.eval(logits)
                prefill_s = time.perf_counter() - t_prefill
                chunk_diff = None
                chunk_s = None
                if int(args.chunk_size) > 0:
                    t_chunk = time.perf_counter()
                    chunk_logits, chunk_state = model.chunked_prefill(ids, chunk_size=int(args.chunk_size))
                    mx.eval(chunk_logits)
                    chunk_s = time.perf_counter() - t_chunk
                    chunk_diff = max_abs(logits, chunk_logits)
                    if chunk_diff > float(args.chunk_tolerance):
                        raise AssertionError(
                            f"chunked/full MLX prefill mismatch {chunk_diff} "
                            f"for prompt_tokens={prompt_tokens}, chunk_size={args.chunk_size}, "
                            f"repeat={repeat_index}"
                        )
                    if int(chunk_state.seen_tokens) != int(prompt_tokens):
                        raise AssertionError(
                            f"chunked state seen_tokens={chunk_state.seen_tokens}, expected {prompt_tokens}"
                        )
                t_decode = time.perf_counter()
                generated, gen_state = model.decode_greedy(logits, state, max_new_tokens=int(decode_tokens))
                mx.eval(generated)
                decode_s = time.perf_counter() - t_decode
                generated_ids = [int(x) for x in generated.reshape(-1).tolist()]
                expected_seen = int(prompt_tokens) + int(decode_tokens)
                if int(gen_state.seen_tokens) != expected_seen:
                    raise AssertionError(
                        f"seen_tokens={gen_state.seen_tokens}, expected {expected_seen}, repeat={repeat_index}"
                    )
                row = {
                    "axis": "mlx_generation_sweep",
                    "status": "pass",
                    "model": Path(args.model).name,
                    "dtype": args.dtype,
                    "repeat_index": int(repeat_index),
                    "repeat": int(args.repeat),
                    "quantization": args.quantization,
                    "quant_min_params": int(args.quant_min_params),
                    "quant_rkv_min_params": None
                    if int(args.quant_rkv_min_params) < 0
                    else int(args.quant_rkv_min_params),
                    "quant_backend": args.quant_backend,
                    "wkv_backend": args.wkv_backend,
                    "prompt_tokens": int(prompt_tokens),
                    "generated_tokens": int(decode_tokens),
                    "prefill_s": round(float(prefill_s), 6),
                    "decode_s": round(float(decode_s), 6),
                    "prefill_tok_s": round(float(prompt_tokens / prefill_s), 6) if prefill_s > 0 else None,
                    "decode_tok_s": round(float(decode_tokens / decode_s), 6) if decode_s > 0 else None,
                    "seen_tokens_after_generate": int(gen_state.seen_tokens),
                    "expected_seen_tokens": int(expected_seen),
                    "generated_preview": generated_ids[:16],
                    **model_quant_runtime_telemetry(model),
                    **mlx_memory_telemetry(),
                }
                if chunk_diff is not None:
                    row.update(
                        {
                            "chunk_size": int(args.chunk_size),
                            "chunked_prefill_s": round(float(chunk_s), 6) if chunk_s is not None else None,
                            "chunked_prefill_max_abs": round(float(chunk_diff), 8),
                        }
                    )
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False))
                append_result(args.results, row)
    summary = {
        "axis": "mlx_generation_sweep_summary",
        "status": "pass",
        "model": Path(args.model).name,
        "dtype": args.dtype,
        "prompt_lengths": prompt_lengths,
        "decode_lengths": decode_lengths,
        "repeat": int(args.repeat),
        "quantization": args.quantization,
        "quant_min_params": int(args.quant_min_params),
        "quant_rkv_min_params": None if int(args.quant_rkv_min_params) < 0 else int(args.quant_rkv_min_params),
        "quant_backend": args.quant_backend,
        "wkv_backend": args.wkv_backend,
        "rows": len(rows),
        "max_prompt_tokens": max(prompt_lengths),
        "max_generated_tokens": max(decode_lengths),
        "max_mlx_active_memory_bytes": max(int(row.get("mlx_active_memory_bytes", 0)) for row in rows) if rows else None,
        "max_mlx_peak_memory_bytes": max(int(row.get("mlx_peak_memory_bytes", 0)) for row in rows) if rows else None,
        "max_mlx_cache_memory_bytes": max(int(row.get("mlx_cache_memory_bytes", 0)) for row in rows) if rows else None,
        "min_prefill_tok_s": min(float(row["prefill_tok_s"]) for row in rows if row.get("prefill_tok_s") is not None)
        if rows
        else None,
        "min_decode_tok_s": min(float(row["decode_tok_s"]) for row in rows if row.get("decode_tok_s") is not None)
        if rows
        else None,
        **model_quant_runtime_telemetry(model),
    }
    print(json.dumps(summary, ensure_ascii=False))
    append_result(args.results, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
