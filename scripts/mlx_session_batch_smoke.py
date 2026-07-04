#!/usr/bin/env python3
# coding=utf-8
"""Interleaved multi-session MLX generation smoke for RWKV-7.

This validates a serving-shaped scenario above the single-session smoke:
multiple prompts are prefetched once, then each session is advanced in
round-robin decode rounds while preserving its own recurrent state cache.  The
result for every session is compared against independent one-shot MLX generate.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

from rwkv7_hf.mlx_bridge import mlx_available, mlx_memory_telemetry, reset_mlx_peak_memory
from rwkv7_hf.mlx_model import MLXGenerationSessionBatch, load_mlx_rwkv7_model


def append_result(path: str, row: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_rounds(raw: str) -> list[int]:
    rounds = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not rounds:
        raise ValueError("--rounds must contain at least one integer")
    if any(x < 0 for x in rounds):
        raise ValueError("--rounds must be non-negative")
    if sum(rounds) <= 0:
        raise ValueError("--rounds must request at least one generated token")
    return rounds


def positive_int(raw: int, *, name: str) -> int:
    value = int(raw)
    if value <= 0:
        raise ValueError(f"--{name} must be positive, got {value}")
    return value


def min_decode_tok_s(rows: list[dict[str, Any]]) -> float | None:
    values: list[float] = []
    for row in rows:
        for value in row.get("decode_tok_s", []):
            if value is not None:
                values.append(float(value))
    return round(min(values), 6) if values else None


def min_round_decode_tok_s(rows: list[dict[str, Any]]) -> float | None:
    values: list[float] = []
    for row in rows:
        for value in row.get("round_decode_tok_s", []):
            if value is not None:
                values.append(float(value))
    return round(min(values), 6) if values else None


def run_interleaved_batch(
    *,
    model: Any,
    tokenizer: Any,
    model_name: str,
    dtype: str,
    prompts: list[str],
    rounds: list[int],
    repeat_index: int,
    repeat: int,
    skip_special_tokens: bool,
    quantization: str,
    quant_min_params: int,
    quant_backend: str,
    wkv_backend: str,
    session_backend: str,
) -> tuple[dict[str, Any], list[str]]:
    reset_mlx_peak_memory()
    batch = MLXGenerationSessionBatch.from_prompts(
        model,
        tokenizer,
        prompts,
        skip_special_tokens=skip_special_tokens,
    )
    round_rows: list[dict[str, Any]] = []
    for round_index, tokens in enumerate(rounds, start=1):
        outputs = batch.decode_round(tokens, backend=session_backend)
        round_rows.append(
            {
                "round_index": int(round_index),
                "tokens_per_session": int(tokens),
                "session_backend": session_backend,
                "actual_backend": batch.round_backends[-1] if batch.round_backends else None,
                "sessions": [output.telemetry() for output in outputs],
            }
        )

    expected_tokens = sum(rounds)
    per_session = []
    all_token_match = True
    all_text_match = True
    all_seen_match = True
    texts: list[str] = []
    for idx, session in enumerate(batch.sessions):
        one_shot = model.generate_text(
            tokenizer,
            session.prompt,
            max_new_tokens=expected_tokens,
            skip_special_tokens=skip_special_tokens,
        )
        token_match = session.generated_ids == one_shot.generated_ids
        text_match = session.text == one_shot.text
        expected_seen = len(session.prompt_ids) + expected_tokens
        seen_match = int(session.state.seen_tokens) == expected_seen
        all_token_match = all_token_match and token_match
        all_text_match = all_text_match and text_match
        all_seen_match = all_seen_match and seen_match
        texts.append(session.text)
        per_session.append(
            {
                "session_index": int(idx),
                "prompt_preview": session.prompt[:80],
                "prompt_tokens": int(session.prompt_tokens),
                "generated_tokens": int(session.generated_tokens),
                "seen_tokens_after_generate": int(session.state.seen_tokens),
                "expected_seen_tokens": int(expected_seen),
                "session_one_shot_token_match": bool(token_match),
                "session_one_shot_text_match": bool(text_match),
                "seen_tokens_match": bool(seen_match),
                "decode_s": round(float(session.decode_s), 6),
                "decode_tok_s": round(float(session.decode_tok_s), 6) if session.decode_tok_s is not None else None,
                "generated_preview": [int(x) for x in session.generated_ids[:16]],
                "text": session.text,
            }
        )

    if not all_token_match:
        raise AssertionError("at least one session differs from one-shot MLX generate token ids")
    if not all_text_match:
        raise AssertionError("at least one session differs from one-shot MLX generate text")
    if not all_seen_match:
        raise AssertionError("at least one session has unexpected seen_tokens after interleaved decode")

    row = {
        "axis": "mlx_session_batch_generate",
        "status": "pass",
        "model": model_name,
        "dtype": dtype,
        "quantization": quantization,
        "quant_min_params": int(quant_min_params),
        "quant_backend": quant_backend,
        "repeat_index": int(repeat_index),
        "repeat": int(repeat),
        "rounds": rounds,
        "expected_generated_tokens_per_session": int(expected_tokens),
        "session_count": len(prompts),
        "all_session_one_shot_token_match": bool(all_token_match),
        "all_session_one_shot_text_match": bool(all_text_match),
        "all_seen_tokens_match": bool(all_seen_match),
        "wkv_backend": wkv_backend,
        "session_backend": session_backend,
        "wkv_backend_last": model.wkv_backend_last,
        "wkv_backend_counts": dict(model.wkv_backend_counts),
        "round_telemetry": round_rows,
        "per_session": per_session,
        **batch.telemetry(),
        **mlx_memory_telemetry(),
    }
    return row, texts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="Converted RWKV-7 HF model directory.")
    ap.add_argument("--prompt", action="append", required=True, help="Prompt to prefill; pass multiple times.")
    ap.add_argument("--rounds", default="2,2", help="Comma-separated decode rounds applied to every session.")
    ap.add_argument("--dtype", default="fp16", choices=["keep", "fp32", "fp16", "bf16"])
    ap.add_argument("--skip-special-tokens", action="store_true")
    ap.add_argument("--repeat", type=int, default=1, help="Repeat the full interleaved-session workload for pressure telemetry.")
    ap.add_argument("--quantization", default="none", choices=["none", "mm8", "mm4"], help="Optional MLX packed W8/W4 projection path.")
    ap.add_argument("--quant-min-params", type=int, default=8_000_000)
    ap.add_argument("--quant-backend", default="affine", choices=["affine", "reference", "metal"])
    ap.add_argument("--wkv-backend", default="reference", choices=["reference", "metal", "auto"])
    ap.add_argument(
        "--session-backend",
        default="sequential",
        choices=["sequential", "batched", "auto"],
        help=(
            "Session decode scheduler: sequential preserves historical per-session decode; "
            "batched stacks equal-size rounds into one MLX batch; auto batches equal positive rounds."
        ),
    )
    ap.add_argument("--require-mlx", action="store_true")
    ap.add_argument("--json-only", action="store_true")
    ap.add_argument("--results", default="", help="Optional JSONL file to append a generation result row.")
    args = ap.parse_args()

    rounds = parse_rounds(args.rounds)
    prompts = [str(x) for x in args.prompt]
    repeat = positive_int(args.repeat, name="repeat")
    if not mlx_available():
        row = {
            "axis": "mlx_session_batch_generate",
            "status": "skip",
            "reason": "mlx not installed",
            "platform": platform.platform(),
            "machine": platform.machine(),
            "model": Path(args.model).name,
            "batch_size": len(prompts),
            "rounds": rounds,
            "repeat": int(repeat),
            "quantization": args.quantization,
            "quant_min_params": int(args.quant_min_params),
            "quant_backend": args.quant_backend,
            "wkv_backend": args.wkv_backend,
            "session_backend": args.session_backend,
        }
        print(json.dumps(row, ensure_ascii=False))
        append_result(args.results, row)
        return 2 if args.require_mlx else 0

    from transformers import AutoTokenizer

    reset_mlx_peak_memory()
    model = load_mlx_rwkv7_model(
        args.model,
        dtype=args.dtype,
        quantization=args.quantization,
        quant_min_params=int(args.quant_min_params),
        quant_backend=args.quant_backend,
        wkv_backend=args.wkv_backend,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model_name = Path(args.model).name
    header = {
        "axis": "mlx_session_batch_env",
        "status": "info",
        "model": model_name,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "quant_min_params": int(args.quant_min_params),
        "quant_backend": args.quant_backend,
        "wkv_backend": args.wkv_backend,
        "session_backend": args.session_backend,
        "session_count": len(prompts),
        "rounds": rounds,
        "repeat": int(repeat),
        **model.telemetry(),
        **mlx_memory_telemetry(),
    }
    print(json.dumps(header, ensure_ascii=False))
    append_result(args.results, header)

    rows: list[dict[str, Any]] = []
    for repeat_index in range(1, repeat + 1):
        row, texts = run_interleaved_batch(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            dtype=args.dtype,
            prompts=prompts,
            rounds=rounds,
            repeat_index=repeat_index,
            repeat=repeat,
            skip_special_tokens=bool(args.skip_special_tokens),
            quantization=args.quantization,
            quant_min_params=int(args.quant_min_params),
            quant_backend=args.quant_backend,
            wkv_backend=args.wkv_backend,
            session_backend=args.session_backend,
        )
        rows.append(row)
        if not args.json_only:
            for item in texts:
                print(item)
        print(json.dumps(row, ensure_ascii=False))
        append_result(args.results, row)

    summary = {
        "axis": "mlx_session_batch_summary",
        "status": "pass",
        "model": model_name,
        "dtype": args.dtype,
        "quantization": args.quantization,
        "quant_min_params": int(args.quant_min_params),
        "quant_backend": args.quant_backend,
        "repeat": int(repeat),
        "rows": len(rows),
        "session_count": len(prompts),
        "rounds": rounds,
        "expected_generated_tokens_per_session": int(sum(rounds)),
        "all_session_one_shot_token_match": all(bool(row["all_session_one_shot_token_match"]) for row in rows),
        "all_session_one_shot_text_match": all(bool(row["all_session_one_shot_text_match"]) for row in rows),
        "all_seen_tokens_match": all(bool(row["all_seen_tokens_match"]) for row in rows),
        "wkv_backend": args.wkv_backend,
        "session_backend": args.session_backend,
        "round_backends": sorted({backend for row in rows for backend in row.get("round_backends", [])}),
        "max_prompt_tokens": max(max(int(x) for x in row.get("prompt_tokens", [0])) for row in rows) if rows else None,
        "max_generated_tokens": max(max(int(x) for x in row.get("generated_tokens", [0])) for row in rows) if rows else None,
        "max_mlx_active_memory_bytes": max(int(row.get("mlx_active_memory_bytes", 0)) for row in rows) if rows else None,
        "max_mlx_peak_memory_bytes": max(int(row.get("mlx_peak_memory_bytes", 0)) for row in rows) if rows else None,
        "max_mlx_cache_memory_bytes": max(int(row.get("mlx_cache_memory_bytes", 0)) for row in rows) if rows else None,
        "min_decode_tok_s": min_decode_tok_s(rows),
        "min_round_decode_tok_s": min_round_decode_tok_s(rows),
    }
    print(json.dumps(summary, ensure_ascii=False))
    append_result(args.results, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
