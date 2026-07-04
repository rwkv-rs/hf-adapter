#!/usr/bin/env python3
# coding=utf-8
"""Serving-shaped MLX generation session smoke for RWKV-7.

This validates the Apple/MLX API shape needed by later serving adapters:
prefill once, retain the RWKV recurrent state cache, and decode in multiple
chunks without recomputing the prompt.  It also compares chunked session output
against a one-shot MLX generate call for token-id equality.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

from rwkv7_hf.mlx_bridge import mlx_available, mlx_memory_telemetry, reset_mlx_peak_memory
from rwkv7_hf.mlx_model import MLXGenerationSession, load_mlx_rwkv7_model


def append_result(path: str, row: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_step_sizes(raw: str) -> list[int]:
    steps = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not steps:
        raise ValueError("--step-sizes must contain at least one integer")
    if any(x < 0 for x in steps):
        raise ValueError("--step-sizes must be non-negative")
    if sum(steps) <= 0:
        raise ValueError("--step-sizes must request at least one generated token")
    return steps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="Converted RWKV-7 HF model directory.")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--step-sizes", default="4,4", help="Comma-separated decode chunks after one prefill.")
    ap.add_argument("--dtype", default="fp16", choices=["keep", "fp32", "fp16", "bf16"])
    ap.add_argument("--skip-special-tokens", action="store_true")
    ap.add_argument("--require-mlx", action="store_true")
    ap.add_argument("--json-only", action="store_true")
    ap.add_argument("--results", default="", help="Optional JSONL file to append a generation result row.")
    args = ap.parse_args()

    steps = parse_step_sizes(args.step_sizes)
    if not mlx_available():
        row = {
            "axis": "mlx_session_generate",
            "status": "skip",
            "reason": "mlx not installed",
            "platform": platform.platform(),
            "machine": platform.machine(),
            "model": Path(args.model).name,
            "step_sizes": steps,
        }
        print(json.dumps(row, ensure_ascii=False))
        append_result(args.results, row)
        return 2 if args.require_mlx else 0

    from transformers import AutoTokenizer

    reset_mlx_peak_memory()
    model = load_mlx_rwkv7_model(args.model, dtype=args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    session = MLXGenerationSession.from_prompt(
        model,
        tokenizer,
        args.prompt,
        skip_special_tokens=bool(args.skip_special_tokens),
    )
    step_rows = []
    for step in steps:
        step_rows.append(session.decode(step).telemetry())

    one_shot = model.generate_text(
        tokenizer,
        args.prompt,
        max_new_tokens=sum(steps),
        skip_special_tokens=bool(args.skip_special_tokens),
    )
    token_match = session.generated_ids == one_shot.generated_ids
    text_match = session.text == one_shot.text
    expected_seen = len(session.prompt_ids) + sum(steps)
    if not token_match:
        raise AssertionError(
            "session chunked decode token ids differ from one-shot MLX generate: "
            f"session={session.generated_ids[:16]} one_shot={one_shot.generated_ids[:16]}"
        )
    if int(session.state.seen_tokens) != expected_seen:
        raise AssertionError(
            f"session seen_tokens={session.state.seen_tokens} expected {expected_seen}"
        )

    output = session.output()
    row = {
        "axis": "mlx_session_generate",
        "status": "pass",
        "model": Path(args.model).name,
        "dtype": args.dtype,
        "prompt_preview": args.prompt[:80],
        "step_sizes": steps,
        "session_one_shot_token_match": bool(token_match),
        "session_one_shot_text_match": bool(text_match),
        "seen_tokens_after_generate": int(session.state.seen_tokens),
        "expected_seen_tokens": int(expected_seen),
        "text": output.text,
        "step_telemetry": step_rows,
        **output.telemetry(),
        **mlx_memory_telemetry(),
    }
    if not args.json_only:
        print(output.text)
    print(json.dumps(row, ensure_ascii=False))
    append_result(args.results, row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
