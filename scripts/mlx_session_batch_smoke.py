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


def first_mismatch(left: list[int], right: list[int]) -> int | None:
    for idx, (a, b) in enumerate(zip(left, right)):
        if int(a) != int(b):
            return idx
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def run_session_backend_sequence(
    *,
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    rounds: list[int],
    session_backend: str,
    skip_special_tokens: bool,
) -> tuple[MLXGenerationSessionBatch, list[dict[str, Any]]]:
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
                "backend_reason": batch.round_backend_reasons[-1] if batch.round_backend_reasons else None,
                "sessions": [output.telemetry() for output in outputs],
            }
        )
    return batch, round_rows


def run_backend_compare(
    *,
    model: Any,
    tokenizer: Any,
    model_name: str,
    dtype: str,
    prompts: list[str],
    rounds: list[int],
    skip_special_tokens: bool,
    quantization: str,
    quant_min_params: int,
    quant_backend: str,
    wkv_backend: str,
    session_backend: str,
    compare_session_backend: str,
    require_match: bool,
) -> tuple[dict[str, Any], list[str]]:
    """Compare two session scheduling backends without hiding mismatches.

    This is the strict batched-session bring-up harness.  The normal generation
    path intentionally raises on one-shot mismatches; this comparison path
    records whether sequential/batched/auto agree, where the first token
    mismatch appears, and whether each side still agrees with independent
    one-shot greedy generation.
    """

    reset_mlx_peak_memory()
    left_batch, left_rounds = run_session_backend_sequence(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        rounds=rounds,
        session_backend=session_backend,
        skip_special_tokens=skip_special_tokens,
    )
    right_batch, right_rounds = run_session_backend_sequence(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        rounds=rounds,
        session_backend=compare_session_backend,
        skip_special_tokens=skip_special_tokens,
    )

    expected_tokens = sum(rounds)
    per_session: list[dict[str, Any]] = []
    texts: list[str] = []
    all_backend_token_match = True
    all_backend_text_match = True
    all_seen_match = True
    all_left_one_shot_match = True
    all_right_one_shot_match = True

    for idx, (left, right) in enumerate(zip(left_batch.sessions, right_batch.sessions)):
        one_shot = model.generate_text(
            tokenizer,
            left.prompt,
            max_new_tokens=expected_tokens,
            skip_special_tokens=skip_special_tokens,
        )
        backend_token_match = left.generated_ids == right.generated_ids
        backend_text_match = left.text == right.text
        left_one_shot_match = left.generated_ids == one_shot.generated_ids
        right_one_shot_match = right.generated_ids == one_shot.generated_ids
        expected_seen = len(left.prompt_ids) + expected_tokens
        seen_match = int(left.state.seen_tokens) == expected_seen and int(right.state.seen_tokens) == expected_seen
        mismatch_index = first_mismatch(left.generated_ids, right.generated_ids)
        all_backend_token_match = all_backend_token_match and backend_token_match
        all_backend_text_match = all_backend_text_match and backend_text_match
        all_left_one_shot_match = all_left_one_shot_match and left_one_shot_match
        all_right_one_shot_match = all_right_one_shot_match and right_one_shot_match
        all_seen_match = all_seen_match and seen_match
        texts.append(right.text)
        per_session.append(
            {
                "session_index": int(idx),
                "prompt_preview": left.prompt[:80],
                "prompt_tokens": int(left.prompt_tokens),
                "generated_tokens": int(left.generated_tokens),
                "expected_seen_tokens": int(expected_seen),
                "backend_token_match": bool(backend_token_match),
                "backend_text_match": bool(backend_text_match),
                "left_one_shot_token_match": bool(left_one_shot_match),
                "right_one_shot_token_match": bool(right_one_shot_match),
                "seen_tokens_match": bool(seen_match),
                "first_token_mismatch_index": mismatch_index,
                "left_generated_preview": [int(x) for x in left.generated_ids[:16]],
                "right_generated_preview": [int(x) for x in right.generated_ids[:16]],
                "one_shot_preview": [int(x) for x in one_shot.generated_ids[:16]],
                "left_decode_s": round(float(left.decode_s), 6),
                "right_decode_s": round(float(right.decode_s), 6),
                "left_decode_tok_s": round(float(left.decode_tok_s), 6) if left.decode_tok_s is not None else None,
                "right_decode_tok_s": round(float(right.decode_tok_s), 6) if right.decode_tok_s is not None else None,
            }
        )

    strict_match = (
        all_backend_token_match
        and all_backend_text_match
        and all_left_one_shot_match
        and all_right_one_shot_match
        and all_seen_match
    )
    row = {
        "axis": "mlx_session_batch_backend_compare",
        "status": "pass",
        "backend_compare_status": "match" if strict_match else "mismatch",
        "strict_match": bool(strict_match),
        "model": model_name,
        "dtype": dtype,
        "quantization": quantization,
        "quant_min_params": int(quant_min_params),
        "quant_backend": quant_backend,
        "wkv_backend": wkv_backend,
        "session_backend": session_backend,
        "compare_session_backend": compare_session_backend,
        "rounds": rounds,
        "expected_generated_tokens_per_session": int(expected_tokens),
        "session_count": len(prompts),
        "all_backend_token_match": bool(all_backend_token_match),
        "all_backend_text_match": bool(all_backend_text_match),
        "all_left_one_shot_token_match": bool(all_left_one_shot_match),
        "all_right_one_shot_token_match": bool(all_right_one_shot_match),
        "all_seen_tokens_match": bool(all_seen_match),
        "left_round_telemetry": left_rounds,
        "right_round_telemetry": right_rounds,
        "left_batch": left_batch.telemetry(),
        "right_batch": right_batch.telemetry(),
        "per_session": per_session,
        "wkv_backend_last": model.wkv_backend_last,
        "wkv_backend_counts": dict(model.wkv_backend_counts),
        "quantized_linear_last_backend_counts": model.telemetry().get("quantized_linear_last_backend_counts"),
        **mlx_memory_telemetry(),
    }
    if require_match and not strict_match:
        raise AssertionError(
            f"session backend compare mismatch: {session_backend} vs {compare_session_backend}"
        )
    return row, texts


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
    batch, round_rows = run_session_backend_sequence(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        rounds=rounds,
        session_backend=session_backend,
        skip_special_tokens=skip_special_tokens,
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
        "quantized_linear_last_backend_counts": model.telemetry().get("quantized_linear_last_backend_counts"),
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
    ap.add_argument("--quant-backend", default="affine", choices=["affine", "reference", "metal", "auto"])
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
    ap.add_argument(
        "--compare-session-backend",
        default="none",
        choices=["none", "sequential", "batched", "auto"],
        help="Optionally compare --session-backend against another scheduler and record strict match telemetry.",
    )
    ap.add_argument(
        "--compare-only",
        action="store_true",
        help="Only run the backend comparison row; useful when an experimental backend is expected to mismatch.",
    )
    ap.add_argument(
        "--require-session-backend-match",
        action="store_true",
        help="Fail unless both compared backends match each other, one-shot greedy tokens/text, and seen_tokens.",
    )
    ap.add_argument("--require-mlx", action="store_true")
    ap.add_argument("--json-only", action="store_true")
    ap.add_argument("--results", default="", help="Optional JSONL file to append a generation result row.")
    args = ap.parse_args()

    rounds = parse_rounds(args.rounds)
    prompts = [str(x) for x in args.prompt]
    repeat = positive_int(args.repeat, name="repeat")
    if args.compare_only and args.compare_session_backend == "none":
        raise ValueError("--compare-only requires --compare-session-backend")
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
            "compare_session_backend": args.compare_session_backend,
            "compare_only": bool(args.compare_only),
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
        "compare_session_backend": args.compare_session_backend,
        "compare_only": bool(args.compare_only),
        "session_count": len(prompts),
        "rounds": rounds,
        "repeat": int(repeat),
        **model.telemetry(),
        **mlx_memory_telemetry(),
    }
    print(json.dumps(header, ensure_ascii=False))
    append_result(args.results, header)

    rows: list[dict[str, Any]] = []
    if not args.compare_only:
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
            "round_backend_reasons": sorted(
                {reason for row in rows for reason in row.get("round_backend_reasons", [])}
            ),
            "max_prompt_tokens": max(max(int(x) for x in row.get("prompt_tokens", [0])) for row in rows) if rows else None,
            "max_generated_tokens": max(max(int(x) for x in row.get("generated_tokens", [0])) for row in rows) if rows else None,
            "max_mlx_active_memory_bytes": max(int(row.get("mlx_active_memory_bytes", 0)) for row in rows) if rows else None,
            "max_mlx_peak_memory_bytes": max(int(row.get("mlx_peak_memory_bytes", 0)) for row in rows) if rows else None,
            "max_mlx_cache_memory_bytes": max(int(row.get("mlx_cache_memory_bytes", 0)) for row in rows) if rows else None,
            "min_decode_tok_s": min_decode_tok_s(rows),
            "min_round_decode_tok_s": min_round_decode_tok_s(rows),
            "quantized_linear_last_backend_counts": model.telemetry().get("quantized_linear_last_backend_counts"),
        }
        print(json.dumps(summary, ensure_ascii=False))
        append_result(args.results, summary)

    if args.compare_session_backend != "none":
        compare_row, compare_texts = run_backend_compare(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            dtype=args.dtype,
            prompts=prompts,
            rounds=rounds,
            skip_special_tokens=bool(args.skip_special_tokens),
            quantization=args.quantization,
            quant_min_params=int(args.quant_min_params),
            quant_backend=args.quant_backend,
            wkv_backend=args.wkv_backend,
            session_backend=args.session_backend,
            compare_session_backend=args.compare_session_backend,
            require_match=bool(args.require_session_backend_match),
        )
        if not args.json_only:
            for item in compare_texts:
                print(item)
        print(json.dumps(compare_row, ensure_ascii=False))
        append_result(args.results, compare_row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
