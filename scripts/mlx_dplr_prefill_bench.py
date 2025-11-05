#!/usr/bin/env python3
"""Benchmark the correctness-first MLX DPLR/WY three-stage scaffold."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.run_qwen35_apple_baseline import append_jsonl, device_info

AXIS = "mlx_dplr_prefill_stage"


def _summary_arrays(summary: dict[str, Any]) -> list[Any]:
    return [
        summary["transition_diag"],
        summary["transition_left"],
        summary["transition_right"],
        summary["additive_left"],
        summary["additive_right"],
    ]


def _max_abs(mx: Any, left: Any, right: Any) -> float:
    return float(mx.max(mx.abs(left.astype(mx.float32) - right.astype(mx.float32))))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--atol", type=float, default=2e-3)
    parser.add_argument("--results", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for name in ("batch", "tokens", "heads", "head_dim", "chunk_size", "repeat"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.warmup < 0 or args.atol < 0:
        raise ValueError("--warmup and --atol must be non-negative")
    if args.tokens % args.chunk_size != 0:
        raise ValueError("--tokens must be divisible by --chunk-size")

    env = {
        "axis": AXIS + "_env",
        "status": "plan" if args.dry_run else "info",
        "batch": int(args.batch),
        "tokens": int(args.tokens),
        "heads": int(args.heads),
        "head_dim": int(args.head_dim),
        "chunk_size": int(args.chunk_size),
        "dtype": args.dtype,
        "repeat": int(args.repeat),
        "warmup": int(args.warmup),
        "atol": float(args.atol),
        **device_info(),
    }
    print(json.dumps(env, ensure_ascii=False))
    append_jsonl(args.results, env)
    if args.dry_run:
        return 0

    import mlx.core as mx

    from rwkv7_hf.mlx_bridge import mlx_memory_telemetry, reset_mlx_peak_memory
    from rwkv7_hf.mlx_dplr_prefill import (
        mlx_compact_wy_chunk_apply,
        mlx_compact_wy_chunk_apply_metal,
        mlx_compact_wy_chunk_summary,
        mlx_compact_wy_chunk_summary_metal,
        mlx_compact_wy_prefix_combine,
        mlx_compact_wy_three_stage,
        mlx_compact_wy_three_stage_metal,
        mlx_dplr_recurrent_scan_reference,
        mlx_dplr_metal_available,
    )

    mx.random.seed(7007)
    dtype = mx.float16 if args.dtype == "fp16" else mx.float32
    shape = (args.batch, args.tokens, args.heads, args.head_dim)
    r = (mx.random.normal(shape) * 0.2).astype(dtype)
    w = mx.sigmoid(mx.random.normal(shape)).astype(dtype)
    k = (mx.random.normal(shape) * 0.2).astype(dtype)
    v = (mx.random.normal(shape) * 0.2).astype(dtype)
    kk = (mx.random.normal(shape) * 0.2).astype(dtype)
    a = (mx.random.normal(shape) * 0.2).astype(dtype)
    state = (mx.random.normal((args.batch, args.heads, args.head_dim, args.head_dim)) * 0.2).astype(mx.float32)

    ref_out, ref_state = mlx_dplr_recurrent_scan_reference(r, w, k, v, kk, a, state)
    mx.eval(ref_out, ref_state)
    summary = mlx_compact_wy_chunk_summary(w, k, v, kk, a, chunk_size=args.chunk_size)
    mx.eval(*_summary_arrays(summary))
    starts, prefix_state = mlx_compact_wy_prefix_combine(state, summary)
    mx.eval(starts, prefix_state)

    def recurrent() -> tuple[list[Any], Callable[[], dict[str, float]]]:
        out, final = mlx_dplr_recurrent_scan_reference(r, w, k, v, kk, a, state)
        return [out, final], lambda: {
            "output_max_abs": _max_abs(mx, out, ref_out),
            "state_max_abs": _max_abs(mx, final, ref_state),
        }

    def chunk_summary() -> tuple[list[Any], Callable[[], dict[str, float]]]:
        value = mlx_compact_wy_chunk_summary(w, k, v, kk, a, chunk_size=args.chunk_size)
        return _summary_arrays(value), lambda: {}

    def chunk_summary_metal() -> tuple[list[Any], Callable[[], dict[str, float]]]:
        value = mlx_compact_wy_chunk_summary_metal(w, k, v, kk, a, chunk_size=args.chunk_size)
        return _summary_arrays(value), lambda: {
            "factor_max_abs": max(
                _max_abs(mx, value[key], summary[key])
                for key in (
                    "transition_diag",
                    "transition_left",
                    "transition_right",
                    "additive_left",
                    "additive_right",
                )
            )
        }

    def prefix_combine() -> tuple[list[Any], Callable[[], dict[str, float]]]:
        value_starts, value_final = mlx_compact_wy_prefix_combine(state, summary)
        return [value_starts, value_final], lambda: {"state_max_abs": _max_abs(mx, value_final, ref_state)}

    def chunk_apply() -> tuple[list[Any], Callable[[], dict[str, float]]]:
        out, ends = mlx_compact_wy_chunk_apply(r, w, k, v, kk, a, starts, chunk_size=args.chunk_size)
        return [out, ends], lambda: {
            "output_max_abs": _max_abs(mx, out, ref_out),
            "state_max_abs": _max_abs(mx, ends[:, -1], ref_state),
        }

    def chunk_apply_metal() -> tuple[list[Any], Callable[[], dict[str, float]]]:
        out, ends = mlx_compact_wy_chunk_apply_metal(
            r, w, k, v, kk, a, starts, chunk_size=args.chunk_size
        )
        return [out, ends], lambda: {
            "output_max_abs": _max_abs(mx, out, ref_out),
            "state_max_abs": _max_abs(mx, ends[:, -1], ref_state),
        }

    def three_stage() -> tuple[list[Any], Callable[[], dict[str, float]]]:
        out, final, telemetry = mlx_compact_wy_three_stage(
            r, w, k, v, kk, a, state, chunk_size=args.chunk_size
        )
        arrays = [
            out,
            final,
            telemetry["start_states"],
            telemetry["chunk_ends"],
            *_summary_arrays(telemetry["summary"]),
        ]
        return arrays, lambda: {
            "output_max_abs": _max_abs(mx, out, ref_out),
            "state_max_abs": _max_abs(mx, final, ref_state),
        }

    def three_stage_metal_summary() -> tuple[list[Any], Callable[[], dict[str, float]]]:
        value_summary = mlx_compact_wy_chunk_summary_metal(w, k, v, kk, a, chunk_size=args.chunk_size)
        value_starts, value_final = mlx_compact_wy_prefix_combine(state, value_summary)
        out, ends = mlx_compact_wy_chunk_apply(
            r, w, k, v, kk, a, value_starts, chunk_size=args.chunk_size
        )
        arrays = [out, value_final, value_starts, ends, *_summary_arrays(value_summary)]
        return arrays, lambda: {
            "output_max_abs": _max_abs(mx, out, ref_out),
            "state_max_abs": _max_abs(mx, value_final, ref_state),
        }

    def three_stage_metal() -> tuple[list[Any], Callable[[], dict[str, float]]]:
        out, final, telemetry = mlx_compact_wy_three_stage_metal(
            r, w, k, v, kk, a, state, chunk_size=args.chunk_size
        )
        arrays = [
            out,
            final,
            telemetry["start_states"],
            telemetry["chunk_ends"],
            *_summary_arrays(telemetry["summary"]),
        ]
        return arrays, lambda: {
            "output_max_abs": _max_abs(mx, out, ref_out),
            "state_max_abs": _max_abs(mx, final, ref_state),
        }

    stages: list[tuple[str, Callable[[], tuple[list[Any], Callable[[], dict[str, float]]]]]] = [
        ("recurrent_reference", recurrent),
        ("chunk_summary", chunk_summary),
        ("prefix_combine", prefix_combine),
        ("chunk_apply_output", chunk_apply),
        ("three_stage", three_stage),
    ]
    if mlx_dplr_metal_available():
        stages.extend(
            [
                ("chunk_summary_metal", chunk_summary_metal),
                ("chunk_apply_output_metal", chunk_apply_metal),
                ("three_stage_metal_summary", three_stage_metal_summary),
                ("three_stage_metal", three_stage_metal),
            ]
        )

    for _, function in stages:
        for _ in range(args.warmup):
            arrays, _ = function()
            mx.eval(*arrays)

    rows: list[dict[str, Any]] = []
    for repeat_index in range(args.repeat):
        offset = repeat_index % len(stages)
        for order_index, (stage, function) in enumerate(stages[offset:] + stages[:offset], 1):
            reset_mlx_peak_memory()
            started = time.perf_counter()
            arrays, parity_fn = function()
            mx.eval(*arrays)
            elapsed_s = time.perf_counter() - started
            parity = parity_fn()
            parity_pass = all(float(value) <= args.atol for value in parity.values())
            row = {
                "axis": AXIS,
                "status": "pass" if parity_pass else "fail",
                "stage": stage,
                "repeat_index": repeat_index + 1,
                "order_index": order_index,
                "elapsed_s": round(elapsed_s, 6),
                "effective_tok_s": round(args.batch * args.tokens / elapsed_s, 6),
                **parity,
                **mlx_memory_telemetry(),
            }
            print(json.dumps(row, ensure_ascii=False))
            append_jsonl(args.results, row)
            rows.append(row)

    for stage, _ in stages:
        selected = [row for row in rows if row["stage"] == stage]
        elapsed = [float(row["elapsed_s"]) for row in selected]
        rates = [float(row["effective_tok_s"]) for row in selected]
        summary_row = {
            "axis": AXIS + "_summary",
            "status": "pass" if all(row["status"] == "pass" for row in selected) else "fail",
            "stage": stage,
            "repeats": len(selected),
            "median_elapsed_s": round(statistics.median(elapsed), 6),
            "median_effective_tok_s": round(statistics.median(rates), 6),
        }
        print(json.dumps(summary_row, ensure_ascii=False))
        append_jsonl(args.results, summary_row)
        rows.append(summary_row)
    return 1 if any(row["status"] == "fail" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
