#!/usr/bin/env python3
# coding=utf-8
"""Synthetic phase profiler for the Triton full-head fused state-scan path.

This benchmark targets the current main prefill bottleneck:
``fused_recurrent_scan_state_prep``.  It keeps the HF path untouched and times
cumulative phase kernels on target-shaped random tensors:

* phase 0: vector prep, K normalization, W decay, and adjusted K/V writeback;
* phase 1: phase 0 plus state-dot-KK reduction;
* phase 2: phase 1 plus recurrent state update and final-state writeback;
* phase 3: phase 2 plus recurrent readout; should match the normal full-head
  fused state-scan helper for the same inputs.

Rows are direction evidence for the next state-layout/readout rewrite.  The
phase deltas are approximate because each phase is a separately compiled
profiling kernel, but they are more actionable than only seeing the full HF
component as one opaque block.  ``--include-no-kv-write`` also runs the same
cumulative phases with adjusted K/V global stores disabled, isolating how much
of the dominant prep phase is pure writeback traffic.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Callable

import torch

from rwkv7_hf.fused_recurrent_update import (
    fused_recurrent_scan_state_prep,
    fused_recurrent_scan_state_prep_phase_probe,
)


PHASE_NAMES = {
    0: "prep_norm_kv_w",
    1: "prep_norm_kv_w_state_dot",
    2: "prep_norm_kv_w_state_dot_update",
    3: "prep_norm_kv_w_state_dot_update_recurrent",
}


def append_row(path: str, row: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def median_ms(fn: Callable[[], Any], *, warmup: int, steps: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(float(start.elapsed_time(end)))
    return statistics.median(times)


def make_inputs(args: argparse.Namespace) -> dict[str, torch.Tensor]:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    shape = (args.batch_size, args.seq_len, args.heads, args.head_dim)
    return {
        "r": torch.randn(shape, device=device, dtype=dtype),
        "w": torch.randn(shape, device=device, dtype=dtype),
        "k": torch.randn(shape, device=device, dtype=dtype),
        "v": torch.randn(shape, device=device, dtype=dtype),
        "a": torch.randn(shape, device=device, dtype=dtype),
        "state": torch.randn(
            (args.batch_size, args.heads, args.head_dim, args.head_dim),
            device=device,
            dtype=torch.float32,
        ),
        "k_k": torch.randn((args.heads, args.head_dim), device=device, dtype=dtype),
        "k_a": torch.randn((args.heads, args.head_dim), device=device, dtype=dtype),
        "v_first": torch.randn(shape, device=device, dtype=dtype),
        "v_gate": torch.sigmoid(torch.randn(shape, device=device, dtype=dtype)),
    }


def call_phase(tensors: dict[str, torch.Tensor], args: argparse.Namespace, phase: int, *, write_kv: bool = True):
    return fused_recurrent_scan_state_prep_phase_probe(
        tensors["r"],
        tensors["w"],
        tensors["k"],
        tensors["v"],
        tensors["a"],
        tensors["state"],
        tensors["k_k"],
        tensors["k_a"],
        v_first=tensors["v_first"],
        v_gate=tensors["v_gate"],
        phase=phase,
        write_kv=write_kv,
        block_n=args.head_dim,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )


def call_full(tensors: dict[str, torch.Tensor], args: argparse.Namespace):
    return fused_recurrent_scan_state_prep(
        tensors["r"],
        tensors["w"],
        tensors["k"],
        tensors["v"],
        tensors["a"],
        tensors["state"],
        tensors["k_k"],
        tensors["k_a"],
        v_first=tensors["v_first"],
        v_gate=tensors["v_gate"],
        block_n=args.head_dim,
        block_m=args.head_dim,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().detach().cpu())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--num-warps", type=int, default=8)
    ap.add_argument("--num-stages", type=int, default=3)
    ap.add_argument(
        "--include-no-kv-write",
        action="store_true",
        help="Also time phase kernels with adjusted K/V global writeback disabled.",
    )
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--results", default="")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("bench_fused_state_scan_micro requires CUDA")
    if args.head_dim != 64:
        raise ValueError("current phase probe is intended for the N=64 target shape")

    tensors = make_inputs(args)
    tokens_total = args.batch_size * args.seq_len
    device_name = torch.cuda.get_device_name(0)

    with torch.inference_mode():
        full_ref = call_full(tensors, args)
        phase3 = call_phase(tensors, args, 3, write_kv=True)
    correctness = {
        "phase3_out_max_abs_diff": round(max_abs(phase3[0], full_ref[0]), 8),
        "phase3_state_max_abs_diff": round(max_abs(phase3[1], full_ref[1]), 8),
        "phase3_k_max_abs_diff": round(max_abs(phase3[2], full_ref[2]), 8),
        "phase3_v_max_abs_diff": round(max_abs(phase3[3], full_ref[3]), 8),
    }
    status = "pass" if max(correctness.values()) <= 0.0 else "fail"

    full_ms = median_ms(lambda: call_full(tensors, args), warmup=args.warmup, steps=args.steps)
    summary_by_write_mode: dict[bool, dict[str, Any]] = {}
    for write_kv in ([True, False] if args.include_no_kv_write else [True]):
        phase_ms: dict[int, float] = {}
        suffix = "" if write_kv else "_no_kv_write"
        for phase in range(4):
            ms = median_ms(
                lambda phase=phase, write_kv=write_kv: call_phase(tensors, args, phase, write_kv=write_kv),
                warmup=args.warmup,
                steps=args.steps,
            )
            phase_ms[phase] = ms
            row = {
                "axis": "triton_state_scan_micro",
                "backend": "fused_recurrent_update",
                "bench_case": f"fullhead_phase_{phase}_{PHASE_NAMES[phase]}{suffix}",
                "status": status if (phase == 3 and write_kv) else "pass",
                "device": device_name,
                "dtype": args.dtype,
                "batch_size": args.batch_size,
                "seq_len": args.seq_len,
                "heads": args.heads,
                "head_dim": args.head_dim,
                "tokens_total": tokens_total,
                "num_warps": args.num_warps,
                "num_stages": args.num_stages,
                "write_kv": write_kv,
                "phase": phase,
                "phase_name": PHASE_NAMES[phase],
                "triton_ms": round(ms, 6),
                "tokps_total": round(1000.0 * tokens_total / ms, 1) if ms > 0 else None,
                **(correctness if (phase == 3 and write_kv) else {}),
            }
            print(json.dumps(row, ensure_ascii=False))
            append_row(args.results, row)

        component_estimates = {
            "prep_norm_kv_w_ms": phase_ms[0],
            "state_dot_delta_ms": phase_ms[1] - phase_ms[0],
            "state_update_delta_ms": phase_ms[2] - phase_ms[1],
            "recurrent_output_delta_ms": phase_ms[3] - phase_ms[2],
            "phase3_vs_full_delta_ms": phase_ms[3] - full_ms,
        }
        summary = {
            "axis": "triton_state_scan_micro",
            "backend": "fused_recurrent_update",
            "bench_case": f"fullhead_phase_delta_summary{suffix}",
            "status": status if write_kv else "pass",
            "device": device_name,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "tokens_total": tokens_total,
            "num_warps": args.num_warps,
            "num_stages": args.num_stages,
            "write_kv": write_kv,
            "triton_ms": round(phase_ms[3], 6),
            "full_fused_ms": round(full_ms, 6),
            "tokps_total": round(1000.0 * tokens_total / phase_ms[3], 1) if phase_ms[3] > 0 else None,
            "component_ms_estimate": {k: round(v, 6) for k, v in component_estimates.items()},
            **(correctness if write_kv else {}),
        }
        summary_by_write_mode[write_kv] = summary
        print(json.dumps(summary, ensure_ascii=False))
        append_row(args.results, summary)

    if args.include_no_kv_write and True in summary_by_write_mode and False in summary_by_write_mode:
        with_kv = summary_by_write_mode[True]
        no_kv = summary_by_write_mode[False]
        kv_write_delta = float(with_kv["triton_ms"]) - float(no_kv["triton_ms"])
        phase0_kv_write_delta = (
            float((with_kv.get("component_ms_estimate") or {}).get("prep_norm_kv_w_ms", 0.0))
            - float((no_kv.get("component_ms_estimate") or {}).get("prep_norm_kv_w_ms", 0.0))
        )
        comparison = {
            "axis": "triton_state_scan_micro",
            "backend": "fused_recurrent_update",
            "bench_case": "fullhead_no_kv_write_delta_summary",
            "status": "pass",
            "device": device_name,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "tokens_total": tokens_total,
            "num_warps": args.num_warps,
            "num_stages": args.num_stages,
            "write_kv": None,
            "with_kv_triton_ms": with_kv["triton_ms"],
            "no_kv_triton_ms": no_kv["triton_ms"],
            "kv_write_delta_ms": round(kv_write_delta, 6),
            "phase0_kv_write_delta_ms": round(phase0_kv_write_delta, 6),
            "kv_write_delta_ratio": round(kv_write_delta / float(with_kv["triton_ms"]), 6)
            if float(with_kv["triton_ms"]) > 0
            else None,
        }
        print(json.dumps(comparison, ensure_ascii=False))
        append_row(args.results, comparison)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
