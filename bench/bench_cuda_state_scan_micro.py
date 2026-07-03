#!/usr/bin/env python3
# coding=utf-8
"""Synthetic micro-profiler for the experimental CUDA N=64 state-scan path.

This benchmark intentionally stays below the HF/model layer.  It times the
row-block CUDA scaffold on target-shaped random tensors and emits cumulative
phase rows:

* phase 0: duplicated vector prep + K normalization inside the row-block grid
* phase 1: phase 0 + state-dot-KK reduction
* phase 2: phase 1 + recurrent state update
* phase 3: phase 2 + recurrent output reduction

The phase deltas are approximate because they are compiled as separate
profiling kernels, but they give a more useful direction signal than another
full-HF row when deciding whether to continue CUDA persistent/inter-CTA work or
return to Triton/DPLR apply-output fusion.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Callable

import torch

from rwkv7_hf.cuda_state_scan import cuda_state_scan_prep, cuda_state_scan_prep_sk, cuda_state_scan_rowblock_phase


PHASE_NAMES = {
    0: "prep_norm",
    1: "prep_norm_state_dot",
    2: "prep_norm_state_dot_update",
    3: "prep_norm_state_dot_update_recurrent",
}


def append_row(path: str, row: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def median_ms(fn: Callable[[], Any], *, warmup: int, steps: int, setup: Callable[[], Any] | None = None) -> float:
    for _ in range(warmup):
        if setup is not None:
            setup()
        fn()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(steps):
        if setup is not None:
            setup()
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
    shape = (args.batch_size, args.seq_len, args.heads, 64)
    w = torch.randn(shape, device=device, dtype=torch.float16)
    return {
        "r": torch.randn(shape, device=device, dtype=torch.float16),
        "w": w,
        "w_decay": torch.exp(-0.606531 * torch.sigmoid(w.float())).to(torch.float16),
        "k": torch.randn(shape, device=device, dtype=torch.float16),
        "v": torch.randn(shape, device=device, dtype=torch.float16),
        "a": torch.randn(shape, device=device, dtype=torch.float16),
        "state": torch.randn((args.batch_size, args.heads, 64, 64), device=device, dtype=torch.float32),
        "k_k": torch.randn((args.heads, 64), device=device, dtype=torch.float16),
        "k_a": torch.randn((args.heads, 64), device=device, dtype=torch.float16),
        "r_k": torch.randn((args.heads, 64), device=device, dtype=torch.float16),
        "v_first": torch.randn(shape, device=device, dtype=torch.float16),
        "v_gate": torch.sigmoid(torch.randn(shape, device=device, dtype=torch.float16)),
    }


def call_phase(tensors: dict[str, torch.Tensor], phase: int):
    return cuda_state_scan_rowblock_phase(
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
    )


def call_full(
    tensors: dict[str, torch.Tensor],
    *,
    rows_per_block: int = 1,
    schedule: str = "default",
    precompute_mode: str = "none",
    w_precomputed: bool = False,
    inplace_kv: bool = False,
    inplace_kka: bool = False,
):
    return cuda_state_scan_prep(
        tensors["r"],
        tensors["w_decay"] if w_precomputed else tensors["w"],
        tensors["k"],
        tensors["v"],
        tensors["a"],
        tensors["state"],
        tensors["k_k"],
        tensors["k_a"],
        v_first=tensors["v_first"],
        v_gate=tensors["v_gate"],
        lanes_per_row=64,
        precompute_mode=precompute_mode,
        rows_per_block=rows_per_block,
        schedule=schedule,
        w_precomputed=w_precomputed,
        inplace_kv=inplace_kv,
        inplace_kka=inplace_kka,
    )


def call_full_inplace_scratch(
    tensors: dict[str, torch.Tensor],
    k_scratch: torch.Tensor,
    v_scratch: torch.Tensor,
    a_scratch: torch.Tensor,
    *,
    rows_per_block: int,
    schedule: str,
    inplace_kka: bool = False,
):
    return cuda_state_scan_prep(
        tensors["r"],
        tensors["w"],
        k_scratch,
        v_scratch,
        a_scratch,
        tensors["state"],
        tensors["k_k"],
        tensors["k_a"],
        v_first=tensors["v_first"],
        v_gate=tensors["v_gate"],
        lanes_per_row=64,
        precompute_mode="wk_half",
        rows_per_block=rows_per_block,
        schedule=schedule,
        inplace_kv=True,
        inplace_kka=inplace_kka,
    )


def call_full_sk(tensors: dict[str, torch.Tensor], *, rows_per_block: int = 1, schedule: str = "warp_specialized"):
    return cuda_state_scan_prep_sk(
        tensors["r"],
        tensors["w"],
        tensors["k"],
        tensors["v"],
        tensors["a"],
        tensors["state"],
        tensors["k_k"],
        tensors["k_a"],
        tensors["r_k"],
        v_first=tensors["v_first"],
        v_gate=tensors["v_gate"],
        rows_per_block=rows_per_block,
        schedule=schedule,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--results", default="")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("bench_cuda_state_scan_micro requires CUDA")
    tensors = make_inputs(args)
    tokens_total = args.batch_size * args.seq_len
    device_name = torch.cuda.get_device_name(0)

    phase_ms: dict[int, float] = {}
    for phase in range(4):
        ms = median_ms(lambda phase=phase: call_phase(tensors, phase), warmup=args.warmup, steps=args.steps)
        phase_ms[phase] = ms
        row = {
            "axis": "cuda_state_scan_micro",
            "backend": "cuda_state_scan",
            "bench_case": f"rowblock_phase_{phase}_{PHASE_NAMES[phase]}",
            "status": "pass",
            "device": device_name,
            "dtype": "fp16",
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "heads": args.heads,
            "head_dim": 64,
            "tokens_total": tokens_total,
            "phase": phase,
            "phase_name": PHASE_NAMES[phase],
            "cuda_ms": round(ms, 6),
            "tokps_total": round(1000.0 * tokens_total / ms, 1) if ms > 0 else None,
        }
        print(json.dumps(row, ensure_ascii=False))
        append_row(args.results, row)

    component_estimates = {
        "duplicated_vector_prep_norm_ms": phase_ms[0],
        "state_dot_delta_ms": phase_ms[1] - phase_ms[0],
        "state_update_delta_ms": phase_ms[2] - phase_ms[1],
        "recurrent_output_delta_ms": phase_ms[3] - phase_ms[2],
    }
    summary = {
        "axis": "cuda_state_scan_micro",
        "backend": "cuda_state_scan",
        "bench_case": "rowblock_phase_delta_summary",
        "status": "pass",
        "device": device_name,
        "dtype": "fp16",
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "heads": args.heads,
        "head_dim": 64,
        "tokens_total": tokens_total,
        "cuda_ms": round(phase_ms[3], 6),
        "component_ms_estimate": {k: round(v, 6) for k, v in component_estimates.items()},
    }
    print(json.dumps(summary, ensure_ascii=False))
    append_row(args.results, summary)

    for schedule, rpb in [
        ("default", 1),
        ("default", 16),
        ("warp_specialized", 1),
        ("warp_specialized", 8),
        ("warp_specialized", 16),
        ("warp2", 1),
        ("warp2", 2),
        ("warp2", 4),
        ("warp2", 8),
        ("warp_pair", 2),
        ("warp_pair", 4),
        ("warp_pair", 8),
        ("warp_pair", 16),
        ("halfwarp_pair", 2),
        ("halfwarp_pair", 4),
        ("halfwarp_pair", 8),
        ("halfwarp_pair", 16),
        ("head_reg16", 1),
        ("head_reg8", 1),
        ("warp_pipelined", 1),
        ("warp_pipelined", 8),
        ("warp_pipelined", 16),
        ("warp_pipelined_half", 1),
        ("warp_pipelined_half", 8),
        ("warp_pipelined_half", 16),
        ("precomputed_warp", 1),
        ("precomputed_warp", 4),
        ("precomputed_warp", 8),
        ("precomputed_warp", 16),
    ]:
        precompute_mode = "wk_half" if schedule == "precomputed_warp" else "none"
        ms = median_ms(
            lambda schedule=schedule, rpb=rpb, precompute_mode=precompute_mode: call_full(
                tensors,
                rows_per_block=rpb,
                schedule=schedule,
                precompute_mode=precompute_mode,
            ),
            warmup=args.warmup,
            steps=args.steps,
        )
        row = {
            "axis": "cuda_state_scan_micro",
            "backend": "cuda_state_scan",
            "bench_case": f"full_{schedule}_rpb{rpb}",
            "status": "pass",
            "device": device_name,
            "dtype": "fp16",
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "heads": args.heads,
            "head_dim": 64,
            "tokens_total": tokens_total,
            "schedule": schedule,
            "rows_per_block": rpb,
            "precompute_mode": precompute_mode,
            "cuda_ms": round(ms, 6),
            "tokps_total": round(1000.0 * tokens_total / ms, 1) if ms > 0 else None,
        }
        print(json.dumps(row, ensure_ascii=False))
        append_row(args.results, row)
    for mode in ["full", "wk", "wk_half"]:
        ms = median_ms(
            lambda mode=mode: call_full(tensors, rows_per_block=1, schedule="default", precompute_mode=mode),
            warmup=args.warmup,
            steps=args.steps,
        )
        row = {
            "axis": "cuda_state_scan_micro",
            "backend": "cuda_state_scan",
            "bench_case": f"full_precompute_{mode}_rpb1",
            "status": "pass",
            "device": device_name,
            "dtype": "fp16",
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "heads": args.heads,
            "head_dim": 64,
            "tokens_total": tokens_total,
            "schedule": "default",
            "rows_per_block": 1,
            "precompute_mode": mode,
            "cuda_ms": round(ms, 6),
            "tokps_total": round(1000.0 * tokens_total / ms, 1) if ms > 0 else None,
        }
        print(json.dumps(row, ensure_ascii=False))
        append_row(args.results, row)
    for schedule, rpb in [("default", 1), ("precomputed_warp", 8), ("precomputed_warp", 16)]:
        k_scratch = torch.empty_like(tensors["k"])
        v_scratch = torch.empty_like(tensors["v"])
        a_scratch = torch.empty_like(tensors["a"])
        setup_inplace = lambda k_scratch=k_scratch, v_scratch=v_scratch, a_scratch=a_scratch: (
            k_scratch.copy_(tensors["k"]),
            v_scratch.copy_(tensors["v"]),
            a_scratch.copy_(tensors["a"]),
        )
        ms = median_ms(
            lambda schedule=schedule, rpb=rpb, k_scratch=k_scratch, v_scratch=v_scratch, a_scratch=a_scratch: call_full_inplace_scratch(
                tensors,
                k_scratch,
                v_scratch,
                a_scratch,
                rows_per_block=rpb,
                schedule=schedule,
            ),
            warmup=args.warmup,
            steps=args.steps,
            setup=setup_inplace,
        )
        row = {
            "axis": "cuda_state_scan_micro",
            "backend": "cuda_state_scan",
            "bench_case": f"full_{schedule}_wkhalf_inplace_kv_rpb{rpb}",
            "status": "pass",
            "device": device_name,
            "dtype": "fp16",
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "heads": args.heads,
            "head_dim": 64,
            "tokens_total": tokens_total,
            "schedule": schedule,
            "rows_per_block": rpb,
            "precompute_mode": "wk_half",
            "inplace_kv": True,
            "cuda_ms": round(ms, 6),
            "tokps_total": round(1000.0 * tokens_total / ms, 1) if ms > 0 else None,
        }
        print(json.dumps(row, ensure_ascii=False))
        append_row(args.results, row)
    for schedule, rpb in [("default", 1), ("precomputed_warp", 8), ("precomputed_warp", 16)]:
        k_scratch = torch.empty_like(tensors["k"])
        v_scratch = torch.empty_like(tensors["v"])
        a_scratch = torch.empty_like(tensors["a"])
        setup_inplace = lambda k_scratch=k_scratch, v_scratch=v_scratch, a_scratch=a_scratch: (
            k_scratch.copy_(tensors["k"]),
            v_scratch.copy_(tensors["v"]),
            a_scratch.copy_(tensors["a"]),
        )
        ms = median_ms(
            lambda schedule=schedule, rpb=rpb, k_scratch=k_scratch, v_scratch=v_scratch, a_scratch=a_scratch: call_full_inplace_scratch(
                tensors,
                k_scratch,
                v_scratch,
                a_scratch,
                rows_per_block=rpb,
                schedule=schedule,
                inplace_kka=True,
            ),
            warmup=args.warmup,
            steps=args.steps,
            setup=setup_inplace,
        )
        row = {
            "axis": "cuda_state_scan_micro",
            "backend": "cuda_state_scan",
            "bench_case": f"full_{schedule}_wkhalf_inplace_kv_kka_rpb{rpb}",
            "status": "pass",
            "device": device_name,
            "dtype": "fp16",
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "heads": args.heads,
            "head_dim": 64,
            "tokens_total": tokens_total,
            "schedule": schedule,
            "rows_per_block": rpb,
            "precompute_mode": "wk_half",
            "inplace_kv": True,
            "inplace_kka": True,
            "cuda_ms": round(ms, 6),
            "tokps_total": round(1000.0 * tokens_total / ms, 1) if ms > 0 else None,
        }
        print(json.dumps(row, ensure_ascii=False))
        append_row(args.results, row)
    for rpb in [1, 8]:
        ms = median_ms(
            lambda rpb=rpb: call_full(tensors, rows_per_block=rpb, schedule="warp_specialized", w_precomputed=True),
            warmup=args.warmup,
            steps=args.steps,
        )
        row = {
            "axis": "cuda_state_scan_micro",
            "backend": "cuda_state_scan",
            "bench_case": f"full_warp_specialized_wpre_rpb{rpb}",
            "status": "pass",
            "device": device_name,
            "dtype": "fp16",
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "heads": args.heads,
            "head_dim": 64,
            "tokens_total": tokens_total,
            "schedule": "warp_specialized",
            "rows_per_block": rpb,
            "w_precomputed": True,
            "cuda_ms": round(ms, 6),
            "tokps_total": round(1000.0 * tokens_total / ms, 1) if ms > 0 else None,
        }
        print(json.dumps(row, ensure_ascii=False))
        append_row(args.results, row)
    for rpb in [8, 16]:
        ms = median_ms(
            lambda rpb=rpb: call_full(tensors, rows_per_block=rpb, schedule="warp_pipelined", w_precomputed=True),
            warmup=args.warmup,
            steps=args.steps,
        )
        row = {
            "axis": "cuda_state_scan_micro",
            "backend": "cuda_state_scan",
            "bench_case": f"full_warp_pipelined_wpre_rpb{rpb}",
            "status": "pass",
            "device": device_name,
            "dtype": "fp16",
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "heads": args.heads,
            "head_dim": 64,
            "tokens_total": tokens_total,
            "schedule": "warp_pipelined",
            "rows_per_block": rpb,
            "w_precomputed": True,
            "cuda_ms": round(ms, 6),
            "tokps_total": round(1000.0 * tokens_total / ms, 1) if ms > 0 else None,
        }
        print(json.dumps(row, ensure_ascii=False))
        append_row(args.results, row)
    for rpb in [1, 2, 4, 8, 16]:
        ms = median_ms(
            lambda rpb=rpb: call_full_sk(tensors, rows_per_block=rpb),
            warmup=args.warmup,
            steps=args.steps,
        )
        row = {
            "axis": "cuda_state_scan_micro",
            "backend": "cuda_state_scan",
            "bench_case": f"full_warp_specialized_sk_rpb{rpb}",
            "status": "pass",
            "device": device_name,
            "dtype": "fp16",
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "heads": args.heads,
            "head_dim": 64,
            "tokens_total": tokens_total,
            "schedule": "warp_specialized",
            "rows_per_block": rpb,
            "cuda_state_scan_sk": True,
            "cuda_ms": round(ms, 6),
            "tokps_total": round(1000.0 * tokens_total / ms, 1) if ms > 0 else None,
        }
        print(json.dumps(row, ensure_ascii=False))
        append_row(args.results, row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
