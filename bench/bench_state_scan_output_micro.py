#!/usr/bin/env python3
# coding=utf-8
"""Synthetic micro-split for RWKV-7 state-scan/output-prep route variants.

This benchmark targets the current sk-scale no-K/V-writeback branch.  It times
scan-only, output-prep-only, and scan+output route pairs for target-shaped
random tensors:

* full_kv: baseline full-head state-prep scan writes adjusted K/V, then output
  prep consumes recurrent + adjusted K/V;
* nokv_raw: scan skips adjusted K/V writeback, then output prep recomputes
  adjusted K and V interpolation from raw K/V/A;
* sk_raw_v: scan skips adjusted K/V writeback and emits one sk scalar per
  token/head, then output prep consumes sk + raw V;
* correction: scan skips adjusted K/V writeback but writes a full correction
  vector, then output prep consumes recurrent + correction.

Rows are synthetic direction evidence, not an HF promotion gate.
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
    fused_recurrent_scan_state_prep_correction,
    fused_recurrent_scan_state_prep_nokv,
    fused_recurrent_scan_state_prep_sk,
)
from rwkv7_hf.fused_output import (
    fused_attn_output_prepare,
    fused_attn_output_prepare_from_correction,
    fused_attn_output_prepare_from_sk_raw_v,
    fused_attn_output_prepare_raw_kv,
)


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
    hidden = args.heads * args.head_dim
    return {
        "r": torch.randn(shape, device=device, dtype=dtype),
        "w": torch.randn(shape, device=device, dtype=dtype),
        "k": torch.randn(shape, device=device, dtype=dtype),
        "v": torch.randn(shape, device=device, dtype=dtype),
        "a": torch.sigmoid(torch.randn(shape, device=device, dtype=dtype)),
        "state": torch.randn(
            (args.batch_size, args.heads, args.head_dim, args.head_dim),
            device=device,
            dtype=torch.float32,
        ),
        "k_k": torch.randn((args.heads, args.head_dim), device=device, dtype=dtype),
        "k_a": torch.randn((args.heads, args.head_dim), device=device, dtype=dtype),
        "r_k": torch.randn((args.heads, args.head_dim), device=device, dtype=dtype),
        "v_first": torch.randn(shape, device=device, dtype=dtype),
        "v_gate": torch.sigmoid(torch.randn(shape, device=device, dtype=dtype)),
        "g": torch.randn((args.batch_size, args.seq_len, hidden), device=device, dtype=dtype),
        "gn_w": torch.randn((hidden,), device=device, dtype=dtype),
        "gn_b": torch.randn((hidden,), device=device, dtype=dtype),
    }


def call_scan_full(t: dict[str, torch.Tensor], args: argparse.Namespace):
    return fused_recurrent_scan_state_prep(
        t["r"],
        t["w"],
        t["k"],
        t["v"],
        t["a"],
        t["state"],
        t["k_k"],
        t["k_a"],
        v_first=t["v_first"],
        v_gate=t["v_gate"],
        block_n=args.head_dim,
        block_m=args.head_dim,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )


def call_scan_nokv(t: dict[str, torch.Tensor], args: argparse.Namespace):
    return fused_recurrent_scan_state_prep_nokv(
        t["r"],
        t["w"],
        t["k"],
        t["v"],
        t["a"],
        t["state"],
        t["k_k"],
        t["k_a"],
        v_first=t["v_first"],
        v_gate=t["v_gate"],
        block_n=args.head_dim,
        block_m=args.head_dim,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )


def call_scan_sk(t: dict[str, torch.Tensor], args: argparse.Namespace):
    return fused_recurrent_scan_state_prep_sk(
        t["r"],
        t["w"],
        t["k"],
        t["v"],
        t["a"],
        t["state"],
        t["k_k"],
        t["k_a"],
        t["r_k"],
        v_first=t["v_first"],
        v_gate=t["v_gate"],
        block_n=args.head_dim,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )


def call_scan_correction(t: dict[str, torch.Tensor], args: argparse.Namespace):
    return fused_recurrent_scan_state_prep_correction(
        t["r"],
        t["w"],
        t["k"],
        t["v"],
        t["a"],
        t["state"],
        t["k_k"],
        t["k_a"],
        t["r_k"],
        v_first=t["v_first"],
        v_gate=t["v_gate"],
        block_n=args.head_dim,
        block_m=args.head_dim,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )


def output_full(t: dict[str, torch.Tensor], args: argparse.Namespace, scan_full: tuple[torch.Tensor, ...]):
    out, _state, k_adj, v_adj = scan_full
    b, seq, h, n = out.shape
    hidden = h * n
    return fused_attn_output_prepare(
        out.reshape(b * seq, hidden),
        t["r"].reshape(b * seq, h, n),
        k_adj.reshape(b * seq, h, n),
        v_adj.reshape(b * seq, h, n),
        t["g"].reshape(b * seq, hidden),
        t["r_k"],
        t["gn_w"],
        t["gn_b"],
        num_heads=h,
        head_dim=n,
        head_v_dim=n,
        eps=args.head_dim * 1e-5,
    )


def output_raw(t: dict[str, torch.Tensor], args: argparse.Namespace, scan_nokv: tuple[torch.Tensor, ...]):
    out, _state = scan_nokv
    b, seq, h, n = out.shape
    hidden = h * n
    return fused_attn_output_prepare_raw_kv(
        out.reshape(b * seq, hidden),
        t["r"].reshape(b * seq, h, n),
        t["k"].reshape(b * seq, h, n),
        t["v"].reshape(b * seq, h, n),
        t["a"].reshape(b * seq, h, n),
        t["g"].reshape(b * seq, hidden),
        t["k_a"],
        t["r_k"],
        t["gn_w"],
        t["gn_b"],
        v_first=t["v_first"].reshape(b * seq, h, n),
        v_gate=t["v_gate"].reshape(b * seq, h, n),
        num_heads=h,
        head_dim=n,
        head_v_dim=n,
        eps=args.head_dim * 1e-5,
    )


def output_sk(t: dict[str, torch.Tensor], args: argparse.Namespace, scan_sk: tuple[torch.Tensor, ...]):
    out, _state, sk = scan_sk
    b, seq, h, n = out.shape
    hidden = h * n
    return fused_attn_output_prepare_from_sk_raw_v(
        out.reshape(b * seq, hidden),
        sk.reshape(b * seq, h),
        t["v"].reshape(b * seq, h, n),
        t["g"].reshape(b * seq, hidden),
        t["gn_w"],
        t["gn_b"],
        v_first=t["v_first"].reshape(b * seq, h, n),
        v_gate=t["v_gate"].reshape(b * seq, h, n),
        num_heads=h,
        head_v_dim=n,
        eps=args.head_dim * 1e-5,
    )


def output_correction(t: dict[str, torch.Tensor], args: argparse.Namespace, scan_correction: tuple[torch.Tensor, ...]):
    out, _state, correction = scan_correction
    b, seq, h, n = out.shape
    hidden = h * n
    return fused_attn_output_prepare_from_correction(
        out.reshape(b * seq, hidden),
        correction.reshape(b * seq, hidden),
        t["g"].reshape(b * seq, hidden),
        t["gn_w"],
        t["gn_b"],
        num_heads=h,
        head_v_dim=n,
        eps=args.head_dim * 1e-5,
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
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--results", default="")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("bench_state_scan_output_micro requires CUDA")
    if args.head_dim != 64:
        raise ValueError("current micro-split is intended for the N=64 target shape")

    tensors = make_inputs(args)
    tokens_total = args.batch_size * args.seq_len
    device_name = torch.cuda.get_device_name(0)

    with torch.inference_mode():
        ref_scan = call_scan_full(tensors, args)
        nokv_scan = call_scan_nokv(tensors, args)
        sk_scan = call_scan_sk(tensors, args)
        corr_scan = call_scan_correction(tensors, args)
        ref_output = output_full(tensors, args, ref_scan)
        raw_output = output_raw(tensors, args, nokv_scan)
        sk_output = output_sk(tensors, args, sk_scan)
        corr_output = output_correction(tensors, args, corr_scan)

    correctness = {
        "nokv_out_max_abs_diff": round(max_abs(ref_scan[0], nokv_scan[0]), 8),
        "nokv_state_max_abs_diff": round(max_abs(ref_scan[1], nokv_scan[1]), 8),
        "sk_out_max_abs_diff": round(max_abs(ref_scan[0], sk_scan[0]), 8),
        "sk_state_max_abs_diff": round(max_abs(ref_scan[1], sk_scan[1]), 8),
        "correction_out_max_abs_diff": round(max_abs(ref_scan[0], corr_scan[0]), 8),
        "correction_state_max_abs_diff": round(max_abs(ref_scan[1], corr_scan[1]), 8),
        "raw_output_max_abs_diff": round(max_abs(ref_output, raw_output), 8),
        "sk_output_max_abs_diff": round(max_abs(ref_output, sk_output), 8),
        "correction_output_max_abs_diff": round(max_abs(ref_output, corr_output), 8),
    }
    status = "pass" if max(correctness.values()) <= 0.125 else "fail"

    # Precompute scan outputs for output-only timing.  This intentionally
    # excludes the scan launch and allocations from output-prep rows.
    scan_outputs = {
        "full_kv": ref_scan,
        "nokv_raw": nokv_scan,
        "sk_raw_v": sk_scan,
        "correction": corr_scan,
    }
    cases: list[tuple[str, Callable[[], Any]]] = [
        ("scan_full_kv", lambda: call_scan_full(tensors, args)),
        ("scan_nokv", lambda: call_scan_nokv(tensors, args)),
        ("scan_sk", lambda: call_scan_sk(tensors, args)),
        ("scan_correction", lambda: call_scan_correction(tensors, args)),
        ("output_full_kv", lambda: output_full(tensors, args, scan_outputs["full_kv"])),
        ("output_raw_kv", lambda: output_raw(tensors, args, scan_outputs["nokv_raw"])),
        ("output_sk_raw_v", lambda: output_sk(tensors, args, scan_outputs["sk_raw_v"])),
        ("output_correction", lambda: output_correction(tensors, args, scan_outputs["correction"])),
        ("route_full_kv", lambda: output_full(tensors, args, call_scan_full(tensors, args))),
        ("route_nokv_raw", lambda: output_raw(tensors, args, call_scan_nokv(tensors, args))),
        ("route_sk_raw_v", lambda: output_sk(tensors, args, call_scan_sk(tensors, args))),
        ("route_correction", lambda: output_correction(tensors, args, call_scan_correction(tensors, args))),
    ]

    timings: dict[str, float] = {}
    for bench_case, fn in cases:
        ms = median_ms(fn, warmup=args.warmup, steps=args.steps)
        timings[bench_case] = ms
        row = {
            "axis": "state_scan_output_micro",
            "backend": "hf_adapter",
            "bench_case": bench_case,
            "status": status,
            "device": device_name,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "heads": args.heads,
            "head_dim": args.head_dim,
            "tokens_total": tokens_total,
            "num_warps": args.num_warps,
            "num_stages": args.num_stages,
            "triton_ms": round(ms, 6),
            "tokps_total": round(1000.0 * tokens_total / ms, 1) if ms > 0 else None,
            **correctness,
        }
        print(json.dumps(row, ensure_ascii=False))
        append_row(args.results, row)

    summary = {
        "axis": "state_scan_output_micro",
        "backend": "hf_adapter",
        "bench_case": "summary",
        "status": status,
        "device": device_name,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "heads": args.heads,
        "head_dim": args.head_dim,
        "tokens_total": tokens_total,
        "num_warps": args.num_warps,
        "num_stages": args.num_stages,
        "component_ms": {k: round(v, 6) for k, v in timings.items()},
        "delta_ms": {
            "scan_sk_minus_full_kv": round(timings["scan_sk"] - timings["scan_full_kv"], 6),
            "scan_nokv_minus_full_kv": round(timings["scan_nokv"] - timings["scan_full_kv"], 6),
            "output_sk_minus_full_kv": round(timings["output_sk_raw_v"] - timings["output_full_kv"], 6),
            "route_sk_minus_full_kv": round(timings["route_sk_raw_v"] - timings["route_full_kv"], 6),
            "route_nokv_raw_minus_full_kv": round(timings["route_nokv_raw"] - timings["route_full_kv"], 6),
            "route_correction_minus_full_kv": round(timings["route_correction"] - timings["route_full_kv"], 6),
        },
        **correctness,
    }
    print(json.dumps(summary, ensure_ascii=False))
    append_row(args.results, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
