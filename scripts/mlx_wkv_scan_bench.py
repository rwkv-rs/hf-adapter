#!/usr/bin/env python3
# coding=utf-8
"""Microbenchmark the MLX multi-token WKV scan seam.

This isolates the recurrent WKV update only: projections and layer norms are not
included.  It compares the existing per-token Metal WKV update loop with the new
single-launch sequence scan kernel, which is the core kernel needed before the
full MLX prefill path can become layer-major.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path


def append_jsonl(path: str | Path | None, row: dict) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def bench(fn, *, warmup: int, runs: int, mx):
    for _ in range(int(warmup)):
        values = fn()
        mx.eval(*values)
    times = []
    last = None
    for _ in range(int(runs)):
        t0 = time.perf_counter()
        last = fn()
        mx.eval(*last)
        times.append(time.perf_counter() - t0)
    return sum(times) / max(len(times), 1), last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--results", default="")
    args = ap.parse_args()

    import mlx.core as mx

    from rwkv7_hf.mlx_scan import metal_wkv_scan_available, wkv_scan
    from rwkv7_hf.mlx_wkv import metal_wkv_available, wkv_update

    B, T, H, N = int(args.batch), int(args.tokens), int(args.heads), int(args.head_dim)
    mx.random.seed(20260707)
    state = (mx.random.normal((B, H, N, N)).astype(mx.float32) * 0.01)
    r = mx.random.normal((B, T, H, N)).astype(mx.float16)
    w = mx.sigmoid(mx.random.normal((B, T, H, N)).astype(mx.float32)).astype(mx.float16)
    v = mx.random.normal((B, T, H, N)).astype(mx.float16)
    k = mx.random.normal((B, T, H, N)).astype(mx.float16)
    kk_raw = mx.random.normal((B, T, H, N)).astype(mx.float32)
    kk = (kk_raw / mx.sqrt(mx.maximum(mx.sum(kk_raw * kk_raw, axis=-1, keepdims=True), 1e-12))).astype(mx.float16)
    a = mx.sigmoid(mx.random.normal((B, T, H, N)).astype(mx.float32)).astype(mx.float16)
    mx.eval(state, r, w, v, k, kk, a)

    if not metal_wkv_available() or not metal_wkv_scan_available():
        row = {
            "axis": "mlx_wkv_scan_bench",
            "status": "skip",
            "reason": "MLX Metal WKV or scan kernels unavailable",
            "platform": platform.platform(),
            "machine": platform.machine(),
        }
        print(json.dumps(row, ensure_ascii=False))
        append_jsonl(args.results, row)
        return 0

    def loop_metal():
        cur = state
        outs = []
        for t in range(T):
            out_t, cur, _backend = wkv_update(cur, w[:, t], v[:, t], k[:, t], kk[:, t], a[:, t], r[:, t], backend="metal")
            outs.append(out_t)
        return (mx.stack(outs, axis=1), cur)

    def scan_metal():
        out, cur, _backend = wkv_scan(state, w, v, k, kk, a, r, backend="metal")
        return (out, cur)

    loop_s, loop_last = bench(loop_metal, warmup=args.warmup, runs=args.runs, mx=mx)
    scan_s, scan_last = bench(scan_metal, warmup=args.warmup, runs=args.runs, mx=mx)
    loop_out, loop_state = loop_last
    scan_out, scan_state = scan_last
    max_abs_out = float(mx.max(mx.abs(loop_out.astype(mx.float32) - scan_out.astype(mx.float32))))
    max_abs_state = float(mx.max(mx.abs(loop_state.astype(mx.float32) - scan_state.astype(mx.float32))))
    row = {
        "axis": "mlx_wkv_scan_bench",
        "status": "pass",
        "batch": B,
        "tokens": T,
        "heads": H,
        "head_dim": N,
        "warmup": int(args.warmup),
        "runs": int(args.runs),
        "loop_metal_s": round(float(loop_s), 6),
        "scan_metal_s": round(float(scan_s), 6),
        "speedup_scan_vs_loop": round(float(loop_s / scan_s), 6) if scan_s > 0 else None,
        "loop_tok_s": round(float(B * T / loop_s), 6) if loop_s > 0 else None,
        "scan_tok_s": round(float(B * T / scan_s), 6) if scan_s > 0 else None,
        "max_abs_out_vs_loop": round(max_abs_out, 8),
        "max_abs_state_vs_loop": round(max_abs_state, 8),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    print(json.dumps(row, ensure_ascii=False))
    append_jsonl(args.results, row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
