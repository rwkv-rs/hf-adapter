#!/usr/bin/env python3
# coding=utf-8
"""Speed bench: fp16 F.linear vs native int4 (mm4) fused Triton GEMV.

The 4-bit path's sweet spot is the largest layers (lm_head, 13B+ body), where
the 4x bandwidth saving + paired-nibble load beats both fp16 and int8 (mm8).
Small/launch-bound layers lose, but the size-gate keeps those in fp16.
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from rwkv7_hf.native_quant_mm4 import quantize_mm4, mm4_gemv_triton


def bench(fn, iters: int = 200, warmup: int = 30) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1000


def sweep(sizes):
    print("GPU:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
    print("%-14s %9s %9s %9s  %9s %9s" %
          ("size NxM", "fp16 ms", "mm4 ms", "cos vs fp16", "speedup", "VRAM x"))
    for (N, M) in sizes:
        w = torch.randn(M, N, dtype=torch.float16, device="cuda")
        x = torch.randn(N, dtype=torch.float16, device="cuda")
        packed, mx, rx_s, my, ry_s, m_orig, _ = quantize_mm4(w.t().contiguous())
        with torch.no_grad():
            cos = F.cosine_similarity(
                mm4_gemv_triton(x, packed, mx, rx_s, my, ry_s, m_orig).unsqueeze(0),
                F.linear(x, w).unsqueeze(0)).item()
        t_fp = bench(lambda: F.linear(x, w))
        t_mm4 = bench(lambda: mm4_gemv_triton(x, packed, mx, rx_s, my, ry_s, m_orig))
        print("%-14s %9.4f %9.4f %9.4f  %8.2fx %8s" % (
            f"{N}x{M}", t_fp, t_mm4, cos, t_fp / t_mm4, "4x"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="768x768,2048x2048,4096x4096,8192x8192,768x65536")
    args = ap.parse_args()
    sizes = [tuple(int(v) for v in s.split("x")) for s in args.sizes.split(",")]
    sweep(sizes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
