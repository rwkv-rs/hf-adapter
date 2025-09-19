#!/usr/bin/env python3
# coding=utf-8
"""Speed bench: fp16 F.linear vs ported int8 (mm8) Triton GEMV (naive + split-K).

Demonstrates the launch->memory-bound crossover: int8 loses on tiny launch-bound
layers and wins on large memory-bound layers; the crossover is GPU-dependent
(wins wider on Blackwell than on V100, whose cuBLAS fp16 GEMV is near peak).

No model required (synthetic weights). Add --hf-dir to also bench a real
model's representative layer weights.
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from rwkv7_hf.native_quant_mm8 import quantize_mm8, mm8_gemv_triton, mm8_gemv_triton_sk


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
          ("size NxM", "fp16 ms", "naive ms", "splitK ms", "naive x", "splitK x"))
    for (N, M) in sizes:
        w = torch.randn(M, N, dtype=torch.float16, device="cuda")
        x = torch.randn(N, dtype=torch.float16, device="cuda")
        wu8, mx, rx, my, ry = quantize_mm8(w.t().contiguous())
        t_fp = bench(lambda: F.linear(x, w))
        t_naive = bench(lambda: mm8_gemv_triton(x, wu8, mx, rx, my, ry))
        t_sk = bench(lambda: mm8_gemv_triton_sk(x, wu8, mx, rx, my, ry))
        print("%-14s %9.4f %9.4f %9.4f  %8.2fx %8.2fx" % (
            f"{N}x{M}", t_fp, t_naive, t_sk, t_fp / t_naive, t_fp / t_sk))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", default="")
    ap.add_argument("--sizes", default="768x768,2048x2048,4096x4096,8192x8192,768x65536")
    args = ap.parse_args()
    sizes = [tuple(int(v) for v in s.split("x")) for s in args.sizes.split(",")]
    sweep(sizes)
    if args.hf_dir:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            args.hf_dir, trust_remote_code=True, torch_dtype=torch.float16, device_map="cuda").eval()
        wanted = ["attn.r_proj", "attn.o_proj", "ffn.key", "ffn.value", "lm_head"]
        print("\n-- real weights --")
        print("%-12s %-14s %9s %9s %9s  %9s %9s" %
              ("layer", "shape", "fp16 ms", "naive ms", "splitK ms", "naive x", "splitK x"))
        seen = set()
        for n, m in model.named_modules():
            if not isinstance(m, torch.nn.Linear):
                continue
            for w in wanted:
                if n.endswith(w) and w not in seen:
                    seen.add(w)
                    wt = m.weight.detach()
                    wu8, mx, rx, my, ry = quantize_mm8(wt.t().contiguous())
                    x = torch.randn(wt.shape[1], dtype=wt.dtype, device=wt.device)
                    tf = bench(lambda: F.linear(x, wt))
                    tn = bench(lambda: mm8_gemv_triton(x, wu8, mx, rx, my, ry))
                    tsk = bench(lambda: mm8_gemv_triton_sk(x, wu8, mx, rx, my, ry))
                    print("%-12s %-14s %9.4f %9.4f %9.4f  %8.2fx %8.2fx" % (
                        w, str(tuple(wt.shape)), tf, tn, tsk, tf / tn, tf / tsk))
                    break
            if len(seen) == len(wanted):
                break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
