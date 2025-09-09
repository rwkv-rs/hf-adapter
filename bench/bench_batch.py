#!/usr/bin/env python3
# coding=utf-8
"""Throughput across batch sizes (fla HF path, which natively handles [B,T]).

Prefill: one forward of [B,T] -> tok/s = B*T/time.
Decode : thread a B-sequence recurrent cache, single-token steps -> tok/s = B*N/time.

Usage: python bench/bench_batch.py --hf-dir <dir> [--prompt-tokens 128] [--batches 1 2 4 8]
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("RWKV_V7_ON", "1")
import torch
from transformers import AutoModelForCausalLM


def measure(model, T, decode_n, batches, device="cuda"):
    base = model.model
    V = base.embeddings.weight.shape[0]
    print(f"{'B':>3} {'prefill tok/s':>14} {'decode tok/s':>13} {'peak VRAM MB':>13}")
    for B in batches:
        ids = torch.randint(0, V, (B, T), device=device)
        torch.cuda.reset_peak_memory_stats()
        # prefill
        try:
            with torch.no_grad():
                for _ in range(2):
                    model(ids, use_cache=False)
            torch.cuda.synchronize(); t0 = time.time()
            with torch.no_grad():
                for _ in range(5):
                    model(ids, use_cache=False)
            torch.cuda.synchronize()
            prefill = B * T / ((time.time() - t0) / 5)
        except RuntimeError as e:
            prefill = float("nan")
        # decode
        try:
            with torch.no_grad():
                out = model(ids[:, :8], use_cache=True)
                st = out.past_key_values
                nx = out.logits[:, -1:].argmax(dim=-1)
                for _ in range(3):
                    out = model(nx, past_key_values=st, use_cache=True)
                    st = out.past_key_values
                    nx = out.logits[:, -1:].argmax(dim=-1)
            torch.cuda.synchronize(); t0 = time.time()
            with torch.no_grad():
                for _ in range(decode_n):
                    out = model(nx, past_key_values=st, use_cache=True)
                    st = out.past_key_values
                    nx = out.logits[:, -1:].argmax(dim=-1)
            torch.cuda.synchronize()
            decode = B * decode_n / (time.time() - t0)
        except RuntimeError as e:
            decode = float("nan")
        vram = torch.cuda.max_memory_allocated() / 1048576
        print(f"{B:>3} {prefill:>14.0f} {decode:>13.0f} {vram:>13.0f}")
        del ids
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--decode-tokens", type=int, default=32)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    args = ap.parse_args()
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_dir, trust_remote_code=True, torch_dtype=dt, device_map="cuda").eval()
    print(f"\n=== {args.hf_dir} | {args.dtype} | T={args.prompt_tokens} ===")
    measure(model, args.prompt_tokens, args.decode_tokens, args.batches)


if __name__ == "__main__":
    main()
