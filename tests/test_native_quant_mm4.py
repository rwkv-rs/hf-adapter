#!/usr/bin/env python3
# coding=utf-8
"""Correctness gate for the native int4 (mm4) quantization path.

Verifies:
1. Per-layer: int4 (mm4) dequant matmul vs fp16 F.linear, cosine floor (int4 is
   lossier than int8, so the floor is lower than test_native_quant_mm8).
2. Triton fused GEMV vs the torch reference, max_abs floor.
3. End-to-end: size-gated quantize_model_mm4 forward logits vs fp16.

Run: python tests/test_native_quant_mm4.py --model <hf_dir>
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("RWKV_V7_ON", "1")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from rwkv7_hf.native_quant_mm4 import (
    quantize_mm4,
    mm4_matmul,
    mm4_gemv_triton,
    mm4_gemv_available,
    quantize_model_mm4,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--per-layer-cos-min", type=float, default=0.98)
    ap.add_argument("--e2e-cos-min", type=float, default=0.998)
    ap.add_argument("--triton-max-abs", type=float, default=0.5)
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float16, device_map="cuda").eval()
    torch.manual_seed(0)
    linears = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]

    # 1. per-layer int4 vs fp16
    worst = 1.0
    for name, lin in linears:
        w = lin.weight.detach()
        packed, mx, rx_s, my, ry_s, m_orig, _ = quantize_mm4(w.t().contiguous())
        x = torch.randn(8, w.shape[1], dtype=w.dtype, device=w.device)
        with torch.no_grad():
            ref = lin(x)
            q = mm4_matmul(x, packed, mx, rx_s, my, ry_s, m_orig)
            if lin.bias is not None:
                q = q + lin.bias
        cos = F.cosine_similarity(ref.flatten().unsqueeze(0), q.flatten().unsqueeze(0)).item()
        worst = min(worst, cos)
    print(f"per-layer worst cos = {worst:.6f} (>= {args.per_layer_cos_min})", flush=True)
    ok = worst >= args.per_layer_cos_min

    # 2. triton fused GEMV vs torch reference
    if mm4_gemv_available():
        lin = linears[0][1]
        w = lin.weight.detach()
        packed, mx, rx_s, my, ry_s, m_orig, _ = quantize_mm4(w.t().contiguous())
        x1 = torch.randn(w.shape[1], dtype=w.dtype, device=w.device)
        with torch.no_grad():
            ref = mm4_matmul(x1, packed, mx, rx_s, my, ry_s, m_orig)
            t = mm4_gemv_triton(x1, packed, mx, rx_s, my, ry_s, m_orig)
        d = (t - ref).abs().max().item()
        print(f"triton vs torch-ref max_abs = {d:.6f} (<= {args.triton_max_abs})", flush=True)
        ok = ok and d <= args.triton_max_abs

    # 3. end-to-end size-gated quantization
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ids = tok("The quick brown fox jumps over the lazy dog.",
              return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    with torch.no_grad():
        ref = model(ids).logits[0, -1].float().cpu()
    n = quantize_model_mm4(model, min_params=8_000_000)
    with torch.no_grad():
        q = model(ids).logits[0, -1].float().cpu()
    e2e = F.cosine_similarity(ref.unsqueeze(0), q.unsqueeze(0)).item()
    print(f"e2e ({n} layer(s) quantized) cos = {e2e:.6f} (>= {args.e2e_cos_min})", flush=True)
    ok = ok and e2e >= args.e2e_cos_min and n >= 1

    if not ok:
        print("FAIL", flush=True)
        return 1
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
