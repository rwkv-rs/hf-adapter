#!/usr/bin/env python3
# coding=utf-8
"""Correctness gate for the ported official int8 (mm8) quantization path.

Verifies:
1. Per-layer: int8 (mm8) dequant matmul vs fp16 F.linear, cosine floor.
2. Triton fused GEMV (naive + split-K) vs the torch reference, max_abs floor.
3. End-to-end: size-gated quantize_model_mm8 forward logits vs fp16.

Run: python tests/test_native_quant_mm8.py --model <hf_dir>
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("RWKV_V7_ON", "1")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from rwkv7_hf.native_quant_mm8 import (
    quantize_mm8,
    mm8_matmul,
    mm8_gemv_triton,
    mm8_gemv_triton_sk,
    mm8_batched_gemv_triton,
    mm8_batched_dot_triton,
    mm8_matmul_triton,
    mm8_gemv_available,
    quantize_model_mm8,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--per-layer-cos-min", type=float, default=0.999)
    ap.add_argument("--e2e-cos-min", type=float, default=0.999)
    ap.add_argument("--triton-max-abs", type=float, default=0.5)
    ap.add_argument("--fast-token-cos-min", type=float, default=0.999)
    ap.add_argument("--relu2-cos-min", type=float, default=0.999)
    args = ap.parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float16, device_map="cuda").eval()
    torch.manual_seed(0)
    linears = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]

    # 1. per-layer int8 vs fp16
    worst = 1.0
    for name, lin in linears:
        w = lin.weight.detach()
        wu8, mx, rx, my, ry = quantize_mm8(w.t().contiguous())
        x = torch.randn(8, w.shape[1], dtype=w.dtype, device=w.device)
        with torch.no_grad():
            ref = lin(x)
            q = mm8_matmul(x, wu8, mx, rx, my, ry)
            if lin.bias is not None:
                q = q + lin.bias
        cos = F.cosine_similarity(ref.flatten().unsqueeze(0), q.flatten().unsqueeze(0)).item()
        worst = min(worst, cos)
    print(f"per-layer worst cos = {worst:.6f} (>= {args.per_layer_cos_min})", flush=True)
    ok = worst >= args.per_layer_cos_min

    # 2. triton fused GEMV vs torch reference
    if mm8_gemv_available():
        lin = linears[0][1]
        w = lin.weight.detach()
        wu8, mx, rx, my, ry = quantize_mm8(w.t().contiguous())
        x1 = torch.randn(w.shape[1], dtype=w.dtype, device=w.device)
        with torch.no_grad():
            ref = mm8_matmul(x1, wu8, mx, rx, my, ry)
            t = mm8_gemv_triton(x1, wu8, mx, rx, my, ry)
            relu2 = mm8_gemv_triton(x1, wu8, mx, rx, my, ry, relu2=True)
            sk = mm8_gemv_triton_sk(x1, wu8, mx, rx, my, ry)
        d = (t - ref).abs().max().item()
        dsk = (sk - ref).abs().max().item()
        print(f"triton vs torch-ref max_abs = {d:.6f}; split-K = {dsk:.6f} (<= {args.triton_max_abs})", flush=True)
        ok = ok and d <= args.triton_max_abs and dsk <= args.triton_max_abs
        relu2_cos = F.cosine_similarity(
            relu2.float().unsqueeze(0), (torch.relu(t) ** 2).float().unsqueeze(0),
        ).item()
        print(f"fused relu2 cosine = {relu2_cos:.6f} (>= {args.relu2_cos_min})", flush=True)
        ok = ok and relu2_cos >= args.relu2_cos_min

        # Blackwell decode uses one launch for the complete batch.  Exercise
        # both the low-batch GEMV kernel and the tensor-core path selected for
        # bsz>=4, including the public dispatcher used by MM8Linear.forward.
        batched_cases = [(2, mm8_batched_gemv_triton)]
        if torch.cuda.get_device_capability(x1.device)[0] >= 12:
            batched_cases.append((8, mm8_batched_dot_triton))
        for batch, kernel in batched_cases:
            xb = torch.randn(batch, w.shape[1], dtype=w.dtype, device=w.device)
            with torch.no_grad():
                refb = mm8_matmul(xb, wu8, mx, rx, my, ry)
                fused = kernel(xb, wu8, mx, rx, my, ry)
                dispatched = mm8_matmul_triton(xb, wu8, mx, rx, my, ry)
                dispatched_relu2 = mm8_matmul_triton(xb, wu8, mx, rx, my, ry, relu2=True)
            db = (fused - refb).abs().max().item()
            dd = (dispatched - refb).abs().max().item()
            print(
                f"batched triton bsz={batch} vs torch-ref max_abs = {db:.6f}; "
                f"dispatcher = {dd:.6f} (<= {args.triton_max_abs})",
                flush=True,
            )
            ok = ok and db <= args.triton_max_abs and dd <= args.triton_max_abs
            relu2_cos = F.cosine_similarity(
                dispatched_relu2.float().flatten().unsqueeze(0),
                (torch.relu(dispatched) ** 2).float().flatten().unsqueeze(0),
            ).item()
            print(
                f"batched fused relu2 bsz={batch} cosine = {relu2_cos:.6f} "
                f"(>= {args.relu2_cos_min})",
                flush=True,
            )
            ok = ok and relu2_cos >= args.relu2_cos_min

    # 3. end-to-end size-gated quantization
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ids = tok("The quick brown fox jumps over the lazy dog.",
              return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    with torch.no_grad():
        ref = model(ids).logits[0, -1].float().cpu()
    n = quantize_model_mm8(model, min_params=8_000_000)
    with torch.no_grad():
        q = model(ids).logits[0, -1].float().cpu()
    e2e = F.cosine_similarity(ref.unsqueeze(0), q.unsqueeze(0)).item()
    print(f"e2e ({n} layer(s) quantized) cos = {e2e:.6f} (>= {args.e2e_cos_min})", flush=True)
    ok = ok and e2e >= args.e2e_cos_min and n >= 1

    # 4. Native fast-token backends must accept quantized lm_head modules.
    # Regression coverage for MM8/MM4Linear, which intentionally do not expose
    # a dense `.weight` tensor. Compare native decode to the quantized FLA
    # one-token fallback from the same empty recurrent state.
    one = ids[:, :1]
    old_backend = os.environ.get("RWKV7_FAST_TOKEN_BACKEND")
    try:
        with torch.no_grad():
            os.environ["RWKV7_FAST_TOKEN_BACKEND"] = "fla"
            fast_ref = model.rwkv7_forward_token(one, past_key_values=None).logits[0, -1].float().cpu()
            for backend in ("native_jit", "native_graph"):
                os.environ["RWKV7_FAST_TOKEN_BACKEND"] = backend
                fast_q = model.rwkv7_forward_token(one, past_key_values=None).logits[0, -1].float().cpu()
                used = getattr(model, "_rwkv7_last_fast_token_backend", None)
                fast_cos = F.cosine_similarity(fast_ref.unsqueeze(0), fast_q.unsqueeze(0)).item()
                print(
                    f"fast-token {backend} used={used} cos vs quantized-fla = "
                    f"{fast_cos:.6f} (>= {args.fast_token_cos_min})",
                    flush=True,
                )
                ok = ok and used == backend and fast_cos >= args.fast_token_cos_min
    finally:
        if old_backend is None:
            os.environ.pop("RWKV7_FAST_TOKEN_BACKEND", None)
        else:
            os.environ["RWKV7_FAST_TOKEN_BACKEND"] = old_backend

    if not ok:
        print("FAIL", flush=True)
        return 1
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
