#!/usr/bin/env python3
# coding=utf-8
"""HF save_pretrained/from_pretrained roundtrip smoke test for RWKV-7 adapter."""
from __future__ import annotations

import argparse
import tempfile

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--max-diff", type=float, default=0.0)
    args = ap.parse_args()
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=args.device if args.device.startswith("cuda") else None,
    ).eval()
    enc = tok("User: Hello!\n\nAssistant:", return_tensors="pt")
    if args.device.startswith("cuda"):
        enc = {k: v.cuda() for k, v in enc.items()}
    with torch.no_grad():
        ref = model(**enc, use_cache=True, logits_to_keep=1).logits.detach().float().cpu()

    with tempfile.TemporaryDirectory(prefix="rwkv7_hf_roundtrip_") as tmp:
        model.save_pretrained(tmp, safe_serialization=True)
        tok.save_pretrained(tmp)
        re_tok = AutoTokenizer.from_pretrained(tmp, trust_remote_code=True)
        re_model = AutoModelForCausalLM.from_pretrained(
            tmp,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=args.device if args.device.startswith("cuda") else None,
        ).eval()
        re_enc = re_tok("User: Hello!\n\nAssistant:", return_tensors="pt")
        if args.device.startswith("cuda"):
            re_enc = {k: v.cuda() for k, v in re_enc.items()}
        with torch.no_grad():
            got = re_model(**re_enc, use_cache=True, logits_to_keep=1).logits.detach().float().cpu()
        diff = float((ref - got).abs().max().item())
        print(f"roundtrip_dir={tmp}")
        print(f"ref_shape={tuple(ref.shape)} got_shape={tuple(got.shape)} max_abs_diff={diff}")
        print(f"tokenizer_ids={re_enc['input_ids'][0].tolist()}")
        if tuple(ref.shape) != tuple(got.shape):
            print("FAIL: shape mismatch")
            return 1
        if diff > args.max_diff:
            print(f"FAIL: max_abs_diff {diff} > {args.max_diff}")
            return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
