#!/usr/bin/env python3
# coding=utf-8
"""Batched HF cache smoke test for RWKV-7 adapter.

This covers the non-fast-path recurrent HF `forward` API used by PEFT/Trainer and
serving frameworks before a dedicated batched fast path exists. The test uses a
repeated prompt so every batch row should produce identical logits and greedy
next tokens; that catches broken batch-state layout and cache-length handling.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("RWKV_V7_ON", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def set_attn_mode(model, attn_mode: str) -> None:
    model.config.attn_mode = attn_mode
    for layer in getattr(model.model, "layers", []):
        attn = getattr(layer, "attn", None)
        if hasattr(attn, "mode"):
            attn.mode = attn_mode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16", choices=sorted(DTYPES))
    ap.add_argument("--attn-mode", default="fused_recurrent", choices=["chunk", "fused_recurrent"])
    ap.add_argument("--fuse-norm", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--decode-steps", type=int, default=8)
    ap.add_argument("--max-row-diff", type=float, default=1e-5)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=DTYPES[args.dtype],
        device_map=args.device if args.device.startswith("cuda") else None,
    ).eval()
    if args.fuse_norm != "auto":
        desired = args.fuse_norm == "true"
        actual = bool(getattr(model.config, "fuse_norm", False))
        if actual != desired:
            raise ValueError(f"Loaded model config has fuse_norm={actual}; use a converted model dir with fuse_norm={desired}")
    set_attn_mode(model, args.attn_mode)

    seed = "The quick brown fox jumps over the lazy dog. " * 128
    base = tok(seed, return_tensors="pt", add_special_tokens=False).input_ids[:, : args.prompt_tokens]
    if args.device.startswith("cuda"):
        base = base.to(args.device)

    with torch.inference_mode():
        for bsz in args.batch_sizes:
            ids = base.repeat(bsz, 1)
            out = model(ids, use_cache=True, logits_to_keep=1)
            logits = out.logits.float()
            assert tuple(logits.shape) == (bsz, 1, model.config.vocab_size), tuple(logits.shape)
            state = out.past_key_values
            assert state.get_seq_length() == ids.shape[1], (state.get_seq_length(), ids.shape[1])
            row_diff = float((logits - logits[:1]).abs().max().detach().cpu())
            print(f"bsz={bsz} prefill_shape={tuple(logits.shape)} row_max_abs_diff={row_diff}")
            assert row_diff <= args.max_row_diff, row_diff
            nxt = out.logits[:, -1:].argmax(dim=-1)
            assert torch.equal(nxt, nxt[:1].repeat(bsz, 1))
            for step in range(args.decode_steps):
                out = model(nxt, past_key_values=state, use_cache=True, logits_to_keep=1)
                state = out.past_key_values
                logits = out.logits.float()
                row_diff = float((logits - logits[:1]).abs().max().detach().cpu())
                print(f"bsz={bsz} step={step + 1} row_max_abs_diff={row_diff}")
                assert row_diff <= args.max_row_diff, row_diff
                nxt = out.logits[:, -1:].argmax(dim=-1)
                assert torch.equal(nxt, nxt[:1].repeat(bsz, 1))
            assert state.get_seq_length() == ids.shape[1] + args.decode_steps
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
