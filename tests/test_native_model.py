#!/usr/bin/env python3
# coding=utf-8
"""Regression test for the native (fla-free) RWKV-7 model (gate H1).

Verifies NativeRWKV7ForCausalLM (pure PyTorch, no fla) loads the converted
weights, forwards bit-exact vs the FLA wrapper, and generates token-identical
greedy output.

  python tests/test_native_model.py --model <hf_dir>
"""
from __future__ import annotations

import argparse
import types

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from rwkv7_hf.native_model import NativeRWKV7ForCausalLM

PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Once upon a time, in a faraway land,",
    "User: Hello!\n\nAssistant:",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gen-tokens", type=int, default=16)
    args = ap.parse_args()
    d = args.model
    tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
    fla = AutoModelForCausalLM.from_pretrained(
        d, trust_remote_code=True, torch_dtype=torch.float32, device_map="cuda").eval()
    nat = NativeRWKV7ForCausalLM.from_pretrained(
        d, torch_dtype=torch.float32, device_map="cuda").eval()

    worst_cos, worst_abs, argmax_ok = 1.0, 0.0, 0
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        with torch.no_grad():
            lf = fla(ids).logits[0, -1].float().cpu()
            ln = nat(ids).logits[0, -1].float().cpu()
        cos = F.cosine_similarity(lf.unsqueeze(0), ln.unsqueeze(0)).item()
        worst_cos = min(worst_cos, cos)
        worst_abs = max(worst_abs, (lf - ln).abs().max().item())
        argmax_ok += int(lf.argmax() == ln.argmax())
    print(f"[forward] min_cos={worst_cos:.6f} max_abs={worst_abs:.6f} "
          f"argmax {argmax_ok}/{len(PROMPTS)}")

    # greedy generate token-identical
    ids = tok(PROMPTS[2], return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    with torch.no_grad():
        no = nat.generate(ids, max_new_tokens=args.gen_tokens, do_sample=False)
        fo = fla.generate(ids, max_new_tokens=args.gen_tokens, do_sample=False,
                          use_cache=True, pad_token_id=0)
    nt = no[0, ids.shape[1]:].tolist()
    ft = fo[0, ids.shape[1]:].tolist()
    match = sum(int(a == b) for a, b in zip(nt, ft))
    print(f"[generate] greedy token-identical {match}/{len(nt)}")

    # GenerationMixin must exercise the incremental cache path rather than
    # recomputing the full prefix on every token.
    calls = []
    original_forward = nat.forward

    def counted_forward(self, input_ids, past_key_values=None, use_cache=None, **kwargs):
        calls.append((tuple(input_ids.shape), past_key_values is not None, bool(use_cache)))
        return original_forward(input_ids, past_key_values=past_key_values, use_cache=use_cache, **kwargs)

    nat.forward = types.MethodType(counted_forward, nat)
    with torch.no_grad():
        nat.generate(ids, max_new_tokens=3, do_sample=False)
    cache_ok = (
        bool(calls)
        and calls[0] == ((1, ids.shape[1]), False, True)
        and all(shape == (1, 1) and has_cache and use_cache for shape, has_cache, use_cache in calls[1:])
    )
    print(f"[generate-cache] incremental_cache={cache_ok} calls={calls}")

    ok = worst_cos >= 0.999 and argmax_ok == len(PROMPTS) and match == len(nt) and cache_ok
    print("NATIVE MODEL PASS" if ok else "NATIVE MODEL FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
