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
import json
import types
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from rwkv7_hf.native_model import NativeRWKV7ForCausalLM

PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Once upon a time, in a faraway land,",
    "User: Hello!\n\nAssistant:",
]


def build_batch_ids(tok, batch_size: int, prompt_tokens: int, device: str) -> torch.LongTensor:
    rows = []
    for i in range(batch_size):
        text = (PROMPTS[i % len(PROMPTS)] + " ") * 32
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if ids.numel() < prompt_tokens:
            raise ValueError(f"prompt {i} only produced {ids.numel()} tokens; need {prompt_tokens}")
        rows.append(ids[:prompt_tokens])
    return torch.stack(rows, dim=0).to(device)


def append_row(path: str, row: dict) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gen-tokens", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=3)
    ap.add_argument("--batch-prompt-tokens", type=int, default=16)
    ap.add_argument("--expect-jit-decode", action="store_true")
    ap.add_argument("--results", default="")
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

    batch_ids = build_batch_ids(tok, args.batch_size, args.batch_prompt_tokens, "cuda")
    with torch.no_grad():
        bf = fla(batch_ids, use_cache=True, logits_to_keep=1)
        bn = nat(batch_ids, use_cache=True)
    bf_logits = bf.logits[:, -1].float().cpu()
    bn_logits = bn.logits[:, -1].float().cpu()
    batch_cos_vals = F.cosine_similarity(bf_logits, bn_logits, dim=-1)
    batch_min_cos = float(batch_cos_vals.min().item())
    batch_max_abs = float((bf_logits - bn_logits).abs().max().item())
    batch_argmax_ok = int((bf_logits.argmax(dim=-1) == bn_logits.argmax(dim=-1)).sum().item())
    batch_next = bf.logits[:, -1:].argmax(dim=-1)
    with torch.no_grad():
        bf_next = fla(batch_next, past_key_values=bf.past_key_values, use_cache=True, logits_to_keep=1)
        bn_next = nat(batch_next, past_key_values=bn.past_key_values, use_cache=True)
    batch_decode_max_abs = float((bf_next.logits[:, -1].float().cpu() - bn_next.logits[:, -1].float().cpu()).abs().max().item())
    batch_decode_argmax_ok = int((bf_next.logits[:, -1].argmax(dim=-1).cpu() == bn_next.logits[:, -1].argmax(dim=-1).cpu()).sum().item())
    native_decode_backend = (
        nat.rwkv7_native_model_last_decode_backend()
        if hasattr(nat, "rwkv7_native_model_last_decode_backend")
        else None
    )
    state0, xpa0, xpf0, vfirst = bn_next.past_key_values
    batch_cache_shape_ok = (
        len(state0) == nat.config.num_hidden_layers
        and state0[0].shape[0] == args.batch_size
        and xpa0[0].shape[0] == args.batch_size
        and xpf0[0].shape[0] == args.batch_size
        and vfirst.shape[0] == args.batch_size
    )
    print(
        f"[batch-forward] bsz={args.batch_size} min_cos={batch_min_cos:.6f} "
        f"max_abs={batch_max_abs:.6f} argmax {batch_argmax_ok}/{args.batch_size}"
    )
    print(
        f"[batch-cache] decode_max_abs={batch_decode_max_abs:.6f} "
        f"argmax {batch_decode_argmax_ok}/{args.batch_size} cache_shape={batch_cache_shape_ok} "
        f"backend={native_decode_backend}"
    )

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

    ok = (
        worst_cos >= 0.999
        and argmax_ok == len(PROMPTS)
        and batch_min_cos >= 0.999
        and batch_argmax_ok == args.batch_size
        and batch_decode_argmax_ok == args.batch_size
        and batch_cache_shape_ok
        and (not args.expect_jit_decode or native_decode_backend == "native_jit")
        and match == len(nt)
        and cache_ok
    )
    append_row(
        args.results,
        {
            "axis": "native_model_smoke",
            "backend": "hf_native_model",
            "status": "pass" if ok else "fail",
            "dtype": "fp32",
            "device": torch.cuda.get_device_name(0),
            "model_name": Path(args.model).name,
            "hf_model_dir": args.model,
            "prompt_count": len(PROMPTS),
            "forward_min_cos": round(float(worst_cos), 8),
            "forward_max_abs": round(float(worst_abs), 8),
            "forward_argmax_match": argmax_ok,
            "forward_argmax_total": len(PROMPTS),
            "batch_size": args.batch_size,
            "batch_prompt_tokens": args.batch_prompt_tokens,
            "batch_forward_min_cos": round(float(batch_min_cos), 8),
            "batch_forward_max_abs": round(float(batch_max_abs), 8),
            "batch_forward_argmax_match": batch_argmax_ok,
            "batch_forward_argmax_total": args.batch_size,
            "batch_decode_max_abs": round(float(batch_decode_max_abs), 8),
            "batch_decode_argmax_match": batch_decode_argmax_ok,
            "batch_decode_argmax_total": args.batch_size,
            "batch_cache_shape_ok": bool(batch_cache_shape_ok),
            "native_decode_backend": native_decode_backend,
            "generate_tokens": len(nt),
            "generate_token_match": match,
            "generate_token_total": len(nt),
            "incremental_cache": bool(cache_ok),
        },
    )
    print("NATIVE MODEL PASS" if ok else "NATIVE MODEL FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
