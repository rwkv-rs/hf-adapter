#!/usr/bin/env python3
"""Smoke-test HF-compatible RWKV speculative decoding.

The default test uses the same checkpoint as target and draft, so greedy
speculative decoding should accept every draft token and exactly match
`model.generate()`. Passing `--draft-model` lets CI or a benchmark machine use a
smaller draft model while keeping the same API contract.
"""
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def load_model(path: str, dtype: str, device: str):
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=DTYPES[dtype],
        attn_mode="fused_recurrent",
        fuse_norm=False,
    )
    model.to(device)
    model.eval()
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Target HF model directory")
    ap.add_argument("--draft-model", default=None, help="Draft HF model directory; defaults to --model")
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="fp16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prompt", default="User: Hello!\n\nAssistant:")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--draft-tokens", type=int, default=4)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    target = load_model(args.model, args.dtype, args.device)
    if args.draft_model is None or args.draft_model == args.model:
        draft = target
    else:
        draft = load_model(args.draft_model, args.dtype, args.device)

    enc = tokenizer(args.prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(args.device)
    pad_token_id = getattr(tokenizer, "pad_token_id", None) or 0

    with torch.inference_mode():
        expected = target.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=pad_token_id,
        )
        got = target.rwkv7_speculative_generate(
            input_ids,
            draft_model=draft,
            max_new_tokens=args.max_new_tokens,
            draft_tokens=args.draft_tokens,
            return_stats=True,
        )

    seq = got["sequences"]
    stats = got["stats"]
    print("speculative_stats", stats)
    print("decoded", tokenizer.decode(seq[0].detach().cpu().tolist()))
    assert torch.equal(seq, expected), (seq.detach().cpu().tolist(), expected.detach().cpu().tolist())
    assert stats["generated_tokens"] == args.max_new_tokens, stats
    assert stats["proposed_tokens"] >= args.max_new_tokens, stats
    assert stats["accepted_tokens"] == args.max_new_tokens, stats
    assert stats["corrected_tokens"] == 0, stats
    assert stats["acceptance_rate"] == 1.0, stats
    # With block verification the target performs prompt prefill plus at most
    # ceil(max_new_tokens / draft_tokens) verification calls when draft==target.
    assert stats["target_forward_calls"] <= 1 + ((args.max_new_tokens + args.draft_tokens - 1) // args.draft_tokens), stats
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
