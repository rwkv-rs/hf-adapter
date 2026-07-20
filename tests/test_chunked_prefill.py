#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
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
    ap.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog. " * 16)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--chunk-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--max-diff", type=float, default=0.15)
    ap.add_argument("--min-cosine", type=float, default=0.999)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=DTYPES[args.dtype],
        device_map=args.device if args.device.startswith("cuda") else None,
    ).eval()
    set_attn_mode(model, args.attn_mode)
    if args.fuse_norm != "auto":
        desired = args.fuse_norm == "true"
        actual = bool(getattr(model.config, "fuse_norm", False))
        if actual != desired:
            raise ValueError(f"Loaded model config has fuse_norm={actual}; use a converted model dir with fuse_norm={desired}")
    assert hasattr(model, "rwkv7_prefill_chunks"), "Model does not expose rwkv7_prefill_chunks"

    enc = tok(args.prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc.input_ids.repeat(args.batch_size, 1)
    if args.device.startswith("cuda"):
        input_ids = input_ids.cuda()
    assert input_ids.shape[1] >= 2, "Prompt must tokenize to at least two tokens"

    with torch.inference_mode():
        full = model(input_ids, use_cache=True, logits_to_keep=1)
        seq_full = full.past_key_values.get_seq_length()
        next_token = full.logits[:, -1:].argmax(dim=-1)
        full_next = model(next_token, past_key_values=full.past_key_values, use_cache=True, logits_to_keep=1)

        for chunk_size in args.chunk_sizes:
            chunked = model.rwkv7_prefill_chunks(input_ids, chunk_size=chunk_size, logits_to_keep=1)
            diff = float((full.logits.float() - chunked.logits.float()).abs().max().detach().cpu())
            seq_chunked = chunked.past_key_values.get_seq_length()
            chunk_next = model(next_token, past_key_values=chunked.past_key_values, use_cache=True, logits_to_keep=1)
            decode_diff = float((full_next.logits.float() - chunk_next.logits.float()).abs().max().detach().cpu())
            prompt_cosine = float(
                F.cosine_similarity(full.logits.float(), chunked.logits.float(), dim=-1).min().detach().cpu()
            )
            decode_cosine = float(
                F.cosine_similarity(full_next.logits.float(), chunk_next.logits.float(), dim=-1).min().detach().cpu()
            )
            prompt_greedy = torch.equal(full.logits.argmax(dim=-1), chunked.logits.argmax(dim=-1))
            decode_greedy = torch.equal(full_next.logits.argmax(dim=-1), chunk_next.logits.argmax(dim=-1))
            print(
                "chunked_prefill",
                "chunk_size", chunk_size,
                "max_abs_diff", diff,
                "decode_max_abs_diff", decode_diff,
                "min_cosine", prompt_cosine,
                "decode_min_cosine", decode_cosine,
                "greedy_match", prompt_greedy,
                "decode_greedy_match", decode_greedy,
                "seq_full", seq_full,
                "seq_chunked", seq_chunked,
            )
            assert diff <= args.max_diff, (chunk_size, diff)
            assert decode_diff <= args.max_diff, (chunk_size, decode_diff)
            assert prompt_cosine >= args.min_cosine, (chunk_size, prompt_cosine)
            assert decode_cosine >= args.min_cosine, (chunk_size, decode_cosine)
            assert prompt_greedy, chunk_size
            assert decode_greedy, chunk_size
            assert seq_full == seq_chunked, (chunk_size, seq_full, seq_chunked)

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
