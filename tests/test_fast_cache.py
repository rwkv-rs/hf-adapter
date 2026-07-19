#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from contextlib import contextmanager

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


@contextmanager
def fast_cache(enabled: bool):
    old = os.environ.get("RWKV7_FAST_CACHE")
    os.environ["RWKV7_FAST_CACHE"] = "1" if enabled else "0"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("RWKV7_FAST_CACHE", None)
        else:
            os.environ["RWKV7_FAST_CACHE"] = old


def _to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    if device.startswith("cuda"):
        return {k: v.cuda() for k, v in batch.items()}
    return batch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16", choices=sorted(DTYPES))
    ap.add_argument("--attn-mode", default="chunk", choices=["chunk", "fused_recurrent"])
    ap.add_argument("--fuse-norm", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--prompt", default="User: Give one short sentence about RWKV.\n\nAssistant:")
    ap.add_argument("--decode-steps", type=int, default=8)
    ap.add_argument("--max-diff", type=float, default=0.0)
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
    model.config.attn_mode = args.attn_mode
    for layer in getattr(model.model, "layers", []):
        attn = getattr(layer, "attn", None)
        if hasattr(attn, "mode"):
            attn.mode = args.attn_mode
    batch = _to_device(tok(args.prompt, return_tensors="pt"), args.device)

    with torch.inference_mode(), fast_cache(False):
        ref = model(**batch, use_cache=True, logits_to_keep=1)
    with torch.inference_mode(), fast_cache(True):
        fast = model(**batch, use_cache=True, logits_to_keep=1)

    fast_cache_name = type(fast.past_key_values).__name__
    print("fast_cache_type", fast_cache_name)
    assert fast_cache_name in {"RWKV7StateCache", "NativeRWKV7Cache"}, fast_cache_name
    assert hasattr(fast.past_key_values, "get_seq_length")
    assert hasattr(fast.past_key_values, "clone")
    max_diff = float((ref.logits.float() - fast.logits.float()).abs().max().detach().cpu())
    print("prefill_max_abs_diff", max_diff)
    assert max_diff <= args.max_diff, max_diff
    assert ref.past_key_values.get_seq_length() == fast.past_key_values.get_seq_length()

    ref_state = ref.past_key_values
    fast_state = fast.past_key_values
    ref_next = ref.logits[:, -1:].argmax(dim=-1)
    fast_next = fast.logits[:, -1:].argmax(dim=-1)
    greedy_equal = 0
    decode_max_diff = 0.0
    with torch.inference_mode():
        for _ in range(args.decode_steps):
            with fast_cache(False):
                ref = model(ref_next, past_key_values=ref_state, use_cache=True, logits_to_keep=1)
            with fast_cache(True):
                fast = model(fast_next, past_key_values=fast_state, use_cache=True, logits_to_keep=1)
            ref_state = ref.past_key_values
            fast_state = fast.past_key_values
            diff = float((ref.logits.float() - fast.logits.float()).abs().max().detach().cpu())
            decode_max_diff = max(decode_max_diff, diff)
            ref_next = ref.logits[:, -1:].argmax(dim=-1)
            fast_next = fast.logits[:, -1:].argmax(dim=-1)
            greedy_equal += int(torch.equal(ref_next, fast_next))
    print("decode_max_abs_diff", decode_max_diff)
    print("greedy_equal", greedy_equal, "/", args.decode_steps)
    print("seq_length", fast_state.get_seq_length())
    assert decode_max_diff <= args.max_diff, decode_max_diff
    assert greedy_equal == args.decode_steps
    assert ref_state.get_seq_length() == fast_state.get_seq_length()
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
