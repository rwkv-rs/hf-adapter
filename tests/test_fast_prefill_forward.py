#!/usr/bin/env python3
# coding=utf-8
"""Smoke the opt-in native prefill path through ordinary HF forward().

Usage:
  python tests/test_fast_prefill_forward.py --model <hf_dir> --fused-scan
  python tests/test_fast_prefill_forward.py --model <hf_dir> \
    --reference-backend native-token-loop --prompt-tokens 512
"""
from __future__ import annotations

import argparse
import os

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:  # pragma: no cover - lightweight local envs
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    AutoModelForCausalLM = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]


PROMPT = "The quick brown fox jumps over the lazy dog. " * 256


def _restore_env(old: dict[str, str | None]) -> None:
    for key, value in old.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _native_token_loop_prefill(model, input_ids):
    """Independent recurrent reference that never enters FLA prompt prefill."""

    out = None
    past = None
    for index in range(int(input_ids.shape[1])):
        out = model.rwkv7_forward_token(
            input_ids[:, index],
            past_key_values=past,
            return_dict=True,
        )
        past = out.past_key_values
    if out is None:
        raise ValueError("native token-loop reference requires a non-empty prompt")
    return out


def _native_token_loop_generate(model, input_ids, max_new_tokens: int):
    sequence = input_ids
    out = _native_token_loop_prefill(model, input_ids)
    for index in range(int(max_new_tokens)):
        next_token = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        sequence = torch.cat((sequence, next_token), dim=1)
        if index + 1 < int(max_new_tokens):
            out = model.rwkv7_forward_token(
                next_token,
                past_key_values=out.past_key_values,
                return_dict=True,
            )
    return sequence


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prompt-tokens", type=int, default=32)
    ap.add_argument("--gen-tokens", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--reference-backend", choices=("hf", "native-token-loop"), default="hf")
    ap.add_argument("--fused-scan", action="store_true")
    args = ap.parse_args()
    if torch is None or not args.model:
        print("SKIP fast prefill forward test: torch/model unavailable")
        return 0

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
        device_map=args.device if args.device.startswith("cuda") else None,
    ).eval()
    ids = tok(PROMPT, return_tensors="pt", add_special_tokens=False).input_ids[:, : args.prompt_tokens]
    ids = ids.repeat(args.batch_size, 1)
    if args.device.startswith("cuda"):
        ids = ids.to(args.device)

    old = {k: os.environ.get(k) for k in ("RWKV7_FAST_PREFILL", "RWKV7_NATIVE_PREFILL_FUSED_SCAN")}
    try:
        with torch.inference_mode():
            if args.reference_backend == "native-token-loop":
                ref = _native_token_loop_prefill(model, ids)
            else:
                os.environ["RWKV7_FAST_PREFILL"] = "0"
                ref = model(ids, use_cache=True, logits_to_keep=1, return_dict=True)

            os.environ["RWKV7_FAST_PREFILL"] = "1"
            os.environ["RWKV7_NATIVE_PREFILL_FUSED_SCAN"] = "1" if args.fused_scan else "0"
            fast = model(ids, use_cache=True, logits_to_keep=1, return_dict=True)
            seen_after_prefill = fast.past_key_values.get_seq_length() if hasattr(fast.past_key_values, "get_seq_length") else None

            ref_logits = ref.logits[:, -1].float()
            fast_logits = fast.logits[:, -1].float()
            max_abs = float((ref_logits - fast_logits).abs().max().detach().cpu())
            min_cos = float(F.cosine_similarity(ref_logits, fast_logits, dim=-1).min().detach().cpu())
            greedy_match = bool(torch.equal(ref_logits.argmax(dim=-1).detach().cpu(), fast_logits.argmax(dim=-1).detach().cpu()))

            next_token = ref_logits.argmax(dim=-1, keepdim=True)
            if args.reference_backend == "native-token-loop":
                ref_next = model.rwkv7_forward_token(next_token, past_key_values=ref.past_key_values, return_dict=True)
            else:
                ref_next = model(next_token, past_key_values=ref.past_key_values, use_cache=True, logits_to_keep=1, return_dict=True)
            fast_next = model(next_token, past_key_values=fast.past_key_values, use_cache=True, logits_to_keep=1, return_dict=True)
            decode_max_abs = float((ref_next.logits[:, -1].float() - fast_next.logits[:, -1].float()).abs().max().detach().cpu())
            decode_match = bool(torch.equal(ref_next.logits[:, -1].argmax(dim=-1).detach().cpu(), fast_next.logits[:, -1].argmax(dim=-1).detach().cpu()))

            if args.reference_backend == "native-token-loop":
                ref_gen = _native_token_loop_generate(model, ids, args.gen_tokens)
            else:
                os.environ["RWKV7_FAST_PREFILL"] = "0"
                ref_gen = model.generate(ids, max_new_tokens=args.gen_tokens, do_sample=False, use_cache=True, pad_token_id=0)
            os.environ["RWKV7_FAST_PREFILL"] = "1"
            fast_gen = model.generate(ids, max_new_tokens=args.gen_tokens, do_sample=False, use_cache=True, pad_token_id=0)
            generate_match = bool(torch.equal(ref_gen.detach().cpu(), fast_gen.detach().cpu()))
    finally:
        _restore_env(old)

    print(
        f"FAST PREFILL FORWARD PASS reference={args.reference_backend} fused_scan={args.fused_scan} "
        f"max_abs={max_abs:.6f} min_cos={min_cos:.8f} greedy={greedy_match} "
        f"decode_max_abs={decode_max_abs:.6f} decode_greedy={decode_match} "
        f"generate_match={generate_match} seen={seen_after_prefill}"
    )
    assert greedy_match
    assert decode_match
    assert generate_match
    assert seen_after_prefill == int(ids.shape[1])
    assert min_cos >= 0.999
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
