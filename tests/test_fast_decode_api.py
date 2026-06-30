#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def set_attn_mode(model, attn_mode: str) -> None:
    model.config.attn_mode = attn_mode
    for layer in getattr(model.model, "layers", []):
        attn = getattr(layer, "attn", None)
        if hasattr(attn, "mode"):
            attn.mode = attn_mode


def run_decode_case(model, input_ids: torch.Tensor, decode_steps: int, max_diff_limit: float, fast_fn, label: str) -> None:
    prefill_ids = input_ids[:, :-1]
    next_forward = input_ids[:, -1:]
    next_fast = next_forward.clone()

    max_diff = 0.0
    greedy_equal = 0
    with torch.inference_mode():
        forward_out = model(prefill_ids, use_cache=True, logits_to_keep=1)
        fast_out = model(prefill_ids, use_cache=True, logits_to_keep=1)
        forward_state = forward_out.past_key_values
        fast_state = fast_out.past_key_values
        for _ in range(decode_steps):
            forward_out = model(next_forward, past_key_values=forward_state, use_cache=True, logits_to_keep=1)
            fast_out = fast_fn(next_fast, past_key_values=fast_state)
            forward_state = forward_out.past_key_values
            fast_state = fast_out.past_key_values
            diff = float((forward_out.logits.float() - fast_out.logits.float()).abs().max().detach().cpu())
            max_diff = max(max_diff, diff)
            next_forward = forward_out.logits[:, -1:].argmax(dim=-1)
            next_fast = fast_out.logits[:, -1:].argmax(dim=-1)
            greedy_equal += int(torch.equal(next_forward, next_fast))

    print(f"{label} max_abs_diff", max_diff)
    print(f"{label} greedy_equal", greedy_equal, "/", decode_steps)
    print(f"{label} seq_length_forward", forward_state.get_seq_length())
    print(f"{label} seq_length_fast", fast_state.get_seq_length())
    assert max_diff <= max_diff_limit, (label, max_diff)
    assert greedy_equal == decode_steps, label
    assert forward_state.get_seq_length() == fast_state.get_seq_length(), label


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16", choices=sorted(DTYPES))
    ap.add_argument("--attn-mode", default="fused_recurrent", choices=["chunk", "fused_recurrent"])
    ap.add_argument("--fuse-norm", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--prompt", default="The quick brown fox jumps over the lazy dog.")
    ap.add_argument("--decode-steps", type=int, default=32)
    ap.add_argument("--max-diff", type=float, default=0.15)
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--fast-token-layouts", nargs="+", default=["3d"], choices=["3d", "2d"],
                    help="Fast-token tensor layouts to validate; 3d is the current production baseline")
    ap.add_argument("--fast-token-backends", nargs="+", default=["fla"], choices=["fla", "native_jit"],
                    help="Fast-token backends to validate; native_jit is bsz=1 only")
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
    assert hasattr(model, "rwkv7_forward_one"), "Model does not expose rwkv7_forward_one"
    assert hasattr(model, "rwkv7_forward_token"), "Model does not expose rwkv7_forward_token"
    set_attn_mode(model, args.attn_mode)

    enc = tok(args.prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc.input_ids.to(args.device) if args.device.startswith("cuda") else enc.input_ids
    assert input_ids.shape[1] >= 2, "Prompt must tokenize to at least two tokens"

    old_layout = os.environ.get("RWKV7_FAST_TOKEN_LAYOUT")
    old_backend = os.environ.get("RWKV7_FAST_TOKEN_BACKEND")
    try:
        for backend in args.fast_token_backends:
            os.environ["RWKV7_FAST_TOKEN_BACKEND"] = backend
            for layout in args.fast_token_layouts:
                os.environ["RWKV7_FAST_TOKEN_LAYOUT"] = layout
                batch_sizes = [1] if backend == "native_jit" else args.batch_sizes
                for bsz in batch_sizes:
                    ids = input_ids.repeat(bsz, 1)
                    run_decode_case(
                        model,
                        ids,
                        args.decode_steps,
                        args.max_diff,
                        model.rwkv7_forward_token,
                        label=f"rwkv7_forward_token backend={backend} layout={layout} bsz={bsz}",
                    )
                run_decode_case(
                    model,
                    input_ids,
                    args.decode_steps,
                    args.max_diff,
                    model.rwkv7_forward_one,
                    label=f"rwkv7_forward_one backend={backend} layout={layout} bsz=1",
                )
    finally:
        if old_layout is None:
            os.environ.pop("RWKV7_FAST_TOKEN_LAYOUT", None)
        else:
            os.environ["RWKV7_FAST_TOKEN_LAYOUT"] = old_layout
        if old_backend is None:
            os.environ.pop("RWKV7_FAST_TOKEN_BACKEND", None)
        else:
            os.environ["RWKV7_FAST_TOKEN_BACKEND"] = old_backend
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
