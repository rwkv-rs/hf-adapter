#!/usr/bin/env python3
# coding=utf-8
"""HF API contract smoke tests for the RWKV-7 adapter.

This covers integration points commonly touched by PEFT/Trainer/generation
stacks but not exercised by a plain forward pass: fixed-vocab resize handling,
generation input preparation, recurrent cache beam reorder, and gradient
checkpointing toggles.
"""
from __future__ import annotations

import argparse
import os

# Keep the V100 training smoke path out of Dynamo/Triton compile trouble.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--attn-mode", default="fused_recurrent", choices=["chunk", "fused_recurrent"])
    ap.add_argument("--fuse-norm", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--beam-new-tokens", type=int, default=2)
    args = ap.parse_args()
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=args.device if args.device.startswith("cuda") else None,
    ).eval()
    set_attn_mode(model, args.attn_mode)
    if args.fuse_norm != "auto":
        desired = args.fuse_norm == "true"
        actual = bool(getattr(model.config, "fuse_norm", False))
        if actual != desired:
            raise ValueError(f"Loaded model config has fuse_norm={actual}; use a converted model dir with fuse_norm={desired}")

    # Fixed official RWKV trie vocab: no-op resize should be accepted, changing
    # the size should fail loudly instead of creating a broken model/tokenizer.
    emb = model.get_input_embeddings()
    same = model.resize_token_embeddings(model.config.vocab_size)
    assert same is emb, "same-size resize should be a no-op returning input embeddings"
    try:
        model.resize_token_embeddings(model.config.vocab_size + 1)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("changing RWKV vocab size should raise NotImplementedError")

    prompts = ["User: Alpha.\n\nAssistant:", "User: Beta.\n\nAssistant:"]
    batch = tok(prompts, return_tensors="pt", padding=True)
    if args.device.startswith("cuda"):
        batch = {k: v.cuda() for k, v in batch.items()}

    with torch.no_grad():
        out = model(**batch, use_cache=True, logits_to_keep=1)
    assert out.past_key_values is not None, "use_cache=True should return recurrent state"
    prepared = model.prepare_inputs_for_generation(
        batch["input_ids"],
        past_key_values=out.past_key_values,
        attention_mask=batch.get("attention_mask"),
        use_cache=True,
        logits_to_keep=1,
    )
    assert prepared["input_ids"].shape[1] == 1, prepared["input_ids"].shape
    assert prepared["past_key_values"] is out.past_key_values

    beam_idx = torch.tensor([1, 0], dtype=torch.long, device=batch["input_ids"].device)
    reordered = model._reorder_cache(out.past_key_values, beam_idx)
    assert reordered is out.past_key_values
    assert reordered.get_seq_length() >= batch["input_ids"].shape[1]

    if args.beam_new_tokens > 0:
        with torch.no_grad():
            beam = model.generate(
                **{k: v[:1] for k, v in batch.items()},
                max_new_tokens=args.beam_new_tokens,
                num_beams=2,
                do_sample=False,
                use_cache=True,
            )
        assert beam.shape[0] == 1 and beam.shape[1] >= batch["input_ids"].shape[1]
        backend_getter = getattr(model, "rwkv7_last_fast_token_backend", None)
        if callable(backend_getter):
            effective_backend = backend_getter()
            print("generate_fast_token_backend", effective_backend)
            assert effective_backend in {"native_graph", "native_jit", "fla"}, effective_backend
        print("beam_ids", beam[0, -args.beam_new_tokens :].tolist())

    model.train()
    model.config.use_cache = True
    model.gradient_checkpointing_enable()
    assert getattr(model, "is_gradient_checkpointing", True), "gradient checkpointing flag was not enabled"
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
