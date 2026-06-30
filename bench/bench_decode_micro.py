#!/usr/bin/env python3
# coding=utf-8
"""Microbenchmarks for RWKV-7 HF one-token decode.

This complements profiler traces with stable JSONL numbers that are easy to
compare across commits. It times the recurrent one-token paths plus cheap
isolated operations so decode regressions can be attributed before deeper kernel
work:

- standard HF recurrent `forward`, fixed-token and greedy-token loops
- optional fast one-token API, fixed-token and greedy-token loops
- embedding, final norm + lm_head, lm_head only, argmax only, and empty-loop cost
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("RWKV_V7_ON", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
SEED = "The quick brown fox jumps over the lazy dog. " * 128
_FALSE_VALUES = {"0", "false", "False", "no", "off"}


def cuda_sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def device_name(device: str) -> str:
    return torch.cuda.get_device_name(0) if device.startswith("cuda") else device


def peak_mb(device: str) -> float | None:
    if not device.startswith("cuda"):
        return None
    return round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)


def set_attn_mode(model, attn_mode: str) -> None:
    model.config.attn_mode = attn_mode
    for layer in getattr(model.model, "layers", []):
        attn = getattr(layer, "attn", None)
        if hasattr(attn, "mode"):
            attn.mode = attn_mode


def encode(tok, n: int, device: str) -> torch.Tensor:
    ids = tok(SEED, return_tensors="pt", add_special_tokens=False).input_ids[:, :n]
    return ids.to(device) if device.startswith("cuda") else ids


def timed(fn: Callable[[], Any], device: str, warmup: int, steps: int) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
    cuda_sync(device)
    t0 = time.time()
    with torch.inference_mode():
        for _ in range(steps):
            fn()
    cuda_sync(device)
    return time.time() - t0


def load_model(args, dtype):
    if args.fast_cache != "auto":
        os.environ["RWKV7_FAST_CACHE"] = "1" if args.fast_cache == "true" else "0"
    if args.fast_token_layout != "auto":
        os.environ["RWKV7_FAST_TOKEN_LAYOUT"] = args.fast_token_layout
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_dir,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=args.device if args.device.startswith("cuda") else None,
    ).eval()
    if args.fuse_norm != "auto":
        desired = args.fuse_norm == "true"
        actual = bool(getattr(model.config, "fuse_norm", False))
        if actual != desired:
            raise ValueError(f"Loaded model config has fuse_norm={actual}; use a converted model dir with fuse_norm={desired}")
    set_attn_mode(model, args.attn_mode)
    return model


def metric(dt_s: float, steps: int) -> dict[str, float]:
    ms = 1000.0 * dt_s / max(steps, 1)
    return {"ms": round(ms, 4), "tokps": round(1000.0 / ms, 1) if ms > 0 else float("inf")}


def bench_decode_paths(args, model, ids: torch.Tensor) -> dict[str, Any]:
    fixed = ids[:, -1:]

    def seed_state():
        out = model(ids[:, : min(8, ids.shape[1])], use_cache=True, logits_to_keep=1)
        return out.past_key_values, out.logits[:, -1:].argmax(dim=-1)

    with torch.inference_mode():
        state, nxt = seed_state()
        cache_type = type(state).__name__ if state is not None else None

    state, _ = seed_state()

    def hf_fixed_step():
        nonlocal state
        out = model(fixed, past_key_values=state, use_cache=True, logits_to_keep=1)
        state = out.past_key_values

    hf_fixed_dt = timed(hf_fixed_step, args.device, args.warmup, args.steps)

    state, nxt = seed_state()

    def hf_greedy_step():
        nonlocal state, nxt
        out = model(nxt, past_key_values=state, use_cache=True, logits_to_keep=1)
        state = out.past_key_values
        nxt = out.logits[:, -1:].argmax(dim=-1)

    hf_greedy_dt = timed(hf_greedy_step, args.device, args.warmup, args.steps)

    row: dict[str, Any] = {
        "cache_type": cache_type,
        "hf_forward_fixed": metric(hf_fixed_dt, args.steps),
        "hf_forward_greedy": metric(hf_greedy_dt, args.steps),
    }

    fast_fn = getattr(model, "rwkv7_forward_token", None)
    fast_name = "rwkv7_forward_token" if fast_fn is not None else None
    if fast_fn is None:
        fast_fn = getattr(model, "rwkv7_forward_one", None)
        fast_name = "rwkv7_forward_one" if fast_fn is not None else None
    row["fast_decode_api_name"] = fast_name
    has_fast = fast_fn is not None
    if args.fast_decode_api == "true" and not has_fast:
        raise ValueError("Loaded model does not expose a fast one-token decode API")
    if args.fast_decode_api != "false" and has_fast:
        state, _ = seed_state()

        def fast_fixed_step():
            nonlocal state
            out = fast_fn(fixed, past_key_values=state)
            state = out.past_key_values

        fast_fixed_dt = timed(fast_fixed_step, args.device, args.warmup, args.steps)

        state, nxt = seed_state()

        def fast_greedy_step():
            nonlocal state, nxt
            out = fast_fn(nxt, past_key_values=state)
            state = out.past_key_values
            nxt = out.logits[:, -1:].argmax(dim=-1)

        fast_greedy_dt = timed(fast_greedy_step, args.device, args.warmup, args.steps)
        row["fast_decode_fixed"] = metric(fast_fixed_dt, args.steps)
        row["fast_decode_greedy"] = metric(fast_greedy_dt, args.steps)
    else:
        row["fast_decode_fixed"] = None
        row["fast_decode_greedy"] = None
    return row


def bench_isolated_ops(args, model, ids: torch.Tensor) -> dict[str, Any]:
    fixed = ids[:, -1:]
    weight = model.lm_head.weight
    hidden_size = int(getattr(model.config, "hidden_size", weight.shape[1]))
    hidden = torch.randn(1, 1, hidden_size, device=weight.device, dtype=weight.dtype)
    with torch.inference_mode():
        logits = model.lm_head(hidden)

    def empty_step():
        return None

    def embedding_step():
        return model.model.embeddings(fixed)

    def norm_lm_head_step():
        return model.lm_head(model.model.norm(hidden))

    def lm_head_step():
        return model.lm_head(hidden)

    def argmax_step():
        return logits[:, -1:].argmax(dim=-1)

    empty_dt = timed(empty_step, args.device, args.warmup, args.steps)
    emb_dt = timed(embedding_step, args.device, args.warmup, args.steps)
    norm_head_dt = timed(norm_lm_head_step, args.device, args.warmup, args.steps)
    head_dt = timed(lm_head_step, args.device, args.warmup, args.steps)
    argmax_dt = timed(argmax_step, args.device, args.warmup, args.steps)
    return {
        "empty_loop": metric(empty_dt, args.steps),
        "embedding": metric(emb_dt, args.steps),
        "norm_lm_head": metric(norm_head_dt, args.steps),
        "lm_head": metric(head_dt, args.steps),
        "argmax": metric(argmax_dt, args.steps),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--dtype", default="fp16", choices=sorted(DTYPES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-mode", default="fused_recurrent", choices=["chunk", "fused_recurrent"])
    ap.add_argument("--fuse-norm", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--fast-cache", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--fast-decode-api", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--fast-token-layout", choices=["auto", "3d", "2d"], default="auto",
                    help="HF fast-token layout; 3d is the validated baseline, 2d is an experimental A/B path")
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--results", default=str(Path(__file__).parent / "results.jsonl"))
    args = ap.parse_args()

    dtype = DTYPES[args.dtype]
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    tok = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)
    model = load_model(args, dtype)
    ids = encode(tok, args.prompt_tokens, args.device)

    row: dict[str, Any] = {
        "axis": "decode_micro",
        "backend": "hf_adapter",
        "dtype": args.dtype,
        "device": device_name(args.device),
        "attn_mode": args.attn_mode,
        "fuse_norm": getattr(model.config, "fuse_norm", None),
        "fast_cache": os.environ.get("RWKV7_FAST_CACHE", "1") not in _FALSE_VALUES,
        "fast_decode_api_requested": args.fast_decode_api,
        "fast_decode_api_available": hasattr(model, "rwkv7_forward_token") or hasattr(model, "rwkv7_forward_one"),
        "fast_token_layout": os.environ.get("RWKV7_FAST_TOKEN_LAYOUT", "3d"),
        "prompt_tokens": int(ids.shape[1]),
        "steps": args.steps,
    }
    row.update(bench_decode_paths(args, model, ids))
    row.update(bench_isolated_ops(args, model, ids))
    row["peak_vram_mb"] = peak_mb(args.device)

    print(json.dumps(row, indent=2, ensure_ascii=False), flush=True)
    if args.results:
        out = Path(args.results)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nappended 1 row -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
