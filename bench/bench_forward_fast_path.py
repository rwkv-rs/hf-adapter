#!/usr/bin/env python3
# coding=utf-8
"""Benchmark ordinary HF cached forward with RWKV7_FAST_FORWARD enabled.

This is the production-facing path used by `model.generate(..., use_cache=True)`:
callers still invoke normal HF `forward`, while the adapter internally routes
one-token cached inference to `rwkv7_forward_token` when safe. The benchmark
keeps a reference row with `RWKV7_FAST_FORWARD=0` so gates can verify both
correctness and speedup against the unoptimized recurrent HF forward.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
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


@contextmanager
def fast_forward_env(enabled: bool):
    old = os.environ.get("RWKV7_FAST_FORWARD")
    os.environ["RWKV7_FAST_FORWARD"] = "1" if enabled else "0"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("RWKV7_FAST_FORWARD", None)
        else:
            os.environ["RWKV7_FAST_FORWARD"] = old


def set_attn_mode(model, attn_mode: str) -> None:
    model.config.attn_mode = attn_mode
    for layer in getattr(model.model, "layers", []):
        attn = getattr(layer, "attn", None)
        if hasattr(attn, "mode"):
            attn.mode = attn_mode


def last_fast_token_backend(model):
    getter = getattr(model, "rwkv7_last_fast_token_backend", None)
    if callable(getter):
        return getter()
    return getattr(model, "_rwkv7_last_fast_token_backend", None)


def metric(dt_s: float, steps: int) -> dict[str, float]:
    ms = 1000.0 * dt_s / max(steps, 1)
    return {"ms": round(ms, 4), "tokps": round(1000.0 / ms, 1) if ms > 0 else float("inf")}


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


def configure_env(args: argparse.Namespace) -> None:
    if args.fast_cache != "auto":
        os.environ["RWKV7_FAST_CACHE"] = "1" if args.fast_cache == "true" else "0"
    if args.fast_token_layout != "auto":
        os.environ["RWKV7_FAST_TOKEN_LAYOUT"] = args.fast_token_layout
    os.environ["RWKV7_FAST_TOKEN_BACKEND"] = args.fast_token_backend


def encode(tok, prompt_tokens: int, device: str) -> torch.Tensor:
    ids = tok(SEED, return_tensors="pt", add_special_tokens=False).input_ids[:, :prompt_tokens]
    return ids.to(device) if device.startswith("cuda") else ids


def load_model(args: argparse.Namespace, dtype: torch.dtype):
    configure_env(args)
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


def seed_state(model, ids: torch.Tensor):
    out = model(ids[:, : min(8, ids.shape[1])], use_cache=True, logits_to_keep=1)
    return out.past_key_values, out.logits[:, -1:].argmax(dim=-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--dtype", default="fp16", choices=sorted(DTYPES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-mode", default="fused_recurrent", choices=["chunk", "fused_recurrent"])
    ap.add_argument("--fuse-norm", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--fast-cache", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--fast-token-layout", choices=["auto", "3d", "2d"], default="auto")
    ap.add_argument("--fast-token-backend", choices=["auto", "fla", "native_jit", "native_graph"], default="auto")
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--results", default=str(Path(__file__).parent / "results.jsonl"))
    args = ap.parse_args()

    dtype = DTYPES[args.dtype]
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    tok = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)
    model = load_model(args, dtype)
    ids = encode(tok, args.prompt_tokens, args.device)
    fixed = ids[:, -1:]

    with torch.inference_mode():
        ref_state, _ = seed_state(model, ids)
        auto_state, _ = seed_state(model, ids)
        direct_state, _ = seed_state(model, ids)
        with fast_forward_env(False):
            ref = model(fixed, past_key_values=ref_state, use_cache=True, logits_to_keep=1)
        with fast_forward_env(True):
            auto = model(fixed, past_key_values=auto_state, use_cache=True, logits_to_keep=1)
        direct = model.rwkv7_forward_token(fixed, past_key_values=direct_state)
        max_abs_auto = float((ref.logits.float() - auto.logits.float()).abs().max().detach().cpu())
        max_abs_direct = float((ref.logits.float() - direct.logits.float()).abs().max().detach().cpu())
        auto_backend = last_fast_token_backend(model)

    ref_state, _ = seed_state(model, ids)

    def ref_step():
        nonlocal ref_state
        with fast_forward_env(False):
            out = model(fixed, past_key_values=ref_state, use_cache=True, logits_to_keep=1)
        ref_state = out.past_key_values

    ref_dt = timed(ref_step, args.device, args.warmup, args.steps)

    auto_state, _ = seed_state(model, ids)

    def auto_step():
        nonlocal auto_state
        with fast_forward_env(True):
            out = model(fixed, past_key_values=auto_state, use_cache=True, logits_to_keep=1)
        auto_state = out.past_key_values

    auto_dt = timed(auto_step, args.device, args.warmup, args.steps)
    auto_backend = last_fast_token_backend(model) or auto_backend

    direct_state, _ = seed_state(model, ids)

    def direct_step():
        nonlocal direct_state
        out = model.rwkv7_forward_token(fixed, past_key_values=direct_state)
        direct_state = out.past_key_values

    direct_dt = timed(direct_step, args.device, args.warmup, args.steps)
    direct_backend = last_fast_token_backend(model)

    row: dict[str, Any] = {
        "axis": "forward_fast_path",
        "backend": "hf_adapter",
        "dtype": args.dtype,
        "device": device_name(args.device),
        "attn_mode": args.attn_mode,
        "fuse_norm": getattr(model.config, "fuse_norm", None),
        "fast_cache": os.environ.get("RWKV7_FAST_CACHE", "1") not in _FALSE_VALUES,
        "fast_token_layout": os.environ.get("RWKV7_FAST_TOKEN_LAYOUT", "3d"),
        "fast_token_backend": os.environ.get("RWKV7_FAST_TOKEN_BACKEND", "auto"),
        "prompt_tokens": int(ids.shape[1]),
        "steps": args.steps,
        "reference_forward": metric(ref_dt, args.steps),
        "hf_forward_fast": metric(auto_dt, args.steps),
        "direct_fast_token": metric(direct_dt, args.steps),
        "hf_forward_fast_backend": auto_backend,
        "direct_fast_token_backend": direct_backend,
        "max_abs_diff_auto_vs_reference": round(max_abs_auto, 6),
        "max_abs_diff_direct_vs_reference": round(max_abs_direct, 6),
        "peak_vram_mb": peak_mb(args.device),
    }
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
