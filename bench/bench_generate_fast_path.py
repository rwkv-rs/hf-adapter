#!/usr/bin/env python3
# coding=utf-8
"""Benchmark `model.generate()` with and without RWKV-7 cached fast-forward.

This covers the highest-level HF inference API. It verifies that enabling
`RWKV7_FAST_FORWARD=1` preserves greedy output while speeding up cached
one-token decode inside `generate(..., use_cache=True)`.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

os.environ.setdefault("RWKV_V7_ON", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
_FALSE_VALUES = {"0", "false", "False", "no", "off"}


@contextmanager
def fast_forward_env(enabled: bool):
    old = os.environ.get("RWKV7_FAST_FORWARD")
    old_native_backend = os.environ.get("RWKV7_NATIVE_MODEL_BACKEND")
    os.environ["RWKV7_FAST_FORWARD"] = "1" if enabled else "0"
    if not enabled:
        os.environ["RWKV7_NATIVE_MODEL_BACKEND"] = "eager"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("RWKV7_FAST_FORWARD", None)
        else:
            os.environ["RWKV7_FAST_FORWARD"] = old
        if old_native_backend is None:
            os.environ.pop("RWKV7_NATIVE_MODEL_BACKEND", None)
        else:
            os.environ["RWKV7_NATIVE_MODEL_BACKEND"] = old_native_backend


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


def last_fast_token_backend(model):
    getter = getattr(model, "rwkv7_last_fast_token_backend", None)
    if callable(getter):
        return getter()
    return getattr(model, "_rwkv7_last_fast_token_backend", None)


def configure_env(args: argparse.Namespace) -> None:
    if args.fast_cache != "auto":
        os.environ["RWKV7_FAST_CACHE"] = "1" if args.fast_cache == "true" else "0"
    if args.fast_token_layout != "auto":
        os.environ["RWKV7_FAST_TOKEN_LAYOUT"] = args.fast_token_layout
    os.environ["RWKV7_FAST_TOKEN_BACKEND"] = args.fast_token_backend
    os.environ["RWKV7_NATIVE_MODEL_BACKEND"] = args.fast_token_backend


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


def generate_once(model, enc: dict[str, torch.Tensor], *, enabled: bool, max_new_tokens: int) -> torch.Tensor:
    with fast_forward_env(enabled):
        return model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )


def time_generate(model, enc: dict[str, torch.Tensor], args: argparse.Namespace, *, enabled: bool) -> tuple[float, torch.Tensor]:
    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = generate_once(model, enc, enabled=enabled, max_new_tokens=args.warmup_new_tokens)
    cuda_sync(args.device)
    generated = None
    t0 = time.time()
    with torch.inference_mode():
        for _ in range(args.runs):
            generated = generate_once(model, enc, enabled=enabled, max_new_tokens=args.max_new_tokens)
    cuda_sync(args.device)
    return (time.time() - t0) / max(args.runs, 1), generated


def gen_metric(dt_s: float, batch: int, new_tokens: int) -> dict[str, float]:
    total = batch * new_tokens
    return {
        "s": round(dt_s, 4),
        "tokps": round(total / dt_s, 1) if dt_s > 0 else float("inf"),
        "ms_per_token": round(1000.0 * dt_s / max(total, 1), 4),
    }


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
    ap.add_argument("--prompt", default="User: Hello!\n\nAssistant:")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--warmup-new-tokens", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--results", default=str(Path(__file__).parent / "results.jsonl"))
    args = ap.parse_args()

    dtype = DTYPES[args.dtype]
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    tok = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)
    model = load_model(args, dtype)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    prompt_input = args.prompt if args.batch_size == 1 else [args.prompt for _ in range(args.batch_size)]
    enc = tok(prompt_input, return_tensors="pt", add_special_tokens=False)
    if args.device.startswith("cuda"):
        enc = {k: v.cuda() for k, v in enc.items()}
    batch = int(enc["input_ids"].shape[0])
    prompt_tokens = int(enc["input_ids"].shape[1])

    ref_dt, ref_gen = time_generate(model, enc, args, enabled=False)
    fast_dt, fast_gen = time_generate(model, enc, args, enabled=True)
    fast_backend = last_fast_token_backend(model)

    ref_tail = ref_gen[:, -args.max_new_tokens :].detach().cpu()
    fast_tail = fast_gen[:, -args.max_new_tokens :].detach().cpu()
    token_matches = int((ref_tail == fast_tail).sum().item())
    token_total = int(ref_tail.numel())
    generated_equal = bool(torch.equal(ref_tail, fast_tail))
    ref_metric = gen_metric(ref_dt, batch, args.max_new_tokens)
    fast_metric = gen_metric(fast_dt, batch, args.max_new_tokens)
    speedup = fast_metric["tokps"] / ref_metric["tokps"] if ref_metric["tokps"] else None

    row: dict[str, Any] = {
        "axis": "generate_fast_path",
        "backend": "hf_adapter",
        "dtype": args.dtype,
        "device": device_name(args.device),
        "attn_mode": args.attn_mode,
        "fuse_norm": getattr(model.config, "fuse_norm", None),
        "fast_cache": os.environ.get("RWKV7_FAST_CACHE", "1") not in _FALSE_VALUES,
        "fast_token_layout": os.environ.get("RWKV7_FAST_TOKEN_LAYOUT", "3d"),
        "fast_token_backend": os.environ.get("RWKV7_FAST_TOKEN_BACKEND", "auto"),
        "fast_token_backend_effective": fast_backend,
        "batch_size": batch,
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": args.max_new_tokens,
        "runs": args.runs,
        "reference_generate": ref_metric,
        "hf_generate_fast": fast_metric,
        "speedup_vs_reference": round(speedup, 4) if speedup is not None else None,
        "generated_equal": generated_equal,
        "generated_tokens_matched": token_matches,
        "generated_tokens_total": token_total,
        "reference_tail": ref_tail[0].tolist() if batch == 1 else None,
        "fast_tail": fast_tail[0].tolist() if batch == 1 else None,
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
