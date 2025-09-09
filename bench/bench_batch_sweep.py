#!/usr/bin/env python3
# coding=utf-8
"""Batch-size sweep benchmark for the RWKV-7 HF adapter.

Measures serving-style prefill and recurrent decode for multiple batch sizes.
The batched `rwkv7_forward_token` API is included when available; older adapter
builds fall back to the bsz=1 `rwkv7_forward_one` API.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("RWKV_V7_ON", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
SEED = "The quick brown fox jumps over the lazy dog. " * 256


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


def timed(fn, device: str, runs: int) -> float:
    cuda_sync(device)
    t0 = time.time()
    for _ in range(runs):
        fn()
    cuda_sync(device)
    return (time.time() - t0) / runs


def load_model(args, dtype):
    if args.fast_cache != "auto":
        os.environ["RWKV7_FAST_CACHE"] = "1" if args.fast_cache == "true" else "0"
    if args.fast_token_backend != "auto":
        os.environ["RWKV7_FAST_TOKEN_BACKEND"] = args.fast_token_backend
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


def encode(tok, prompt_tokens: int, bsz: int, device: str) -> torch.Tensor:
    ids = tok(SEED, return_tensors="pt", add_special_tokens=False).input_ids[:, :prompt_tokens]
    ids = ids.repeat(bsz, 1)
    return ids.to(device) if device.startswith("cuda") else ids


def bench_one(args, tok, model, bsz: int) -> list[dict[str, Any]]:
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    ids = encode(tok, args.prompt_tokens, bsz, args.device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = model(ids, use_cache=True, logits_to_keep=args.hf_logits_to_keep)
    prefill_dt = timed(lambda: model(ids, use_cache=True, logits_to_keep=args.hf_logits_to_keep), args.device, args.runs)

    with torch.inference_mode():
        out = model(ids[:, :8], use_cache=True, logits_to_keep=args.hf_logits_to_keep)
        state = out.past_key_values
        nxt = out.logits[:, -1:].argmax(dim=-1)
        for _ in range(args.warmup):
            out = model(nxt, past_key_values=state, use_cache=True, logits_to_keep=args.hf_logits_to_keep)
            state = out.past_key_values
            nxt = out.logits[:, -1:].argmax(dim=-1)
        cuda_sync(args.device)
        t0 = time.time()
        for _ in range(args.decode_tokens):
            out = model(nxt, past_key_values=state, use_cache=True, logits_to_keep=args.hf_logits_to_keep)
            state = out.past_key_values
            nxt = out.logits[:, -1:].argmax(dim=-1)
        cuda_sync(args.device)
        decode_dt = time.time() - t0

    rows = [{
        "axis": "batch_sweep",
        "backend": "hf_adapter",
        "decode_api": "forward",
        "dtype": args.dtype,
        "device": device_name(args.device),
        "attn_mode": args.attn_mode,
        "fuse_norm": getattr(model.config, "fuse_norm", None),
        "fast_cache": os.environ.get("RWKV7_FAST_CACHE", "1") not in {"0", "false", "False", "no", "off"},
        "cache_type": type(state).__name__ if state is not None else None,
        "batch_size": bsz,
        "prompt_tokens": int(ids.shape[1]),
        "decode_tokens": args.decode_tokens,
        "prefill_tokps_total": round((bsz * int(ids.shape[1])) / prefill_dt, 1),
        "prefill_tokps_per_seq": round(int(ids.shape[1]) / prefill_dt, 1),
        "prefill_ms": round(1000 * prefill_dt, 2),
        "decode_tokps_total": round((bsz * args.decode_tokens) / decode_dt, 1),
        "decode_tokps_per_seq": round(args.decode_tokens / decode_dt, 1),
        "decode_ms_per_step": round(1000 * decode_dt / args.decode_tokens, 2),
        "peak_vram_mb": peak_mb(args.device),
    }]

    fast_fn = getattr(model, "rwkv7_forward_token", None)
    fast_name = "rwkv7_forward_token" if fast_fn is not None else None
    if fast_fn is None and bsz == 1:
        fast_fn = getattr(model, "rwkv7_forward_one", None)
        fast_name = "rwkv7_forward_one" if fast_fn is not None else None

    if args.fast_decode_api != "false" and fast_fn is not None:
        requested_backend = os.environ.get("RWKV7_FAST_TOKEN_BACKEND", "fla")
        effective_backend = "native_jit" if requested_backend == "native_jit" else "fla"
        with torch.inference_mode():
            out = model(ids[:, :8], use_cache=True, logits_to_keep=args.hf_logits_to_keep)
            state = out.past_key_values
            nxt = out.logits[:, -1:].argmax(dim=-1)
            for _ in range(args.warmup):
                out = fast_fn(nxt, past_key_values=state)
                state = out.past_key_values
                nxt = out.logits[:, -1:].argmax(dim=-1)
            cuda_sync(args.device)
            t0 = time.time()
            for _ in range(args.decode_tokens):
                out = fast_fn(nxt, past_key_values=state)
                state = out.past_key_values
                nxt = out.logits[:, -1:].argmax(dim=-1)
            cuda_sync(args.device)
            fast_dt = time.time() - t0
        rows.append({**rows[0],
            "decode_api": fast_name,
            "fast_token_backend": requested_backend,
            "fast_token_backend_effective": effective_backend,
            "decode_tokps_total": round((bsz * args.decode_tokens) / fast_dt, 1),
            "decode_tokps_per_seq": round(args.decode_tokens / fast_dt, 1),
            "decode_ms_per_step": round(1000 * fast_dt / args.decode_tokens, 2),
            "cache_type": type(state).__name__ if state is not None else None,
            "peak_vram_mb": peak_mb(args.device),
        })
    elif args.fast_decode_api == "true":
        raise ValueError("Loaded model does not expose a fast one-token decode API for this batch size")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--dtype", default="fp16", choices=sorted(DTYPES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-mode", default="fused_recurrent", choices=["chunk", "fused_recurrent"])
    ap.add_argument("--fuse-norm", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--fast-cache", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--fast-decode-api", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--fast-token-backend", choices=["auto", "fla", "native_jit"], default="auto",
                    help="Fast-token backend; native_jit applies to bsz=1 and falls back to FLA for batched requests")
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--decode-tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--hf-logits-to-keep", type=int, default=1)
    ap.add_argument("--results", default=str(Path(__file__).parent / "results.jsonl"))
    args = ap.parse_args()

    dtype = DTYPES[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)
    model = load_model(args, dtype)
    all_rows: list[dict[str, Any]] = []
    for bsz in args.batch_sizes:
        print(f"\n===== batch_size={bsz} =====", flush=True)
        rows = bench_one(args, tok, model, bsz)
        all_rows.extend(rows)
        for row in rows:
            print(json.dumps(row, indent=2), flush=True)
    if args.results:
        out = Path(args.results)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as f:
            for row in all_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nappended {len(all_rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
