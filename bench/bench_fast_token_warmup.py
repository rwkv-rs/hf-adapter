#!/usr/bin/env python3
# coding=utf-8
"""Benchmark/preflight fast-token backend warmup for serving.

For native CUDA graph serving, the first request for a new active batch size has
to capture a graph. This script calls the public `rwkv7_warmup_fast_token()` API
for a set of serving batch sizes, records the effective backend per size, and
verifies the native-graph runner cache contains the expected sizes.
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
from transformers import AutoModelForCausalLM

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
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


def configure_env(args: argparse.Namespace) -> None:
    if args.fast_cache != "auto":
        os.environ["RWKV7_FAST_CACHE"] = "1" if args.fast_cache == "true" else "0"
    os.environ["RWKV7_FAST_TOKEN_BACKEND"] = args.fast_token_backend
    os.environ["RWKV7_NATIVE_GRAPH_CACHE_SIZE"] = str(args.native_graph_cache_size)


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--dtype", default="fp16", choices=sorted(DTYPES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-mode", default="fused_recurrent", choices=["chunk", "fused_recurrent"])
    ap.add_argument("--fuse-norm", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--fast-cache", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--fast-token-backend", choices=["auto", "fla", "native_jit", "native_graph"], default="auto")
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--native-graph-cache-size", type=int, default=8)
    ap.add_argument("--results", default=str(Path(__file__).parent / "results.jsonl"))
    args = ap.parse_args()

    dtype = DTYPES[args.dtype]
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = load_model(args, dtype)
    if not hasattr(model, "rwkv7_warmup_fast_token"):
        raise ValueError("Loaded model does not expose rwkv7_warmup_fast_token")

    clear_fn = getattr(model, "rwkv7_clear_native_graph_cache", None)
    cleared_before = clear_fn() if callable(clear_fn) else None
    cuda_sync(args.device)
    t0 = time.time()
    warmed = model.rwkv7_warmup_fast_token(args.batch_sizes, backend=args.fast_token_backend)
    cuda_sync(args.device)
    warmup_s = time.time() - t0
    cache_sizes = []
    cache_getter = getattr(model, "rwkv7_native_graph_cache_batch_sizes", None)
    if callable(cache_getter):
        cache_sizes = cache_getter()

    row: dict[str, Any] = {
        "axis": "fast_token_warmup",
        "backend": "hf_adapter",
        "dtype": args.dtype,
        "device": device_name(args.device),
        "attn_mode": args.attn_mode,
        "fuse_norm": getattr(model.config, "fuse_norm", None),
        "fast_cache": os.environ.get("RWKV7_FAST_CACHE", "1") not in _FALSE_VALUES,
        "fast_token_backend": os.environ.get("RWKV7_FAST_TOKEN_BACKEND", "auto"),
        "batch_sizes": [int(v) for v in args.batch_sizes],
        "effective_backend_by_batch": {str(k): v for k, v in warmed.items()},
        "native_graph_cache_batch_sizes": cache_sizes,
        "native_graph_cache_size_limit": args.native_graph_cache_size,
        "cleared_before": cleared_before,
        "warmup_s": round(warmup_s, 4),
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
