#!/usr/bin/env python3
# coding=utf-8
"""Benchmark optional fused fp16 FFN path prototype."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("RWKV_V7_ON", "1")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from rwkv7_hf.fused_ffn import fused_ffn, fused_ffn_available

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


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


def load_model(args, dtype):
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


def make_inputs(hidden_size: int, batch_size: int, device: str, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    gen_device = device if device.startswith("cuda") else "cpu"
    x = torch.randn(batch_size, 1, hidden_size, device=gen_device, dtype=dtype)
    prev = torch.randn_like(x)
    return x, prev


def current_ffn(ffn, x: torch.Tensor, prev: torch.Tensor):
    shifted = torch.addcmul(x, prev - x, ffn.x_k.view(1, 1, -1))
    out = F.linear(torch.relu(F.linear(shifted, ffn.key.weight)) ** 2, ffn.value.weight)
    return out, x[:, -1:].contiguous()


def prototype_ffn(ffn, x: torch.Tensor, prev: torch.Tensor, block_m: int, block_k: int):
    return fused_ffn(
        x,
        prev,
        ffn.x_k,
        ffn.key.weight,
        ffn.value.weight,
        block_m=block_m,
        block_k=block_k,
    )


def timed(fn: Callable[[], Any], device: str, warmup: int, steps: int) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
    cuda_sync(device)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(steps):
            fn()
    cuda_sync(device)
    return (time.perf_counter() - t0) * 1000.0 / steps


def maxdiff_tuple(a, b) -> float:
    return max(float((x.float() - y.float()).abs().max().detach().cpu()) for x, y in zip(a, b, strict=False))


def cos_min_tuple(a, b) -> float:
    vals = []
    for x, y in zip(a, b, strict=False):
        xf = x.float().reshape(x.shape[0], -1)
        yf = y.float().reshape(y.shape[0], -1)
        vals.append(float(torch.nn.functional.cosine_similarity(xf, yf, dim=-1).min().detach().cpu()))
    return min(vals)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--dtype", default="fp16", choices=sorted(DTYPES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-mode", default="fused_recurrent", choices=["chunk", "fused_recurrent"])
    ap.add_argument("--fuse-norm", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--layers", nargs="+", type=int, default=[0, 1, 11])
    ap.add_argument("--block-m", type=int, default=64)
    ap.add_argument("--block-k", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--results", default=str(Path(__file__).parent / "results.jsonl"))
    args = ap.parse_args()

    dtype = DTYPES[args.dtype]
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = load_model(args, dtype)
    hidden_size = int(model.config.hidden_size)
    x, prev = make_inputs(hidden_size, args.batch_size, args.device, dtype)

    layer_rows = []
    intermediates = []
    for layer_idx in args.layers:
        ffn = model.model.layers[layer_idx].ffn
        intermediate_size = int(ffn.key.weight.shape[0])
        intermediates.append(intermediate_size)
        with torch.inference_mode():
            cur = current_ffn(ffn, x, prev)
            proto = prototype_ffn(ffn, x, prev, args.block_m, args.block_k)
        current_ms = timed(lambda: current_ffn(ffn, x, prev), args.device, args.warmup, args.steps)
        prototype_ms = timed(lambda: prototype_ffn(ffn, x, prev, args.block_m, args.block_k), args.device, args.warmup, args.steps)
        layer_rows.append(
            {
                "layer_idx": layer_idx,
                "intermediate_size": intermediate_size,
                "current_ms": round(current_ms, 5),
                "prototype_ms": round(prototype_ms, 5),
                "speedup": round(current_ms / prototype_ms, 4) if prototype_ms else None,
                "max_abs_diff": maxdiff_tuple(cur, proto),
                "min_cosine": cos_min_tuple(cur, proto),
            }
        )

    avg_current = sum(float(r["current_ms"]) for r in layer_rows) / len(layer_rows)
    avg_prototype = sum(float(r["prototype_ms"]) for r in layer_rows) / len(layer_rows)
    row = {
        "axis": "fused_ffn_proto",
        "backend": "hf_adapter",
        "prototype_backend": "triton_ffn_shift_key_relu_value" if fused_ffn_available() else "torch_fallback",
        "status": "pass",
        "dtype": args.dtype,
        "device": device_name(args.device),
        "attn_mode": args.attn_mode,
        "fuse_norm": getattr(model.config, "fuse_norm", None),
        "batch_size": args.batch_size,
        "hidden_size": hidden_size,
        "intermediate_sizes": sorted(set(intermediates)),
        "layers": args.layers,
        "block_m": args.block_m,
        "block_k": args.block_k,
        "steps": args.steps,
        "avg_current_ms": round(avg_current, 5),
        "avg_prototype_ms": round(avg_prototype, 5),
        "avg_speedup": round(avg_current / avg_prototype, 4) if avg_prototype else None,
        "max_abs_diff": max(float(r["max_abs_diff"]) for r in layer_rows),
        "min_cosine": min(float(r["min_cosine"]) for r in layer_rows),
        "layer_rows": layer_rows,
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
