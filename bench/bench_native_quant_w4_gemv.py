#!/usr/bin/env python3
# coding=utf-8
"""Benchmark native row-wise W4 dequant-GEMV prototypes for RWKV-7 linears.

This is the first RWKV-native W4 quantization serving prototype. It compares the
current fp16 linear calls with a row-wise int4 packed weight plus fused dequant
GEMV. The row is telemetry for replacing generic bitsandbytes kernels only after
speed and accuracy are acceptable.
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
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from rwkv7_hf.native_quant import (
    int4_rowwise_gemv,
    int4_weight_footprint_bytes,
    native_int4_gemv_available,
    quantize_int4_rowwise,
)

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
DEFAULT_MODULES = ["attn.r_proj", "attn.k_proj", "attn.v_proj", "attn.o_proj", "ffn.key", "ffn.value"]


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


def get_module(layer, name: str):
    obj: Any = layer
    for part in name.split("."):
        obj = getattr(obj, part)
    if not hasattr(obj, "weight"):
        raise ValueError(f"{name} did not resolve to a Linear-like module")
    return obj


def make_input(batch_size: int, in_features: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    gen_device = device if device.startswith("cuda") else "cpu"
    return torch.randn(batch_size, in_features, device=gen_device, dtype=dtype)


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


def cosine_min(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().reshape(a.shape[0], -1)
    bf = b.float().reshape(b.shape[0], -1)
    return float(torch.nn.functional.cosine_similarity(af, bf, dim=-1).min().detach().cpu())


def _mb(num_bytes: float) -> float:
    return round(float(num_bytes) / 1024.0 / 1024.0, 5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--dtype", default="fp16", choices=sorted(DTYPES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-mode", default="fused_recurrent", choices=["chunk", "fused_recurrent"])
    ap.add_argument("--fuse-norm", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--layers", nargs="+", type=int, default=[0, 1, 11])
    ap.add_argument("--modules", nargs="+", default=DEFAULT_MODULES)
    ap.add_argument("--block-m", type=int, default=16)
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

    rows = []
    for layer_idx in args.layers:
        layer = model.model.layers[layer_idx]
        for module_name in args.modules:
            module = get_module(layer, module_name)
            weight = module.weight.detach()
            bias = getattr(module, "bias", None)
            q_weight, scales = quantize_int4_rowwise(weight)
            q_weight = q_weight.to(weight.device)
            scales = scales.to(weight.device)
            x = make_input(args.batch_size, int(weight.shape[1]), args.device, dtype)
            with torch.inference_mode():
                current = F.linear(x, weight, bias)
                proto = int4_rowwise_gemv(x, q_weight, scales, bias, block_m=args.block_m, block_k=args.block_k)
            current_ms = timed(lambda: F.linear(x, weight, bias), args.device, args.warmup, args.steps)
            prototype_ms = timed(
                lambda: int4_rowwise_gemv(x, q_weight, scales, bias, block_m=args.block_m, block_k=args.block_k),
                args.device,
                args.warmup,
                args.steps,
            )
            fp16_bytes = int(weight.numel()) * int(weight.element_size()) + (int(bias.numel()) * int(bias.element_size()) if bias is not None else 0)
            int4_bytes = int4_weight_footprint_bytes(q_weight, scales, bias)
            rows.append(
                {
                    "layer_idx": layer_idx,
                    "module": module_name,
                    "shape": [int(weight.shape[0]), int(weight.shape[1])],
                    "has_bias": bias is not None,
                    "current_ms": round(current_ms, 5),
                    "prototype_ms": round(prototype_ms, 5),
                    "speedup": round(current_ms / prototype_ms, 4) if prototype_ms else None,
                    "max_abs_diff": float((current.float() - proto.float()).abs().max().detach().cpu()),
                    "mean_abs_diff": float((current.float() - proto.float()).abs().mean().detach().cpu()),
                    "min_cosine": cosine_min(current, proto),
                    "fp16_weight_mb": _mb(fp16_bytes),
                    "int4_weight_mb": _mb(int4_bytes),
                    "footprint_ratio": round(float(int4_bytes) / float(fp16_bytes), 4) if fp16_bytes else None,
                }
            )

    pass_rows = rows
    avg_current = sum(float(r["current_ms"]) for r in pass_rows) / len(pass_rows)
    avg_prototype = sum(float(r["prototype_ms"]) for r in pass_rows) / len(pass_rows)
    total_fp16_mb = sum(float(r["fp16_weight_mb"]) for r in pass_rows)
    total_int4_mb = sum(float(r["int4_weight_mb"]) for r in pass_rows)
    row = {
        "axis": "native_quant_w4_gemv_proto",
        "backend": "hf_adapter",
        "prototype_backend": "triton_int4_rowwise_gemv" if native_int4_gemv_available() else "torch_fallback",
        "status": "pass",
        "quantization": "int4_rowwise",
        "dtype": args.dtype,
        "device": device_name(args.device),
        "attn_mode": args.attn_mode,
        "fuse_norm": getattr(model.config, "fuse_norm", None),
        "batch_size": args.batch_size,
        "layers": args.layers,
        "modules": args.modules,
        "block_m": args.block_m,
        "block_k": args.block_k,
        "steps": args.steps,
        "avg_current_ms": round(avg_current, 5),
        "avg_prototype_ms": round(avg_prototype, 5),
        "avg_speedup": round(avg_current / avg_prototype, 4) if avg_prototype else None,
        "max_abs_diff": max(float(r["max_abs_diff"]) for r in pass_rows),
        "mean_abs_diff_max": max(float(r["mean_abs_diff"]) for r in pass_rows),
        "min_cosine": min(float(r["min_cosine"]) for r in pass_rows),
        "sample_fp16_weight_mb": round(total_fp16_mb, 5),
        "sample_int4_weight_mb": round(total_int4_mb, 5),
        "sample_footprint_ratio": round(total_int4_mb / total_fp16_mb, 4) if total_fp16_mb else None,
        "layer_rows": rows,
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
