#!/usr/bin/env python3
# coding=utf-8
"""Benchmark native fused row-wise W8 R/K/V dequant-GEMV prototype.

This measures the launch-fusion step missing from per-module W8 GEMV telemetry:
three decode-hot RWKV attention projections are computed in one Triton launch
and compared with both fp16 linears and three separate native W8 GEMVs.
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
    int8_fused_rkv_gemv,
    int8_rowwise_gemv,
    int8_weight_footprint_bytes,
    native_int8_fused_rkv_available,
    quantize_int8_rowwise,
)

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


def make_inputs(hidden_size: int, batch_size: int, device: str, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    gen_device = device if device.startswith("cuda") else "cpu"
    base = torch.randn(batch_size, hidden_size, device=gen_device, dtype=dtype)
    prev = torch.randn_like(base)
    delta = prev - base
    return {
        "xr": base + 0.01 * delta,
        "xk": base + 0.03 * delta,
        "xv": base + 0.04 * delta,
    }


def current_fp16_rkv(attn, xs: dict[str, torch.Tensor]):
    return (
        F.linear(xs["xr"], attn.r_proj.weight),
        F.linear(xs["xk"], attn.k_proj.weight),
        F.linear(xs["xv"], attn.v_proj.weight),
    )


def separate_int8_rkv(qpack: dict[str, Any], xs: dict[str, torch.Tensor], block_m: int, block_k: int):
    return (
        int8_rowwise_gemv(xs["xr"], qpack["qr"], qpack["sr"], block_m=block_m, block_k=block_k),
        int8_rowwise_gemv(xs["xk"], qpack["qk"], qpack["sk"], block_m=block_m, block_k=block_k),
        int8_rowwise_gemv(xs["xv"], qpack["qv"], qpack["sv"], block_m=block_m, block_k=block_k),
    )


def fused_int8_rkv(qpack: dict[str, Any], xs: dict[str, torch.Tensor], block_m: int, block_k: int):
    return int8_fused_rkv_gemv(
        xs["xr"],
        xs["xk"],
        xs["xv"],
        qpack["qr"],
        qpack["qk"],
        qpack["qv"],
        qpack["sr"],
        qpack["sk"],
        qpack["sv"],
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


def _mb(num_bytes: float) -> float:
    return round(float(num_bytes) / 1024.0 / 1024.0, 5)


def quantize_attn_rkv(attn) -> dict[str, Any]:
    qr, sr = quantize_int8_rowwise(attn.r_proj.weight.detach())
    qk, sk = quantize_int8_rowwise(attn.k_proj.weight.detach())
    qv, sv = quantize_int8_rowwise(attn.v_proj.weight.detach())
    device = attn.r_proj.weight.device
    return {
        "qr": qr.to(device),
        "qk": qk.to(device),
        "qv": qv.to(device),
        "sr": sr.to(device),
        "sk": sk.to(device),
        "sv": sv.to(device),
    }


def qpack_footprint_bytes(qpack: dict[str, Any]) -> int:
    return (
        int8_weight_footprint_bytes(qpack["qr"], qpack["sr"])
        + int8_weight_footprint_bytes(qpack["qk"], qpack["sk"])
        + int8_weight_footprint_bytes(qpack["qv"], qpack["sv"])
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--dtype", default="fp16", choices=sorted(DTYPES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-mode", default="fused_recurrent", choices=["chunk", "fused_recurrent"])
    ap.add_argument("--fuse-norm", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--layers", nargs="+", type=int, default=[0, 1, 11])
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
    hidden_size = int(model.config.hidden_size)
    xs = make_inputs(hidden_size, args.batch_size, args.device, dtype)

    rows = []
    for layer_idx in args.layers:
        attn = model.model.layers[layer_idx].attn
        qpack = quantize_attn_rkv(attn)
        fp16_bytes = int(attn.r_proj.weight.numel() + attn.k_proj.weight.numel() + attn.v_proj.weight.numel()) * int(attn.r_proj.weight.element_size())
        int8_bytes = qpack_footprint_bytes(qpack)
        with torch.inference_mode():
            fp16_out = current_fp16_rkv(attn, xs)
            separate_out = separate_int8_rkv(qpack, xs, args.block_m, args.block_k)
            fused_out = fused_int8_rkv(qpack, xs, args.block_m, args.block_k)
        fp16_ms = timed(lambda: current_fp16_rkv(attn, xs), args.device, args.warmup, args.steps)
        separate_ms = timed(lambda: separate_int8_rkv(qpack, xs, args.block_m, args.block_k), args.device, args.warmup, args.steps)
        fused_ms = timed(lambda: fused_int8_rkv(qpack, xs, args.block_m, args.block_k), args.device, args.warmup, args.steps)
        rows.append(
            {
                "layer_idx": layer_idx,
                "fp16_current_ms": round(fp16_ms, 5),
                "separate_int8_ms": round(separate_ms, 5),
                "fused_int8_ms": round(fused_ms, 5),
                "fused_speedup_vs_fp16": round(fp16_ms / fused_ms, 4) if fused_ms else None,
                "fused_speedup_vs_separate_int8": round(separate_ms / fused_ms, 4) if fused_ms else None,
                "separate_speedup_vs_fp16": round(fp16_ms / separate_ms, 4) if separate_ms else None,
                "max_abs_diff_fp16_vs_fused": maxdiff_tuple(fp16_out, fused_out),
                "max_abs_diff_separate_vs_fused": maxdiff_tuple(separate_out, fused_out),
                "min_cosine_fp16_vs_fused": cos_min_tuple(fp16_out, fused_out),
                "min_cosine_separate_vs_fused": cos_min_tuple(separate_out, fused_out),
                "fp16_weight_mb": _mb(fp16_bytes),
                "int8_weight_mb": _mb(int8_bytes),
                "footprint_ratio": round(float(int8_bytes) / float(fp16_bytes), 4) if fp16_bytes else None,
            }
        )

    avg_fp16 = sum(float(r["fp16_current_ms"]) for r in rows) / len(rows)
    avg_separate = sum(float(r["separate_int8_ms"]) for r in rows) / len(rows)
    avg_fused = sum(float(r["fused_int8_ms"]) for r in rows) / len(rows)
    total_fp16_mb = sum(float(r["fp16_weight_mb"]) for r in rows)
    total_int8_mb = sum(float(r["int8_weight_mb"]) for r in rows)
    row = {
        "axis": "native_quant_rkv_proto",
        "backend": "hf_adapter",
        "prototype_backend": "triton_int8_fused_rkv_gemv" if native_int8_fused_rkv_available() else "torch_fallback",
        "status": "pass",
        "quantization": "int8_rowwise_fused_rkv",
        "dtype": args.dtype,
        "device": device_name(args.device),
        "attn_mode": args.attn_mode,
        "fuse_norm": getattr(model.config, "fuse_norm", None),
        "batch_size": args.batch_size,
        "hidden_size": hidden_size,
        "layers": args.layers,
        "block_m": args.block_m,
        "block_k": args.block_k,
        "steps": args.steps,
        "avg_fp16_current_ms": round(avg_fp16, 5),
        "avg_separate_int8_ms": round(avg_separate, 5),
        "avg_fused_int8_ms": round(avg_fused, 5),
        "fused_speedup_vs_fp16": round(avg_fp16 / avg_fused, 4) if avg_fused else None,
        "fused_speedup_vs_separate_int8": round(avg_separate / avg_fused, 4) if avg_fused else None,
        "separate_speedup_vs_fp16": round(avg_fp16 / avg_separate, 4) if avg_separate else None,
        "max_abs_diff_fp16_vs_fused": max(float(r["max_abs_diff_fp16_vs_fused"]) for r in rows),
        "max_abs_diff_separate_vs_fused": max(float(r["max_abs_diff_separate_vs_fused"]) for r in rows),
        "min_cosine_fp16_vs_fused": min(float(r["min_cosine_fp16_vs_fused"]) for r in rows),
        "min_cosine_separate_vs_fused": min(float(r["min_cosine_separate_vs_fused"]) for r in rows),
        "sample_fp16_weight_mb": round(total_fp16_mb, 5),
        "sample_int8_weight_mb": round(total_int8_mb, 5),
        "sample_footprint_ratio": round(total_int8_mb / total_fp16_mb, 4) if total_fp16_mb else None,
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
