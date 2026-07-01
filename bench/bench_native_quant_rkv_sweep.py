#!/usr/bin/env python3
# coding=utf-8
"""Single-load block sweep for native W8/W4 fused R/K/V dequant-GEMV.

The per-config prototype benches are useful, but loading the model once per
configuration makes the fp16 cuBLAS baseline drift enough to hide the real gap.
This script loads once, measures a shared fp16 baseline, then sweeps block_m /
block_k for the fused quant R/K/V kernels under the same process and inputs.
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
    int4_fused_rkv_gemv,
    int4_rowwise_gemv,
    int4_weight_footprint_bytes,
    int8_fused_rkv_gemv,
    int8_rowwise_gemv,
    int8_weight_footprint_bytes,
    native_int4_fused_rkv_available,
    native_int8_fused_rkv_available,
    quantize_int4_rowwise,
    quantize_int8_rowwise,
)

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
QUANT_CHOICES = {"w8", "w4"}


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


def _quantize_attn_rkv(attn, quantization: str) -> dict[str, Any]:
    if quantization == "w8":
        pack = quantize_int8_rowwise
    elif quantization == "w4":
        pack = quantize_int4_rowwise
    else:
        raise ValueError(f"unknown quantization {quantization!r}")
    qr, sr = pack(attn.r_proj.weight.detach())
    qk, sk = pack(attn.k_proj.weight.detach())
    qv, sv = pack(attn.v_proj.weight.detach())
    device = attn.r_proj.weight.device
    return {
        "qr": qr.to(device),
        "qk": qk.to(device),
        "qv": qv.to(device),
        "sr": sr.to(device),
        "sk": sk.to(device),
        "sv": sv.to(device),
    }


def _footprint_bytes(qpack: dict[str, Any], quantization: str) -> int:
    if quantization == "w8":
        fn = int8_weight_footprint_bytes
    elif quantization == "w4":
        fn = int4_weight_footprint_bytes
    else:
        raise ValueError(f"unknown quantization {quantization!r}")
    return int(fn(qpack["qr"], qpack["sr"]) + fn(qpack["qk"], qpack["sk"]) + fn(qpack["qv"], qpack["sv"]))


def _separate_rkv(qpack: dict[str, Any], xs: dict[str, torch.Tensor], quantization: str, block_m: int, block_k: int):
    if quantization == "w8":
        fn = int8_rowwise_gemv
    elif quantization == "w4":
        fn = int4_rowwise_gemv
    else:
        raise ValueError(f"unknown quantization {quantization!r}")
    return (
        fn(xs["xr"], qpack["qr"], qpack["sr"], block_m=block_m, block_k=block_k),
        fn(xs["xk"], qpack["qk"], qpack["sk"], block_m=block_m, block_k=block_k),
        fn(xs["xv"], qpack["qv"], qpack["sv"], block_m=block_m, block_k=block_k),
    )


def _fused_rkv(qpack: dict[str, Any], xs: dict[str, torch.Tensor], quantization: str, block_m: int, block_k: int):
    if quantization == "w8":
        fn = int8_fused_rkv_gemv
    elif quantization == "w4":
        fn = int4_fused_rkv_gemv
    else:
        raise ValueError(f"unknown quantization {quantization!r}")
    return fn(
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


def _prototype_backend(quantization: str) -> str:
    if quantization == "w8":
        return "triton_int8_fused_rkv_gemv" if native_int8_fused_rkv_available() else "torch_fallback"
    if quantization == "w4":
        return "triton_int4_fused_rkv_gemv" if native_int4_fused_rkv_available() else "torch_fallback"
    raise ValueError(f"unknown quantization {quantization!r}")


def run_quantization(args: argparse.Namespace, model, xs: dict[str, torch.Tensor], quantization: str) -> dict[str, Any]:
    layer_refs = []
    total_fp16_bytes = 0
    total_quant_bytes = 0
    for layer_idx in args.layers:
        attn = model.model.layers[layer_idx].attn
        qpack = _quantize_attn_rkv(attn, quantization)
        fp16_bytes = int(attn.r_proj.weight.numel() + attn.k_proj.weight.numel() + attn.v_proj.weight.numel()) * int(attn.r_proj.weight.element_size())
        quant_bytes = _footprint_bytes(qpack, quantization)
        fp16_out = current_fp16_rkv(attn, xs)
        fp16_ms = timed(lambda a=attn: current_fp16_rkv(a, xs), args.device, args.warmup, args.steps)
        total_fp16_bytes += fp16_bytes
        total_quant_bytes += quant_bytes
        layer_refs.append(
            {
                "layer_idx": layer_idx,
                "attn": attn,
                "qpack": qpack,
                "fp16_out": fp16_out,
                "fp16_ms": fp16_ms,
                "fp16_weight_mb": _mb(fp16_bytes),
                "quant_weight_mb": _mb(quant_bytes),
                "footprint_ratio": round(float(quant_bytes) / float(fp16_bytes), 4) if fp16_bytes else None,
            }
        )

    configs = []
    for block_m in args.block_m:
        for block_k in args.block_k:
            layer_rows = []
            for ref in layer_refs:
                with torch.inference_mode():
                    separate_out = _separate_rkv(ref["qpack"], xs, quantization, block_m, block_k)
                    fused_out = _fused_rkv(ref["qpack"], xs, quantization, block_m, block_k)
                separate_ms = timed(lambda r=ref: _separate_rkv(r["qpack"], xs, quantization, block_m, block_k), args.device, args.warmup, args.steps)
                fused_ms = timed(lambda r=ref: _fused_rkv(r["qpack"], xs, quantization, block_m, block_k), args.device, args.warmup, args.steps)
                layer_rows.append(
                    {
                        "layer_idx": ref["layer_idx"],
                        "fp16_current_ms": round(float(ref["fp16_ms"]), 5),
                        "separate_quant_ms": round(float(separate_ms), 5),
                        "fused_quant_ms": round(float(fused_ms), 5),
                        "fused_speedup_vs_fp16": round(float(ref["fp16_ms"]) / float(fused_ms), 4) if fused_ms else None,
                        "fused_speedup_vs_separate": round(float(separate_ms) / float(fused_ms), 4) if fused_ms else None,
                        "separate_speedup_vs_fp16": round(float(ref["fp16_ms"]) / float(separate_ms), 4) if separate_ms else None,
                        "max_abs_diff_fp16_vs_fused": maxdiff_tuple(ref["fp16_out"], fused_out),
                        "max_abs_diff_separate_vs_fused": maxdiff_tuple(separate_out, fused_out),
                        "min_cosine_fp16_vs_fused": cos_min_tuple(ref["fp16_out"], fused_out),
                        "min_cosine_separate_vs_fused": cos_min_tuple(separate_out, fused_out),
                    }
                )
            avg_fp16 = sum(float(r["fp16_current_ms"]) for r in layer_rows) / len(layer_rows)
            avg_separate = sum(float(r["separate_quant_ms"]) for r in layer_rows) / len(layer_rows)
            avg_fused = sum(float(r["fused_quant_ms"]) for r in layer_rows) / len(layer_rows)
            configs.append(
                {
                    "block_m": int(block_m),
                    "block_k": int(block_k),
                    "avg_fp16_current_ms": round(avg_fp16, 5),
                    "avg_separate_quant_ms": round(avg_separate, 5),
                    "avg_fused_quant_ms": round(avg_fused, 5),
                    "fused_speedup_vs_fp16": round(avg_fp16 / avg_fused, 4) if avg_fused else None,
                    "fused_speedup_vs_separate": round(avg_separate / avg_fused, 4) if avg_fused else None,
                    "separate_speedup_vs_fp16": round(avg_fp16 / avg_separate, 4) if avg_separate else None,
                    "max_abs_diff_fp16_vs_fused": max(float(r["max_abs_diff_fp16_vs_fused"]) for r in layer_rows),
                    "max_abs_diff_separate_vs_fused": max(float(r["max_abs_diff_separate_vs_fused"]) for r in layer_rows),
                    "min_cosine_fp16_vs_fused": min(float(r["min_cosine_fp16_vs_fused"]) for r in layer_rows),
                    "min_cosine_separate_vs_fused": min(float(r["min_cosine_separate_vs_fused"]) for r in layer_rows),
                    "layer_rows": layer_rows if args.include_layer_rows else None,
                }
            )

    def compact_config(cfg: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in cfg.items() if k != "layer_rows" or args.include_layer_rows}

    best_by_speedup = max(configs, key=lambda r: float(r.get("fused_speedup_vs_fp16") or 0.0))
    best_by_latency = min(configs, key=lambda r: float(r.get("avg_fused_quant_ms") or float("inf")))
    avg_fp16_baseline = sum(float(r["fp16_ms"]) for r in layer_refs) / len(layer_refs)
    row = {
        "axis": "native_quant_rkv_sweep",
        "backend": "hf_adapter",
        "prototype_backend": _prototype_backend(quantization),
        "status": "pass",
        "quantization": "int8_rowwise_fused_rkv" if quantization == "w8" else "int4_rowwise_fused_rkv",
        "dtype": args.dtype,
        "device": device_name(args.device),
        "attn_mode": args.attn_mode,
        "fuse_norm": getattr(model.config, "fuse_norm", None),
        "batch_size": args.batch_size,
        "hidden_size": int(model.config.hidden_size),
        "layers": args.layers,
        "block_m_values": [int(v) for v in args.block_m],
        "block_k_values": [int(v) for v in args.block_k],
        "warmup": args.warmup,
        "steps": args.steps,
        "avg_fp16_baseline_ms": round(avg_fp16_baseline, 5),
        "best_by_speedup_vs_fp16": compact_config(best_by_speedup),
        "best_by_latency": compact_config(best_by_latency),
        "configs": [compact_config(c) for c in configs],
        "sample_fp16_weight_mb": _mb(total_fp16_bytes),
        "sample_quant_weight_mb": _mb(total_quant_bytes),
        "sample_footprint_ratio": round(float(total_quant_bytes) / float(total_fp16_bytes), 4) if total_fp16_bytes else None,
        "peak_vram_mb": peak_mb(args.device),
    }
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--dtype", default="fp16", choices=sorted(DTYPES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-mode", default="fused_recurrent", choices=["chunk", "fused_recurrent"])
    ap.add_argument("--fuse-norm", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--layers", nargs="+", type=int, default=[0, 1, 11])
    ap.add_argument("--quantizations", nargs="+", choices=sorted(QUANT_CHOICES), default=["w8", "w4"])
    ap.add_argument("--block-m", nargs="+", type=int, default=[8, 16, 32])
    ap.add_argument("--block-k", nargs="+", type=int, default=[32, 64, 128])
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--include-layer-rows", action="store_true")
    ap.add_argument("--results", default=str(Path(__file__).parent / "results.jsonl"))
    args = ap.parse_args()

    dtype = DTYPES[args.dtype]
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = load_model(args, dtype)
    xs = make_inputs(int(model.config.hidden_size), args.batch_size, args.device, dtype)

    rows = []
    for quantization in args.quantizations:
        row = run_quantization(args, model, xs, quantization)
        rows.append(row)
        print(json.dumps(row, indent=2, ensure_ascii=False), flush=True)

    if args.results:
        out = Path(args.results)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nappended {len(rows)} row(s) -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
