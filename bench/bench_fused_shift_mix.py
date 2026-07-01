#!/usr/bin/env python3
# coding=utf-8
"""Benchmark optional fused RWKV-7 attention shift-mix prototype.

The first fused R/K/V GEMV prototype proved that a naive custom GEMV does not
beat cuBLAS on V100.  This benchmark tests a lower-level decode bottleneck that
is launch-bound in the native_graph path: materializing the six attention
shift-mix inputs.  Rows are telemetry for deciding whether to integrate the
Triton shift-mix kernel into the captured fast-token graph.
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
from transformers import AutoModelForCausalLM

from rwkv7_hf.fused_time_mix import fused_attn_shift_mix, fused_attn_shift_mix_available

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


def make_inputs(hidden_size: int, batch_size: int, device: str, dtype: torch.dtype, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    gen_device = device if device.startswith("cuda") else "cpu"
    x = torch.randn(batch_size, hidden_size, device=gen_device, dtype=dtype)
    prev = torch.randn_like(x)
    if rank == 3:
        return x.unsqueeze(1), prev.unsqueeze(1)
    return x, prev


def current_shift_mix(attn, x: torch.Tensor, prev: torch.Tensor):
    delta = prev - x
    if x.dim() == 3:
        return (
            torch.addcmul(x, delta, attn.x_r),
            torch.addcmul(x, delta, attn.x_w),
            torch.addcmul(x, delta, attn.x_k),
            torch.addcmul(x, delta, attn.x_v),
            torch.addcmul(x, delta, attn.x_a),
            torch.addcmul(x, delta, attn.x_g),
        )
    return (
        torch.addcmul(x, delta, attn.x_r.view(1, -1)),
        torch.addcmul(x, delta, attn.x_w.view(1, -1)),
        torch.addcmul(x, delta, attn.x_k.view(1, -1)),
        torch.addcmul(x, delta, attn.x_v.view(1, -1)),
        torch.addcmul(x, delta, attn.x_a.view(1, -1)),
        torch.addcmul(x, delta, attn.x_g.view(1, -1)),
    )


def prototype_shift_mix(attn, x: torch.Tensor, prev: torch.Tensor, block_size: int):
    return fused_attn_shift_mix(
        x,
        prev,
        attn.x_r,
        attn.x_w,
        attn.x_k,
        attn.x_v,
        attn.x_a,
        attn.x_g,
        block_size=block_size,
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
    diffs = []
    for x, y in zip(a, b, strict=False):
        diffs.append(float((x.float() - y.float()).abs().max().detach().cpu()))
    return max(diffs) if diffs else 0.0


def cos_min_tuple(a, b) -> float:
    vals = []
    for x, y in zip(a, b, strict=False):
        xf = x.float().reshape(x.shape[0], -1)
        yf = y.float().reshape(y.shape[0], -1)
        vals.append(float(torch.nn.functional.cosine_similarity(xf, yf, dim=-1).min().detach().cpu()))
    return min(vals) if vals else 1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--dtype", default="fp16", choices=sorted(DTYPES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-mode", default="fused_recurrent", choices=["chunk", "fused_recurrent"])
    ap.add_argument("--fuse-norm", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--input-rank", type=int, choices=[2, 3], default=2)
    ap.add_argument("--layers", nargs="+", type=int, default=[0, 1, 11])
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--warmup", type=int, default=32)
    ap.add_argument("--steps", type=int, default=512)
    ap.add_argument("--results", default=str(Path(__file__).parent / "results.jsonl"))
    args = ap.parse_args()

    dtype = DTYPES[args.dtype]
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = load_model(args, dtype)
    hidden_size = int(model.config.hidden_size)
    x, prev = make_inputs(hidden_size, args.batch_size, args.device, dtype, args.input_rank)

    layer_rows = []
    for layer_idx in args.layers:
        attn = model.model.layers[layer_idx].attn
        with torch.inference_mode():
            cur = current_shift_mix(attn, x, prev)
            proto = prototype_shift_mix(attn, x, prev, args.block_size)
        current_ms = timed(lambda: current_shift_mix(attn, x, prev), args.device, args.warmup, args.steps)
        prototype_ms = timed(lambda: prototype_shift_mix(attn, x, prev, args.block_size), args.device, args.warmup, args.steps)
        layer_rows.append(
            {
                "layer_idx": layer_idx,
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
        "axis": "fused_shift_mix_proto",
        "backend": "hf_adapter",
        "prototype_backend": "triton_attn_shift_mix" if fused_attn_shift_mix_available() else "torch_fallback",
        "status": "pass",
        "dtype": args.dtype,
        "device": device_name(args.device),
        "attn_mode": args.attn_mode,
        "fuse_norm": getattr(model.config, "fuse_norm", None),
        "batch_size": args.batch_size,
        "input_rank": args.input_rank,
        "hidden_size": hidden_size,
        "layers": args.layers,
        "block_size": args.block_size,
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
