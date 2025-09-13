#!/usr/bin/env python3
# coding=utf-8
"""Benchmark fused attention output-prep plus o_proj prototype."""
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

from rwkv7_hf.fused_output import (
    fused_attn_output_prepare,
    fused_attn_output_project,
    fused_attn_output_project_available,
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


def make_inputs(attn, batch_size: int, device: str, dtype: torch.dtype, scale: float) -> dict[str, torch.Tensor]:
    gen_device = device if device.startswith("cuda") else "cpu"
    h = int(attn.num_heads)
    n = int(attn.head_dim)
    vn = int(attn.head_v_dim)
    value_dim = int(attn.value_dim)
    return {
        "recurrent": scale * torch.randn(batch_size, 1, value_dim, device=gen_device, dtype=dtype),
        "r": scale * torch.randn(batch_size, 1, h, n, device=gen_device, dtype=dtype),
        "k": scale * torch.randn(batch_size, 1, h, n, device=gen_device, dtype=dtype),
        "v": scale * torch.randn(batch_size, 1, h, vn, device=gen_device, dtype=dtype),
        "g": scale * torch.randn(batch_size, 1, value_dim, device=gen_device, dtype=dtype),
    }


def current_attn_output(attn, xs: dict[str, torch.Tensor]):
    batch_size = int(xs["recurrent"].shape[0])
    seq_len = int(xs["recurrent"].shape[1])
    recurrent = xs["recurrent"]
    r = xs["r"]
    k = xs["k"]
    v = xs["v"]
    g = xs["g"]
    o = attn.g_norm(recurrent.reshape(batch_size * seq_len, attn.value_dim)).view(batch_size, seq_len, attn.value_dim)
    correction = ((r * k * attn.r_k.view(1, 1, attn.num_heads, attn.head_dim)).sum(-1, keepdim=True) * v).reshape(o.shape)
    prep = (o + correction) * g
    return attn.o_proj(prep)


def prep_plus_cublas_output(attn, xs: dict[str, torch.Tensor]):
    prep = fused_attn_output_prepare(
        xs["recurrent"],
        xs["r"],
        xs["k"],
        xs["v"],
        xs["g"],
        attn.r_k,
        attn.g_norm.weight,
        attn.g_norm.bias,
        num_heads=int(attn.num_heads),
        head_dim=int(attn.head_dim),
        head_v_dim=int(attn.head_v_dim),
        eps=float(attn.g_norm.eps),
    )
    return attn.o_proj(prep)


def fused_project_output(attn, xs: dict[str, torch.Tensor], block_m: int):
    return fused_attn_output_project(
        xs["recurrent"],
        xs["r"],
        xs["k"],
        xs["v"],
        xs["g"],
        attn.r_k,
        attn.g_norm.weight,
        attn.g_norm.bias,
        attn.o_proj.weight,
        getattr(attn.o_proj, "bias", None),
        num_heads=int(attn.num_heads),
        head_dim=int(attn.head_dim),
        head_v_dim=int(attn.head_v_dim),
        eps=float(attn.g_norm.eps),
        block_m=block_m,
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


def min_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.float().reshape(a.shape[0], -1), b.float().reshape(b.shape[0], -1), dim=-1).min().detach().cpu())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--dtype", default="fp16", choices=sorted(DTYPES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-mode", default="fused_recurrent", choices=["chunk", "fused_recurrent"])
    ap.add_argument("--fuse-norm", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--layers", nargs="+", type=int, default=[0, 1, 11])
    ap.add_argument("--block-m", nargs="+", type=int, default=[16, 32, 64])
    ap.add_argument("--input-scale", type=float, default=0.3)
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--results", default=str(Path(__file__).parent / "results.jsonl"))
    args = ap.parse_args()

    dtype = DTYPES[args.dtype]
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = load_model(args, dtype)

    layer_rows = []
    for layer_idx in args.layers:
        attn = model.model.layers[layer_idx].attn
        xs = make_inputs(attn, args.batch_size, args.device, dtype, args.input_scale)
        with torch.inference_mode():
            current = current_attn_output(attn, xs)
            prep_cublas = prep_plus_cublas_output(attn, xs)
        current_ms = timed(lambda: current_attn_output(attn, xs), args.device, args.warmup, args.steps)
        prep_cublas_ms = timed(lambda: prep_plus_cublas_output(attn, xs), args.device, args.warmup, args.steps)
        configs = []
        for block_m in args.block_m:
            with torch.inference_mode():
                fused = fused_project_output(attn, xs, block_m)
            fused_ms = timed(lambda bm=block_m: fused_project_output(attn, xs, bm), args.device, args.warmup, args.steps)
            configs.append(
                {
                    "block_m": int(block_m),
                    "fused_project_ms": round(fused_ms, 5),
                    "speedup_vs_current": round(current_ms / fused_ms, 4) if fused_ms else None,
                    "speedup_vs_prep_cublas": round(prep_cublas_ms / fused_ms, 4) if fused_ms else None,
                    "output_max_abs_diff": float((current.float() - fused.float()).abs().max().detach().cpu()),
                    "output_min_cosine": min_cos(current, fused),
                }
            )
        best = min(configs, key=lambda c: float(c["fused_project_ms"]))
        layer_rows.append(
            {
                "layer_idx": layer_idx,
                "num_heads": int(attn.num_heads),
                "head_dim": int(attn.head_dim),
                "head_v_dim": int(attn.head_v_dim),
                "current_ms": round(current_ms, 5),
                "prep_cublas_ms": round(prep_cublas_ms, 5),
                "prep_cublas_speedup": round(current_ms / prep_cublas_ms, 4) if prep_cublas_ms else None,
                "prep_cublas_max_abs_diff": float((current.float() - prep_cublas.float()).abs().max().detach().cpu()),
                "prep_cublas_min_cosine": min_cos(current, prep_cublas),
                "best_fused_project": best,
                "configs": configs,
            }
        )

    avg_current = sum(float(r["current_ms"]) for r in layer_rows) / len(layer_rows)
    avg_prep_cublas = sum(float(r["prep_cublas_ms"]) for r in layer_rows) / len(layer_rows)
    block_rows = []
    for block_m in args.block_m:
        vals = []
        for r in layer_rows:
            cfg = next(c for c in r["configs"] if int(c["block_m"]) == int(block_m))
            vals.append(float(cfg["fused_project_ms"]))
        avg_fused = sum(vals) / len(vals)
        block_rows.append(
            {
                "block_m": int(block_m),
                "avg_fused_project_ms": round(avg_fused, 5),
                "speedup_vs_current": round(avg_current / avg_fused, 4) if avg_fused else None,
                "speedup_vs_prep_cublas": round(avg_prep_cublas / avg_fused, 4) if avg_fused else None,
            }
        )
    best_block = min(block_rows, key=lambda c: float(c["avg_fused_project_ms"]))
    row = {
        "axis": "fused_attn_output_project_proto",
        "backend": "hf_adapter",
        "prototype_backend": "triton_attn_output_prepare_o_proj" if fused_attn_output_project_available() else "torch_fallback",
        "status": "pass",
        "dtype": args.dtype,
        "device": device_name(args.device),
        "attn_mode": args.attn_mode,
        "fuse_norm": getattr(model.config, "fuse_norm", None),
        "batch_size": args.batch_size,
        "hidden_size": int(model.config.hidden_size),
        "layers": args.layers,
        "block_m_values": [int(v) for v in args.block_m],
        "input_scale": args.input_scale,
        "steps": args.steps,
        "avg_current_ms": round(avg_current, 5),
        "avg_prep_cublas_ms": round(avg_prep_cublas, 5),
        "avg_prep_cublas_speedup": round(avg_current / avg_prep_cublas, 4) if avg_prep_cublas else None,
        "best_fused_project": best_block,
        "max_abs_diff": max(float(c["output_max_abs_diff"]) for r in layer_rows for c in r["configs"]),
        "min_cosine": min(float(c["output_min_cosine"]) for r in layer_rows for c in r["configs"]),
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
