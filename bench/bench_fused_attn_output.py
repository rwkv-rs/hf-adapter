#!/usr/bin/env python3
# coding=utf-8
"""Benchmark optional fused attention output-prepare prototype."""
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

from rwkv7_hf.fused_output import fused_attn_output_prepare, fused_attn_output_prepare_available

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
    return attn.o_proj(prep), prep


def prototype_attn_output(attn, xs: dict[str, torch.Tensor]):
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
    return attn.o_proj(prep), prep


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
    head_dims = []
    head_v_dims = []
    for layer_idx in args.layers:
        attn = model.model.layers[layer_idx].attn
        xs = make_inputs(attn, args.batch_size, args.device, dtype, args.input_scale)
        head_dims.append(int(attn.head_dim))
        head_v_dims.append(int(attn.head_v_dim))
        with torch.inference_mode():
            cur = current_attn_output(attn, xs)
            proto = prototype_attn_output(attn, xs)
        current_ms = timed(lambda: current_attn_output(attn, xs), args.device, args.warmup, args.steps)
        prototype_ms = timed(lambda: prototype_attn_output(attn, xs), args.device, args.warmup, args.steps)
        output_max_abs_diff = float((cur[0].float() - proto[0].float()).abs().max().detach().cpu())
        prep_max_abs_diff = float((cur[1].float() - proto[1].float()).abs().max().detach().cpu())
        output_min_cosine = float(torch.nn.functional.cosine_similarity(cur[0].float().reshape(cur[0].shape[0], -1), proto[0].float().reshape(proto[0].shape[0], -1), dim=-1).min().detach().cpu())
        prep_min_cosine = float(torch.nn.functional.cosine_similarity(cur[1].float().reshape(cur[1].shape[0], -1), proto[1].float().reshape(proto[1].shape[0], -1), dim=-1).min().detach().cpu())
        layer_rows.append(
            {
                "layer_idx": layer_idx,
                "num_heads": int(attn.num_heads),
                "head_dim": int(attn.head_dim),
                "head_v_dim": int(attn.head_v_dim),
                "current_ms": round(current_ms, 5),
                "prototype_ms": round(prototype_ms, 5),
                "speedup": round(current_ms / prototype_ms, 4) if prototype_ms else None,
                "max_abs_diff": maxdiff_tuple(cur, proto),
                "output_max_abs_diff": output_max_abs_diff,
                "prep_max_abs_diff": prep_max_abs_diff,
                "min_cosine": cos_min_tuple(cur, proto),
                "output_min_cosine": output_min_cosine,
                "prep_min_cosine": prep_min_cosine,
            }
        )

    avg_current = sum(float(r["current_ms"]) for r in layer_rows) / len(layer_rows)
    avg_prototype = sum(float(r["prototype_ms"]) for r in layer_rows) / len(layer_rows)
    row = {
        "axis": "fused_attn_output_proto",
        "backend": "hf_adapter",
        "prototype_backend": "triton_attn_output_prepare_plus_cublas_o" if fused_attn_output_prepare_available() else "torch_fallback",
        "status": "pass",
        "dtype": args.dtype,
        "device": device_name(args.device),
        "attn_mode": args.attn_mode,
        "fuse_norm": getattr(model.config, "fuse_norm", None),
        "batch_size": args.batch_size,
        "hidden_size": int(model.config.hidden_size),
        "head_dims": sorted(set(head_dims)),
        "head_v_dims": sorted(set(head_v_dims)),
        "layers": args.layers,
        "input_scale": args.input_scale,
        "steps": args.steps,
        "avg_current_ms": round(avg_current, 5),
        "avg_prototype_ms": round(avg_prototype, 5),
        "avg_speedup": round(avg_current / avg_prototype, 4) if avg_prototype else None,
        "max_abs_diff": max(float(r["max_abs_diff"]) for r in layer_rows),
        "output_max_abs_diff": max(float(r["output_max_abs_diff"]) for r in layer_rows),
        "prep_max_abs_diff": max(float(r["prep_max_abs_diff"]) for r in layer_rows),
        "min_cosine": min(float(r["min_cosine"]) for r in layer_rows),
        "output_min_cosine": min(float(r["output_min_cosine"]) for r in layer_rows),
        "prep_min_cosine": min(float(r["prep_min_cosine"]) for r in layer_rows),
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
