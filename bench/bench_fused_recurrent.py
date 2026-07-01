#!/usr/bin/env python3
# coding=utf-8
"""Benchmark optional fused RWKV-7 recurrent state-update prototype.

Projection and shift-mix standalone prototypes are slower on V100 because they
are too small.  The recurrent update is a better candidate for deeper fusion:
it owns the state write, rank-1 state transition, and readout.  This benchmark
compares the current torch expression against a Triton rank-1 update prototype.
Rows are telemetry only until the prototype is fast and integrated behind the
HF native_graph fast-token backend.
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

from rwkv7_hf.fused_recurrent_update import fused_recurrent_update, fused_recurrent_update_available, torch_recurrent_update

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


def make_inputs(batch_size: int, heads: int, head_dim: int, device: str, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    gen_device = device if device.startswith("cuda") else "cpu"
    shape = (batch_size, heads, head_dim)
    r = torch.randn(shape, device=gen_device, dtype=dtype)
    k = torch.randn(shape, device=gen_device, dtype=dtype)
    v = torch.randn(shape, device=gen_device, dtype=dtype)
    kk = F.normalize(torch.randn(shape, device=gen_device, dtype=dtype).float(), dim=-1, p=2.0).to(dtype)
    a = torch.sigmoid(torch.randn(shape, device=gen_device, dtype=dtype))
    # Match native_jit post-transform w: a positive decay vector in fp32 math.
    w = torch.exp(-0.606531 * torch.sigmoid(torch.randn(shape, device=gen_device, dtype=dtype).float())).to(dtype)
    state = torch.randn(batch_size, heads, head_dim, head_dim, device=gen_device, dtype=torch.float32) * 0.05
    return {"r": r, "w": w, "k": k, "v": v, "kk": kk, "a": a, "state": state}


def current_recurrent(xs: dict[str, torch.Tensor]):
    return torch_recurrent_update(xs["r"], xs["w"], xs["k"], xs["v"], xs["kk"], xs["a"], xs["state"])


def prototype_recurrent(xs: dict[str, torch.Tensor], block_n: int):
    return fused_recurrent_update(xs["r"], xs["w"], xs["k"], xs["v"], xs["kk"], xs["a"], xs["state"], block_n=block_n)


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


def maxdiff_pair(a, b) -> tuple[float, float]:
    out_a, state_a = a
    out_b, state_b = b
    out_diff = float((out_a.float() - out_b.float()).abs().max().detach().cpu())
    state_diff = float((state_a.float() - state_b.float()).abs().max().detach().cpu())
    return out_diff, state_diff


def cosine_min(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.float().reshape(a.shape[0], -1)
    bf = b.float().reshape(b.shape[0], -1)
    return float(torch.nn.functional.cosine_similarity(af, bf, dim=-1).min().detach().cpu())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--dtype", default="fp16", choices=sorted(DTYPES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-mode", default="fused_recurrent", choices=["chunk", "fused_recurrent"])
    ap.add_argument("--fuse-norm", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--layers", nargs="+", type=int, default=[0, 1, 11])
    ap.add_argument("--block-n", type=int, default=64)
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
        heads = int(attn.num_heads)
        head_dim = int(attn.head_dim)
        xs = make_inputs(args.batch_size, heads, head_dim, args.device, dtype)
        with torch.inference_mode():
            cur = current_recurrent(xs)
            proto = prototype_recurrent(xs, args.block_n)
        current_ms = timed(lambda: current_recurrent(xs), args.device, args.warmup, args.steps)
        prototype_ms = timed(lambda: prototype_recurrent(xs, args.block_n), args.device, args.warmup, args.steps)
        out_diff, state_diff = maxdiff_pair(cur, proto)
        layer_rows.append(
            {
                "layer_idx": layer_idx,
                "num_heads": heads,
                "head_dim": head_dim,
                "current_ms": round(current_ms, 5),
                "prototype_ms": round(prototype_ms, 5),
                "speedup": round(current_ms / prototype_ms, 4) if prototype_ms else None,
                "out_max_abs_diff": out_diff,
                "state_max_abs_diff": state_diff,
                "out_min_cosine": cosine_min(cur[0], proto[0]),
            }
        )

    avg_current = sum(float(r["current_ms"]) for r in layer_rows) / len(layer_rows)
    avg_prototype = sum(float(r["prototype_ms"]) for r in layer_rows) / len(layer_rows)
    row = {
        "axis": "fused_recurrent_proto",
        "backend": "hf_adapter",
        "prototype_backend": "triton_rank1_recurrent" if fused_recurrent_update_available() else "torch_fallback",
        "status": "pass",
        "dtype": args.dtype,
        "device": device_name(args.device),
        "attn_mode": args.attn_mode,
        "fuse_norm": getattr(model.config, "fuse_norm", None),
        "batch_size": args.batch_size,
        "hidden_size": int(model.config.hidden_size),
        "layers": args.layers,
        "block_n": args.block_n,
        "steps": args.steps,
        "avg_current_ms": round(avg_current, 5),
        "avg_prototype_ms": round(avg_prototype, 5),
        "avg_speedup": round(avg_current / avg_prototype, 4) if avg_prototype else None,
        "out_max_abs_diff": max(float(r["out_max_abs_diff"]) for r in layer_rows),
        "state_max_abs_diff": max(float(r["state_max_abs_diff"]) for r in layer_rows),
        "out_min_cosine": min(float(r["out_min_cosine"]) for r in layer_rows),
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
