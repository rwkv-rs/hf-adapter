#!/usr/bin/env python3
# coding=utf-8
"""Albatross-inspired small-B R/K/V projection layout sweep.

Albatross gets much of its small B/T decode speed from shape-specialized
linear kernels plus GPU-specific layout tuning.  This benchmark keeps the HF
adapter path unchanged and evaluates the same idea as telemetry:

* ``single``: existing one-launch Triton R/K/V GEMV prototype;
* ``splitk``: new two-launch split-K R/K/V GEMV prototype.

Rows identify the fastest backend/block layout for decode-sized batches so we
can decide what to integrate behind ``native_graph`` without copying the
standalone Albatross engine.
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

from rwkv7_hf.fused_projection import (
    fused_rkv_available,
    fused_rkv_projection,
    fused_rkv_projection_splitk,
    fused_rkv_splitk_available,
)


DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def cuda_sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def device_name(device: str) -> str:
    return torch.cuda.get_device_name(0) if device.startswith("cuda") and torch.cuda.is_available() else device


def peak_mb(device: str) -> float | None:
    if not device.startswith("cuda"):
        return None
    return round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)


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
    return (time.perf_counter() - t0) * 1000.0 / max(1, steps)


def set_attn_mode(model, attn_mode: str) -> None:
    model.config.attn_mode = attn_mode
    for layer in getattr(model.model, "layers", []):
        attn = getattr(layer, "attn", None)
        if hasattr(attn, "mode"):
            attn.mode = attn_mode


def load_model(args, dtype: torch.dtype):
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_dir,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=args.device if args.device.startswith("cuda") else None,
    ).eval()
    set_attn_mode(model, args.attn_mode)
    return model


def resolve_layers(model, requested: list[int]) -> list[int]:
    layers = list(getattr(model.model, "layers", []))
    n = len(layers)
    out = []
    for idx in requested:
        real = n + idx if idx < 0 else idx
        if 0 <= real < n and real not in out:
            out.append(real)
    if not out:
        raise ValueError(f"no valid layers from {requested}; model has {n} layers")
    return out


def make_inputs(hidden_size: int, batch_size: int, device: str, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    gen_device = device if device.startswith("cuda") else "cpu"
    g = torch.Generator(device=gen_device)
    g.manual_seed(1000 + batch_size * 17 + hidden_size)
    base = torch.randn(batch_size, 1, hidden_size, device=gen_device, dtype=dtype, generator=g)
    prev = torch.randn_like(base)
    delta = prev - base
    return {
        "xr": base + 0.01 * delta,
        "xk": base + 0.03 * delta,
        "xv": base + 0.04 * delta,
    }


def current_rkv(attn, xs: dict[str, torch.Tensor]):
    return attn.r_proj(xs["xr"]), attn.k_proj(xs["xk"]), attn.v_proj(xs["xv"])


def prototype_rkv(attn, xs: dict[str, torch.Tensor], backend: str, block_m: int, block_k: int):
    fn = fused_rkv_projection_splitk if backend == "splitk" else fused_rkv_projection
    return fn(
        xs["xr"],
        xs["xk"],
        xs["xv"],
        attn.r_proj.weight,
        attn.k_proj.weight,
        attn.v_proj.weight,
        block_m=block_m,
        block_k=block_k,
    )


def maxdiff_tuple(a, b) -> float:
    diffs = [float((x.float() - y.float()).abs().max().detach().cpu()) for x, y in zip(a, b, strict=False)]
    return max(diffs) if diffs else 0.0


def cos_min_tuple(a, b) -> float:
    vals = []
    for x, y in zip(a, b, strict=False):
        xf = x.float().reshape(x.shape[0], -1)
        yf = y.float().reshape(y.shape[0], -1)
        vals.append(float(torch.nn.functional.cosine_similarity(xf, yf, dim=-1).min().detach().cpu()))
    return min(vals) if vals else 1.0


def append_jsonl(path: str, row: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def bench_batch(model, args, dtype: torch.dtype, batch_size: int) -> dict[str, Any]:
    hidden_size = int(model.config.hidden_size)
    layers = resolve_layers(model, args.layers)
    xs = make_inputs(hidden_size, batch_size, args.device, dtype)

    baseline_rows = []
    for layer_idx in layers:
        attn = model.model.layers[layer_idx].attn
        baseline_rows.append(
            {
                "layer_idx": layer_idx,
                "current_ms": timed(lambda attn=attn: current_rkv(attn, xs), args.device, args.warmup, args.steps),
            }
        )
    avg_current = sum(r["current_ms"] for r in baseline_rows) / len(baseline_rows)

    config_rows = []
    for backend in args.backends:
        if backend == "single" and not fused_rkv_available():
            continue
        if backend == "splitk" and not fused_rkv_splitk_available():
            continue
        for block_m in args.block_ms:
            for block_k in args.block_ks:
                layer_rows = []
                for layer_idx in layers:
                    attn = model.model.layers[layer_idx].attn
                    with torch.inference_mode():
                        cur = current_rkv(attn, xs)
                        proto = prototype_rkv(attn, xs, backend, block_m, block_k)
                    proto_ms = timed(
                        lambda attn=attn, backend=backend, block_m=block_m, block_k=block_k: prototype_rkv(
                            attn, xs, backend, block_m, block_k
                        ),
                        args.device,
                        args.warmup,
                        args.steps,
                    )
                    layer_rows.append(
                        {
                            "layer_idx": layer_idx,
                            "prototype_ms": round(proto_ms, 5),
                            "max_abs_diff": maxdiff_tuple(cur, proto),
                            "min_cosine": cos_min_tuple(cur, proto),
                        }
                    )
                avg_proto = sum(float(r["prototype_ms"]) for r in layer_rows) / len(layer_rows)
                config_rows.append(
                    {
                        "backend": backend,
                        "block_m": int(block_m),
                        "block_k": int(block_k),
                        "avg_prototype_ms": round(avg_proto, 5),
                        "avg_speedup": round(avg_current / avg_proto, 4) if avg_proto else None,
                        "max_abs_diff": max(float(r["max_abs_diff"]) for r in layer_rows),
                        "min_cosine": min(float(r["min_cosine"]) for r in layer_rows),
                        "layer_rows": layer_rows,
                    }
                )

    config_rows.sort(key=lambda r: float(r["avg_prototype_ms"]))
    best = config_rows[0] if config_rows else None
    row: dict[str, Any] = {
        "axis": "albatross_projection_layout_tune",
        "backend": "hf_adapter",
        "prototype_backend": "triton_rkv_small_b_layout_sweep",
        "borrowed_from": "Albatross small-B split-K linear/layout tuning idea",
        "status": "pass" if best is not None else "skip",
        "dtype": args.dtype,
        "device": device_name(args.device),
        "attn_mode": args.attn_mode,
        "fuse_norm": getattr(model.config, "fuse_norm", None),
        "batch_size": int(batch_size),
        "hidden_size": hidden_size,
        "layers": layers,
        "warmup": args.warmup,
        "steps": args.steps,
        "avg_current_ms": round(avg_current, 5),
        "baseline_rows": [{**r, "current_ms": round(float(r["current_ms"]), 5)} for r in baseline_rows],
        "config_count": len(config_rows),
        "best_config": best,
        "top_configs": config_rows[: int(args.keep_top)],
        "peak_vram_mb": peak_mb(args.device),
    }
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--dtype", default="fp16", choices=sorted(DTYPES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-mode", default="fused_recurrent", choices=["chunk", "fused_recurrent"])
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4])
    ap.add_argument("--layers", nargs="+", type=int, default=[0, 1, -1])
    ap.add_argument("--backends", nargs="+", choices=["single", "splitk"], default=["single", "splitk"])
    ap.add_argument("--block-ms", nargs="+", type=int, default=[8, 16, 32, 64])
    ap.add_argument("--block-ks", nargs="+", type=int, default=[64, 128, 256])
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--keep-top", type=int, default=6)
    ap.add_argument("--results", default=str(Path(__file__).parent / "results.jsonl"))
    args = ap.parse_args()

    dtype = DTYPES[args.dtype]
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = load_model(args, dtype)

    rows = []
    for batch_size in args.batch_sizes:
        row = bench_batch(model, args, dtype, batch_size)
        print(json.dumps(row, indent=2, ensure_ascii=False), flush=True)
        append_jsonl(args.results, row)
        rows.append(row)

    print(f"\nappended {len(rows)} row(s) -> {args.results}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
