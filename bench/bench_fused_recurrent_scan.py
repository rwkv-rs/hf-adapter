#!/usr/bin/env python3
# coding=utf-8
"""Benchmark native RWKV-7 recurrent scan prototype for prefill.

This isolates the recurrent scan part of prefill after R/K/V/W/A projection.
The production HF prefill path still uses FLA's full layer implementation; this
prototype gives us a concrete native fused-scan target to compare against
``fla.ops.rwkv7.chunk_rwkv7`` before wiring it into the full HF wrapper.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from rwkv7_hf.fused_recurrent_update import (
    fused_recurrent_scan,
    fused_recurrent_scan_available,
    torch_recurrent_scan,
)

try:  # pragma: no cover - only available in the benchmark environment
    from fla.ops.rwkv7 import chunk_rwkv7
except Exception:  # pragma: no cover
    chunk_rwkv7 = None


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


def make_inputs(
    batch_size: int,
    tokens: int,
    heads: int,
    head_dim: int,
    device: str,
    dtype: torch.dtype,
    seed: int,
) -> dict[str, torch.Tensor]:
    gen_device = device if device.startswith("cuda") else "cpu"
    g = torch.Generator(device=gen_device)
    g.manual_seed(seed)
    shape = (batch_size, tokens, heads, head_dim)
    r = torch.randn(shape, device=gen_device, dtype=dtype, generator=g)
    k = torch.randn(shape, device=gen_device, dtype=dtype, generator=g) * 0.25
    v = torch.randn(shape, device=gen_device, dtype=dtype, generator=g) * 0.25
    kk = F.normalize(torch.randn(shape, device=gen_device, dtype=dtype, generator=g).float(), dim=-1, p=2.0).to(dtype)
    a = torch.sigmoid(torch.randn(shape, device=gen_device, dtype=dtype, generator=g))
    # FLA chunk_rwkv7 expects the log-decay gate used by the layer forward,
    # whereas native_jit/recurrent_update consume the already exponentiated
    # positive decay vector.
    w_log = (-0.6065306597126334 * torch.sigmoid(torch.randn(shape, device=gen_device, dtype=dtype, generator=g).float())).to(dtype)
    w_decay = torch.exp(w_log.float()).to(dtype)
    state = torch.randn(batch_size, heads, head_dim, head_dim, device=gen_device, dtype=torch.float32, generator=g) * 0.01
    return {"r": r, "w_log": w_log, "w_decay": w_decay, "k": k, "v": v, "kk": kk, "a": a, "state": state}


def native_scan(xs: dict[str, torch.Tensor], block_n: int, block_m: int | None = None, num_warps: int | None = None):
    return fused_recurrent_scan(
        xs["r"],
        xs["w_decay"],
        xs["k"],
        xs["v"],
        xs["kk"],
        xs["a"],
        xs["state"],
        block_n=block_n,
        block_m=block_m,
        num_warps=num_warps,
    )


def torch_scan(xs: dict[str, torch.Tensor]):
    return torch_recurrent_scan(xs["r"], xs["w_decay"], xs["k"], xs["v"], xs["kk"], xs["a"], xs["state"])


def fla_scan(xs: dict[str, torch.Tensor], chunk_size: int):
    if chunk_rwkv7 is None:
        raise RuntimeError("FLA chunk_rwkv7 is unavailable")
    return chunk_rwkv7(
        r=xs["r"],
        w=xs["w_log"],
        k=xs["k"],
        v=xs["v"],
        a=-xs["kk"],
        b=xs["kk"] * xs["a"],
        scale=1.0,
        initial_state=xs["state"],
        output_final_state=True,
        safe_gate=True,
        chunk_size=chunk_size,
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
    return (time.perf_counter() - t0) * 1000.0 / max(1, steps)


def pair_diff(a, b) -> dict[str, float]:
    out_a, state_a = a
    out_b, state_b = b
    return {
        "out_max_abs_diff": float((out_a.float() - out_b.float()).abs().max().detach().cpu()),
        "state_max_abs_diff": float((state_a.float() - state_b.float()).abs().max().detach().cpu()),
        "out_min_cosine": float(
            F.cosine_similarity(out_a.float().reshape(out_a.shape[0], -1), out_b.float().reshape(out_b.shape[0], -1), dim=-1)
            .min()
            .detach()
            .cpu()
        ),
    }


def append_jsonl(path: str, row: dict[str, Any]) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="fp16", choices=sorted(DTYPES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4])
    ap.add_argument("--tokens", nargs="+", type=int, default=[128, 512])
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--block-n", type=int, default=64)
    ap.add_argument("--block-m", type=int, default=64)
    ap.add_argument("--num-warps", type=int, default=0, choices=(0, 1, 2, 4, 8), help="Triton num_warps for native scan; 0 uses kernel default")
    ap.add_argument("--chunk-size", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--torch-reference-max-tokens", type=int, default=128)
    ap.add_argument("--results", default=str(Path(__file__).parent / "results.jsonl"))
    args = ap.parse_args()

    dtype = DTYPES[args.dtype]
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    rows = []
    num_warps = int(args.num_warps) if int(args.num_warps) > 0 else None
    for batch_size in args.batch_sizes:
        for tokens in args.tokens:
            xs = make_inputs(batch_size, tokens, args.heads, args.head_dim, args.device, dtype, args.seed + batch_size * 1000 + tokens)
            with torch.inference_mode():
                native_out = native_scan(xs, args.block_n, args.block_m, num_warps)
                fla_out = fla_scan(xs, args.chunk_size) if chunk_rwkv7 is not None else None
                torch_out = torch_scan(xs) if int(tokens) <= int(args.torch_reference_max_tokens) else None

            native_ms = timed(lambda: native_scan(xs, args.block_n, args.block_m, num_warps), args.device, args.warmup, args.steps)
            fla_ms = timed(lambda: fla_scan(xs, args.chunk_size), args.device, args.warmup, args.steps) if chunk_rwkv7 is not None else None
            torch_ms = (
                timed(lambda: torch_scan(xs), args.device, max(1, args.warmup // 2), max(1, args.steps // 4))
                if torch_out is not None
                else None
            )

            row: dict[str, Any] = {
                "axis": "fused_recurrent_scan_proto",
                "backend": "hf_adapter",
                "prototype_backend": "triton_rank1_recurrent_scan" if fused_recurrent_scan_available() else "torch_fallback",
                "status": "pass",
                "dtype": args.dtype,
                "device": device_name(args.device),
                "batch_size": batch_size,
                "tokens": tokens,
                "heads": args.heads,
                "head_dim": args.head_dim,
                "block_n": args.block_n,
                "block_m": args.block_m,
                "num_warps": num_warps,
                "chunk_size": args.chunk_size,
                "warmup": args.warmup,
                "steps": args.steps,
                "native_scan_ms": round(native_ms, 5),
                "native_scan_tokps_total": round((batch_size * tokens) / (native_ms / 1000.0), 1) if native_ms else None,
                "fla_chunk_ms": round(fla_ms, 5) if fla_ms is not None else None,
                "fla_chunk_tokps_total": round((batch_size * tokens) / (fla_ms / 1000.0), 1) if fla_ms else None,
                "native_vs_fla_speedup": round(fla_ms / native_ms, 4) if fla_ms and native_ms else None,
                "torch_reference_ms": round(torch_ms, 5) if torch_ms is not None else None,
                "native_vs_torch_speedup": round(torch_ms / native_ms, 4) if torch_ms and native_ms else None,
                "peak_vram_mb": peak_mb(args.device),
            }
            if fla_out is not None:
                row.update({f"native_vs_fla_{k}": v for k, v in pair_diff(native_out, fla_out).items()})
            if torch_out is not None:
                row.update({f"native_vs_torch_{k}": v for k, v in pair_diff(native_out, torch_out).items()})
            print(json.dumps(row, indent=2, ensure_ascii=False), flush=True)
            append_jsonl(args.results, row)
            rows.append(row)
    print(f"\nappended {len(rows)} row(s) -> {args.results}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
