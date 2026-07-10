#!/usr/bin/env python3
# coding=utf-8
"""KernelBench-Mega-inspired W4A16 R/K/V decode experiment.

The existing RWKV W4 fused R/K/V prototype uses one Triton launch but keeps
three projection tiles and three fp32 accumulators live in every program.  This
benchmark compares it with a projection-axis layout that still uses one launch
while assigning one R/K/V projection to each grid program.  The candidate takes
pre-stacked activations and packed weights, matching the layout a fused native
shift/mix producer could emit without intermediate copies.

Reported timings intentionally include two candidate views:

* ``stacked_prepacked``: the kernel ceiling when the producer already emits
  ``[batch, 3, hidden]`` activations;
* ``stacked_runtime_copy``: the current integration cost when ``torch.stack``
  must be paid immediately before the kernel.

Only the prepacked result can justify a later producer/consumer fusion.  The
runtime-copy row prevents claiming a speedup that the current graph cannot yet
realize end to end.
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
    int4_stacked_rkv_gemv,
    int4_weight_footprint_bytes,
    native_int4_stacked_rkv_available,
    quantize_int4_rowwise,
)

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def cuda_sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def device_name(device: str) -> str:
    return torch.cuda.get_device_name(0) if device.startswith("cuda") else device


def timed(fn: Callable[[], Any], device: str, warmup: int, steps: int) -> float:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
    cuda_sync(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(steps):
            fn()
    cuda_sync(device)
    return (time.perf_counter() - start) * 1000.0 / steps


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
    set_attn_mode(model, args.attn_mode)
    return model


def make_inputs(hidden: int, batch: int, device: str, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    gen_device = device if device.startswith("cuda") else "cpu"
    base = torch.randn(batch, hidden, device=gen_device, dtype=dtype)
    previous = torch.randn_like(base)
    delta = previous - base
    return {
        "xr": (base + 0.01 * delta).contiguous(),
        "xk": (base + 0.03 * delta).contiguous(),
        "xv": (base + 0.04 * delta).contiguous(),
    }


def fp16_rkv(attn, xs: dict[str, torch.Tensor]):
    return (
        F.linear(xs["xr"], attn.r_proj.weight),
        F.linear(xs["xk"], attn.k_proj.weight),
        F.linear(xs["xv"], attn.v_proj.weight),
    )


def pack_rkv(attn) -> dict[str, torch.Tensor]:
    qr, sr = quantize_int4_rowwise(attn.r_proj.weight.detach())
    qk, sk = quantize_int4_rowwise(attn.k_proj.weight.detach())
    qv, sv = quantize_int4_rowwise(attn.v_proj.weight.detach())
    device = attn.r_proj.weight.device
    qr, qk, qv = qr.to(device), qk.to(device), qv.to(device)
    sr, sk, sv = sr.to(device), sk.to(device), sv.to(device)
    return {
        "qr": qr,
        "qk": qk,
        "qv": qv,
        "sr": sr,
        "sk": sk,
        "sv": sv,
        "q_stacked": torch.stack((qr, qk, qv), dim=0).contiguous(),
        "s_stacked": torch.stack((sr, sk, sv), dim=0).contiguous(),
    }


def current_fused(qpack: dict[str, torch.Tensor], xs: dict[str, torch.Tensor], block_m: int, block_k: int):
    return int4_fused_rkv_gemv(
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


def stacked_prepacked(
    qpack: dict[str, torch.Tensor],
    x_stacked: torch.Tensor,
    block_m: int,
    block_k: int,
    num_warps: int,
):
    return int4_stacked_rkv_gemv(
        x_stacked,
        qpack["q_stacked"],
        qpack["s_stacked"],
        block_m=block_m,
        block_k=block_k,
        num_warps=num_warps,
    )


def stacked_runtime_copy(
    qpack: dict[str, torch.Tensor],
    xs: dict[str, torch.Tensor],
    block_m: int,
    block_k: int,
    num_warps: int,
):
    x_stacked = torch.stack((xs["xr"], xs["xk"], xs["xv"]), dim=1)
    return stacked_prepacked(qpack, x_stacked, block_m, block_k, num_warps)


def tensor_to_tuple(value: torch.Tensor):
    return value[:, 0, :], value[:, 1, :], value[:, 2, :]


def maxdiff_tuple(a, b) -> float:
    return max(float((x.float() - y.float()).abs().max().detach().cpu()) for x, y in zip(a, b, strict=True))


def cosine_min_tuple(a, b) -> float:
    values = []
    for x, y in zip(a, b, strict=True):
        values.append(
            float(
                F.cosine_similarity(x.float().reshape(x.shape[0], -1), y.float().reshape(y.shape[0], -1), dim=-1)
                .min()
                .detach()
                .cpu()
            )
        )
    return min(values)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", required=True)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-mode", choices=["chunk", "fused_recurrent"], default="fused_recurrent")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--layers", nargs="+", type=int, default=[0, 1, 11])
    parser.add_argument("--block-m", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--block-k", nargs="+", type=int, default=[32, 64, 128, 256])
    parser.add_argument("--num-warps", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--results", default=str(Path(__file__).parent / "results.jsonl"))
    args = parser.parse_args()

    dtype = DTYPES[args.dtype]
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    model = load_model(args, dtype)
    hidden = int(model.config.hidden_size)
    xs = make_inputs(hidden, args.batch_size, args.device, dtype)
    x_stacked = torch.stack((xs["xr"], xs["xk"], xs["xv"]), dim=1).contiguous()

    layer_refs = []
    for layer_idx in args.layers:
        attn = model.model.layers[layer_idx].attn
        qpack = pack_rkv(attn)
        fp16_out = fp16_rkv(attn, xs)
        fp16_ms = timed(lambda a=attn: fp16_rkv(a, xs), args.device, args.warmup, args.steps)
        dense_bytes = int(
            attn.r_proj.weight.numel() + attn.k_proj.weight.numel() + attn.v_proj.weight.numel()
        ) * int(attn.r_proj.weight.element_size())
        packed_bytes = sum(
            int4_weight_footprint_bytes(qpack[f"q{name}"], qpack[f"s{name}"]) for name in ("r", "k", "v")
        )
        layer_refs.append(
            {
                "layer_idx": layer_idx,
                "attn": attn,
                "qpack": qpack,
                "fp16_out": fp16_out,
                "fp16_ms": fp16_ms,
                "dense_bytes": dense_bytes,
                "packed_bytes": packed_bytes,
            }
        )

    current_configs = []
    for block_m in args.block_m:
        for block_k in args.block_k:
            latencies = []
            cosine = []
            for ref in layer_refs:
                candidate = current_fused(ref["qpack"], xs, block_m, block_k)
                latencies.append(
                    timed(
                        lambda r=ref: current_fused(r["qpack"], xs, block_m, block_k),
                        args.device,
                        args.warmup,
                        args.steps,
                    )
                )
                cosine.append(cosine_min_tuple(ref["fp16_out"], candidate))
            current_configs.append(
                {
                    "block_m": block_m,
                    "block_k": block_k,
                    "num_warps": 4,
                    "avg_ms": round(mean(latencies), 6),
                    "min_cosine_vs_fp16": min(cosine),
                }
            )

    stacked_configs = []
    for block_m in args.block_m:
        for block_k in args.block_k:
            for num_warps in args.num_warps:
                latencies = []
                maxdiff_current = []
                cosine = []
                for ref in layer_refs:
                    candidate_tensor = stacked_prepacked(ref["qpack"], x_stacked, block_m, block_k, num_warps)
                    candidate = tensor_to_tuple(candidate_tensor)
                    current = current_fused(ref["qpack"], xs, block_m, block_k)
                    latencies.append(
                        timed(
                            lambda r=ref: stacked_prepacked(r["qpack"], x_stacked, block_m, block_k, num_warps),
                            args.device,
                            args.warmup,
                            args.steps,
                        )
                    )
                    maxdiff_current.append(maxdiff_tuple(current, candidate))
                    cosine.append(cosine_min_tuple(ref["fp16_out"], candidate))
                stacked_configs.append(
                    {
                        "block_m": block_m,
                        "block_k": block_k,
                        "num_warps": num_warps,
                        "avg_ms": round(mean(latencies), 6),
                        "max_abs_diff_vs_current_w4": max(maxdiff_current),
                        "min_cosine_vs_fp16": min(cosine),
                    }
                )

    best_current = min(current_configs, key=lambda item: float(item["avg_ms"]))
    best_stacked = min(stacked_configs, key=lambda item: float(item["avg_ms"]))
    runtime_copy_latencies = []
    for ref in layer_refs:
        runtime_copy_latencies.append(
            timed(
                lambda r=ref: stacked_runtime_copy(
                    r["qpack"],
                    xs,
                    int(best_stacked["block_m"]),
                    int(best_stacked["block_k"]),
                    int(best_stacked["num_warps"]),
                ),
                args.device,
                args.warmup,
                args.steps,
            )
        )
    runtime_copy_ms = mean(runtime_copy_latencies)
    fp16_ms = mean([float(ref["fp16_ms"]) for ref in layer_refs])
    current_ms = float(best_current["avg_ms"])
    stacked_ms = float(best_stacked["avg_ms"])
    dense_bytes = sum(int(ref["dense_bytes"]) for ref in layer_refs)
    packed_bytes = sum(int(ref["packed_bytes"]) for ref in layer_refs)

    row = {
        "axis": "kernelbench_mega_w4a16_rkv",
        "backend": "hf_adapter",
        "status": "pass" if float(best_stacked["max_abs_diff_vs_current_w4"]) == 0.0 else "fail",
        "candidate_backend": "triton_int4_projection_axis_rkv" if native_int4_stacked_rkv_available() else "torch_fallback",
        "device": device_name(args.device),
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "hidden_size": hidden,
        "layers": args.layers,
        "warmup": args.warmup,
        "steps": args.steps,
        "avg_fp16_ms": round(fp16_ms, 6),
        "best_current_fused": best_current,
        "best_stacked_prepacked": best_stacked,
        "stacked_prepacked_speedup_vs_current": round(current_ms / stacked_ms, 4),
        "stacked_prepacked_speedup_vs_fp16": round(fp16_ms / stacked_ms, 4),
        "stacked_runtime_copy_ms": round(runtime_copy_ms, 6),
        "stacked_runtime_copy_speedup_vs_current": round(current_ms / runtime_copy_ms, 4),
        "stacked_runtime_copy_speedup_vs_fp16": round(fp16_ms / runtime_copy_ms, 4),
        "packed_footprint_ratio_vs_fp16": round(float(packed_bytes) / float(dense_bytes), 4),
        "kernel_launches": {
            "current_fused": 1,
            "stacked_prepacked": 1,
            "stacked_runtime_copy": 2,
        },
        "current_configs": current_configs,
        "stacked_configs": stacked_configs,
        "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
        if args.device.startswith("cuda")
        else None,
    }
    print(json.dumps(row, indent=2, ensure_ascii=False), flush=True)
    if args.results:
        output = Path(args.results)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nappended 1 row -> {output}", flush=True)
    return 0 if row["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
