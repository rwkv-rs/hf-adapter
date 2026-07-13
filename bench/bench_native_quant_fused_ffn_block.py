#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from rwkv7_hf.native_quant_mm8 import MM8Linear


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def timed_ms(fn, *, warmup: int, runs: int, device: str) -> float:
    for _ in range(warmup):
        fn()
    sync(device)
    samples = []
    for _ in range(runs):
        started = time.perf_counter()
        fn()
        sync(device)
        samples.append((time.perf_counter() - started) * 1000.0)
    return float(statistics.median(samples))


def append_row(path: Path | None, row: dict) -> None:
    line = json.dumps(row, sort_keys=True)
    print(f"NATIVE_QUANT_FUSED_FFN_BLOCK_RESULT {line}", flush=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def benchmark_case(args, batch_size: int) -> dict:
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    torch.manual_seed(args.seed)
    key_dense = torch.nn.Linear(
        args.hidden_size, args.intermediate_size, bias=False,
        device=args.device, dtype=dtype,
    ).eval()
    value_dense = torch.nn.Linear(
        args.intermediate_size, args.hidden_size, bias=False,
        device=args.device, dtype=dtype,
    ).eval()
    key = MM8Linear(key_dense, fused=True).eval()
    value = MM8Linear(value_dense, fused=True).eval()
    del key_dense, value_dense

    x = torch.randn(batch_size, args.hidden_size, device=args.device, dtype=dtype) * 0.05
    residual = torch.randn_like(x) * 0.05

    def separate():
        return residual + value(torch.relu(key(x)) ** 2)

    def up_fused():
        return residual + value(key.rwkv7_forward_relu2(x))

    def deep_fused():
        return value.rwkv7_forward_add(key.rwkv7_forward_relu2(x), residual)

    with torch.inference_mode():
        reference = separate()
        up_only = up_fused()
        deep = deep_fused()
    sync(args.device)
    reference_f = reference.float().flatten().unsqueeze(0)
    deep_f = deep.float().flatten().unsqueeze(0)
    cosine = float(F.cosine_similarity(reference_f, deep_f).item())
    max_abs = float((reference_f - deep_f).abs().max().item())
    up_max_abs = float((up_only.float() - deep.float()).abs().max().item())

    with torch.inference_mode():
        separate_ms = timed_ms(separate, warmup=args.warmup, runs=args.runs, device=args.device)
        up_fused_ms = timed_ms(up_fused, warmup=args.warmup, runs=args.runs, device=args.device)
        deep_fused_ms = timed_ms(deep_fused, warmup=args.warmup, runs=args.runs, device=args.device)

    return {
        "axis": "native_quant_fused_ffn_block",
        "status": "pass" if cosine >= args.min_cosine else "fail",
        "quantization": "mm8",
        "device": torch.cuda.get_device_name(torch.device(args.device)),
        "capability": list(torch.cuda.get_device_capability(torch.device(args.device))),
        "torch_version": torch.__version__,
        "dtype": args.dtype,
        "batch_size": batch_size,
        "hidden_size": args.hidden_size,
        "intermediate_size": args.intermediate_size,
        "separate_ms": round(separate_ms, 6),
        "up_fused_ms": round(up_fused_ms, 6),
        "deep_fused_ms": round(deep_fused_ms, 6),
        "speedup_vs_separate": round(separate_ms / deep_fused_ms, 6),
        "speedup_vs_up_fused": round(up_fused_ms / deep_fused_ms, 6),
        "min_cosine": cosine,
        "max_abs_diff": max_abs,
        "up_vs_deep_max_abs_diff": up_max_abs,
        "warmup": args.warmup,
        "runs": args.runs,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--hidden-size", type=int, default=2048)
    ap.add_argument("--intermediate-size", type=int, default=8192)
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=713)
    ap.add_argument("--min-cosine", type=float, default=0.999)
    ap.add_argument("--results", type=Path)
    args = ap.parse_args()

    failures = 0
    for batch_size in args.batch_sizes:
        row = benchmark_case(args, batch_size)
        append_row(args.results, row)
        failures += row["status"] != "pass"
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
