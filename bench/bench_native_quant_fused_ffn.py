#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from rwkv7_hf.native_quant_mm4 import MM4Linear
from rwkv7_hf.native_quant_mm8 import MM8Linear


DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


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


def payload_mb(module: torch.nn.Module) -> float:
    total = sum(t.numel() * t.element_size() for t in list(module.parameters()) + list(module.buffers()))
    return total / 1024 / 1024


def append_row(path: Path | None, row: dict) -> None:
    line = json.dumps(row, sort_keys=True)
    print(f"NATIVE_QUANT_FUSED_FFN_RESULT {line}", flush=True)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def benchmark_case(args, quantization: str, batch_size: int) -> dict:
    dtype = DTYPES[args.dtype]
    torch.manual_seed(args.seed)
    dense = torch.nn.Linear(
        args.input_size,
        args.output_size,
        bias=False,
        device=args.device,
        dtype=dtype,
    ).eval()
    module_type = MM8Linear if quantization == "mm8" else MM4Linear
    module = module_type(dense, fused=True).eval()
    del dense
    x = torch.randn(batch_size, args.input_size, device=args.device, dtype=dtype) * 0.05

    with torch.inference_mode():
        separate = torch.relu(module(x)) ** 2
        fused = module.rwkv7_forward_relu2(x)
    sync(args.device)
    separate_f = separate.float().flatten()
    fused_f = fused.float().flatten()
    cosine = float(F.cosine_similarity(separate_f.unsqueeze(0), fused_f.unsqueeze(0)).item())
    max_abs = float((separate_f - fused_f).abs().max().item())

    with torch.inference_mode():
        separate_ms = timed_ms(
            lambda: torch.relu(module(x)) ** 2,
            warmup=args.warmup,
            runs=args.runs,
            device=args.device,
        )
        fused_ms = timed_ms(
            lambda: module.rwkv7_forward_relu2(x),
            warmup=args.warmup,
            runs=args.runs,
            device=args.device,
        )

    sm70_rowwise = bool(getattr(module, "sm70_rowwise", False))
    if not args.device.startswith("cuda"):
        backend = "portable_relu2"
    elif sm70_rowwise:
        backend = "sm70_cuda_relu2"
    else:
        backend = f"{quantization}_triton_relu2"
    return {
        "axis": "native_quant_fused_ffn_relu2",
        "status": "pass" if cosine >= args.min_cosine else "fail",
        "quantization": quantization,
        "device": torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith("cuda") else args.device,
        "capability": list(torch.cuda.get_device_capability(torch.device(args.device))) if args.device.startswith("cuda") else None,
        "torch_version": torch.__version__,
        "dtype": args.dtype,
        "batch_size": batch_size,
        "input_size": args.input_size,
        "output_size": args.output_size,
        "backend": backend,
        "separate_ms": round(separate_ms, 6),
        "fused_ms": round(fused_ms, 6),
        "speedup": round(separate_ms / fused_ms, 6),
        "min_cosine": cosine,
        "max_abs_diff": max_abs,
        "quant_payload_mb": round(payload_mb(module), 3),
        "warmup": args.warmup,
        "runs": args.runs,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="fp16")
    ap.add_argument("--quantizations", nargs="+", choices=["mm8", "mm4"], default=["mm8", "mm4"])
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--input-size", type=int, default=2048)
    ap.add_argument("--output-size", type=int, default=8192)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=712)
    ap.add_argument("--min-cosine", type=float, default=0.999)
    ap.add_argument("--results", type=Path)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if any(batch <= 0 for batch in args.batch_sizes):
        raise ValueError("batch sizes must be positive")
    if args.input_size <= 0 or args.output_size <= 0:
        raise ValueError("input and output sizes must be positive")
    failures = 0
    for quantization in args.quantizations:
        for batch_size in args.batch_sizes:
            row = benchmark_case(args, quantization, batch_size)
            append_row(args.results, row)
            failures += row["status"] != "pass"
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
