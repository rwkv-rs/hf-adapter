#!/usr/bin/env python3
"""Sweep TorchAO W4A16 over RWKV projection roles on one CUDA card.

The result separates projection-level evidence from the required end-to-end
``bench_native_quant_e2e_decode.py`` acceptance row.  It counts packed tensor
payloads rather than the logical dense shape exposed by TorchAO wrappers.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from rwkv7_hf.native_quant_torchao import _torchao_api


DEFAULT_SHAPES = (
    "1p5_square:2048x2048",
    "1p5_up:2048x8192",
    "1p5_down:8192x2048",
    "1p5_head:2048x65536",
    "7p2_square:4096x4096",
    "7p2_up:4096x16384",
    "7p2_down:16384x4096",
    "7p2_head:4096x65536",
)


def parse_shape(value: str) -> tuple[str, int, int]:
    label, separator, dims = value.partition(":")
    if not separator:
        label = value
        dims = value
    k_text, x, n_text = dims.lower().partition("x")
    if not x:
        raise argparse.ArgumentTypeError(f"shape must be [label:]KxN, got {value!r}")
    try:
        k, n = int(k_text), int(n_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid shape {value!r}") from exc
    if k <= 0 or n <= 0:
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return label, k, n


def tensor_payload_bytes(tensor, seen: set[int] | None = None) -> int:
    if seen is None:
        seen = set()
    ident = id(tensor)
    if ident in seen:
        return 0
    seen.add(ident)
    flatten = getattr(tensor, "__tensor_flatten__", None)
    if callable(flatten) and type(tensor) not in {torch.Tensor, torch.nn.Parameter}:
        try:
            payload = sum(
                tensor_payload_bytes(getattr(tensor, name), seen)
                for name in flatten()[0]
                if isinstance(getattr(tensor, name), torch.Tensor)
            )
            if payload:
                return payload
        except Exception:
            pass
    return int(tensor.numel()) * int(tensor.element_size())


def timed(call, warmup: int, runs: int) -> float:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(runs):
        call()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / runs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--shapes", type=parse_shape, nargs="+", default=list(DEFAULT_SHAPES))
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    shapes = [parse_shape(item) if isinstance(item, str) else item for item in args.shapes]
    if not torch.cuda.is_available():
        parser.error("CUDA is required")

    quantize_, _, int4_weight_only = _torchao_api()
    output = Path(args.output) if args.output else None
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("", encoding="utf-8")

    def emit(row: dict) -> None:
        line = json.dumps(row, ensure_ascii=False)
        print(line, flush=True)
        if output:
            with output.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    for batch in args.batch_sizes:
        for label, inputs, outputs in shapes:
            gc.collect()
            torch.cuda.empty_cache()
            try:
                dense = torch.nn.Linear(
                    inputs, outputs, bias=False, device="cuda", dtype=torch.bfloat16
                )
                quant = torch.nn.Linear(
                    inputs, outputs, bias=False, device="cuda", dtype=torch.bfloat16
                )
                quant.weight.data.copy_(dense.weight.data)
                dense_bytes = tensor_payload_bytes(dense.weight)
                quantize_(quant, int4_weight_only(group_size=args.group_size))
                quant_bytes = tensor_payload_bytes(quant.weight)
                x = torch.randn(batch, inputs, device="cuda", dtype=torch.bfloat16)
                dense_out = dense(x)
                quant_out = quant(x)
                dense_ms = timed(lambda: dense(x), args.warmup, args.runs)
                quant_ms = timed(lambda: quant(x), args.warmup, args.runs)
                emit(
                    {
                        "axis": "torchao_w4_role_sweep",
                        "status": "pass",
                        "device": torch.cuda.get_device_name(),
                        "compute_capability": list(torch.cuda.get_device_capability()),
                        "torch_version": torch.__version__,
                        "cuda_version": torch.version.cuda,
                        "label": label,
                        "batch_size": batch,
                        "inputs": inputs,
                        "outputs": outputs,
                        "group_size": args.group_size,
                        "dense_ms": round(dense_ms, 6),
                        "w4_ms": round(quant_ms, 6),
                        "speedup_vs_bf16": round(dense_ms / quant_ms, 6),
                        "dense_weight_bytes": dense_bytes,
                        "w4_weight_bytes": quant_bytes,
                        "weight_ratio": round(quant_bytes / dense_bytes, 6),
                        "cosine_vs_bf16": round(
                            torch.nn.functional.cosine_similarity(
                                dense_out.float().flatten(),
                                quant_out.float().flatten(),
                                dim=0,
                            ).item(),
                            8,
                        ),
                        "max_abs_vs_bf16": round(
                            (dense_out.float() - quant_out.float()).abs().max().item(),
                            6,
                        ),
                        "weight_type": type(quant.weight).__name__,
                    }
                )
            except Exception as exc:
                emit(
                    {
                        "axis": "torchao_w4_role_sweep",
                        "status": "fail",
                        "label": label,
                        "batch_size": batch,
                        "inputs": inputs,
                        "outputs": outputs,
                        "error": repr(exc),
                    }
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
