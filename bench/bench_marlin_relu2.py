#!/usr/bin/env python3
"""Benchmark the explicit RWKV FFN-key ReLU-square Marlin epilogue."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import torch
import torch.nn.functional as F

from rwkv7_hf.fused_elementwise import fused_relu_square
from rwkv7_hf.native_quant_marlin import MarlinW4Linear


def parse_shape(raw: str) -> tuple[int, int]:
    k, n = (int(value) for value in raw.lower().split("x", 1))
    return k, n


def timed_ms(fn, *, warmup: int, runs: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(runs):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / int(runs)


def paired_samples(first, second, *, warmup: int, runs: int, repeats: int):
    first_samples = []
    second_samples = []
    for repeat in range(repeats):
        if repeat % 2:
            second_samples.append(timed_ms(second, warmup=warmup, runs=runs))
            first_samples.append(timed_ms(first, warmup=warmup, runs=runs))
        else:
            first_samples.append(timed_ms(first, warmup=warmup, runs=runs))
            second_samples.append(timed_ms(second, warmup=warmup, runs=runs))
    return first_samples, second_samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", nargs="+", type=parse_shape, required=True)
    parser.add_argument("--rows", nargs="+", type=int, default=(1, 8, 128, 1024))
    parser.add_argument("--group-size", type=int, choices=(32, 64, 128), default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    torch.manual_seed(20260716)
    records = []
    for k, n in args.shapes:
        linear = torch.nn.Linear(k, n, bias=False, device="cuda", dtype=torch.bfloat16)
        packed = MarlinW4Linear(
            linear,
            group_size=args.group_size,
            fp32_reduce=False,
            production_bn_tn=True,
            fuse_relu2=True,
        )
        del linear
        torch.cuda.empty_cache()
        for rows in args.rows:
            x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
            base = packed(x)
            reference = fused_relu_square(base)
            got = packed.rwkv7_forward_relu2(x)
            torch.cuda.synchronize()
            difference = (got.float() - reference.float()).abs()

            unfused = lambda: fused_relu_square(packed(x))
            fused = lambda: packed.rwkv7_forward_relu2(x)
            unfused_samples, fused_samples = paired_samples(
                unfused,
                fused,
                warmup=args.warmup,
                runs=args.runs,
                repeats=args.repeats,
            )
            unfused_ms = statistics.median(unfused_samples)
            fused_ms = statistics.median(fused_samples)
            plan = packed.effective_bn_tn_plan(rows)
            grid = plan.launches[0].grid
            record = {
                "status": "pass",
                "rows": int(rows),
                "k": int(k),
                "n": int(n),
                "group_size": int(args.group_size),
                "block_n": int(grid.block_n),
                "thread_n": int(grid.thread_n),
                "tile_k": int(grid.tile_k),
                "cuda_threads": int(grid.cuda_threads),
                "stages": int(grid.stages),
                "launch_plan": plan.as_dict(),
                "exact_vs_unfused": bool(torch.equal(got, reference)),
                "cosine_vs_unfused": float(
                    F.cosine_similarity(
                        got.float().reshape(1, -1),
                        reference.float().reshape(1, -1),
                    ).item()
                ),
                "max_abs_vs_unfused": float(difference.max().item()),
                "unfused_ms": round(unfused_ms, 6),
                "fused_ms": round(fused_ms, 6),
                "speedup_vs_unfused": round(unfused_ms / fused_ms, 6),
                "unfused_ms_samples": [round(value, 6) for value in unfused_samples],
                "fused_ms_samples": [round(value, 6) for value in fused_samples],
                "device": torch.cuda.get_device_name(),
                "compute_capability": list(torch.cuda.get_device_capability()),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
            }
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
            del x, base, reference, got, difference
        del packed
        torch.cuda.empty_cache()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
