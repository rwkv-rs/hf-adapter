#!/usr/bin/env python3
"""Shape sweep for the optional Volta small-row fp16 linear kernel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from rwkv7_hf.sm70_linear import sm70_linear, sm70_linear_build_error


DEFAULT_SHAPES = (
    (768, 768),
    (3072, 768),
    (768, 3072),
    (1024, 1024),
    (4096, 1024),
    (1024, 4096),
    (2048, 2048),
    (8192, 2048),
    (2048, 8192),
    (4096, 4096),
    (16384, 4096),
    (4096, 16384),
    (65536, 768),
    (65536, 1024),
    (65536, 2048),
    (65536, 4096),
)


def bench(fn, warmup: int, steps: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(steps):
        graph.replay()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / float(steps)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", type=int, default=[1, 2, 4, 8])
    ap.add_argument("--threads", nargs="+", type=int, default=[64, 128, 256])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--results", default="")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    rows = []
    for batch in args.rows:
        for outputs, inputs in DEFAULT_SHAPES:
            x = torch.randn(batch, inputs, device="cuda", dtype=torch.float16)
            weight = torch.randn(outputs, inputs, device="cuda", dtype=torch.float16)
            reference = F.linear(x, weight)
            current_ms = bench(lambda: F.linear(x, weight), args.warmup, args.steps)
            for threads in args.threads:
                candidate = sm70_linear(x, weight, threads=threads)
                torch.cuda.synchronize()
                candidate_ms = bench(
                    lambda threads=threads: sm70_linear(x, weight, threads=threads),
                    args.warmup,
                    args.steps,
                )
                cosine = float(
                    F.cosine_similarity(reference.float(), candidate.float(), dim=-1).min().cpu()
                )
                row = {
                    "axis": "sm70_small_row_linear",
                    "device": torch.cuda.get_device_name(0),
                    "dtype": "fp16",
                    "rows": batch,
                    "outputs": outputs,
                    "inputs": inputs,
                    "threads": threads,
                    "cublas_ms": round(current_ms, 6),
                    "candidate_ms": round(candidate_ms, 6),
                    "speedup": round(current_ms / candidate_ms, 4),
                    "max_abs_diff": float((reference.float() - candidate.float()).abs().max().cpu()),
                    "min_cosine": cosine,
                    "build_error": sm70_linear_build_error(),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
            del x, weight, reference, candidate
    if args.results:
        path = Path(args.results)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
