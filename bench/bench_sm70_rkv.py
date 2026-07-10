#!/usr/bin/env python3
"""V100 A/B for grouped R/K/V projection without duplicate weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from rwkv7_hf.sm70_linear import sm70_rkv, sm70_rkv_should_use


def bench(fn, warmup: int, steps: int) -> float:
    for _ in range(warmup):
        fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
    begin.record()
    for _ in range(steps):
        graph.replay()
    end.record()
    end.synchronize()
    return float(begin.elapsed_time(end)) / float(steps)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", type=int, default=[1, 2])
    ap.add_argument("--hidden", nargs="+", type=int, default=[768, 1024, 2048, 4096])
    ap.add_argument("--threads", nargs="+", type=int, default=[64, 128, 256])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--results", default="")
    args = ap.parse_args()
    rows = []
    for batch in args.rows:
        for hidden in args.hidden:
            xs = [torch.randn(batch, hidden, device="cuda", dtype=torch.float16) for _ in range(3)]
            ws = [torch.randn(hidden, hidden, device="cuda", dtype=torch.float16) for _ in range(3)]

            def baseline():
                return tuple(F.linear(x, w) for x, w in zip(xs, ws))

            references = baseline()
            baseline_ms = bench(baseline, args.warmup, args.steps)
            for threads in args.threads:
                candidate = sm70_rkv(*xs, *ws, threads=threads)
                candidate_ms = bench(
                    lambda threads=threads: sm70_rkv(*xs, *ws, threads=threads),
                    args.warmup,
                    args.steps,
                )
                row = {
                    "axis": "sm70_grouped_rkv",
                    "device": torch.cuda.get_device_name(0),
                    "dtype": "fp16",
                    "rows": batch,
                    "hidden": hidden,
                    "threads": threads,
                    "route_selected": sm70_rkv_should_use(batch, hidden),
                    "cublas_ms": round(baseline_ms, 6),
                    "candidate_ms": round(candidate_ms, 6),
                    "speedup": round(baseline_ms / candidate_ms, 4),
                    "max_abs_diff": max(
                        float((a.float() - b.float()).abs().max().cpu())
                        for a, b in zip(candidate, references)
                    ),
                    "min_cosine": min(
                        float(F.cosine_similarity(a.float(), b.float(), dim=-1).min().cpu())
                        for a, b in zip(candidate, references)
                    ),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
            del xs, ws, references, candidate
    if args.results:
        path = Path(args.results)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
