#!/usr/bin/env python3
"""Sweep production Marlin Tensor Core output/reduction tile schedules."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import torch
import torch.nn.functional as F

from rwkv7_hf.native_quant_marlin import MarlinW4Linear


BASE_SCHEDULES = (
    ("auto", None),
    ("tk128_tn128_t256", (128, 128, 256)),
    ("tk64_tn128_t128", (64, 128, 128)),
    ("tk128_tn64_t128", (128, 64, 128)),
    ("tk64_tn256_t256", (64, 256, 256)),
)


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


def paired_samples(auto_fn, candidate_fn, *, warmup: int, runs: int, repeats: int):
    auto_fn()
    candidate_fn()
    auto_samples = []
    candidate_samples = []
    for repeat in range(repeats):
        if repeat % 2:
            candidate_samples.append(timed_ms(candidate_fn, warmup=warmup, runs=runs))
            auto_samples.append(timed_ms(auto_fn, warmup=warmup, runs=runs))
        else:
            auto_samples.append(timed_ms(auto_fn, warmup=warmup, runs=runs))
            candidate_samples.append(timed_ms(candidate_fn, warmup=warmup, runs=runs))
    return auto_samples, candidate_samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", nargs="+", type=parse_shape, required=True)
    parser.add_argument("--rows", nargs="+", type=int, default=(1, 8, 128, 1024))
    parser.add_argument("--group-size", type=int, choices=(32, 64, 128), default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--sms",
        nargs="+",
        type=int,
        default=(),
        help="also sweep persistent scheduler CTA counts; physical SM count is the default",
    )
    parser.add_argument("--stages", nargs="+", type=int, default=())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    torch.manual_seed(1234)
    schedules = list(BASE_SCHEDULES)
    for sms in args.sms:
        schedules.append((f"auto_sms{sms}", (-1, -1, -1, int(sms))))
        for label, schedule in BASE_SCHEDULES[1:]:
            schedules.append((f"{label}_sms{sms}", (*schedule, int(sms))))
    for stages in args.stages:
        schedules.append((f"auto_stages{stages}", (-1, -1, -1, -1, int(stages))))
        for label, schedule in BASE_SCHEDULES[1:]:
            schedules.append((f"{label}_stages{stages}", (*schedule, -1, int(stages))))
    records = []
    for k, n in args.shapes:
        dense = torch.nn.Linear(k, n, bias=False, device="cuda", dtype=torch.bfloat16)
        packed = MarlinW4Linear(dense, group_size=args.group_size)
        weight = dense.weight.detach()
        for rows in args.rows:
            x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
            dense_out = F.linear(x, weight)
            dense_ms = timed_ms(lambda: F.linear(x, weight), warmup=args.warmup, runs=args.runs)
            auto = packed._apply_marlin(x)
            auto_ms = timed_ms(lambda: packed._apply_marlin(x), warmup=args.warmup, runs=args.runs)
            for label, schedule in schedules:
                try:
                    got = packed._apply_marlin(x, schedule=schedule)
                    auto_samples, candidate_samples = paired_samples(
                        lambda: packed._apply_marlin(x),
                        lambda schedule=schedule: packed._apply_marlin(x, schedule=schedule),
                        warmup=args.warmup,
                        runs=args.runs,
                        repeats=args.repeats,
                    )
                    auto_ms = statistics.median(auto_samples)
                    candidate_ms = statistics.median(candidate_samples)
                    speedup_samples = [
                        auto_value / candidate_value
                        for auto_value, candidate_value in zip(auto_samples, candidate_samples)
                    ]
                    row = {
                        "status": "pass",
                        "schedule": label,
                        "tile_k": schedule[0] if schedule else -1,
                        "block_n": schedule[1] if schedule else -1,
                        "thread_n": 8,
                        "num_threads": schedule[2] if schedule else -1,
                        "sms": schedule[3] if schedule and len(schedule) >= 4 else -1,
                        "stages": schedule[4] if schedule and len(schedule) == 5 else -1,
                        "rows": rows,
                        "group_size": int(args.group_size),
                        "k": k,
                        "n": n,
                        "candidate_ms": round(candidate_ms, 6),
                        "auto_ms": round(auto_ms, 6),
                        "dense_bf16_ms": round(dense_ms, 6),
                        "speedup_vs_auto": round(auto_ms / candidate_ms, 6),
                        "speedup_vs_auto_median_of_pairs": round(statistics.median(speedup_samples), 6),
                        "speedup_vs_dense_bf16": round(dense_ms / candidate_ms, 6),
                        "auto_ms_samples": [round(value, 6) for value in auto_samples],
                        "candidate_ms_samples": [round(value, 6) for value in candidate_samples],
                        "speedup_vs_auto_samples": [round(value, 6) for value in speedup_samples],
                        "timing_repeats": args.repeats,
                        "cosine_vs_auto": round(float(F.cosine_similarity(got.float().reshape(1, -1), auto.float().reshape(1, -1)).item()), 9),
                        "cosine_vs_dense_bf16": round(float(F.cosine_similarity(got.float().reshape(1, -1), dense_out.float().reshape(1, -1)).item()), 9),
                    }
                except RuntimeError as exc:
                    row = {
                        "status": "unsupported",
                        "schedule": label,
                        "tile_k": schedule[0] if schedule else -1,
                        "block_n": schedule[1] if schedule else -1,
                        "thread_n": 8,
                        "num_threads": schedule[2] if schedule else -1,
                        "sms": schedule[3] if schedule and len(schedule) >= 4 else -1,
                        "stages": schedule[4] if schedule and len(schedule) == 5 else -1,
                        "rows": rows,
                        "group_size": int(args.group_size),
                        "k": k,
                        "n": n,
                        "error": str(exc).splitlines()[0],
                    }
                row.update(
                    device=torch.cuda.get_device_name(),
                    compute_capability=list(torch.cuda.get_device_capability()),
                    torch_version=torch.__version__,
                    cuda_version=torch.version.cuda,
                )
                records.append(row)
                print(json.dumps(row, sort_keys=True))
            del x, dense_out, auto
        del packed, dense, weight
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
