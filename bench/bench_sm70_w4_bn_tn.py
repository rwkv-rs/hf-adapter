# coding=utf-8
"""Sweep production-shaped BN/TN tiles for the exact-sm70 W4 kernel."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from rwkv7_hf.sm70_quant import (
    SM70_W4_BN_TN_CHOICES,
    build_error,
    is_sm70,
    quantize_w4_groupwise,
    quantize_w4_row,
    w4_groupwise_linear,
    w4_linear,
)


def parse_shape(raw: str) -> tuple[int, int]:
    try:
        k, n = (int(part) for part in raw.lower().split("x", 1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"expected KxN, got {raw!r}") from exc
    if k <= 0 or n <= 0 or k % 8:
        raise argparse.ArgumentTypeError("K and N must be positive and K divisible by 8")
    return k, n


def timed_ms(fn, *, warmup: int, runs: int) -> float:
    for _ in range(int(warmup)):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(int(runs)):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / int(runs)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.float().reshape(-1), b.float().reshape(-1), dim=0))


def select_pairs(
    block_ns: list[int],
    thread_ns: list[int],
) -> tuple[tuple[int, int], ...]:
    requested = tuple(
        pair
        for pair in SM70_W4_BN_TN_CHOICES
        if pair[0] in set(block_ns) and pair[1] in set(thread_ns)
    )
    if not requested:
        raise ValueError("no legal sm70 W4 BN/TN pair selected")
    return requested


def set_pair(pair: tuple[int, int]) -> None:
    os.environ["RWKV7_SM70_W4_BN"] = str(pair[0])
    os.environ["RWKV7_SM70_W4_TN"] = str(pair[1])
    os.environ["RWKV7_SM70_W4_GROUP_BN"] = str(pair[0])
    os.environ["RWKV7_SM70_W4_GROUP_TN"] = str(pair[1])


def prepare_weight(weight: torch.Tensor, group_size: int):
    if group_size:
        return quantize_w4_groupwise(weight, group_size=group_size)
    return quantize_w4_row(weight)


def apply_w4(x, packed, scales, n, k, group_size, out):
    if group_size:
        return w4_groupwise_linear(
            x,
            packed,
            scales,
            n,
            k,
            group_size=group_size,
            out=out,
        )
    return w4_linear(x, packed, scales, n, k, out=out)


def run_case(args, batch: int, shape: tuple[int, int], pairs) -> list[dict[str, object]]:
    k, n = shape
    x = torch.randn((batch, k), device="cuda", dtype=torch.float16) * 0.1
    weight = torch.randn((n, k), device="cuda", dtype=torch.float16) * 0.02
    packed, scales, in_features = prepare_weight(weight, args.group_size)
    out = torch.empty((batch, n), device="cuda", dtype=torch.float16)

    current_pair = (8, 1)
    set_pair(current_pair)
    current = apply_w4(
        x, packed, scales, n, in_features, args.group_size, out
    ).clone()
    current_ms = timed_ms(
        lambda: apply_w4(
            x, packed, scales, n, in_features, args.group_size, out
        ),
        warmup=args.warmup,
        runs=args.runs,
    )
    fp16 = F.linear(x, weight)
    fp16_ms = timed_ms(
        lambda: F.linear(x, weight), warmup=args.warmup, runs=args.runs
    )

    rows = []
    for bn, tn in pairs:
        set_pair((bn, tn))
        candidate = apply_w4(
            x, packed, scales, n, in_features, args.group_size, out
        ).clone()
        candidate_ms = timed_ms(
            lambda: apply_w4(
                x, packed, scales, n, in_features, args.group_size, out
            ),
            warmup=args.warmup,
            runs=args.runs,
        )
        quant_cosine = cosine(candidate, current)
        row = {
            "axis": "sm70_w4_bn_tn",
            "device": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "batch_size": batch,
            "k": k,
            "n": n,
            "block_n": bn,
            "thread_n": tn,
            "group_size": args.group_size,
            "threads": (bn // tn) * (32 if not args.group_size else 16),
            "current_ms": round(current_ms, 6),
            "fp16_ms": round(fp16_ms, 6),
            "candidate_ms": round(candidate_ms, 6),
            "speedup_vs_current": round(current_ms / candidate_ms, 6),
            "speedup_vs_fp16": round(fp16_ms / candidate_ms, 6),
            "cosine_vs_current": round(quant_cosine, 9),
            "cosine_vs_fp16": round(cosine(candidate, fp16), 9),
            "max_abs_vs_current": round(float((candidate.float() - current.float()).abs().max()), 6),
            "status": "pass" if quant_cosine >= args.min_cosine else "fail",
        }
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=(1, 8))
    parser.add_argument("--shapes", nargs="+", type=parse_shape, default=((2048, 2048),))
    parser.add_argument("--block-n", nargs="+", type=int, default=(4, 8, 16))
    parser.add_argument("--thread-n", nargs="+", type=int, default=(1, 2, 4))
    parser.add_argument("--group-size", type=int, choices=(0, 128, 256), default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--min-cosine", type=float, default=0.9999)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not is_sm70():
        raise SystemExit("an exact sm_70 CUDA device is required")
    torch.manual_seed(7070)
    pairs = select_pairs(args.block_n, args.thread_n)
    all_rows = []
    for batch in args.batch_sizes:
        if batch < 1:
            raise SystemExit("sm70 W4 requires at least one activation row")
        for shape in args.shapes:
            rows = run_case(args, batch, shape, pairs)
            all_rows.extend(rows)
            passing = [row for row in rows if row["status"] == "pass"]
            best = min(passing, key=lambda row: row["candidate_ms"], default=None)
            print(json.dumps({"case": [batch, *shape], "best": best}, sort_keys=True))
    if build_error() is not None:
        raise SystemExit(f"sm70 extension build failed: {build_error()}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in all_rows),
            encoding="utf-8",
        )
    return 0 if all(row["status"] == "pass" for row in all_rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
