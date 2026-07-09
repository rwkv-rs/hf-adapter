#!/usr/bin/env python3
"""Fair eager operator speed matrix for reference, optional v1, and FLA."""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from importlib import metadata
from pathlib import Path

import torch

from rwkv7_hf.ops_rwkv7 import (
    get_last_recurrent_route,
    rwkv7_recurrent,
    rwkv7_recurrent_reference,
)


EXPECTED_FLA_COMMIT = "80e494f6c588e091fc8316b612870df29375c5b8"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fla-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation", choices=("graph", "triton"), required=True)
    parser.add_argument("--batches", default="1,4,8")
    parser.add_argument("--lengths", default="1,17,128,512")
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--code-sha", required=True)
    return parser.parse_args()


def version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction + 0.5))
    return ordered[index]


def measure(function, *, warmup: int, repeats: int) -> dict:
    with torch.inference_mode():
        for _ in range(warmup):
            output = function()
            del output
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = function()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - started) * 1000.0)
            del output
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": percentile(samples, 0.95),
        "min_ms": min(samples),
        "samples_ms": samples,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
    }


def main() -> int:
    args = arguments()
    source = args.fla_source.expanduser().resolve()
    marker = source / ".fla-upstream-commit"
    commit = marker.read_text(encoding="utf-8").strip() if marker.is_file() else None
    if commit != EXPECTED_FLA_COMMIT:
        raise SystemExit(f"unexpected FLA commit: {commit!r}")
    sys.path.insert(0, str(source))
    from fla.ops.rwkv7 import chunk_rwkv7, fused_recurrent_rwkv7

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    os.environ["RWKV7_KERNEL_IMPL"] = args.implementation
    device = torch.device("cuda")
    dtype = torch.float16
    batches = [int(value) for value in args.batches.split(",")]
    lengths = [int(value) for value in args.lengths.split(",")]
    generator = torch.Generator(device=device).manual_seed(args.seed)

    report = {
        "schema": "rwkv7-recurrent-speed-v1",
        "code_sha": args.code_sha,
        "implementation": args.implementation,
        "fla_commit": commit,
        "settings": {
            "dtype": "fp16",
            "state_dtype": "fp32",
            "heads": args.heads,
            "head_size": 64,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "cuda_graph_at_model_level": False,
            "torch_compile": False,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "transformers": version("transformers"),
            "triton": version("triton"),
            "rwkv7_hf": version("rwkv7-hf"),
            "rwkv7_kernels": version("rwkv7-kernels"),
        },
        "cases": {},
    }

    for batch in batches:
        for length in lengths:
            shape = (batch, length, args.heads, 64)
            values = {
                name: torch.randn(
                    shape, generator=generator, device=device, dtype=dtype
                )
                * 0.05
                for name in ("r", "k", "v", "a", "b")
            }
            values["log_w"] = -(
                torch.rand(shape, generator=generator, device=device).float() * 0.5
                + 0.1
            )
            values["decay"] = values["log_w"].exp()
            values["state"] = (
                torch.randn(
                    batch,
                    args.heads,
                    64,
                    64,
                    generator=generator,
                    device=device,
                    dtype=torch.float32,
                )
                * 0.01
            )

            def reference():
                return rwkv7_recurrent_reference(
                    values["r"],
                    values["decay"],
                    values["k"],
                    values["v"],
                    values["a"],
                    values["b"],
                    values["state"],
                )

            def optimized():
                return rwkv7_recurrent(
                    values["r"],
                    values["decay"],
                    values["k"],
                    values["v"],
                    values["a"],
                    values["b"],
                    values["state"],
                    backend="optimized",
                )

            def fla_recurrent():
                return fused_recurrent_rwkv7(
                    r=values["r"],
                    w=values["log_w"],
                    k=values["k"],
                    v=values["v"],
                    a=values["a"],
                    b=values["b"],
                    initial_state=values["state"],
                    output_final_state=True,
                )

            def fla_chunk():
                return chunk_rwkv7(
                    r=values["r"],
                    w=values["log_w"],
                    k=values["k"],
                    v=values["v"],
                    a=values["a"],
                    b=values["b"],
                    initial_state=values["state"],
                    output_final_state=True,
                )

            rows = {
                "reference": measure(reference, warmup=args.warmup, repeats=args.repeats),
                "optimized": measure(optimized, warmup=args.warmup, repeats=args.repeats),
                "fla_fused_recurrent": measure(
                    fla_recurrent, warmup=args.warmup, repeats=args.repeats
                ),
                "fla_chunk": measure(
                    fla_chunk, warmup=args.warmup, repeats=args.repeats
                ),
            }
            rows["optimized"]["route"] = get_last_recurrent_route()
            for name, row in rows.items():
                row["tokens_per_second"] = batch * length / (
                    row["median_ms"] / 1000.0
                )
                if name != "reference":
                    row["speedup_vs_reference"] = (
                        rows["reference"]["median_ms"] / row["median_ms"]
                    )
            report["cases"][f"b{batch}_t{length}"] = rows
            del values

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "cases": len(report["cases"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
