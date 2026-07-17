#!/usr/bin/env python3
"""Validate the V100 native recurrent route against pinned FLA fused recurrence.

Pinned FLA's chunk RWKV-7 forward/backward does not lower on SM70.  This gate
therefore uses FLA's inference-only fused recurrent implementation on the same
B/T/H/K matrix, records the external limitation, and compares both correctness
and synchronized kernel latency.  It never presents this as chunk/backward or
full-model FLA coverage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import torch

from common import environment, git_revision, sha256_file
from fla_common import (
    activate_fla_source,
    metric_passed,
    tensor_metric,
    write_json,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fla-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", action="append", type=int, default=[])
    parser.add_argument("--tokens", action="append", type=int, default=[])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--code-sha")
    parser.add_argument("--hf-wheel", type=Path)
    parser.add_argument("--kernel-wheel", type=Path)
    return parser.parse_args()


def measure(
    function: Callable[[], Any], *, warmup: int, repeats: int
) -> dict[str, Any]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        stop.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(stop)))
    samples.sort()
    return {
        "median_ms": samples[len(samples) // 2],
        "samples_ms": samples,
    }


def optimized_route_passed(route: dict[str, Any] | None) -> bool:
    return bool(
        route
        and route.get("selected") == "optimized"
        and route.get("implementation")
        in {
            "native-triton-rank1-scan-v1",
            "torch-cuda-graph-reference-v1",
        }
    )


def main() -> int:
    args = arguments()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    capability = tuple(torch.cuda.get_device_capability())
    if capability != (7, 0):
        raise SystemExit(f"this explicit SM70 gate requires capability 7.0, got {capability}")
    fla = activate_fla_source(args.fla_source)
    from fla.ops.rwkv7 import fused_recurrent_rwkv7
    from rwkv7_hf.ops_rwkv7 import (
        get_last_recurrent_route,
        rwkv7_recurrent,
        rwkv7_recurrent_reference,
    )

    batches = tuple(args.batch or (1, 4))
    lengths = tuple(args.tokens or (1, 17, 128))
    cases = []
    for batch in batches:
        for tokens in lengths:
            generator = torch.Generator(device="cuda").manual_seed(
                args.seed + batch * 1000 + tokens
            )
            shape = (batch, tokens, 2, 64)
            values = {
                name: torch.randn(
                    shape,
                    generator=generator,
                    device="cuda",
                    dtype=torch.float16,
                )
                * 0.1
                for name in ("r", "k", "v", "a", "b")
            }
            values["w"] = -(
                torch.rand(
                    shape,
                    generator=generator,
                    device="cuda",
                    dtype=torch.float16,
                )
                * 0.5
                + 0.1
            )
            values["state"] = (
                torch.randn(
                    (batch, 2, 64, 64),
                    generator=generator,
                    device="cuda",
                    dtype=torch.float32,
                )
                * 0.01
            )

            def reference():
                return rwkv7_recurrent_reference(
                    values["r"],
                    values["w"].exp(),
                    values["k"],
                    values["v"],
                    values["a"],
                    values["b"],
                    values["state"],
                )

            def optimized():
                return rwkv7_recurrent(
                    values["r"],
                    values["w"].exp(),
                    values["k"],
                    values["v"],
                    values["a"],
                    values["b"],
                    values["state"],
                    backend="optimized",
                )

            def fla_fused():
                return fused_recurrent_rwkv7(
                    values["r"],
                    values["w"],
                    values["k"],
                    values["v"],
                    values["a"],
                    values["b"],
                    scale=1.0,
                    initial_state=values["state"],
                    output_final_state=True,
                )

            with torch.inference_mode():
                reference_output, reference_state = reference()
                optimized_output, optimized_state = optimized()
                optimized_route = get_last_recurrent_route()
                fla_output, fla_state = fla_fused()
                comparisons = {}
                for label, output, state in (
                    ("optimized", optimized_output, optimized_state),
                    ("fla_fused_recurrent", fla_output, fla_state),
                ):
                    output_metric = tensor_metric(output.cpu(), reference_output.cpu())
                    state_metric = tensor_metric(state.cpu(), reference_state.cpu())
                    comparisons[label] = {
                        "passed": metric_passed(output_metric, torch.float16)
                        and metric_passed(state_metric, torch.float16),
                        "output": output_metric,
                        "state": state_metric,
                    }
                speed = {
                    "reference": measure(
                        reference, warmup=args.warmup, repeats=args.repeats
                    ),
                    "optimized": measure(
                        optimized, warmup=args.warmup, repeats=args.repeats
                    ),
                    "fla_fused_recurrent": measure(
                        fla_fused, warmup=args.warmup, repeats=args.repeats
                    ),
                }
            speed["optimized_vs_fla"] = float(
                speed["fla_fused_recurrent"]["median_ms"]
                / speed["optimized"]["median_ms"]
            )
            passed = bool(
                comparisons["optimized"]["passed"]
                and comparisons["fla_fused_recurrent"]["passed"]
                and optimized_route_passed(optimized_route)
            )
            cases.append(
                {
                    "case": f"b{batch}-t{tokens}",
                    "passed": passed,
                    "optimized_route": optimized_route,
                    "comparisons": comparisons,
                    "speed": speed,
                }
            )

    wheels = {}
    for name, path in (
        ("rwkv7_hf", args.hf_wheel),
        ("rwkv7_kernels", args.kernel_wheel),
    ):
        if path is not None:
            path = path.expanduser().resolve()
            wheels[name] = {"path": str(path), "sha256": sha256_file(path)}
    passed = all(row["passed"] for row in cases)
    report = {
        "schema": "rwkv7-backend-v2-fla-sm70-fused-recurrent-v1",
        "status": "passed" if passed else "failed",
        "purpose": "SM70 inference-only FLA fused recurrent parity and speed",
        "code_sha": args.code_sha or git_revision(Path(__file__).resolve().parents[1]),
        "fla": fla,
        "environment": environment(),
        "wheels": wheels,
        "settings": {
            "batches": batches,
            "tokens": lengths,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "limitations": [
            "pinned FLA fused_recurrent_dplr_delta_rule has no backward implementation",
            "pinned FLA chunk_dplr_delta_rule fails Triton LLVM lowering on compute capability 7.0",
            "this report is not full-model or FLA chunk/backward coverage",
        ],
        "cases": cases,
    }
    write_json(args.output, report)
    print(json.dumps({"output": str(args.output), "status": report["status"]}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
