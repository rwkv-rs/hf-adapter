#!/usr/bin/env python3
"""Validate one canonical RWKV-7 training recurrence candidate on CUDA.

The readable PyTorch recurrence is the numerical reference.  The selected
candidate and pinned FLA implementation are independent comparison lanes.
Precision uses output, final-state and full-input gradients; timing uses the
ordinary detached-state training contract.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
from typing import Any, Callable

import torch

from common import environment, git_revision, sha256_file
from fla_common import (
    activate_fla_source,
    gradient_metrics,
    gradient_rows_passed,
    metric_passed,
    tensor_metric,
    write_json,
)
from training_metrics import adaptive_fast_domain_expected


DTYPE = torch.bfloat16
IMPLEMENTATIONS = {
    "matrix": "torch-cuda-rwkv7-batched-matrix-recurrent-training-v1",
    "factorized": "native-nvidia-rwkv7-factorized-recurrent-training-v1",
}
VECTOR_NAMES = ("receptance", "raw_decay", "key", "value", "a", "b")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fla-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", action="append", type=int, default=[])
    parser.add_argument("--tokens", action="append", type=int, default=[])
    parser.add_argument(
        "--padding",
        action="append",
        choices=("none", "left", "right"),
        default=[],
        help="repeat to validate multiple mask layouts; default is unpadded",
    )
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--code-sha")
    parser.add_argument("--hf-wheel", type=Path)
    parser.add_argument("--kernel-wheel", type=Path)
    parser.add_argument(
        "--candidate",
        choices=("adaptive", *IMPLEMENTATIONS),
        default="adaptive",
        help=(
            "adaptive uses the factorized route only in the certified dense "
            "zero-state B=4,T=128 domain and the exact matrix route elsewhere; "
            "matrix and factorized isolate one recurrent program"
        ),
    )
    return parser.parse_args()


def make_case(
    batch: int, tokens: int, heads: int, seed: int, padding: str
) -> dict[str, torch.Tensor]:
    if tokens <= 0:
        raise ValueError("training validation requires a non-empty sequence")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (batch, tokens, heads, 64)
    values = {
        name: torch.randn(
            shape,
            device="cuda",
            dtype=DTYPE,
            generator=generator,
        )
        * 0.1
        for name in ("receptance", "key", "value", "a", "b")
    }
    values["raw_decay"] = -(
        torch.rand(shape, device="cuda", dtype=DTYPE, generator=generator) * 0.5 + 0.1
    )
    values["initial_state"] = torch.zeros(
        batch,
        heads,
        64,
        64,
        device="cuda",
        dtype=torch.float32,
    )
    attention_mask = torch.ones(batch, tokens, dtype=torch.bool, device="cuda")
    if padding != "none":
        for batch_idx in range(batch):
            masked = min(tokens - 1, batch_idx % 3 + 1)
            if padding == "left":
                attention_mask[batch_idx, :masked] = False
            elif padding == "right":
                attention_mask[batch_idx, -masked:] = False
            else:
                raise ValueError(f"unknown padding mode: {padding}")
    values["attention_mask"] = attention_mask
    return values


def clone_case(
    base: dict[str, torch.Tensor], *, state_requires_grad: bool
) -> dict[str, torch.Tensor]:
    return {
        name: value.detach()
        .clone()
        .requires_grad_(
            state_requires_grad
            if name == "initial_state"
            else bool(value.is_floating_point())
        )
        for name, value in base.items()
    }


def translated_decay(raw_decay: torch.Tensor) -> torch.Tensor:
    """Translate the public raw-decay parameter in canonical FP32."""

    return raw_decay.float().exp()


def run_reference(values: dict[str, torch.Tensor]):
    from rwkv7_hf.ops_rwkv7 import rwkv7_recurrent_reference

    return rwkv7_recurrent_reference(
        values["receptance"],
        translated_decay(values["raw_decay"]),
        values["key"],
        values["value"],
        values["a"],
        values["b"],
        values["initial_state"],
        values["attention_mask"],
    )


def run_candidate(values: dict[str, torch.Tensor]):
    from rwkv7_hf.ops_rwkv7 import get_last_recurrent_route, rwkv7_recurrent

    output = rwkv7_recurrent(
        values["receptance"],
        translated_decay(values["raw_decay"]),
        values["key"],
        values["value"],
        values["a"],
        values["b"],
        values["initial_state"],
        values["attention_mask"],
        backend="optimized",
        training=True,
        initial_state_zero=True,
    )
    return (*output, get_last_recurrent_route())


def run_fla(values: dict[str, torch.Tensor]):
    from fla.ops.rwkv7 import chunk_rwkv7

    mask = values["attention_mask"]
    if bool(mask.all().detach().cpu()):
        output, final_state = chunk_rwkv7(
            r=values["receptance"],
            w=values["raw_decay"],
            k=values["key"],
            v=values["value"],
            a=values["a"],
            b=values["b"],
            initial_state=values["initial_state"],
            output_final_state=True,
        )
        return output, final_state, None

    outputs = []
    states = []
    for batch_idx in range(int(mask.shape[0])):
        active = torch.nonzero(mask[batch_idx], as_tuple=False).flatten()
        selected = {
            name: values[name][batch_idx : batch_idx + 1].index_select(1, active)
            for name in VECTOR_NAMES
        }
        output, final_state = chunk_rwkv7(
            r=selected["receptance"],
            w=selected["raw_decay"],
            k=selected["key"],
            v=selected["value"],
            a=selected["a"],
            b=selected["b"],
            initial_state=values["initial_state"][batch_idx : batch_idx + 1],
            output_final_state=True,
        )
        restored = torch.zeros_like(
            values["value"][batch_idx : batch_idx + 1]
        ).index_copy(1, active, output)
        outputs.append(restored)
        states.append(final_state)
    return torch.cat(outputs), torch.cat(states), None


LANES: dict[str, Callable] = {
    "reference": lambda values: (*run_reference(values), None),
    "candidate": run_candidate,
    "fla": run_fla,
}


def collect_lane(
    lane: str,
    base: dict[str, torch.Tensor],
    *,
    include_final_state_in_loss: bool,
    state_requires_grad: bool,
) -> dict[str, Any]:
    values = clone_case(base, state_requires_grad=state_requires_grad)
    output, final_state, route = LANES[lane](values)
    loss = output.float().square().mean()
    if include_final_state_in_loss:
        loss = loss + final_state.float().square().mean()
    loss.backward()
    gradients = {
        name: value.grad.detach().cpu()
        for name, value in values.items()
        if value.grad is not None
    }
    return {
        "output": output.detach().cpu(),
        "final_state": final_state.detach().cpu(),
        "loss": float(loss.detach()),
        "gradients": gradients,
        "route": route,
    }


def compare_lane(
    candidate: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    output = tensor_metric(candidate["output"], reference["output"])
    final_state = tensor_metric(candidate["final_state"], reference["final_state"])
    gradients = gradient_metrics(candidate["gradients"], reference["gradients"])
    passed = bool(
        metric_passed(output, DTYPE)
        and metric_passed(final_state, DTYPE)
        and gradient_rows_passed(gradients, DTYPE)
    )
    return {
        "passed": passed,
        "output": output,
        "final_state": final_state,
        "loss": {
            "candidate": candidate["loss"],
            "reference": reference["loss"],
            "absolute_difference": abs(candidate["loss"] - reference["loss"]),
        },
        "gradients": gradients,
    }


def route_passed(route: dict[str, Any] | None, *, implementation: str) -> bool:
    return bool(
        route
        and route.get("selected") == "optimized"
        and route.get("implementation") == implementation
    )


def expected_implementation(
    candidate: str, padding: str, batch: int, tokens: int
) -> str:
    if candidate == "adaptive":
        factorized = adaptive_fast_domain_expected(
            batch=batch,
            tokens=tokens,
            fully_active=padding == "none",
            initial_state_zero=True,
        )
        return IMPLEMENTATIONS["factorized" if factorized else "matrix"]
    return IMPLEMENTATIONS[candidate]


def benchmark_lane(
    lane: str,
    base: dict[str, torch.Tensor],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    timings = []
    routes = []
    for index in range(warmup + iterations):
        values = clone_case(base, state_requires_grad=False)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output, _final_state, route = LANES[lane](values)
        output.float().square().mean().backward()
        end.record()
        end.synchronize()
        if index >= warmup:
            timings.append(float(start.elapsed_time(end)))
            routes.append(route)
    median_ms = statistics.median(timings)
    batch, tokens = base["receptance"].shape[:2]
    return {
        "median_milliseconds": median_ms,
        "minimum_milliseconds": min(timings),
        "maximum_milliseconds": max(timings),
        "samples_per_second": batch * 1000.0 / median_ms,
        "tokens_per_second": batch * tokens * 1000.0 / median_ms,
        "iterations": iterations,
        "routes": routes,
    }


def benchmark_case(
    base: dict[str, torch.Tensor], *, warmup: int, iterations: int
) -> dict[str, Any]:
    lanes = {}
    for lane in LANES:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        row = benchmark_lane(
            lane,
            base,
            warmup=warmup,
            iterations=iterations,
        )
        row["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated())
        lanes[lane] = row
    candidate_ms = lanes["candidate"]["median_milliseconds"]
    return {
        "lanes": lanes,
        "candidate_speedup_vs_reference": (
            lanes["reference"]["median_milliseconds"] / candidate_ms
        ),
        "candidate_speedup_vs_fla": (
            lanes["fla"]["median_milliseconds"] / candidate_ms
        ),
    }


def validate_auto_fallback(base: dict[str, torch.Tensor]) -> dict[str, Any]:
    from rwkv7_hf.ops_rwkv7 import get_last_recurrent_route, rwkv7_recurrent

    previous = os.environ.get("RWKV7_TRAINING_KERNEL_IMPL")
    os.environ["RWKV7_TRAINING_KERNEL_IMPL"] = "auto"
    try:
        values = clone_case(base, state_requires_grad=False)
        candidate_output, candidate_state = rwkv7_recurrent(
            values["receptance"],
            translated_decay(values["raw_decay"]),
            values["key"],
            values["value"],
            values["a"],
            values["b"],
            values["initial_state"],
            values["attention_mask"],
            backend="auto",
            training=True,
            initial_state_zero=True,
        )
        reference_output, reference_state = run_reference(values)
        route = get_last_recurrent_route()
    finally:
        if previous is None:
            os.environ.pop("RWKV7_TRAINING_KERNEL_IMPL", None)
        else:
            os.environ["RWKV7_TRAINING_KERNEL_IMPL"] = previous
    output = tensor_metric(candidate_output.cpu(), reference_output.cpu())
    final_state = tensor_metric(candidate_state.cpu(), reference_state.cpu())
    passed = bool(
        route
        and route.get("selected") == "reference"
        and output["max_abs"] == 0.0
        and final_state["max_abs"] == 0.0
    )
    return {
        "passed": passed,
        "route": route,
        "output": output,
        "final_state": final_state,
    }


def wheel_rows(args: argparse.Namespace) -> dict[str, Any]:
    rows = {}
    for name, path in (
        ("rwkv7_hf", args.hf_wheel),
        ("rwkv7_kernels", args.kernel_wheel),
    ):
        if path is not None:
            resolved = path.expanduser().resolve()
            rows[name] = {"path": str(resolved), "sha256": sha256_file(resolved)}
    return rows


def main() -> int:
    args = arguments()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.heads <= 0 or args.warmup < 0 or args.iterations <= 0:
        raise ValueError(
            "heads and iterations must be positive; warmup cannot be negative"
        )

    fla = activate_fla_source(args.fla_source)
    os.environ["RWKV7_TRAINING_KERNEL_IMPL"] = args.candidate
    os.environ["RWKV7_MODEL_KERNEL_IMPL"] = "auto"
    batches = tuple(args.batch or (1, 4))
    tokens = tuple(args.tokens or (16, 128))
    padding_modes = tuple(args.padding or ("none",))
    cases = []
    for batch in batches:
        for token_count in tokens:
            for padding in padding_modes:
                base = make_case(
                    batch,
                    token_count,
                    args.heads,
                    args.seed
                    + batch * 1000
                    + token_count
                    + {"none": 0, "left": 100_000, "right": 200_000}[padding],
                    padding,
                )
                audit_state_gradient = token_count == min(tokens)
                case_implementation = expected_implementation(
                    args.candidate,
                    padding,
                    batch,
                    token_count,
                )
                precision_lanes = {
                    lane: collect_lane(
                        lane,
                        base,
                        include_final_state_in_loss=audit_state_gradient,
                        state_requires_grad=audit_state_gradient,
                    )
                    for lane in LANES
                }
                comparisons = {
                    lane: compare_lane(
                        precision_lanes[lane], precision_lanes["reference"]
                    )
                    for lane in ("candidate", "fla")
                }
                candidate_vs_fla = compare_lane(
                    precision_lanes["candidate"], precision_lanes["fla"]
                )
                actual_route = precision_lanes["candidate"]["route"]
                performance = benchmark_case(
                    base,
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
                benchmark_routes_passed = all(
                    route_passed(route, implementation=case_implementation)
                    for route in performance["lanes"]["candidate"]["routes"]
                )
                passed = bool(
                    comparisons["candidate"]["passed"]
                    and comparisons["fla"]["passed"]
                    and route_passed(actual_route, implementation=case_implementation)
                    and benchmark_routes_passed
                )
                cases.append(
                    {
                        "case": (
                            f"b{batch}-t{token_count}-h{args.heads}-d64-"
                            f"padding-{padding}"
                        ),
                        "passed": passed,
                        "gradient_contract": (
                            "vectors-and-initial-state"
                            if audit_state_gradient
                            else "vectors"
                        ),
                        "actual_route": actual_route,
                        "benchmark_routes_passed": benchmark_routes_passed,
                        "candidate": args.candidate,
                        "expected_implementation": case_implementation,
                        "candidate_vs_reference": comparisons["candidate"],
                        "fla_vs_reference": comparisons["fla"],
                        "candidate_vs_fla": candidate_vs_fla,
                        "performance": performance,
                    }
                )
                del base, precision_lanes
                torch.cuda.empty_cache()

    fallback_base = make_case(1, 16, args.heads, args.seed + 900_000, "none")
    auto_fallback = validate_auto_fallback(fallback_base)
    passed = all(case["passed"] for case in cases) and auto_fallback["passed"]
    report = {
        "schema": "rwkv7-recurrent-training-three-way-v1",
        "status": "passed" if passed else "failed",
        "code_sha": args.code_sha or git_revision(Path(__file__).resolve().parents[1]),
        "candidate": args.candidate,
        "expected_implementations": (
            {
                "certified_dense_b4_t128_zero_state": IMPLEMENTATIONS["factorized"],
                "all_other_or_unsupported_requests": IMPLEMENTATIONS["matrix"],
            }
            if args.candidate == "adaptive"
            else {"all_cases": IMPLEMENTATIONS[args.candidate]}
        ),
        "dtype": "bfloat16",
        "fla": fla,
        "environment": environment(),
        "wheels": wheel_rows(args),
        "settings": {
            "batches": batches,
            "tokens": tokens,
            "padding": padding_modes,
            "heads": args.heads,
            "head_size": 64,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "seed": args.seed,
        },
        "precision_contract": {
            "reference": "rwkv7_hf.ops_rwkv7.rwkv7_recurrent_reference",
            "vector_gradients": list(VECTOR_NAMES),
            "initial_state_gradient": (
                "included for the shortest requested sequence in each batch"
            ),
            "stateful_loss": "mean(output_fp32^2) + mean(final_state_fp32^2)",
            "ordinary_loss": "mean(output_fp32^2)",
        },
        "performance_contract": {
            "loss": "mean(output_fp32^2)",
            "initial_state": "detached zero state",
            "scope": "warmed recurrent forward plus backward",
        },
        "auto_fallback": auto_fallback,
        "cases": cases,
        "failures": [case for case in cases if not case["passed"]],
    }
    write_json(args.output, report)
    print(json.dumps({"output": str(args.output), "status": report["status"]}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
