#!/usr/bin/env python3
"""Profile RWKV-7 readable-HF training with independently dispatched leaves.

The tool deliberately reuses the three training lanes from
``benchmark_backend_v2``.  It is an evaluation utility rather than a runtime
component: no profiler or FLA dependency crosses the model/kernel package
boundary.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import resource
import sys
import time
from typing import Any, Callable, Iterable

import torch

from benchmark_backend_v2 import (
    last_training_routes,
    load_model,
    training_route_mode,
)
from common import environment, git_revision, model_fingerprint, sha256_file
from fla_common import activate_fla_source, write_json
from training_metrics import adaptive_fast_domain_expected


SCHEMA = "rwkv7-training-hotspot-profile-v3"
PROGRAM_IMPLEMENTATION = "native-nvidia-rwkv7-adaptive-training-program-v1"
LANES = ("reference", "optimized", "fla")
EXACT_HOST_SYNC_OPERATORS = (
    "aten::item",
    "aten::_local_scalar_dense",
)
SELECTED_OPERATORS = (
    "aten::mm",
    "aten::addmm",
    "aten::copy_",
    "aten::cat",
    "aten::item",
    "aten::_local_scalar_dense",
)
TOP_EVENT_COUNT = 25


def _process_peak_rss_bytes() -> int:
    """Return this process' peak resident set size in bytes."""

    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                fields = line.split()
                if len(fields) >= 2:
                    return int(fields[1]) * 1024
    except (OSError, ValueError):
        pass

    # ru_maxrss is bytes on macOS and KiB on Linux/other supported Unix.
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "For formal evidence, run one lane and one shape per OS process. "
            "This bounds profiler host memory and isolates allocator state."
        ),
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--lane",
        action="append",
        choices=LANES,
        default=[],
        help="lane to profile; repeat as needed (default: all three lanes)",
    )
    parser.add_argument(
        "--fla-source",
        type=Path,
        help="pinned FLA checkout; required when the FLA lane is selected",
    )
    parser.add_argument("--batch", action="append", type=int, default=[])
    parser.add_argument("--tokens", action="append", type=int, default=[])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--active",
        type=int,
        default=1,
        help="profiled steps per case (default: 1 to bound profiler host memory)",
    )
    parser.add_argument(
        "--profile-memory-events",
        action="store_true",
        help=(
            "collect per-operator memory events; disabled by default because "
            "it substantially increases profiler host memory"
        ),
    )
    parser.add_argument(
        "--group-by-input-shape",
        action="store_true",
        help=(
            "record input shapes and retain per-shape key-average groups; "
            "disabled by default to keep profiles small"
        ),
    )
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-sha")
    parser.add_argument("--hf-wheel", type=Path)
    parser.add_argument("--kernel-wheel", type=Path)
    return parser.parse_args(argv)


def validate_arguments(args: argparse.Namespace) -> None:
    lanes = tuple(args.lane or LANES)
    batches = tuple(args.batch or (1, 4))
    tokens = tuple(args.tokens or (128,))
    if args.warmup < 0 or args.active <= 0:
        raise ValueError("warmup must be non-negative and active must be positive")
    if any(value <= 0 for value in (*batches, *tokens)):
        raise ValueError("batch and tokens must be positive")
    if any(value < 2 for value in tokens):
        raise ValueError("training profiles require at least two tokens")
    if "optimized" in lanes and args.dtype != "bf16":
        raise ValueError("optimized clean-leaf profiling requires --dtype bf16")
    if "fla" in lanes and args.fla_source is None:
        raise ValueError("--fla-source is required when profiling the FLA lane")


def _event_number(event: Any, name: str) -> float:
    value = getattr(event, name, 0.0)
    return float(value if value is not None else 0.0)


def _jsonable_shape(value: Any) -> Any:
    """Convert profiler shape metadata to a JSON-stable representation."""

    if isinstance(value, (list, tuple)):
        return [_jsonable_shape(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def event_row(event: Any, *, include_input_shapes: bool = False) -> dict[str, Any]:
    """Convert a profiler key-average row into stable JSON units."""

    row = {
        "name": str(event.key),
        "count": int(getattr(event, "count", 0)),
        "self_cpu_time_us": _event_number(event, "self_cpu_time_total"),
        "cpu_time_us": _event_number(event, "cpu_time_total"),
        "self_device_time_us": _event_number(event, "self_device_time_total"),
        "device_time_us": _event_number(event, "device_time_total"),
        "cpu_memory_bytes": int(_event_number(event, "cpu_memory_usage")),
        "device_memory_bytes": int(_event_number(event, "device_memory_usage")),
    }
    if include_input_shapes:
        row["input_shapes"] = _jsonable_shape(getattr(event, "input_shapes", None))
    return row


def _sum_event_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    return {
        "count": sum(row["count"] for row in rows),
        "self_cpu_time_us": sum(row["self_cpu_time_us"] for row in rows),
        "cpu_time_us": sum(row["cpu_time_us"] for row in rows),
        "self_device_time_us": sum(row["self_device_time_us"] for row in rows),
        "device_time_us": sum(row["device_time_us"] for row in rows),
        "cpu_memory_bytes": sum(row["cpu_memory_bytes"] for row in rows),
        "device_memory_bytes": sum(row["device_memory_bytes"] for row in rows),
    }


def _aggregate_exact_names(
    rows: Iterable[dict[str, Any]],
    *,
    include_input_shapes: bool,
) -> list[dict[str, Any]]:
    """Merge duplicate exact-name rows without discarding shape groups.

    ``key_averages(group_by_input_shape=True)`` may emit several rows with the
    same operator name.  Legacy summary fields remain exact-name aggregates;
    optional shape detail is attached separately to each aggregate row.
    """

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["name"], []).append(row)

    aggregated = []
    for name, matching in grouped.items():
        aggregate = {"name": name, **_sum_event_rows(matching)}
        if include_input_shapes:
            aggregate["input_shape_groups"] = [
                {
                    "input_shapes": row.get("input_shapes"),
                    **_sum_event_rows((row,)),
                }
                for row in matching
            ]
        aggregated.append(aggregate)
    return aggregated


def _is_recurrent(name: str) -> bool:
    lowered = name.lower()
    return any(
        marker in lowered
        for marker in (
            "rwkv7_clampw",
            "recurrent_rwkv",
            "rwkv7_recurrent",
            "chunk_rwkv7",
            "chunkdplr",
            "dplrfunction",
        )
    )


def _is_backward(name: str) -> bool:
    lowered = name.lower()
    return "backward" in lowered or "autograd" in lowered


def _matches_any(name: str, markers: tuple[str, ...]) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in markers)


def _category_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the non-GEMM families needed for an actionable profile."""

    categories = {
        "mix6": ("mix6", "tmix_mix6"),
        "causal_loss": (
            "cross_entropy",
            "nll_loss",
            "log_softmax",
            "l2wrap",
        ),
        "normalization": (
            "layer_norm",
            "group_norm",
            "native_layer_norm",
            "native_group_norm",
        ),
        "allocation_or_zeroing": (
            "aten::zero_",
            "aten::zeros",
            "aten::empty",
            "aten::fill_",
        ),
        "copy_or_layout": (
            "aten::copy_",
            "aten::contiguous",
            "aten::clone",
            "aten::cat",
        ),
    }
    result = {}
    for category, markers in categories.items():
        matching = [row for row in rows if _matches_any(row["name"], markers)]
        result[category] = {
            "aggregate": _sum_event_rows(matching),
            "events": sorted(
                matching,
                key=lambda row: row["self_device_time_us"],
                reverse=True,
            ),
        }
    host_sync_rows = [row for row in rows if row["name"] in EXACT_HOST_SYNC_OPERATORS]
    result["host_synchronization"] = {
        "aggregate": _sum_event_rows(host_sync_rows),
        "events": host_sync_rows,
        "exact_operators": list(EXACT_HOST_SYNC_OPERATORS),
    }
    return result


def summarize_events(
    events: Iterable[Any],
    *,
    include_input_shapes: bool = False,
) -> dict[str, Any]:
    """Summarize recurrent, launch-heavy, and top self-time profiler rows."""

    key_average_rows = [
        event_row(event, include_input_shapes=include_input_shapes) for event in events
    ]
    rows = _aggregate_exact_names(
        key_average_rows,
        include_input_shapes=include_input_shapes,
    )
    by_name = {row["name"]: row for row in rows}
    selected = {
        name: by_name.get(
            name,
            {
                "name": name,
                "count": 0,
                "self_cpu_time_us": 0.0,
                "cpu_time_us": 0.0,
                "self_device_time_us": 0.0,
                "device_time_us": 0.0,
                "cpu_memory_bytes": 0,
                "device_memory_bytes": 0,
            },
        )
        for name in SELECTED_OPERATORS
    }
    recurrent_rows = [row for row in rows if _is_recurrent(row["name"])]
    recurrent_backward = [row for row in recurrent_rows if _is_backward(row["name"])]
    recurrent_forward = [row for row in recurrent_rows if not _is_backward(row["name"])]
    top_self_device_time = sorted(
        rows,
        key=lambda row: row["self_device_time_us"],
        reverse=True,
    )[:TOP_EVENT_COUNT]
    top_launch_count = sorted(
        rows,
        key=lambda row: row["count"],
        reverse=True,
    )[:TOP_EVENT_COUNT]
    categories = _category_rows(rows)
    host_sync = categories["host_synchronization"]
    return {
        "selected_operators": selected,
        "recurrent": {
            "forward": {
                "aggregate": _sum_event_rows(recurrent_forward),
                "events": recurrent_forward,
            },
            "backward": {
                "aggregate": _sum_event_rows(recurrent_backward),
                "events": recurrent_backward,
            },
        },
        "categories": categories,
        "host_synchronization_gate": {
            **host_sync,
            "passed": host_sync["aggregate"]["count"] == 0,
        },
        "top_self_device_time": top_self_device_time,
        "top_launch_count": top_launch_count,
        "total_operator_calls": sum(row["count"] for row in rows),
        "event_count": len(rows),
        "key_average_row_count": len(key_average_rows),
        "grouped_by_input_shape": include_input_shapes,
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory(device: torch.device) -> int:
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    return 0


def expected_route(
    lane: str,
    route: dict[str, Any] | None,
    *,
    batch: int = 1,
    tokens: int = 128,
) -> bool:
    if lane == "fla":
        return route is None
    route = route or {}
    model = route.get("model") or {}
    recurrent = route.get("recurrent") or {}
    linear = route.get("linear") or {}
    mix6 = route.get("mix6") or {}
    program = route.get("program") or {}
    if lane == "reference":
        return (
            model.get("selected") == "reference"
            and model.get("implementation") == "torch-reference-model-v1"
            and recurrent.get("selected") == "reference"
            and recurrent.get("implementation") == "torch-reference-v1"
            and linear.get("selected") == "reference"
            and linear.get("implementation") == "torch-reference-linear-v1"
            and mix6.get("selected") == "reference"
            and mix6.get("implementation") == "torch-reference-mix6-v1"
            and program.get("selected") == "reference"
            and program.get("implementation") == "torch-reference-training-program-v1"
        )
    fast_domain = adaptive_fast_domain_expected(batch=batch, tokens=tokens)
    recurrent_implementation = (
        "native-nvidia-rwkv7-factorized-recurrent-training-v1"
        if fast_domain
        else "torch-cuda-rwkv7-batched-matrix-recurrent-training-v1"
    )
    linear_passed = (
        linear.get("selected") == "optimized"
        and linear.get("implementation")
        == "torch-cuda-rwkv7-flattened-linear-training-v1"
        if fast_domain
        else linear.get("selected") == "reference"
        and linear.get("implementation") == "torch-reference-linear-v1"
    )
    return bool(
        model.get("selected") == "reference"
        and model.get("implementation") == "torch-reference-model-v1"
        and recurrent.get("selected") == "optimized"
        and recurrent.get("implementation") == recurrent_implementation
        and linear_passed
        and mix6.get("selected") == "optimized"
        and mix6.get("implementation") == "native-nvidia-rwkv7-mix6-training-v1"
        and program.get("selected") == ("optimized" if fast_domain else "reference")
        and program.get("implementation") == PROGRAM_IMPLEMENTATION
    )


def profile_training_case(
    model: Any,
    ids: torch.Tensor,
    labels: torch.Tensor,
    *,
    lane: str,
    warmup: int,
    active: int,
    profile_memory_events: bool = False,
    group_by_input_shape: bool = False,
    route_getter: Callable[[str], dict[str, Any] | None] = last_training_routes,
    profiler_factory: Callable[..., Any] = torch.profiler.profile,
) -> dict[str, Any]:
    """Profile one shape, using ``output.loss`` exactly once per step."""

    if ids.shape != labels.shape or ids.ndim != 2:
        raise ValueError("input_ids and labels must have the same [B,T] shape")
    device = ids.device
    losses: list[torch.Tensor] = []

    def step() -> torch.Tensor:
        model.zero_grad(set_to_none=True)
        output = model(
            input_ids=ids,
            labels=labels,
            use_cache=False,
            logits_to_keep=0,
        )
        loss = output.loss
        if loss is None:
            raise RuntimeError("training model did not return output.loss")
        loss.backward()
        # Keep profiling asynchronous.  Converting a CUDA scalar to ``float``
        # here would synchronize every step and manufacture a false hotspot.
        losses.append(loss.detach())
        return loss

    for _ in range(warmup):
        step()
    _synchronize(device)
    _reset_peak_memory(device)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with profiler_factory(
        activities=activities,
        record_shapes=group_by_input_shape,
        profile_memory=profile_memory_events,
        with_stack=False,
    ) as profiler:
        started = time.perf_counter()
        for _ in range(active):
            step()
            profiler.step()
        _synchronize(device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
    route = route_getter(lane)
    active_losses = [float(loss) for loss in losses[-active:]]
    hotspots = summarize_events(
        profiler.key_averages(group_by_input_shape=group_by_input_shape),
        include_input_shapes=group_by_input_shape,
    )
    host_sync_gate = hotspots["host_synchronization_gate"]
    return {
        "lane": lane,
        "shape": {"batch": int(ids.shape[0]), "tokens": int(ids.shape[1])},
        "warmup_steps": warmup,
        "active_steps": active,
        "profiler_settings": {
            "record_shapes": group_by_input_shape,
            "group_by_input_shape": group_by_input_shape,
            "profile_memory": profile_memory_events,
            "profile_memory_events": profile_memory_events,
            "with_stack": False,
        },
        "process_peak_rss_bytes": _process_peak_rss_bytes(),
        "loss_mode": "model-output-loss",
        "loss": {
            "samples": active_losses,
            "finite": all(math.isfinite(value) for value in active_losses),
            "last": active_losses[-1],
        },
        # This interval deliberately encloses profiler collection. It is useful
        # only as profile provenance and must never be presented as benchmark
        # latency; benchmark_backend_v2 supplies unprofiled CUDA-event timing.
        "profiled_wall_time_ms": elapsed_ms,
        "profiled_wall_time_per_active_step_ms": elapsed_ms / active,
        "profiled_wall_time_includes_profiler_overhead": True,
        "peak_memory_bytes": _peak_memory(device),
        "route": route,
        "route_passed": expected_route(
            lane,
            route,
            batch=int(ids.shape[0]),
            tokens=int(ids.shape[1]),
        ),
        "host_synchronization_gate": host_sync_gate,
        "host_synchronization_passed": host_sync_gate["passed"],
        "hotspots": hotspots,
    }


def _wheel_rows(args: argparse.Namespace) -> dict[str, Any]:
    result = {}
    for name, value in (
        ("rwkv7_hf", args.hf_wheel),
        ("rwkv7_kernels", args.kernel_wheel),
    ):
        if value is None:
            continue
        path = value.expanduser().resolve()
        result[name] = {"path": str(path), "sha256": sha256_file(path)}
    return result


def build_report(
    args: argparse.Namespace,
    *,
    cases: dict[str, Any],
    fla: dict[str, str] | None,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    route_gate = bool(cases) and all(row["route_passed"] for row in cases.values())
    finite_loss_gate = bool(cases) and all(
        row["loss"]["finite"] for row in cases.values()
    )
    host_sync_gate = bool(cases) and all(
        row.get("host_synchronization_passed", False) for row in cases.values()
    )
    passed = route_gate and finite_loss_gate and host_sync_gate
    return {
        "schema": SCHEMA,
        "status": "passed" if passed else "failed",
        "code_sha": args.code_sha or git_revision(root),
        "command": list(sys.argv),
        "model": model_fingerprint(args.model.expanduser().resolve()),
        "dtype": args.dtype,
        "settings": {
            "lanes": list(args.lane or LANES),
            "batches": list(args.batch or (1, 4)),
            "tokens": list(args.tokens or (128,)),
            "warmup": args.warmup,
            "active": args.active,
            "record_shapes": args.group_by_input_shape,
            "group_by_input_shape": args.group_by_input_shape,
            "profile_memory": args.profile_memory_events,
            "profile_memory_events": args.profile_memory_events,
            "seed": args.seed,
            "loss_mode": "model-output-loss",
        },
        "gates": {
            "routes": route_gate,
            "finite_loss": finite_loss_gate,
            "no_profiled_scalar_host_sync": host_sync_gate,
        },
        "wheels": _wheel_rows(args),
        "fla": fla,
        "environment": environment(),
        "cases": cases,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_arguments(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal training hotspot profiles")
    lanes = tuple(args.lane or LANES)
    batches = tuple(args.batch or (1, 4))
    tokens = tuple(args.tokens or (128,))
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    fla = activate_fla_source(args.fla_source) if "fla" in lanes else None
    path = args.model.expanduser().resolve()
    cases: dict[str, Any] = {}
    for lane in lanes:
        model = load_model(lane, path, dtype, training=True)
        training_route_mode(lane, "adaptive")
        vocab = int(model.config.vocab_size)
        for batch in batches:
            for tokens_per_sample in tokens:
                generator = torch.Generator(device="cuda").manual_seed(
                    args.seed + batch * 1_000 + tokens_per_sample
                )
                ids = torch.randint(
                    1,
                    vocab,
                    (batch, tokens_per_sample),
                    generator=generator,
                    device="cuda",
                )
                labels = ids.clone()
                labels[0, tokens_per_sample // 2] = -100
                key = f"{lane}-b{batch}-t{tokens_per_sample}"
                cases[key] = profile_training_case(
                    model,
                    ids,
                    labels,
                    lane=lane,
                    warmup=args.warmup,
                    active=args.active,
                    profile_memory_events=args.profile_memory_events,
                    group_by_input_shape=args.group_by_input_shape,
                )
                del ids, labels
        del model
        torch.cuda.empty_cache()
    return build_report(args, cases=cases, fla=fla)


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    report = run(args)
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output.expanduser().resolve()),
                "status": report["status"],
            }
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
