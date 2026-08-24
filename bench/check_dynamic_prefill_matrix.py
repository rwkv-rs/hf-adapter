#!/usr/bin/env python3
"""Validate correctness and performance continuity of dynamic prefill shapes.

The gate consumes JSONL rows emitted by ``bench_native_prefill_scan.py`` (or
compatible ``batch_sweep`` rows).  It checks the full requested BxT grid,
shape-safe fused-scan selection, prompt-tile boundary continuity, and the
specific failure mode where a smaller batch is dramatically slower than the
next batch and internal padding would have won.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def parse_int_set(raw: str) -> list[int]:
    values: set[int] = set()
    for item in raw.replace(" ", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start <= 0 or end < start:
                raise ValueError(f"invalid positive integer range: {item!r}")
            values.update(range(start, end + 1))
        else:
            value = int(item)
            if value <= 0:
                raise ValueError(f"values must be positive: {value}")
            values.add(value)
    if not values:
        raise ValueError("at least one positive integer is required")
    return sorted(values)


def load_rows(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _first_number(row: dict[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        value = row.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _first_bool(row: dict[str, Any], names: tuple[str, ...]) -> bool | None:
    for name in names:
        value = row.get(name)
        if isinstance(value, bool):
            return value
    return None


def analyze(
    rows: Iterable[dict[str, Any]],
    *,
    batches: Iterable[int],
    prompts: Iterable[int],
    require_fused_scan: bool = True,
    require_safe_fusions: bool = False,
    max_padding_latency_ratio: float = 1.5,
    max_boundary_throughput_ratio: float = 1.35,
    max_cross_route_boundary_ratio: float = 3.0,
) -> dict[str, Any]:
    requested_batches = sorted({int(value) for value in batches})
    requested_prompts = sorted({int(value) for value in prompts})
    requested = {
        (batch, prompt)
        for batch in requested_batches
        for prompt in requested_prompts
    }
    selected: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        try:
            key = (int(row["batch_size"]), int(row["prompt_tokens"]))
        except (KeyError, TypeError, ValueError):
            continue
        if key not in requested:
            continue
        # Prefer the fast-token row from batch_sweep; native-prefill-scan emits
        # one row per shape and is selected directly.
        previous = selected.get(key)
        previous_is_fast_token = (
            previous is not None
            and previous.get("decode_api") == "rwkv7_forward_token"
        )
        if (
            previous is None
            or row.get("decode_api") == "rwkv7_forward_token"
            or not previous_is_fast_token
        ):
            selected[key] = row

    failures: list[dict[str, Any]] = []
    for batch, prompt in sorted(requested - set(selected)):
        failures.append({"kind": "missing_shape", "batch_size": batch, "prompt_tokens": prompt})

    latency_by_shape: dict[tuple[int, int], float] = {}
    throughput_by_shape: dict[tuple[int, int], float] = {}
    route_by_shape: dict[tuple[int, int], str] = {}
    graph_shapes: list[list[int]] = []
    dynamic_shapes: list[list[int]] = []
    for key, row in sorted(selected.items()):
        batch, prompt = key
        if row.get("status") not in (None, "pass"):
            failures.append({"kind": "status", "batch_size": batch, "prompt_tokens": prompt, "value": row.get("status")})
        for field in ("greedy_match", "decode_after_prefill_greedy_match"):
            if field in row and row[field] is not True:
                failures.append({"kind": "correctness", "field": field, "batch_size": batch, "prompt_tokens": prompt})
        fused_scan = _first_bool(
            row,
            ("fused_scan_effective", "prefill_fused_scan_effective"),
        )
        if require_fused_scan and fused_scan is not True:
            failures.append({"kind": "route", "field": "fused_scan_effective", "batch_size": batch, "prompt_tokens": prompt, "value": fused_scan})
        if require_safe_fusions:
            for field in (
                "prefill_fused_shift_mix_effective",
                "prefill_fused_state_prep_effective",
                "prefill_fused_output_effective",
            ):
                value = _first_bool(row, (field,))
                if value is not True:
                    failures.append(
                        {
                            "kind": "route",
                            "field": field,
                            "batch_size": batch,
                            "prompt_tokens": prompt,
                            "value": value,
                        }
                    )
        graph = _first_bool(row, ("prefill_graph_effective",))
        (graph_shapes if graph else dynamic_shapes).append([batch, prompt])
        route_by_shape[key] = "graph" if graph else "dynamic"

        latency = _first_number(row, ("native_prefill_ms", "prefill_ms"))
        throughput = _first_number(
            row,
            ("native_prefill_tokps_total", "prefill_tokps_total"),
        )
        if latency is not None and latency > 0:
            latency_by_shape[key] = latency
        if throughput is not None and throughput > 0:
            throughput_by_shape[key] = throughput

    padding_ratios: list[dict[str, Any]] = []
    for prompt in requested_prompts:
        for batch in requested_batches:
            next_batch = batch + 1
            left = latency_by_shape.get((batch, prompt))
            right = latency_by_shape.get((next_batch, prompt))
            if left is None or right is None:
                continue
            ratio = left / right
            item = {
                "batch_size": batch,
                "padded_batch_size": next_batch,
                "prompt_tokens": prompt,
                "latency_ratio": round(ratio, 6),
            }
            padding_ratios.append(item)
            if ratio > max_padding_latency_ratio:
                failures.append({"kind": "padding_cliff", **item})

    # Exact CUDA-Graph rows are intentionally retained as card-local hot
    # shapes.  Compare adjacent lengths across that route boundary with a
    # severe-cliff ceiling, then compare the two dynamic neighbours around the
    # hot shape with the tighter continuity ceiling.  This catches a genuinely
    # broken fallback without treating a validated graph replay bonus as a
    # regression (for example T=127/129 around exact T=128).
    cross_route_boundary_ratios: list[dict[str, Any]] = []
    for batch in requested_batches:
        for prompt in requested_prompts:
            next_prompt = prompt + 1
            left = throughput_by_shape.get((batch, prompt))
            right = throughput_by_shape.get((batch, next_prompt))
            if left is None or right is None:
                continue
            ratio = max(left, right) / min(left, right)
            item = {
                "batch_size": batch,
                "prompt_tokens": prompt,
                "next_prompt_tokens": next_prompt,
                "throughput_ratio": round(ratio, 6),
                "left_route": route_by_shape.get((batch, prompt)),
                "right_route": route_by_shape.get((batch, next_prompt)),
            }
            if item["left_route"] != item["right_route"]:
                cross_route_boundary_ratios.append(item)
                if ratio > max_cross_route_boundary_ratio:
                    failures.append({"kind": "cross_route_boundary_cliff", **item})

    same_route_boundary_ratios: list[dict[str, Any]] = []
    for batch in requested_batches:
        for route in ("dynamic", "graph"):
            route_prompts = sorted(
                prompt
                for prompt in requested_prompts
                if route_by_shape.get((batch, prompt)) == route
                and (batch, prompt) in throughput_by_shape
            )
            for prompt, next_prompt in zip(route_prompts, route_prompts[1:]):
                # Boundary triplets use N-1,N,N+1.  Do not compare unrelated
                # prompt bands such as 129 and 255.
                if next_prompt - prompt > 2:
                    continue
                left = throughput_by_shape[(batch, prompt)]
                right = throughput_by_shape[(batch, next_prompt)]
                ratio = max(left, right) / min(left, right)
                item = {
                    "batch_size": batch,
                    "prompt_tokens": prompt,
                    "next_prompt_tokens": next_prompt,
                    "throughput_ratio": round(ratio, 6),
                    "route": route,
                }
                same_route_boundary_ratios.append(item)
                if ratio > max_boundary_throughput_ratio:
                    failures.append({"kind": "prompt_boundary_cliff", **item})

    return {
        "schema_version": 1,
        "status": "pass" if not failures else "fail",
        "requested_batches": requested_batches,
        "requested_prompts": requested_prompts,
        "requested_shape_count": len(requested),
        "observed_shape_count": len(selected),
        "graph_shapes": graph_shapes,
        "dynamic_shapes": dynamic_shapes,
        "max_padding_latency_ratio": max_padding_latency_ratio,
        "max_boundary_throughput_ratio": max_boundary_throughput_ratio,
        "max_cross_route_boundary_ratio": max_cross_route_boundary_ratio,
        "observed_worst_padding_latency_ratio": round(
            max((item["latency_ratio"] for item in padding_ratios), default=0.0),
            6,
        ),
        "observed_worst_boundary_throughput_ratio": round(
            max((item["throughput_ratio"] for item in same_route_boundary_ratios), default=0.0),
            6,
        ),
        "observed_worst_cross_route_boundary_ratio": round(
            max((item["throughput_ratio"] for item in cross_route_boundary_ratios), default=0.0),
            6,
        ),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--batches", default="1-8")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--max-padding-latency-ratio", type=float, default=1.5)
    parser.add_argument("--max-boundary-throughput-ratio", type=float, default=1.35)
    parser.add_argument("--max-cross-route-boundary-ratio", type=float, default=3.0)
    parser.add_argument("--allow-unfused-scan", action="store_true")
    parser.add_argument("--require-safe-fusions", action="store_true")
    parser.add_argument("--summary", default="")
    args = parser.parse_args()

    summary = analyze(
        load_rows(args.results),
        batches=parse_int_set(args.batches),
        prompts=parse_int_set(args.prompts),
        require_fused_scan=not args.allow_unfused_scan,
        require_safe_fusions=args.require_safe_fusions,
        max_padding_latency_ratio=args.max_padding_latency_ratio,
        max_boundary_throughput_ratio=args.max_boundary_throughput_ratio,
        max_cross_route_boundary_ratio=args.max_cross_route_boundary_ratio,
    )
    payload = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.summary:
        path = Path(args.summary)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
