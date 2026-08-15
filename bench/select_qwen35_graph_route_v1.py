#!/usr/bin/env python3
"""Select one whole-model Qwen StaticCache graph route from paired probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

try:
    from bench.validate_qwen35_best_optimized_hf_v1 import (
        RUNTIME_FIELDS,
        _validate_row,
    )
except ModuleNotFoundError:
    from validate_qwen35_best_optimized_hf_v1 import RUNTIME_FIELDS, _validate_row


INDUCTOR = "static_cache_inductor_cudagraph"
RAW = "static_cache_raw_cudagraph"
SHORT_KEYS = {(1, 128, 128), (8, 128, 128)}
BOUNDARY_KEYS = {(1, 2048, 512), (8, 2048, 512)}


def _read_rows(path: Path) -> tuple[list[dict[str, Any]], str]:
    payload = path.read_bytes()
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if type(row) is not dict:
            raise ValueError(f"{path}:{line_number}: row must be a JSON object")
        row["_source"] = f"{path}:{line_number}"
        rows.append(row)
    return rows, hashlib.sha256(payload).hexdigest()


def _key(row: dict[str, Any]) -> tuple[Any, Any, Any]:
    return row.get("batch_size"), row.get("prompt_tokens"), row.get("decode_tokens")


def _index(
    rows: list[dict[str, Any]],
    *,
    route: str,
    keys: set[tuple[int, int, int]],
    expected_device: str,
    errors: list[str],
    cross_cache_full_greedy_policy: str = "strict",
) -> dict[tuple[int, int, int], dict[str, Any]]:
    for row in rows:
        _validate_row(
            row,
            expected_device,
            errors,
            expected_cross_cache_full_greedy_policy=cross_cache_full_greedy_policy,
        )
        if row.get("qwen_decode_optimization_effective") != route:
            errors.append(f"{row.get('_source', '<row>')}: route must be {route!r}")
    observed = [_key(row) for row in rows]
    if (
        len(rows) != len(keys)
        or set(observed) != keys
        or len(set(observed)) != len(rows)
    ):
        errors.append(f"{route}: expected exactly {sorted(keys)!r}, got {observed!r}")
    return {key: row for row in rows if (key := _key(row)) in keys}


def _signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("model_pair"),
        row.get("model_size_label"),
        row.get("model_id_or_path"),
        row.get("benchmark_repository_commit"),
        *(row.get(field) for field in RUNTIME_FIELDS),
    )


def select_route(
    inductor_short: list[dict[str, Any]],
    raw_short: list[dict[str, Any]],
    *,
    expected_device: str,
    inductor_boundary: list[dict[str, Any]] | None = None,
    raw_boundary: list[dict[str, Any]] | None = None,
    cross_cache_full_greedy_policy: str = "strict",
) -> dict[str, Any]:
    errors: list[str] = []
    route_errors: dict[str, list[str]] = {INDUCTOR: [], RAW: []}
    ind_short = _index(
        inductor_short,
        route=INDUCTOR,
        keys=SHORT_KEYS,
        expected_device=expected_device,
        errors=route_errors[INDUCTOR],
        cross_cache_full_greedy_policy=cross_cache_full_greedy_policy,
    )
    raw_short_index = _index(
        raw_short,
        route=RAW,
        keys=SHORT_KEYS,
        expected_device=expected_device,
        errors=route_errors[RAW],
        cross_cache_full_greedy_policy=cross_cache_full_greedy_policy,
    )
    has_boundary = inductor_boundary is not None or raw_boundary is not None
    if has_boundary and (inductor_boundary is None or raw_boundary is None):
        errors.append("boundary evidence must provide both Inductor and raw routes")
    ind_boundary: dict[tuple[int, int, int], dict[str, Any]] = {}
    raw_boundary_index: dict[tuple[int, int, int], dict[str, Any]] = {}
    if inductor_boundary is not None and raw_boundary is not None:
        ind_boundary = _index(
            inductor_boundary,
            route=INDUCTOR,
            keys=BOUNDARY_KEYS,
            expected_device=expected_device,
            errors=route_errors[INDUCTOR],
            cross_cache_full_greedy_policy=cross_cache_full_greedy_policy,
        )
        raw_boundary_index = _index(
            raw_boundary,
            route=RAW,
            keys=BOUNDARY_KEYS,
            expected_device=expected_device,
            errors=route_errors[RAW],
            cross_cache_full_greedy_policy=cross_cache_full_greedy_policy,
        )

    all_rows = [*inductor_short, *raw_short]
    if inductor_boundary:
        all_rows.extend(inductor_boundary)
    if raw_boundary:
        all_rows.extend(raw_boundary)
    valid_routes = [route for route, failures in route_errors.items() if not failures]
    signature_rows = all_rows
    if len(valid_routes) == 1:
        selected_valid_route = valid_routes[0]
        signature_rows = [
            row
            for row in all_rows
            if row.get("qwen_decode_optimization_requested") == selected_valid_route
        ]
        valid_signature = _signature(signature_rows[0]) if signature_rows else ()
        for row in all_rows:
            if row in signature_rows:
                continue
            for index, label in ((0, "model_pair"), (3, "repository commit")):
                value = _signature(row)[index]
                if value not in (None, "") and value != valid_signature[index]:
                    errors.append(
                        f"disqualified route {label} does not match the valid route"
                    )
    signatures = {_signature(row) for row in signature_rows}
    if len(signatures) != 1:
        errors.append(
            "all route probes must use one model, commit, and runtime signature"
        )

    ratios: list[dict[str, Any]] = []
    for stage, left, right in (
        ("short", ind_short, raw_short_index),
        ("boundary", ind_boundary, raw_boundary_index),
    ):
        for key in sorted(set(left) & set(right)):
            ind_rate = left[key].get("decode_tokps_total_raw")
            raw_rate = right[key].get("decode_tokps_total_raw")
            if (
                type(ind_rate) not in {int, float}
                or type(raw_rate) not in {int, float}
                or not math.isfinite(float(ind_rate))
                or not math.isfinite(float(raw_rate))
                or float(ind_rate) <= 0
                or float(raw_rate) <= 0
            ):
                if not route_errors[INDUCTOR] and not route_errors[RAW]:
                    errors.append(f"{stage} {key}: invalid raw Decode throughput")
                continue
            ratios.append(
                {
                    "stage": stage,
                    "batch_size": key[0],
                    "prompt_tokens": key[1],
                    "decode_tokens": key[2],
                    "inductor_decode_tokps_total_raw": float(ind_rate),
                    "raw_decode_tokps_total_raw": float(raw_rate),
                    "inductor_over_raw": float(ind_rate) / float(raw_rate),
                }
            )

    selected: str | None = None
    decision = "fail" if errors else "needs_boundary"
    if not errors and len(valid_routes) == 1:
        selected = valid_routes[0]
        decision = "selected_alternate_after_correctness_failure"
    elif not errors and not valid_routes:
        errors.append("both graph routes failed correctness or coverage validation")
        decision = "fail"
    short_ratios = [
        item["inductor_over_raw"] for item in ratios if item["stage"] == "short"
    ]
    if not errors and selected is None and len(short_ratios) == 2:
        if all(value >= 1.05 for value in short_ratios):
            selected, decision = INDUCTOR, "selected_unanimous_short_5pct"
        elif all(value <= 1 / 1.05 for value in short_ratios):
            selected, decision = RAW, "selected_unanimous_short_5pct"
        elif has_boundary:
            all_ratios = [item["inductor_over_raw"] for item in ratios]
            if len(all_ratios) == 4:
                geometric_mean = math.prod(all_ratios) ** (1 / len(all_ratios))
                if geometric_mean >= 1.02:
                    selected, decision = INDUCTOR, "selected_four_cell_geomean_2pct"
                elif geometric_mean <= 1 / 1.02:
                    selected, decision = RAW, "selected_four_cell_geomean_2pct"
                else:
                    decision = "needs_full_route_matrix"

    return {
        "schema_version": 1,
        "status": "pass" if selected is not None and not errors else decision,
        "expected_device": expected_device,
        "selection_policy": {
            "whole_model_single_route": True,
            "short_unanimous_margin": 0.05,
            "four_cell_geometric_mean_margin": 0.02,
            "per_cell_route_mixing": False,
            "cross_cache_full_greedy_policy": cross_cache_full_greedy_policy,
        },
        "selected_route": selected,
        "decision": decision,
        "ratios": ratios,
        "route_errors": route_errors,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inductor-short", type=Path, required=True)
    parser.add_argument("--raw-short", type=Path, required=True)
    parser.add_argument("--inductor-boundary", type=Path)
    parser.add_argument("--raw-boundary", type=Path)
    parser.add_argument("--expected-device", default="NVIDIA GeForce RTX 4080")
    parser.add_argument(
        "--cross-cache-full-greedy-policy",
        choices=["strict", "informational"],
        default="strict",
    )
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inductor_short, ind_short_sha = _read_rows(args.inductor_short)
    raw_short, raw_short_sha = _read_rows(args.raw_short)
    ind_boundary = raw_boundary = None
    hashes = {
        "inductor_short": ind_short_sha,
        "raw_short": raw_short_sha,
    }
    if args.inductor_boundary is not None:
        ind_boundary, hashes["inductor_boundary"] = _read_rows(args.inductor_boundary)
    if args.raw_boundary is not None:
        raw_boundary, hashes["raw_boundary"] = _read_rows(args.raw_boundary)
    summary = select_route(
        inductor_short,
        raw_short,
        expected_device=args.expected_device,
        inductor_boundary=ind_boundary,
        raw_boundary=raw_boundary,
        cross_cache_full_greedy_policy=args.cross_cache_full_greedy_policy,
    )
    summary["artifact_sha256"] = hashes
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
