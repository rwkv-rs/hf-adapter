#!/usr/bin/env python3
"""Render sorted raw Qwen Prefill/Decode results for the optimized HF lane."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

try:
    from bench.validate_qwen35_best_optimized_hf_v1 import (
        PAIR_RANK,
        read_rows,
        validate_matrix,
    )
except ModuleNotFoundError:
    from validate_qwen35_best_optimized_hf_v1 import (
        PAIR_RANK,
        read_rows,
        validate_matrix,
    )


def display_rate(value: float) -> str:
    """Apply the report-only threshold without modifying machine-readable data."""

    return f"{value:,.0f}" if value >= 100 else f"{value:,.1f}"


def ordered_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            PAIR_RANK.get(str(row.get("model_pair")), 999),
            str(row.get("device", "")),
            int(row.get("batch_size", 0)),
            int(row.get("prompt_tokens", 0)),
            int(row.get("decode_tokens", 0)),
        ),
    )


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = ordered_rows(rows)
    cross_cache_policies = sorted(
        {
            str(row.get("qwen_cross_cache_full_greedy_policy_effective", "strict"))
            for row in ordered
        }
    )
    cross_cache_policy = (
        cross_cache_policies[0] if len(cross_cache_policies) == 1 else "mixed"
    )
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in ordered:
        key = (
            str(row["model_size_label"]),
            str(row["device"]),
            int(row["batch_size"]),
        )
        groups.setdefault(key, []).append(row)
    medians = [
        {
            "model_size_label": model,
            "device": device,
            "batch_size": batch,
            "cells": len(group),
            "decode_route": str(group[0]["qwen_decode_optimization_effective"]),
            "cross_cache_full_greedy_policy": str(
                group[0]["qwen_cross_cache_full_greedy_policy_effective"]
            ),
            "dynamic_static_full_greedy_mismatch_cells": sum(
                int(row["qwen_dynamic_static_full_greedy_mismatch_count"]) > 0
                for row in group
            ),
            "dynamic_candidate_full_greedy_mismatch_cells": sum(
                int(row["qwen_dynamic_candidate_full_greedy_mismatch_count"]) > 0
                for row in group
            ),
            "prefill_tokps_median": statistics.median(
                float(row["prefill_tokps_total_raw"]) for row in group
            ),
            "decode_tokps_median": statistics.median(
                float(row["decode_tokps_total_raw"]) for row in group
            ),
        }
        for (model, device, batch), group in groups.items()
    ]
    correctness_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in ordered:
        key = (str(row["model_size_label"]), str(row["device"]))
        correctness_groups.setdefault(key, []).append(row)
    model_correctness = [
        {
            "model_size_label": model,
            "device": device,
            "cells": len(group),
            "decode_route": str(group[0]["qwen_decode_optimization_effective"]),
            "cross_cache_full_greedy_policy": str(
                group[0]["qwen_cross_cache_full_greedy_policy_effective"]
            ),
            "dynamic_static_full_greedy_mismatch_cells": sum(
                int(row["qwen_dynamic_static_full_greedy_mismatch_count"]) > 0
                for row in group
            ),
            "dynamic_candidate_full_greedy_mismatch_cells": sum(
                int(row["qwen_dynamic_candidate_full_greedy_mismatch_count"]) > 0
                for row in group
            ),
            "same_cache_logits_min_cosine": min(
                float(row["qwen_same_cache_logits_min_cosine"]) for row in group
            ),
            "dynamic_static_logits_min_cosine": min(
                float(row["qwen_dynamic_static_logits_min_cosine"]) for row in group
            ),
            "dynamic_candidate_logits_min_cosine": min(
                float(row["qwen_graph_logits_min_cosine"]) for row in group
            ),
        }
        for (model, device), group in correctness_groups.items()
    ]
    return {
        "schema_version": 2,
        "benchmark_matrix": "qwen35_best_optimized_hf_v1",
        "reference_lane_eligible": True,
        "unified_main_table_eligible": False,
        "unified_main_table_reason": (
            "RWKV candidate rows were not rerun under this runtime, and the "
            "Qwen numbers are an independent best-Prefill/best-Decode axis envelope"
        ),
        "axis_composition": "independent_best_prefill_and_decode",
        "interpretation": {
            "scope": "qwen_reference_only",
            "prefill": "official FLA plus causal_conv1d eager DynamicCache",
            "decode": "fixed per-model StaticCache CUDA Graph route",
            "continuous_end_to_end_path": False,
            "raw_cudagraph_origin": (
                "repository benchmark optimization, not an official Qwen Graph path"
            ),
        },
        "sort_order": ["model_size", "gpu", "batch", "prompt", "decode"],
        "display_rounding": {">=100": 0, "<100": 1},
        "correctness_contract": {
            "same_cache": {
                "comparison": "StaticCache eager vs candidate graph route",
                "hard_gates": [
                    "finite logits trace",
                    "full-horizon greedy match",
                    "minimum cosine >= 0.9999",
                ],
                "cosine_threshold": 0.9999,
            },
            "cross_cache": {
                "comparisons": [
                    "DynamicCache eager vs StaticCache eager",
                    "DynamicCache eager vs candidate graph route",
                ],
                "hard_gates": [
                    "finite logits traces",
                    (
                        "full-horizon greedy match"
                        if cross_cache_policy == "strict"
                        else "short logits-trace greedy match"
                    ),
                    "prefill-next-token match",
                ],
                "full_horizon_greedy_policy": cross_cache_policy,
                "full_horizon_greedy_interpretation": (
                    "hard_gate"
                    if cross_cache_policy == "strict"
                    else "informational_only"
                ),
                "cosine_threshold": None,
                "cosine_interpretation": "informational_only",
            },
        },
        "rows": len(ordered),
        "decode_routes_by_model": {
            model: sorted(
                {
                    str(row["qwen_decode_optimization_effective"])
                    for row in ordered
                    if str(row["model_size_label"]) == model
                }
            )
            for model in sorted({str(row["model_size_label"]) for row in ordered})
        },
        "model_correctness": model_correctness,
        "model_batch_medians": medians,
        "cells": [
            {
                "model_pair": row["model_pair"],
                "model_size_label": row["model_size_label"],
                "device": row["device"],
                "batch_size": row["batch_size"],
                "prompt_tokens": row["prompt_tokens"],
                "decode_tokens": row["decode_tokens"],
                "qwen_decode_optimization_effective": row[
                    "qwen_decode_optimization_effective"
                ],
                "step_backend": row["step_backend"],
                "cache_type": row["cache_type"],
                "qwen_cross_cache_full_greedy_policy_effective": row[
                    "qwen_cross_cache_full_greedy_policy_effective"
                ],
                "qwen_dynamic_static_full_greedy_mismatch_count": row[
                    "qwen_dynamic_static_full_greedy_mismatch_count"
                ],
                "qwen_dynamic_static_full_greedy_first_mismatch_index": row[
                    "qwen_dynamic_static_full_greedy_first_mismatch_index"
                ],
                "qwen_dynamic_candidate_full_greedy_mismatch_count": row[
                    "qwen_dynamic_candidate_full_greedy_mismatch_count"
                ],
                "qwen_dynamic_candidate_full_greedy_first_mismatch_index": row[
                    "qwen_dynamic_candidate_full_greedy_first_mismatch_index"
                ],
                "prefill_tokps_total": row["prefill_tokps_total"],
                "decode_tokps_total": row["decode_tokps_total"],
                "prefill_tokps_total_raw": row["prefill_tokps_total_raw"],
                "decode_tokps_total_raw": row["decode_tokps_total_raw"],
                "prefill_sec_samples": row["prefill_sec_samples"],
                "decode_sec_samples": row["decode_sec_samples"],
            }
            for row in ordered
        ],
    }


def render_markdown(summary: dict[str, Any]) -> str:
    devices = sorted(
        {str(row["device"]) for row in summary.get("model_correctness", [])}
    )
    title_device = devices[0] if len(devices) == 1 else "Card-local"
    cross_cache_policy = summary["correctness_contract"]["cross_cache"][
        "full_horizon_greedy_policy"
    ]
    cross_cache_gate = (
        "full-horizon greedy-token equality"
        if cross_cache_policy == "strict"
        else "short logits-trace greedy-token equality; full-horizon greedy is retained as informational telemetry"
    )
    lines = [
        f"# {title_device} Qwen3.5 best-optimized HF raw results",
        "",
        "Status: **PASS, reference-only**. This Qwen-only artifact is not "
        "eligible for the unified RWKV/Qwen main table because no same-runtime "
        "RWKV candidate was measured.",
        "",
        "Prefill and Decode form an `independent_best_prefill_and_decode` "
        "performance envelope. They are not one continuous end-to-end request, "
        "TTFT result, or DynamicCache-to-StaticCache handoff. Prefill uses the "
        "official FLA plus `causal_conv1d` eager DynamicCache path. Raw CUDA "
        "Graph, where selected, is a repository benchmark optimization around "
        "the official Qwen operators, not an official Qwen Graph path.",
        "",
        "Rows are sorted by model size, GPU, B1/B8, prompt and decode. "
        "Display values use 0 decimals at >=100 tok/s and 1 decimal below 100; "
        "JSONL and JSON retain the original numeric values and all seven samples.",
        "",
        "## Correctness contract",
        "",
        "**Same-cache hard gate:** StaticCache eager vs the candidate graph route "
        "must have finite logits, full-horizon greedy-token equality, and minimum "
        "cosine >= 0.9999.",
        "",
        "**Cross-cache hard gates:** DynamicCache eager vs StaticCache eager and "
        "DynamicCache eager vs the candidate graph route must have finite logits, "
        f"{cross_cache_gate}, and prefill-next-token equality. "
        "Cross-cache cosine is informational only and has no acceptance threshold.",
        "",
        "| Qwen3.5 | GPU | Route | Cells | Same-cache min cosine | Dynamic/Static min cosine | Dynamic/Candidate min cosine | D/S full-greedy mismatch cells | D/C full-greedy mismatch cells |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["model_correctness"]:
        lines.append(
            "| {model_size_label} | {device} | {decode_route} | {cells} | "
            "{same:.6f} | {dynamic_static:.6f} | {dynamic_candidate:.6f} | "
            "{dynamic_static_full_greedy_mismatch_cells} | "
            "{dynamic_candidate_full_greedy_mismatch_cells} |".format(
                **row,
                same=float(row["same_cache_logits_min_cosine"]),
                dynamic_static=float(row["dynamic_static_logits_min_cosine"]),
                dynamic_candidate=float(row["dynamic_candidate_logits_min_cosine"]),
            )
        )
    lines.extend(
        [
            "",
            "## Model / batch medians",
            "",
            "| Qwen3.5 | GPU | Batch | Decode route | Cells | Prefill tok/s | Decode tok/s |",
            "|---|---|---:|---|---:|---:|---:|",
        ]
    )
    for row in summary["model_batch_medians"]:
        lines.append(
            "| {model_size_label} | {device} | B{batch_size} | {decode_route} | {cells} | {prefill} | {decode} |".format(
                **row,
                prefill=display_rate(float(row["prefill_tokps_median"])),
                decode=display_rate(float(row["decode_tokps_median"])),
            )
        )
    lines.extend(
        [
            "",
            "## Complete 48-cell raw matrix",
            "",
            "| Qwen3.5 | GPU | Batch | Prompt | Decode | Route | Prefill tok/s | Decode tok/s |",
            "|---|---|---:|---:|---:|---|---:|---:|",
        ]
    )
    for row in summary["cells"]:
        lines.append(
            "| {model_size_label} | {device} | B{batch_size} | {prompt_tokens} | "
            "{decode_tokens} | {qwen_decode_optimization_effective} | {prefill} | {decode} |".format(
                **row,
                prefill=display_rate(float(row["prefill_tokps_total_raw"])),
                decode=display_rate(float(row["decode_tokps_total_raw"])),
            )
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--expected-device", default="")
    parser.add_argument(
        "--cross-cache-full-greedy-policy",
        choices=["strict", "informational"],
        default="strict",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_rows(args.results)
    validation = validate_matrix(
        rows,
        expected_device=args.expected_device,
        expected_cross_cache_full_greedy_policy=args.cross_cache_full_greedy_policy,
    )
    if validation["status"] != "pass":
        print(
            "QWEN35_BEST_OPTIMIZED_SUMMARY "
            + json.dumps(
                {"status": "fail", "errors": validation["errors"]},
                ensure_ascii=False,
            )
        )
        return 1
    summary = build_summary(rows)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.markdown.write_text(render_markdown(summary), encoding="utf-8")
    print(
        "QWEN35_BEST_OPTIMIZED_SUMMARY "
        + json.dumps({"status": "pass", "rows": summary["rows"]})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
