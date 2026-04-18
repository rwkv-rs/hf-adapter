#!/usr/bin/env python3
"""Gate every RTX 4080 parameter-adjusted Prefill/Decode matrix cell."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any


PAIRS = (
    "rwkv-0.4b__qwen3.5-0.8b",
    "rwkv-1.5b__qwen3.5-2b",
    "rwkv-2.9b__qwen3.5-4b",
)
EXPECTED_DEVICE = "NVIDIA GeForce RTX 4080"
PARAMETERS = {
    "rwkv-0.4b__qwen3.5-0.8b": (450_767_872, 752_393_024),
    "rwkv-1.5b__qwen3.5-2b": (1_527_404_544, 1_881_825_088),
    "rwkv-2.9b__qwen3.5-4b": (2_947_735_040, 4_205_751_296),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def shape(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(row["batch_size"]),
        int(row["prompt_tokens"]),
        int(row["decode_tokens"]),
    )


def rounded(value: float) -> float:
    return round(float(value), 6)


def summarize(
    candidate_path: Path,
    reference_path: Path,
    *,
    gate: float = 1.0,
) -> dict[str, Any]:
    candidates = load_jsonl(candidate_path)
    references = load_jsonl(reference_path)
    errors: list[str] = []
    groups: list[dict[str, Any]] = []
    if len(candidates) != 36:
        errors.append(f"candidate matrix has {len(candidates)} rows, expected 36")
    if len(references) != 36:
        errors.append(f"reference matrix has {len(references)} rows, expected 36")

    for pair in PAIRS:
        for batch_size in (1, 8):
            expected = {
                (batch_size, prompt, decode)
                for prompt in (128, 512, 2048)
                for decode in (128, 512)
            }
            candidate_rows = [
                row
                for row in candidates
                if row.get("model_pair") == pair
                and row.get("model_role") == "candidate"
                and int(row.get("batch_size", -1)) == batch_size
            ]
            reference_rows = [
                row
                for row in references
                if row.get("model_pair") == pair
                and row.get("model_role") == "reference"
                and int(row.get("batch_size", -1)) == batch_size
            ]
            cand = {shape(row): row for row in candidate_rows}
            ref = {shape(row): row for row in reference_rows}
            if len(candidate_rows) != 6 or set(cand) != expected:
                errors.append(f"{pair} B{batch_size}: candidate coverage is not six cells")
                continue
            if len(reference_rows) != 6 or set(ref) != expected:
                errors.append(f"{pair} B{batch_size}: reference coverage is not six cells")
                continue

            candidate_parameters, reference_parameters = PARAMETERS[pair]
            if {
                int(row["active_parameter_count"]) for row in candidate_rows
            } != {candidate_parameters}:
                errors.append(f"{pair} B{batch_size}: candidate parameter count drifted")
            if {
                int(row["active_parameter_count"]) for row in reference_rows
            } != {reference_parameters}:
                errors.append(f"{pair} B{batch_size}: reference parameter count drifted")

            raw_prefill: list[float] = []
            raw_decode: list[float] = []
            adjusted_prefill: list[float] = []
            adjusted_decode: list[float] = []
            raw_e2e: list[float] = []
            adjusted_e2e: list[float] = []
            cells: list[dict[str, Any]] = []
            param_ratio: float | None = None
            for key in sorted(expected):
                c = cand[key]
                r = ref[key]
                for label, row in (("candidate", c), ("reference", r)):
                    if row.get("device") != EXPECTED_DEVICE:
                        errors.append(f"{pair} B{batch_size} {key}: {label} is not RTX 4080")
                    if row.get("dtype") != "fp16" or row.get("status") != "pass":
                        errors.append(f"{pair} B{batch_size} {key}: {label} failed FP16 contract")
                    if row.get("logits_finite") is not True:
                        errors.append(f"{pair} B{batch_size} {key}: {label} logits are not finite")
                if r.get("qwen_fast_path_verified") is not True:
                    errors.append(
                        f"{pair} B{batch_size} {key}: Qwen full-FLA path is unverified"
                    )
                if r.get("effective_backend") != "qwen_fla_gated_delta_rule":
                    errors.append(
                        f"{pair} B{batch_size} {key}: Qwen effective backend drifted"
                    )
                ratio = float(c["active_parameter_count"]) / float(
                    r["active_parameter_count"]
                )
                if param_ratio is None:
                    param_ratio = ratio
                elif abs(param_ratio - ratio) > 1e-12:
                    errors.append(f"{pair} B{batch_size}: parameter ratio changed between cells")
                p = float(c["prefill_tokps_total"]) / float(r["prefill_tokps_total"])
                d = float(c["decode_tokps_total"]) / float(r["decode_tokps_total"])
                tokens_p = key[0] * key[1]
                tokens_d = key[0] * key[2]
                candidate_time = (
                    tokens_p / float(c["prefill_tokps_total"])
                    + tokens_d / float(c["decode_tokps_total"])
                )
                reference_time = (
                    tokens_p / float(r["prefill_tokps_total"])
                    + tokens_d / float(r["decode_tokps_total"])
                )
                e2e = reference_time / candidate_time
                raw_prefill.append(p)
                raw_decode.append(d)
                adjusted_prefill.append(p * ratio)
                adjusted_decode.append(d * ratio)
                raw_e2e.append(e2e)
                adjusted_e2e.append(e2e * ratio)
                cells.append(
                    {
                        "shape": list(key),
                        "raw_prefill_ratio": rounded(p),
                        "raw_decode_ratio": rounded(d),
                        "adjusted_prefill_ratio": rounded(p * ratio),
                        "adjusted_decode_ratio": rounded(d * ratio),
                    }
                )

            assert param_ratio is not None
            p_med = median(adjusted_prefill)
            d_med = median(adjusted_decode)
            p_min = min(adjusted_prefill)
            d_min = min(adjusted_decode)
            passed = p_min > gate and d_min > gate
            if not passed:
                errors.append(
                    f"{pair} B{batch_size}: adjusted P/D cell minima "
                    f"{p_min:.4f}/{d_min:.4f} are not both > {gate:.4f}"
                )
            groups.append(
                {
                    "model_pair": pair,
                    "batch_size": batch_size,
                    "cells": 6,
                    "candidate_active_parameters": int(next(iter(cand.values()))["active_parameter_count"]),
                    "reference_active_parameters": int(next(iter(ref.values()))["active_parameter_count"]),
                    "active_parameter_ratio": rounded(param_ratio),
                    "raw_prefill_median": rounded(median(raw_prefill)),
                    "raw_decode_median": rounded(median(raw_decode)),
                    "adjusted_prefill_median": rounded(p_med),
                    "adjusted_decode_median": rounded(d_med),
                    "adjusted_prefill_min": rounded(p_min),
                    "adjusted_decode_min": rounded(d_min),
                    "adjusted_prefill_cells_passed": sum(
                        value > gate for value in adjusted_prefill
                    ),
                    "adjusted_decode_cells_passed": sum(
                        value > gate for value in adjusted_decode
                    ),
                    "raw_e2e_median": rounded(median(raw_e2e)),
                    "adjusted_e2e_median": rounded(median(adjusted_e2e)),
                    "adjusted_pd_pass": passed,
                    "candidate_runtime": {
                        "torch": next(iter(cand.values())).get("torch_version"),
                        "cuda": next(iter(cand.values())).get("torch_cuda_version"),
                        "triton": next(iter(cand.values())).get("triton_version"),
                        "transformers": next(iter(cand.values())).get("transformers_version"),
                    },
                    "reference_runtime": {
                        "torch": next(iter(ref.values())).get("torch_version"),
                        "cuda": next(iter(ref.values())).get("torch_cuda_version"),
                        "triton": next(iter(ref.values())).get("triton_version"),
                        "transformers": next(iter(ref.values())).get("transformers_version"),
                        "fla": next(iter(ref.values())).get("fla_version"),
                        "causal_conv1d": next(iter(ref.values())).get("causal_conv1d_version"),
                    },
                    "cell_ratios": cells,
                }
            )

    prefill_cells_passed = sum(
        int(group["adjusted_prefill_cells_passed"]) for group in groups
    )
    decode_cells_passed = sum(
        int(group["adjusted_decode_cells_passed"]) for group in groups
    )
    return {
        "axis": "rtx4080_parameter_adjusted_pd",
        "status": "pass" if not errors else "fail",
        "gate": "every adjusted Prefill cell > 1.0 and every adjusted Decode cell > 1.0",
        "formula": "raw_speed_ratio * candidate_active_parameters / reference_active_parameters",
        "adjusted_prefill_cells_passed": prefill_cells_passed,
        "adjusted_decode_cells_passed": decode_cells_passed,
        "cells_total": 36,
        "groups": groups,
        "errors": errors,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# RTX 4080 parameter-adjusted Prefill/Decode",
        "",
        f"Status: **{report['status']}**",
        "",
        "| Pair | Batch | Raw P / D median | Adjusted P / D median | Adjusted P / D minimum | E2E raw / adjusted |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["groups"]:
        pair = row["model_pair"].replace("rwkv-", "").replace("__qwen3.5-", " / ")
        lines.append(
            f"| {pair} | B{row['batch_size']} | "
            f"**{row['raw_prefill_median']:.2f}x / {row['raw_decode_median']:.2f}x** | "
            f"**{row['adjusted_prefill_median']:.2f}x / {row['adjusted_decode_median']:.2f}x** | "
            f"**{row['adjusted_prefill_min']:.2f}x / {row['adjusted_decode_min']:.2f}x** | "
            f"**{row['raw_e2e_median']:.2f}x / {row['adjusted_e2e_median']:.2f}x** |"
        )
    if report["errors"]:
        lines.extend(["", "## Errors", "", *[f"- {value}" for value in report["errors"]]])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--gate", type=float, default=1.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    report = summarize(args.candidate, args.reference, gate=args.gate)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.write_text(markdown(report), encoding="utf-8")
    print(text, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
