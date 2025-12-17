#!/usr/bin/env python3
"""Fail-closed summary for the exact RTX 4080 bsz8 acceptance matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_SHAPES = {
    (8, prompt, decode)
    for prompt in (128, 512, 2048)
    for decode in (128, 512)
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(value)
    return rows


def shape(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(row.get("batch_size", -1)),
        int(row.get("prompt_tokens", -1)),
        int(row.get("decode_tokens", -1)),
    )


def ratio(numerator: Any, denominator: Any) -> float:
    return float(numerator) / float(denominator)


def metric_range(values: list[float]) -> dict[str, float | None]:
    return {
        "min": round(min(values), 6) if values else None,
        "max": round(max(values), 6) if values else None,
    }


def summarize(
    root: Path,
    *,
    min_dense_prefill: float = 1.0,
    min_dense_decode: float = 1.4,
    min_quant_speed: float = 1.0,
    min_quant_cosine: float = 0.999,
) -> dict[str, Any]:
    errors: list[str] = []

    def read_required(name: str) -> list[dict[str, Any]]:
        path = root / name
        if not path.is_file():
            errors.append(f"missing {name}")
            return []
        try:
            return load_jsonl(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid {name}: {exc}")
            return []

    dense_rows = read_required("dense.jsonl")
    memory_rows = read_required("memory.jsonl")
    quant_rows = read_required("paired_quant.jsonl")
    candidates = [row for row in dense_rows if row.get("model_role") == "candidate"]
    references = [row for row in dense_rows if row.get("model_role") == "reference"]

    for label, rows in (("dense candidate", candidates), ("Qwen reference", references)):
        if len(rows) != 6 or {shape(row) for row in rows} != EXPECTED_SHAPES:
            errors.append(f"{label} coverage is not the exact 6-cell bsz8 matrix")

    reference_by_shape = {shape(row): row for row in references}
    dense_by_shape = {shape(row): row for row in candidates}
    dense_prefill: list[float] = []
    dense_decode: list[float] = []
    dense_active_decode: list[float] = []
    for row in candidates:
        key = shape(row)
        reference = reference_by_shape.get(key)
        if reference is None:
            continue
        if row.get("status") != "pass" or row.get("logits_finite") is not True:
            errors.append(f"dense candidate {key} failed status/finite-logits")
        if row.get("effective_backend") != "native_graph":
            errors.append(f"dense candidate {key} did not use native_graph decode")
        if row.get("prefill_effective_backend") != "native_prefill_graph":
            errors.append(f"dense candidate {key} did not use native_prefill_graph")
        if reference.get("status") != "pass" or reference.get("logits_finite") is not True:
            errors.append(f"Qwen reference {key} failed status/finite-logits")
        if reference.get("qwen_backend_requested") != "fla":
            errors.append(f"Qwen reference {key} did not request FLA")
        if reference.get("qwen_full_fused_contract_pass") is not True:
            errors.append(f"Qwen reference {key} failed the full-FLA operator contract")
        if reference.get("qwen_fast_path_verified") is not True:
            errors.append(f"Qwen reference {key} failed live operator binding verification")

        prefill_value = ratio(row["prefill_tokps_total"], reference["prefill_tokps_total"])
        decode_value = ratio(row["decode_tokps_total"], reference["decode_tokps_total"])
        active_value = ratio(
            row["decode_tokps_per_active_billion"],
            reference["decode_tokps_per_active_billion"],
        )
        dense_prefill.append(prefill_value)
        dense_decode.append(decode_value)
        dense_active_decode.append(active_value)
        if prefill_value < min_dense_prefill:
            errors.append(f"dense candidate {key} prefill ratio {prefill_value:.4f} < {min_dense_prefill:.4f}")
        if decode_value < min_dense_decode:
            errors.append(f"dense candidate {key} decode ratio {decode_value:.4f} < {min_dense_decode:.4f}")
        if active_value < 1.0:
            errors.append(f"dense candidate {key} active-work decode ratio {active_value:.4f} < 1.0")

    memory_summary: dict[str, Any] = {}
    for quantization in ("bnb8", "bnb4"):
        rows = [row for row in memory_rows if row.get("quantization") == quantization]
        footprints: list[float] = []
        if len(rows) != 6 or {shape(row) for row in rows} != EXPECTED_SHAPES:
            errors.append(f"{quantization} coverage is not the exact 6-cell bsz8 matrix")
        for row in rows:
            key = shape(row)
            dense = dense_by_shape.get(key)
            if dense is None:
                continue
            if row.get("status") != "pass" or row.get("logits_finite") is not True:
                errors.append(f"{quantization} {key} failed status/finite-logits")
            footprint = ratio(row["model_footprint_mb"], dense["model_footprint_mb"])
            footprints.append(footprint)
            if footprint >= 1.0:
                errors.append(f"{quantization} {key} footprint ratio {footprint:.4f} is not lower")
        memory_summary[quantization] = {
            "rows": len(rows),
            "footprint_ratio_vs_fp16": metric_range(footprints),
        }

    quant_summary: dict[str, Any] = {}
    for quantization in ("a8w8", "torchao_w4"):
        rows = [row for row in quant_rows if row.get("quantization") == quantization]
        prefills: list[float] = []
        decodes: list[float] = []
        totals: list[float] = []
        footprints: list[float] = []
        cosines: list[float] = []
        if len(rows) != 6 or {shape(row) for row in rows} != EXPECTED_SHAPES:
            errors.append(f"{quantization} paired coverage is not the exact 6-cell bsz8 matrix")
        for row in rows:
            key = shape(row)
            if row.get("status") != "pass" or row.get("paired_baseline") is not True:
                errors.append(f"{quantization} {key} is not a passing paired-baseline row")
            if row.get("same_greedy_tokens_as_fp16") is not True:
                errors.append(f"{quantization} {key} greedy tokens differ from fp16")
            prefill_value = float(row.get("prefill_speed_ratio_vs_fp16") or 0.0)
            decode_value = float(row.get("decode_speed_ratio_vs_fp16") or 0.0)
            baseline_prefill_tokps = float(row.get("baseline_prefill_tokps_total") or 0.0)
            baseline_decode_tokps = float(row.get("baseline_decode_tokps_total") or 0.0)
            quant_prefill_tokps = float(row.get("prefill_tokps_total") or 0.0)
            quant_decode_tokps = float(row.get("decode_tokps_total") or 0.0)
            total_value = 0.0
            if min(
                baseline_prefill_tokps,
                baseline_decode_tokps,
                quant_prefill_tokps,
                quant_decode_tokps,
            ) <= 0.0:
                errors.append(f"{quantization} {key} is missing positive paired throughput")
            else:
                prefill_tokens = int(key[0]) * int(key[1])
                decode_tokens = int(key[0]) * int(key[2])
                baseline_total = (
                    prefill_tokens / baseline_prefill_tokps
                    + decode_tokens / baseline_decode_tokps
                )
                quant_total = (
                    prefill_tokens / quant_prefill_tokps
                    + decode_tokens / quant_decode_tokps
                )
                total_value = baseline_total / quant_total
            footprint = float(row.get("footprint_ratio_vs_fp16") or 1.0)
            cosine = min(
                float(row.get("prompt_logits_cos_vs_fp16") or 0.0),
                float(row.get("final_logits_cos_vs_fp16") or 0.0),
            )
            prefills.append(prefill_value)
            decodes.append(decode_value)
            totals.append(total_value)
            footprints.append(footprint)
            cosines.append(cosine)
            # Match the promoted RTX 3090/4090 quant contract: cached decode
            # and complete-cell latency are gated. Prefill remains visible
            # telemetry, but it is not an independent rejection axis.
            if total_value < min_quant_speed or decode_value < min_quant_speed:
                errors.append(
                    f"{quantization} {key} total/decode speed ratios "
                    f"{total_value:.4f}/{decode_value:.4f} "
                    f"are below {min_quant_speed:.4f}"
                )
            if footprint >= 1.0:
                errors.append(f"{quantization} {key} footprint ratio {footprint:.4f} is not lower")
            if cosine < min_quant_cosine:
                errors.append(f"{quantization} {key} minimum logits cosine {cosine:.8f} is too low")
        quant_summary[quantization] = {
            "rows": len(rows),
            "prefill_speed_ratio_vs_fp16": metric_range(prefills),
            "decode_speed_ratio_vs_fp16": metric_range(decodes),
            "total_speed_ratio_vs_fp16": metric_range(totals),
            "footprint_ratio_vs_fp16": metric_range(footprints),
            "minimum_logits_cosine": metric_range(cosines),
            "greedy_pass_rows": sum(row.get("same_greedy_tokens_as_fp16") is True for row in rows),
        }

    return {
        "axis": "rtx4080_qwen35_bsz8_acceptance",
        "status": "pass" if not errors else "fail",
        "scope": {
            "device": "NVIDIA GeForce RTX 4080",
            "model_pair": "rwkv-1.5b__qwen3.5-2b",
            "batch_size": 8,
            "prompt_tokens": [128, 512, 2048],
            "decode_tokens": [128, 512],
            "dtype": "fp16",
            "qwen_backend": "full FLA",
        },
        "gates": {
            "min_dense_prefill_ratio": min_dense_prefill,
            "min_dense_decode_ratio": min_dense_decode,
            "min_active_work_decode_ratio": 1.0,
            "min_quant_total_speed_ratio": min_quant_speed,
            "min_quant_decode_speed_ratio": min_quant_speed,
            "quant_prefill_speed": "reported, not independently gated",
            "max_quant_footprint_ratio": "<1.0",
            "min_quant_logits_cosine": min_quant_cosine,
            "quant_greedy_match": True,
        },
        "coverage": {
            "dense_candidate_rows": len(candidates),
            "qwen_reference_rows": len(references),
            "memory_rows": len(memory_rows),
            "paired_quant_rows": len(quant_rows),
        },
        "dense_vs_qwen": {
            "prefill_speed_ratio": metric_range(dense_prefill),
            "decode_speed_ratio": metric_range(dense_decode),
            "active_work_decode_ratio": metric_range(dense_active_decode),
            "qwen_full_fla_rows": sum(row.get("qwen_full_fused_contract_pass") is True for row in references),
        },
        "memory_routes": memory_summary,
        "paired_speed_routes": quant_summary,
        "errors": errors,
    }


def markdown(report: dict[str, Any]) -> str:
    dense = report["dense_vs_qwen"]

    def display(value: Any) -> str:
        return f"{float(value):.4f}x" if value is not None else "n/a"

    lines = [
        "# RTX 4080 RWKV-7 / Qwen3.5 acceptance",
        "",
        f"Status: **{report['status']}**",
        "",
        "| Axis | Measured range |",
        "|---|---:|",
        f"| Dense prefill / full-FLA Qwen | {display(dense['prefill_speed_ratio']['min'])} - {display(dense['prefill_speed_ratio']['max'])} |",
        f"| Dense decode / full-FLA Qwen | {display(dense['decode_speed_ratio']['min'])} - {display(dense['decode_speed_ratio']['max'])} |",
        f"| Active-work decode ratio | {display(dense['active_work_decode_ratio']['min'])} - {display(dense['active_work_decode_ratio']['max'])} |",
    ]
    for name, values in report["paired_speed_routes"].items():
        lines.append(
            f"| {name} paired prefill/decode/total | "
            f"{display(values['prefill_speed_ratio_vs_fp16']['min'])} / "
            f"{display(values['decode_speed_ratio_vs_fp16']['min'])} / "
            f"{display(values['total_speed_ratio_vs_fp16']['min'])} minimum |"
        )
    if report["errors"]:
        lines.extend(["", "## Failed gates", "", *[f"- {item}" for item in report["errors"]]])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--min-dense-prefill", type=float, default=1.0)
    parser.add_argument("--min-dense-decode", type=float, default=1.4)
    parser.add_argument("--min-quant-speed", type=float, default=1.0)
    parser.add_argument("--min-quant-cosine", type=float, default=0.999)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    report = summarize(
        args.artifact_root,
        min_dense_prefill=args.min_dense_prefill,
        min_dense_decode=args.min_dense_decode,
        min_quant_speed=args.min_quant_speed,
        min_quant_cosine=args.min_quant_cosine,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(markdown(report), encoding="utf-8")
    print(text, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
