#!/usr/bin/env python3
"""Fail-closed summary for the exact RTX 5090 Qwen3.5 acceptance matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PAIR_DIRS = {
    "rwkv-0.4b__qwen3.5-0.8b": "pair_0.4b_0.8b",
    "rwkv-1.5b__qwen3.5-2b": "pair_1.5b_2b",
    "rwkv-2.9b__qwen3.5-4b": "pair_2.9b_4b",
    "rwkv-7.2b__qwen3.5-9b": "pair_7.2b_9b",
}
BATCH_SIZES = (1, 8)
CORRECTNESS_FILES = (
    "full-fla-vs-transformers-conv-oracle.json",
    "rwkv-prefill-correctness-none.json",
    "rwkv-prefill-correctness-bnb8.json",
    "rwkv-prefill-correctness-bnb4.json",
)
EXIT_CODE_FILES = (
    "pipeline_exit_code.txt",
    "matrix_failures.txt",
    "compose_exit_code.txt",
    "compare_memory_exit_code.txt",
    "compare_speed_exit_code.txt",
    "compare_active_work_exit_code.txt",
    "correctness-failures.txt",
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(row)
    return rows


def _read_zero(path: Path) -> bool:
    return int(path.read_text(encoding="utf-8").strip()) == 0


def _shape(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(row.get("batch_size", -1)),
        int(row.get("prompt_tokens", -1)),
        int(row.get("decode_tokens", -1)),
    )


def _family(summary: dict[str, Any], name: str) -> dict[str, Any]:
    value = (summary.get("speed_by_quantization") or {}).get(name)
    return value if isinstance(value, dict) else {}


def _validate_pair(
    root: Path,
    pair_label: str,
    batch_size: int,
) -> tuple[dict[str, Any], list[str]]:
    label = f"B{batch_size} {pair_label}"
    expected_shapes = {
        (batch_size, prompt_tokens, decode_tokens)
        for prompt_tokens in (128, 512, 2048)
        for decode_tokens in (128, 512)
    }
    errors: list[str] = []

    def require(path: Path) -> bool:
        if path.is_file():
            return True
        errors.append(f"{label}: missing {path.name}")
        return False

    for name in EXIT_CODE_FILES:
        path = root / name
        if require(path):
            try:
                if not _read_zero(path):
                    errors.append(f"{label}: {name} is nonzero")
            except (OSError, ValueError) as exc:
                errors.append(f"{label}: invalid {name}: {exc}")

    correctness: dict[str, Any] = {}
    for name in CORRECTNESS_FILES:
        path = root / name
        if not require(path):
            continue
        try:
            report = _load_json(path)
            correctness[name] = report
            if report.get("status") != "pass":
                errors.append(f"{label}: {name} status is not pass")
            if not report.get("greedy_tokens_match"):
                errors.append(f"{label}: {name} greedy tokens do not match")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{label}: invalid {name}: {exc}")

    summaries: dict[str, dict[str, Any]] = {}
    for kind in ("speed", "memory", "active_work"):
        name = f"summary_{kind}.json"
        path = root / name
        if not require(path):
            continue
        try:
            report = _load_json(path)
            summaries[kind] = report
            gates = report.get("gates") or {}
            if not gates.get("overall_pass"):
                errors.append(f"{label}: {name} overall gate failed")
            coverage = report.get("coverage") or {}
            expected = 6 if kind == "active_work" else 18
            if coverage.get("expected_cells") != expected or coverage.get("joined_cells") != expected:
                errors.append(f"{label}: {name} coverage is not {expected}/{expected}")
            red_cells = report.get("red_cells", [])
            red_count = len(red_cells) if isinstance(red_cells, list) else int(red_cells)
            if red_count != 0:
                errors.append(f"{label}: {name} has red cells")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{label}: invalid {name}: {exc}")

    manifest_path = root / "route_manifest.json"
    route_manifest: dict[str, Any] = {}
    if require(manifest_path):
        try:
            route_manifest = _load_json(manifest_path)
            if route_manifest.get("status") != "pass" or route_manifest.get("failures"):
                errors.append(f"{label}: quant route manifest failed")
            if len(route_manifest.get("decisions") or []) != 12:
                errors.append(f"{label}: quant route manifest must contain 12 decisions")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{label}: invalid route_manifest.json: {exc}")

    rows_path = root / "combined_auto.jsonl"
    rows: list[dict[str, Any]] = []
    if require(rows_path):
        try:
            rows = _load_jsonl(rows_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{label}: invalid combined_auto.jsonl: {exc}")
    candidates = [row for row in rows if row.get("model_role") == "candidate"]
    references = [row for row in rows if row.get("model_role") == "reference"]
    if len(candidates) != 18 or len(references) != 18:
        errors.append(f"{label}: combined rows must be 18 candidate + 18 reference")
    if {_shape(row) for row in candidates} != expected_shapes:
        errors.append(f"{label}: candidate shape matrix is incomplete")
    if {_shape(row) for row in references} != expected_shapes:
        errors.append(f"{label}: reference shape matrix is incomplete")
    for row in candidates:
        if row.get("model_pair") != pair_label or row.get("status") != "pass" or not row.get("logits_finite"):
            errors.append(f"{label}: candidate row failed identity/status/finite-logits contract")
            break
        if row.get("prefill_backend_effective") != "native_prefill_graph":
            errors.append(f"{label}: candidate row did not use native_prefill_graph")
            break
    for row in references:
        if row.get("model_pair") != pair_label or row.get("status") != "pass" or not row.get("logits_finite"):
            errors.append(f"{label}: reference row failed identity/status/finite-logits contract")
            break
        if row.get("qwen_backend_requested") != "fla":
            errors.append(f"{label}: reference row did not request FLA")
            break
        if row.get("qwen_conv_backend_effective") != "fla_triton":
            errors.append(f"{label}: reference row did not use the FLA Triton conv bridge")
            break
        if not row.get("qwen_full_fused_contract_pass") or not row.get("qwen_fast_path_verified"):
            errors.append(f"{label}: reference row failed the full-FLA operator contract")
            break

    if batch_size == 8 and pair_label == "rwkv-1.5b__qwen3.5-2b":
        exact_rows = [
            row
            for row in candidates
            if row.get("quantization") == "none" and int(row.get("prompt_tokens", -1)) == 512
        ]
        if len(exact_rows) != 2:
            errors.append(f"{label}: missing two dense P512 exact-policy rows")
        elif not all(
            row.get("rwkv_prefill_clampw_scan_effective")
            and row.get("rwkv_prefill_stacked_rkv_effective")
            and row.get("rwkv_prefill_sequence_ffn_effective")
            for row in exact_rows
        ):
            errors.append(f"{label}: dense P512 exact-policy telemetry is incomplete")

    speed = summaries.get("speed") or {}
    dense = _family(speed, "none")
    active_work = (summaries.get("active_work") or {}).get("active_parameter_work") or {}
    pair_summary = {
        "status": "pass" if not errors else "fail",
        "batch_size": batch_size,
        "coverage": {
            "candidate_rows": len(candidates),
            "reference_rows": len(references),
            "shapes": len({_shape(row) for row in candidates}),
            "qwen_full_fla_rows": sum(bool(row.get("qwen_full_fused_contract_pass")) for row in references),
        },
        "dense": dense,
        "w8": _family(speed, "w8"),
        "w4": _family(speed, "w4"),
        "dense_active_work": active_work,
        "correctness": {
            name: {
                "status": report.get("status"),
                "greedy_tokens_match": report.get("greedy_tokens_match"),
                "prompt_logits_cosine": report.get("prompt_logits_cosine"),
                "final_logits_cosine": report.get("final_logits_cosine"),
            }
            for name, report in correctness.items()
        },
        "route_decisions": len(route_manifest.get("decisions") or []),
    }
    return pair_summary, errors


def summarize(
    root: Path,
    pair_labels: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    selected_labels = tuple(PAIR_DIRS) if pair_labels is None else tuple(pair_labels)
    if not selected_labels:
        raise ValueError("at least one model pair is required")
    unknown_labels = sorted(set(selected_labels) - set(PAIR_DIRS))
    if unknown_labels:
        raise ValueError(f"unknown model pairs: {', '.join(unknown_labels)}")

    errors: list[str] = []
    batches: dict[str, Any] = {}
    all_pairs = []
    for batch_size in BATCH_SIZES:
        pairs: dict[str, Any] = {}
        for label in selected_labels:
            directory = PAIR_DIRS[label]
            pair_summary, pair_errors = _validate_pair(
                root / f"b{batch_size}" / directory,
                label,
                batch_size,
            )
            pairs[label] = pair_summary
            all_pairs.append(pair_summary)
            errors.extend(pair_errors)
        batches[f"b{batch_size}"] = pairs
    return {
        "axis": "rtx5090_g1h_qwen35_b1_b8_acceptance",
        "status": "pass" if not errors else "fail",
        "scope": {
            "device": "NVIDIA GeForce RTX 5090",
            "batch_size": list(BATCH_SIZES),
            "prompt_tokens": [128, 512, 2048],
            "decode_tokens": [128, 512],
            "dtype": "fp16",
            "qwen_backend": "full FLA with FLA Triton causal-conv bridge",
            "expected_joined_cells": len(selected_labels) * len(BATCH_SIZES) * 18,
            "selected_model_pairs": list(selected_labels),
        },
        "matrix_complete": set(selected_labels) == set(PAIR_DIRS),
        "coverage": {
            "batch_pairs": sum(pair.get("status") == "pass" for pair in all_pairs),
            "expected_batch_pairs": len(selected_labels) * len(BATCH_SIZES),
            "candidate_rows": sum(pair["coverage"]["candidate_rows"] for pair in all_pairs),
            "reference_rows": sum(pair["coverage"]["reference_rows"] for pair in all_pairs),
            "qwen_full_fla_rows": sum(pair["coverage"]["qwen_full_fla_rows"] for pair in all_pairs),
        },
        "batches": batches,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--pair",
        action="append",
        choices=tuple(PAIR_DIRS),
        dest="pair_labels",
        help="Validate only this model pair; repeat for an explicit partial matrix.",
    )
    args = parser.parse_args()
    report = summarize(args.artifact_root, args.pair_labels)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
