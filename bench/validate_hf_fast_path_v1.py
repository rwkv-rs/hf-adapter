#!/usr/bin/env python3
"""Validate and assemble the fail-closed 96-row HF fast-path v1 card matrix."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Iterable

MATRIX = "hf_fast_path_v1"
BATCHES = (1, 8)
PROMPTS = (128, 512, 2048)
DECODES = (128, 512)
EXPECTED_SHAPES = set(product(BATCHES, PROMPTS, DECODES))
PAIRS = {
    "rwkv-0.4b__qwen3.5-0.8b": ("0.4b", "0.8b"),
    "rwkv-1.5b__qwen3.5-2b": ("1.5b", "2b"),
    "rwkv-2.9b__qwen3.5-4b": ("2.9b", "4b"),
    "rwkv-7.2b__qwen3.5-9b": ("7.2b", "9b"),
}
RUNTIME_FIELDS = (
    "torch_version",
    "torch_cuda_version",
    "triton_version",
    "transformers_version",
    "fla_version",
    "causal_conv1d_version",
)


def read_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_source"] = f"{path}:{line_number}"
            rows.append(row)
    return rows


def _require(row: dict[str, Any], field: str, expected: Any, errors: list[str]) -> None:
    if row.get(field) != expected:
        errors.append(
            f"{row.get('_source', '<row>')}: {field}={row.get(field)!r}, expected {expected!r}"
        )


def _validate_common(
    row: dict[str, Any], *, role: str, kind: str, expected_device: str, errors: list[str]
) -> None:
    for field, expected in (
        ("axis", "qwen35_cross_model_speed"),
        ("benchmark_matrix", MATRIX),
        ("model_role", role),
        ("model_kind", kind),
        ("dtype", "fp16"),
        ("quantization", "none"),
        ("prefill_chunk_size", 512),
        ("warmup", 3),
        ("runs", 7),
        ("timing_statistic", "median"),
        ("mtp_enabled", False),
        ("speculative_decoding_enabled", False),
        ("resident_sweep", True),
        ("status", "pass"),
    ):
        _require(row, field, expected, errors)
    if expected_device:
        _require(row, "device", expected_device, errors)
    pair = str(row.get("model_pair", ""))
    if pair not in PAIRS:
        errors.append(f"{row.get('_source', '<row>')}: unexpected model_pair={pair!r}")
    else:
        _require(row, "model_size_label", PAIRS[pair][0 if role == "candidate" else 1], errors)
    shape = (row.get("batch_size"), row.get("prompt_tokens"), row.get("decode_tokens"))
    if shape not in EXPECTED_SHAPES:
        errors.append(f"{row.get('_source', '<row>')}: unexpected B/P/D cell {shape!r}")


def _validate_candidate(row: dict[str, Any], errors: list[str]) -> None:
    for field, expected in (
        ("rwkv_fast_token_backend_requested", "native_jit"),
        ("rwkv_prefill_graph_requested", "0"),
        ("effective_backend", "native_jit"),
    ):
        _require(row, field, expected, errors)
    for field in ("effective_backend", "step_backend", "prefill_backend_effective"):
        if "graph" in str(row.get(field, "")).lower():
            errors.append(
                f"{row.get('_source', '<row>')}: {field} proves a CUDA Graph route: {row.get(field)!r}"
            )


def _validate_reference(row: dict[str, Any], errors: list[str]) -> None:
    for field, expected in (
        ("qwen_backend_requested", "fla"),
        ("qwen_conv_backend_requested", "causal_conv1d"),
        ("qwen_fast_path_required", True),
        ("qwen_fast_path_available", True),
        ("qwen_fast_path_verified", True),
        ("qwen_full_fused_contract_pass", True),
        ("qwen_causal_conv1d_importable", True),
        ("qwen_conv_backend_effective", "causal_conv1d"),
        ("qwen_force_torch", False),
    ):
        _require(row, field, expected, errors)


def validate_matrix(
    candidate_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    *,
    expected_device: str = "",
) -> dict[str, Any]:
    errors: list[str] = []
    if len(candidate_rows) != 48:
        errors.append(f"candidate row count={len(candidate_rows)}, expected 48")
    if len(reference_rows) != 48:
        errors.append(f"reference row count={len(reference_rows)}, expected 48")

    for row in candidate_rows:
        _validate_common(row, role="candidate", kind="rwkv", expected_device=expected_device, errors=errors)
        _validate_candidate(row, errors)
    for row in reference_rows:
        _validate_common(row, role="reference", kind="qwen35", expected_device=expected_device, errors=errors)
        _validate_reference(row, errors)

    expected_keys = {(pair, *shape) for pair in PAIRS for shape in EXPECTED_SHAPES}
    for label, rows in (("candidate", candidate_rows), ("reference", reference_rows)):
        keys = [
            (row.get("model_pair"), row.get("batch_size"), row.get("prompt_tokens"), row.get("decode_tokens"))
            for row in rows
        ]
        counts = Counter(keys)
        duplicates = sorted(key for key, count in counts.items() if count > 1)
        missing = sorted(expected_keys - set(keys))
        extras = sorted(set(keys) - expected_keys)
        if duplicates:
            errors.append(f"{label} duplicate cells: {duplicates}")
        if missing:
            errors.append(f"{label} missing cells: {missing}")
        if extras:
            errors.append(f"{label} extra cells: {extras}")

    all_rows = candidate_rows + reference_rows
    runtime_signatures = {
        tuple(row.get(field) for field in RUNTIME_FIELDS)
        for row in all_rows
        if row.get("status") == "pass"
    }
    if len(runtime_signatures) != 1:
        errors.append(
            "rows were not produced by one locked runtime signature: "
            + json.dumps(sorted(runtime_signatures, key=repr), ensure_ascii=False)
        )
    devices = sorted({str(row.get("device")) for row in all_rows})
    summary = {
        "schema_version": 1,
        "benchmark_matrix": MATRIX,
        "status": "pass" if not errors else "fail",
        "candidate_rows": len(candidate_rows),
        "reference_rows": len(reference_rows),
        "total_rows": len(all_rows),
        "expected_rows": 96,
        "devices": devices,
        "runtime_fields": list(RUNTIME_FIELDS),
        "runtime_signature_count": len(runtime_signatures),
        "qwen_contract": "official_fla_plus_causal_conv1d",
        "rwkv_contract": "native_jit_without_cuda_graph",
        "errors": errors,
    }
    return summary


def _clean_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "_source"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-results", type=Path, nargs="+", required=True)
    parser.add_argument("--reference-results", type=Path, nargs="+", required=True)
    parser.add_argument("--expected-device", default="")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--main-table", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = read_rows(args.candidate_results)
    references = read_rows(args.reference_results)
    summary = validate_matrix(candidates, references, expected_device=args.expected_device)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if summary["status"] != "pass":
        print("HF_FAST_PATH_V1_VALIDATION " + json.dumps(summary, ensure_ascii=False), flush=True)
        return 1
    ordered = sorted(
        (_clean_row(row) for row in candidates + references),
        key=lambda row: (
            row["model_pair"],
            0 if row["model_role"] == "candidate" else 1,
            row["batch_size"],
            row["prompt_tokens"],
            row["decode_tokens"],
        ),
    )
    args.main_table.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered),
        encoding="utf-8",
    )
    print("HF_FAST_PATH_V1_VALIDATION " + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
