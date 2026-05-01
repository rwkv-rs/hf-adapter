#!/usr/bin/env python3
"""Validate and assemble the frozen-Qwen/RWKV paired Decode v1 matrix.

This protocol deliberately keeps the Qwen-only reference artifact immutable.
It accepts a separately captured 48-row RWKV candidate matrix, validates both
inputs fail-closed, and gates the unrounded active-parameter-adjusted Decode
ratio in every matched cell.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import torch

try:
    from bench.compare_rwkv_prefill_probe import compare as compare_rwkv_probes
    from bench.summarize_qwen35_best_optimized_hf_v1 import display_rate
    from bench.validate_qwen35_best_optimized_hf_v1 import (
        EXPECTED_SHAPES,
        PAIR_RANK,
        RUNTIME_FIELDS,
        validate_matrix as validate_qwen_reference,
    )
except ModuleNotFoundError:
    from compare_rwkv_prefill_probe import compare as compare_rwkv_probes
    from summarize_qwen35_best_optimized_hf_v1 import display_rate
    from validate_qwen35_best_optimized_hf_v1 import (
        EXPECTED_SHAPES,
        PAIR_RANK,
        RUNTIME_FIELDS,
        validate_matrix as validate_qwen_reference,
    )


PROTOCOL = "qwen35_paired_decode_v1"
CANDIDATE_MATRIX = PROTOCOL
CANDIDATE_LANE = "best_optimized_hf"
EXPECTED_DEVICE = "NVIDIA GeForce RTX 5090"
FROZEN_REFERENCE_SHA256 = (
    "b02378fe14d455f52940a3d24e4f515f49c18a06f57c65ad0b461a2330b5f6d1"
)
STRICT_ADJUSTED_DECODE_GATE = 1.0
HEX_DIGITS = frozenset("0123456789abcdef")
SM120_AB_PROTOCOL = "sm120_b8_decode_ab_v1"
DECODE_CORRECTNESS_PROTOCOL = "rwkv_native_graph_fla_correctness_v1"
PARAMETERS = {
    "rwkv-0.4b__qwen3.5-0.8b": (450_767_872, 752_393_024),
    "rwkv-1.5b__qwen3.5-2b": (1_527_404_544, 1_881_825_088),
    "rwkv-2.9b__qwen3.5-4b": (2_947_735_040, 4_205_751_296),
    "rwkv-7.2b__qwen3.5-9b": (7_199_141_888, 8_953_803_264),
}
RWKV_PAIR_SIZES = {
    "rwkv-0.4b__qwen3.5-0.8b": "0.4b",
    "rwkv-1.5b__qwen3.5-2b": "1.5b",
    "rwkv-2.9b__qwen3.5-4b": "2.9b",
    "rwkv-7.2b__qwen3.5-9b": "7.2b",
}
EXPECTED_KEYS = {
    (pair, batch, prompt, decode)
    for pair in PARAMETERS
    for batch, prompt, decode in EXPECTED_SHAPES
}


def _is_finite_real(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_jsonl_bytes(data: bytes, path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text = data.decode("utf-8")
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"JSON value at {path}:{line_number} is not an object")
        row["_source"] = f"{path}:{line_number}"
        row["_line_number"] = line_number
        rows.append(row)
    return rows


def _strict_equal(actual: Any, expected: Any) -> bool:
    """Compare JSON-like contract values without bool/int coercion."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _strict_equal(actual[key], expected[key]) for key in expected
        )
    return bool(actual == expected)


def _require(row: dict[str, Any], field: str, expected: Any, errors: list[str]) -> None:
    actual = row.get(field)
    matches = _strict_equal(actual, expected)
    if not matches:
        errors.append(
            f"{row.get('_source', '<row>')}: {field}={actual!r}, expected {expected!r}"
        )


def _require_positive(
    row: dict[str, Any], field: str, errors: list[str]
) -> float | None:
    value = row.get(field)
    if not _is_finite_real(value) or float(value) <= 0:
        errors.append(f"{row.get('_source', '<row>')}: {field}={value!r}, expected > 0")
        return None
    return float(value)


def _cell_key(row: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    return (
        row.get("model_pair"),
        row.get("batch_size"),
        row.get("prompt_tokens"),
        row.get("decode_tokens"),
    )


def _validate_candidate_row(
    row: dict[str, Any], *, expected_device: str, errors: list[str]
) -> None:
    for field, expected in (
        ("axis", "qwen35_cross_model_speed"),
        ("benchmark_matrix", CANDIDATE_MATRIX),
        ("optimization_lane", CANDIDATE_LANE),
        ("model_role", "candidate"),
        ("model_kind", "rwkv"),
        ("rwkv_implementation_requested", "auto"),
        ("rwkv_implementation_effective", "native_model"),
        ("dtype", "fp16"),
        ("quantization", "none"),
        ("quantization_backend", "dense"),
        ("native_quant_kernel_active", False),
        ("prefill_chunk_size", 512),
        ("warmup", 3),
        ("runs", 7),
        ("timing_statistic", "median"),
        ("mtp_enabled", False),
        ("speculative_decoding_enabled", False),
        ("resident_sweep", True),
        ("status", "pass"),
        ("logits_finite", True),
        ("device", expected_device),
        ("gpu_arch", "sm_120"),
        ("gpu_compute_capability", [12, 0]),
        ("rwkv_fast_token_backend_requested", "native_graph"),
        ("rwkv_native_model_backend_requested", "native_graph"),
        ("effective_backend", "native_graph"),
        ("step_backend", "rwkv_fast_token"),
        ("cache_type", "NativeRWKV7Cache"),
    ):
        _require(row, field, expected, errors)

    pair = str(row.get("model_pair") or "")
    if pair not in PARAMETERS:
        errors.append(f"{row.get('_source', '<row>')}: unexpected model_pair={pair!r}")
    else:
        _require(row, "model_size_label", RWKV_PAIR_SIZES[pair], errors)
        _require(row, "active_parameter_count", PARAMETERS[pair][0], errors)

    shape = (row.get("batch_size"), row.get("prompt_tokens"), row.get("decode_tokens"))
    if shape not in EXPECTED_SHAPES:
        errors.append(f"{row.get('_source', '<row>')}: unexpected B/P/D cell {shape!r}")

    batch_size = row.get("batch_size")
    if batch_size == 8 and pair in {
        "rwkv-0.4b__qwen3.5-0.8b",
        "rwkv-1.5b__qwen3.5-2b",
    }:
        for field, expected in (
            ("rwkv_native_graph_ada_wagv_bmm_requested", True),
            ("rwkv_native_graph_ada_wagv_bmm_selected", True),
            ("rwkv_native_graph_ada_wagv_bmm_effective", True),
            ("rwkv_native_graph_ada_wagv_bmm_effective_layer_count", 24),
            ("rwkv_native_graph_ada_wagv_bmm_full_model_effective", True),
            ("rwkv_native_graph_sm120_wagv_bmm_g_requested", True),
            ("rwkv_native_graph_sm120_wagv_bmm_g_selected", True),
            ("rwkv_native_graph_sm120_wagv_bmm_g_effective", True),
            ("rwkv_native_graph_sm120_wagv_bmm_g_full_model_effective", True),
            ("rwkv_native_graph_sm120_compiled_ffn_requested", True),
            ("rwkv_native_graph_sm120_compiled_ffn_selected", True),
            ("rwkv_native_graph_sm120_compiled_ffn_effective", True),
            ("rwkv_native_graph_sm120_compiled_ffn_full_model_effective", True),
        ):
            _require(row, field, expected, errors)
        expected_layers = list(range(24))
        for route in ("sm120_wagv_bmm_g", "sm120_compiled_ffn"):
            _require(
                row,
                f"rwkv_native_graph_{route}_selected_layers",
                expected_layers,
                errors,
            )
            _require(
                row,
                f"rwkv_native_graph_{route}_effective_layers",
                expected_layers,
                errors,
            )
            _require(
                row,
                f"rwkv_native_graph_{route}_effective_layer_count",
                len(expected_layers),
                errors,
            )
        for field, expected in (
            ("rwkv_native_graph_sm120_compiled_ffn_compile_effective", True),
            ("rwkv_native_graph_sm120_compiled_ffn_compile_reused", True),
            ("rwkv_native_graph_sm120_compiled_ffn_unique_graphs", 1),
            ("rwkv_native_graph_sm120_compiled_ffn_graph_breaks", 0),
            (
                "rwkv_native_graph_sm120_compiled_ffn_compile_mode",
                "max-autotune-no-cudagraphs",
            ),
            ("rwkv_native_graph_sm120_compiled_ffn_prewarm_all_finite", True),
            (
                "rwkv_native_graph_sm120_compiled_ffn_prewarm_argmax_all_equal",
                True,
            ),
            (
                "rwkv_native_graph_sm120_compiled_ffn_prewarm_layer_indices",
                expected_layers,
            ),
            ("rwkv_native_graph_sm120_compiled_ffn_prewarm_layer_count", 24),
        ):
            _require(row, field, expected, errors)
        prewarm_cosine = row.get(
            "rwkv_native_graph_sm120_compiled_ffn_prewarm_min_cosine"
        )
        if not _is_finite_real(prewarm_cosine) or float(prewarm_cosine) < 0.9999:
            errors.append(
                f"{row.get('_source', '<row>')}: "
                "rwkv_native_graph_sm120_compiled_ffn_prewarm_min_cosine="
                f"{prewarm_cosine!r}, expected finite >= 0.9999"
            )
        prewarm_max_abs = row.get(
            "rwkv_native_graph_sm120_compiled_ffn_prewarm_max_abs_diff"
        )
        if not _is_finite_real(prewarm_max_abs) or float(prewarm_max_abs) < 0:
            errors.append(
                f"{row.get('_source', '<row>')}: "
                "rwkv_native_graph_sm120_compiled_ffn_prewarm_max_abs_diff="
                f"{prewarm_max_abs!r}, expected finite >= 0"
            )
    else:
        for field, expected in (
            ("rwkv_native_graph_ada_wagv_bmm_requested", False),
            ("rwkv_native_graph_ada_wagv_bmm_selected", False),
            ("rwkv_native_graph_ada_wagv_bmm_effective", False),
            ("rwkv_native_graph_ada_wagv_bmm_effective_layer_count", 0),
            ("rwkv_native_graph_ada_wagv_bmm_full_model_effective", False),
        ):
            _require(row, field, expected, errors)
        for route in ("sm120_wagv_bmm_g", "sm120_compiled_ffn"):
            for suffix, expected in (
                ("requested", False),
                ("selected", False),
                ("effective", False),
                ("selected_layers", []),
                ("effective_layers", []),
                ("effective_layer_count", 0),
                ("full_model_effective", False),
            ):
                _require(
                    row,
                    f"rwkv_native_graph_{route}_{suffix}",
                    expected,
                    errors,
                )
        for field in (
            "rwkv_native_graph_sm120_compiled_ffn_compile_effective",
            "rwkv_native_graph_sm120_compiled_ffn_compile_reused",
            "rwkv_native_graph_sm120_compiled_ffn_unique_graphs",
            "rwkv_native_graph_sm120_compiled_ffn_graph_breaks",
            "rwkv_native_graph_sm120_compiled_ffn_compile_mode",
            "rwkv_native_graph_sm120_compiled_ffn_prewarm_all_finite",
            "rwkv_native_graph_sm120_compiled_ffn_prewarm_min_cosine",
            "rwkv_native_graph_sm120_compiled_ffn_prewarm_argmax_all_equal",
            "rwkv_native_graph_sm120_compiled_ffn_prewarm_max_abs_diff",
            "rwkv_native_graph_sm120_compiled_ffn_prewarm_layer_indices",
            "rwkv_native_graph_sm120_compiled_ffn_prewarm_layer_count",
        ):
            _require(row, field, None, errors)

    for field in RUNTIME_FIELDS:
        value = row.get(field)
        if type(value) is not str or not value.strip():
            errors.append(
                f"{row.get('_source', '<row>')}: {field}={value!r}, "
                "expected a non-empty string"
            )
    commit = row.get("benchmark_repository_commit")
    normalized_commit = commit.lower() if type(commit) is str else ""
    if len(normalized_commit) != 40 or any(
        character not in HEX_DIGITS for character in normalized_commit
    ):
        errors.append(
            f"{row.get('_source', '<row>')}: benchmark_repository_commit "
            "must be exactly 40 hexadecimal characters"
        )

    samples = row.get("decode_sec_samples")
    if not isinstance(samples, list) or len(samples) != 7:
        errors.append(
            f"{row.get('_source', '<row>')}: decode_sec_samples must contain 7 raw samples"
        )
        return
    if not all(_is_finite_real(value) and float(value) > 0 for value in samples):
        errors.append(
            f"{row.get('_source', '<row>')}: decode_sec_samples contains a non-positive sample"
        )
        return
    sample_median = float(statistics.median(samples))
    raw_median = _require_positive(row, "decode_sec_median_raw", errors)
    rounded_median = _require_positive(row, "decode_sec_median", errors)
    raw_tokps = _require_positive(row, "decode_tokps_total_raw", errors)
    _require_positive(row, "decode_tokps_total", errors)
    if raw_median is not None and raw_median != sample_median:
        errors.append(
            f"{row.get('_source', '<row>')}: decode_sec_median_raw={raw_median!r} "
            f"does not exactly match sample median {sample_median!r}"
        )
    if rounded_median is not None and abs(rounded_median - sample_median) > 1e-6:
        errors.append(
            f"{row.get('_source', '<row>')}: decode_sec_median={rounded_median!r} "
            f"does not match sample median {sample_median!r}"
        )
    if raw_median is not None and raw_tokps is not None and shape in EXPECTED_SHAPES:
        batch, _prompt, decode = (int(value) for value in shape)
        expected_tokps = batch * decode / raw_median
        if not math.isclose(raw_tokps, expected_tokps, rel_tol=1e-12, abs_tol=1e-12):
            errors.append(
                f"{row.get('_source', '<row>')}: decode_tokps_total_raw={raw_tokps!r}, "
                f"expected tokens/raw_median={expected_tokps!r}"
            )


def _validate_coverage(
    rows: list[dict[str, Any]], label: str, errors: list[str]
) -> dict[tuple[Any, Any, Any, Any], dict[str, Any]]:
    keys = [_cell_key(row) for row in rows]
    counts = Counter(keys)
    duplicates = sorted((key for key, count in counts.items() if count > 1), key=repr)
    missing = sorted(EXPECTED_KEYS - set(keys), key=repr)
    extras = sorted(set(keys) - EXPECTED_KEYS, key=repr)
    if len(rows) != 48:
        errors.append(f"{label} row count={len(rows)}, expected 48")
    if duplicates:
        errors.append(f"{label} duplicate cells: {duplicates}")
    if missing:
        errors.append(f"{label} missing cells: {missing}")
    if extras:
        errors.append(f"{label} extra cells: {extras}")
    return {
        key: row
        for key, row in zip(keys, rows)
        if counts[key] == 1 and key in EXPECTED_KEYS
    }


def _sort_key(cell: dict[str, Any]) -> tuple[Any, ...]:
    return (
        PAIR_RANK.get(str(cell.get("model_pair")), 999),
        str(cell.get("device") or ""),
        int(cell.get("batch_size") or 0),
        int(cell.get("prompt_tokens") or 0),
        int(cell.get("decode_tokens") or 0),
    )


def validate_paired_decode(
    candidate_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    *,
    expected_device: str = EXPECTED_DEVICE,
    expected_reference_sha256: str = FROZEN_REFERENCE_SHA256,
    reference_sha256_before: str,
    reference_sha256_after: str,
    candidate_sha256: str = "",
) -> dict[str, Any]:
    """Validate both matrices and derive the strict per-cell Decode table."""

    errors: list[str] = []
    expected_hash = expected_reference_sha256.strip().lower()
    if len(expected_hash) != 64 or any(
        ch not in "0123456789abcdef" for ch in expected_hash
    ):
        errors.append(
            "expected reference SHA256 must be 64 lowercase hexadecimal characters"
        )
    if reference_sha256_before.lower() != expected_hash:
        errors.append(
            "Qwen reference SHA256 mismatch: "
            f"observed {reference_sha256_before.lower()}, expected {expected_hash}"
        )
    if reference_sha256_after.lower() != reference_sha256_before.lower():
        errors.append(
            "Qwen reference bytes changed while validation was running: "
            f"{reference_sha256_before.lower()} -> {reference_sha256_after.lower()}"
        )

    for row in reference_rows:
        for field in RUNTIME_FIELDS:
            value = row.get(field)
            if type(value) is not str or not value.strip():
                errors.append(
                    f"{row.get('_source', '<row>')}: reference {field}={value!r}, "
                    "expected a non-empty string"
                )
        commit = row.get("benchmark_repository_commit")
        normalized_commit = commit.lower() if type(commit) is str else ""
        if len(normalized_commit) != 40 or any(
            character not in HEX_DIGITS for character in normalized_commit
        ):
            errors.append(
                f"{row.get('_source', '<row>')}: reference "
                "benchmark_repository_commit must be exactly 40 hexadecimal characters"
            )
    try:
        qwen_validation = validate_qwen_reference(
            reference_rows, expected_device=expected_device
        )
    except (TypeError, ValueError) as exc:
        qwen_validation = {
            "status": "fail",
            "errors": [f"validator rejected malformed reference telemetry: {exc}"],
        }
    if qwen_validation.get("status") != "pass":
        errors.extend(
            f"Qwen reference: {message}"
            for message in qwen_validation.get("errors", [])
        )

    for row in candidate_rows:
        _validate_candidate_row(row, expected_device=expected_device, errors=errors)
    candidate_index = _validate_coverage(candidate_rows, "candidate", errors)
    reference_index = _validate_coverage(reference_rows, "reference", errors)

    for row in reference_rows:
        pair = str(row.get("model_pair") or "")
        if pair in PARAMETERS:
            _require(row, "active_parameter_count", PARAMETERS[pair][1], errors)

    candidate_commits = sorted(
        {
            str(row.get("benchmark_repository_commit"))
            for row in candidate_rows
            if row.get("benchmark_repository_commit") not in (None, "")
        }
    )
    reference_commits = sorted(
        {
            str(row.get("benchmark_repository_commit"))
            for row in reference_rows
            if row.get("benchmark_repository_commit") not in (None, "")
        }
    )
    if len(candidate_commits) != 1:
        errors.append(
            "candidate rows did not record exactly one repository commit: "
            + json.dumps(candidate_commits)
        )
    if len(reference_commits) != 1:
        errors.append(
            "reference rows did not record exactly one repository commit: "
            + json.dumps(reference_commits)
        )

    def runtime_signature(row: dict[str, Any]) -> tuple[str, ...]:
        return tuple(
            value
            if type(value := row.get(field)) is str
            else f"<invalid-{field}-{type(value).__name__}:{value!r}>"
            for field in RUNTIME_FIELDS
        )

    candidate_runtime_signatures = {runtime_signature(row) for row in candidate_rows}
    reference_runtime_signatures = {runtime_signature(row) for row in reference_rows}
    if len(candidate_runtime_signatures) != 1:
        errors.append(
            "candidate rows were not produced by one locked runtime signature: "
            + json.dumps(
                sorted(candidate_runtime_signatures, key=repr), ensure_ascii=False
            )
        )
    if len(reference_runtime_signatures) != 1:
        errors.append(
            "reference rows were not produced by one locked runtime signature: "
            + json.dumps(
                sorted(reference_runtime_signatures, key=repr), ensure_ascii=False
            )
        )
    if (
        len(candidate_runtime_signatures) == 1
        and len(reference_runtime_signatures) == 1
        and candidate_runtime_signatures != reference_runtime_signatures
    ):
        errors.append(
            "candidate/reference runtime signatures differ: candidate="
            + json.dumps(
                sorted(candidate_runtime_signatures, key=repr), ensure_ascii=False
            )
            + ", reference="
            + json.dumps(
                sorted(reference_runtime_signatures, key=repr), ensure_ascii=False
            )
        )

    cells: list[dict[str, Any]] = []
    for key in EXPECTED_KEYS:
        candidate = candidate_index.get(key)
        reference = reference_index.get(key)
        if candidate is None or reference is None:
            continue
        candidate_tokps = candidate.get("decode_tokps_total_raw")
        reference_tokps = reference.get("decode_tokps_total_raw")
        candidate_parameters = candidate.get("active_parameter_count")
        reference_parameters = reference.get("active_parameter_count")
        if not (
            _is_finite_real(candidate_tokps)
            and float(candidate_tokps) > 0
            and _is_finite_real(reference_tokps)
            and float(reference_tokps) > 0
            and _is_finite_real(candidate_parameters)
            and float(candidate_parameters) > 0
            and _is_finite_real(reference_parameters)
            and float(reference_parameters) > 0
        ):
            errors.append(
                f"cell {key}: cannot derive ratios from invalid raw telemetry"
            )
            continue
        candidate_tokps = float(candidate_tokps)
        reference_tokps = float(reference_tokps)
        candidate_parameters = int(candidate_parameters)
        reference_parameters = int(reference_parameters)
        raw_ratio = candidate_tokps / reference_tokps
        parameter_ratio = candidate_parameters / reference_parameters
        adjusted_ratio = raw_ratio * parameter_ratio
        required_tokps = reference_tokps * reference_parameters / candidate_parameters
        margin_tokps = candidate_tokps - required_tokps
        strict_pass = adjusted_ratio > STRICT_ADJUSTED_DECODE_GATE
        cells.append(
            {
                "axis": PROTOCOL,
                "model_pair": key[0],
                "rwkv_model_size_label": candidate.get("model_size_label"),
                "qwen_model_size_label": reference.get("model_size_label"),
                "device": reference.get("device"),
                "batch_size": key[1],
                "prompt_tokens": key[2],
                "decode_tokens": key[3],
                "candidate_decode_tokps_total_raw": candidate_tokps,
                "reference_decode_tokps_total_raw": reference_tokps,
                "candidate_active_parameter_count": candidate_parameters,
                "reference_active_parameter_count": reference_parameters,
                "raw_decode_ratio": raw_ratio,
                "active_parameter_ratio": parameter_ratio,
                "adjusted_decode_ratio": adjusted_ratio,
                "required_candidate_decode_tokps": required_tokps,
                "candidate_margin_tokps": margin_tokps,
                "candidate_margin_percent": (candidate_tokps / required_tokps - 1.0)
                * 100.0,
                "strict_gate": "adjusted_decode_ratio > 1.0",
                "strict_pass": strict_pass,
                "candidate_decode_backend": candidate.get("effective_backend"),
                "reference_decode_backend": reference.get("step_backend"),
                "reference_axis_composition": reference.get("qwen_axis_composition"),
                "b8_decode_semantics": (
                    "aggregate_throughput_across_8_sequences"
                    if key[1] == 8
                    else "single_sequence_throughput"
                ),
                "continuous_e2e_eligible": False,
                "candidate_source": candidate.get("_source"),
                "reference_source": reference.get("_source"),
                "candidate_line_number": candidate.get("_line_number"),
                "reference_line_number": reference.get("_line_number"),
                "candidate_sha256": candidate_sha256,
                "reference_sha256": reference_sha256_before.lower(),
            }
        )
    cells.sort(key=_sort_key)
    strict_passed = sum(cell["strict_pass"] is True for cell in cells)
    eligible = bool(not errors and len(cells) == 48 and strict_passed == 48)
    if len(cells) == 48 and strict_passed != 48:
        errors.append(
            f"parameter-adjusted Decode gate passed {strict_passed}/48 cells; "
            "every unrounded ratio must be strictly > 1.0"
        )

    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "pass" if eligible else "fail",
        "expected_device": expected_device,
        "frozen_reference": {
            "expected_sha256": expected_hash,
            "sha256_before": reference_sha256_before.lower(),
            "sha256_after": reference_sha256_after.lower(),
            "immutable_during_validation": (
                reference_sha256_before.lower() == reference_sha256_after.lower()
            ),
            "qwen_validation_status": qwen_validation.get("status"),
        },
        "candidate_sha256": candidate_sha256,
        "repository_commits": {
            "candidate": candidate_commits,
            "reference": reference_commits,
        },
        "runtime_fields": list(RUNTIME_FIELDS),
        "candidate_runtime_signatures": [
            list(value) for value in sorted(candidate_runtime_signatures, key=repr)
        ],
        "reference_runtime_signatures": [
            list(value) for value in sorted(reference_runtime_signatures, key=repr)
        ],
        "coverage": {
            "candidate_rows": len(candidate_rows),
            "reference_rows": len(reference_rows),
            "joined_cells": len(cells),
            "expected_cells": 48,
        },
        "gate": {
            "formula": (
                "(candidate_decode_tokps_total_raw / reference_decode_tokps_total_raw) "
                "* (candidate_active_parameter_count / reference_active_parameter_count)"
            ),
            "comparison": ">",
            "threshold": STRICT_ADJUSTED_DECODE_GATE,
            "uses_unrounded_raw_throughput": True,
            "passing_cells": strict_passed,
            "total_cells": 48,
        },
        "sort_order": ["rwkv_model", "device", "batch", "prompt", "decode"],
        "display_rounding": {">=100_tokps": 0, "<100_tokps": 1, "ratios": 6},
        "paired_decode_table_eligible": eligible,
        "continuous_e2e_eligible": False,
        "continuous_e2e_reason": (
            "the frozen Qwen artifact is an independent best-Prefill/best-Decode envelope"
        ),
        "red_cells": [cell for cell in cells if cell["strict_pass"] is not True],
        "cells": cells,
        "errors": errors,
    }


def _display_signed_rate(value: float) -> str:
    rendered = display_rate(abs(value))
    return f"-{rendered}" if value < 0 else rendered


def render_markdown(summary: dict[str, Any]) -> str:
    status = "PASS" if summary["paired_decode_table_eligible"] else "FAIL"
    gate = summary["gate"]
    lines = [
        "# Frozen Qwen3.5 vs RWKV paired Decode v1",
        "",
        f"Status: **{status}**; adjusted Decode: "
        f"**{gate['passing_cells']}/{gate['total_cells']}** cells strictly above 1.0x.",
        "",
        "Adjusted ratio = raw RWKV/Qwen Decode ratio * RWKV/Qwen active-parameter ratio.",
        "The gate uses unrounded `decode_tokps_total_raw`; rendered values are display-only.",
        "B8 Decode is aggregate throughput across eight sequences. This is not a continuous E2E route.",
        "",
        "| RWKV / Qwen | GPU | B | P | D | RWKV tok/s | Qwen tok/s | Raw | Param | Adjusted | Required RWKV tok/s | Margin tok/s | Pass |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for cell in summary["cells"]:
        pair = str(cell["model_pair"]).replace("rwkv-", "").replace("__qwen3.5-", " / ")
        lines.append(
            "| {pair} | {device} | {batch} | {prompt} | {decode} | {candidate} | "
            "{reference} | {raw:.6f}x | {parameters:.6f}x | {adjusted:.6f}x | "
            "{required} | {margin} | {passed} |".format(
                pair=pair,
                device=cell["device"],
                batch=cell["batch_size"],
                prompt=cell["prompt_tokens"],
                decode=cell["decode_tokens"],
                candidate=display_rate(cell["candidate_decode_tokps_total_raw"]),
                reference=display_rate(cell["reference_decode_tokps_total_raw"]),
                raw=cell["raw_decode_ratio"],
                parameters=cell["active_parameter_ratio"],
                adjusted=cell["adjusted_decode_ratio"],
                required=display_rate(cell["required_candidate_decode_tokps"]),
                margin=_display_signed_rate(cell["candidate_margin_tokps"]),
                passed="PASS" if cell["strict_pass"] else "FAIL",
            )
        )
    if summary["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {message}" for message in summary["errors"])
    return "\n".join(lines) + "\n"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _evidence_path(
    manifest: Path,
    value: Any,
    label: str,
    errors: list[str],
) -> Path | None:
    if type(value) is not dict:
        errors.append(f"{label} must be an evidence object")
        return None
    raw_path = value.get("path")
    expected_sha = value.get("sha256")
    if type(raw_path) is not str or not raw_path or Path(raw_path).name != raw_path:
        errors.append(f"{label}.path must be one local artifact basename")
        return None
    if (
        type(expected_sha) is not str
        or len(expected_sha) != 64
        or any(character not in HEX_DIGITS for character in expected_sha.lower())
    ):
        errors.append(f"{label}.sha256 must be 64 hexadecimal characters")
        return None
    path = manifest.parent / raw_path
    try:
        actual_sha = _sha256(path.read_bytes())
    except OSError as exc:
        errors.append(f"{label} could not be read: {exc}")
        return None
    if actual_sha != expected_sha.lower():
        errors.append(
            f"{label} SHA256 mismatch: observed {actual_sha}, expected {expected_sha.lower()}"
        )
        return None
    return path


def _validate_ab_row_common(
    row: dict[str, Any],
    *,
    pair: str,
    size: str,
    lane: str,
    commit: str,
    runtime: dict[str, str],
    errors: list[str],
) -> None:
    for field, expected in (
        ("axis", "qwen35_cross_model_speed"),
        ("benchmark_matrix", SM120_AB_PROTOCOL),
        ("optimization_lane", lane),
        ("benchmark_repository_commit", commit),
        ("model_pair", pair),
        ("model_size_label", size),
        ("model_role", "candidate"),
        ("model_kind", "rwkv"),
        ("dtype", "fp16"),
        ("quantization", "none"),
        ("quantization_backend", "dense"),
        ("native_quant_kernel_active", False),
        ("batch_size", 8),
        ("prompt_tokens", 2048),
        ("decode_tokens", 512),
        ("prefill_chunk_size", 512),
        ("warmup", 3),
        ("runs", 7),
        ("timing_statistic", "median"),
        ("resident_sweep", True),
        ("status", "pass"),
        ("logits_finite", True),
        ("device", EXPECTED_DEVICE),
        ("gpu_arch", "sm_120"),
        ("gpu_compute_capability", [12, 0]),
        ("rwkv_fast_token_backend_requested", "native_graph"),
        ("rwkv_native_model_backend_requested", "native_graph"),
        ("effective_backend", "native_graph"),
        ("step_backend", "rwkv_fast_token"),
        ("cache_type", "NativeRWKV7Cache"),
        ("probe_tokens", 512),
        ("probe_batch_size", 8),
        ("probe_distinct_batch_prompts", True),
        ("probe_decode_logits_all_finite", True),
    ):
        _require(row, field, expected, errors)
    for field, expected in runtime.items():
        _require(row, field, expected, errors)
    samples = row.get("decode_sec_samples")
    if (
        type(samples) is not list
        or len(samples) != 7
        or not all(_is_finite_real(value) and float(value) > 0 for value in samples)
    ):
        errors.append(f"{row.get('_source', '<row>')}: invalid 7-run Decode samples")
        return
    raw_median = row.get("decode_sec_median_raw")
    raw_tokps = row.get("decode_tokps_total_raw")
    expected_median = float(statistics.median(samples))
    if not _is_finite_real(raw_median) or float(raw_median) != expected_median:
        errors.append(f"{row.get('_source', '<row>')}: invalid raw Decode median")
    if not _is_finite_real(raw_tokps) or not math.isclose(
        float(raw_tokps), 8 * 512 / expected_median, rel_tol=1e-12, abs_tol=1e-12
    ):
        errors.append(f"{row.get('_source', '<row>')}: invalid raw Decode tok/s")


def _recorded_path(value: Any) -> str | None:
    """Normalize an artifact-recorded path without requiring it to exist locally."""

    if type(value) is not str or not value.strip():
        return None
    return value.replace("\\", "/").rstrip("/")


def _parse_model_hash_sections(
    path: Path, errors: list[str]
) -> dict[str, dict[str, str]]:
    """Parse the immutable model-input manifest used by the formal runner."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        errors.append(f"model hashes evidence could not be read: {exc}")
        return {}
    sections: dict[str, dict[str, str]] = {}
    current: str | None = None
    for line_number, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = _recorded_path(line[1:-1])
            if current is None:
                errors.append(f"{path}:{line_number}: empty model hash section")
                continue
            if current in sections:
                errors.append(
                    f"{path}:{line_number}: duplicate model hash section {current!r}"
                )
            sections.setdefault(current, {})
            continue
        if current is None:
            errors.append(f"{path}:{line_number}: hash appears before a model section")
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            errors.append(f"{path}:{line_number}: malformed model hash entry")
            continue
        digest, filename = parts
        filename = filename.strip()
        relative_path = Path(filename.replace("\\", "/"))
        if (
            len(digest) != 64
            or any(character not in HEX_DIGITS for character in digest.lower())
            or not filename
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or filename in sections[current]
        ):
            errors.append(f"{path}:{line_number}: invalid model hash entry")
            continue
        sections[current][filename] = digest.lower()
    for model_path, files in sections.items():
        if "config.json" not in files:
            errors.append(f"{path}: {model_path!r} is missing config.json hash")
        if not any(name.endswith(".safetensors") for name in files):
            errors.append(f"{path}: {model_path!r} is missing safetensors hashes")
        if not any(
            name
            in {
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt",
            }
            for name in files
        ):
            errors.append(f"{path}: {model_path!r} is missing tokenizer hashes")
    return sections


def _validate_sm120_ab_evidence(
    *,
    manifest_path: Path,
    model_hashes_path: Path,
    candidate_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"status": "fail"}, [f"SM120 A/B manifest could not be read: {exc}"]
    if type(manifest) is not dict:
        return {"status": "fail"}, ["SM120 A/B manifest must be a JSON object"]

    commits = {
        row.get("benchmark_repository_commit")
        for row in candidate_rows
        if type(row.get("benchmark_repository_commit")) is str
    }
    commit = next(iter(commits)) if len(commits) == 1 else ""
    for field, expected in (
        ("schema_version", 1),
        ("protocol", SM120_AB_PROTOCOL),
        ("benchmark_repository_commit", commit),
        (
            "cell",
            {
                "batch_size": 8,
                "prompt_tokens": 2048,
                "decode_tokens": 512,
                "probe_tokens": 512,
                "distinct_batch_prompts": True,
            },
        ),
    ):
        _require({**manifest, "_source": str(manifest_path)}, field, expected, errors)
    try:
        model_hashes_sha = _sha256(model_hashes_path.read_bytes())
    except OSError as exc:
        errors.append(f"model hashes evidence could not be read: {exc}")
        model_hashes_sha = ""
    if manifest.get("model_hashes_sha256") != model_hashes_sha:
        errors.append("SM120 A/B manifest model_hashes_sha256 mismatch")
    model_hash_sections = _parse_model_hash_sections(model_hashes_path, errors)

    main_model_paths: dict[str, str] = {}
    for pair in PARAMETERS:
        paths = {
            normalized
            for row in candidate_rows
            if row.get("model_pair") == pair
            and (normalized := _recorded_path(row.get("model_id_or_path"))) is not None
        }
        if len(paths) != 1:
            errors.append(
                f"candidate {pair}: expected exactly one model_id_or_path, got {sorted(paths)!r}"
            )
        else:
            main_model_paths[pair] = next(iter(paths))
    if set(model_hash_sections) != set(main_model_paths.values()):
        errors.append(
            "model hash sections must match exactly the four candidate model paths"
        )

    first_candidate = candidate_rows[0] if candidate_rows else {}
    runtime = {field: first_candidate.get(field) for field in RUNTIME_FIELDS}
    expected_entries = {
        "rwkv-0.4b__qwen3.5-0.8b": "0.4b",
        "rwkv-1.5b__qwen3.5-2b": "1.5b",
    }
    entries = manifest.get("entries")
    if type(entries) is not list:
        errors.append("SM120 A/B manifest entries must be a list")
        entries = []
    by_pair = {
        entry.get("model_pair"): entry
        for entry in entries
        if type(entry) is dict and type(entry.get("model_pair")) is str
    }
    if set(by_pair) != set(expected_entries) or len(entries) != 2:
        errors.append(
            "SM120 A/B manifest must contain exactly the 0.4B and 1.5B entries"
        )

    evidence_summary: list[dict[str, Any]] = []
    for pair, size in expected_entries.items():
        entry = by_pair.get(pair)
        if type(entry) is not dict:
            continue
        if entry.get("model_size_label") != size:
            errors.append(f"SM120 A/B {pair}: wrong model_size_label")
        loaded_rows: dict[str, dict[str, Any]] = {}
        loaded_probes: dict[str, dict[str, Any]] = {}
        artifact_summary: dict[str, Any] = {"model_pair": pair}
        for lane in ("baseline", "candidate"):
            lane_value = entry.get(lane)
            if type(lane_value) is not dict:
                errors.append(f"SM120 A/B {pair} {lane} must be an object")
                continue
            row_path = _evidence_path(
                manifest_path, lane_value.get("row"), f"{pair}.{lane}.row", errors
            )
            probe_path = _evidence_path(
                manifest_path,
                lane_value.get("probe"),
                f"{pair}.{lane}.probe",
                errors,
            )
            if row_path is not None:
                try:
                    rows = _read_jsonl_bytes(row_path.read_bytes(), row_path)
                except (UnicodeDecodeError, ValueError) as exc:
                    errors.append(f"{row_path}: invalid row evidence: {exc}")
                    rows = []
                if len(rows) != 1:
                    errors.append(f"{row_path}: expected exactly one A/B row")
                else:
                    row = rows[0]
                    loaded_rows[lane] = row
                    _validate_ab_row_common(
                        row,
                        pair=pair,
                        size=size,
                        lane=lane,
                        commit=commit,
                        runtime=runtime,
                        errors=errors,
                    )
                    row_model_path = _recorded_path(row.get("model_id_or_path"))
                    if row_model_path != main_model_paths.get(pair):
                        errors.append(
                            f"{row_path}: model_id_or_path {row_model_path!r} does not "
                            f"match promoted {pair} path {main_model_paths.get(pair)!r}"
                        )
                    row_probe_path = _recorded_path(row.get("probe_output"))
                    if probe_path is not None and (
                        row_probe_path is None
                        or row_probe_path.rsplit("/", 1)[-1] != probe_path.name
                    ):
                        errors.append(
                            f"{row_path}: probe_output does not bind manifest probe {probe_path.name!r}"
                        )
                    if lane == "candidate":
                        promoted = dict(row)
                        promoted["benchmark_matrix"] = CANDIDATE_MATRIX
                        promoted["optimization_lane"] = CANDIDATE_LANE
                        _validate_candidate_row(
                            promoted, expected_device=EXPECTED_DEVICE, errors=errors
                        )
                    else:
                        for field, expected in (
                            ("rwkv_native_graph_ada_wagv_bmm_requested", False),
                            ("rwkv_native_graph_ada_wagv_bmm_selected", False),
                            ("rwkv_native_graph_ada_wagv_bmm_effective", False),
                            ("rwkv_native_graph_ada_wagv_bmm_effective_layer_count", 0),
                            (
                                "rwkv_native_graph_ada_wagv_bmm_full_model_effective",
                                False,
                            ),
                        ):
                            _require(row, field, expected, errors)
                        for route in ("sm120_wagv_bmm_g", "sm120_compiled_ffn"):
                            for suffix, expected in (
                                ("requested", False),
                                ("selected", False),
                                ("effective", False),
                                ("selected_layers", []),
                                ("effective_layers", []),
                                ("effective_layer_count", 0),
                                ("full_model_effective", False),
                            ):
                                _require(
                                    row,
                                    f"rwkv_native_graph_{route}_{suffix}",
                                    expected,
                                    errors,
                                )
            if probe_path is not None:
                try:
                    probe = torch.load(
                        probe_path, map_location="cpu", weights_only=True
                    )
                except (
                    EOFError,
                    OSError,
                    pickle.UnpicklingError,
                    RuntimeError,
                    TypeError,
                    ValueError,
                ) as exc:
                    errors.append(f"{probe_path}: invalid probe payload: {exc}")
                    probe = None
                if type(probe) is dict:
                    loaded_probes[lane] = probe
                    for field, expected in (
                        ("probe_schema_version", 2),
                        ("benchmark_repository_commit", commit),
                        ("model_pair", pair),
                        ("model_size_label", size),
                        ("model_id_or_path", main_model_paths.get(pair)),
                    ):
                        _require(
                            {**probe, "_source": str(probe_path)},
                            field,
                            expected,
                            errors,
                        )
                    recorded_probe_output = _recorded_path(probe.get("probe_output"))
                    if (
                        recorded_probe_output is None
                        or recorded_probe_output.rsplit("/", 1)[-1] != probe_path.name
                    ):
                        errors.append(
                            f"{probe_path}: payload probe_output does not bind its artifact"
                        )
                    input_ids = probe.get("input_ids")
                    greedy = probe.get("greedy_tokens")
                    decode_finite = probe.get("decode_logits_finite_by_batch")
                    prompt = probe.get("prompt_logits")
                    final = probe.get("final_logits")
                    if not isinstance(input_ids, torch.Tensor) or tuple(
                        input_ids.shape
                    ) != (8, 2048):
                        errors.append(
                            f"{probe_path}: input_ids shape must be [8, 2048]"
                        )
                    elif torch.unique(input_ids, dim=0).shape[0] != 8:
                        errors.append(
                            f"{probe_path}: batch prompts must all be distinct"
                        )
                    if not isinstance(greedy, torch.Tensor) or tuple(greedy.shape) != (
                        512,
                        8,
                    ):
                        errors.append(
                            f"{probe_path}: greedy_tokens shape must be [512, 8]"
                        )
                    if (
                        not isinstance(decode_finite, torch.Tensor)
                        or tuple(decode_finite.shape) != (8,)
                        or not bool(decode_finite.bool().all())
                        or probe.get("decode_logits_all_finite") is not True
                    ):
                        errors.append(
                            f"{probe_path}: all 512 Decode logits must be finite for every batch row"
                        )
                    for label, logits in (("prompt", prompt), ("final", final)):
                        if (
                            not isinstance(logits, torch.Tensor)
                            or logits.dim() != 2
                            or logits.shape[0] != 8
                            or logits.numel() == 0
                            or not bool(torch.isfinite(logits).all())
                        ):
                            errors.append(
                                f"{probe_path}: {label} logits must be finite [8, vocab]"
                            )
                else:
                    errors.append(f"{probe_path}: probe payload must be a dictionary")
        comparison_path = _evidence_path(
            manifest_path, entry.get("comparison"), f"{pair}.comparison", errors
        )
        recomputed: dict[str, Any] | None = None
        if set(loaded_probes) == {"baseline", "candidate"}:
            recomputed = compare_rwkv_probes(
                loaded_probes["baseline"], loaded_probes["candidate"], 0.9999
            )
            if recomputed.get("status") != "pass":
                errors.append(f"SM120 A/B {pair}: recomputed probe comparison failed")
        if comparison_path is not None:
            try:
                recorded = json.loads(comparison_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                errors.append(f"{comparison_path}: invalid comparison JSON: {exc}")
                recorded = {}
            for field, expected in (
                ("status", "pass"),
                ("min_cosine_required", 0.9999),
                ("input_ids_match", True),
                ("greedy_tokens_match", True),
                ("prompt_logits_shape_match", True),
                ("prompt_logits_finite", True),
                ("final_logits_shape_match", True),
                ("final_logits_finite", True),
                ("probe_batch_size", 8),
                ("probe_tokens", 512),
                ("distinct_batch_prompts", True),
                ("decode_finite_shape_match", True),
                ("reference_decode_logits_all_finite", True),
                ("native_decode_logits_all_finite", True),
                ("contract_errors", []),
            ):
                _require(
                    {**recorded, "_source": str(comparison_path)},
                    field,
                    expected,
                    errors,
                )
            for field in ("prompt_logits_cosine", "final_logits_cosine"):
                value = recorded.get(field)
                if not _is_finite_real(value) or float(value) < 0.9999:
                    errors.append(
                        f"{comparison_path}: {field} must be finite >= 0.9999"
                    )
            if recomputed is not None:
                for field in (
                    "status",
                    "input_ids_match",
                    "greedy_tokens_match",
                    "prompt_logits_cosine",
                    "final_logits_cosine",
                ):
                    if not _strict_equal(recorded.get(field), recomputed.get(field)):
                        errors.append(
                            f"{comparison_path}: {field} differs from recomputed probes"
                        )
        if set(loaded_rows) == {"baseline", "candidate"}:
            baseline_rate = loaded_rows["baseline"].get("decode_tokps_total_raw")
            candidate_rate = loaded_rows["candidate"].get("decode_tokps_total_raw")
            if (
                not _is_finite_real(baseline_rate)
                or not _is_finite_real(candidate_rate)
                or float(candidate_rate) <= float(baseline_rate)
            ):
                errors.append(
                    f"SM120 A/B {pair}: candidate raw Decode must be strictly faster "
                    f"than baseline, got {candidate_rate!r} <= {baseline_rate!r}"
                )
            artifact_summary.update(
                {
                    "baseline_decode_tokps": baseline_rate,
                    "candidate_decode_tokps": candidate_rate,
                }
            )
        if recomputed is not None:
            artifact_summary.update(
                {
                    "prompt_logits_min_cosine": recomputed.get("prompt_logits_cosine"),
                    "final_logits_min_cosine": recomputed.get("final_logits_cosine"),
                    "greedy_tokens_match": recomputed.get("greedy_tokens_match"),
                }
            )
        evidence_summary.append(artifact_summary)

    return {
        "status": "pass" if not errors else "fail",
        "protocol": SM120_AB_PROTOCOL,
        "manifest": str(manifest_path),
        "model_hashes": str(model_hashes_path),
        "entries": evidence_summary,
    }, errors


def _row_payload(row: dict[str, Any]) -> dict[str, Any]:
    """Return the JSON artifact payload without validator-only source metadata."""

    return {
        key: value
        for key, value in row.items()
        if key not in {"_source", "_line_number"}
    }


def _read_single_evidence_row(
    path: Path, label: str, errors: list[str]
) -> dict[str, Any] | None:
    try:
        rows = _read_jsonl_bytes(path.read_bytes(), path)
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        errors.append(f"{label} is not valid JSONL row evidence: {exc}")
        return None
    if len(rows) != 1:
        errors.append(f"{label} must contain exactly one JSON object, got {len(rows)}")
        return None
    return rows[0]


def _load_decode_probe(
    path: Path,
    *,
    pair: str,
    size: str,
    model_path: str | None,
    commit: str,
    batch: int,
    errors: list[str],
) -> dict[str, Any] | None:
    try:
        probe = torch.load(path, map_location="cpu", weights_only=True)
    except (
        EOFError,
        OSError,
        pickle.UnpicklingError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        errors.append(f"{path}: invalid Decode correctness probe payload: {exc}")
        return None
    if type(probe) is not dict:
        errors.append(f"{path}: Decode correctness probe payload must be a dictionary")
        return None

    source = {**probe, "_source": str(path)}
    for field, expected in (
        ("probe_schema_version", 2),
        ("benchmark_repository_commit", commit),
        ("model_pair", pair),
        ("model_size_label", size),
        ("model_id_or_path", model_path),
        ("decode_logits_all_finite", True),
    ):
        _require(source, field, expected, errors)
    recorded_output = _recorded_path(probe.get("probe_output"))
    if recorded_output is None or recorded_output.rsplit("/", 1)[-1] != path.name:
        errors.append(f"{path}: payload probe_output does not bind its artifact")

    input_ids = probe.get("input_ids")
    greedy = probe.get("greedy_tokens")
    decode_finite = probe.get("decode_logits_finite_by_batch")
    expected_greedy_shape = (512,) if batch == 1 else (512, batch)
    if (
        not isinstance(input_ids, torch.Tensor)
        or tuple(input_ids.shape) != (batch, 2048)
        or input_ids.dtype.is_floating_point
    ):
        errors.append(f"{path}: input_ids must be an integer [{batch}, 2048] tensor")
    elif batch == 8 and int(torch.unique(input_ids, dim=0).shape[0]) != 8:
        errors.append(f"{path}: all B8 prompt rows must be distinct")
    if (
        not isinstance(greedy, torch.Tensor)
        or tuple(greedy.shape) != expected_greedy_shape
        or greedy.dtype.is_floating_point
    ):
        errors.append(
            f"{path}: greedy_tokens must be an integer {expected_greedy_shape} tensor"
        )
    if (
        not isinstance(decode_finite, torch.Tensor)
        or tuple(decode_finite.shape) != (batch,)
        or not bool(decode_finite.bool().all())
    ):
        errors.append(
            f"{path}: every one of 512 Decode steps must remain finite for every batch row"
        )
    for label, logits in (
        ("prompt", probe.get("prompt_logits")),
        ("final", probe.get("final_logits")),
    ):
        if (
            not isinstance(logits, torch.Tensor)
            or logits.dim() != 2
            or logits.shape[0] != batch
            or logits.numel() == 0
            or not bool(torch.isfinite(logits).all())
        ):
            errors.append(f"{path}: {label} logits must be finite [{batch}, vocab]")
    return probe


def _validate_decode_row_probe_binding(
    row: dict[str, Any],
    probe: dict[str, Any],
    probe_path: Path,
    *,
    batch: int,
    errors: list[str],
) -> None:
    row_output = _recorded_path(row.get("probe_output"))
    if row_output is None or row_output.rsplit("/", 1)[-1] != probe_path.name:
        errors.append(
            f"{row.get('_source', '<row>')}: probe_output does not bind "
            f"manifest probe {probe_path.name!r}"
        )
    raw_output = row.get("probe_output")
    if type(raw_output) is not str or not (
        raw_output.startswith(("/", "\\\\"))
        or (len(raw_output) >= 3 and raw_output[1:3] in {":/", ":\\"})
    ):
        errors.append(
            f"{row.get('_source', '<row>')}: probe_output must record an absolute path"
        )
    for field, expected in (
        ("probe_tokens", 512),
        ("probe_batch_size", batch),
        ("probe_distinct_batch_prompts", batch == 8),
        ("probe_decode_logits_all_finite", True),
        ("probe_decode_logits_finite_by_batch", [True] * batch),
    ):
        _require(row, field, expected, errors)
    greedy = probe.get("greedy_tokens")
    recorded_greedy = row.get("probe_greedy_tokens")
    expected_greedy = greedy.tolist() if isinstance(greedy, torch.Tensor) else None
    if not _strict_equal(recorded_greedy, expected_greedy):
        errors.append(
            f"{row.get('_source', '<row>')}: probe_greedy_tokens does not bind "
            "the probe payload"
        )


def _validate_decode_correctness_row(
    row: dict[str, Any],
    *,
    pair: str,
    size: str,
    model_path: str | None,
    batch: int,
    lane: str,
    commit: str,
    runtime: dict[str, Any],
    errors: list[str],
) -> None:
    is_reference = lane == "fla_reference"
    for field, expected in (
        ("axis", "qwen35_cross_model_speed"),
        (
            "benchmark_matrix",
            DECODE_CORRECTNESS_PROTOCOL if is_reference else CANDIDATE_MATRIX,
        ),
        ("optimization_lane", lane if is_reference else CANDIDATE_LANE),
        ("benchmark_repository_commit", commit),
        ("model_pair", pair),
        ("model_size_label", size),
        ("model_id_or_path", model_path),
        ("model_role", "candidate"),
        ("model_kind", "rwkv"),
        ("dtype", "fp16"),
        ("quantization", "none"),
        ("quantization_backend", "dense"),
        ("native_quant_kernel_active", False),
        ("active_parameter_count", PARAMETERS[pair][0]),
        ("batch_size", batch),
        ("prompt_tokens", 2048),
        ("decode_tokens", 512),
        ("prefill_chunk_size", 512),
        ("timing_statistic", "median"),
        ("resident_sweep", True),
        ("resident_cell_index", 1 if is_reference else 6),
        ("resident_cells_total", 1 if is_reference else 6),
        ("resident_probe_cell", [batch, 2048, 512]),
        ("resident_probe_cell_selected", True),
        ("status", "pass"),
        ("logits_finite", True),
        ("device", EXPECTED_DEVICE),
        ("gpu_arch", "sm_120"),
        ("gpu_compute_capability", [12, 0]),
        ("step_backend", "rwkv_fast_token"),
    ):
        _require(row, field, expected, errors)
    for field, expected in runtime.items():
        _require(row, field, expected, errors)

    if is_reference:
        for field, expected in (
            ("warmup", 1),
            ("runs", 1),
            ("rwkv_implementation_requested", "wrapper_repo"),
            ("rwkv_implementation_effective", "wrapper_repo"),
            ("rwkv_fast_token_backend_requested", "fla"),
            ("rwkv_native_model_backend_requested", "eager"),
            ("rwkv_fast_prefill_requested", "0"),
            ("rwkv_prefill_graph_requested", "0"),
            ("effective_backend", "fla"),
            ("prefill_backend_effective", None),
            ("cache_type", "RWKV7StateCache"),
        ):
            _require(row, field, expected, errors)
        if row.get("prefill_backend_effective") in {
            "native_prefill",
            "native_prefill_graph",
        }:
            errors.append(
                f"{row.get('_source', '<row>')}: FLA reference used native prefill"
            )
        # The FLA lane never constructs a native-graph runner, so every
        # native-graph route field is unavailable rather than false/empty.
        for field in (
            "rwkv_native_graph_ada_wagv_bmm_requested",
            "rwkv_native_graph_ada_wagv_bmm_selected",
            "rwkv_native_graph_ada_wagv_bmm_effective",
            "rwkv_native_graph_ada_wagv_bmm_effective_layer_count",
            "rwkv_native_graph_ada_wagv_bmm_full_model_effective",
            "rwkv_native_graph_sm120_wagv_bmm_g_requested",
            "rwkv_native_graph_sm120_wagv_bmm_g_selected",
            "rwkv_native_graph_sm120_wagv_bmm_g_effective",
            "rwkv_native_graph_sm120_wagv_bmm_g_selected_layers",
            "rwkv_native_graph_sm120_wagv_bmm_g_effective_layers",
            "rwkv_native_graph_sm120_wagv_bmm_g_effective_layer_count",
            "rwkv_native_graph_sm120_wagv_bmm_g_full_model_effective",
            "rwkv_native_graph_sm120_compiled_ffn_requested",
            "rwkv_native_graph_sm120_compiled_ffn_selected",
            "rwkv_native_graph_sm120_compiled_ffn_effective",
            "rwkv_native_graph_sm120_compiled_ffn_selected_layers",
            "rwkv_native_graph_sm120_compiled_ffn_effective_layers",
            "rwkv_native_graph_sm120_compiled_ffn_effective_layer_count",
            "rwkv_native_graph_sm120_compiled_ffn_full_model_effective",
            "rwkv_native_graph_sm120_compiled_ffn_compile_effective",
            "rwkv_native_graph_sm120_compiled_ffn_compile_reused",
            "rwkv_native_graph_sm120_compiled_ffn_unique_graphs",
            "rwkv_native_graph_sm120_compiled_ffn_graph_breaks",
            "rwkv_native_graph_sm120_compiled_ffn_compile_mode",
            "rwkv_native_graph_sm120_compiled_ffn_prewarm_all_finite",
            "rwkv_native_graph_sm120_compiled_ffn_prewarm_min_cosine",
            "rwkv_native_graph_sm120_compiled_ffn_prewarm_argmax_all_equal",
            "rwkv_native_graph_sm120_compiled_ffn_prewarm_max_abs_diff",
            "rwkv_native_graph_sm120_compiled_ffn_prewarm_layer_indices",
            "rwkv_native_graph_sm120_compiled_ffn_prewarm_layer_count",
        ):
            _require(row, field, None, errors)
    else:
        _validate_candidate_row(row, expected_device=EXPECTED_DEVICE, errors=errors)

    runs = row.get("runs")
    samples = row.get("decode_sec_samples")
    if (
        type(runs) is not int
        or type(samples) is not list
        or len(samples) != runs
        or not all(_is_finite_real(value) and float(value) > 0 for value in samples)
    ):
        errors.append(
            f"{row.get('_source', '<row>')}: invalid raw Decode timing samples"
        )
        return
    median = float(statistics.median(samples))
    raw_median = row.get("decode_sec_median_raw")
    raw_tokps = row.get("decode_tokps_total_raw")
    if not _is_finite_real(raw_median) or float(raw_median) != median:
        errors.append(f"{row.get('_source', '<row>')}: invalid raw Decode median")
    expected_tokps = batch * 512 / median
    if not _is_finite_real(raw_tokps) or not math.isclose(
        float(raw_tokps), expected_tokps, rel_tol=1e-12, abs_tol=1e-12
    ):
        errors.append(f"{row.get('_source', '<row>')}: invalid raw Decode tok/s")


def _validate_decode_correctness_evidence(
    *,
    manifest_path: Path,
    model_hashes_path: Path,
    runtime_lock_path: Path,
    candidate_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """Validate the formal 4-model, B1/B8 native_graph-vs-FLA oracle."""

    errors: list[str] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"status": "fail"}, [
            f"native_graph-vs-FLA correctness manifest could not be read: {exc}"
        ]
    if type(manifest) is not dict:
        return {"status": "fail"}, [
            "native_graph-vs-FLA correctness manifest must be a JSON object"
        ]

    commits = {
        row.get("benchmark_repository_commit")
        for row in candidate_rows
        if type(row.get("benchmark_repository_commit")) is str
    }
    commit = next(iter(commits)) if len(commits) == 1 else ""
    top_contract = (
        ("schema_version", 1),
        ("protocol", DECODE_CORRECTNESS_PROTOCOL),
        ("benchmark_repository_commit", commit),
        (
            "coverage",
            {
                "models": 4,
                "batch_sizes": [1, 8],
                "entries": 8,
                "baseline_fresh_gpu_processes": 8,
                "candidate_additional_gpu_processes": 0,
                "candidate_formal_lane_processes": 8,
                "prompt_tokens": 2048,
                "decode_tokens": 512,
                "probe_tokens": 512,
            },
        ),
        (
            "reference_contract",
            {
                "rwkv_implementation": "wrapper_repo",
                "RWKV7_FAST_TOKEN_BACKEND": "fla",
                "RWKV7_NATIVE_MODEL_BACKEND": "eager",
                "RWKV7_FAST_PREFILL": 0,
                "RWKV7_NATIVE_PREFILL_GRAPH": 0,
            },
        ),
        (
            "candidate_contract",
            {
                "rwkv_implementation": "auto",
                "RWKV7_FAST_TOKEN_BACKEND": "native_graph",
                "RWKV7_NATIVE_MODEL_BACKEND": "native_graph",
                "RWKV7_FAST_PREFILL": "unset_exact_card_policy",
                "RWKV7_NATIVE_PREFILL_GRAPH": "unset_exact_card_policy",
                "small_model_b8_promoted_bundle": True,
            },
        ),
        (
            "gates",
            {
                "greedy_tokens": "exact_all_512",
                "prompt_logits_min_row_cosine": 0.9999,
                "final_logits_min_row_cosine": 0.9999,
                "decode_logits_all_finite": True,
                "b8_distinct_prompts": True,
            },
        ),
    )
    manifest_source = {**manifest, "_source": str(manifest_path)}
    for field, expected in top_contract:
        _require(manifest_source, field, expected, errors)

    try:
        model_hashes_sha = _sha256(model_hashes_path.read_bytes())
    except OSError as exc:
        errors.append(f"model hashes evidence could not be read: {exc}")
        model_hashes_sha = ""
    if manifest.get("model_hashes_sha256") != model_hashes_sha:
        errors.append("Decode correctness manifest model_hashes_sha256 mismatch")
    model_hash_sections = _parse_model_hash_sections(model_hashes_path, errors)

    runtime_path = _evidence_path(
        manifest_path, manifest.get("runtime"), "Decode correctness runtime", errors
    )
    if runtime_path is not None and _resolved(runtime_path) != _resolved(
        runtime_lock_path
    ):
        errors.append(
            "Decode correctness runtime artifact does not bind the supplied runtime lock"
        )
    runtime_lock: dict[str, Any] = {}
    if runtime_path is not None:
        try:
            decoded_runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"{runtime_path}: invalid runtime lock JSON: {exc}")
        else:
            if type(decoded_runtime) is not dict:
                errors.append(f"{runtime_path}: runtime lock must be an object")
            else:
                runtime_lock = decoded_runtime
    locked_versions = {
        "python": "3.10.12",
        "torch": "2.8.0+cu128",
        "torch_cuda": "12.8",
        "triton": "3.4.0",
        "transformers": "5.12.1",
        "fla": "0.5.1",
        "causal_conv1d": "1.6.2.post1",
    }
    for field, expected in (
        ("schema_version", 1),
        ("protocol", PROTOCOL),
        ("repository_commit", commit),
        ("runtime", locked_versions),
        (
            "pip_freeze_sha256",
            "f5bf8ef181f2c1b29b79d6fae5c8019fa85008df120569b9e18646bd09eee5cf",
        ),
        ("torch_cuda_arch_list", "12.0"),
    ):
        _require(
            {**runtime_lock, "_source": str(runtime_path or manifest_path)},
            field,
            expected,
            errors,
        )

    main_model_paths: dict[str, str] = {}
    for pair in PARAMETERS:
        paths = {
            normalized
            for row in candidate_rows
            if row.get("model_pair") == pair
            and (normalized := _recorded_path(row.get("model_id_or_path"))) is not None
        }
        if len(paths) != 1:
            errors.append(
                f"candidate {pair}: expected exactly one model_id_or_path, got {sorted(paths)!r}"
            )
        else:
            main_model_paths[pair] = next(iter(paths))
    if set(model_hash_sections) != set(main_model_paths.values()):
        errors.append(
            "model hash sections must match exactly the four candidate model paths"
        )

    runtime = {
        field: candidate_rows[0].get(field) if candidate_rows else None
        for field in RUNTIME_FIELDS
    }
    candidate_index = {
        _cell_key(row): row for row in candidate_rows if _cell_key(row) in EXPECTED_KEYS
    }
    expected_entry_keys = {(pair, batch) for pair in PARAMETERS for batch in (1, 8)}
    entries = manifest.get("entries")
    if type(entries) is not list:
        errors.append("Decode correctness manifest entries must be a list")
        entries = []
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    entry_counts: Counter[tuple[str, int]] = Counter()
    for index, entry in enumerate(entries):
        if type(entry) is not dict:
            errors.append(f"Decode correctness entry {index} must be an object")
            continue
        pair = entry.get("model_pair")
        batch = entry.get("batch_size")
        if type(pair) is not str or type(batch) is not int:
            errors.append(f"Decode correctness entry {index} has an invalid key")
            continue
        key = (pair, batch)
        entry_counts[key] += 1
        if entry_counts[key] == 1:
            by_key[key] = entry
    duplicates = sorted(key for key, count in entry_counts.items() if count > 1)
    if len(entries) != 8 or set(by_key) != expected_entry_keys or duplicates:
        errors.append(
            "Decode correctness manifest must contain exactly one entry for each "
            f"4-model x B1/B8 key; duplicates={duplicates!r}"
        )

    used_artifacts: set[Path] = (
        {runtime_path.resolve(strict=False)} if runtime_path is not None else set()
    )

    def evidence_path(value: Any, label: str) -> Path | None:
        path = _evidence_path(manifest_path, value, label, errors)
        if path is None:
            return None
        resolved = path.resolve(strict=False)
        if resolved in used_artifacts:
            errors.append(f"{label} reuses an artifact already bound elsewhere")
        used_artifacts.add(resolved)
        return path

    expected_source_shapes = [
        (prompt, decode) for prompt in (128, 512, 2048) for decode in (128, 512)
    ]
    evidence_summary: list[dict[str, Any]] = []
    for pair in PARAMETERS:
        size = RWKV_PAIR_SIZES[pair]
        model_path = main_model_paths.get(pair)
        for batch in (1, 8):
            entry = by_key.get((pair, batch))
            if type(entry) is not dict:
                continue
            entry_source = {**entry, "_source": f"{manifest_path}:{pair}:B{batch}"}
            for field, expected in (
                ("model_pair", pair),
                ("model_size_label", size),
                ("model_path", model_path),
                ("batch_size", batch),
                ("prompt_tokens", 2048),
                ("decode_tokens", 512),
                ("probe_tokens", 512),
            ):
                _require(entry_source, field, expected, errors)

            loaded_rows: dict[str, dict[str, Any]] = {}
            loaded_probes: dict[str, dict[str, Any]] = {}
            artifact_summary: dict[str, Any] = {
                "model_pair": pair,
                "batch_size": batch,
            }
            for lane in ("fla_reference", "native_candidate"):
                lane_value = entry.get(lane)
                if type(lane_value) is not dict:
                    errors.append(f"{pair} B{batch} {lane} must be an object")
                    continue
                row_path = evidence_path(
                    lane_value.get("row"), f"{pair}.B{batch}.{lane}.row"
                )
                probe_path = evidence_path(
                    lane_value.get("probe"), f"{pair}.B{batch}.{lane}.probe"
                )
                row = (
                    _read_single_evidence_row(
                        row_path, f"{pair} B{batch} {lane} row", errors
                    )
                    if row_path is not None
                    else None
                )
                if row is not None:
                    loaded_rows[lane] = row
                    _validate_decode_correctness_row(
                        row,
                        pair=pair,
                        size=size,
                        model_path=model_path,
                        batch=batch,
                        lane=lane,
                        commit=commit,
                        runtime=runtime,
                        errors=errors,
                    )
                probe = (
                    _load_decode_probe(
                        probe_path,
                        pair=pair,
                        size=size,
                        model_path=model_path,
                        commit=commit,
                        batch=batch,
                        errors=errors,
                    )
                    if probe_path is not None
                    else None
                )
                if probe is not None:
                    loaded_probes[lane] = probe
                if row is not None and probe is not None and probe_path is not None:
                    _validate_decode_row_probe_binding(
                        row, probe, probe_path, batch=batch, errors=errors
                    )

                if lane == "native_candidate":
                    source_cell = lane_value.get("source_cell")
                    expected_source_cell = {
                        "batch_size": batch,
                        "prompt_tokens": 2048,
                        "decode_tokens": 512,
                    }
                    if not _strict_equal(source_cell, expected_source_cell):
                        errors.append(
                            f"{pair} B{batch} native source_cell={source_cell!r}, "
                            f"expected {expected_source_cell!r}"
                        )
                    source_lane_path = evidence_path(
                        lane_value.get("source_lane"),
                        f"{pair}.B{batch}.native_candidate.source_lane",
                    )
                    source_rows: list[dict[str, Any]] = []
                    if source_lane_path is not None:
                        try:
                            source_rows = _read_jsonl_bytes(
                                source_lane_path.read_bytes(), source_lane_path
                            )
                        except (OSError, UnicodeDecodeError, ValueError) as exc:
                            errors.append(
                                f"{source_lane_path}: invalid production source lane: {exc}"
                            )
                    source_keys = [
                        (item.get("prompt_tokens"), item.get("decode_tokens"))
                        for item in source_rows
                    ]
                    if len(source_rows) != 6 or source_keys != expected_source_shapes:
                        errors.append(
                            f"{pair} B{batch}: production source lane must contain "
                            "exactly the six formal P/D cells in stable order"
                        )
                    for source_row in source_rows:
                        source_key = _cell_key(source_row)
                        production_row = candidate_index.get(source_key)
                        if (
                            source_row.get("model_pair") != pair
                            or source_row.get("batch_size") != batch
                            or production_row is None
                            or not _strict_equal(
                                _row_payload(source_row), _row_payload(production_row)
                            )
                        ):
                            errors.append(
                                f"{source_row.get('_source', source_lane_path)}: source "
                                "lane row does not bind the promoted candidate matrix"
                            )
                    production_target = candidate_index.get((pair, batch, 2048, 512))
                    source_target = next(
                        (
                            item
                            for item in source_rows
                            if item.get("prompt_tokens") == 2048
                            and item.get("decode_tokens") == 512
                        ),
                        None,
                    )
                    for label, target in (
                        ("promoted candidate", production_target),
                        ("production source cell", source_target),
                    ):
                        if (
                            row is None
                            or target is None
                            or not _strict_equal(
                                _row_payload(row), _row_payload(target)
                            )
                        ):
                            errors.append(
                                f"{pair} B{batch}: native evidence row does not bind "
                                f"the {label} P2048/D512 row"
                            )

            recomputed: dict[str, Any] | None = None
            if set(loaded_probes) == {"fla_reference", "native_candidate"}:
                recomputed = compare_rwkv_probes(
                    loaded_probes["fla_reference"],
                    loaded_probes["native_candidate"],
                    0.9999,
                )
                if recomputed.get("status") != "pass":
                    errors.append(
                        f"{pair} B{batch}: recomputed native_graph-vs-FLA "
                        "correctness comparison failed"
                    )
            comparison_path = evidence_path(
                entry.get("comparison"), f"{pair}.B{batch}.comparison"
            )
            recorded: dict[str, Any] = {}
            if comparison_path is not None:
                try:
                    decoded = json.loads(comparison_path.read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    errors.append(f"{comparison_path}: invalid comparison JSON: {exc}")
                else:
                    if type(decoded) is not dict:
                        errors.append(
                            f"{comparison_path}: comparison must be an object"
                        )
                    else:
                        recorded = decoded
            for field, expected in (
                ("status", "pass"),
                ("min_cosine_required", 0.9999),
                ("input_ids_match", True),
                ("greedy_tokens_match", True),
                ("prompt_logits_shape_match", True),
                ("prompt_logits_finite", True),
                ("final_logits_shape_match", True),
                ("final_logits_finite", True),
                ("probe_batch_size", batch),
                ("probe_tokens", 512),
                ("distinct_batch_prompts", batch == 8),
                ("decode_finite_shape_match", True),
                ("reference_decode_logits_all_finite", True),
                ("native_decode_logits_all_finite", True),
                ("contract_errors", []),
            ):
                _require(
                    {**recorded, "_source": str(comparison_path or manifest_path)},
                    field,
                    expected,
                    errors,
                )
            for field in ("prompt_logits_cosine", "final_logits_cosine"):
                value = recorded.get(field)
                if not _is_finite_real(value) or float(value) < 0.9999:
                    errors.append(
                        f"{comparison_path}: {field} must be finite minimum-row "
                        "cosine >= 0.9999"
                    )
            if recomputed is not None:
                for field, expected in recomputed.items():
                    if not _strict_equal(recorded.get(field), expected):
                        errors.append(
                            f"{comparison_path}: {field} differs from recomputed probes"
                        )
                artifact_summary.update(
                    {
                        "prompt_logits_min_row_cosine": recomputed.get(
                            "prompt_logits_cosine"
                        ),
                        "final_logits_min_row_cosine": recomputed.get(
                            "final_logits_cosine"
                        ),
                        "greedy_tokens_match": recomputed.get("greedy_tokens_match"),
                        "decode_logits_all_finite": bool(
                            recomputed.get("reference_decode_logits_all_finite")
                            and recomputed.get("native_decode_logits_all_finite")
                        ),
                    }
                )
            evidence_summary.append(artifact_summary)

    return {
        "status": "pass" if not errors else "fail",
        "protocol": DECODE_CORRECTNESS_PROTOCOL,
        "manifest": str(manifest_path),
        "model_hashes": str(model_hashes_path),
        "entries": evidence_summary,
    }, errors


def _validate_candidate_route_manifest(
    *,
    manifest_path: Path,
    candidate_path: Path,
    candidate_rows: list[dict[str, Any]],
    model_hashes_path: Path,
    sm120_ab_manifest_path: Path,
    decode_correctness_manifest_path: Path,
    runtime_lock_path: Path,
) -> tuple[dict[str, Any], list[str]]:
    """Authenticate the formal runner and every artifact enclosing the matrix."""

    errors: list[str] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {"status": "fail"}, [
            f"candidate route manifest could not be read: {exc}"
        ]
    if type(manifest) is not dict:
        return {"status": "fail"}, ["candidate route manifest must be a JSON object"]

    commits = {
        row.get("benchmark_repository_commit")
        for row in candidate_rows
        if type(row.get("benchmark_repository_commit")) is str
    }
    commit = next(iter(commits)) if len(commits) == 1 else ""
    source = {**manifest, "_source": str(manifest_path)}
    for field, expected in (
        ("schema_version", 1),
        ("protocol", PROTOCOL),
        ("benchmark_repository_commit", commit),
        ("repository_clean_pre_and_post", True),
        ("candidate_rows", 48),
        ("qwen_rerun", False),
        ("rwkv_implementation_requested", "auto"),
        ("rwkv_implementation_effective", "native_model"),
    ):
        _require(source, field, expected, errors)

    repository_root = _recorded_path(manifest.get("repository_root"))
    if repository_root is None or not repository_root.startswith("/"):
        errors.append("candidate route repository_root must be an absolute POSIX path")
    expected_environment = {
        "CUDA_VISIBLE_DEVICES": "0",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "PYTHONPATH": repository_root,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_CUDA_ARCH_LIST": "12.0",
        "TORCHDYNAMO_DISABLE": "0",
        "TORCH_COMPILE_DISABLE": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "RWKV7_FAST_TOKEN_BACKEND": "native_graph",
        "RWKV7_NATIVE_MODEL_BACKEND": "native_graph",
        "RWKV7_NATIVE_PREFILL_GRAPH": "unset_exact_card_policy",
    }
    _require(source, "forced_environment", expected_environment, errors)

    used_paths: set[Path] = {_resolved(manifest_path)}

    def route_artifact(
        value: Any,
        label: str,
        *,
        expected_path: Path | None = None,
    ) -> tuple[Path | None, str | None]:
        if type(value) is not dict:
            errors.append(f"{label} must be an artifact object")
            return None, None
        recorded = value.get("path")
        digest = value.get("sha256")
        normalized = _recorded_path(recorded)
        if (
            normalized is None
            or not normalized.startswith("/")
            or "/../" in f"/{normalized.strip('/')}/"
        ):
            errors.append(f"{label}.path must be one absolute normalized POSIX path")
            return None, None
        basename = normalized.rsplit("/", 1)[-1]
        if not basename:
            errors.append(f"{label}.path has no artifact basename")
            return None, None
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in HEX_DIGITS for character in digest.lower())
        ):
            errors.append(f"{label}.sha256 must be 64 hexadecimal characters")
            return None, None
        path = (
            expected_path
            if expected_path is not None
            else manifest_path.parent / basename
        )
        if expected_path is not None and expected_path.name != basename:
            errors.append(
                f"{label}.path basename {basename!r} does not bind {expected_path.name!r}"
            )
        resolved = _resolved(path)
        if resolved in used_paths:
            errors.append(f"{label} reuses an artifact already bound elsewhere")
        used_paths.add(resolved)
        try:
            payload = path.read_bytes()
        except OSError as exc:
            errors.append(f"{label} could not be read: {exc}")
            return None, digest.lower()
        observed = _sha256(payload)
        if observed != digest.lower():
            errors.append(
                f"{label} SHA256 mismatch: observed {observed}, expected {digest.lower()}"
            )
        return path, digest.lower()

    candidate_artifact, candidate_digest = route_artifact(
        manifest.get("candidate_result"),
        "candidate_result",
        expected_path=candidate_path,
    )
    sidecar_path, _sidecar_digest = route_artifact(
        manifest.get("candidate_sha256_sidecar"),
        "candidate_sha256_sidecar",
    )
    if candidate_artifact is not None and candidate_digest is not None:
        try:
            observed_candidate_digest = _sha256(candidate_artifact.read_bytes())
        except OSError:
            observed_candidate_digest = ""
        if observed_candidate_digest != candidate_digest:
            errors.append("candidate_result does not bind the supplied candidate bytes")
        if sidecar_path is not None:
            try:
                sidecar = sidecar_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                errors.append(f"candidate SHA256 sidecar could not be read: {exc}")
            else:
                expected_sidecar = f"{candidate_digest}  {candidate_path.name}\n"
                if sidecar != expected_sidecar:
                    errors.append(
                        "candidate SHA256 sidecar must contain the exact digest and basename"
                    )

    model_contract = manifest.get("model_hash_contract")
    if type(model_contract) is not dict:
        errors.append("model_hash_contract must be an object")
        model_contract = {}
    for field, expected in (
        ("algorithm", "sha256"),
        ("scope", "every recursive regular file"),
        ("byte_identical", True),
    ):
        _require(
            {**model_contract, "_source": str(manifest_path)}, field, expected, errors
        )
    before_path, _ = route_artifact(
        model_contract.get("before"),
        "model_hash_contract.before",
        expected_path=model_hashes_path,
    )
    after_path, _ = route_artifact(
        model_contract.get("after"), "model_hash_contract.after"
    )
    if before_path is not None and after_path is not None:
        try:
            if before_path.read_bytes() != after_path.read_bytes():
                errors.append("model hash snapshots are not byte-identical")
        except OSError as exc:
            errors.append(f"model hash snapshots could not be compared: {exc}")

    bound_artifacts = (
        (
            "sm120_b8_ab_manifest",
            sm120_ab_manifest_path,
        ),
        (
            "native_graph_fla_correctness_manifest",
            decode_correctness_manifest_path,
        ),
    )
    for field, path in bound_artifacts:
        route_artifact(manifest.get(field), field, expected_path=path)
    runtime_path, _ = route_artifact(
        manifest.get("runtime_lock"),
        "runtime_lock",
        expected_path=runtime_lock_path,
    )
    pip_freeze_path, _ = route_artifact(manifest.get("pip_freeze"), "pip_freeze")
    system_identity_path, _ = route_artifact(
        manifest.get("system_identity"), "system_identity"
    )
    if system_identity_path is not None:
        try:
            with system_identity_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, skipinitialspace=True)
                raw_fieldnames = reader.fieldnames
                system_rows = list(reader)
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            errors.append(f"system identity CSV could not be read: {exc}")
        else:
            if raw_fieldnames is None:
                errors.append("system identity CSV has no header")
            else:
                fieldnames = [str(name).strip() for name in raw_fieldnames]
                if any(not name for name in fieldnames) or len(fieldnames) != len(
                    set(fieldnames)
                ):
                    errors.append(
                        "system identity CSV header must contain unique non-empty columns"
                    )
                required_columns = {
                    "name",
                    "uuid",
                    "pci.bus_id",
                    "compute_cap",
                    "driver_version",
                    "memory.total [MiB]",
                }
                missing_columns = sorted(required_columns - set(fieldnames))
                if missing_columns:
                    errors.append(
                        "system identity CSV is missing required columns: "
                        + ", ".join(missing_columns)
                    )
            if len(system_rows) != 1:
                errors.append(
                    "system identity CSV must contain exactly one GPU data row"
                )
            elif raw_fieldnames is not None:
                raw_row = system_rows[0]
                if None in raw_row or any(value is None for value in raw_row.values()):
                    errors.append("system identity CSV row does not match its header")
                row = {
                    str(name).strip(): str(value).strip()
                    for name, value in raw_row.items()
                    if name is not None and value is not None
                }
                for field, expected in (
                    ("name", EXPECTED_DEVICE),
                    ("compute_cap", "12.0"),
                    ("driver_version", "595.58.03"),
                    ("memory.total [MiB]", "32607 MiB"),
                ):
                    if row.get(field) != expected:
                        errors.append(
                            f"system identity {field}={row.get(field)!r}, "
                            f"expected {expected!r}"
                        )
                for field in ("uuid", "pci.bus_id"):
                    if not row.get(field):
                        errors.append(f"system identity {field} must be non-empty")

    if runtime_path is not None and pip_freeze_path is not None:
        try:
            route_runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
            pip_freeze_digest = _sha256(pip_freeze_path.read_bytes())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"route runtime/pip-freeze evidence could not be read: {exc}")
        else:
            if (
                type(route_runtime) is not dict
                or route_runtime.get("pip_freeze_sha256") != pip_freeze_digest
            ):
                errors.append(
                    "runtime lock pip_freeze_sha256 does not bind the pip-freeze artifact"
                )

    # The correctness manifest independently hashes the same runtime lock. This
    # closes the route-manifest/correctness-manifest provenance chain.
    if runtime_path is not None:
        try:
            correctness = json.loads(
                decode_correctness_manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            correctness = {}
        runtime_evidence = (
            correctness.get("runtime") if type(correctness) is dict else None
        )
        if type(runtime_evidence) is not dict:
            errors.append("correctness manifest does not bind the route runtime lock")
        else:
            if runtime_evidence.get("path") != runtime_path.name:
                errors.append(
                    "correctness manifest runtime basename differs from route"
                )
            try:
                runtime_digest = _sha256(runtime_path.read_bytes())
            except OSError:
                runtime_digest = ""
            if runtime_evidence.get("sha256") != runtime_digest:
                errors.append("correctness manifest runtime SHA256 differs from route")

    main_model_paths: dict[str, str] = {}
    for pair in PARAMETERS:
        paths = {
            normalized
            for row in candidate_rows
            if row.get("model_pair") == pair
            and (normalized := _recorded_path(row.get("model_id_or_path"))) is not None
        }
        if len(paths) == 1:
            main_model_paths[pair] = next(iter(paths))
        else:
            errors.append(f"route manifest cannot bind candidate model path for {pair}")
    expected_lane_keys = [(pair, batch) for pair in PARAMETERS for batch in (1, 8)]
    lanes = manifest.get("lanes")
    if type(lanes) is not list:
        errors.append("candidate route lanes must be a list")
        lanes = []
    actual_lane_keys = [
        (lane.get("model_pair"), lane.get("batch_size"))
        for lane in lanes
        if type(lane) is dict
    ]
    if len(lanes) != 8 or actual_lane_keys != expected_lane_keys:
        errors.append(
            "candidate route lanes must contain the eight pair/B1/B8 lanes in stable order"
        )
    lane_summaries: list[dict[str, Any]] = []
    for index, key in enumerate(expected_lane_keys):
        if index >= len(lanes) or type(lanes[index]) is not dict:
            continue
        lane = lanes[index]
        pair, batch = key
        size = RWKV_PAIR_SIZES[pair]
        promoted = batch == 8 and size in {"0.4b", "1.5b"}
        expected = {
            "model_pair": pair,
            "model_size_label": size,
            "model_path": main_model_paths.get(pair),
            "batch_size": batch,
            "cells": 6,
            "fresh_process": True,
            "rwkv_implementation_requested": "auto",
            "rwkv_implementation_effective": "native_model",
            "RWKV7_NATIVE_PREFILL_GRAPH": "exact_card_policy",
            "RWKV7_NATIVE_GRAPH_ADA_WAGV_BMM": int(promoted),
            "RWKV7_NATIVE_GRAPH_SM120_WAGV_BMM_G": int(promoted),
            "RWKV7_NATIVE_GRAPH_SM120_COMPILED_FFN": int(promoted),
            "RWKV7_NATIVE_GRAPH_RKV_POLICY": "vkwr_auto" if promoted else None,
            "RWKV7_BLACKWELL_TORCH_COMPILE": 1 if promoted else None,
            "compile_cache": "fresh_unique_directory" if promoted else None,
        }
        if not _strict_equal(lane, expected):
            errors.append(
                f"candidate route lane {pair} B{batch} does not match the exact route contract"
            )
        lane_summaries.append(
            {"model_pair": pair, "batch_size": batch, "promoted": promoted}
        )

    return {
        "status": "pass" if not errors else "fail",
        "protocol": PROTOCOL,
        "manifest": str(manifest_path),
        "candidate_sha256": candidate_digest,
        "repository_clean_pre_and_post": manifest.get("repository_clean_pre_and_post"),
        "lanes": lane_summaries,
    }, errors


def validate_files(
    *,
    qwen_reference: Path,
    rwkv_candidate: Path,
    sm120_ab_manifest: Path,
    decode_correctness_manifest: Path,
    candidate_route_manifest: Path,
    runtime_lock: Path,
    model_hashes: Path,
    expected_candidate_commit: str | None = None,
    expected_reference_sha256: str = FROZEN_REFERENCE_SHA256,
    expected_device: str = EXPECTED_DEVICE,
) -> dict[str, Any]:
    reference_bytes = qwen_reference.read_bytes()
    candidate_bytes = rwkv_candidate.read_bytes()
    reference_sha256_before = _sha256(reference_bytes)
    candidate_sha256 = _sha256(candidate_bytes)
    reference_rows = _read_jsonl_bytes(reference_bytes, qwen_reference)
    candidate_rows = _read_jsonl_bytes(candidate_bytes, rwkv_candidate)
    summary = validate_paired_decode(
        candidate_rows,
        reference_rows,
        expected_device=expected_device,
        expected_reference_sha256=expected_reference_sha256,
        reference_sha256_before=reference_sha256_before,
        reference_sha256_after=_sha256(qwen_reference.read_bytes()),
        candidate_sha256=candidate_sha256,
    )
    if expected_candidate_commit is not None:
        normalized_expected = expected_candidate_commit.lower()
        if len(normalized_expected) != 40 or any(
            character not in HEX_DIGITS for character in normalized_expected
        ):
            summary["errors"].append(
                "expected candidate commit must be exactly 40 hexadecimal characters"
            )
        candidate_commits = {
            str(row.get("benchmark_repository_commit", "")).lower()
            for row in candidate_rows
        }
        if candidate_commits != {normalized_expected}:
            summary["errors"].append(
                "candidate rows do not match the externally expected repository commit"
            )
        if summary["errors"]:
            summary["status"] = "fail"
            summary["paired_decode_table_eligible"] = False
    ab_summary, ab_errors = _validate_sm120_ab_evidence(
        manifest_path=sm120_ab_manifest,
        model_hashes_path=model_hashes,
        candidate_rows=candidate_rows,
    )
    summary["sm120_b8_decode_ab"] = ab_summary
    if ab_errors:
        summary["errors"].extend(ab_errors)
        summary["status"] = "fail"
        summary["paired_decode_table_eligible"] = False
    correctness_summary, correctness_errors = _validate_decode_correctness_evidence(
        manifest_path=decode_correctness_manifest,
        model_hashes_path=model_hashes,
        runtime_lock_path=runtime_lock,
        candidate_rows=candidate_rows,
    )
    summary["native_graph_fla_decode_correctness"] = correctness_summary
    if correctness_errors:
        summary["errors"].extend(correctness_errors)
        summary["status"] = "fail"
        summary["paired_decode_table_eligible"] = False
    route_summary, route_errors = _validate_candidate_route_manifest(
        manifest_path=candidate_route_manifest,
        candidate_path=rwkv_candidate,
        candidate_rows=candidate_rows,
        model_hashes_path=model_hashes,
        sm120_ab_manifest_path=sm120_ab_manifest,
        decode_correctness_manifest_path=decode_correctness_manifest,
        runtime_lock_path=runtime_lock,
    )
    summary["candidate_route_manifest"] = route_summary
    if route_errors:
        summary["errors"].extend(route_errors)
        summary["status"] = "fail"
        summary["paired_decode_table_eligible"] = False
    summary["inputs"] = {
        "qwen_reference": str(qwen_reference),
        "rwkv_candidate": str(rwkv_candidate),
        "sm120_ab_manifest": str(sm120_ab_manifest),
        "decode_correctness_manifest": str(decode_correctness_manifest),
        "candidate_route_manifest": str(candidate_route_manifest),
        "runtime_lock": str(runtime_lock),
        "model_hashes": str(model_hashes),
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-reference", type=Path, required=True)
    parser.add_argument("--rwkv-candidate", type=Path, required=True)
    parser.add_argument("--sm120-ab-manifest", type=Path, required=True)
    parser.add_argument("--decode-correctness-manifest", type=Path, required=True)
    parser.add_argument("--candidate-route-manifest", type=Path, required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    parser.add_argument("--model-hashes", type=Path, required=True)
    parser.add_argument(
        "--expected-candidate-commit",
        required=True,
        help="exact 40-hex clean repository commit used for the RWKV capture",
    )
    parser.add_argument(
        "--expected-reference-sha256",
        choices=(FROZEN_REFERENCE_SHA256,),
        default=FROZEN_REFERENCE_SHA256,
        help="immutable v1 Qwen reference digest (versioned in this validator)",
    )
    parser.add_argument("--expected-device", default=EXPECTED_DEVICE)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--paired-table", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = {
        _resolved(args.qwen_reference),
        _resolved(args.rwkv_candidate),
        _resolved(args.sm120_ab_manifest),
        _resolved(args.decode_correctness_manifest),
        _resolved(args.candidate_route_manifest),
        _resolved(args.runtime_lock),
        _resolved(args.model_hashes),
    }
    outputs = [_resolved(args.validation), _resolved(args.paired_table)]
    if args.markdown:
        outputs.append(_resolved(args.markdown))
    collision_errors: list[str] = []
    if len(inputs) != 7:
        collision_errors.append("all seven evidence inputs must be different files")
    if len(outputs) != len(set(outputs)):
        collision_errors.append("output paths must be distinct")
    overlap = sorted(str(path) for path in inputs & set(outputs))
    if overlap:
        collision_errors.append(
            "input and output paths overlap: " + json.dumps(overlap)
        )
    if collision_errors:
        # Output collisions are a failed rerun. Remove any prior promotable
        # table when doing so cannot touch either immutable input.
        paired_output = _resolved(args.paired_table)
        if paired_output not in inputs:
            args.paired_table.unlink(missing_ok=True)
        print(
            "QWEN35_PAIRED_DECODE_V1 "
            + json.dumps(
                {"status": "fail", "errors": collision_errors}, ensure_ascii=False
            )
        )
        return 1

    try:
        summary = validate_files(
            qwen_reference=args.qwen_reference,
            rwkv_candidate=args.rwkv_candidate,
            sm120_ab_manifest=args.sm120_ab_manifest,
            decode_correctness_manifest=args.decode_correctness_manifest,
            candidate_route_manifest=args.candidate_route_manifest,
            runtime_lock=args.runtime_lock,
            model_hashes=args.model_hashes,
            expected_candidate_commit=args.expected_candidate_commit,
            expected_reference_sha256=args.expected_reference_sha256,
            expected_device=args.expected_device,
        )
    except (
        EOFError,
        KeyError,
        OSError,
        pickle.UnpicklingError,
        RuntimeError,
        TypeError,
        UnicodeDecodeError,
        ValueError,
    ) as exc:
        summary = {
            "schema_version": 1,
            "protocol": PROTOCOL,
            "status": "fail",
            "paired_decode_table_eligible": False,
            "continuous_e2e_eligible": False,
            "gate": {"passing_cells": 0, "total_cells": 48},
            "cells": [],
            "errors": [str(exc)],
        }

    _write_json(args.validation, summary)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(summary), encoding="utf-8")
    if summary["paired_decode_table_eligible"]:
        args.paired_table.parent.mkdir(parents=True, exist_ok=True)
        args.paired_table.write_text(
            "".join(
                json.dumps(cell, ensure_ascii=False) + "\n" for cell in summary["cells"]
            ),
            encoding="utf-8",
        )
    else:
        # A failed rerun must never leave a prior promotable PASS table behind.
        args.paired_table.unlink(missing_ok=True)
    print("QWEN35_PAIRED_DECODE_V1 " + json.dumps(summary, ensure_ascii=False))
    return 0 if summary["paired_decode_table_eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
