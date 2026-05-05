#!/usr/bin/env python3
"""Validate the strict Tesla V100 RWKV/Qwen Prefill+Decode matrix."""

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
    from bench.validate_qwen35_best_optimized_hf_v1 import (
        EXPECTED_SHAPES,
        PAIR_RANK,
        RUNTIME_FIELDS,
        validate_matrix as validate_qwen_reference,
    )
except ModuleNotFoundError:
    from validate_qwen35_best_optimized_hf_v1 import (
        EXPECTED_SHAPES,
        PAIR_RANK,
        RUNTIME_FIELDS,
        validate_matrix as validate_qwen_reference,
    )

try:
    from bench.compare_rwkv_prefill_probe import compare as compare_rwkv_probes
except ModuleNotFoundError:
    from compare_rwkv_prefill_probe import compare as compare_rwkv_probes


PROTOCOL = "qwen35_v100_paired_pd_v1"
QWEN_MATRIX = "qwen35_v100_best_optimized_hf_v1"
QWEN_CONTRACT = "official_fla_triton_static_cache_cudagraph_same_cache_v100_v1"
CANDIDATE_LANE = "best_optimized_hf"
QWEN_LANE = "qwen_best_optimized_hf"
EXPECTED_DEVICE = "Tesla V100-PCIE-32GB"
STRICT_RATIO_GATE = 1.0
PAIRS = (
    "rwkv-0.4b__qwen3.5-0.8b",
    "rwkv-1.5b__qwen3.5-2b",
    "rwkv-2.9b__qwen3.5-4b",
    "rwkv-7.2b__qwen3.5-9b",
)
PARAMETERS = {
    PAIRS[0]: (450_767_872, 752_393_024),
    PAIRS[1]: (1_527_404_544, 1_881_825_088),
    PAIRS[2]: (2_947_735_040, 4_205_751_296),
    PAIRS[3]: (7_199_141_888, 8_953_803_264),
}
RWKV_SIZES = {
    PAIRS[0]: ("0.4b", 24),
    PAIRS[1]: ("1.5b", 24),
    PAIRS[2]: ("2.9b", 32),
    PAIRS[3]: ("7.2b", 32),
}
QWEN_ROUTES = {pair: "static_cache_raw_cudagraph" for pair in PAIRS}
QWEN_SDPA_POLICIES = {
    PAIRS[0]: "auto",
    PAIRS[1]: "auto",
    PAIRS[2]: "auto",
    PAIRS[3]: "math_only",
}
EXPECTED_KEYS = {
    (pair, batch, prompt, decode)
    for pair in PAIRS
    for batch, prompt, decode in EXPECTED_SHAPES
}
HEX_DIGITS = frozenset("0123456789abcdef")


def _is_finite_real(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value))


def _strict_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_equal(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return bool(actual == expected)


def _require(row: dict[str, Any], field: str, expected: Any, errors: list[str]) -> None:
    actual = row.get(field)
    if not _strict_equal(actual, expected):
        errors.append(
            f"{row.get('_source', '<row>')}: {field}={actual!r}, expected {expected!r}"
        )


def _read_jsonl_bytes(data: bytes, path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: row is not a JSON object")
        row["_source"] = f"{path}:{line_number}"
        rows.append(row)
    return rows


def _cell_key(row: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    return (
        row.get("model_pair"),
        row.get("batch_size"),
        row.get("prompt_tokens"),
        row.get("decode_tokens"),
    )


def _validate_samples(
    row: dict[str, Any], axis: str, tokens: int, errors: list[str]
) -> None:
    samples_field = f"{axis}_sec_samples"
    median_field = f"{axis}_sec_median"
    raw_median_field = f"{median_field}_raw"
    raw_tokps_field = f"{axis}_tokps_total_raw"
    samples = row.get(samples_field)
    if type(samples) is not list or len(samples) != 7:
        errors.append(
            f"{row.get('_source', '<row>')}: {samples_field} must contain 7 samples"
        )
        return
    if not all(_is_finite_real(value) and float(value) > 0 for value in samples):
        errors.append(
            f"{row.get('_source', '<row>')}: {samples_field} must be positive and finite"
        )
        return
    expected_median = float(statistics.median(samples))
    if row.get(raw_median_field) != expected_median:
        errors.append(
            f"{row.get('_source', '<row>')}: {raw_median_field} does not exactly match samples"
        )
    rounded = row.get(median_field)
    if not _is_finite_real(rounded) or abs(float(rounded) - expected_median) > 1e-6:
        errors.append(
            f"{row.get('_source', '<row>')}: {median_field} does not match samples"
        )
    expected_tokps = tokens / expected_median
    raw_tokps = row.get(raw_tokps_field)
    if not _is_finite_real(raw_tokps) or not math.isclose(
        float(raw_tokps), expected_tokps, rel_tol=1e-12, abs_tol=1e-12
    ):
        errors.append(
            f"{row.get('_source', '<row>')}: {raw_tokps_field} does not equal tokens/raw median"
        )


def _validate_candidate_row(
    row: dict[str, Any], *, expected_device: str, errors: list[str]
) -> None:
    for field, expected in (
        ("axis", "qwen35_cross_model_speed"),
        ("benchmark_matrix", PROTOCOL),
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
        ("gpu_arch", "sm_70"),
        ("gpu_compute_capability", [7, 0]),
        ("rwkv_fast_token_backend_requested", "native_graph"),
        ("rwkv_native_model_backend_requested", "native_graph"),
        ("effective_backend", "native_graph"),
        ("step_backend", "rwkv_fast_token"),
        ("cache_type", "NativeRWKV7Cache"),
    ):
        _require(row, field, expected, errors)
    pair = row.get("model_pair")
    if pair not in PAIRS:
        errors.append(f"{row.get('_source', '<row>')}: unexpected model_pair={pair!r}")
        return
    size, layer_count = RWKV_SIZES[pair]
    _require(row, "model_size_label", size, errors)
    _require(row, "active_parameter_count", PARAMETERS[pair][0], errors)
    shape = (row.get("batch_size"), row.get("prompt_tokens"), row.get("decode_tokens"))
    if shape not in EXPECTED_SHAPES:
        errors.append(f"{row.get('_source', '<row>')}: unexpected B/P/D={shape!r}")
        return
    batch, prompt, decode = (int(value) for value in shape)
    large_b8_closure = pair == PAIRS[3] and batch == 8
    _require(
        row,
        "rwkv_native_graph_rkv_policy",
        "vkwr_auto" if large_b8_closure else "manual",
        errors,
    )
    _require(
        row,
        "rwkv_native_graph_fused_norm_mix_num_warps",
        8 if large_b8_closure else 4,
        errors,
    )
    fp16_state = batch == 8 and (pair in PAIRS[:2] or pair == PAIRS[3])
    _require(
        row,
        "rwkv_native_graph_state_dtype",
        "torch.float16" if fp16_state else "torch.float32",
        errors,
    )
    _require(row, "rwkv_native_graph_triton_fp16_state", fp16_state, errors)
    _require(row, "rwkv_native_graph_fp16_recurrent", False, errors)
    _require(
        row,
        "rwkv_native_graph_sm70_wagv_lora_extension_required",
        batch == 1,
        errors,
    )
    _require(
        row,
        "rwkv_native_graph_sm70_wagv_lora_extension_available",
        batch == 1,
        errors,
    )

    eligible_layers = list(range(1, layer_count))
    for route, enabled in (
        ("sm70_wagv_lora", batch == 1),
        ("fused_wavg_lora", batch == 8 and not large_b8_closure),
    ):
        layers = eligible_layers if enabled else []
        for suffix, expected in (
            ("selected", enabled),
            ("effective", enabled),
            ("selected_layers", layers),
            ("effective_layers", layers),
            ("effective_layer_count", len(layers)),
            ("full_eligible_layers_effective", enabled),
        ):
            _require(row, f"rwkv_native_graph_{route}_{suffix}", expected, errors)

    for route in ("ada_wagv_bmm", "sm120_wagv_bmm_g", "sm120_compiled_ffn"):
        for suffix, expected in (
            ("requested", False),
            ("selected", False),
            ("effective", False),
            ("selected_layers", []),
            ("effective_layers", []),
            ("effective_layer_count", 0),
            ("full_model_effective", False),
        ):
            _require(row, f"rwkv_native_graph_{route}_{suffix}", expected, errors)

    _validate_samples(row, "prefill", batch * prompt, errors)
    _validate_samples(row, "decode", batch * decode, errors)
    for field in RUNTIME_FIELDS:
        value = row.get(field)
        if field == "causal_conv1d_version" and value is None:
            continue
        if type(value) is not str or not value.strip():
            errors.append(
                f"{row.get('_source', '<row>')}: {field} must be a non-empty string"
            )
    commit = row.get("benchmark_repository_commit")
    normalized = commit.lower() if type(commit) is str else ""
    if len(normalized) != 40 or any(value not in HEX_DIGITS for value in normalized):
        errors.append(
            f"{row.get('_source', '<row>')}: benchmark_repository_commit must be 40 hex"
        )


def _validate_qwen_sdpa_row(row: dict[str, Any], errors: list[str]) -> None:
    pair = row.get("model_pair")
    if pair not in QWEN_SDPA_POLICIES:
        errors.append(f"unexpected Qwen model_pair {pair!r}")
        return
    policy = QWEN_SDPA_POLICIES[pair]
    automatic = policy == "auto"
    for field, expected in (
        ("qwen_sdpa_policy_requested", policy),
        ("qwen_sdpa_policy_effective", policy),
        ("qwen_sdp_flash_enabled", automatic),
        ("qwen_sdp_mem_efficient_enabled", automatic),
        ("qwen_sdp_math_enabled", True),
        ("qwen_sdp_cudnn_enabled", automatic),
    ):
        _require(row, field, expected, errors)


def _index_rows(
    rows: list[dict[str, Any]], label: str, errors: list[str]
) -> dict[tuple[Any, Any, Any, Any], dict[str, Any]]:
    keys = [_cell_key(row) for row in rows]
    counts = Counter(keys)
    duplicates = sorted((key for key, count in counts.items() if count > 1), key=repr)
    missing = sorted(EXPECTED_KEYS - set(keys), key=repr)
    extra = sorted(set(keys) - EXPECTED_KEYS, key=repr)
    if len(rows) != 48:
        errors.append(f"{label} row count={len(rows)}, expected 48")
    if duplicates:
        errors.append(f"{label} duplicate cells: {duplicates}")
    if missing:
        errors.append(f"{label} missing cells: {missing}")
    if extra:
        errors.append(f"{label} extra cells: {extra}")
    return {
        key: row
        for key, row in zip(keys, rows, strict=True)
        if counts[key] == 1 and key in EXPECTED_KEYS
    }


def _runtime_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in RUNTIME_FIELDS)


def _sort_key(cell: dict[str, Any]) -> tuple[Any, ...]:
    return (
        PAIR_RANK.get(str(cell["model_pair"]), 999),
        str(cell["device"]),
        int(cell["batch_size"]),
        int(cell["prompt_tokens"]),
        int(cell["decode_tokens"]),
    )


def validate_paired_pd(
    candidate_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    *,
    expected_device: str = EXPECTED_DEVICE,
    candidate_sha256: str = "",
    reference_sha256: str = "",
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        qwen_validation = validate_qwen_reference(
            reference_rows,
            expected_device=expected_device,
            expected_pairs=PAIRS,
            expected_routes_by_pair=QWEN_ROUTES,
            expected_matrix=QWEN_MATRIX,
            expected_lane=QWEN_LANE,
            expected_conv_backend="fla_triton",
            expected_causal_conv1d_importable=False,
            expected_fast_path_available=False,
            nullable_runtime_fields=("causal_conv1d_version",),
            qwen_contract=QWEN_CONTRACT,
        )
    except (TypeError, ValueError) as exc:
        qwen_validation = {"status": "fail", "errors": [str(exc)]}
    if qwen_validation.get("status") != "pass":
        errors.extend(
            f"Qwen reference: {message}"
            for message in qwen_validation.get("errors", [])
        )

    for row in candidate_rows:
        _validate_candidate_row(row, expected_device=expected_device, errors=errors)
    candidate_index = _index_rows(candidate_rows, "candidate", errors)
    reference_index = _index_rows(reference_rows, "reference", errors)
    for row in reference_rows:
        pair = row.get("model_pair")
        if pair in PARAMETERS:
            _require(row, "active_parameter_count", PARAMETERS[pair][1], errors)
        _validate_qwen_sdpa_row(row, errors)

    candidate_runtime = {_runtime_signature(row) for row in candidate_rows}
    reference_runtime = {_runtime_signature(row) for row in reference_rows}
    if len(candidate_runtime) != 1:
        errors.append("candidate rows do not have one runtime signature")
    if len(reference_runtime) != 1:
        errors.append("reference rows do not have one runtime signature")
    if (
        len(candidate_runtime) == len(reference_runtime) == 1
        and candidate_runtime != reference_runtime
    ):
        errors.append("candidate/reference runtime signatures differ")

    candidate_commits = sorted(
        {str(row.get("benchmark_repository_commit")) for row in candidate_rows}
    )
    reference_commits = sorted(
        {str(row.get("benchmark_repository_commit")) for row in reference_rows}
    )
    if len(candidate_commits) != 1:
        errors.append("candidate rows do not have exactly one repository commit")
    if len(reference_commits) != 1:
        errors.append("reference rows do not have exactly one repository commit")

    cells: list[dict[str, Any]] = []
    for key in EXPECTED_KEYS:
        candidate = candidate_index.get(key)
        reference = reference_index.get(key)
        if candidate is None or reference is None:
            continue
        pair, batch, prompt, decode = key
        candidate_params, reference_params = PARAMETERS[pair]
        parameter_ratio = candidate_params / reference_params
        values: dict[str, float] = {}
        valid = True
        for axis in ("prefill", "decode"):
            candidate_tokps = candidate.get(f"{axis}_tokps_total_raw")
            reference_tokps = reference.get(f"{axis}_tokps_total_raw")
            if not (
                _is_finite_real(candidate_tokps)
                and float(candidate_tokps) > 0
                and _is_finite_real(reference_tokps)
                and float(reference_tokps) > 0
            ):
                errors.append(f"cell {key}: invalid raw {axis} throughput")
                valid = False
                continue
            raw_ratio = float(candidate_tokps) / float(reference_tokps)
            adjusted_ratio = raw_ratio * parameter_ratio
            required_tokps = float(reference_tokps) / parameter_ratio
            values[f"candidate_{axis}_tokps_total_raw"] = float(candidate_tokps)
            values[f"reference_{axis}_tokps_total_raw"] = float(reference_tokps)
            values[f"raw_{axis}_ratio"] = raw_ratio
            values[f"adjusted_{axis}_ratio"] = adjusted_ratio
            values[f"required_candidate_{axis}_tokps"] = required_tokps
            values[f"candidate_{axis}_margin_tokps"] = (
                float(candidate_tokps) - required_tokps
            )
        if not valid:
            continue
        gates = {
            name: values[name] > STRICT_RATIO_GATE
            for name in (
                "raw_prefill_ratio",
                "raw_decode_ratio",
                "adjusted_prefill_ratio",
                "adjusted_decode_ratio",
            )
        }
        cells.append(
            {
                "axis": PROTOCOL,
                "model_pair": pair,
                "device": reference.get("device"),
                "batch_size": batch,
                "prompt_tokens": prompt,
                "decode_tokens": decode,
                "candidate_active_parameter_count": candidate_params,
                "reference_active_parameter_count": reference_params,
                "active_parameter_ratio": parameter_ratio,
                **values,
                **{f"{name}_pass": passed for name, passed in gates.items()},
                "strict_pass": all(gates.values()),
                "candidate_source": candidate.get("_source"),
                "reference_source": reference.get("_source"),
            }
        )
    cells.sort(key=_sort_key)
    gate_names = (
        "raw_prefill_ratio_pass",
        "raw_decode_ratio_pass",
        "adjusted_prefill_ratio_pass",
        "adjusted_decode_ratio_pass",
    )
    passing = {
        name.removesuffix("_pass"): sum(cell[name] is True for cell in cells)
        for name in gate_names
    }
    eligible = bool(
        not errors
        and len(cells) == 48
        and all(value == 48 for value in passing.values())
    )
    if len(cells) == 48 and not all(value == 48 for value in passing.values()):
        errors.append(
            "strict raw/adjusted Prefill+Decode gates require 48/48 cells each"
        )
    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "pass" if eligible else "fail",
        "expected_device": expected_device,
        "candidate_sha256": candidate_sha256,
        "reference_sha256": reference_sha256,
        "qwen_reference_validation_status": qwen_validation.get("status"),
        "repository_commits": {
            "candidate": candidate_commits,
            "reference": reference_commits,
        },
        "runtime_fields": list(RUNTIME_FIELDS),
        "candidate_runtime_signatures": [list(value) for value in candidate_runtime],
        "reference_runtime_signatures": [list(value) for value in reference_runtime],
        "coverage": {
            "candidate_rows": len(candidate_rows),
            "reference_rows": len(reference_rows),
            "joined_cells": len(cells),
            "expected_cells": 48,
        },
        "gate": {
            "comparison": ">",
            "threshold": STRICT_RATIO_GATE,
            "uses_unrounded_raw_throughput": True,
            "passing_cells": passing,
            "total_cells": 48,
        },
        "paired_pd_table_eligible": eligible,
        "continuous_e2e_eligible": False,
        "cells": cells,
        "red_cells": [cell for cell in cells if cell["strict_pass"] is not True],
        "errors": errors,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Tesla V100 strict paired Prefill/Decode v1",
        "",
        f"Status: **{'PASS' if summary['paired_pd_table_eligible'] else 'FAIL'}**.",
        "",
        "All gates use unrounded raw throughput and require strict `> 1.0x`.",
        "B8 is aggregate throughput across eight sequences; this is not continuous E2E.",
        "",
        "| RWKV / Qwen | B | P | D | Raw P / D | Adjusted P / D | Pass |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for cell in summary["cells"]:
        pair = str(cell["model_pair"]).replace("rwkv-", "").replace("__qwen3.5-", " / ")
        lines.append(
            f"| {pair} | {cell['batch_size']} | {cell['prompt_tokens']} | {cell['decode_tokens']} | "
            f"{cell['raw_prefill_ratio']:.6f}x / {cell['raw_decode_ratio']:.6f}x | "
            f"{cell['adjusted_prefill_ratio']:.6f}x / {cell['adjusted_decode_ratio']:.6f}x | "
            f"{'PASS' if cell['strict_pass'] else 'FAIL'} |"
        )
    if summary["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in summary["errors"])
    return "\n".join(lines) + "\n"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _resolve_artifact(base: Path, record: Any, label: str, errors: list[str]) -> Path:
    if type(record) is not dict:
        errors.append(f"{label} must be an artifact object")
        return base / "<invalid>"
    recorded_path = record.get("path")
    recorded_sha = record.get("sha256")
    if type(recorded_path) is not str or not recorded_path:
        errors.append(f"{label}.path must be non-empty")
        return base / "<invalid>"
    path = base / Path(recorded_path).name
    if not path.is_file():
        errors.append(f"{label} is missing: {path}")
        return path
    actual_sha = _sha256(path.read_bytes())
    if type(recorded_sha) is not str or recorded_sha != actual_sha:
        errors.append(f"{label} sha256 mismatch")
    return path


def _validate_targeted_same_implementation_closure(
    doc: dict[str, Any],
    path: Path,
    candidate_by_key: dict[tuple[Any, Any, Any, Any], dict[str, Any]],
    errors: list[str],
) -> dict[str, Any]:
    label = "targeted same-implementation closure"
    error_count_before = len(errors)
    closure = doc.get("targeted_same_implementation_closure")
    if type(closure) is not dict:
        errors.append(f"{label} must be an object")
        return {"status": "fail"}
    for field, expected in (
        ("model_pair", PAIRS[3]),
        ("model_size_label", "7.2b"),
        ("batch_size", 8),
        ("prompt_tokens", 128),
        ("decode_tokens", 128),
        ("probe_tokens", 512),
    ):
        if not _strict_equal(closure.get(field), expected):
            errors.append(
                f"{label}.{field}={closure.get(field)!r}, expected {expected!r}"
            )
    reference_contract = closure.get("reference_contract")
    expected_reference_contract = {
        "rwkv_implementation": "native_model",
        "effective_backend": "eager",
        "RWKV7_FAST_TOKEN_BACKEND": "module_call",
        "RWKV7_NATIVE_MODEL_BACKEND": "eager",
        "RWKV7_FAST_PREFILL": "0",
        "RWKV7_NATIVE_PREFILL_GRAPH": "0",
    }
    if not _strict_equal(reference_contract, expected_reference_contract):
        errors.append(f"{label} reference contract mismatch")
    candidate_contract = closure.get("candidate_contract")
    expected_candidate_contract = {
        "rwkv_implementation": "native_model",
        "effective_backend": "native_graph",
        "rkv_policy": "vkwr_auto",
        "state_dtype": "torch.float16",
        "triton_fp16_state": True,
        "fused_norm_mix_num_warps": 8,
        "fused_wavg_lora": False,
    }
    if not _strict_equal(candidate_contract, expected_candidate_contract):
        errors.append(f"{label} candidate contract mismatch")
    reference = closure.get("native_eager_reference")
    candidate = closure.get("native_graph_candidate")
    if type(reference) is not dict or type(candidate) is not dict:
        errors.append(f"{label} must contain native eager/graph artifacts")
        return {"status": "fail"}
    base = path.parent
    reference_row_path = _resolve_artifact(
        base, reference.get("row"), f"{label}.reference.row", errors
    )
    reference_probe_path = _resolve_artifact(
        base, reference.get("probe"), f"{label}.reference.probe", errors
    )
    candidate_row_path = _resolve_artifact(
        base, candidate.get("row"), f"{label}.candidate.row", errors
    )
    candidate_probe_path = _resolve_artifact(
        base, candidate.get("probe"), f"{label}.candidate.probe", errors
    )
    comparison_path = _resolve_artifact(
        base, closure.get("comparison"), f"{label}.comparison", errors
    )
    if not all(
        item.is_file()
        for item in (
            reference_row_path,
            reference_probe_path,
            candidate_row_path,
            candidate_probe_path,
            comparison_path,
        )
    ):
        return {"status": "fail"}
    try:
        reference_rows = _read_jsonl_bytes(
            reference_row_path.read_bytes(), reference_row_path
        )
        candidate_rows = _read_jsonl_bytes(
            candidate_row_path.read_bytes(), candidate_row_path
        )
        reference_probe = torch.load(
            reference_probe_path, map_location="cpu", weights_only=True
        )
        candidate_probe = torch.load(
            candidate_probe_path, map_location="cpu", weights_only=True
        )
        recorded_comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    except (
        OSError,
        UnicodeError,
        ValueError,
        EOFError,
        RuntimeError,
        KeyError,
        pickle.UnpicklingError,
    ) as exc:
        errors.append(f"{label} evidence could not be read: {exc}")
        return {"status": "fail"}
    if len(reference_rows) != 1 or len(candidate_rows) != 1:
        errors.append(f"{label} row artifacts must contain exactly one row")
        return {"status": "fail"}
    reference_row = reference_rows[0]
    candidate_row = candidate_rows[0]
    formal_row = candidate_by_key.get((PAIRS[3], 8, 128, 128))
    for field, expected in (
        ("rwkv_implementation_requested", "auto"),
        ("rwkv_implementation_effective", "native_model"),
        ("rwkv_fast_token_backend_requested", "module_call"),
        ("rwkv_native_model_backend_requested", "eager"),
        ("effective_backend", "eager"),
        ("step_backend", "rwkv_fast_token"),
        ("cache_type", "NativeRWKV7Cache"),
        ("batch_size", 8),
        ("prompt_tokens", 128),
        ("decode_tokens", 128),
        ("warmup", 1),
        ("runs", 1),
        ("status", "pass"),
        ("logits_finite", True),
    ):
        if not _strict_equal(reference_row.get(field), expected):
            errors.append(
                f"{label} reference {field}={reference_row.get(field)!r}, expected {expected!r}"
            )
    for field, expected in (
        ("rwkv_implementation_requested", "auto"),
        ("rwkv_implementation_effective", "native_model"),
        ("rwkv_fast_token_backend_requested", "native_graph"),
        ("rwkv_native_model_backend_requested", "native_graph"),
        ("effective_backend", "native_graph"),
        ("step_backend", "rwkv_fast_token"),
        ("cache_type", "NativeRWKV7Cache"),
        ("batch_size", 8),
        ("prompt_tokens", 128),
        ("decode_tokens", 128),
        ("warmup", 3),
        ("runs", 7),
        ("status", "pass"),
        ("logits_finite", True),
        ("rwkv_native_graph_rkv_policy", "vkwr_auto"),
        ("rwkv_native_graph_fused_norm_mix_num_warps", 8),
        ("rwkv_native_graph_state_dtype", "torch.float16"),
        ("rwkv_native_graph_triton_fp16_state", True),
        ("rwkv_native_graph_fp16_recurrent", False),
        ("rwkv_native_graph_fused_wavg_lora_selected", False),
        ("rwkv_native_graph_fused_wavg_lora_effective", False),
    ):
        if not _strict_equal(candidate_row.get(field), expected):
            errors.append(
                f"{label} candidate {field}={candidate_row.get(field)!r}, expected {expected!r}"
            )
    model_path = closure.get("model_path")
    if type(model_path) is not str or not model_path:
        errors.append(f"{label} model_path must be non-empty")
    for row_name, row in (
        ("reference", reference_row),
        ("candidate", candidate_row),
        ("formal", formal_row),
    ):
        if type(row) is not dict or row.get("model_id_or_path") != model_path:
            errors.append(f"{label} {row_name} model path mismatch")
    if type(formal_row) is dict:
        for field, value in formal_row.items():
            if field.startswith("rwkv_native_graph_") and not _strict_equal(
                candidate_row.get(field), value
            ):
                errors.append(f"{label} route differs from formal row at {field}")
        for field in RUNTIME_FIELDS:
            if not _strict_equal(candidate_row.get(field), formal_row.get(field)):
                errors.append(f"{label} runtime differs from formal row at {field}")
    if type(reference_probe) is not dict or type(candidate_probe) is not dict:
        errors.append(f"{label} probe payloads must be dictionaries")
        return {"status": "fail"}
    recomputed = compare_rwkv_probes(reference_probe, candidate_probe, 0.9999)
    contract_errors: list[str] = []
    if recomputed.get("probe_batch_size") != 8:
        contract_errors.append("probe batch size mismatch")
    if recomputed.get("probe_tokens") != 512:
        contract_errors.append("probe token count mismatch")
    if recomputed.get("distinct_batch_prompts") is not True:
        contract_errors.append("B8 prompts are not distinct")
    recomputed["contract_errors"] = contract_errors
    if contract_errors:
        recomputed["status"] = "fail"
    if recorded_comparison != recomputed:
        errors.append(f"{label} recorded comparison does not match recomputation")
    if recomputed.get("status") != "pass":
        errors.append(f"{label} comparison failed")
    closure_pass = (
        recomputed.get("status") == "pass" and len(errors) == error_count_before
    )
    return {
        "status": "pass" if closure_pass else "fail",
        "prompt_logits_cosine": recomputed.get("prompt_logits_cosine"),
        "final_logits_cosine": recomputed.get("final_logits_cosine"),
        "greedy_tokens_match": recomputed.get("greedy_tokens_match"),
        "reference_decode_logits_all_finite": recomputed.get(
            "reference_decode_logits_all_finite"
        ),
        "candidate_decode_logits_all_finite": recomputed.get(
            "native_decode_logits_all_finite"
        ),
    }


def validate_correctness_manifest(
    path: Path, candidate_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {"status": "fail", "errors": [f"correctness manifest: {exc}"]}
    if type(doc) is not dict:
        return {"status": "fail", "errors": ["correctness manifest must be an object"]}
    if doc.get("schema_version") != 1:
        errors.append("correctness manifest schema_version must be 1")
    if doc.get("protocol") != "rwkv_native_graph_fla_correctness_v100_v1":
        errors.append("correctness manifest protocol mismatch")
    candidate_commits = {
        row.get("benchmark_repository_commit") for row in candidate_rows
    }
    if (
        len(candidate_commits) != 1
        or doc.get("benchmark_repository_commit") not in candidate_commits
    ):
        errors.append("correctness manifest commit does not match candidate rows")
    coverage = doc.get("coverage")
    expected_coverage = {
        "models": 4,
        "batch_sizes": [1, 8],
        "entries": 8,
        "baseline_fresh_gpu_processes": 8,
        "candidate_additional_gpu_processes": 1,
        "candidate_formal_lane_processes": 8,
        "targeted_native_eager_fresh_gpu_processes": 1,
        "prompt_tokens": 2048,
        "decode_tokens": 512,
        "probe_tokens": 512,
        "targeted_closure_entries": 1,
    }
    if not _strict_equal(coverage, expected_coverage):
        errors.append(
            f"correctness coverage={coverage!r}, expected {expected_coverage!r}"
        )
    entries = doc.get("entries")
    if type(entries) is not list:
        errors.append("correctness entries must be a list")
        entries = []
    expected_keys = {(pair, batch) for pair in PAIRS for batch in (1, 8)}
    seen_keys: set[tuple[Any, Any]] = set()
    prompt_cosines: list[float] = []
    final_cosines: list[float] = []
    base = path.parent
    candidate_by_key = {
        _cell_key(row): {
            key: value for key, value in row.items() if not key.startswith("_")
        }
        for row in candidate_rows
    }
    for index, entry in enumerate(entries):
        label = f"correctness entry {index}"
        if type(entry) is not dict:
            errors.append(f"{label} must be an object")
            continue
        pair = entry.get("model_pair")
        batch = entry.get("batch_size")
        key = (pair, batch)
        if key in seen_keys:
            errors.append(f"duplicate correctness key {key!r}")
        seen_keys.add(key)
        if key not in expected_keys:
            errors.append(f"unexpected correctness key {key!r}")
            continue
        for field, expected in (
            ("model_size_label", RWKV_SIZES[pair][0]),
            ("prompt_tokens", 2048),
            ("decode_tokens", 512),
            ("probe_tokens", 512),
        ):
            if not _strict_equal(entry.get(field), expected):
                errors.append(
                    f"{label}.{field}={entry.get(field)!r}, expected {expected!r}"
                )
        fla = entry.get("fla_reference")
        native = entry.get("native_candidate")
        if type(fla) is not dict or type(native) is not dict:
            errors.append(f"{label} must contain FLA/native artifact objects")
            continue
        fla_row_path = _resolve_artifact(
            base, fla.get("row"), f"{label}.fla.row", errors
        )
        fla_probe_path = _resolve_artifact(
            base, fla.get("probe"), f"{label}.fla.probe", errors
        )
        native_row_path = _resolve_artifact(
            base, native.get("row"), f"{label}.native.row", errors
        )
        native_probe_path = _resolve_artifact(
            base, native.get("probe"), f"{label}.native.probe", errors
        )
        _resolve_artifact(
            base, native.get("source_lane"), f"{label}.native.source_lane", errors
        )
        comparison_path = _resolve_artifact(
            base, entry.get("comparison"), f"{label}.comparison", errors
        )
        if not all(
            item.is_file()
            for item in (
                fla_row_path,
                fla_probe_path,
                native_row_path,
                native_probe_path,
                comparison_path,
            )
        ):
            continue
        try:
            fla_rows = _read_jsonl_bytes(fla_row_path.read_bytes(), fla_row_path)
            native_rows = _read_jsonl_bytes(
                native_row_path.read_bytes(), native_row_path
            )
            fla_probe = torch.load(
                fla_probe_path, map_location="cpu", weights_only=True
            )
            native_probe = torch.load(
                native_probe_path, map_location="cpu", weights_only=True
            )
            recorded_comparison = json.loads(
                comparison_path.read_text(encoding="utf-8")
            )
        except (
            OSError,
            UnicodeError,
            ValueError,
            EOFError,
            RuntimeError,
            KeyError,
        ) as exc:
            errors.append(f"{label} evidence could not be read: {exc}")
            continue
        if len(fla_rows) != 1 or len(native_rows) != 1:
            errors.append(f"{label} row artifacts must contain exactly one row")
            continue
        fla_row = fla_rows[0]
        native_row = {
            key: value
            for key, value in native_rows[0].items()
            if not key.startswith("_")
        }
        expected_candidate = candidate_by_key.get((pair, batch, 2048, 512))
        if native_row != expected_candidate:
            errors.append(
                f"{label} native row does not equal the formal candidate cell"
            )
        for field, expected in (
            ("rwkv_implementation_requested", "wrapper_repo"),
            ("rwkv_implementation_effective", "wrapper_repo"),
            ("effective_backend", "fla"),
            ("cache_type", "RWKV7StateCache"),
            ("batch_size", batch),
            ("prompt_tokens", 2048),
            ("decode_tokens", 512),
            ("status", "pass"),
            ("logits_finite", True),
        ):
            if not _strict_equal(fla_row.get(field), expected):
                errors.append(
                    f"{label} FLA {field}={fla_row.get(field)!r}, expected {expected!r}"
                )
        if type(fla_probe) is not dict or type(native_probe) is not dict:
            errors.append(f"{label} probe payloads must be dictionaries")
            continue
        recomputed = compare_rwkv_probes(fla_probe, native_probe, 0.9999)
        contract_errors: list[str] = []
        if recomputed["probe_batch_size"] != batch:
            contract_errors.append("probe batch size mismatch")
        if recomputed["probe_tokens"] != 512:
            contract_errors.append("probe token count mismatch")
        if batch == 8 and recomputed["distinct_batch_prompts"] is not True:
            contract_errors.append("B8 prompts are not distinct")
        recomputed["contract_errors"] = contract_errors
        if contract_errors:
            recomputed["status"] = "fail"
        if recorded_comparison != recomputed:
            errors.append(
                f"{label} recorded comparison does not match probe recomputation"
            )
        if recomputed.get("status") != "pass":
            errors.append(f"{label} correctness comparison failed")
        for field, bucket in (
            ("prompt_logits_cosine", prompt_cosines),
            ("final_logits_cosine", final_cosines),
        ):
            value = recomputed.get(field)
            if _is_finite_real(value):
                bucket.append(float(value))
    if seen_keys != expected_keys:
        errors.append(
            f"correctness keys={sorted(seen_keys, key=repr)!r}, expected all 8 pair/batch keys"
        )
    closure_metrics = _validate_targeted_same_implementation_closure(
        doc, path, candidate_by_key, errors
    )
    return {
        "status": "pass" if not errors else "fail",
        "entries": len(entries),
        "minimum_prompt_logits_cosine": min(prompt_cosines) if prompt_cosines else None,
        "minimum_final_logits_cosine": min(final_cosines) if final_cosines else None,
        "targeted_same_implementation_closure": closure_metrics,
        "errors": errors,
    }


def validate_provenance(
    *,
    candidate_path: Path,
    reference_paths: list[Path],
    candidate_rows: list[dict[str, Any]],
    route_manifest_path: Path,
    qwen_route_manifest_paths: list[Path],
    correctness_manifest_path: Path,
    runtime_lock_path: Path,
    candidate_model_hashes_path: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    base = route_manifest_path.parent
    try:
        route = json.loads(route_manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {"status": "fail", "errors": [f"candidate route manifest: {exc}"]}
    if type(route) is not dict:
        return {
            "status": "fail",
            "errors": ["candidate route manifest must be an object"],
        }
    for field, expected in (
        ("schema_version", 1),
        ("protocol", PROTOCOL),
        ("repository_clean_pre_and_post", True),
        ("candidate_rows", 48),
    ):
        if not _strict_equal(route.get(field), expected):
            errors.append(
                f"candidate route {field}={route.get(field)!r}, expected {expected!r}"
            )
    commits = {row.get("benchmark_repository_commit") for row in candidate_rows}
    if len(commits) != 1 or route.get("benchmark_repository_commit") not in commits:
        errors.append("candidate route commit does not match candidate rows")
    candidate_artifact = _resolve_artifact(
        base, route.get("candidate_result"), "candidate_result", errors
    )
    if (
        candidate_artifact.is_file()
        and candidate_artifact.read_bytes() != candidate_path.read_bytes()
    ):
        errors.append("candidate route result bytes differ from --candidate")
    correctness_artifact = _resolve_artifact(
        base,
        route.get("native_graph_fla_correctness_manifest"),
        "correctness_manifest",
        errors,
    )
    if (
        correctness_artifact.is_file()
        and correctness_artifact.read_bytes() != correctness_manifest_path.read_bytes()
    ):
        errors.append("candidate route correctness manifest differs from CLI input")
    runtime_artifact = _resolve_artifact(
        base, route.get("runtime_lock"), "runtime_lock", errors
    )
    if (
        runtime_artifact.is_file()
        and runtime_artifact.read_bytes() != runtime_lock_path.read_bytes()
    ):
        errors.append("candidate route runtime lock differs from CLI input")
    model_contract = route.get("model_hash_contract")
    if type(model_contract) is not dict:
        errors.append("candidate route model_hash_contract must be an object")
    else:
        for field, expected in (
            ("algorithm", "sha256"),
            ("scope", "every recursive regular file"),
            ("byte_identical", True),
        ):
            if not _strict_equal(model_contract.get(field), expected):
                errors.append(f"candidate model hash {field} mismatch")
        before = _resolve_artifact(
            base, model_contract.get("before"), "candidate model hashes before", errors
        )
        after = _resolve_artifact(
            base, model_contract.get("after"), "candidate model hashes after", errors
        )
        if (
            before.is_file()
            and before.read_bytes() != candidate_model_hashes_path.read_bytes()
        ):
            errors.append("candidate model hash input differs from route manifest")
        if (
            before.is_file()
            and after.is_file()
            and before.read_bytes() != after.read_bytes()
        ):
            errors.append("candidate model hashes changed during capture")
    forced = route.get("forced_environment")
    expected_forced = {
        "CUDA_VISIBLE_DEVICES": "0",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_CUDA_ARCH_LIST": "7.0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "RWKV7_FAST_TOKEN_BACKEND": "native_graph",
        "RWKV7_NATIVE_MODEL_BACKEND": "native_graph",
        "RWKV7_NATIVE_PREFILL_GRAPH": "unset_exact_card_policy",
        "RWKV7_NATIVE_GRAPH_ADA_WAGV_LORA_REQUIRE_EXTENSION": "0",
        "RWKV7_NATIVE_GRAPH_SM70_WAGV_LORA_REQUIRE_EXTENSION": "per_lane_exact_v100_policy",
        "RWKV7_NATIVE_GRAPH_RKV_POLICY": "per_lane_exact_v100_policy",
        "RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX_NUM_WARPS": "per_lane_exact_v100_policy",
        "RWKV7_NATIVE_GRAPH_FUSED_WAVG_LORA": "per_lane_exact_v100_policy",
        "RWKV7_NATIVE_GRAPH_ADA_WAGV_BMM": "0",
        "RWKV7_NATIVE_GRAPH_SM120_WAGV_BMM_G": "0",
        "RWKV7_NATIVE_GRAPH_SM120_COMPILED_FFN": "0",
    }
    if type(forced) is not dict:
        errors.append("candidate forced_environment must be an object")
    else:
        for field, expected in expected_forced.items():
            if not _strict_equal(forced.get(field), expected):
                errors.append(f"candidate forced environment {field} mismatch")
        pythonpath = forced.get("PYTHONPATH")
        if type(pythonpath) is not str or len(pythonpath.split(":")) != 3:
            errors.append(
                "candidate PYTHONPATH must bind repository, Triton, and FLA roots"
            )
    lanes = route.get("lanes")
    if type(lanes) is not list:
        errors.append("candidate lanes must be a list")
        lanes = []
    expected_lane_keys = {(pair, batch) for pair in PAIRS for batch in (1, 8)}
    seen_lane_keys: set[tuple[Any, Any]] = set()
    candidate_by_key = {_cell_key(row): row for row in candidate_rows}
    for index, lane in enumerate(lanes):
        label = f"candidate lane {index}"
        if type(lane) is not dict:
            errors.append(f"{label} must be an object")
            continue
        pair = lane.get("model_pair")
        batch = lane.get("batch_size")
        key = (pair, batch)
        if key in seen_lane_keys:
            errors.append(f"duplicate {label} key {key!r}")
        seen_lane_keys.add(key)
        if key not in expected_lane_keys:
            errors.append(f"unexpected {label} key {key!r}")
            continue
        closure_lane = pair == PAIRS[3] and batch == 8
        for field, expected in (
            ("rows", 6),
            ("probe_cell", [batch, 2048, 512]),
            ("ada_wagv_lora_require_extension", False),
            ("sm70_wagv_lora_require_extension", batch == 1),
            ("rkv_policy", "vkwr_auto" if closure_lane else "manual"),
            ("fused_norm_mix_num_warps", 8 if closure_lane else 4),
            ("fused_wavg_lora", not closure_lane),
        ):
            if not _strict_equal(lane.get(field), expected):
                errors.append(f"{label} {field} mismatch")
        lane_path = _resolve_artifact(
            base, lane.get("artifact"), f"{label} artifact", errors
        )
        if lane_path.is_file():
            try:
                lane_rows = _read_jsonl_bytes(lane_path.read_bytes(), lane_path)
            except (OSError, UnicodeError, ValueError) as exc:
                errors.append(f"{label} could not be read: {exc}")
                lane_rows = []
            expected_rows = [
                candidate_by_key.get((pair, batch, prompt, decode))
                for prompt in (128, 512, 2048)
                for decode in (128, 512)
            ]
            normalized_lane_rows = [
                {name: value for name, value in row.items() if not name.startswith("_")}
                for row in lane_rows
            ]
            normalized_expected_rows = [
                {name: value for name, value in row.items() if not name.startswith("_")}
                if type(row) is dict
                else None
                for row in expected_rows
            ]
            if normalized_lane_rows != normalized_expected_rows:
                errors.append(f"{label} rows differ from candidate matrix")
    if seen_lane_keys != expected_lane_keys:
        errors.append("candidate lane coverage must contain all four pairs and B1/B8")
    system_path = _resolve_artifact(
        base, route.get("system_identity"), "system_identity", errors
    )
    if system_path.is_file():
        try:
            with system_path.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle, skipinitialspace=True)
                system_rows = list(reader)
        except (OSError, UnicodeError, csv.Error) as exc:
            errors.append(f"system identity could not be read: {exc}")
            system_rows = []
        if len(system_rows) != 2:
            errors.append(
                f"system identity must contain exactly two V100 rows, got {len(system_rows)}"
            )
        for row in system_rows:
            if row.get("name") != EXPECTED_DEVICE:
                errors.append(
                    f"system GPU name={row.get('name')!r}, expected {EXPECTED_DEVICE!r}"
                )
            if row.get("compute_cap") != "7.0":
                errors.append("system compute_cap must be 7.0")
            if row.get("driver_version") != "580.173.02":
                errors.append("system driver_version must be 580.173.02")
            if row.get("memory.total [MiB]") != "32768 MiB":
                errors.append("system memory.total must be 32768 MiB")

    try:
        runtime = json.loads(runtime_lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"runtime lock could not be read: {exc}")
        runtime = {}
    expected_runtime = {
        "python": "3.11.15",
        "torch": "2.5.1+cu124",
        "torch_cuda": "12.4",
        "triton": "3.4.0",
        "transformers": "5.12.1",
        "fla": "0.5.1",
        "causal_conv1d": None,
    }
    if runtime.get("runtime") != expected_runtime:
        errors.append(
            f"runtime lock={runtime.get('runtime')!r}, expected {expected_runtime!r}"
        )
    if runtime.get("torch_cuda_arch_list") != "7.0":
        errors.append("runtime lock torch_cuda_arch_list must be 7.0")

    if len(qwen_route_manifest_paths) != 4 or len(reference_paths) != 4:
        errors.append("exactly four Qwen result and route manifests are required")
    seen_qwen: set[str] = set()
    reference_by_name = {path.name: path for path in reference_paths}
    for manifest_path in qwen_route_manifest_paths:
        try:
            qwen = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(f"Qwen route {manifest_path}: {exc}")
            continue
        pair = qwen.get("model_pair")
        if pair in seen_qwen:
            errors.append(f"duplicate Qwen route pair {pair!r}")
        seen_qwen.add(pair)
        if pair not in PAIRS:
            errors.append(f"unexpected Qwen route pair {pair!r}")
            continue
        for field, expected in (
            ("schema_version", 1),
            ("protocol", "qwen35_v100_best_optimized_hf_v1"),
            ("repository_clean_pre_and_post", True),
            ("decode_route", QWEN_ROUTES[pair]),
            ("sdpa_policy", QWEN_SDPA_POLICIES[pair]),
            ("compile_mode", None),
        ):
            if not _strict_equal(qwen.get(field), expected):
                errors.append(f"Qwen route {pair} {field} mismatch")
        result_record = qwen.get("result")
        result_path = _resolve_artifact(
            manifest_path.parent, result_record, f"Qwen route {pair} result", errors
        )
        expected_result = reference_by_name.get(result_path.name)
        if (
            not result_path.is_file()
            or expected_result is None
            or not expected_result.is_file()
            or result_path.read_bytes() != expected_result.read_bytes()
        ):
            errors.append(
                f"Qwen route {pair} result is not one of the four reference inputs"
            )
        qwen_hashes = qwen.get("model_hash_contract")
        if (
            type(qwen_hashes) is not dict
            or qwen_hashes.get("byte_identical") is not True
        ):
            errors.append(f"Qwen route {pair} model hashes were not stable")
        else:
            before = _resolve_artifact(
                manifest_path.parent,
                qwen_hashes.get("before"),
                f"Qwen {pair} hashes before",
                errors,
            )
            after = _resolve_artifact(
                manifest_path.parent,
                qwen_hashes.get("after"),
                f"Qwen {pair} hashes after",
                errors,
            )
            if (
                before.is_file()
                and after.is_file()
                and before.read_bytes() != after.read_bytes()
            ):
                errors.append(f"Qwen route {pair} model hashes changed")
    if seen_qwen != set(PAIRS):
        errors.append(
            f"Qwen route coverage={sorted(seen_qwen)!r}, expected all four pairs"
        )
    return {"status": "pass" if not errors else "fail", "errors": errors}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--correctness-manifest", type=Path, required=True)
    parser.add_argument("--candidate-route-manifest", type=Path, required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    parser.add_argument("--candidate-model-hashes", type=Path, required=True)
    parser.add_argument("--qwen-result", type=Path, action="append", required=True)
    parser.add_argument(
        "--qwen-route-manifest", type=Path, action="append", required=True
    )
    parser.add_argument("--expected-device", default=EXPECTED_DEVICE)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--paired-table", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate_bytes = args.candidate.read_bytes()
    reference_bytes = args.reference.read_bytes()
    summary = validate_paired_pd(
        _read_jsonl_bytes(candidate_bytes, args.candidate),
        _read_jsonl_bytes(reference_bytes, args.reference),
        expected_device=args.expected_device,
        candidate_sha256=_sha256(candidate_bytes),
        reference_sha256=_sha256(reference_bytes),
    )
    correctness = validate_correctness_manifest(
        args.correctness_manifest, _read_jsonl_bytes(candidate_bytes, args.candidate)
    )
    summary["rwkv_native_graph_fla_correctness"] = correctness
    if correctness["status"] != "pass":
        summary["errors"].extend(
            f"RWKV correctness: {error}" for error in correctness["errors"]
        )
        summary["status"] = "fail"
        summary["paired_pd_table_eligible"] = False
    provenance = validate_provenance(
        candidate_path=args.candidate,
        reference_paths=args.qwen_result,
        candidate_rows=_read_jsonl_bytes(candidate_bytes, args.candidate),
        route_manifest_path=args.candidate_route_manifest,
        qwen_route_manifest_paths=args.qwen_route_manifest,
        correctness_manifest_path=args.correctness_manifest,
        runtime_lock_path=args.runtime_lock,
        candidate_model_hashes_path=args.candidate_model_hashes,
    )
    summary["provenance"] = provenance
    if provenance["status"] != "pass":
        summary["errors"].extend(
            f"Provenance: {error}" for error in provenance["errors"]
        )
        summary["status"] = "fail"
        summary["paired_pd_table_eligible"] = False
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(summary), encoding="utf-8")
    if summary["paired_pd_table_eligible"]:
        args.paired_table.write_text(
            "".join(json.dumps(cell) + "\n" for cell in summary["cells"]),
            encoding="utf-8",
        )
    elif args.paired_table.exists():
        args.paired_table.unlink()
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
