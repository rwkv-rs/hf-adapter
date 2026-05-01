#!/usr/bin/env python3
"""Validate the strict RTX 4080 RWKV/Qwen Prefill+Decode matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

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


PROTOCOL = "qwen35_paired_pd_v1"
CANDIDATE_LANE = "best_optimized_hf"
EXPECTED_DEVICE = "NVIDIA GeForce RTX 4080"
STRICT_RATIO_GATE = 1.0
PAIRS = (
    "rwkv-0.4b__qwen3.5-0.8b",
    "rwkv-1.5b__qwen3.5-2b",
    "rwkv-2.9b__qwen3.5-4b",
)
PARAMETERS = {
    "rwkv-0.4b__qwen3.5-0.8b": (450_767_872, 752_393_024),
    "rwkv-1.5b__qwen3.5-2b": (1_527_404_544, 1_881_825_088),
    "rwkv-2.9b__qwen3.5-4b": (2_947_735_040, 4_205_751_296),
}
RWKV_SIZES = {
    "rwkv-0.4b__qwen3.5-0.8b": ("0.4b", 24),
    "rwkv-1.5b__qwen3.5-2b": ("1.5b", 24),
    "rwkv-2.9b__qwen3.5-4b": ("2.9b", 32),
}
QWEN_ROUTES = {
    PAIRS[0]: "static_cache_inductor_cudagraph",
    PAIRS[1]: "static_cache_inductor_cudagraph",
    PAIRS[2]: "module_call_dynamic",
}
EXPECTED_KEYS = {
    (pair, batch, prompt, decode)
    for pair in PAIRS
    for batch, prompt, decode in EXPECTED_SHAPES
}
HEX_DIGITS = frozenset("0123456789abcdef")


def _is_finite_real(value: Any) -> bool:
    return (
        type(value) in {int, float}
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


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
        row["_line_number"] = line_number
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
    raw_median = row.get(raw_median_field)
    rounded_median = row.get(median_field)
    raw_tokps = row.get(raw_tokps_field)
    if not _is_finite_real(raw_median) or float(raw_median) != expected_median:
        errors.append(
            f"{row.get('_source', '<row>')}: {raw_median_field} does not exactly match samples"
        )
    if (
        not _is_finite_real(rounded_median)
        or abs(float(rounded_median) - expected_median) > 1e-6
    ):
        errors.append(
            f"{row.get('_source', '<row>')}: {median_field} does not match samples"
        )
    expected_tokps = tokens / expected_median
    if not _is_finite_real(raw_tokps) or not math.isclose(
        float(raw_tokps), expected_tokps, rel_tol=1e-12, abs_tol=1e-12
    ):
        errors.append(
            f"{row.get('_source', '<row>')}: {raw_tokps_field} does not equal tokens/raw median"
        )


def _expected_accumulation(pair: str, batch: int, prompt: int) -> tuple[bool, bool]:
    hidden, layers = {
        "rwkv-0.4b__qwen3.5-0.8b": (1024, 24),
        "rwkv-1.5b__qwen3.5-2b": (2048, 24),
        "rwkv-2.9b__qwen3.5-4b": (2560, 32),
    }[pair]
    block_only = {
        (1024, 24, 8, 512),
        (1024, 24, 8, 2048),
        (2048, 24, 8, 512),
        (2048, 24, 8, 2048),
        (2560, 32, 1, 512),
        (2560, 32, 1, 2048),
    }
    block = (hidden, layers, batch, prompt) in block_only
    return (not block, block)


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
        ("gpu_arch", "sm_89"),
        ("gpu_compute_capability", [8, 9]),
        ("rwkv_fast_token_backend_requested", "native_graph"),
        ("rwkv_native_model_backend_requested", "native_graph"),
        ("effective_backend", "native_graph"),
        ("step_backend", "rwkv_fast_token"),
        ("cache_type", "NativeRWKV7Cache"),
        ("prefill_effective_backend", "native_prefill_graph"),
        ("prefill_backend_effective", "native_prefill_graph"),
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
    expected_rkv_policy = (
        "manual" if pair == PAIRS[2] and batch == 8 else "vkwr_auto"
    )
    _require(row, "rwkv_native_graph_rkv_policy", expected_rkv_policy, errors)
    extension_layers = list(range(layer_count)) if batch == 1 else []
    for suffix, expected in (
        ("requested", batch == 1),
        ("selected", batch == 1),
        ("effective", batch == 1),
        ("selected_layers", extension_layers),
        ("effective_layers", extension_layers),
        ("effective_layer_count", len(extension_layers)),
        ("full_model_effective", batch == 1),
    ):
        _require(
            row,
            f"rwkv_native_graph_ada_wagv_lora_extension_{suffix}",
            expected,
            errors,
        )
    _require(row, "rwkv_native_graph_ada_wagv_bmm_requested", True, errors)
    expected_layers = list(range(layer_count)) if batch == 8 else []
    for suffix, expected in (
        ("selected", batch == 8),
        ("effective", batch == 8),
        ("selected_layers", expected_layers),
        ("effective_layers", expected_layers),
        ("effective_layer_count", len(expected_layers)),
        ("full_model_effective", batch == 8),
    ):
        _require(
            row,
            f"rwkv_native_graph_ada_wagv_bmm_{suffix}",
            expected,
            errors,
        )
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
            _require(row, f"rwkv_native_graph_{route}_{suffix}", expected, errors)

    global_accum, block_accum = _expected_accumulation(pair, batch, prompt)
    _require(row, "rwkv_prefill_global_fp16_accum_effective", global_accum, errors)
    _require(row, "rwkv_prefill_block_fp16_accum_effective", block_accum, errors)
    _validate_samples(row, "prefill", batch * prompt, errors)
    _validate_samples(row, "decode", batch * decode, errors)
    for field in RUNTIME_FIELDS:
        value = row.get(field)
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


def _index_rows(
    rows: list[dict[str, Any]], label: str, errors: list[str]
) -> dict[tuple[Any, Any, Any, Any], dict[str, Any]]:
    keys = [_cell_key(row) for row in rows]
    counts = Counter(keys)
    duplicates = sorted((key for key, count in counts.items() if count > 1), key=repr)
    missing = sorted(EXPECTED_KEYS - set(keys), key=repr)
    extra = sorted(set(keys) - EXPECTED_KEYS, key=repr)
    if len(rows) != 36:
        errors.append(f"{label} row count={len(rows)}, expected 36")
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


def _runtime_signature(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        value
        if type(value := row.get(field)) is str
        else f"<invalid-{field}:{value!r}>"
        for field in RUNTIME_FIELDS
    )


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

    candidate_runtime = {_runtime_signature(row) for row in candidate_rows}
    reference_runtime = {_runtime_signature(row) for row in reference_rows}
    if len(candidate_runtime) != 1:
        errors.append("candidate rows do not have one runtime signature")
    if len(reference_runtime) != 1:
        errors.append("reference rows do not have one runtime signature")
    if len(candidate_runtime) == len(reference_runtime) == 1 and (
        candidate_runtime != reference_runtime
    ):
        errors.append("candidate/reference runtime signatures differ")

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
        and len(cells) == 36
        and all(value == 36 for value in passing.values())
    )
    if len(cells) == 36 and not all(value == 36 for value in passing.values()):
        errors.append(
            "strict raw/adjusted Prefill+Decode gates require 36/36 cells each"
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
            "expected_cells": 36,
        },
        "gate": {
            "comparison": ">",
            "threshold": STRICT_RATIO_GATE,
            "uses_unrounded_raw_throughput": True,
            "passing_cells": passing,
            "total_cells": 36,
        },
        "paired_pd_table_eligible": eligible,
        "continuous_e2e_eligible": False,
        "cells": cells,
        "red_cells": [cell for cell in cells if cell["strict_pass"] is not True],
        "errors": errors,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# RTX 4080 strict paired Prefill/Decode v1",
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
            f"| {pair} | {cell['batch_size']} | {cell['prompt_tokens']} | "
            f"{cell['decode_tokens']} | {cell['raw_prefill_ratio']:.6f}x / "
            f"{cell['raw_decode_ratio']:.6f}x | "
            f"{cell['adjusted_prefill_ratio']:.6f}x / "
            f"{cell['adjusted_decode_ratio']:.6f}x | "
            f"{'PASS' if cell['strict_pass'] else 'FAIL'} |"
        )
    if summary["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in summary["errors"])
    return "\n".join(lines) + "\n"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--expected-device", default=EXPECTED_DEVICE)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--paired-table", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate_bytes = args.candidate.read_bytes()
    reference_bytes = args.reference.read_bytes()
    candidate_rows = _read_jsonl_bytes(candidate_bytes, args.candidate)
    reference_rows = _read_jsonl_bytes(reference_bytes, args.reference)
    summary = validate_paired_pd(
        candidate_rows,
        reference_rows,
        expected_device=args.expected_device,
        candidate_sha256=_sha256(candidate_bytes),
        reference_sha256=_sha256(reference_bytes),
    )
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
