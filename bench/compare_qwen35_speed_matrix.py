#!/usr/bin/env python3
"""Join RWKV/Qwen3.5 speed rows and enforce declared matrix gates."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

CELL_FIELDS = (
    "model_pair",
    "prompt_tokens",
    "decode_tokens",
    "batch_size",
    "dtype",
    "quantization",
)


def quantization_family(value: Any) -> str:
    """Normalize implementation names to the W8/W4 acceptance contract."""

    value = str(value or "none").strip().lower().replace("-", "_")
    if value in {
        "bnb8",
        "bnb8_a8w8_head",
        "torchao_w8",
        "mm8",
        "a8w8",
        "w8",
        "int8",
    }:
        return "w8"
    if value in {"bnb4", "torchao_w4", "mm4", "w4", "int4"}:
        return "w4"
    return value


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{lineno}: {exc}") from exc
        if row.get("axis") == "qwen35_cross_model_speed":
            row["_lineno"] = lineno
            rows.append(row)
    return rows


def cell_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        quantization_family(row.get(field)) if field == "quantization" else row.get(field)
        for field in CELL_FIELDS
    )


def key_dict(key: tuple[Any, ...]) -> dict[str, Any]:
    return dict(zip(CELL_FIELDS, key))


def ratio(candidate: Any, reference: Any) -> float | None:
    if candidate is None or reference in (None, 0):
        return None
    return round(float(candidate) / float(reference), 6)


def active_parameter_efficiency(row: dict[str, Any], phase: str) -> float | None:
    recorded = row.get(f"{phase}_tokps_per_active_billion")
    if recorded is not None:
        return float(recorded)
    tokps = row.get(f"{phase}_tokps_total")
    active = row.get("active_parameter_count")
    if tokps is None or active is None or float(active) <= 0:
        return None
    return float(tokps) / (float(active) / 1e9)


def median_or_none(values: list[float]) -> float | None:
    return round(float(statistics.median(values)), 6) if values else None


def reference_backend_matches(row: dict[str, Any], required: str) -> bool:
    if required == "any":
        return True
    requested = row.get("qwen_backend_requested")
    effective = row.get("effective_backend")
    if required == "fla":
        return bool(
            requested == "fla"
            and effective
            in {
                "qwen_fla_gated_delta_rule",
                "qwen_fla_gated_delta_rule_torch_conv",
                "qwen_fla_gated_delta_rule_fla_triton_conv",
            }
            and row.get("qwen_operator_contract_pass") is True
        )
    return bool(
        requested == "torch"
        and effective == "transformers_torch_fallback"
        and row.get("qwen_force_torch") is True
    )


def compare(
    rows: list[dict[str, Any]],
    *,
    expected_cells: int,
    min_prefill_speedup: float,
    min_decode_speedup: float,
    min_quant_prefill_speedup: float | None = None,
    min_quant_decode_speedup: float | None = None,
    require_native_candidate: bool = False,
    require_qwen_fast_path: bool = False,
    require_qwen_full_fused: bool = False,
    require_quant_memory_reduction: bool = False,
    require_prefill_mode_match: bool = False,
    require_quant_not_slower_than_dense: bool = False,
    required_reference_backend: str = "any",
    require_memory_not_larger: bool = False,
    min_active_parameter_throughput_ratio: float | None = None,
    min_active_parameter_efficiency_ratio: float | None = None,
) -> dict[str, Any]:
    indexed: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {"candidate": {}, "reference": {}}
    for row in rows:
        role = str(row.get("model_role") or "")
        if role in indexed:
            indexed[role][cell_key(row)] = row

    candidate_keys = set(indexed["candidate"])
    reference_keys = set(indexed["reference"])
    joined_keys = sorted(candidate_keys & reference_keys, key=lambda key: tuple(str(x) for x in key))
    missing_candidate = sorted(reference_keys - candidate_keys, key=lambda key: tuple(str(x) for x in key))
    missing_reference = sorted(candidate_keys - reference_keys, key=lambda key: tuple(str(x) for x in key))

    cells: list[dict[str, Any]] = []
    red_cells: list[dict[str, Any]] = []
    prefill_ratios: list[float] = []
    decode_ratios: list[float] = []
    backend_failures = 0
    memory_failures = 0
    prefill_mode_failures = 0
    quant_dense_speed_failures = 0
    family_metrics: dict[str, dict[str, list[float]]] = {}
    footprint_ratios: list[float] = []
    peak_vram_ratios: list[float] = []
    working_set_ratios: list[float] = []
    active_parameter_ratios: list[float] = []
    prefill_parameter_throughput_ratios: list[float] = []
    decode_parameter_throughput_ratios: list[float] = []
    prefill_parameter_efficiency_ratios: list[float] = []
    decode_parameter_efficiency_ratios: list[float] = []
    active_parameter_work_failures = 0
    active_parameter_efficiency_failures = 0
    active_parameter_failures = 0
    backend_matches = 0
    for key in joined_keys:
        candidate = indexed["candidate"][key]
        reference = indexed["reference"][key]
        both_pass = candidate.get("status") == "pass" and reference.get("status") == "pass"
        reference_backend_pass = reference_backend_matches(reference, required_reference_backend)
        backend_matches += int(reference_backend_pass)
        prefill_speedup = ratio(candidate.get("prefill_tokps_total"), reference.get("prefill_tokps_total"))
        decode_speedup = ratio(candidate.get("decode_tokps_total"), reference.get("decode_tokps_total"))
        footprint_ratio = ratio(candidate.get("model_footprint_mb"), reference.get("model_footprint_mb"))
        peak_vram_ratio = ratio(candidate.get("peak_vram_mb"), reference.get("peak_vram_mb"))
        working_set_ratio = ratio(
            candidate.get("runtime_working_set_mb"), reference.get("runtime_working_set_mb")
        )
        active_parameter_ratio = ratio(
            candidate.get("active_parameter_count"), reference.get("active_parameter_count")
        )
        prefill_parameter_throughput_ratio = ratio(
            candidate.get("prefill_active_parameter_tops"),
            reference.get("prefill_active_parameter_tops"),
        )
        decode_parameter_throughput_ratio = ratio(
            candidate.get("decode_active_parameter_tops"),
            reference.get("decode_active_parameter_tops"),
        )
        prefill_parameter_efficiency_ratio = ratio(
            active_parameter_efficiency(candidate, "prefill"),
            active_parameter_efficiency(reference, "prefill"),
        )
        decode_parameter_efficiency_ratio = ratio(
            active_parameter_efficiency(candidate, "decode"),
            active_parameter_efficiency(reference, "decode"),
        )
        active_parameter_work_pass = bool(
            min_active_parameter_throughput_ratio is None
            or (
                prefill_parameter_throughput_ratio is not None
                and decode_parameter_throughput_ratio is not None
                and prefill_parameter_throughput_ratio >= min_active_parameter_throughput_ratio
                and decode_parameter_throughput_ratio >= min_active_parameter_throughput_ratio
            )
        )
        active_parameter_efficiency_pass = bool(
            min_active_parameter_efficiency_ratio is None
            or (
                prefill_parameter_efficiency_ratio is not None
                and decode_parameter_efficiency_ratio is not None
                and prefill_parameter_efficiency_ratio >= min_active_parameter_efficiency_ratio
                and decode_parameter_efficiency_ratio >= min_active_parameter_efficiency_ratio
            )
        )
        active_parameter_pass = (
            active_parameter_work_pass and active_parameter_efficiency_pass
        )
        memory_pass = bool(
            not require_memory_not_larger
            or (
                footprint_ratio is not None
                and peak_vram_ratio is not None
                and footprint_ratio <= 1.0
                and peak_vram_ratio <= 1.0
            )
        )
        if prefill_speedup is not None:
            prefill_ratios.append(prefill_speedup)
        if decode_speedup is not None:
            decode_ratios.append(decode_speedup)
        quantized = key_dict(key).get("quantization") != "none"
        quant_family = str(key_dict(key).get("quantization") or "none")
        metrics = family_metrics.setdefault(
            quant_family,
            {
                "prefill": [],
                "decode": [],
                "quant_prefill_vs_dense": [],
                "quant_decode_vs_dense": [],
                "footprint_vs_dense": [],
                "peak_vram_vs_dense": [],
            },
        )
        if prefill_speedup is not None:
            metrics["prefill"].append(prefill_speedup)
        if decode_speedup is not None:
            metrics["decode"].append(decode_speedup)
        prefill_gate = (
            min_quant_prefill_speedup
            if quantized and min_quant_prefill_speedup is not None
            else min_prefill_speedup
        )
        decode_gate = (
            min_quant_decode_speedup
            if quantized and min_quant_decode_speedup is not None
            else min_decode_speedup
        )
        candidate_prefill_backend = candidate.get("prefill_effective_backend")
        candidate_decode_backend = candidate.get("effective_backend")
        reference_backend = reference.get("effective_backend")
        native_backend_pass = bool(
            not require_native_candidate
            or (
                candidate_prefill_backend in {"native_prefill", "native_prefill_graph"}
                and candidate_decode_backend == "native_graph"
            )
        )
        qwen_backend_pass = bool(
            not require_qwen_fast_path
            or (
                reference.get("qwen_operator_contract_pass") is True
                or (
                    reference_backend == "fla+causal_conv1d"
                    and reference.get("qwen_fast_path_verified") is True
                )
            )
        )
        qwen_full_fused_pass = bool(
            not require_qwen_full_fused
            or (
                reference.get("qwen_full_fused_contract_pass") is True
                and reference.get("qwen_fast_path_verified") is True
                and reference.get("qwen_conv_backend_effective")
                in {"causal_conv1d", "fla_triton", "mixed_accelerated"}
            )
        )
        candidate_prefill_chunk_size = int(candidate.get("prefill_chunk_size") or 0)
        reference_prefill_chunk_size = int(reference.get("prefill_chunk_size") or 0)
        prefill_mode_pass = bool(
            not require_prefill_mode_match
            or candidate_prefill_chunk_size == reference_prefill_chunk_size
        )
        quant_memory_ratio = None
        quant_peak_memory_ratio = None
        quant_memory_pass = True
        quant_prefill_speedup_vs_dense = None
        quant_decode_speedup_vs_dense = None
        quant_dense_prefill_mode_pass = True
        quant_dense_speed_pass = True
        dense_candidate = None
        if quantized and (require_quant_memory_reduction or require_quant_not_slower_than_dense):
            dense_key = tuple("none" if field == "quantization" else value for field, value in zip(CELL_FIELDS, key))
            dense_candidate = indexed["candidate"].get(dense_key)
        if quantized and require_quant_memory_reduction:
            quant_memory_ratio = ratio(
                candidate.get("model_footprint_mb"),
                dense_candidate.get("model_footprint_mb") if dense_candidate else None,
            )
            quant_peak_memory_ratio = ratio(
                candidate.get("peak_vram_mb"),
                dense_candidate.get("peak_vram_mb") if dense_candidate else None,
            )
            quant_memory_pass = bool(
                quant_memory_ratio is not None
                and quant_peak_memory_ratio is not None
                and quant_memory_ratio < 1.0
                and quant_peak_memory_ratio < 1.0
            )
            if quant_memory_ratio is not None:
                metrics["footprint_vs_dense"].append(quant_memory_ratio)
            if quant_peak_memory_ratio is not None:
                metrics["peak_vram_vs_dense"].append(quant_peak_memory_ratio)
        if quantized and require_quant_not_slower_than_dense:
            quant_dense_prefill_mode_pass = bool(
                dense_candidate is not None
                and candidate_prefill_chunk_size
                == int(dense_candidate.get("prefill_chunk_size") or 0)
            )
            quant_prefill_speedup_vs_dense = ratio(
                candidate.get("prefill_tokps_total"),
                dense_candidate.get("prefill_tokps_total") if dense_candidate else None,
            )
            quant_decode_speedup_vs_dense = ratio(
                candidate.get("decode_tokps_total"),
                dense_candidate.get("decode_tokps_total") if dense_candidate else None,
            )
            quant_dense_speed_pass = bool(
                quant_prefill_speedup_vs_dense is not None
                and quant_decode_speedup_vs_dense is not None
                and quant_dense_prefill_mode_pass
                and quant_prefill_speedup_vs_dense >= 1.0
                and quant_decode_speedup_vs_dense >= 1.0
            )
            if quant_prefill_speedup_vs_dense is not None:
                metrics["quant_prefill_vs_dense"].append(quant_prefill_speedup_vs_dense)
            if quant_decode_speedup_vs_dense is not None:
                metrics["quant_decode_vs_dense"].append(quant_decode_speedup_vs_dense)
        backend_pass = native_backend_pass and qwen_backend_pass and qwen_full_fused_pass
        backend_failures += int(not backend_pass)
        memory_failures += int(not quant_memory_pass)
        prefill_mode_failures += int(not prefill_mode_pass)
        quant_dense_speed_failures += int(not quant_dense_speed_pass)
        active_parameter_work_failures += int(not active_parameter_work_pass)
        active_parameter_efficiency_failures += int(not active_parameter_efficiency_pass)
        active_parameter_failures += int(not active_parameter_pass)
        if footprint_ratio is not None:
            footprint_ratios.append(footprint_ratio)
        if peak_vram_ratio is not None:
            peak_vram_ratios.append(peak_vram_ratio)
        if working_set_ratio is not None:
            working_set_ratios.append(working_set_ratio)
        if active_parameter_ratio is not None:
            active_parameter_ratios.append(active_parameter_ratio)
        if prefill_parameter_throughput_ratio is not None:
            prefill_parameter_throughput_ratios.append(prefill_parameter_throughput_ratio)
        if decode_parameter_throughput_ratio is not None:
            decode_parameter_throughput_ratios.append(decode_parameter_throughput_ratio)
        if prefill_parameter_efficiency_ratio is not None:
            prefill_parameter_efficiency_ratios.append(prefill_parameter_efficiency_ratio)
        if decode_parameter_efficiency_ratio is not None:
            decode_parameter_efficiency_ratios.append(decode_parameter_efficiency_ratio)
        passed = bool(
            both_pass
            and reference_backend_pass
            and prefill_speedup is not None
            and decode_speedup is not None
            and prefill_speedup >= prefill_gate
            and decode_speedup >= decode_gate
            and backend_pass
            and quant_memory_pass
            and prefill_mode_pass
            and quant_dense_speed_pass
            and memory_pass
            and active_parameter_pass
        )
        cell = {
            **key_dict(key),
            "candidate_status": candidate.get("status"),
            "reference_status": reference.get("status"),
            "reference_backend_requested": reference.get("qwen_backend_requested"),
            "reference_effective_backend": reference.get("effective_backend"),
            "reference_backend_pass": reference_backend_pass,
            "candidate_prefill_tokps_total": candidate.get("prefill_tokps_total"),
            "reference_prefill_tokps_total": reference.get("prefill_tokps_total"),
            "prefill_speedup": prefill_speedup,
            "candidate_decode_tokps_total": candidate.get("decode_tokps_total"),
            "reference_decode_tokps_total": reference.get("decode_tokps_total"),
            "decode_speedup": decode_speedup,
            "candidate_model_footprint_mb": candidate.get("model_footprint_mb"),
            "reference_model_footprint_mb": reference.get("model_footprint_mb"),
            "model_footprint_ratio": footprint_ratio,
            "candidate_peak_vram_mb": candidate.get("peak_vram_mb"),
            "reference_peak_vram_mb": reference.get("peak_vram_mb"),
            "peak_vram_ratio": peak_vram_ratio,
            "candidate_runtime_working_set_mb": candidate.get("runtime_working_set_mb"),
            "reference_runtime_working_set_mb": reference.get("runtime_working_set_mb"),
            "runtime_working_set_ratio": working_set_ratio,
            "memory_pass": memory_pass,
            "candidate_active_parameter_count": candidate.get("active_parameter_count"),
            "reference_active_parameter_count": reference.get("active_parameter_count"),
            "active_parameter_ratio": active_parameter_ratio,
            "prefill_active_parameter_throughput_ratio": prefill_parameter_throughput_ratio,
            "decode_active_parameter_throughput_ratio": decode_parameter_throughput_ratio,
            "prefill_active_parameter_efficiency_ratio": prefill_parameter_efficiency_ratio,
            "decode_active_parameter_efficiency_ratio": decode_parameter_efficiency_ratio,
            "active_parameter_work_pass": active_parameter_work_pass,
            "active_parameter_efficiency_pass": active_parameter_efficiency_pass,
            "active_parameter_pass": active_parameter_pass,
            "candidate_prefill_backend": candidate_prefill_backend,
            "candidate_decode_backend": candidate_decode_backend,
            "reference_backend": reference_backend,
            "candidate_quantization_backend": candidate.get("quantization_backend"),
            "reference_quantization_backend": reference.get("quantization_backend"),
            "native_backend_pass": native_backend_pass,
            "qwen_backend_pass": qwen_backend_pass,
            "qwen_full_fused_pass": qwen_full_fused_pass,
            "quant_memory_ratio_vs_dense": quant_memory_ratio,
            "quant_peak_memory_ratio_vs_dense": quant_peak_memory_ratio,
            "quant_memory_pass": quant_memory_pass,
            "candidate_prefill_chunk_size": candidate_prefill_chunk_size,
            "reference_prefill_chunk_size": reference_prefill_chunk_size,
            "prefill_mode_pass": prefill_mode_pass,
            "quant_prefill_speedup_vs_dense": quant_prefill_speedup_vs_dense,
            "quant_decode_speedup_vs_dense": quant_decode_speedup_vs_dense,
            "quant_dense_prefill_mode_pass": quant_dense_prefill_mode_pass,
            "quant_dense_speed_pass": quant_dense_speed_pass,
            "passed": passed,
        }
        cells.append(cell)
        if not passed:
            red_cells.append(cell)

    expected = expected_cells if expected_cells > 0 else len(candidate_keys | reference_keys)
    coverage_complete = bool(
        len(joined_keys) == expected and not missing_candidate and not missing_reference
    )
    ratios_pass = not red_cells and bool(cells)
    speed_by_quantization = {}
    for family, metrics in sorted(family_metrics.items()):
        speed_by_quantization[family] = {
            "cells": len(metrics["prefill"]),
            "min_prefill_speedup": min(metrics["prefill"]) if metrics["prefill"] else None,
            "median_prefill_speedup": median_or_none(metrics["prefill"]),
            "min_decode_speedup": min(metrics["decode"]) if metrics["decode"] else None,
            "median_decode_speedup": median_or_none(metrics["decode"]),
            "min_prefill_speedup_vs_dense": (
                min(metrics["quant_prefill_vs_dense"])
                if metrics["quant_prefill_vs_dense"]
                else None
            ),
            "median_prefill_speedup_vs_dense": median_or_none(
                metrics["quant_prefill_vs_dense"]
            ),
            "min_decode_speedup_vs_dense": (
                min(metrics["quant_decode_vs_dense"])
                if metrics["quant_decode_vs_dense"]
                else None
            ),
            "median_decode_speedup_vs_dense": median_or_none(
                metrics["quant_decode_vs_dense"]
            ),
            "max_footprint_ratio_vs_dense": (
                max(metrics["footprint_vs_dense"])
                if metrics["footprint_vs_dense"]
                else None
            ),
            "max_peak_vram_ratio_vs_dense": (
                max(metrics["peak_vram_vs_dense"])
                if metrics["peak_vram_vs_dense"]
                else None
            ),
        }
    reference_backend_complete = bool(cells) and backend_matches == len(cells)
    cells_pass = not red_cells and bool(cells)
    overall_pass = coverage_complete and reference_backend_complete and cells_pass
    return {
        "axis": "qwen35_cross_model_speed_comparison",
        "coverage": {
            "expected_cells": expected,
            "joined_cells": len(joined_keys),
            "complete": coverage_complete,
        },
        "thresholds": {
            "min_prefill_speedup": min_prefill_speedup,
            "min_decode_speedup": min_decode_speedup,
            "min_quant_prefill_speedup": min_quant_prefill_speedup,
            "min_quant_decode_speedup": min_quant_decode_speedup,
            "require_native_candidate": require_native_candidate,
            "require_qwen_fast_path": require_qwen_fast_path,
            "require_qwen_full_fused": require_qwen_full_fused,
            "require_quant_memory_reduction": require_quant_memory_reduction,
            "require_prefill_mode_match": require_prefill_mode_match,
            "require_quant_not_slower_than_dense": require_quant_not_slower_than_dense,
            "required_reference_backend": required_reference_backend,
            "require_memory_not_larger": require_memory_not_larger,
            "min_active_parameter_throughput_ratio": min_active_parameter_throughput_ratio,
            "min_active_parameter_efficiency_ratio": min_active_parameter_efficiency_ratio,
        },
        "reference_backend": {
            "required": required_reference_backend,
            "matching_cells": backend_matches,
            "total_cells": len(cells),
            "complete": reference_backend_complete,
        },
        "speed": {
            "min_prefill_speedup": min(prefill_ratios) if prefill_ratios else None,
            "median_prefill_speedup": median_or_none(prefill_ratios),
            "max_prefill_speedup": max(prefill_ratios) if prefill_ratios else None,
            "prefill_at_least_equal_cells": sum(value >= 1.0 for value in prefill_ratios),
            "prefill_gate_cells": sum(value >= min_prefill_speedup for value in prefill_ratios),
            "min_decode_speedup": min(decode_ratios) if decode_ratios else None,
            "median_decode_speedup": median_or_none(decode_ratios),
            "max_decode_speedup": max(decode_ratios) if decode_ratios else None,
            "decode_at_least_equal_cells": sum(value >= 1.0 for value in decode_ratios),
            "decode_gate_cells": sum(value >= min_decode_speedup for value in decode_ratios),
            "strict_gate_cells": len(cells) - len(red_cells),
            "total_cells": len(cells),
        },
        "memory": {
            "min_model_footprint_ratio": min(footprint_ratios) if footprint_ratios else None,
            "median_model_footprint_ratio": median_or_none(footprint_ratios),
            "max_model_footprint_ratio": max(footprint_ratios) if footprint_ratios else None,
            "model_footprint_not_larger_cells": sum(value <= 1.0 for value in footprint_ratios),
            "min_peak_vram_ratio": min(peak_vram_ratios) if peak_vram_ratios else None,
            "median_peak_vram_ratio": median_or_none(peak_vram_ratios),
            "max_peak_vram_ratio": max(peak_vram_ratios) if peak_vram_ratios else None,
            "peak_vram_not_larger_cells": sum(value <= 1.0 for value in peak_vram_ratios),
            "min_runtime_working_set_ratio": min(working_set_ratios) if working_set_ratios else None,
            "median_runtime_working_set_ratio": median_or_none(working_set_ratios),
            "max_runtime_working_set_ratio": max(working_set_ratios) if working_set_ratios else None,
            "runtime_working_set_not_larger_cells": sum(value <= 1.0 for value in working_set_ratios),
            "total_cells": len(cells),
        },
        "active_parameter_work": {
            "gate_enabled": min_active_parameter_throughput_ratio is not None,
            "min_active_parameter_ratio": min(active_parameter_ratios) if active_parameter_ratios else None,
            "median_active_parameter_ratio": median_or_none(active_parameter_ratios),
            "max_active_parameter_ratio": max(active_parameter_ratios) if active_parameter_ratios else None,
            "min_prefill_throughput_ratio": (
                min(prefill_parameter_throughput_ratios)
                if prefill_parameter_throughput_ratios
                else None
            ),
            "median_prefill_throughput_ratio": median_or_none(prefill_parameter_throughput_ratios),
            "min_decode_throughput_ratio": (
                min(decode_parameter_throughput_ratios)
                if decode_parameter_throughput_ratios
                else None
            ),
            "median_decode_throughput_ratio": median_or_none(decode_parameter_throughput_ratios),
            "passing_cells": len(cells) - active_parameter_work_failures,
            "total_cells": len(cells),
        },
        "active_parameter_efficiency": {
            "gate_enabled": min_active_parameter_efficiency_ratio is not None,
            "min_prefill_ratio": (
                min(prefill_parameter_efficiency_ratios)
                if prefill_parameter_efficiency_ratios
                else None
            ),
            "median_prefill_ratio": median_or_none(prefill_parameter_efficiency_ratios),
            "max_prefill_ratio": (
                max(prefill_parameter_efficiency_ratios)
                if prefill_parameter_efficiency_ratios
                else None
            ),
            "min_decode_ratio": (
                min(decode_parameter_efficiency_ratios)
                if decode_parameter_efficiency_ratios
                else None
            ),
            "median_decode_ratio": median_or_none(decode_parameter_efficiency_ratios),
            "max_decode_ratio": (
                max(decode_parameter_efficiency_ratios)
                if decode_parameter_efficiency_ratios
                else None
            ),
            "passing_cells": len(cells) - active_parameter_efficiency_failures,
            "total_cells": len(cells),
        },
        "speed_by_quantization": speed_by_quantization,
        "missing": {
            "candidate": [key_dict(key) for key in missing_candidate],
            "reference": [key_dict(key) for key in missing_reference],
        },
        "red_cells": red_cells,
        "cells": cells,
        "gates": {
            "coverage_pass": coverage_complete,
            "speed_pass": ratios_pass,
            "backend_pass": backend_failures == 0,
            "quant_memory_pass": memory_failures == 0,
            "prefill_mode_pass": prefill_mode_failures == 0,
            "quant_dense_speed_pass": quant_dense_speed_failures == 0,
            "reference_backend_pass": reference_backend_complete,
            "memory_pass": all(cell["memory_pass"] for cell in cells) and bool(cells),
            "active_parameter_work_pass": active_parameter_work_failures == 0 and bool(cells),
            "active_parameter_efficiency_pass": (
                active_parameter_efficiency_failures == 0 and bool(cells)
            ),
            "active_parameter_pass": active_parameter_failures == 0 and bool(cells),
            "overall_pass": overall_pass,
        },
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(summary: dict[str, Any]) -> str:
    coverage = summary["coverage"]
    backend = summary["reference_backend"]
    speed = summary["speed"]
    memory = summary["memory"]
    active = summary["active_parameter_work"]
    efficiency = summary["active_parameter_efficiency"]
    active_work_cells = (
        f"{active['passing_cells']}/{active['total_cells']}"
        if active["gate_enabled"]
        else f"reported {active['total_cells']}/{active['total_cells']}"
    )
    full_fused_required = bool(summary["thresholds"].get("require_qwen_full_fused"))
    full_fused_cells = sum(cell.get("qwen_full_fused_pass") is True for cell in summary["cells"])
    overall = "PASS" if summary["gates"]["overall_pass"] else "FAIL"
    lines = [
        "# RWKV-7 vs Qwen3.5 HF speed matrix",
        "",
        f"Overall: {overall}",
        "",
        f"Coverage: `{coverage['joined_cells']}/{coverage['expected_cells']}` cells.",
        "",
        f"Required Qwen backend: `{backend['required']}`; verified: "
        f"`{backend['matching_cells']}/{backend['total_cells']}` cells.",
        "",
        f"Required Qwen full fusion: `{str(full_fused_required).lower()}`; verified: "
        f"`{full_fused_cells}/{backend['total_cells']}` cells.",
        "",
        "| Metric | Minimum | Median | Maximum | Passing cells |",
        "|---|---:|---:|---:|---:|",
        f"| Prefill RWKV/Qwen | {fmt(speed['min_prefill_speedup'])}x | {fmt(speed['median_prefill_speedup'])}x | "
        f"{fmt(speed['max_prefill_speedup'])}x | {speed['prefill_gate_cells']}/{speed['total_cells']} |",
        f"| Decode RWKV/Qwen | {fmt(speed['min_decode_speedup'])}x | {fmt(speed['median_decode_speedup'])}x | "
        f"{fmt(speed['max_decode_speedup'])}x | {speed['decode_gate_cells']}/{speed['total_cells']} |",
        f"| Model footprint RWKV/Qwen | {fmt(memory['min_model_footprint_ratio'])}x | "
        f"{fmt(memory['median_model_footprint_ratio'])}x | {fmt(memory['max_model_footprint_ratio'])}x | "
        f"{memory['model_footprint_not_larger_cells']}/{memory['total_cells']} |",
        f"| Peak VRAM RWKV/Qwen | {fmt(memory['min_peak_vram_ratio'])}x | "
        f"{fmt(memory['median_peak_vram_ratio'])}x | {fmt(memory['max_peak_vram_ratio'])}x | "
        f"{memory['peak_vram_not_larger_cells']}/{memory['total_cells']} |",
        f"| Runtime working set RWKV/Qwen | {fmt(memory['min_runtime_working_set_ratio'])}x | "
        f"{fmt(memory['median_runtime_working_set_ratio'])}x | {fmt(memory['max_runtime_working_set_ratio'])}x | "
        f"{memory['runtime_working_set_not_larger_cells']}/{memory['total_cells']} |",
        f"| Active parameters RWKV/Qwen | {fmt(active['min_active_parameter_ratio'])}x | "
        f"{fmt(active['median_active_parameter_ratio'])}x | {fmt(active['max_active_parameter_ratio'])}x | "
        f"{active['total_cells']}/{active['total_cells']} |",
        f"| Prefill tok/s per active-B | {fmt(efficiency['min_prefill_ratio'])}x | "
        f"{fmt(efficiency['median_prefill_ratio'])}x | {fmt(efficiency['max_prefill_ratio'])}x | "
        f"{efficiency['passing_cells']}/{efficiency['total_cells']} |",
        f"| Decode tok/s per active-B | {fmt(efficiency['min_decode_ratio'])}x | "
        f"{fmt(efficiency['median_decode_ratio'])}x | {fmt(efficiency['max_decode_ratio'])}x | "
        f"{efficiency['passing_cells']}/{efficiency['total_cells']} |",
        f"| Prefill active-param work rate | {fmt(active['min_prefill_throughput_ratio'])}x | "
        f"{fmt(active['median_prefill_throughput_ratio'])}x | - | "
        f"{active_work_cells} |",
        f"| Decode active-param work rate | {fmt(active['min_decode_throughput_ratio'])}x | "
        f"{fmt(active['median_decode_throughput_ratio'])}x | - | "
        f"{active_work_cells} |",
        "",
        f"Strict speed cells: `{speed['strict_gate_cells']}/{speed['total_cells']}`.",
        "",
        "## Precision families",
        "",
        "| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Footprint max | Peak max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for family, metrics in summary.get("speed_by_quantization", {}).items():
        lines.append(
            "| {family} | {cells} | {prefill_min}x / {prefill_median}x | "
            "{decode_min}x / {decode_median}x | {dense_prefill} | {dense_decode} | "
            "{footprint} | {peak} |".format(
                family=family,
                cells=metrics["cells"],
                prefill_min=fmt(metrics["min_prefill_speedup"]),
                prefill_median=fmt(metrics["median_prefill_speedup"]),
                decode_min=fmt(metrics["min_decode_speedup"]),
                decode_median=fmt(metrics["median_decode_speedup"]),
                dense_prefill=(
                    f"{fmt(metrics['min_prefill_speedup_vs_dense'])}x"
                    if metrics["min_prefill_speedup_vs_dense"] is not None
                    else "-"
                ),
                dense_decode=(
                    f"{fmt(metrics['min_decode_speedup_vs_dense'])}x"
                    if metrics["min_decode_speedup_vs_dense"] is not None
                    else "-"
                ),
                footprint=(
                    f"{fmt(metrics['max_footprint_ratio_vs_dense'])}x"
                    if metrics["max_footprint_ratio_vs_dense"] is not None
                    else "-"
                ),
                peak=(
                    f"{fmt(metrics['max_peak_vram_ratio_vs_dense'])}x"
                    if metrics["max_peak_vram_ratio_vs_dense"] is not None
                    else "-"
                ),
            )
        )
    lines.extend(
        [
        "",
        "## Red cells",
        "",
        ]
    )
    if not summary["red_cells"]:
        lines.append("None.")
    else:
        lines.extend(
            [
                "| Pair | Prompt | Decode | Bsz | Quant | Prefill | Decode | Candidate backend | Reference backend | Quant/dense memory | Quant/dense P/D | chunks C/R |",
                "|---|---:|---:|---:|---|---:|---:|---|---|---:|---:|---:|",
            ]
        )
        for cell in summary["red_cells"]:
            lines.append(
                "| {model_pair} | {prompt_tokens} | {decode_tokens} | {batch_size} | {quantization} | "
                "{prefill}x | {decode}x | {candidate_prefill_backend}/{candidate_decode_backend} | "
                    "{reference_backend} | {memory} | {quant_dense} | "
                    "{candidate_prefill_chunk_size}/{reference_prefill_chunk_size} |".format(
                    **cell,
                    prefill=fmt(cell["prefill_speedup"]),
                    decode=fmt(cell["decode_speedup"]),
                    memory=(
                        f"{fmt(cell['quant_memory_ratio_vs_dense'])}x"
                        if cell["quant_memory_ratio_vs_dense"] is not None
                        else "-"
                    ),
                    quant_dense=(
                        f"{fmt(cell['quant_prefill_speedup_vs_dense'])}x/"
                        f"{fmt(cell['quant_decode_speedup_vs_dense'])}x"
                        if cell["quant_prefill_speedup_vs_dense"] is not None
                        else "-"
                    ),
                )
            )
    lines.extend(
        [
            "",
            f"Missing candidate rows: `{len(summary['missing']['candidate'])}`.",
            f"Missing reference rows: `{len(summary['missing']['reference'])}`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--expected-cells", type=int, default=216)
    ap.add_argument("--min-prefill-speedup", type=float, default=1.05)
    ap.add_argument("--min-decode-speedup", type=float, default=1.05)
    ap.add_argument("--min-quant-prefill-speedup", type=float, default=None)
    ap.add_argument("--min-quant-decode-speedup", type=float, default=None)
    ap.add_argument("--require-native-candidate", action="store_true")
    ap.add_argument("--require-qwen-fast-path", action="store_true")
    ap.add_argument("--require-qwen-full-fused", action="store_true")
    ap.add_argument("--require-quant-memory-reduction", action="store_true")
    ap.add_argument("--require-prefill-mode-match", action="store_true")
    ap.add_argument("--require-quant-not-slower-than-dense", action="store_true")
    ap.add_argument("--required-reference-backend", choices=["fla", "torch", "any"], default="fla")
    ap.add_argument("--require-memory-not-larger", action="store_true")
    ap.add_argument("--min-active-parameter-throughput-ratio", type=float, default=None)
    ap.add_argument("--min-active-parameter-efficiency-ratio", type=float, default=None)
    ap.add_argument("--json-output", default="")
    ap.add_argument("--markdown-output", default="")
    ap.add_argument("--fail-on-gate", action="store_true")
    args = ap.parse_args()

    summary = compare(
        load_rows(Path(args.results)),
        expected_cells=args.expected_cells,
        min_prefill_speedup=args.min_prefill_speedup,
        min_decode_speedup=args.min_decode_speedup,
        min_quant_prefill_speedup=args.min_quant_prefill_speedup,
        min_quant_decode_speedup=args.min_quant_decode_speedup,
        require_native_candidate=args.require_native_candidate,
        require_qwen_fast_path=args.require_qwen_fast_path,
        require_qwen_full_fused=args.require_qwen_full_fused,
        require_quant_memory_reduction=args.require_quant_memory_reduction,
        require_prefill_mode_match=args.require_prefill_mode_match,
        require_quant_not_slower_than_dense=args.require_quant_not_slower_than_dense,
        required_reference_backend=args.required_reference_backend,
        require_memory_not_larger=args.require_memory_not_larger,
        min_active_parameter_throughput_ratio=args.min_active_parameter_throughput_ratio,
        min_active_parameter_efficiency_ratio=args.min_active_parameter_efficiency_ratio,
    )
    markdown = render_markdown(summary)
    if args.json_output:
        out = Path(args.json_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown_output:
        out = Path(args.markdown_output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown, encoding="utf-8")
    print(markdown)
    return 1 if args.fail_on_gate and not summary["gates"]["overall_pass"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
