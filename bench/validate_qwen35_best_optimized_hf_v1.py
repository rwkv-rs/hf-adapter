#!/usr/bin/env python3
"""Validate a complete card-local Qwen3.5 best-optimized HF reference matrix."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Iterable

MATRIX = "qwen35_best_optimized_hf_v1"
LANE = "qwen_best_optimized_hf"
BATCHES = (1, 8)
PROMPTS = (128, 512, 2048)
DECODES = (128, 512)
EXPECTED_SHAPES = set(product(BATCHES, PROMPTS, DECODES))
PAIRS = {
    "rwkv-0.4b__qwen3.5-0.8b": "0.8b",
    "rwkv-1.5b__qwen3.5-2b": "2b",
    "rwkv-2.9b__qwen3.5-4b": "4b",
    "rwkv-7.2b__qwen3.5-9b": "9b",
}
PAIR_RANK = {pair: rank for rank, pair in enumerate(PAIRS)}
RUNTIME_FIELDS = (
    "torch_version",
    "torch_cuda_version",
    "triton_version",
    "transformers_version",
    "fla_version",
    "causal_conv1d_version",
)
QWEN_CONTRACT = "official_fla_causal_conv1d_static_cache_cudagraph_same_cache_v2"
QWEN_COMPILE_MODES = {"reduce-overhead", "max-autotune"}
QWEN_GRAPH_ROUTES = {
    "static_cache_inductor_cudagraph",
    "static_cache_raw_cudagraph",
}
QWEN_DYNAMIC_ROUTE = "module_call_dynamic"


def read_rows(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_source"] = f"{path}:{line_number}"
            rows.append(row)
    return rows


def _require(row: dict[str, Any], field: str, expected: Any, errors: list[str]) -> None:
    actual = row.get(field)
    matches = (
        type(actual) is bool and actual is expected
        if isinstance(expected, bool)
        else actual == expected
    )
    if not matches:
        errors.append(
            f"{row.get('_source', '<row>')}: {field}={actual!r}, expected {expected!r}"
        )


def _require_positive(row: dict[str, Any], field: str, errors: list[str]) -> None:
    value = row.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        errors.append(f"{row.get('_source', '<row>')}: {field}={value!r}, expected > 0")


def _require_int(
    row: dict[str, Any], field: str, expected: int, errors: list[str]
) -> None:
    value = row.get(field)
    if type(value) is not int or value != expected:
        errors.append(
            f"{row.get('_source', '<row>')}: {field}={value!r}, expected integer {expected}"
        )


def _is_finite_real_number(value: Any) -> bool:
    """Return true only for finite int/float telemetry, never bool sentinels."""

    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
    )


def _validate_mismatch_telemetry(
    row: dict[str, Any], prefix: str, errors: list[str]
) -> None:
    count_field = f"{prefix}_mismatch_count"
    index_field = f"{prefix}_first_mismatch_index"
    count = row.get(count_field)
    index = row.get(index_field)
    if type(count) is not int or count < 0:
        errors.append(
            f"{row.get('_source', '<row>')}: {count_field} must be a non-negative int"
        )
        return
    if count == 0:
        if index is not None:
            errors.append(
                f"{row.get('_source', '<row>')}: {index_field} must be null when count is zero"
            )
        return
    probe_tokens = row.get("qwen_graph_probe_tokens")
    if (
        type(index) is not int
        or type(probe_tokens) is not int
        or not 0 <= index <= probe_tokens
    ):
        errors.append(
            f"{row.get('_source', '<row>')}: {index_field}={index!r} is outside the token trace"
        )


def _validate_samples(
    row: dict[str, Any], sample_field: str, median_field: str, errors: list[str]
) -> None:
    samples = row.get(sample_field)
    if not isinstance(samples, list) or len(samples) != 7:
        errors.append(
            f"{row.get('_source', '<row>')}: {sample_field} must contain 7 raw samples"
        )
        return
    if not all(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value > 0
        for value in samples
    ):
        errors.append(
            f"{row.get('_source', '<row>')}: {sample_field} contains a non-positive sample"
        )
        return
    recorded = row.get(median_field)
    if (
        not isinstance(recorded, (int, float))
        or abs(statistics.median(samples) - recorded) > 1e-6
    ):
        errors.append(
            f"{row.get('_source', '<row>')}: {median_field} does not match raw-sample median"
        )
    raw_median_field = f"{median_field}_raw"
    raw_recorded = row.get(raw_median_field)
    if not isinstance(raw_recorded, (int, float)) or raw_recorded != statistics.median(
        samples
    ):
        errors.append(
            f"{row.get('_source', '<row>')}: {raw_median_field} does not exactly match "
            "the raw-sample median"
        )


def _validate_row(
    row: dict[str, Any],
    expected_device: str,
    errors: list[str],
    *,
    expected_route: str | None = None,
    expected_matrix: str = MATRIX,
    expected_lane: str = LANE,
    expected_conv_backend: str = "causal_conv1d",
    expected_causal_conv1d_importable: bool = True,
    expected_fast_path_available: bool = True,
    nullable_runtime_fields: frozenset[str] = frozenset(),
    expected_cross_cache_full_greedy_policy: str = "strict",
) -> None:
    for field, expected in (
        ("axis", "qwen35_cross_model_speed"),
        ("benchmark_matrix", expected_matrix),
        ("optimization_lane", expected_lane),
        ("model_role", "reference"),
        ("model_kind", "qwen35"),
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
        ("qwen_backend_requested", "fla"),
        ("qwen_conv_backend_requested", expected_conv_backend),
        ("qwen_fast_path_required", True),
        ("qwen_fast_path_available", expected_fast_path_available),
        ("qwen_fast_path_verified", True),
        ("qwen_full_fused_contract_pass", True),
        ("qwen_causal_conv1d_importable", expected_causal_conv1d_importable),
        ("qwen_conv_backend_effective", expected_conv_backend),
        ("qwen_force_torch", False),
        ("logits_finite", True),
    ):
        _require(row, field, expected, errors)
    repository_commit = row.get("benchmark_repository_commit")
    if not isinstance(repository_commit, str) or not repository_commit.strip():
        errors.append(
            f"{row.get('_source', '<row>')}: benchmark_repository_commit must be non-empty"
        )
    requested_route = row.get("qwen_decode_optimization_requested")
    effective_route = row.get("qwen_decode_optimization_effective")
    allowed_routes = (
        {expected_route} if expected_route is not None else QWEN_GRAPH_ROUTES
    )
    if requested_route not in allowed_routes:
        errors.append(
            f"{row.get('_source', '<row>')}: qwen_decode_optimization_requested="
            f"{requested_route!r}, expected one of {sorted(allowed_routes)!r}"
        )
    if requested_route != effective_route:
        errors.append(
            f"{row.get('_source', '<row>')}: requested/effective decode route mismatch: "
            f"{requested_route!r} != {effective_route!r}"
        )
    if effective_route in QWEN_GRAPH_ROUTES:
        cross_cache_full_greedy_required = (
            expected_cross_cache_full_greedy_policy == "strict"
        )
        for field, expected in (
            ("prefill_backend_effective", "module_call_dynamic_cache"),
            ("prefill_cache_type", "DynamicCache"),
            ("cache_type", "StaticCache"),
            ("qwen_cuda_graph_requested", True),
            ("qwen_cuda_graph_effective", True),
            ("qwen_decode_cuda_graph_verified", True),
            ("qwen_cache_pointer_stable", True),
            ("qwen_graph_parity_verified", True),
            (
                "qwen_cross_cache_full_greedy_policy_requested",
                expected_cross_cache_full_greedy_policy,
            ),
            (
                "qwen_cross_cache_full_greedy_policy_effective",
                expected_cross_cache_full_greedy_policy,
            ),
            (
                "qwen_cross_cache_full_greedy_required",
                cross_cache_full_greedy_required,
            ),
            ("qwen_graph_prefill_next_token_match", True),
            ("qwen_axis_composition", "independent_best_prefill_and_decode"),
            ("qwen_same_cache_greedy_match", True),
            ("qwen_graph_logits_greedy_match", True),
            ("qwen_dynamic_static_logits_greedy_match", True),
            ("qwen_same_cache_logits_greedy_match", True),
            ("qwen_graph_logits_trace_finite", True),
            ("qwen_dynamic_static_logits_finite", True),
            ("qwen_same_cache_logits_finite", True),
            ("qwen_static_compiled_logits_finite", True),
        ):
            _require(row, field, expected, errors)
        _require_int(row, "qwen_same_cache_full_greedy_mismatch_count", 0, errors)
        if cross_cache_full_greedy_required:
            for field, expected in (
                ("qwen_graph_greedy_match", True),
                ("qwen_static_cache_eager_greedy_match", True),
            ):
                _require(row, field, expected, errors)
            for field in (
                "qwen_dynamic_static_full_greedy_mismatch_count",
                "qwen_dynamic_candidate_full_greedy_mismatch_count",
            ):
                _require_int(row, field, 0, errors)
        else:
            for field in (
                "qwen_graph_greedy_match",
                "qwen_static_cache_eager_greedy_match",
            ):
                if type(row.get(field)) is not bool:
                    errors.append(
                        f"{row.get('_source', '<row>')}: {field} must be bool telemetry"
                    )
            for field in (
                "qwen_dynamic_static_full_greedy_mismatch_count",
                "qwen_dynamic_candidate_full_greedy_mismatch_count",
            ):
                value = row.get(field)
                if type(value) is not int or value < 0:
                    errors.append(
                        f"{row.get('_source', '<row>')}: {field} must be a non-negative int"
                    )
        for prefix in (
            "qwen_dynamic_static_full_greedy",
            "qwen_dynamic_candidate_full_greedy",
            "qwen_same_cache_full_greedy",
        ):
            _validate_mismatch_telemetry(row, prefix, errors)
        _require(row, "step_backend", f"qwen_{effective_route}", errors)
    if effective_route == "static_cache_inductor_cudagraph":
        _require(row, "qwen_compile_backend_effective", "inductor", errors)
        _require(row, "qwen_compile_fullgraph_effective", False, errors)
        _require(row, "qwen_compile_dynamic_effective", False, errors)
        _require(row, "qwen_graph_scope", "single_token_hf_qwen_forward", errors)
        _require(row, "qwen_graph_break_count", 0, errors)
        _require(row, "qwen_cudagraph_skip_count", 0, errors)
        requested_compile_mode = row.get("qwen_compile_mode_requested")
        compile_mode = row.get("qwen_compile_mode_effective")
        if requested_compile_mode not in QWEN_COMPILE_MODES:
            errors.append(
                f"{row.get('_source', '<row>')}: qwen_compile_mode_requested="
                f"{requested_compile_mode!r}, expected one of {sorted(QWEN_COMPILE_MODES)!r}"
            )
        if compile_mode not in QWEN_COMPILE_MODES:
            errors.append(
                f"{row.get('_source', '<row>')}: qwen_compile_mode_effective="
                f"{compile_mode!r}, expected one of {sorted(QWEN_COMPILE_MODES)!r}"
            )
        if requested_compile_mode != compile_mode:
            errors.append(
                f"{row.get('_source', '<row>')}: requested/effective compile mode mismatch: "
                f"{requested_compile_mode!r} != {compile_mode!r}"
            )
        _require_positive(row, "qwen_cudagraph_recorded_non_static_inputs", errors)
        _require_positive(row, "qwen_cuda_graph_launch_count", errors)
    elif effective_route == "static_cache_raw_cudagraph":
        _require(
            row,
            "qwen_graph_scope",
            "single_token_hf_qwen_forward_argmax_token_copy",
            errors,
        )
        for field in (
            "qwen_compile_mode_requested",
            "qwen_compile_backend_effective",
            "qwen_compile_mode_effective",
            "qwen_compile_fullgraph_effective",
            "qwen_compile_dynamic_effective",
            "qwen_graph_break_count",
            "qwen_cudagraph_skip_count",
            "qwen_cudagraph_recorded_non_static_inputs",
        ):
            _require(row, field, None, errors)
        launches = row.get("qwen_cuda_graph_launch_count")
        if not _is_finite_real_number(launches) or launches != 1:
            errors.append(
                f"{row.get('_source', '<row>')}: qwen_cuda_graph_launch_count="
                f"{launches!r}, expected exactly 1"
            )
    elif effective_route == QWEN_DYNAMIC_ROUTE:
        for field, expected in (
            ("step_backend", "module_call"),
            ("prefill_backend_effective", "module_call"),
            ("prefill_cache_type", "DynamicCache"),
            ("cache_type", "DynamicCache"),
            ("qwen_axis_composition", "continuous_single_cache_path"),
            ("qwen_cuda_graph_requested", False),
            ("qwen_cuda_graph_effective", False),
            ("qwen_decode_cuda_graph_verified", False),
        ):
            _require(row, field, expected, errors)
        for field in (
            "qwen_compile_mode_requested",
            "qwen_compile_backend_effective",
            "qwen_compile_mode_effective",
            "qwen_compile_fullgraph_effective",
            "qwen_compile_dynamic_effective",
            "qwen_graph_break_count",
            "qwen_cudagraph_skip_count",
            "qwen_cudagraph_recorded_non_static_inputs",
            "qwen_cuda_graph_launch_count",
            "qwen_cache_pointer_stable",
            "qwen_cache_tensor_pointer_count",
            "qwen_graph_parity_verified",
            "qwen_cross_cache_full_greedy_policy_effective",
            "qwen_cross_cache_full_greedy_required",
            "qwen_graph_prefill_next_token_match",
            "qwen_graph_greedy_match",
            "qwen_same_cache_greedy_match",
            "qwen_dynamic_static_full_greedy_mismatch_count",
            "qwen_dynamic_static_full_greedy_first_mismatch_index",
            "qwen_dynamic_candidate_full_greedy_mismatch_count",
            "qwen_dynamic_candidate_full_greedy_first_mismatch_index",
            "qwen_same_cache_full_greedy_mismatch_count",
            "qwen_same_cache_full_greedy_first_mismatch_index",
            "qwen_static_cache_eager_greedy_match",
            "qwen_graph_logits_greedy_match",
            "qwen_dynamic_static_logits_greedy_match",
            "qwen_same_cache_logits_greedy_match",
            "qwen_graph_logits_trace_finite",
            "qwen_dynamic_static_logits_finite",
            "qwen_same_cache_logits_finite",
            "qwen_static_compiled_logits_finite",
            "qwen_graph_logits_min_cosine",
            "qwen_graph_logits_max_abs_diff",
            "qwen_dynamic_static_logits_min_cosine",
            "qwen_dynamic_static_logits_max_abs_diff",
            "qwen_same_cache_logits_min_cosine",
            "qwen_same_cache_logits_max_abs_diff",
            "qwen_static_compiled_logits_min_cosine",
            "qwen_static_compiled_logits_max_abs_diff",
            "qwen_graph_probe_tokens",
            "qwen_graph_logits_probe_tokens",
            "qwen_graph_distinct_batch_prompts",
            "qwen_graph_scope",
            "qwen_graph_capture_s",
            "qwen_graph_setup_s",
            "qwen_graph_max_cache_len",
        ):
            _require(row, field, None, errors)
    if expected_device:
        _require(row, "device", expected_device, errors)

    pair = str(row.get("model_pair", ""))
    if pair not in PAIRS:
        errors.append(f"{row.get('_source', '<row>')}: unexpected model_pair={pair!r}")
    else:
        _require(row, "model_size_label", PAIRS[pair], errors)
    shape = (row.get("batch_size"), row.get("prompt_tokens"), row.get("decode_tokens"))
    if shape not in EXPECTED_SHAPES:
        errors.append(f"{row.get('_source', '<row>')}: unexpected B/P/D cell {shape!r}")
        return

    batch, prompt, decode = (int(value) for value in shape)
    if effective_route in QWEN_GRAPH_ROUTES:
        _require(row, "qwen_graph_max_cache_len", prompt + 3 + decode, errors)
        _require(row, "qwen_graph_probe_tokens", 3 + decode, errors)
        _require(row, "qwen_graph_logits_probe_tokens", 16, errors)
        _require(row, "qwen_graph_distinct_batch_prompts", batch > 1, errors)
        for field in (
            "qwen_graph_logits_min_cosine",
            "qwen_dynamic_static_logits_min_cosine",
        ):
            minimum_cosine = row.get(field)
            if not _is_finite_real_number(minimum_cosine):
                errors.append(
                    f"{row.get('_source', '<row>')}: {field}="
                    f"{minimum_cosine!r}, expected finite cross-cache telemetry"
                )
        same_cache_cosine = row.get("qwen_same_cache_logits_min_cosine")
        if not _is_finite_real_number(same_cache_cosine) or same_cache_cosine < 0.9999:
            errors.append(
                f"{row.get('_source', '<row>')}: qwen_same_cache_logits_min_cosine="
                f"{same_cache_cosine!r}, expected >=0.9999"
            )
    if effective_route in QWEN_GRAPH_ROUTES:
        _require_positive(row, "qwen_cache_tensor_pointer_count", errors)
    for field in (
        "prefill_tokps_total",
        "decode_tokps_total",
        "prefill_tokps_total_raw",
        "decode_tokps_total_raw",
        "prefill_sec_median_raw",
        "decode_sec_median_raw",
    ):
        _require_positive(row, field, errors)
    _validate_samples(row, "prefill_sec_samples", "prefill_sec_median", errors)
    _validate_samples(row, "decode_sec_samples", "decode_sec_median", errors)
    for field in RUNTIME_FIELDS:
        value = row.get(field)
        if field in nullable_runtime_fields and value is None:
            continue
        if type(value) is not str or not value.strip():
            errors.append(f"{row.get('_source', '<row>')}: {field} must be non-empty")
    expected_prefill = (
        batch * prompt / float(row.get("prefill_sec_median_raw", math.nan))
    )
    expected_decode = batch * decode / float(row.get("decode_sec_median_raw", math.nan))
    for field, actual, expected in (
        (
            "prefill_tokps_total_raw",
            row.get("prefill_tokps_total_raw"),
            expected_prefill,
        ),
        ("decode_tokps_total_raw", row.get("decode_tokps_total_raw"), expected_decode),
    ):
        if (
            not isinstance(actual, (int, float))
            or not math.isfinite(actual)
            or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
        ):
            errors.append(
                f"{row.get('_source', '<row>')}: {field}={actual!r}, "
                f"expected tokens/raw_median={expected!r}"
            )


def validate_matrix(
    rows: list[dict[str, Any]],
    *,
    expected_device: str = "",
    expected_pairs: Iterable[str] | None = None,
    expected_routes_by_pair: dict[str, str] | None = None,
    expected_matrix: str = MATRIX,
    expected_lane: str = LANE,
    expected_conv_backend: str = "causal_conv1d",
    expected_causal_conv1d_importable: bool = True,
    expected_fast_path_available: bool = True,
    nullable_runtime_fields: Iterable[str] = (),
    qwen_contract: str = QWEN_CONTRACT,
    expected_cross_cache_full_greedy_policy: str = "strict",
) -> dict[str, Any]:
    errors: list[str] = []
    selected_pairs = (
        tuple(expected_pairs) if expected_pairs is not None else tuple(PAIRS)
    )
    if not selected_pairs:
        errors.append("expected model pairs must not be empty")
    unknown_pairs = sorted(set(selected_pairs) - set(PAIRS))
    if unknown_pairs:
        errors.append(f"unknown expected model pairs: {unknown_pairs}")
    if len(set(selected_pairs)) != len(selected_pairs):
        errors.append("expected model pairs contain duplicates")
    if expected_cross_cache_full_greedy_policy not in {"strict", "informational"}:
        errors.append(
            "expected_cross_cache_full_greedy_policy must be strict or informational"
        )
    expected_rows = len(selected_pairs) * len(EXPECTED_SHAPES)
    if len(rows) != expected_rows:
        errors.append(f"reference row count={len(rows)}, expected {expected_rows}")
    if expected_routes_by_pair is not None and set(expected_routes_by_pair) != set(
        selected_pairs
    ):
        errors.append("expected route map must exactly cover the selected model pairs")
    for row in rows:
        pair = str(row.get("model_pair", ""))
        expected_route = (
            expected_routes_by_pair.get(pair)
            if expected_routes_by_pair is not None
            else None
        )
        _validate_row(
            row,
            expected_device,
            errors,
            expected_route=expected_route,
            expected_matrix=expected_matrix,
            expected_lane=expected_lane,
            expected_conv_backend=expected_conv_backend,
            expected_causal_conv1d_importable=expected_causal_conv1d_importable,
            expected_fast_path_available=expected_fast_path_available,
            nullable_runtime_fields=frozenset(nullable_runtime_fields),
            expected_cross_cache_full_greedy_policy=expected_cross_cache_full_greedy_policy,
        )
    repository_commits = sorted(
        {
            str(row.get("benchmark_repository_commit"))
            for row in rows
            if row.get("benchmark_repository_commit")
        }
    )
    if len(repository_commits) != 1:
        errors.append(
            "rows did not record exactly one benchmark repository commit: "
            + json.dumps(repository_commits)
        )

    expected_keys = {
        (pair, *shape) for pair in selected_pairs for shape in EXPECTED_SHAPES
    }
    keys = [
        (
            row.get("model_pair"),
            row.get("batch_size"),
            row.get("prompt_tokens"),
            row.get("decode_tokens"),
        )
        for row in rows
    ]
    counts = Counter(keys)
    duplicates = sorted(key for key, count in counts.items() if count > 1)
    missing = sorted(expected_keys - set(keys))
    extras = sorted(set(keys) - expected_keys)
    if duplicates:
        errors.append(f"duplicate cells: {duplicates}")
    if missing:
        errors.append(f"missing cells: {missing}")
    if extras:
        errors.append(f"extra cells: {extras}")

    runtime_signatures = {
        tuple(row.get(field) for field in RUNTIME_FIELDS)
        for row in rows
        if row.get("status") == "pass"
    }
    if len(runtime_signatures) != 1:
        errors.append(
            "rows were not produced by one locked runtime signature: "
            + json.dumps(sorted(runtime_signatures, key=repr), ensure_ascii=False)
        )
    compile_modes_by_model: dict[str, list[str] | None] = {}
    decode_routes_by_model: dict[str, list[str]] = {}
    for pair in selected_pairs:
        if pair not in PAIRS:
            continue
        routes = sorted(
            {
                str(row.get("qwen_decode_optimization_effective"))
                for row in rows
                if row.get("model_pair") == pair
            }
        )
        decode_routes_by_model[PAIRS[pair]] = routes
        if len(routes) != 1:
            errors.append(
                f"model pair {pair} mixed decode routes across its 12 cells: {routes}"
            )
        if routes == ["static_cache_inductor_cudagraph"]:
            modes = sorted(
                {
                    str(row.get("qwen_compile_mode_effective"))
                    for row in rows
                    if row.get("model_pair") == pair
                }
            )
            compile_modes_by_model[PAIRS[pair]] = modes
            if len(modes) != 1:
                errors.append(
                    f"model pair {pair} mixed compile modes across its 12 cells: {modes}"
                )
        else:
            compile_modes_by_model[PAIRS[pair]] = None
    devices = sorted({str(row.get("device")) for row in rows})
    return {
        "schema_version": 2,
        "benchmark_matrix": expected_matrix,
        "optimization_lane": expected_lane,
        "status": "pass" if not errors else "fail",
        "reference_rows": len(rows),
        "expected_rows": expected_rows,
        "expected_model_pairs": list(selected_pairs),
        "devices": devices,
        "runtime_fields": list(RUNTIME_FIELDS),
        "runtime_signature_count": len(runtime_signatures),
        "benchmark_repository_commits": repository_commits,
        "compile_modes_by_model": compile_modes_by_model,
        "decode_routes_by_model": decode_routes_by_model,
        "qwen_contract": qwen_contract,
        "cross_cache_full_greedy_policy": expected_cross_cache_full_greedy_policy,
        "reference_lane_eligible": not errors,
        "unified_main_table_eligible": False,
        "unified_main_table_reason": (
            "RWKV candidate rows were not rerun under this runtime, and the Qwen "
            "numbers are an independent best-Prefill/best-Decode axis envelope"
        ),
        "errors": errors,
    }


def _clean_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "_source"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-results", type=Path, nargs="+", required=True)
    parser.add_argument("--expected-device", default="")
    parser.add_argument(
        "--cross-cache-full-greedy-policy",
        choices=["strict", "informational"],
        default="strict",
    )
    parser.add_argument("--qwen-contract", default=QWEN_CONTRACT)
    parser.add_argument(
        "--expected-model-pair",
        action="append",
        choices=tuple(PAIRS),
        dest="expected_model_pairs",
        help=(
            "Required model pair; repeat to validate a card-local subset. "
            "The default remains all four reference pairs."
        ),
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--reference-table", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_rows(args.reference_results)
    summary = validate_matrix(
        rows,
        expected_device=args.expected_device,
        expected_pairs=args.expected_model_pairs,
        expected_cross_cache_full_greedy_policy=args.cross_cache_full_greedy_policy,
        qwen_contract=args.qwen_contract,
    )
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if summary["status"] != "pass":
        print("QWEN35_BEST_OPTIMIZED_HF_V1 " + json.dumps(summary, ensure_ascii=False))
        return 1
    ordered = sorted(
        (_clean_row(row) for row in rows),
        key=lambda row: (
            PAIR_RANK[row["model_pair"]],
            str(row["device"]),
            int(row["batch_size"]),
            int(row["prompt_tokens"]),
            int(row["decode_tokens"]),
        ),
    )
    args.reference_table.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered),
        encoding="utf-8",
    )
    print("QWEN35_BEST_OPTIMIZED_HF_V1 " + json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
