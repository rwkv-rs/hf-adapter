from __future__ import annotations

from itertools import product

import pytest

from bench.validate_qwen35_best_optimized_hf_v1 import PAIRS, validate_matrix
from bench.summarize_qwen35_best_optimized_hf_v1 import (
    build_summary,
    display_rate,
    ordered_rows,
    render_markdown,
)


def row(pair: str, batch: int, prompt: int, decode: int) -> dict:
    return {
        "axis": "qwen35_cross_model_speed",
        "benchmark_matrix": "qwen35_best_optimized_hf_v1",
        "benchmark_repository_commit": "test-commit",
        "optimization_lane": "qwen_best_optimized_hf",
        "model_pair": pair,
        "model_size_label": PAIRS[pair],
        "model_role": "reference",
        "model_kind": "qwen35",
        "dtype": "fp16",
        "quantization": "none",
        "batch_size": batch,
        "prompt_tokens": prompt,
        "decode_tokens": decode,
        "prefill_chunk_size": 512,
        "warmup": 3,
        "runs": 7,
        "timing_statistic": "median",
        "mtp_enabled": False,
        "speculative_decoding_enabled": False,
        "resident_sweep": True,
        "status": "pass",
        "device": "NVIDIA GeForce RTX 5090",
        "torch_version": "2.8.0+cu128",
        "torch_cuda_version": "12.8",
        "triton_version": "3.4.0",
        "transformers_version": "5.12.1",
        "fla_version": "0.5.1",
        "causal_conv1d_version": "1.6.2.post1",
        "qwen_backend_requested": "fla",
        "qwen_conv_backend_requested": "causal_conv1d",
        "qwen_fast_path_required": True,
        "qwen_fast_path_available": True,
        "qwen_fast_path_verified": True,
        "qwen_full_fused_contract_pass": True,
        "qwen_causal_conv1d_importable": True,
        "qwen_conv_backend_effective": "causal_conv1d",
        "qwen_force_torch": False,
        "qwen_decode_optimization_requested": "static_cache_inductor_cudagraph",
        "qwen_compile_mode_requested": "max-autotune",
        "qwen_decode_optimization_effective": "static_cache_inductor_cudagraph",
        "step_backend": "qwen_static_cache_inductor_cudagraph",
        "prefill_backend_effective": "module_call_dynamic_cache",
        "prefill_cache_type": "DynamicCache",
        "cache_type": "StaticCache",
        "qwen_compile_backend_effective": "inductor",
        "qwen_compile_mode_effective": "max-autotune",
        "qwen_compile_fullgraph_effective": False,
        "qwen_compile_dynamic_effective": False,
        "qwen_graph_scope": "single_token_hf_qwen_forward",
        "qwen_cuda_graph_requested": True,
        "qwen_cuda_graph_effective": True,
        "qwen_decode_cuda_graph_verified": True,
        "qwen_graph_break_count": 0,
        "qwen_cudagraph_skip_count": 0,
        "qwen_cudagraph_recorded_non_static_inputs": 1,
        "qwen_cuda_graph_launch_count": 1,
        "qwen_cache_pointer_stable": True,
        "qwen_cache_tensor_pointer_count": 54,
        "qwen_graph_parity_verified": True,
        "qwen_cross_cache_full_greedy_policy_requested": "strict",
        "qwen_cross_cache_full_greedy_policy_effective": "strict",
        "qwen_cross_cache_full_greedy_required": True,
        "qwen_graph_prefill_next_token_match": True,
        "qwen_axis_composition": "independent_best_prefill_and_decode",
        "qwen_graph_greedy_match": True,
        "qwen_same_cache_greedy_match": True,
        "qwen_dynamic_static_full_greedy_mismatch_count": 0,
        "qwen_dynamic_static_full_greedy_first_mismatch_index": None,
        "qwen_dynamic_candidate_full_greedy_mismatch_count": 0,
        "qwen_dynamic_candidate_full_greedy_first_mismatch_index": None,
        "qwen_same_cache_full_greedy_mismatch_count": 0,
        "qwen_same_cache_full_greedy_first_mismatch_index": None,
        "qwen_static_cache_eager_greedy_match": True,
        "qwen_graph_logits_greedy_match": True,
        "qwen_dynamic_static_logits_greedy_match": True,
        "qwen_same_cache_logits_greedy_match": True,
        "qwen_graph_logits_trace_finite": True,
        "qwen_dynamic_static_logits_finite": True,
        "qwen_same_cache_logits_finite": True,
        "qwen_static_compiled_logits_finite": True,
        "qwen_graph_logits_min_cosine": 0.99999,
        "qwen_dynamic_static_logits_min_cosine": 0.99999,
        "qwen_static_compiled_logits_min_cosine": 0.99999,
        "qwen_same_cache_logits_min_cosine": 0.99999,
        "qwen_graph_max_cache_len": prompt + 3 + decode,
        "qwen_graph_probe_tokens": 3 + decode,
        "qwen_graph_logits_probe_tokens": 16,
        "qwen_graph_distinct_batch_prompts": batch > 1,
        "prefill_sec_samples": [0.1] * 7,
        "prefill_sec_median": 0.1,
        "prefill_sec_median_raw": 0.1,
        "decode_sec_samples": [0.2] * 7,
        "decode_sec_median": 0.2,
        "decode_sec_median_raw": 0.2,
        "prefill_tokps_total": round(batch * prompt / 0.1, 3),
        "prefill_tokps_total_raw": batch * prompt / 0.1,
        "decode_tokps_total": round(batch * decode / 0.2, 3),
        "decode_tokps_total_raw": batch * decode / 0.2,
        "logits_finite": True,
    }


def complete_rows() -> list[dict]:
    return [
        row(pair, batch, prompt, decode)
        for pair, (batch, prompt, decode) in product(
            PAIRS, product((1, 8), (128, 512, 2048), (128, 512))
        )
    ]


def use_raw_graph(item: dict) -> None:
    item.update(
        {
            "qwen_decode_optimization_requested": "static_cache_raw_cudagraph",
            "qwen_decode_optimization_effective": "static_cache_raw_cudagraph",
            "step_backend": "qwen_static_cache_raw_cudagraph",
            "qwen_compile_mode_requested": None,
            "qwen_compile_backend_effective": None,
            "qwen_compile_mode_effective": None,
            "qwen_compile_fullgraph_effective": None,
            "qwen_compile_dynamic_effective": None,
            "qwen_graph_scope": "single_token_hf_qwen_forward_argmax_token_copy",
            "qwen_graph_break_count": None,
            "qwen_cudagraph_skip_count": None,
            "qwen_cudagraph_recorded_non_static_inputs": None,
        }
    )


def test_complete_best_optimized_qwen_matrix_passes() -> None:
    summary = validate_matrix(
        complete_rows(), expected_device="NVIDIA GeForce RTX 5090"
    )
    assert summary["status"] == "pass"
    assert summary["reference_rows"] == 48
    assert summary["reference_lane_eligible"] is True
    assert summary["unified_main_table_eligible"] is False


def test_card_local_three_pair_reference_subset_passes_fail_closed() -> None:
    expected_pairs = tuple(list(PAIRS)[:3])
    rows = [item for item in complete_rows() if item["model_pair"] in expected_pairs]
    for item in rows:
        item["device"] = "NVIDIA GeForce RTX 4080"
    summary = validate_matrix(
        rows,
        expected_device="NVIDIA GeForce RTX 4080",
        expected_pairs=expected_pairs,
    )
    assert summary["status"] == "pass"
    assert summary["reference_rows"] == 36
    assert summary["expected_rows"] == 36
    assert summary["expected_model_pairs"] == list(expected_pairs)
    assert set(summary["decode_routes_by_model"]) == {
        "0.8b",
        "2b",
        "4b",
    }

    summary = validate_matrix(
        rows[:-1],
        expected_device="NVIDIA GeForce RTX 4080",
        expected_pairs=expected_pairs,
    )
    assert summary["status"] == "fail"
    assert any("missing cells" in error for error in summary["errors"])


def test_reference_subset_rejects_unrequested_fourth_pair() -> None:
    expected_pairs = tuple(list(PAIRS)[:3])
    rows = [item for item in complete_rows() if item["model_pair"] in expected_pairs]
    rows[0] = next(
        item for item in complete_rows() if item["model_pair"] not in expected_pairs
    )
    summary = validate_matrix(rows, expected_pairs=expected_pairs)
    assert summary["status"] == "fail"
    assert any("missing cells" in error for error in summary["errors"])
    assert any("extra cells" in error for error in summary["errors"])


def test_reference_subset_rejects_an_empty_pair_contract() -> None:
    summary = validate_matrix([], expected_pairs=())
    assert summary["status"] == "fail"
    assert any("must not be empty" in error for error in summary["errors"])


def test_reduce_overhead_rows_are_valid_when_all_graph_gates_pass() -> None:
    rows = complete_rows()
    for item in rows:
        if item["model_size_label"] == "0.8b":
            item["qwen_compile_mode_requested"] = "reduce-overhead"
            item["qwen_compile_mode_effective"] = "reduce-overhead"
    summary = validate_matrix(rows, expected_device="NVIDIA GeForce RTX 5090")
    assert summary["status"] == "pass"


def test_models_may_choose_one_consistent_graph_route_each() -> None:
    rows = complete_rows()
    for item in rows:
        if item["model_size_label"] in {"4b", "9b"}:
            use_raw_graph(item)
    summary = validate_matrix(rows, expected_device="NVIDIA GeForce RTX 5090")
    assert summary["status"] == "pass"
    assert summary["decode_routes_by_model"]["4b"] == ["static_cache_raw_cudagraph"]
    assert summary["compile_modes_by_model"]["4b"] is None

    use_raw_graph(rows[0])
    summary = validate_matrix(rows)
    assert summary["status"] == "fail"
    assert any("mixed decode routes" in error for error in summary["errors"])


def test_cross_cache_cosine_is_informational_but_same_cache_is_strict() -> None:
    rows = complete_rows()
    rows[0]["qwen_graph_logits_min_cosine"] = 0.99
    rows[0]["qwen_dynamic_static_logits_min_cosine"] = 0.99
    assert validate_matrix(rows)["status"] == "pass"

    rows[0]["qwen_same_cache_logits_min_cosine"] = 0.9998
    summary = validate_matrix(rows)
    assert summary["status"] == "fail"
    assert any(
        "qwen_same_cache_logits_min_cosine" in error for error in summary["errors"]
    )


def test_cross_cache_full_greedy_can_be_explicitly_informational() -> None:
    rows = complete_rows()
    for item in rows:
        item["qwen_cross_cache_full_greedy_policy_requested"] = "informational"
        item["qwen_cross_cache_full_greedy_policy_effective"] = "informational"
        item["qwen_cross_cache_full_greedy_required"] = False
    target = rows[0]
    target["qwen_graph_greedy_match"] = False
    target["qwen_static_cache_eager_greedy_match"] = False
    target["qwen_dynamic_static_full_greedy_mismatch_count"] = 1
    target["qwen_dynamic_static_full_greedy_first_mismatch_index"] = 100
    target["qwen_dynamic_candidate_full_greedy_mismatch_count"] = 1
    target["qwen_dynamic_candidate_full_greedy_first_mismatch_index"] = 100
    summary = validate_matrix(
        rows, expected_cross_cache_full_greedy_policy="informational"
    )
    assert summary["status"] == "pass"
    assert summary["cross_cache_full_greedy_policy"] == "informational"

    assert validate_matrix(rows)["status"] == "fail"


def test_same_cache_full_greedy_never_becomes_informational() -> None:
    rows = complete_rows()
    for item in rows:
        item["qwen_cross_cache_full_greedy_policy_requested"] = "informational"
        item["qwen_cross_cache_full_greedy_policy_effective"] = "informational"
        item["qwen_cross_cache_full_greedy_required"] = False
    rows[0]["qwen_same_cache_greedy_match"] = False
    rows[0]["qwen_same_cache_full_greedy_mismatch_count"] = 1
    rows[0]["qwen_same_cache_full_greedy_first_mismatch_index"] = 7
    summary = validate_matrix(
        rows, expected_cross_cache_full_greedy_policy="informational"
    )
    assert summary["status"] == "fail"
    assert any("qwen_same_cache_greedy_match" in error for error in summary["errors"])


@pytest.mark.parametrize(
    "field",
    (
        "qwen_graph_logits_min_cosine",
        "qwen_dynamic_static_logits_min_cosine",
        "qwen_same_cache_logits_min_cosine",
    ),
)
@pytest.mark.parametrize("invalid", (True, float("nan"), float("inf")))
def test_cosine_gates_require_finite_real_numbers(field: str, invalid: object) -> None:
    rows = complete_rows()
    rows[0][field] = invalid
    summary = validate_matrix(rows)
    assert summary["status"] == "fail"
    assert any(field in error for error in summary["errors"])


def test_raw_graph_requires_exactly_one_launch_but_inductor_accepts_positive() -> None:
    rows = complete_rows()
    rows[0]["qwen_cuda_graph_launch_count"] = 2
    assert validate_matrix(rows)["status"] == "pass"

    for item in rows:
        use_raw_graph(item)
    rows[0]["qwen_cuda_graph_launch_count"] = 2
    summary = validate_matrix(rows)
    assert summary["status"] == "fail"
    assert any("qwen_cuda_graph_launch_count=2" in error for error in summary["errors"])

    rows[0]["qwen_cuda_graph_launch_count"] = True
    summary = validate_matrix(rows)
    assert summary["status"] == "fail"
    assert any("expected exactly 1" in error for error in summary["errors"])


def test_one_model_cannot_mix_compile_modes_across_cells() -> None:
    rows = complete_rows()
    rows[0]["qwen_compile_mode_effective"] = "reduce-overhead"
    summary = validate_matrix(rows)
    assert summary["status"] == "fail"
    assert any("mixed compile modes" in error for error in summary["errors"])


def test_rows_must_record_one_benchmark_repository_commit() -> None:
    rows = complete_rows()
    rows[0]["benchmark_repository_commit"] = "other-commit"
    summary = validate_matrix(rows)
    assert summary["status"] == "fail"
    assert any("repository commit" in error for error in summary["errors"])

    rows = complete_rows()
    for item in rows[:-1]:
        item.pop("benchmark_repository_commit")
    summary = validate_matrix(rows)
    assert summary["status"] == "fail"
    assert any("must be non-empty" in error for error in summary["errors"])


def test_non_finite_cosine_and_requested_mode_mismatch_fail() -> None:
    rows = complete_rows()
    rows[0]["qwen_same_cache_logits_min_cosine"] = float("nan")
    rows[1]["qwen_compile_mode_requested"] = "reduce-overhead"
    summary = validate_matrix(rows)
    assert summary["status"] == "fail"
    assert any("expected >=0.9999" in error for error in summary["errors"])
    assert any("requested/effective" in error for error in summary["errors"])


def test_graph_fallback_or_missing_cell_fails() -> None:
    rows = complete_rows()
    rows[0]["qwen_decode_cuda_graph_verified"] = False
    summary = validate_matrix(rows[:-1])
    assert summary["status"] == "fail"
    assert any(
        "qwen_decode_cuda_graph_verified=False" in error for error in summary["errors"]
    )
    assert any("missing cells" in error for error in summary["errors"])


@pytest.mark.parametrize(
    "field",
    (
        "qwen_graph_logits_trace_finite",
        "qwen_dynamic_static_logits_finite",
        "qwen_same_cache_logits_finite",
    ),
)
def test_trace_finite_gates_require_real_booleans(field: str) -> None:
    for invalid in (False, 1, None):
        rows = complete_rows()
        rows[0][field] = invalid
        summary = validate_matrix(rows)
        assert summary["status"] == "fail"
        assert any(field in error for error in summary["errors"])


def test_summary_sort_and_display_rounding_do_not_change_raw_values() -> None:
    rows = complete_rows()
    rows.reverse()
    rows[0]["decode_tokps_total"] = 99.94
    rows[0]["decode_tokps_total_raw"] = 100.04
    target_model = rows[0]["model_size_label"]
    target_batch = rows[0]["batch_size"]
    target_group = [
        item
        for item in rows
        if item["model_size_label"] == target_model
        and item["batch_size"] == target_batch
    ]
    for index, item in enumerate(target_group):
        item["prefill_tokps_total"] = float(index + 1)
        item["prefill_tokps_total_raw"] = float(1000 + index)
    summary = build_summary(rows)
    ordered = ordered_rows(rows)
    assert ordered[0]["model_size_label"] == "0.8b"
    assert ordered[0]["batch_size"] == 1
    assert summary["cells"][-1]["decode_tokps_total"] == 99.94
    assert summary["cells"][-1]["decode_tokps_total_raw"] == 100.04
    target_median = next(
        item
        for item in summary["model_batch_medians"]
        if item["model_size_label"] == target_model
        and item["batch_size"] == target_batch
    )
    assert target_median["prefill_tokps_median"] == 1002.5
    assert display_rate(99.94) == "99.9"
    assert display_rate(100.0) == "100"
    assert summary["correctness_contract"]["same_cache"]["cosine_threshold"] == 0.9999
    assert summary["correctness_contract"]["cross_cache"]["cosine_threshold"] is None
    assert (
        summary["correctness_contract"]["cross_cache"]["cosine_interpretation"]
        == "informational_only"
    )
    assert summary["reference_lane_eligible"] is True
    assert summary["unified_main_table_eligible"] is False
    assert summary["axis_composition"] == "independent_best_prefill_and_decode"
    assert summary["interpretation"]["continuous_end_to_end_path"] is False
    assert len(summary["model_correctness"]) == 4
    assert summary["model_correctness"][0]["same_cache_logits_min_cosine"] == 0.99999
    markdown = render_markdown(summary)
    assert "Same-cache hard gate" in markdown
    assert "minimum cosine >= 0.9999" in markdown
    assert "Cross-cache hard gates" in markdown
    assert "cosine is informational only" in markdown
    assert "PASS, reference-only" in markdown
    assert "not eligible for the unified RWKV/Qwen main table" in markdown
    assert "not an official Qwen Graph path" in markdown

    first_pair = rows[0]["model_pair"]
    gpu_rows = [
        {**rows[0], "model_pair": first_pair, "device": "NVIDIA GeForce RTX 5090"},
        {**rows[0], "model_pair": first_pair, "device": "NVIDIA GeForce RTX 4090"},
    ]
    assert [item["device"] for item in ordered_rows(gpu_rows)] == [
        "NVIDIA GeForce RTX 4090",
        "NVIDIA GeForce RTX 5090",
    ]
