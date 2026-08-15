from __future__ import annotations

from itertools import product

from bench.validate_qwen35_best_optimized_hf_v1 import PAIRS as QWEN_PAIRS
from bench.validate_qwen35_paired_pd_v1 import (
    PAIRS,
    PARAMETERS,
    RWKV_SIZES,
    validate_paired_pd,
)
from tests.test_qwen35_best_optimized_hf_v1 import row as qwen_row
from bench.bench_cross_model_speed import rwkv_native_graph_decode_route


COMMIT = "a" * 40
RUNTIME = {
    "torch_version": "2.11.0+cu130",
    "torch_cuda_version": "13.0",
    "triton_version": "3.6.0",
    "transformers_version": "5.12.1",
    "fla_version": "0.5.1",
    "causal_conv1d_version": "1.6.2.post1",
}


def references() -> list[dict]:
    rows: list[dict] = []
    for pair, (batch, prompt, decode) in product(
        PAIRS, product((1, 8), (128, 512, 2048), (128, 512))
    ):
        item = qwen_row(pair, batch, prompt, decode)
        item.update(
            {
                "benchmark_repository_commit": COMMIT,
                "device": "NVIDIA GeForce RTX 4080",
                "active_parameter_count": PARAMETERS[pair][1],
                **RUNTIME,
            }
        )
        if pair == PAIRS[2]:
            item.update(
                {
                    "qwen_decode_optimization_requested": "module_call_dynamic",
                    "qwen_decode_optimization_effective": "module_call_dynamic",
                    "step_backend": "module_call",
                    "prefill_backend_effective": "module_call",
                    "prefill_cache_type": "DynamicCache",
                    "cache_type": "DynamicCache",
                    "qwen_axis_composition": "continuous_single_cache_path",
                    "qwen_cuda_graph_requested": False,
                    "qwen_cuda_graph_effective": False,
                    "qwen_decode_cuda_graph_verified": False,
                }
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
                "qwen_dynamic_static_logits_min_cosine",
                "qwen_same_cache_logits_min_cosine",
                "qwen_static_compiled_logits_min_cosine",
                "qwen_graph_max_cache_len",
                "qwen_graph_probe_tokens",
                "qwen_graph_logits_probe_tokens",
                "qwen_graph_distinct_batch_prompts",
                "qwen_graph_scope",
            ):
                item[field] = None
        rows.append(item)
    return rows


def candidate_row(pair: str, batch: int, prompt: int, decode: int) -> dict:
    size, layers = RWKV_SIZES[pair]
    selected_layers = list(range(layers)) if batch == 8 else []
    hidden, layer_count = {
        PAIRS[0]: (1024, 24),
        PAIRS[1]: (2048, 24),
        PAIRS[2]: (2560, 32),
    }[pair]
    block = (hidden, layer_count, batch, prompt) in {
        (1024, 24, 8, 512),
        (1024, 24, 8, 2048),
        (2048, 24, 8, 512),
        (2048, 24, 8, 2048),
        (2560, 32, 1, 512),
        (2560, 32, 1, 2048),
    }
    result = {
        "axis": "qwen35_cross_model_speed",
        "benchmark_matrix": "qwen35_paired_pd_v1",
        "benchmark_repository_commit": COMMIT,
        "optimization_lane": "best_optimized_hf",
        "model_pair": pair,
        "model_size_label": size,
        "model_role": "candidate",
        "model_kind": "rwkv",
        "rwkv_implementation_requested": "auto",
        "rwkv_implementation_effective": "native_model",
        "dtype": "fp16",
        "quantization": "none",
        "quantization_backend": "dense",
        "native_quant_kernel_active": False,
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
        "logits_finite": True,
        "device": "NVIDIA GeForce RTX 4080",
        "gpu_arch": "sm_89",
        "gpu_compute_capability": [8, 9],
        "rwkv_fast_token_backend_requested": "native_graph",
        "rwkv_native_model_backend_requested": "native_graph",
        "effective_backend": "native_graph",
        "step_backend": "rwkv_fast_token",
        "cache_type": "NativeRWKV7Cache",
        "prefill_effective_backend": "native_prefill_graph",
        "prefill_backend_effective": "native_prefill_graph",
        "active_parameter_count": PARAMETERS[pair][0],
        "rwkv_native_graph_ada_wagv_lora_extension_requested": batch == 1,
        "rwkv_native_graph_ada_wagv_lora_extension_selected": batch == 1,
        "rwkv_native_graph_ada_wagv_lora_extension_effective": batch == 1,
        "rwkv_native_graph_ada_wagv_lora_extension_selected_layers": (
            list(range(layers)) if batch == 1 else []
        ),
        "rwkv_native_graph_ada_wagv_lora_extension_effective_layers": (
            list(range(layers)) if batch == 1 else []
        ),
        "rwkv_native_graph_ada_wagv_lora_extension_effective_layer_count": (
            layers if batch == 1 else 0
        ),
        "rwkv_native_graph_ada_wagv_lora_extension_full_model_effective": batch == 1,
        "rwkv_native_graph_rkv_policy": (
            "manual" if pair == PAIRS[2] and batch == 8 else "vkwr_auto"
        ),
        "rwkv_native_graph_ada_wagv_bmm_requested": True,
        "rwkv_native_graph_ada_wagv_bmm_selected": batch == 8,
        "rwkv_native_graph_ada_wagv_bmm_effective": batch == 8,
        "rwkv_native_graph_ada_wagv_bmm_selected_layers": selected_layers,
        "rwkv_native_graph_ada_wagv_bmm_effective_layers": selected_layers,
        "rwkv_native_graph_ada_wagv_bmm_effective_layer_count": len(selected_layers),
        "rwkv_native_graph_ada_wagv_bmm_full_model_effective": batch == 8,
        "rwkv_prefill_global_fp16_accum_effective": not block,
        "rwkv_prefill_block_fp16_accum_effective": block,
        "prefill_sec_samples": [0.05] * 7,
        "prefill_sec_median": 0.05,
        "prefill_sec_median_raw": 0.05,
        "prefill_tokps_total": round(batch * prompt / 0.05, 3),
        "prefill_tokps_total_raw": batch * prompt / 0.05,
        "decode_sec_samples": [0.1] * 7,
        "decode_sec_median": 0.1,
        "decode_sec_median_raw": 0.1,
        "decode_tokps_total": round(batch * decode / 0.1, 3),
        "decode_tokps_total_raw": batch * decode / 0.1,
        **RUNTIME,
    }
    for route in ("sm120_wagv_bmm_g", "sm120_compiled_ffn"):
        result.update(
            {
                f"rwkv_native_graph_{route}_requested": False,
                f"rwkv_native_graph_{route}_selected": False,
                f"rwkv_native_graph_{route}_effective": False,
                f"rwkv_native_graph_{route}_selected_layers": [],
                f"rwkv_native_graph_{route}_effective_layers": [],
                f"rwkv_native_graph_{route}_effective_layer_count": 0,
                f"rwkv_native_graph_{route}_full_model_effective": False,
            }
        )
    return result


def candidates() -> list[dict]:
    return [
        candidate_row(pair, batch, prompt, decode)
        for pair, (batch, prompt, decode) in product(
            PAIRS, product((1, 8), (128, 512, 2048), (128, 512))
        )
    ]


def test_strict_4080_paired_pd_matrix_passes_all_four_gates() -> None:
    summary = validate_paired_pd(candidates(), references())
    assert summary["status"] == "pass"
    assert summary["paired_pd_table_eligible"] is True
    assert summary["coverage"]["joined_cells"] == 36
    assert summary["gate"]["passing_cells"] == {
        "raw_prefill_ratio": 36,
        "raw_decode_ratio": 36,
        "adjusted_prefill_ratio": 36,
        "adjusted_decode_ratio": 36,
    }


def test_any_raw_or_adjusted_red_cell_fails_closed() -> None:
    candidate = candidates()
    candidate[0]["decode_sec_samples"] = [0.3] * 7
    candidate[0]["decode_sec_median"] = 0.3
    candidate[0]["decode_sec_median_raw"] = 0.3
    candidate[0]["decode_tokps_total"] = round(
        candidate[0]["batch_size"] * candidate[0]["decode_tokens"] / 0.3, 3
    )
    candidate[0]["decode_tokps_total_raw"] = (
        candidate[0]["batch_size"] * candidate[0]["decode_tokens"] / 0.3
    )
    summary = validate_paired_pd(candidate, references())
    assert summary["status"] == "fail"
    assert summary["gate"]["passing_cells"]["raw_decode_ratio"] == 35
    assert summary["red_cells"]


def test_runtime_route_sample_and_coverage_drift_fail() -> None:
    candidate = candidates()
    candidate[0]["torch_version"] = "different"
    candidate[1]["rwkv_native_graph_ada_wagv_bmm_requested"] = False
    candidate[2]["prefill_sec_samples"] = [0.05] * 6
    summary = validate_paired_pd(candidate[:-1], references())
    assert summary["status"] == "fail"
    assert any(
        "candidate rows do not have one runtime signature" in error
        for error in summary["errors"]
    )
    assert any("ada_wagv_bmm_requested" in error for error in summary["errors"])
    assert any("must contain 7 samples" in error for error in summary["errors"])
    assert any("missing cells" in error for error in summary["errors"])


def test_reference_contract_still_uses_known_qwen_pairs() -> None:
    assert tuple(QWEN_PAIRS)[:3] == PAIRS


def test_4b_reference_route_is_locked_to_continuous_dynamic_cache() -> None:
    reference = references()
    row = next(item for item in reference if item["model_pair"] == PAIRS[2])
    row["qwen_decode_optimization_requested"] = "static_cache_raw_cudagraph"
    row["qwen_decode_optimization_effective"] = "static_cache_raw_cudagraph"
    summary = validate_paired_pd(candidates(), reference)
    assert summary["status"] == "fail"
    assert any(
        "expected one of ['module_call_dynamic']" in error
        for error in summary["errors"]
    )


def test_cross_model_row_exports_exact_base_bmm_layer_sets() -> None:
    class Model:
        def rwkv7_native_graph_runner_copy_stats(self) -> dict:
            return {
                "runners": [
                    {
                        "batch_size": 8,
                        "ada_wagv_bmm_selected_layers": [0, 1],
                        "ada_wagv_bmm_effective_layers": [0, 1],
                    }
                ]
            }

    route = rwkv_native_graph_decode_route(Model(), 8)
    assert route["rwkv_native_graph_ada_wagv_bmm_selected_layers"] == [0, 1]
    assert route["rwkv_native_graph_ada_wagv_bmm_effective_layers"] == [0, 1]
