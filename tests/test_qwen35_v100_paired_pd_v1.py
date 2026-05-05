from __future__ import annotations

import runpy
import hashlib
import json
from argparse import Namespace
from pathlib import Path

import torch

from bench.bench_cross_model_speed import configure_qwen_sdpa_policy
from bench.validate_qwen35_best_optimized_hf_v1 import validate_matrix
from bench.validate_qwen35_v100_paired_pd_v1 import (
    PAIRS,
    PARAMETERS,
    QWEN_CONTRACT,
    QWEN_LANE,
    QWEN_MATRIX,
    QWEN_ROUTES,
    QWEN_SDPA_POLICIES,
    RWKV_SIZES,
    _validate_candidate_row,
    _validate_qwen_sdpa_row,
    _validate_targeted_same_implementation_closure,
    validate_correctness_manifest,
    validate_provenance,
)
from bench.compare_rwkv_prefill_probe import compare as compare_rwkv_probes

HELPERS = runpy.run_path(
    str(Path(__file__).with_name("test_qwen35_best_optimized_hf_v1.py"))
)
complete_rows = HELPERS["complete_rows"]
use_raw_graph = HELPERS["use_raw_graph"]


def v100_qwen_rows() -> list[dict]:
    rows = complete_rows()
    for item in rows:
        use_raw_graph(item)
        sdpa_policy = QWEN_SDPA_POLICIES[item["model_pair"]]
        automatic = sdpa_policy == "auto"
        item.update(
            {
                "benchmark_matrix": QWEN_MATRIX,
                "device": "Tesla V100-PCIE-32GB",
                "torch_version": "2.5.1+cu124",
                "torch_cuda_version": "12.4",
                "triton_version": "3.4.0",
                "qwen_conv_backend_requested": "fla_triton",
                "qwen_fast_path_available": False,
                "qwen_causal_conv1d_importable": False,
                "qwen_conv_backend_effective": "fla_triton",
                "causal_conv1d_version": None,
                "qwen_sdpa_policy_requested": sdpa_policy,
                "qwen_sdpa_policy_effective": sdpa_policy,
                "qwen_sdp_flash_enabled": automatic,
                "qwen_sdp_mem_efficient_enabled": automatic,
                "qwen_sdp_math_enabled": True,
                "qwen_sdp_cudnn_enabled": automatic,
            }
        )
    return rows


def v100_candidate(batch: int, pair: str = PAIRS[0]) -> dict:
    size, layer_count = RWKV_SIZES[pair]
    item = {
        "axis": "qwen35_cross_model_speed",
        "benchmark_matrix": "qwen35_v100_paired_pd_v1",
        "benchmark_repository_commit": "1" * 40,
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
        "prompt_tokens": 128,
        "decode_tokens": 128,
        "prefill_chunk_size": 512,
        "warmup": 3,
        "runs": 7,
        "timing_statistic": "median",
        "mtp_enabled": False,
        "speculative_decoding_enabled": False,
        "resident_sweep": True,
        "status": "pass",
        "logits_finite": True,
        "device": "Tesla V100-PCIE-32GB",
        "gpu_arch": "sm_70",
        "gpu_compute_capability": [7, 0],
        "rwkv_fast_token_backend_requested": "native_graph",
        "rwkv_native_model_backend_requested": "native_graph",
        "effective_backend": "native_graph",
        "step_backend": "rwkv_fast_token",
        "cache_type": "NativeRWKV7Cache",
        "active_parameter_count": PARAMETERS[pair][0],
        "prefill_sec_samples": [0.05] * 7,
        "prefill_sec_median": 0.05,
        "prefill_sec_median_raw": 0.05,
        "prefill_tokps_total_raw": batch * 128 / 0.05,
        "decode_sec_samples": [0.1] * 7,
        "decode_sec_median": 0.1,
        "decode_sec_median_raw": 0.1,
        "decode_tokps_total_raw": batch * 128 / 0.1,
        "torch_version": "2.5.1+cu124",
        "torch_cuda_version": "12.4",
        "triton_version": "3.4.0",
        "transformers_version": "5.12.1",
        "fla_version": "0.5.1",
        "causal_conv1d_version": None,
    }
    layer_indices = list(range(1, layer_count))
    fp16_state = batch == 8 and (pair in PAIRS[:2] or pair == PAIRS[3])
    item.update(
        {
            "rwkv_native_graph_rkv_policy": (
                "vkwr_auto" if pair == PAIRS[3] and batch == 8 else "manual"
            ),
            "rwkv_native_graph_fused_norm_mix_num_warps": (
                8 if pair == PAIRS[3] and batch == 8 else 4
            ),
            "rwkv_native_graph_state_dtype": (
                "torch.float16" if fp16_state else "torch.float32"
            ),
            "rwkv_native_graph_triton_fp16_state": fp16_state,
            "rwkv_native_graph_fp16_recurrent": False,
            "rwkv_native_graph_sm70_wagv_lora_extension_required": batch == 1,
            "rwkv_native_graph_sm70_wagv_lora_extension_available": batch == 1,
            "rwkv_native_graph_sm70_wagv_lora_selected": batch == 1,
            "rwkv_native_graph_sm70_wagv_lora_effective": batch == 1,
            "rwkv_native_graph_sm70_wagv_lora_selected_layers": (
                layer_indices if batch == 1 else []
            ),
            "rwkv_native_graph_sm70_wagv_lora_effective_layers": (
                layer_indices if batch == 1 else []
            ),
            "rwkv_native_graph_sm70_wagv_lora_effective_layer_count": (
                layer_count - 1 if batch == 1 else 0
            ),
            "rwkv_native_graph_sm70_wagv_lora_full_eligible_layers_effective": (
                batch == 1
            ),
            "rwkv_native_graph_fused_wavg_lora_selected": (
                batch == 8 and pair != PAIRS[3]
            ),
            "rwkv_native_graph_fused_wavg_lora_effective": (
                batch == 8 and pair != PAIRS[3]
            ),
            "rwkv_native_graph_fused_wavg_lora_selected_layers": (
                layer_indices if batch == 8 and pair != PAIRS[3] else []
            ),
            "rwkv_native_graph_fused_wavg_lora_effective_layers": (
                layer_indices if batch == 8 and pair != PAIRS[3] else []
            ),
            "rwkv_native_graph_fused_wavg_lora_effective_layer_count": (
                layer_count - 1 if batch == 8 and pair != PAIRS[3] else 0
            ),
            "rwkv_native_graph_fused_wavg_lora_full_eligible_layers_effective": (
                batch == 8 and pair != PAIRS[3]
            ),
            "rwkv_native_graph_ada_wagv_bmm_requested": False,
            "rwkv_native_graph_ada_wagv_bmm_selected": False,
            "rwkv_native_graph_ada_wagv_bmm_effective": False,
            "rwkv_native_graph_ada_wagv_bmm_selected_layers": [],
            "rwkv_native_graph_ada_wagv_bmm_effective_layers": [],
            "rwkv_native_graph_ada_wagv_bmm_effective_layer_count": 0,
            "rwkv_native_graph_ada_wagv_bmm_full_model_effective": False,
        }
    )
    for route in ("sm120_wagv_bmm_g", "sm120_compiled_ffn"):
        item.update(
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
    return item


def test_v100_qwen_profile_accepts_fla_triton_raw_graph() -> None:
    summary = validate_matrix(
        v100_qwen_rows(),
        expected_device="Tesla V100-PCIE-32GB",
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
    assert summary["status"] == "pass", summary["errors"]
    assert summary["reference_rows"] == 48


def test_v100_qwen_sdpa_policy_is_exact_per_model() -> None:
    rows = v100_qwen_rows()
    for item in rows:
        errors: list[str] = []
        _validate_qwen_sdpa_row(item, errors)
        assert errors == []

    qwen9 = next(item for item in rows if item["model_pair"] == PAIRS[3])
    qwen9["qwen_sdp_mem_efficient_enabled"] = True
    errors = []
    _validate_qwen_sdpa_row(qwen9, errors)
    assert any("qwen_sdp_mem_efficient_enabled" in error for error in errors)


def test_qwen_math_only_sdpa_policy_disables_fused_backends() -> None:
    args = Namespace(model_kind="qwen35", qwen_sdpa_policy="math_only")
    try:
        configure_qwen_sdpa_policy(args)
        assert args._qwen_sdpa_policy_effective == "math_only"
        assert torch.backends.cuda.flash_sdp_enabled() is False
        assert torch.backends.cuda.mem_efficient_sdp_enabled() is False
        assert torch.backends.cuda.math_sdp_enabled() is True
        if callable(getattr(torch.backends.cuda, "cudnn_sdp_enabled", None)):
            assert torch.backends.cuda.cudnn_sdp_enabled() is False
    finally:
        configure_qwen_sdpa_policy(
            Namespace(model_kind="qwen35", qwen_sdpa_policy="auto")
        )


def test_v100_candidate_routes_are_exact_for_b1_and_b8() -> None:
    for pair in (PAIRS[0], PAIRS[3]):
        for batch in (1, 8):
            errors: list[str] = []
            _validate_candidate_row(
                v100_candidate(batch, pair),
                expected_device="Tesla V100-PCIE-32GB",
                errors=errors,
            )
            assert errors == []


def test_v100_7p2_b8_closure_route_is_fail_closed() -> None:
    item = v100_candidate(8, PAIRS[3])
    for field, bad in (
        ("rwkv_native_graph_rkv_policy", "manual"),
        ("rwkv_native_graph_fused_norm_mix_num_warps", 4),
        ("rwkv_native_graph_state_dtype", "torch.float32"),
        ("rwkv_native_graph_triton_fp16_state", False),
    ):
        changed = dict(item)
        changed[field] = bad
        errors: list[str] = []
        _validate_candidate_row(
            changed,
            expected_device="Tesla V100-PCIE-32GB",
            errors=errors,
        )
        assert any(field in error for error in errors)


def test_v100_candidate_rejects_route_telemetry_pollution() -> None:
    item = v100_candidate(8)
    item["rwkv_native_graph_sm70_wagv_lora_effective"] = True
    errors: list[str] = []
    _validate_candidate_row(
        item,
        expected_device="Tesla V100-PCIE-32GB",
        errors=errors,
    )
    assert any("sm70_wagv_lora_effective" in error for error in errors)


def test_v100_evidence_parsers_fail_closed_on_malformed_inputs(tmp_path) -> None:
    malformed = tmp_path / "malformed.json"
    malformed.write_text("[]", encoding="utf-8")
    correctness = validate_correctness_manifest(malformed, [])
    assert correctness["status"] == "fail"
    provenance = validate_provenance(
        candidate_path=malformed,
        reference_paths=[],
        candidate_rows=[],
        route_manifest_path=malformed,
        qwen_route_manifest_paths=[],
        correctness_manifest_path=malformed,
        runtime_lock_path=malformed,
        candidate_model_hashes_path=malformed,
    )
    assert provenance["status"] == "fail"


def test_v100_targeted_native_eager_closure_is_bound_and_fail_closed(
    tmp_path: Path,
) -> None:
    model_path = "/models/rwkv7-g1i-7.2b-hf"
    formal = v100_candidate(8, PAIRS[3])
    formal["model_id_or_path"] = model_path
    reference = {
        **formal,
        "rwkv_fast_token_backend_requested": "module_call",
        "rwkv_native_model_backend_requested": "eager",
        "effective_backend": "eager",
        "warmup": 1,
        "runs": 1,
    }
    candidate = {
        **formal,
        "warmup": 3,
        "runs": 7,
    }
    reference_row = tmp_path / "reference.jsonl"
    candidate_row = tmp_path / "candidate.jsonl"
    reference_row.write_text(json.dumps(reference) + "\n", encoding="utf-8")
    candidate_row.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
    probe = {
        "input_ids": torch.arange(8 * 128).reshape(8, 128),
        "greedy_tokens": torch.arange(512 * 8).reshape(512, 8),
        "prompt_logits": torch.arange(1, 33, dtype=torch.float32).reshape(8, 4),
        "final_logits": torch.arange(1, 33, dtype=torch.float32).reshape(8, 4),
        "decode_logits_all_finite": True,
        "decode_logits_finite_by_batch": torch.ones(8, dtype=torch.bool),
        "probe_batch_size": 8,
        "probe_tokens": 512,
        "distinct_batch_prompts": True,
    }
    reference_probe = tmp_path / "reference.pt"
    candidate_probe = tmp_path / "candidate.pt"
    torch.save(probe, reference_probe)
    torch.save(probe, candidate_probe)
    comparison = compare_rwkv_probes(probe, probe, 0.9999)
    comparison["contract_errors"] = []
    comparison_path = tmp_path / "comparison.json"
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")

    def artifact(path: Path) -> dict[str, str]:
        return {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    closure = {
        "model_pair": PAIRS[3],
        "model_size_label": "7.2b",
        "model_path": model_path,
        "batch_size": 8,
        "prompt_tokens": 128,
        "decode_tokens": 128,
        "probe_tokens": 512,
        "reference_contract": {
            "rwkv_implementation": "native_model",
            "effective_backend": "eager",
            "RWKV7_FAST_TOKEN_BACKEND": "module_call",
            "RWKV7_NATIVE_MODEL_BACKEND": "eager",
            "RWKV7_FAST_PREFILL": "0",
            "RWKV7_NATIVE_PREFILL_GRAPH": "0",
        },
        "candidate_contract": {
            "rwkv_implementation": "native_model",
            "effective_backend": "native_graph",
            "rkv_policy": "vkwr_auto",
            "state_dtype": "torch.float16",
            "triton_fp16_state": True,
            "fused_norm_mix_num_warps": 8,
            "fused_wavg_lora": False,
        },
        "native_eager_reference": {
            "row": artifact(reference_row),
            "probe": artifact(reference_probe),
        },
        "native_graph_candidate": {
            "row": artifact(candidate_row),
            "probe": artifact(candidate_probe),
        },
        "comparison": artifact(comparison_path),
    }
    errors: list[str] = []
    result = _validate_targeted_same_implementation_closure(
        {"targeted_same_implementation_closure": closure},
        tmp_path / "manifest.json",
        {(PAIRS[3], 8, 128, 128): formal},
        errors,
    )
    assert result["status"] == "pass", errors
    assert errors == []

    closure["candidate_contract"]["fused_wavg_lora"] = True
    errors = []
    result = _validate_targeted_same_implementation_closure(
        {"targeted_same_implementation_closure": closure},
        tmp_path / "manifest.json",
        {(PAIRS[3], 8, 128, 128): formal},
        errors,
    )
    assert result["status"] == "fail"
    assert any("candidate contract mismatch" in error for error in errors)
