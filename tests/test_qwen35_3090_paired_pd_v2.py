from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "bench" / "validate_qwen35_3090_paired_pd_v2.py"
QWEN_RUNNER_PATH = ROOT / "bench" / "run_5090_qwen35_best_optimized_hf.sh"
QWEN_ROUTE_PROBE_PATH = ROOT / "bench" / "run_3090_qwen_graph_route_probe_v1.sh"
QWEN_3090_RUNNER_PATH = ROOT / "bench" / "run_3090_qwen35_best_optimized_hf.sh"
RWKV_3090_RUNNER_PATH = ROOT / "bench" / "run_3090_rwkv_paired_pd_v2.sh"


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_3090_pd", VALIDATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _contract(module) -> dict:
    return {
        "schema_version": 2,
        "protocol": module.CONTRACT_PROTOCOL,
        "device": module.EXPECTED_DEVICE,
        "gpu_arch": "sm_86",
        "compute_cap": "8.6",
        "driver_version": "550.142",
        "memory_total_mib": 24576,
        "runtime": module.EXPECTED_RUNTIME,
        "torch_cuda_arch_list": "8.6",
        "cross_cache_full_greedy_policy": "informational",
        "reference_sha256": "a" * 64,
        "routes_by_pair": {
            pair: "static_cache_raw_cudagraph" for pair in module.base.PAIRS
        },
    }


def _route_fields(route: str, enabled: bool, layer_count: int) -> dict:
    layers = list(range(layer_count)) if enabled else []
    prefix = f"rwkv_native_graph_{route}_"
    return {
        prefix + "requested": enabled,
        prefix + "selected": enabled,
        prefix + "effective": enabled,
        prefix + "selected_layers": layers,
        prefix + "effective_layers": layers,
        prefix + "effective_layer_count": len(layers),
        prefix + "full_model_effective": enabled,
    }


def _sm86_candidate(module, pair: str, batch: int) -> dict:
    base = module.base
    small_b8 = pair in base.PAIRS[:2] and batch == 8
    bmm = small_b8 or (pair == base.PAIRS[2] and batch == 8)
    layer_count = base.LAYERS[pair]
    row = {
        "_source": "fixture",
        "axis": "qwen35_cross_model_speed",
        "benchmark_matrix": module.PROTOCOL,
        "optimization_lane": "best_optimized_hf",
        "model_role": "candidate",
        "model_kind": "rwkv",
        "model_pair": pair,
        "model_size_label": base.SIZES[pair],
        "active_parameter_count": base.PARAMETERS[pair][0],
        "rwkv_implementation_requested": "auto",
        "rwkv_implementation_effective": "native_model",
        "dtype": "fp16",
        "quantization": "none",
        "quantization_backend": "dense",
        "native_quant_kernel_active": False,
        "device": module.EXPECTED_DEVICE,
        "gpu_arch": "sm_86",
        "gpu_compute_capability": [8, 6],
        "prefill_chunk_size": 512,
        "warmup": 3,
        "runs": 7,
        "timing_statistic": "median",
        "resident_sweep": True,
        "status": "pass",
        "logits_finite": True,
        "rwkv_fast_token_backend_requested": "native_graph",
        "rwkv_native_model_backend_requested": "native_graph",
        "effective_backend": "native_graph",
        "step_backend": "rwkv_fast_token",
        "cache_type": "NativeRWKV7Cache",
        "batch_size": batch,
        "prompt_tokens": 128,
        "decode_tokens": 128,
        "prefill_sec_samples": [1.0] * 7,
        "prefill_sec_median_raw": 1.0,
        "prefill_tokps_total_raw": float(batch * 128),
        "decode_sec_samples": [1.0] * 7,
        "decode_sec_median_raw": 1.0,
        "decode_tokps_total_raw": float(batch * 128),
        "benchmark_repository_commit": "a" * 40,
        "rwkv_decode_route_profile": "sm86_qwen_alignment",
        "rwkv_native_graph_rkv_policy": (
            "manual" if pair == base.PAIRS[3] else "vkwr_auto"
        ),
        "rwkv_native_graph_fused_norm_mix_num_warps": 8,
        "rwkv_native_graph_state_dtype": (
            "torch.float32" if small_b8 else "torch.float16"
        ),
        "rwkv_native_graph_fp16_recurrent": not small_b8,
        "rwkv_native_graph_ada_linear_requested": (
            batch == 1 and pair in {base.PAIRS[0], base.PAIRS[3]}
        ),
        "rwkv_native_graph_ada_linear_extension_required": (
            batch == 1 and pair in {base.PAIRS[0], base.PAIRS[3]}
        ),
        "rwkv_native_graph_ada_linear_rows_requested": "1",
        "rwkv_native_graph_ada_linear_roles_requested": "hidden,ffn_up,ffn_down",
        "rwkv_native_graph_ada_sparse_ffn_requested": False,
        "rwkv_native_graph_ada_wagv_lora_extension_required": batch == 1,
        **_route_fields("ada_wagv_bmm", bmm, layer_count),
        **_route_fields("sm120_wagv_bmm_g", small_b8, layer_count),
        **_route_fields("sm120_compiled_ffn", small_b8, layer_count),
        **_route_fields("ada_wagv_lora_extension", batch == 1, layer_count),
    }
    if small_b8:
        row.update(
            {
                "rwkv_native_graph_sm120_compiled_ffn_compile_effective": True,
                "rwkv_native_graph_sm120_compiled_ffn_compile_reused": True,
                "rwkv_native_graph_sm120_compiled_ffn_unique_graphs": 1,
                "rwkv_native_graph_sm120_compiled_ffn_graph_breaks": 0,
                "rwkv_native_graph_sm120_compiled_ffn_compile_mode": "max-autotune-no-cudagraphs",
                "rwkv_native_graph_sm120_compiled_ffn_prewarm_all_finite": True,
                "rwkv_native_graph_sm120_compiled_ffn_prewarm_argmax_all_equal": True,
                "rwkv_native_graph_sm120_compiled_ffn_prewarm_min_cosine": 0.99999,
            }
        )
    return row


def test_3090_contract_is_current_48_cell_protocol(tmp_path: Path) -> None:
    module = _load_validator()
    assert module.PROTOCOL == "qwen35_3090_paired_pd_v2"
    assert module.EXPECTED_DEVICE == "NVIDIA GeForce RTX 3090"
    assert module.EXPECTED_RUNTIME["python"] == "3.10.12"
    assert len(module.base.EXPECTED_KEYS) == 48
    path = tmp_path / "reference-contract.json"
    path.write_text(json.dumps(_contract(module)), encoding="utf-8")
    doc, errors = module._load_contract(path)
    assert errors == []
    assert doc["compute_cap"] == "8.6"


def test_3090_reference_contract_rejects_bool_and_partial_routes(
    tmp_path: Path,
) -> None:
    module = _load_validator()
    contract = _contract(module)
    contract["memory_total_mib"] = True
    contract["routes_by_pair"].pop(next(iter(contract["routes_by_pair"])))
    path = tmp_path / "reference-contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    _, errors = module._load_contract(path)
    assert any("memory_total_mib" in error for error in errors)
    assert any("routes_by_pair coverage" in error for error in errors)


def test_3090_validator_selects_the_exact_sm86_alignment_profile() -> None:
    source = VALIDATOR_PATH.read_text(encoding="utf-8")
    assert "base.SPECIAL_SMALL_B8_BUNDLE = False" in source
    assert "base.BASE_ADA_WAGV_BMM_EXPECTED = None" in source
    assert 'base.CANDIDATE_ROUTE_PROFILE = "sm86_qwen_alignment"' in source
    assert 'base.EXPECTED_ARCH = "8.6"' in source
    assert 'base.EXPECTED_MEMORY = "24576 MiB"' in source
    assert 'base.QWEN_CROSS_CACHE_FULL_GREEDY_POLICY = "informational"' in source


def test_3090_validator_accepts_all_eight_exact_lane_routes() -> None:
    module = _load_validator()
    base = module.base
    base.PROTOCOL = module.PROTOCOL
    base.EXPECTED_DEVICE = module.EXPECTED_DEVICE
    base.EXPECTED_ARCH = "8.6"
    base.CANDIDATE_ROUTE_PROFILE = "sm86_qwen_alignment"
    for pair in base.PAIRS:
        for batch in (1, 8):
            errors: list[str] = []
            base._validate_candidate(_sm86_candidate(module, pair, batch), errors)
            assert errors == [], (pair, batch, errors)


def test_3090_validator_rejects_route_layer_or_linear_drift() -> None:
    module = _load_validator()
    base = module.base
    base.PROTOCOL = module.PROTOCOL
    base.EXPECTED_DEVICE = module.EXPECTED_DEVICE
    base.EXPECTED_ARCH = "8.6"
    base.CANDIDATE_ROUTE_PROFILE = "sm86_qwen_alignment"

    b8 = _sm86_candidate(module, base.PAIRS[0], 8)
    b8["rwkv_native_graph_sm120_compiled_ffn_effective_layers"] = list(range(23))
    errors: list[str] = []
    base._validate_candidate(b8, errors)
    assert any("compiled_ffn_effective_layers" in error for error in errors)

    b1 = _sm86_candidate(module, base.PAIRS[3], 1)
    b1["rwkv_native_graph_ada_linear_extension_required"] = False
    errors = []
    base._validate_candidate(b1, errors)
    assert any("ada_linear_extension_required" in error for error in errors)


def test_3090_rwkv_runner_invokes_non_executable_base_via_bash() -> None:
    source = RWKV_3090_RUNNER_PATH.read_text(encoding="utf-8")
    assert "ROUTE_PROFILE=sm86_qwen_alignment" in source
    assert "SMALL_B8_MODE=sm86_qwen_alignment" in source
    assert "unset ADA_WAGV_BMM_OVERRIDE" in source
    assert 'exec bash "${ROOT}/bench/run_4090_rwkv_paired_pd_v2.sh" "$@"' in source


def test_qwen_formal_runner_accepts_an_exact_card_override() -> None:
    source = QWEN_RUNNER_PATH.read_text(encoding="utf-8")
    assert 'EXPECTED_GPU_MODEL="${EXPECTED_GPU_MODEL:-5090}"' in source
    assert '--model "${EXPECTED_GPU_MODEL}"' in source

    wrapper = QWEN_3090_RUNNER_PATH.read_text(encoding="utf-8")
    assert "EXPECTED_GPU_MODEL=3090" in wrapper
    assert "TORCH_CUDA_ARCH_LIST=8.6" in wrapper
    assert "QWEN_CROSS_CACHE_FULL_GREEDY_POLICY=informational" in wrapper
    assert "run_5090_qwen35_best_optimized_hf.sh" in wrapper


def test_3090_route_probe_compares_short_and_boundary_without_cell_mixing() -> None:
    source = QWEN_ROUTE_PROBE_PATH.read_text(encoding="utf-8")
    assert "--model 3090" in source
    assert "TORCH_CUDA_ARCH_LIST=8.6" in source
    assert "static_cache_inductor_cudagraph" in source
    assert "static_cache_raw_cudagraph" in source
    assert "1x128x128 8x128x128 1x2048x512 8x2048x512" in source
    assert "select_qwen35_graph_route_v1.py" in source
    assert '--expected-device "NVIDIA GeForce RTX 3090"' in source
    assert "--cross-cache-full-greedy-policy informational" in source
