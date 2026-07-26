from __future__ import annotations

import importlib
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def source_kernel_package(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "kernels"))
    for name in tuple(sys.modules):
        if name == "rwkv7_kernels" or name.startswith("rwkv7_kernels."):
            sys.modules.pop(name)


def policy(name: str, capability: tuple[int, int]):
    module = importlib.import_module("rwkv7_kernels.nvidia.kernel_policy")
    profile = module.classify_gpu(name, capability)
    return profile, module.policy_for_profile(profile)


def test_v100_policy_maps_sm70_prefill_decode_and_state_families():
    profile, route = policy("Tesla V100-SXM2-32GB", (7, 0))
    assert profile.family == "volta"
    assert route.fast_prefill
    assert route.fused_prefill_scan
    assert route.prefill_graph and route.prefill_graph_cache_size == 4
    assert route.fused_prefill_shift_mix
    assert route.fused_prefill_state_prep and route.fused_prefill_state_scan
    assert route.fused_prefill_output
    assert route.fused_recurrent_raw and route.fused_recurrent_output
    assert route.fused_norm_mix and route.fused_output
    assert route.sm70_linear and route.sm70_wagv_lora
    assert route.fused_wavg_lora and route.ada_sparse_ffn
    assert route.native_graph_triton_fp16_state
    assert (2048, 24, 8) in route.native_graph_triton_fp16_state_model_shapes


def test_rtx4080_policy_maps_dynamic_prefill_graph_and_grouped_decode():
    profile, route = policy("NVIDIA GeForce RTX 4080", (8, 9))
    assert profile.family == "ada"
    assert route.fast_prefill and route.fused_prefill_scan
    assert route.fused_prefill_self_chunk and route.prefill_graph
    assert route.prefill_graph_cache_size == 4
    assert route.prefill_scan_model_profiles == (
        (1024, 24, 8, 4096, 32768),
        (2048, 24, 8, 4096, 32768),
    )
    assert route.fused_prefill_shift_mix
    assert route.fused_prefill_stacked_rkv
    assert route.fused_prefill_state_prep and route.fused_prefill_output
    assert route.ada_wagv_lora and route.ada_wagv_bmm
    assert not route.ada_linear
    assert not route.ada_sparse_ffn
    assert route.a8w8_gemv_max_rows == 32
    assert route.native_graph_triton_fp16_state
    assert route.native_graph_triton_fp16_state_model_shapes == ((4096, 32, 8),)


def test_rtx4090_policy_maps_quant_graph_sparse_ffn_and_prefill_routes():
    profile, route = policy("NVIDIA GeForce RTX 4090", (8, 9))
    assert profile.family == "ada"
    assert route.fast_prefill and route.fused_prefill_scan
    assert route.fused_prefill_self_chunk and route.prefill_graph
    assert route.fused_prefill_stacked_rkv
    assert route.ada_linear and route.ada_wagv_lora and route.ada_wagv_bmm
    assert route.ada_sparse_ffn and route.ada_sparse_ffn_inplace
    assert route.native_external_quant_prefill
    assert route.native_external_quant_graph
    assert route.native_external_quant_prefill_graph
    assert route.native_bnb8_direct and route.native_bnb8_relu_quant
    assert route.a8w8_gemv_max_rows == 1
    assert route.mm4_fused_max_rows == 16
    assert route.prefill_block_fp16_accum_model_shapes


def test_rtx5090_policy_maps_blackwell_prefill_decode_quant_and_cmix():
    profile, route = policy("NVIDIA GeForce RTX 5090", (12, 0))
    assert profile.family == "blackwell"
    assert route.fused_prefill_scan and route.prefill_graph
    assert route.fused_prefill_shift_mix
    assert route.fused_prefill_state_prep and route.fused_prefill_state_scan
    assert route.fused_prefill_output and route.fused_prefill_residual_gemm
    assert route.fused_prefill_stacked_rkv and route.fused_prefill_sequence_ffn
    assert route.native_graph_state_dtype == "fp16"
    assert route.native_graph_fp16_recurrent and route.prefill_fp16_recurrent
    assert route.native_graph_precompute_embedding
    assert route.ada_linear and route.ada_wagv_lora
    assert route.ada_sparse_ffn and route.blackwell_cmix
    assert route.marlin_w4_ffn_shapes and route.marlin_w4_model_profiles


def test_exact_card_routes_do_not_leak_to_adjacent_product_names():
    _, desktop = policy("NVIDIA GeForce RTX 4080 SUPER", (8, 9))
    assert not desktop.fast_prefill
    assert not desktop.fused_prefill_scan
    assert not desktop.prefill_graph
    assert not desktop.ada_wagv_bmm

    _, laptop = policy("NVIDIA GeForce RTX 4090 Laptop GPU", (8, 9))
    assert not laptop.fast_prefill
    assert not laptop.native_external_quant_graph
    assert not laptop.ada_sparse_ffn

    _, blackwell = policy("NVIDIA GeForce RTX 5080", (12, 0))
    assert not blackwell.fused_prefill_shift_mix
    assert not blackwell.blackwell_cmix
    assert not blackwell.marlin_w4_model_profiles
