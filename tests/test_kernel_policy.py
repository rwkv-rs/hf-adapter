#!/usr/bin/env python3
from __future__ import annotations

import os

from rwkv7_hf.kernel_policy import (
    ADAPTATION_RULES,
    adaptation_rule_for_profile,
    classify_gpu,
    detect_gpu_profile,
    env_flag,
    env_int,
    policy_for_profile,
)


def test_gpu_family_classification() -> None:
    cases = [
        ("Tesla P100-PCIE-16GB", (6, 0), "pascal"),
        ("Tesla V100-PCIE-32GB", (7, 0), "volta"),
        ("NVIDIA A800-SXM4-80GB", (8, 0), "ampere"),
        ("NVIDIA RTX A6000", (8, 6), "ampere"),
        ("NVIDIA GeForce RTX 4090", (8, 9), "ada"),
        ("NVIDIA H100 SXM", (9, 0), "hopper"),
        ("NVIDIA GeForce RTX 5070 Laptop GPU", (12, 0), "blackwell"),
        ("NVIDIA GeForce RTX 5090", (12, 0), "blackwell"),
        ("AMD Instinct MI300X", None, "amd_hip"),
        ("Apple M5", None, "apple_mps"),
    ]
    for name, capability, family in cases:
        profile = classify_gpu(
            name,
            capability,
            is_hip=name.startswith("AMD"),
            is_mps=name.startswith("Apple"),
        )
        assert profile.family == family, (name, profile)


def test_policy_defaults_are_conservative() -> None:
    pascal = policy_for_profile(classify_gpu("Tesla P100", (6, 0)))
    assert not pascal.fused_output
    assert not pascal.fused_recurrent_output

    v100 = policy_for_profile(classify_gpu("Tesla V100-PCIE-32GB", (7, 0)))
    assert v100.fused_output
    assert v100.fused_recurrent_output
    assert v100.fused_recurrent_raw
    assert v100.fast_prefill
    assert v100.fused_prefill_scan
    assert v100.prefill_graph
    assert v100.prefill_graph_cache_size == 4
    assert v100.fused_prefill_shift_mix
    assert v100.fused_prefill_state_prep
    assert v100.fused_prefill_state_scan
    assert v100.fused_prefill_state_scan_max_batch == 1
    assert v100.fused_prefill_output
    assert v100.fused_norm_mix
    assert v100.fused_wavg_lora
    assert v100.wavg_lora_bsz1_max_hidden == 4096
    assert v100.wavg_lora_blocks == (32, 64, 256)
    assert v100.wavg_lora_num_warps == 8
    assert v100.sm70_linear
    assert v100.sm70_wagv_lora
    assert v100.ada_sparse_ffn
    assert v100.ada_sparse_ffn_max_rows == 4
    assert v100.ada_sparse_ffn_inplace
    assert not v100.ada_sparse_ffn_up
    assert not v100.fused_projection
    assert not v100.fused_output_project

    ada = policy_for_profile(classify_gpu("NVIDIA GeForce RTX 4090", (8, 9)))
    assert ada.fused_output
    assert ada.fused_recurrent_output
    assert ada.fused_recurrent_raw
    assert ada.fused_norm_mix
    assert ada.fast_prefill
    assert ada.bnb_skip_policy == "memory"
    assert ada.bnb_int8_threshold == 0.0
    assert ada.native_external_quant_prefill
    assert ada.native_external_quant_graph
    assert ada.native_external_quant_prefill_graph
    assert ada.native_bnb8_direct
    assert ada.native_bnb8_relu_quant
    assert ada.native_bnb8_rkv_mix_quant
    assert ada.native_bnb8_ffn_mix_quant
    assert ada.native_bnb8_attn_mix_block == 4096
    assert ada.native_bnb8_ffn_mix_block == 2048
    assert ada.mm4_fused_max_rows == 16
    assert ada.mm4_gemv_block_pairs == 128
    assert ada.mm4_gemv_block_n == 128
    assert ada.mm4_dot_min_rows == 2
    assert ada.mm4_dot_block_b == 16
    assert ada.mm4_dot_block_pairs == 64
    assert ada.mm4_dot_block_n == 64
    assert ada.mm4_dot_warps == 4
    assert ada.prefill_scan_block_m_shapes == ((8, 128, 32),)
    assert ada.prefill_scan_block_m_model_shapes == ((2048, 8, 512, 32),)
    assert ada.prefill_graph
    assert ada.fused_prefill_scan
    assert ada.fused_prefill_state_prep
    assert ada.fused_prefill_output
    assert ada.fused_prefill_shift_mix
    assert not ada.fused_prefill_state_scan
    assert not ada.fused_projection
    assert ada.ada_linear
    assert ada.ada_linear_rows == "1 2 4"
    assert ada.ada_wagv_lora
    assert ada.ada_sparse_ffn
    assert ada.ada_sparse_ffn_max_rows == 2
    assert ada.ada_sparse_ffn_inplace
    assert ada.rkv_policy == "vkwr_auto"
    assert ada.norm_mix_num_warps == 8

    other_ada = policy_for_profile(classify_gpu("NVIDIA GeForce RTX 4070", (8, 9)))
    assert not other_ada.fast_prefill
    assert other_ada.bnb_int8_threshold is None
    assert not other_ada.native_external_quant_prefill
    assert not other_ada.native_external_quant_graph
    assert not other_ada.native_external_quant_prefill_graph
    assert not other_ada.native_bnb8_direct
    assert not other_ada.native_bnb8_relu_quant
    assert not other_ada.native_bnb8_rkv_mix_quant
    assert not other_ada.native_bnb8_ffn_mix_quant
    assert other_ada.mm4_fused_max_rows is None
    assert other_ada.mm4_dot_min_rows is None
    assert other_ada.prefill_scan_block_m_shapes == ()
    assert other_ada.prefill_scan_block_m_model_shapes == ()
    assert not other_ada.prefill_graph
    assert not other_ada.fused_prefill_scan
    assert not other_ada.ada_sparse_ffn
    assert other_ada.rkv_policy == "manual"
    assert other_ada.ada_linear_rows == "2 4"
    assert other_ada.norm_mix_num_warps == 4

    rtx3090 = policy_for_profile(classify_gpu("NVIDIA GeForce RTX 3090", (8, 6)))
    assert rtx3090.fast_prefill
    assert rtx3090.fused_prefill_scan
    assert rtx3090.fused_prefill_self_chunk
    assert rtx3090.prefill_self_chunk_min_tokens == 1024
    assert rtx3090.prefill_self_chunk_size == 32
    assert rtx3090.prefill_graph
    assert rtx3090.prefill_graph_cache_size == 4
    assert rtx3090.bnb_skip_policy == "memory"
    assert rtx3090.bnb_int8_threshold == 0.0
    assert rtx3090.native_external_quant_prefill
    assert rtx3090.native_external_quant_graph
    assert rtx3090.native_external_quant_prefill_graph
    assert rtx3090.native_bnb8_direct
    assert rtx3090.native_bnb8_relu_quant
    assert rtx3090.native_bnb8_rkv_mix_quant
    assert rtx3090.native_bnb8_ffn_mix_quant
    assert rtx3090.native_bnb8_attn_mix_block == 4096
    assert rtx3090.native_bnb8_ffn_mix_block == 2048
    assert rtx3090.a8w8_gemv_max_rows == 8
    assert rtx3090.mm4_fused_max_rows == 16
    assert rtx3090.mm4_gemv_block_pairs == 128
    assert rtx3090.mm4_gemv_block_n == 128
    assert rtx3090.mm4_dot_min_rows == 2
    assert rtx3090.mm4_dot_block_b == 16
    assert rtx3090.mm4_dot_block_pairs == 64
    assert rtx3090.mm4_dot_block_n == 64
    assert rtx3090.mm4_dot_warps == 4
    assert rtx3090.prefill_scan_block_m == 8
    assert rtx3090.prefill_scan_block_m_b2 == 8
    assert rtx3090.prefill_scan_block_m_b4 == 8
    assert rtx3090.prefill_scan_num_warps == 4
    assert rtx3090.prefill_blas_library == "cublaslt"
    assert rtx3090.prefill_blas_large_library == "cublas"
    assert rtx3090.prefill_blas_large_min_rows == 4096
    assert rtx3090.prefill_self_chunk_shape_sizes == (
        (2, 512, 16),
        (2, 2048, 16),
        (8, 128, 16),
    )
    assert rtx3090.prefill_self_chunk_h_tile_shapes == ((4, 2048, 16, 16),)
    assert rtx3090.prefill_self_chunk_model_shapes == (
        (4096, 32, 1, 512),
        (4096, 32, 2, 512),
        (4096, 32, 4, 512),
        (4096, 32, 8, 512),
        (4096, 32, 8, 128),
    )
    assert rtx3090.fused_prefill_shift_mix
    assert rtx3090.fused_prefill_state_prep
    assert rtx3090.fused_prefill_output
    assert rtx3090.fused_prefill_residual_gemm
    assert rtx3090.fused_prefill_stacked_rkv
    assert rtx3090.prefill_stacked_rkv_min_rows == 192
    assert rtx3090.prefill_stacked_rkv_max_rows == 384
    assert rtx3090.prefill_stacked_rkv_extra_rows == ()
    assert rtx3090.prefill_stacked_rkv_shapes == ()
    assert rtx3090.prefill_stacked_rkv_model_shapes == (
        (4096, 32, 1, 512),
        (4096, 32, 2, 512),
        (4096, 32, 4, 512),
        (4096, 32, 4, 128),
    )
    assert rtx3090.fused_prefill_sequence_ffn
    assert rtx3090.prefill_sequence_ffn_min_rows == 192
    assert rtx3090.prefill_sequence_ffn_max_rows == 384
    assert rtx3090.prefill_sequence_ffn_extra_rows == ()
    assert rtx3090.prefill_sequence_ffn_model_shapes == (
        (4096, 32, 2, 2048),
        (4096, 32, 8, 512),
    )
    assert rtx3090.prefill_sequence_ffn_blocks == (64, 64, 32, 64, 8)
    assert rtx3090.prefill_sequence_ffn_large_blocks == (128, 128, 32, 64, 8)
    assert rtx3090.prefill_sequence_ffn_num_stages == 4
    assert rtx3090.prefill_sequence_ffn_num_warps == 8
    assert not rtx3090.fused_prefill_state_scan

    a6000 = policy_for_profile(classify_gpu("NVIDIA RTX A6000", (8, 6)))
    assert not a6000.fast_prefill
    assert not a6000.fused_prefill_scan
    assert not a6000.fused_prefill_self_chunk
    assert not a6000.prefill_graph
    assert a6000.bnb_skip_policy == "memory"
    assert a6000.bnb_int8_threshold is None
    assert not a6000.native_external_quant_prefill
    assert not a6000.native_external_quant_graph
    assert not a6000.native_bnb8_direct
    assert not a6000.native_bnb8_relu_quant
    assert not a6000.native_bnb8_rkv_mix_quant
    assert not a6000.native_bnb8_ffn_mix_quant
    assert a6000.prefill_scan_block_m is None
    assert a6000.prefill_scan_block_m_b2 is None
    assert a6000.prefill_scan_block_m_b4 is None
    assert not a6000.fused_prefill_sequence_ffn
    assert not a6000.fused_prefill_stacked_rkv
    assert a6000.prefill_blas_library is None

    blackwell = policy_for_profile(classify_gpu("NVIDIA GeForce RTX 5090", (12, 0)))
    assert blackwell.fused_output
    assert blackwell.fused_recurrent_output
    assert not blackwell.fused_projection
    assert blackwell.prefill_graph
    assert blackwell.prefill_fp16_recurrent
    assert (4096, 61, 8, 2048) not in blackwell.prefill_graph_model_shapes
    assert (4096, 61, 8, 512) in blackwell.prefill_graph_model_shapes
    assert blackwell.native_graph_state_dtype == "fp16"
    assert blackwell.native_graph_fp16_recurrent
    assert blackwell.native_graph_precompute_embedding
    assert blackwell.fused_norm_mix
    assert blackwell.norm_mix_num_warps == 8
    assert blackwell.ada_linear
    assert blackwell.ada_linear_rows == "1"
    assert blackwell.ada_linear_roles == "hidden,ffn_up,ffn_down"
    assert blackwell.ada_wagv_lora
    assert blackwell.ada_wag_lora
    assert blackwell.ada_sparse_ffn
    assert blackwell.ada_sparse_ffn_low_memory_pack
    assert blackwell.ada_sparse_ffn_share_pack
    assert blackwell.ada_sparse_ffn_deterministic_splits == 4
    assert blackwell.ada_sparse_ffn_official_boundary
    assert blackwell.blackwell_cmix
    assert blackwell.prefill_scan_block_m_model_shapes == ((2048, 8, 512, 8),)
    assert blackwell.fused_prefill_shift_mix
    assert blackwell.prefill_shift_mix_model_shapes == (
        (2048, 24, 8, 128),
        (2048, 24, 8, 512),
        (2048, 24, 8, 2048),
        (4096, 61, 1, 128),
        (4096, 61, 1, 512),
        (4096, 61, 1, 2048),
        (4096, 61, 8, 128),
        (4096, 61, 8, 512),
        (4096, 61, 8, 2048),
    )
    assert blackwell.prefill_attn_shift_mix_strict_fp16_model_shapes == (
        (4096, 61, 1, 128),
        (4096, 61, 1, 512),
        (4096, 61, 1, 2048),
        (4096, 61, 8, 128),
    )
    assert blackwell.prefill_ffn_shift_mix_strict_fp16_model_shapes == (
        (4096, 61, 1, 128),
    )
    assert blackwell.prefill_attn_shift_mix_launch_profiles[-1] == (
        4096, 61, 8, 2048, 2048, 8
    )
    assert blackwell.prefill_ffn_shift_mix_launch_profiles[-1] == (
        4096, 61, 8, 2048, 2048, 8
    )
    assert blackwell.fused_prefill_state_prep
    assert blackwell.prefill_state_prep_model_shapes == (
        (2048, 24, 8, 512),
        (2048, 24, 8, 2048),
        (4096, 61, 1, 128),
        (4096, 61, 1, 512),
        (4096, 61, 1, 2048),
        (4096, 61, 8, 128),
        (4096, 61, 8, 512),
        (4096, 61, 8, 2048),
    )
    assert blackwell.prefill_state_prep_layer_counts == (
        (2048, 24, 8, 512, 24),
        (2048, 24, 8, 2048, 18),
    )
    assert not blackwell.fused_prefill_clampw_scan
    assert blackwell.prefill_clampw_scan_model_shapes == ((2048, 24, 8, 512),)
    assert blackwell.fused_prefill_residual_gemm
    assert blackwell.fused_prefill_stacked_rkv
    assert blackwell.prefill_stacked_rkv_min_rows == 1
    assert blackwell.prefill_stacked_rkv_max_rows == 1
    assert blackwell.prefill_stacked_rkv_model_shapes == (
        (4096, 32, 8, 128),
    )
    assert blackwell.fused_prefill_sequence_ffn
    assert blackwell.prefill_sequence_ffn_min_rows == 1
    assert blackwell.prefill_sequence_ffn_max_rows == 1
    assert blackwell.prefill_sequence_ffn_model_shapes == (
        (2048, 24, 8, 128),
        (2048, 24, 8, 512),
        (2048, 24, 8, 2048),
    )
    assert blackwell.prefill_sequence_ffn_large_blocks == (64, 128, 32, 64, 8)
    assert blackwell.prefill_sequence_ffn_num_stages == 3
    assert blackwell.prefill_sequence_ffn_num_warps == 8
    assert blackwell.prefill_fp16_accum_ffn_key_model_shapes == (
        (2560, 32, 8, 128),
        (4096, 32, 8, 128),
        (4096, 61, 1, 128),
    )
    assert blackwell.prefill_fp16_accum_ffn_key_layer_counts == (
        (2560, 32, 8, 128, 28),
        (4096, 61, 1, 128, 12),
    )
    assert blackwell.marlin_w4_ffn_shapes == (
        (8192, 2048),
        (2048, 8192),
        (10240, 2560),
        (2560, 10240),
        (16384, 4096),
        (4096, 16384),
    )
    assert blackwell.marlin_w4_model_profiles == (
        (2048, 8192, 24, 128, False, 1),
        (2560, 10240, 32, 128, False, 0),
        (4096, 16384, 32, 128, True, 0),
        (4096, 16384, 61, 128, True, 1),
    )
    assert "triton_compat" in blackwell.notes
    other_blackwell = policy_for_profile(
        classify_gpu("NVIDIA GeForce RTX 5070 Laptop GPU", (12, 0))
    )
    assert other_blackwell.prefill_scan_block_m_model_shapes == ()
    assert other_blackwell.prefill_clampw_scan_model_shapes == ()
    assert not other_blackwell.fused_prefill_stacked_rkv
    assert not other_blackwell.fused_prefill_sequence_ffn
    assert not other_blackwell.prefill_graph
    assert other_blackwell.prefill_graph_model_shapes == ()
    assert other_blackwell.native_graph_state_dtype == "fp32"
    assert not other_blackwell.native_graph_fp16_recurrent
    assert not other_blackwell.ada_sparse_ffn_low_memory_pack
    assert other_blackwell.prefill_fp16_accum_ffn_key_model_shapes == ()
    assert other_blackwell.prefill_fp16_accum_ffn_key_layer_counts == ()
    assert other_blackwell.marlin_w4_ffn_shapes == ()
    assert other_blackwell.marlin_w4_model_profiles == ()

    apple = policy_for_profile(classify_gpu("Apple M5", None, is_mps=True))
    assert apple.profile.family == "apple_mps"
    assert apple.fast_token_backend == "native"
    assert not apple.fused_output
    assert not apple.fused_recurrent_output


def test_mps_runtime_detection() -> None:
    class FakeMPS:
        @staticmethod
        def is_available():
            return True

    class FakeBackends:
        mps = FakeMPS()

    class FakeTorch:
        backends = FakeBackends()

    profile = detect_gpu_profile(torch_module=FakeTorch())
    assert profile.family == "apple_mps"
    assert profile.vendor == "apple"
    assert profile.is_mps


def test_every_policy_family_has_an_adaptation_rule() -> None:
    cases = [
        classify_gpu(None, None),
        classify_gpu("old cuda", (5, 2)),
        classify_gpu("Tesla P100", (6, 0)),
        classify_gpu("Tesla V100-PCIE-32GB", (7, 0)),
        classify_gpu("NVIDIA T4", (7, 5)),
        classify_gpu("NVIDIA A100-SXM4-80GB", (8, 0)),
        classify_gpu("NVIDIA A800-SXM4-80GB", (8, 0)),
        classify_gpu("NVIDIA RTX A6000", (8, 6)),
        classify_gpu("NVIDIA GeForce RTX 4090", (8, 9)),
        classify_gpu("NVIDIA H100 SXM", (9, 0)),
        classify_gpu("NVIDIA GeForce RTX 5070 Laptop GPU", (12, 0)),
        classify_gpu("NVIDIA GeForce RTX 5090", (12, 0)),
        classify_gpu("AMD Instinct MI300X", None, is_hip=True),
        classify_gpu("Apple M5", None, is_mps=True),
    ]
    for profile in cases:
        rule = adaptation_rule_for_profile(profile)
        assert rule.family == profile.family, (profile, rule)
        assert rule.required_functional
        assert rule.required_benchmarks
        assert rule.promotion_rule

    # The registry is intentionally broader than the live test cases because it
    # also documents unvalidated fallback families.
    for family in ("unknown_cuda", "legacy_cuda", "pascal", "volta", "ada", "blackwell", "amd_hip", "apple_mps"):
        assert family in ADAPTATION_RULES
    assert any("A6000" in card for card in ADAPTATION_RULES["ampere"].cards)


def test_env_helpers_override_defaults() -> None:
    old = os.environ.get("RWKV7_TEST_FLAG")
    old_int = os.environ.get("RWKV7_TEST_INT")
    try:
        os.environ.pop("RWKV7_TEST_FLAG", None)
        assert env_flag("RWKV7_TEST_FLAG", True)
        assert not env_flag("RWKV7_TEST_FLAG", False)
        os.environ["RWKV7_TEST_FLAG"] = "0"
        assert not env_flag("RWKV7_TEST_FLAG", True)
        os.environ["RWKV7_TEST_FLAG"] = "1"
        assert env_flag("RWKV7_TEST_FLAG", False)
        os.environ["RWKV7_TEST_FLAG"] = "TRUE"
        assert env_flag("RWKV7_TEST_FLAG", False)

        os.environ["RWKV7_TEST_INT"] = "999"
        assert env_int("RWKV7_TEST_INT", 16, lower=1, upper=128) == 128
        os.environ["RWKV7_TEST_INT"] = "bad"
        assert env_int("RWKV7_TEST_INT", 16, lower=1, upper=128) == 16
    finally:
        if old is None:
            os.environ.pop("RWKV7_TEST_FLAG", None)
        else:
            os.environ["RWKV7_TEST_FLAG"] = old
        if old_int is None:
            os.environ.pop("RWKV7_TEST_INT", None)
        else:
            os.environ["RWKV7_TEST_INT"] = old_int


def main() -> int:
    test_gpu_family_classification()
    test_policy_defaults_are_conservative()
    test_mps_runtime_detection()
    test_every_policy_family_has_an_adaptation_rule()
    test_env_helpers_override_defaults()
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
