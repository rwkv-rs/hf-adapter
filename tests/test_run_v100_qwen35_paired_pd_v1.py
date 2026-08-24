from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def text(name: str) -> str:
    path = ROOT / "bench" / name
    if not path.exists():
        path = ROOT / "bench" / "validators" / name
    return path.read_text(encoding="utf-8")


def test_v100_qwen_runner_locks_sm70_fla_triton_raw_graph() -> None:
    script = text("run_v100_qwen35_best_optimized_hf_v1.sh")
    assert "TORCH_CUDA_ARCH_LIST=7.0" in script
    assert '--exact-name "Tesla V100-PCIE-32GB"' in script
    assert "--benchmark-matrix qwen35_v100_best_optimized_hf_v1" in script
    assert "--qwen-conv-backend fla_triton" in script
    assert "static_cache_raw_cudagraph" in script
    assert '--qwen-sdpa-policy "${QWEN_SDPA_POLICY}"' in script
    assert '"sdpa_policy": sdpa_policy' in script
    assert '"triton": "3.4.0"' in script
    assert '"causal_conv1d": None' in script
    assert "PYTHONPATH=${ROOT}:${TRITON_TARGET}:${FLA_TARGET}" in script


def test_v100_rwkv_runner_covers_four_pairs_and_exact_routes() -> None:
    script = text("run_v100_rwkv_paired_pd_v1.sh")
    assert "RWKV_72_MODEL" in script
    assert "rwkv-7.2b__qwen3.5-9b" in script
    assert '"candidate_rows":48' in script
    assert '"models":4' in script
    assert '"entries":8' in script
    assert "RWKV7_NATIVE_GRAPH_RKV_POLICY=${rkv_policy}" in script
    assert (
        "local ada_require_extension=0 sm70_require_extension=0 rkv_policy=manual"
        in script
    )
    assert "sm70_require_extension=1" in script
    assert (
        "RWKV7_NATIVE_GRAPH_SM70_WAGV_LORA_REQUIRE_EXTENSION=${sm70_require_extension}"
        in script
    )
    assert '"${tag}" == "7p2" && "${batch}" == "8"' in script
    assert "rkv_policy=vkwr_auto" in script
    assert "norm_mix_warps=8" in script
    assert "RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX_NUM_WARPS=${norm_mix_warps}" in script
    assert "RWKV7_NATIVE_GRAPH_FUSED_WAVG_LORA=${fused_wavg_lora}" in script
    assert "fused_wavg_lora=0" in script
    assert '"RWKV7_NATIVE_GRAPH_RKV_POLICY":"per_lane_exact_v100_policy"' in script
    assert "RWKV7_NATIVE_GRAPH_ADA_WAGV_BMM=0" in script
    assert '--probe-tokens 512 --probe-batch-size "${batch}"' in script
    assert "rwkv_native_graph_fla_correctness_v100_v1" in script


def test_v100_top_level_is_append_never_and_includes_9b() -> None:
    script = text("run_v100_qwen35_paired_pd_v1.sh")
    assert "formal OUT_DIR and CACHE_ROOT must both be absent" in script
    assert "QWEN_9_MODEL" in script
    assert "qwen_9b.jsonl" in script
    assert "rwkv-7.2b__qwen3.5-9b" in script
    assert "qwen3.5-9b 9b static_cache_raw_cudagraph math_only" in script
    assert script.count("static_cache_raw_cudagraph auto") == 3
    assert "validate_qwen35_v100_paired_pd_v1.py" in script
    assert "FROZEN_QWEN_DIR" in script
    assert "FROZEN_QWEN_REFERENCE_SHA256" in script
    assert "frozen Qwen reference SHA mismatch" in script


def test_exact_gpu_helper_supports_literal_product_names() -> None:
    helper = text("check_exact_gpu.py")
    assert 'product.add_argument("--exact-name"' in helper
    assert "matches_gpu_product" in helper
