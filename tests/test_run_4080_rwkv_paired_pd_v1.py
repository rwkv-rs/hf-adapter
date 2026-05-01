from __future__ import annotations

from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1] / "bench" / "run_4080_rwkv_paired_pd_v1.sh"
).read_text(encoding="utf-8")


def test_formal_runner_locks_paths_runtime_and_append_never() -> None:
    for text in (
        "formal OUT_DIR must not already exist",
        "CACHE_ROOT must be absent or empty",
        "status --porcelain --untracked-files=all",
        "3.12.2",
        "2.11.0+cu130",
        "TORCH_CUDA_ARCH_LIST=8.9",
        "CUDA_COMPONENT_INCLUDE lacks cusparse.h",
        '"CPATH=${CUDA_COMPONENT_INCLUDE}"',
        "TORCHINDUCTOR_CACHE_DIR",
        "--model 4080 --name",
        "model_hashes.after.sha256",
        "every recursive regular file",
    ):
        assert text in SCRIPT


def test_formal_runner_captures_complete_matrix_and_long_correctness() -> None:
    for text in (
        "--cells \"${batch}x128x128\"",
        "\"${batch}x2048x512\"",
        "--warmup 3 --runs 7",
        "--probe-cell \"${batch}x2048x512\"",
        "--probe-tokens 512",
        "--required-probe-tokens 512",
        "--require-distinct-batch-prompts",
        "baseline_fresh_gpu_processes",
        "candidate_formal_lane_processes",
        '"TORCHDYNAMO_DISABLE=1"',
        '"TORCH_COMPILE_DISABLE=1"',
        '"performance_role":False',
    ):
        assert text in SCRIPT


def test_formal_runner_locks_exact_ada_routes_without_sm120_pollution() -> None:
    assert '"RWKV7_NATIVE_GRAPH_ADA_WAGV_LORA_REQUIRE_EXTENSION=${require_extension}"' in SCRIPT
    assert '"RWKV7_NATIVE_GRAPH_RKV_POLICY=${rkv_policy}"' in SCRIPT
    assert 'if [[ "${batch}" == 1 ]]; then require_extension=1; fi' in SCRIPT
    assert 'if [[ "${batch}" == 1 || "${tag}" == 0p4 || "${tag}" == 1p5 ]]' in SCRIPT
    assert '"RWKV7_NATIVE_GRAPH_ADA_WAGV_BMM=1"' in SCRIPT
    assert '"RWKV7_NATIVE_GRAPH_SM120_WAGV_BMM_G=0"' in SCRIPT
    assert '"RWKV7_NATIVE_GRAPH_SM120_COMPILED_FFN=0"' in SCRIPT
    assert '"RWKV7_NATIVE_PREFILL_GRAPH":"unset_exact_card_policy"' in SCRIPT
