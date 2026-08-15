from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "bench" / "validate_qwen35_4090_paired_pd_v2.py"
RUNNER_PATH = ROOT / "bench" / "run_4090_rwkv_paired_pd_v2.sh"
RUNNER_3090_PATH = ROOT / "bench" / "run_3090_rwkv_paired_pd_v2.sh"


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_4090_pd", VALIDATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_4090_validator_contract_is_four_pairs_and_48_cells() -> None:
    module = _load_validator()
    assert module.PROTOCOL == "qwen35_4090_paired_pd_v2"
    assert module.EXPECTED_DEVICE == "NVIDIA GeForce RTX 4090"
    assert len(module.PAIRS) == 4
    assert len(module.EXPECTED_KEYS) == 48
    assert set(module.QWEN_ROUTES.values()) == {
        "static_cache_inductor_cudagraph",
        "static_cache_raw_cudagraph",
    }


def test_4090_validator_strict_types_reject_bool_as_integer() -> None:
    module = _load_validator()
    assert module._strict(1, 1)
    assert not module._strict(True, 1)
    assert not module._strict([False, 1], [0, 1])


def test_4090_runner_locks_commit_runtime_and_small_b8_bundle() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    assert "validate_repository" in source
    assert 'python_dir="$(realpath -e -- "$(dirname -- "${PYTHON_BIN}")")"' in source
    assert 'PYTHON_BIN="${python_dir}/$(basename -- "${PYTHON_BIN}")"' in source
    assert 'EXPECTED_PYTHON="${EXPECTED_PYTHON:-3.12.8}"' in source
    assert '"torch":"2.7.1+cu126"' in source
    assert 'TORCH_CUDA_ARCH="${TORCH_CUDA_ARCH:-8.9}"' in source
    assert '"TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH}"' in source
    assert '"RWKV7_NATIVE_GRAPH_ADA_WAGV_BMM=1"' in source
    assert '"RWKV7_NATIVE_GRAPH_SM120_WAGV_BMM_G=1"' in source
    assert '"RWKV7_NATIVE_GRAPH_SM120_COMPILED_FFN=1"' in source
    assert '"TORCH_COMPILE_DISABLE=1" "TORCHDYNAMO_DISABLE=1"' in source
    assert '--probe-tokens 512 --probe-batch-size "${batch}"' in source


def test_3090_runner_locks_ampere_contract_and_disables_foreign_routes() -> None:
    source = RUNNER_3090_PATH.read_text(encoding="utf-8")
    assert "qwen35_3090_paired_pd_v2" in source
    assert 'EXPECTED_GPU_NAME="NVIDIA GeForce RTX 3090"' in source
    assert "EXPECTED_PYTHON=3.10.12" in source
    assert "TORCH_CUDA_ARCH=8.6" in source
    assert "SMALL_B8_MODE=base" in source
    assert "SPLIT_7P2_B8=0" in source
    assert "ADA_WAGV_BMM_OVERRIDE=0" in source


def test_4090_validator_requires_strict_unrounded_four_axis_pass() -> None:
    source = VALIDATOR_PATH.read_text(encoding="utf-8")
    assert 'for axis in ("prefill", "decode")' in source
    assert "gates.extend((raw > 1.0, adjusted > 1.0))" in source
    assert (
        "all 48 cells must strictly pass raw and adjusted Prefill and Decode" in source
    )


def test_4090_correctness_validator_uses_flat_probe_comparison_schema() -> None:
    source = VALIDATOR_PATH.read_text(encoding="utf-8")
    assert 'result.get(f"{axis}_cosine")' in source
    assert 'result.get(f"{axis}_shape_match")' in source
    assert 'result.get(f"{axis}_finite")' in source
