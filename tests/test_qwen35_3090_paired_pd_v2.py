from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "bench" / "validate_qwen35_3090_paired_pd_v2.py"
QWEN_RUNNER_PATH = ROOT / "bench" / "run_5090_qwen35_best_optimized_hf.sh"


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_3090_pd", VALIDATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _contract(module) -> dict:
    return {
        "schema_version": 1,
        "protocol": module.CONTRACT_PROTOCOL,
        "device": module.EXPECTED_DEVICE,
        "gpu_arch": "sm_86",
        "compute_cap": "8.6",
        "driver_version": "550.142",
        "memory_total_mib": 24576,
        "runtime": module.EXPECTED_RUNTIME,
        "torch_cuda_arch_list": "8.6",
        "reference_sha256": "a" * 64,
        "routes_by_pair": {
            pair: "static_cache_raw_cudagraph" for pair in module.base.PAIRS
        },
    }


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


def test_3090_validator_disables_4090_sm120_bundle() -> None:
    source = VALIDATOR_PATH.read_text(encoding="utf-8")
    assert "base.SPECIAL_SMALL_B8_BUNDLE = False" in source
    assert "base.BASE_ADA_WAGV_BMM_EXPECTED = False" in source
    assert 'base.EXPECTED_ARCH = "8.6"' in source
    assert 'base.EXPECTED_MEMORY = "24576 MiB"' in source


def test_qwen_formal_runner_accepts_an_exact_card_override() -> None:
    source = QWEN_RUNNER_PATH.read_text(encoding="utf-8")
    assert 'EXPECTED_GPU_MODEL="${EXPECTED_GPU_MODEL:-5090}"' in source
    assert '--model "${EXPECTED_GPU_MODEL}"' in source
