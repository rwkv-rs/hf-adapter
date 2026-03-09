from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from conftest import REGISTERED_MARKERS, classify_test_path, validate_marker_set


ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_registers_and_strictly_enforces_every_policy_marker() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pytest_config = config["tool"]["pytest"]["ini_options"]
    assert "--strict-markers" in pytest_config["addopts"]
    registered = {entry.split(":", 1)[0] for entry in pytest_config["markers"]}
    assert registered == REGISTERED_MARKERS


def test_path_policy_assigns_execution_and_hardware_relationships() -> None:
    assert classify_test_path("tests/test_kernel_policy.py") == {"cpu"}
    assert classify_test_path("tests/test_mlx_quant.py") == {"cpu", "apple"}
    assert classify_test_path("tests/test_musa_contract.py") == {"cpu", "musa"}
    assert classify_test_path("tests/test_ascend_runtime.py") == {"cpu", "ascend"}
    assert classify_test_path("tests/test_huawei_ascend_smoke.py") == {
        "cpu",
        "ascend",
    }
    assert classify_test_path("tests/test_metax_runtime.py") == {"cpu", "metax"}
    assert classify_test_path("tests/test_metax_c500_smoke.py") == {
        "cpu",
        "metax",
    }
    assert classify_test_path("tests/test_v100_sm70_mm4_production_gate.py") == {
        "cpu",
        "cuda",
        "sm70",
    }
    assert classify_test_path("tests/test_4080_acceptance_summary.py") == {
        "cpu",
        "cuda",
        "ada",
    }
    assert classify_test_path("tests/test_5090_qwen35_acceptance_summary.py") == {
        "cpu",
        "cuda",
        "blackwell",
    }


def test_model_backed_modules_are_also_slow_and_remain_collectable_offline() -> None:
    markers = classify_test_path("tests/test_hf_training_smoke.py")
    assert {"cpu", "model_required", "slow"} <= markers
    validate_marker_set(markers)
