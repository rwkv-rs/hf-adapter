from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from examples import generate
from scripts.adapter_manifest import ADAPTER_FILES


ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "47322bfaffc2e662fa989863c3fda4d74f02fc32"


def test_biren_runtime_is_in_converted_checkpoint_manifest() -> None:
    assert "biren_runtime.py" in ADAPTER_FILES
    facade = (ROOT / "rwkv7_hf" / "native_model.py").read_text(encoding="utf-8")
    assert "_native_biren_runtime_dependency_sentinel" in facade
    runtime = (ROOT / "rwkv7_hf" / "model_runtime.py").read_text(encoding="utf-8")
    assert 'weight.device.type == "supa"' in runtime


def test_public_package_exposes_import_safe_biren_api() -> None:
    import rwkv7_hf

    for name in (
        "biren_available",
        "biren_runtime_policy",
        "configure_biren_defaults",
        "enable_biren",
        "biren_memory_stats",
        "biren_synchronize",
        "validate_biren_model_dtype",
    ):
        assert callable(getattr(rwkv7_hf, name))


def test_generate_auto_selects_biren_and_forces_bf16(monkeypatch) -> None:
    biren_device = SimpleNamespace(type="supa")
    monkeypatch.setattr(generate, "_biren_available", lambda: True)
    monkeypatch.setattr(generate, "_enable_biren", lambda: biren_device)
    assert generate.resolve_device("auto") is biren_device
    assert generate.resolve_device("biren") is biren_device
    assert generate.resolve_dtype("auto", biren_device) is torch.bfloat16
    with pytest.raises(ValueError, match="BR106M.*bfloat16"):
        generate.resolve_dtype("fp16", biren_device)


def test_docs_pin_source_and_keep_current_main_rerun_open() -> None:
    document = (ROOT / "docs" / "hardware" / "BIREN_BR106M.md").read_text(
        encoding="utf-8"
    )
    assert SOURCE_COMMIT in document
    assert "current-main" in document
    assert "not a performance" in document
