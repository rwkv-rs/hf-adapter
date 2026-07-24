import importlib
import os
import sys
from types import SimpleNamespace

import pytest


def test_import_does_not_require_torch_npu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch_npu", None)
    module = importlib.reload(importlib.import_module("rwkv7_hf.ascend_runtime"))
    assert module.import_torch_npu(required=False) is None


def test_configure_ascend_defaults_is_fail_closed(monkeypatch):
    from rwkv7_hf.ascend_runtime import configure_ascend_defaults

    for name in list(os.environ):
        if name.startswith("RWKV7_"):
            monkeypatch.delenv(name, raising=False)
    values = configure_ascend_defaults()
    assert values["RWKV7_NATIVE_MODEL"] == "1"
    assert values["RWKV7_NATIVE_MODEL_BACKEND"] == "eager"
    assert values["RWKV7_NATIVE_MODEL_JIT"] == "0"
    assert values["RWKV7_NATIVE_GRAPH"] == "0"
    assert values["RWKV7_NATIVE_PREFILL_GRAPH"] == "0"


def test_configure_preserves_explicit_override(monkeypatch):
    from rwkv7_hf.ascend_runtime import configure_ascend_defaults

    monkeypatch.setenv("RWKV7_NATIVE_MODEL_BACKEND", "native_jit")
    values = configure_ascend_defaults(backend="eager")
    assert values["RWKV7_NATIVE_MODEL_BACKEND"] == "native_jit"


def test_rejects_cuda_graph_backend():
    from rwkv7_hf.ascend_runtime import configure_ascend_defaults

    with pytest.raises(ValueError, match="eager, native_jit, or auto"):
        configure_ascend_defaults(backend="native_graph")


def test_exact_stack_validation_has_no_substring_or_version_family_matching():
    from rwkv7_hf.ascend_runtime import normalize_ascend_device_name, validate_ascend_stack

    exact = dict(
        device_name="Ascend 910B3",
        torch_version="2.9.0+cpu",
        torch_npu_version="2.9.0",
        cann_version="8.5.0",
    )
    assert normalize_ascend_device_name("Ascend 910B3") == "Ascend910B3"
    assert validate_ascend_stack(**exact)[0]
    assert not validate_ascend_stack(**{**exact, "device_name": "FakeAscend910B3X"})[0]
    assert not validate_ascend_stack(**{**exact, "device_name": "Ascend910B4"})[0]
    assert not validate_ascend_stack(**{**exact, "cann_version": "8.5.0.1"})[0]
    assert not validate_ascend_stack(**{**exact, "torch_npu_version": "2.9.0.post1"})[0]


def _patch_fake_available_runtime(monkeypatch, runtime, *, device_name, torch_npu_version="2.9.0"):
    import torch

    selected = []
    fake_npu = SimpleNamespace(
        device_count=lambda: 1,
        get_device_name=lambda index: device_name,
        set_device=lambda device: selected.append(device),
    )
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    monkeypatch.setattr(torch, "__version__", "2.9.0+cpu")
    monkeypatch.setattr(runtime, "import_torch_npu", lambda required=False: SimpleNamespace())
    monkeypatch.setattr(runtime, "ascend_available", lambda: True)
    monkeypatch.setattr(
        runtime,
        "_package_version",
        lambda name: torch_npu_version if name == "torch-npu" else None,
    )
    monkeypatch.setenv("ASCEND_TOOLKIT_VERSION", "8.5.0")
    monkeypatch.delenv("RWKV7_ALLOW_UNVALIDATED_ASCEND", raising=False)
    return selected


def test_enable_ascend_accepts_only_exact_validated_stack(monkeypatch):
    runtime = importlib.import_module("rwkv7_hf.ascend_runtime")
    selected = _patch_fake_available_runtime(
        monkeypatch, runtime, device_name="Ascend 910B3"
    )
    info = runtime.enable_ascend("npu:0")
    assert info.validated_stack and info.validation_status == "validated"
    assert not info.allow_unvalidated
    assert selected == ["npu:0"]


def test_enable_ascend_rejects_unknown_card_before_set_device(monkeypatch):
    runtime = importlib.import_module("rwkv7_hf.ascend_runtime")
    selected = _patch_fake_available_runtime(
        monkeypatch, runtime, device_name="Vendor-Ascend910B3-compatible"
    )
    with pytest.raises(RuntimeError, match="unvalidated Huawei Ascend production stack"):
        runtime.enable_ascend("npu:0")
    assert selected == []


def test_enable_ascend_explicit_override_is_reported_unvalidated(monkeypatch):
    runtime = importlib.import_module("rwkv7_hf.ascend_runtime")
    selected = _patch_fake_available_runtime(
        monkeypatch,
        runtime,
        device_name="Ascend910B4",
        torch_npu_version="2.9.0.post1",
    )
    monkeypatch.setenv("RWKV7_ALLOW_UNVALIDATED_ASCEND", "1")
    info = runtime.enable_ascend("npu:0")
    assert not info.validated_stack
    assert info.validation_status == "unvalidated_override"
    assert info.allow_unvalidated
    assert "device_name" in info.validation_reason and "torch_npu_version" in info.validation_reason
    assert selected == ["npu:0"]
