from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from rwkv7_hf import metax_runtime
from rwkv7_hf.model_runtime_policy import native_model_jit_enabled


EXACT_MX_SMI = """
mx-smi  version: 2.2.12
Kernel Mode Driver Version: 3.8.30
MACA Version: 3.5.3.20
Board Name | MetaX C500
"""


def test_metax_name_normalization_is_exact() -> None:
    assert metax_runtime.normalize_metax_device_name("MetaX C500") == "MetaX C500"
    assert metax_runtime.is_metax_c500_name("metax-c500")
    assert not metax_runtime.is_metax_c500_name("MetaX C5000")
    assert not metax_runtime.is_metax_c500_name("NVIDIA C500")


def test_metax_stack_validation_is_fail_closed() -> None:
    valid, reason = metax_runtime.validate_metax_stack(
        device_name="MetaX C500",
        torch_version="2.8.0+metax3.5.3.9",
        torch_cuda_version="11.6",
        mxmaca_version="3.5.3.20",
    )
    assert valid
    assert "exact validated" in reason

    valid, reason = metax_runtime.validate_metax_stack(
        device_name="MetaX C5000",
        torch_version="2.8.0+metax3.5.3.9",
        torch_cuda_version="11.6",
        mxmaca_version="3.5.3.20",
    )
    assert not valid
    assert "device_name" in reason


def test_metax_metadata_parser_uses_redacted_fixed_command_output() -> None:
    assert metax_runtime.detect_mxmaca_version(output=EXACT_MX_SMI) == "3.5.3.20"
    assert metax_runtime.detect_metax_driver_version(output=EXACT_MX_SMI) == "3.8.30"
    assert metax_runtime.detect_mx_smi_version(output=EXACT_MX_SMI) == "2.2.12"


def test_metax_defaults_preserve_explicit_user_values(monkeypatch) -> None:
    monkeypatch.delenv("RWKV7_NATIVE_MODEL_BACKEND", raising=False)
    monkeypatch.setenv("RWKV7_FAST_CACHE", "1")
    values = metax_runtime.configure_metax_defaults()
    assert values["RWKV7_NATIVE_MODEL_BACKEND"] == "eager"
    assert values["RWKV7_NATIVE_MODEL_JIT"] == "0"
    assert values["RWKV7_FAST_CACHE"] == "1"

    values = metax_runtime.configure_metax_defaults(overwrite=True)
    assert values["RWKV7_FAST_CACHE"] == "0"


def test_metax_default_backend_disables_jit_without_overriding_user_env(
    monkeypatch,
) -> None:
    def metax_policy():
        return SimpleNamespace(profile=SimpleNamespace(family="metax"))

    monkeypatch.delenv("RWKV7_NATIVE_MODEL_JIT", raising=False)
    assert not native_model_jit_enabled(kernel_policy_fn=metax_policy)
    monkeypatch.setenv("RWKV7_NATIVE_MODEL_JIT", "1")
    assert native_model_jit_enabled(kernel_policy_fn=metax_policy)


def test_enable_metax_reports_exact_validated_stack(monkeypatch) -> None:
    selected: list[int] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "MetaX C500")
    monkeypatch.setattr(
        torch.cuda, "set_device", lambda index: selected.append(int(index))
    )
    monkeypatch.setattr(torch, "__version__", "2.8.0+metax3.5.3.9")
    monkeypatch.setattr(torch.version, "cuda", "11.6", raising=False)
    monkeypatch.setattr(metax_runtime, "_mx_smi_output", lambda: EXACT_MX_SMI)
    monkeypatch.delenv("RWKV7_ALLOW_UNVALIDATED_METAX", raising=False)

    info = metax_runtime.enable_metax("metax:0")
    assert info.available
    assert info.validated_stack
    assert info.validation_status == "validated"
    assert info.device == "cuda:0"
    assert info.mxmaca_version == "3.5.3.20"
    assert info.driver_version == "3.8.30"
    assert selected == [0]


def test_enable_metax_rejects_adjacent_product(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _index: "MetaX C5000")
    with pytest.raises(RuntimeError, match="expected exact MetaX C500"):
        metax_runtime.enable_metax(0)
