from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from rwkv7_hf import biren_runtime
from rwkv7_hf.model_runtime_policy import native_model_jit_enabled
from rwkv7_hf.native import _decomposed_group_norm


EXACT_BRSMI = """
BR-SMI 1.11.0
Driver Version: 1.11.0
SUPA Version: 1.11
"""


def test_biren_name_normalization_is_exact() -> None:
    assert biren_runtime.normalize_biren_device_name("Biren106M") == "Biren106M"
    assert biren_runtime.is_biren_br106m_name("Biren-106M")
    assert not biren_runtime.is_biren_br106m_name("Biren106M2")
    assert not biren_runtime.is_biren_br106m_name("BR106M")


def test_biren_policy_defaults_to_bfloat16_eager() -> None:
    policy = biren_runtime.biren_runtime_policy(None)
    assert policy.device_type == "supa"
    assert policy.dtype is torch.bfloat16
    assert policy.backend == "eager"
    assert not policy.compile_enabled


def test_biren_policy_rejects_fp16_before_dispatch() -> None:
    with pytest.raises(biren_runtime.BirenDTypeError, match="BR106M.*bfloat16"):
        biren_runtime.validate_biren_model_dtype(
            torch.float16,
            device_type="supa",
        )
    assert (
        biren_runtime.validate_biren_model_dtype(
            torch.float16,
            device_type="cuda",
        )
        is torch.float16
    )


def test_biren_forward_validation_checks_model_and_input_devices() -> None:
    with pytest.raises(biren_runtime.BirenDTypeError, match="BR106M.*bfloat16"):
        biren_runtime.validate_biren_forward_dtype(
            torch.bfloat16,
            input_device="supa:0",
            model_device="supa:0",
            model_dtype=torch.float16,
        )


def test_biren_stack_validation_and_brsmi_parser() -> None:
    versions = biren_runtime.detect_biren_driver_versions(output=EXACT_BRSMI)
    assert versions == {"brsmi": "1.11.0", "driver": "1.11.0", "supa": "1.11"}
    valid, reason = biren_runtime.validate_biren_stack(
        device_name="Biren106M",
        visible_devices=1,
        torch_version="2.9.0+cu128",
        torch_br_version="1.10.0.20900+br1xx",
        sdk_version="1.11.0.0.rc2",
        driver_version="1.11.0",
        supa_version="1.11",
    )
    assert valid
    assert "exact validated" in reason

    valid, reason = biren_runtime.validate_biren_stack(
        device_name="Biren106M2",
        visible_devices=1,
        torch_version="2.9.0+cu128",
        torch_br_version="1.10.0.20900+br1xx",
        sdk_version="1.11.0.0.rc2",
        driver_version="1.11.0",
        supa_version="1.11",
    )
    assert not valid
    assert "device_name" in reason

    valid, reason = biren_runtime.validate_biren_stack(
        device_name="Biren106M",
        visible_devices=2,
        torch_version="2.9.0+cu128",
        torch_br_version="1.10.0.20900+br1xx",
        sdk_version="1.11.0.0.rc2",
        driver_version="1.11.0",
        supa_version="1.11",
    )
    assert not valid
    assert "visible_devices" in reason


def test_biren_defaults_preserve_explicit_user_values(monkeypatch) -> None:
    monkeypatch.delenv("RWKV7_NATIVE_MODEL_BACKEND", raising=False)
    monkeypatch.setenv("RWKV7_FAST_CACHE", "1")
    values = biren_runtime.configure_biren_defaults()
    assert values["RWKV7_NATIVE_MODEL_BACKEND"] == "eager"
    assert values["RWKV7_NATIVE_MODEL_JIT"] == "0"
    assert values["RWKV7_FAST_CACHE"] == "1"

    values = biren_runtime.configure_biren_defaults(overwrite=True)
    assert values["RWKV7_FAST_CACHE"] == "0"


def test_biren_default_backend_disables_jit_without_overriding_user_env(
    monkeypatch,
) -> None:
    def biren_policy():
        return SimpleNamespace(profile=SimpleNamespace(family="biren"))

    monkeypatch.delenv("RWKV7_NATIVE_MODEL_JIT", raising=False)
    assert not native_model_jit_enabled(kernel_policy_fn=biren_policy)
    monkeypatch.setenv("RWKV7_NATIVE_MODEL_JIT", "1")
    assert native_model_jit_enabled(kernel_policy_fn=biren_policy)


def test_decomposed_group_norm_matches_torch_oracle() -> None:
    torch.manual_seed(17)
    value = torch.randn(3, 16, dtype=torch.float32)
    weight = torch.randn(16, dtype=torch.float32)
    bias = torch.randn(16, dtype=torch.float32)
    expected = F.group_norm(value, 4, weight=weight, bias=bias, eps=4e-5)
    actual = _decomposed_group_norm(
        value,
        4,
        weight=weight,
        bias=bias,
        eps=4e-5,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)

    bf16_value = value.to(torch.bfloat16)
    bf16_weight = weight.to(torch.bfloat16)
    bf16_bias = bias.to(torch.bfloat16)
    bf16_expected = F.group_norm(
        bf16_value,
        4,
        weight=bf16_weight,
        bias=bf16_bias,
        eps=4e-5,
    )
    bf16_actual = _decomposed_group_norm(
        bf16_value,
        4,
        weight=bf16_weight,
        bias=bf16_bias,
        eps=4e-5,
    )
    torch.testing.assert_close(bf16_actual, bf16_expected, rtol=2e-2, atol=2e-2)


def test_enable_biren_reports_exact_validated_stack(monkeypatch) -> None:
    selected: list[str] = []
    fake_supa = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_name=lambda _index: "Biren106M",
        set_device=lambda device: selected.append(str(device)),
    )
    monkeypatch.setattr(torch, "supa", fake_supa, raising=False)
    monkeypatch.setattr(biren_runtime, "import_torch_br", lambda **_kwargs: object())
    monkeypatch.setattr(biren_runtime, "_package_version", lambda _name: "1.10.0.20900+br1xx")
    monkeypatch.setattr(biren_runtime, "detect_biren_sdk_version", lambda: "1.11.0.0.rc2")
    monkeypatch.setattr(
        biren_runtime,
        "detect_biren_driver_versions",
        lambda: {"brsmi": "1.11.0", "driver": "1.11.0", "supa": "1.11"},
    )
    monkeypatch.setattr(torch, "__version__", "2.9.0+cu128")
    monkeypatch.delenv("RWKV7_ALLOW_UNVALIDATED_BIREN", raising=False)

    info = biren_runtime.enable_biren("biren:0")
    assert info.available
    assert info.validated_stack
    assert info.validation_status == "validated"
    assert info.device == "supa:0"
    assert info.dtype == "bfloat16"
    assert selected == ["supa:0"]


def test_enable_biren_rejects_adjacent_product(monkeypatch) -> None:
    fake_supa = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_name=lambda _index: "Biren106M2",
    )
    monkeypatch.setattr(torch, "supa", fake_supa, raising=False)
    monkeypatch.setattr(biren_runtime, "import_torch_br", lambda **_kwargs: object())
    with pytest.raises(RuntimeError, match="expected exact Biren106M"):
        biren_runtime.enable_biren(0)
