#!/usr/bin/env python3
from __future__ import annotations

import ast
import importlib
from pathlib import Path

import torch

from rwkv7_hf.kernel_policy import classify_gpu, detect_gpu_profile, policy_for_profile

musa_wkv_module = importlib.import_module("rwkv7_hf.musa_wkv")



ROOT = Path(__file__).resolve().parents[1]


def test_musa_profile_and_policy_are_distinct_and_conservative() -> None:
    profile = classify_gpu("Moore Threads MTT S70", None, is_musa=True)
    assert profile.family == "musa"
    assert profile.vendor == "moore_threads"
    assert profile.is_musa
    policy = policy_for_profile(profile)
    assert policy.fast_token_backend == "native"
    assert policy.fast_cache
    assert not policy.fused_recurrent_output
    assert not policy.fused_output
    assert not policy.fused_prefill_scan
    assert policy.quant_policy == "musa_unvalidated"


def test_musa_runtime_detection_does_not_depend_on_cuda() -> None:
    class FakeMusa:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def current_device():
            return 0

        @staticmethod
        def get_device_name(index):
            assert index == 0
            return "Moore Threads MTT S70"

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

    class FakeTorch:
        musa = FakeMusa()
        cuda = FakeCuda()
        version = type("Version", (), {"hip": None})()

    profile = detect_gpu_profile(torch_module=FakeTorch())
    assert profile.family == "musa"
    assert profile.device_index == 0
    assert profile.is_musa
    assert not profile.is_cuda


def test_musa_wkv_fails_closed_without_runtime(monkeypatch) -> None:
    monkeypatch.setattr(musa_wkv_module, "_musa_available", lambda: False)
    operands = [torch.zeros(1, 1, 1, 64, dtype=torch.float16) for _ in range(6)]
    assert not musa_wkv_module.musa_wkv_available()
    assert not musa_wkv_module.musa_wkv_can_run(*operands)


def test_musa_wkv_is_disabled_while_autograd_is_enabled(monkeypatch) -> None:
    monkeypatch.setattr(musa_wkv_module, "_musa_available", lambda: True)

    class FakeDevice:
        type = "musa"

        def __eq__(self, other):
            return isinstance(other, FakeDevice)

    class FakeTensor:
        device = FakeDevice()
        dtype = torch.float16
        shape = (1, 1, 1, 64)

        @staticmethod
        def dim():
            return 4

    assert torch.is_grad_enabled()
    assert not musa_wkv_module.musa_wkv_can_run(*([FakeTensor()] * 6))


def test_musa_build_failure_returns_to_eager_fallback(monkeypatch) -> None:
    monkeypatch.setattr(musa_wkv_module, "musa_wkv_can_run", lambda *args: True)
    monkeypatch.setattr(
        musa_wkv_module,
        "musa_wkv",
        lambda *args: (_ for _ in ()).throw(RuntimeError("build failed")),
    )
    assert musa_wkv_module.try_musa_wkv(*([object()] * 7)) is None


def test_musa_extension_is_lazy_and_does_not_import_torch_musa_at_module_import() -> None:
    source = (ROOT / "rwkv7_hf" / "musa_wkv.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    top_level_imports = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "torch_musa" not in top_level_imports
    assert "triton" not in source


def test_musa_kernel_keeps_validated_fp16_io_fp32_state_contract() -> None:
    wrapper = (ROOT / "rwkv7_hf" / "musa_wkv.py").read_text(encoding="utf-8")
    kernel = (ROOT / "rwkv7_hf" / "csrc" / "musa" / "wkv7_musa.muh").read_text(
        encoding="utf-8"
    )
    assert "head_size=64" in wrapper
    assert "state shape must be [B,H,64,64]" in wrapper
    assert "PrivateUse1" not in wrapper
    assert "#include <musa_fp16.h>" in kernel
    assert "__shfl" not in kernel
    assert "musa_bf16" not in kernel
