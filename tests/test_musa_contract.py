#!/usr/bin/env python3
from __future__ import annotations

import ast
import importlib
from pathlib import Path

import torch

from rwkv7_hf.kernel_policy import (
    classify_gpu,
    detect_gpu_profile,
    is_mtt_s70_name,
    policy_for_profile,
)

check_environment_module = importlib.import_module("examples.check_environment")
musa_build_module = importlib.import_module("rwkv7_hf.musa_build")
musa_fused_module = importlib.import_module("rwkv7_hf.musa_fused")
musa_wkv_module = importlib.import_module("rwkv7_hf.musa_wkv")
musa_wkv_source_module = importlib.import_module("rwkv7_hf.musa_wkv_source")


ROOT = Path(__file__).resolve().parents[1]


def test_musa_profile_and_policy_are_distinct_and_conservative() -> None:
    profile = classify_gpu("Moore Threads MTT S70", None, is_musa=True)
    assert profile.family == "musa"
    assert profile.vendor == "moore_threads"
    assert profile.is_musa
    assert profile.hardware_generation == "musa_legacy_s70"
    assert profile.validation_scope == "exact_card_smoke"
    assert profile.compute_profile == "fp32_compute_fp16_io"
    policy = policy_for_profile(profile)
    assert policy.fast_token_backend == "native"
    assert policy.fast_cache
    assert not policy.fused_recurrent_output
    assert not policy.fused_output
    assert not policy.fused_prefill_scan
    assert policy.quant_policy == "musa_unvalidated"
    assert "legacy SDK 4.2.0" in policy.notes

    for name in ("Moore Threads MTT S4000", "Moore Threads MTT S5000"):
        later = classify_gpu(name, None, is_musa=True)
        assert later.family == "musa"
        assert later.hardware_generation == "musa_post_s70"
        assert later.validation_scope == "unvalidated"
        assert later.compute_profile == "device_specific_unvalidated"
        assert "do not inherit legacy S70" in policy_for_profile(later).notes


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

    class FakeDevice:
        def __init__(self, value):
            self.index = str(value).split(":", 1)[-1]

    class FakeTorch:
        musa = FakeMusa()
        cuda = FakeCuda()
        device = FakeDevice
        version = type("Version", (), {"hip": None})()

    profile = detect_gpu_profile(torch_module=FakeTorch())
    explicit = detect_gpu_profile(device="musa:1", torch_module=FakeTorch())
    assert profile.family == "musa"
    assert profile.device_index == 0
    assert explicit.device_index == 1
    assert type(explicit.device_index) is int
    assert profile.is_musa
    assert not profile.is_cuda
    assert profile.hardware_generation == "musa_legacy_s70"
    assert profile.validation_scope == "exact_card_smoke"


def test_environment_doctor_keeps_musa_available_when_device_count_fails(
    capsys,
) -> None:
    class FakeMusa:
        @staticmethod
        def device_count():
            raise RuntimeError("runtime not fully initialized")

    check_environment_module.report_musa_devices(FakeMusa())
    output = capsys.readouterr().out
    assert "[PASS] MUSA: available" in output
    assert "[WARN] MUSA device count unavailable: RuntimeError" in output


def test_environment_doctor_continues_when_one_musa_device_name_fails(capsys) -> None:
    class FakeMusa:
        @staticmethod
        def device_count():
            return 2

        @staticmethod
        def get_device_name(index):
            if index == 0:
                raise RuntimeError("device metadata unavailable")
            return "Moore Threads MTT S70"

    check_environment_module.report_musa_devices(FakeMusa())
    output = capsys.readouterr().out
    assert "[PASS] MUSA: available (2 device(s))" in output
    assert "[WARN] MUSA device 0 name unavailable: RuntimeError" in output
    assert "[INFO] MUSA device 1: Moore Threads MTT S70" in output


def test_environment_doctor_completes_when_musa_metadata_fails(
    monkeypatch,
    capsys,
) -> None:
    class FakeMusa:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def device_count():
            raise RuntimeError("runtime not fully initialized")

    monkeypatch.setattr(torch, "musa", FakeMusa(), raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(check_environment_module, "package_version", lambda name: "test")
    monkeypatch.setattr(check_environment_module, "MIN_PYTHON", (3, 9))

    assert check_environment_module.main([]) == 0
    output = capsys.readouterr().out
    assert "[PASS] MUSA: available" in output
    assert "[WARN] MUSA device count unavailable: RuntimeError" in output
    assert "RESULT: READY" in output


def test_musa_gcc_include_candidates_use_numeric_version_order(monkeypatch) -> None:
    monkeypatch.setattr(
        musa_build_module.glob,
        "glob",
        lambda pattern: ["/usr/include/c++/9", "/usr/include/c++/13"],
    )
    monkeypatch.setattr(musa_build_module.os.path, "isdir", lambda path: True)
    includes = musa_build_module._gcc_includes()
    assert includes[1] == "/usr/include/c++/13"
    assert includes[3] == "/usr/include/x86_64-linux-gnu/c++/13"


def test_musa_wkv_fails_closed_without_runtime(monkeypatch) -> None:
    monkeypatch.setattr(musa_wkv_module, "_musa_available", lambda: False)
    operands = [torch.zeros(1, 1, 1, 64, dtype=torch.float16) for _ in range(6)]
    assert not musa_wkv_module.musa_wkv_available()
    assert not musa_wkv_module.musa_wkv_can_run(*operands)


def test_musa_wkv_auto_mode_is_exact_card_fail_closed(monkeypatch) -> None:
    monkeypatch.setattr(musa_wkv_module, "_musa_available", lambda: True)
    monkeypatch.delenv("RWKV7_MUSA_WKV", raising=False)
    assert musa_wkv_module._mode() == "auto"
    assert not musa_wkv_module.musa_wkv_available()
    monkeypatch.setenv("RWKV7_MUSA_WKV", "0")
    assert musa_wkv_module._mode() == "off"
    assert not musa_wkv_module.musa_wkv_available()
    monkeypatch.setenv("RWKV7_MUSA_WKV", "1")
    assert musa_wkv_module._mode() == "on"
    assert musa_wkv_module.musa_wkv_available()
    monkeypatch.delenv("RWKV7_MUSA_WKV", raising=False)

    calls = []

    class FakeMusa:
        @staticmethod
        def get_device_name(index):
            calls.append(index)
            return "Moore Threads MTT S70"

    monkeypatch.setattr(torch, "musa", FakeMusa(), raising=False)
    musa_wkv_module._VALIDATED_DEVICE_CACHE.clear()
    device = type("FakeDevice", (), {"index": 0})()
    assert musa_wkv_module._is_validated_device(device)
    assert musa_wkv_module._is_validated_device(device)
    assert musa_wkv_module.musa_wkv_available(device)
    assert calls == [0]

    musa_wkv_module._VALIDATED_DEVICE_CACHE.clear()
    monkeypatch.setattr(
        torch,
        "musa",
        type("LaterMusa", (), {"get_device_name": staticmethod(lambda index: "MTT S5000")})(),
        raising=False,
    )
    assert not musa_wkv_module._is_validated_device(device)
    assert not musa_wkv_module.musa_wkv_available(device)
    monkeypatch.setenv("RWKV7_MUSA_WKV", "1")
    assert musa_wkv_module.musa_wkv_available(device)


def test_native_musa_wkv_route_requires_fp16_model_dtype() -> None:
    source = (ROOT / "rwkv7_hf" / "native.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    route_guard = next(
        node.test
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(descendant, ast.Call)
            and isinstance(descendant.func, ast.Name)
            and descendant.func.id == "musa_wkv_available"
            for descendant in ast.walk(node.test)
        )
    )
    assert any(
        isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Attribute)
        and isinstance(node.left.value, ast.Name)
        and node.left.value.id == "x"
        and node.left.attr == "dtype"
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.Eq)
        and len(node.comparators) == 1
        and isinstance(node.comparators[0], ast.Attribute)
        and isinstance(node.comparators[0].value, ast.Name)
        and node.comparators[0].value.id == "torch"
        and node.comparators[0].attr == "float16"
        for node in ast.walk(route_guard)
    )


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
    for error in (
        RuntimeError("build failed"),
        ValueError("invalid state"),
        OSError("compiler unavailable"),
    ):
        monkeypatch.setattr(
            musa_wkv_module,
            "musa_wkv",
            lambda *args, error=error: (_ for _ in ()).throw(error),
        )
        assert musa_wkv_module.try_musa_wkv(*([object()] * 7)) is None


def test_musa_extension_is_lazy_and_does_not_import_torch_musa_at_module_import() -> None:
    for name in ("musa_wkv.py", "musa_fused.py"):
        source = (ROOT / "rwkv7_hf" / name).read_text(encoding="utf-8")
        tree = ast.parse(source)
        top_level_imports = {
            alias.name
            for node in tree.body
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        assert "torch_musa" not in top_level_imports
        assert "triton" not in source


def test_musa_attention_shift_mix_is_opt_in_and_fail_closed(monkeypatch) -> None:
    monkeypatch.delenv("RWKV7_MUSA_ATTN_SHIFT_MIX", raising=False)
    monkeypatch.setattr(musa_fused_module, "_musa_available", lambda: True)
    musa_fused_module._MODULE_ERROR = None
    assert not musa_fused_module.musa_attn_shift_mix_available()

    monkeypatch.setenv("RWKV7_MUSA_ATTN_SHIFT_MIX", "1")
    assert musa_fused_module.musa_attn_shift_mix_available()
    monkeypatch.setattr(
        musa_fused_module,
        "musa_attn_shift_mix_can_run",
        lambda *args: True,
    )
    monkeypatch.setattr(
        musa_fused_module,
        "_load_module",
        lambda: (_ for _ in ()).throw(RuntimeError("build failed")),
    )
    assert musa_fused_module.try_musa_attn_shift_mix(*([object()] * 8)) is None


def test_musa_attention_shift_mix_is_exact_s70_only(monkeypatch) -> None:
    assert is_mtt_s70_name("Moore Threads MTT S70")
    assert is_mtt_s70_name("MTT-S70")
    assert not is_mtt_s70_name("Moore Threads MTT S80")
    assert not is_mtt_s70_name("Moore Threads MTT S700")
    assert not is_mtt_s70_name("Moore Threads MTT S4000")
    assert not is_mtt_s70_name("Moore Threads MTT S5000")
    assert not is_mtt_s70_name("Generic MUSA Device")

    calls = []

    class FakeMusa:
        @staticmethod
        def get_device_name(device):
            calls.append(device)
            return "Moore Threads MTT S70"

    monkeypatch.setattr(torch, "musa", FakeMusa(), raising=False)
    musa_fused_module._S70_DEVICE_CACHE.clear()
    device = type("FakeDevice", (), {"index": 0})()
    assert musa_fused_module._is_mtt_s70_device(device)
    assert musa_fused_module._is_mtt_s70_device(device)
    assert len(calls) == 1


def test_musa_attention_shift_mix_keeps_separate_lazy_module() -> None:
    source = (ROOT / "rwkv7_hf" / "musa_fused.py").read_text(encoding="utf-8")
    wkv_source = (ROOT / "rwkv7_hf" / "musa_wkv.py").read_text(encoding="utf-8")
    assert "rwkv7_hf_attn_shift_mix_musa_v2_strict" in source
    assert '"-ffp-contract=off"' in source
    assert "RWKV7_MUSA_ATTN_SHIFT_MIX" in source
    assert "attn_shift_mix_6" not in wkv_source


def test_musa_remote_code_embeds_the_licensed_kernel_resource() -> None:
    from scripts.adapter_manifest import ADAPTER_FILES

    kernel = (ROOT / "rwkv7_hf" / "csrc" / "musa" / "wkv7_musa.muh").read_text(
        encoding="utf-8"
    )
    assert musa_wkv_source_module.WKV7_MUSA_HEADER == kernel
    assert "musa_fused.py" in ADAPTER_FILES
    assert "musa_wkv_source.py" in ADAPTER_FILES
    assert "from .musa_wkv_source import WKV7_MUSA_HEADER" in (
        ROOT / "rwkv7_hf" / "musa_wkv.py"
    ).read_text(encoding="utf-8")


def test_musa_hardware_acceptance_tools_move_and_measure_on_musa() -> None:
    speed = (ROOT / "bench" / "bench_speed.py").read_text(encoding="utf-8")
    batch = (ROOT / "bench" / "bench_batch_sweep.py").read_text(encoding="utf-8")
    smoke = (ROOT / "tests" / "smoke_hf_generate.py").read_text(encoding="utf-8")
    api = (ROOT / "tests" / "test_hf_api_contract.py").read_text(encoding="utf-8")

    assert 'device.startswith("musa")' in speed
    assert "model.to(args.device)" in speed
    assert "peak_memory_mb(args.device)" in speed
    assert "m.forward(id_list[:8], None)\n    device_sync(args.device)\n    t0 = time.time()" in speed
    assert 'torch.device(device) if device.startswith(("cuda", "musa"))' in speed
    assert 'device.startswith("musa")' in batch
    assert "model.to(args.device)" in batch
    assert "return ids.to(device)" in batch
    assert 'torch.device(device) if device.startswith(("cuda", "musa"))' in batch
    assert 'importlib.import_module(package + ".musa_wkv")' in batch
    assert 'importlib.import_module(package + ".musa_fused")' in batch
    assert '"musa_wkv_mode": wkv._mode()' in batch
    assert '"musa_wkv_available": bool(wkv.musa_wkv_available(next(model.parameters()).device))' in batch
    assert '"musa_wkv_module_loaded": wkv._MODULE is not None' in batch
    assert '"musa_attn_shift_mix_module_loaded": fused._MODULE is not None' in batch
    assert '"musa_attn_shift_mix_calls": int(fused._CALLS)' in batch
    assert 'forward_route["musa_attn_shift_mix_calls_delta"]' in batch
    assert 'fast_route["musa_attn_shift_mix_calls_delta"]' in batch
    assert 'args.device.startswith(("cuda", "npu", "musa", "mps"))' in smoke
    assert "torch.musa.synchronize()" in smoke
    assert '"--trust-remote-code"' not in smoke
    assert smoke.count("trust_remote_code=True") == 2
    assert "model.to(args.device)" in api
    assert "v.to(args.device)" in api


def test_musa_kernel_keeps_validated_fp16_io_fp32_state_contract() -> None:
    wrapper = (ROOT / "rwkv7_hf" / "musa_wkv.py").read_text(encoding="utf-8")
    kernel = musa_wkv_source_module.WKV7_MUSA_HEADER
    assert "head_size=64" in wrapper
    assert "state shape must be [B,H,64,64]" in wrapper
    assert "PrivateUse1" not in wrapper
    assert "c10::musa::MUSAGuard" in wrapper
    assert "c10::musa::getCurrentMUSAStream" in wrapper
    assert "C10_MUSA_KERNEL_LAUNCH_CHECK" in wrapper
    assert "#include <musa_fp16.h>" in kernel
    assert "__shfl" not in kernel
    assert "musa_bf16" not in kernel
