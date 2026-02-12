#!/usr/bin/env python3
from __future__ import annotations

import ast
import importlib
from pathlib import Path

import torch

from rwkv7_hf.kernel_policy import classify_gpu, detect_gpu_profile, policy_for_profile

check_environment_module = importlib.import_module("examples.check_environment")
musa_build_module = importlib.import_module("rwkv7_hf.musa_build")
musa_wkv_module = importlib.import_module("rwkv7_hf.musa_wkv")
musa_wkv_source_module = importlib.import_module("rwkv7_hf.musa_wkv_source")


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


def test_musa_remote_code_embeds_the_licensed_kernel_resource() -> None:
    from scripts.adapter_manifest import ADAPTER_FILES

    kernel = (ROOT / "rwkv7_hf" / "csrc" / "musa" / "wkv7_musa.muh").read_text(
        encoding="utf-8"
    )
    assert musa_wkv_source_module.WKV7_MUSA_HEADER == kernel
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
    assert '"musa_wkv_module_loaded": module._MODULE is not None' in batch
    assert 'args.device.startswith(("cuda", "musa", "mps"))' in smoke
    assert "torch.musa.synchronize()" in smoke
    assert '"--trust-remote-code"' in smoke
    assert smoke.count("trust_remote_code=args.trust_remote_code") == 2
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
