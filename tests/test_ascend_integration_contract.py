from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import torch

from examples import check_environment, generate
from rwkv7_hf import model_runtime, native_model
from scripts.adapter_manifest import ADAPTER_FILES


ROOT = Path(__file__).resolve().parents[1]
ASCEND_REMOTE_FILES = {
    "ascend_graph_runtime.py",
    "ascend_quant.py",
    "ascend_quant_w4.py",
    "ascend_runtime.py",
    "ascend_w4_cle.py",
}


def test_ascend_modules_are_in_converted_checkpoint_manifest() -> None:
    assert ASCEND_REMOTE_FILES <= set(ADAPTER_FILES)
    facade = (ROOT / "rwkv7_hf" / "native_model.py").read_text(encoding="utf-8")
    for name in ASCEND_REMOTE_FILES:
        assert name.removesuffix(".py") in facade


def test_public_package_exposes_import_safe_ascend_api() -> None:
    import rwkv7_hf

    for name in (
        "ascend_available",
        "enable_ascend",
        "quantize_ascend_w8a16",
        "quantize_ascend_w4a16_candidate",
    ):
        assert callable(getattr(rwkv7_hf, name))


def test_user_entrypoints_and_docs_include_npu_lane() -> None:
    generate = (ROOT / "examples" / "generate.py").read_text(encoding="utf-8")
    environment = (ROOT / "examples" / "check_environment.py").read_text(
        encoding="utf-8"
    )
    smoke = (ROOT / "tests" / "smoke_hf_generate.py").read_text(encoding="utf-8")
    assert '"npu"' in generate and "enable_ascend" in generate
    assert "torch-npu" in environment and "report_ascend_devices" in environment
    assert 'args.device.startswith("npu")' in smoke
    assert (ROOT / "docs" / "hardware" / "HUAWEI_ASCEND.md").is_file()


def test_generate_auto_selects_ascend_before_other_private_backends(
    monkeypatch,
) -> None:
    npu_device = SimpleNamespace(type="npu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(generate, "_ascend_available", lambda: True)
    monkeypatch.setattr(generate, "_enable_ascend", lambda: npu_device)
    monkeypatch.setattr(generate, "_musa_available", lambda: True)
    assert generate.resolve_device("auto") is npu_device
    assert generate.resolve_dtype("auto", npu_device) is torch.bfloat16


def test_environment_doctor_keeps_ascend_available_when_metadata_fails(
    capsys,
) -> None:
    class FakeNPU:
        @staticmethod
        def device_count():
            raise RuntimeError("runtime metadata unavailable")

    check_environment.report_ascend_devices(FakeNPU())
    output = capsys.readouterr().out
    assert "[PASS] Huawei Ascend NPU: available" in output
    assert "[WARN] Ascend NPU device count unavailable: RuntimeError" in output


def test_npu_graph_runner_uses_ascend_runner_without_cuda_packs(monkeypatch) -> None:
    created: list[tuple[object, int]] = []

    class FakeAscendRunner:
        def __init__(self, owner, batch_size: int) -> None:
            created.append((owner, int(batch_size)))

    fake_device = SimpleNamespace(type="npu", index=0)
    owner = SimpleNamespace(
        model=SimpleNamespace(
            embeddings=SimpleNamespace(
                weight=SimpleNamespace(device=fake_device, dtype=torch.float16)
            ),
            layers=torch.nn.ModuleList(),
        ),
        _rwkv7_native_mm_quantization="none",
        _rwkv7_native_mm_replaced_modules=0,
        _rwkv7_native_graph_runner_cache=OrderedDict(),
    )
    monkeypatch.setattr(native_model, "_AscendGraphRunner", FakeAscendRunner)
    monkeypatch.setattr(native_model, "_ascend_graph_module_signature", lambda _: ())
    monkeypatch.setattr(native_model, "_ascend_graph_runtime_signature", lambda: ())
    monkeypatch.setattr(native_model, "_ascend_graph_cache_size", lambda: 3)
    monkeypatch.setattr(
        native_model,
        "_native_graph_extract",
        lambda _: (_ for _ in ()).throw(AssertionError("CUDA packs used on NPU")),
    )

    runner = model_runtime._NativeRuntimeMixin._native_graph_runner_current_device(
        owner, 4
    )
    assert isinstance(runner, FakeAscendRunner)
    assert created == [(owner, 4)]
    assert owner._rwkv7_native_graph_cache_stats == {
        "requests": 1,
        "hits": 0,
        "misses": 1,
        "evictions": 0,
    }
