from __future__ import annotations

from types import SimpleNamespace

import pytest

from rwkv7_hf import kernel_package


class _FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def current_device() -> int:
        return 0

    @staticmethod
    def get_device_capability(_index: int = 0) -> tuple[int, int]:
        return 8, 9


class _FakeTorch:
    __version__ = "2.5.1+cu124"
    cuda = _FakeCuda()
    version = SimpleNamespace(cuda="12.4", hip=None)
    _C = SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=False)

    @staticmethod
    def device(value):
        text = str(value)
        return SimpleNamespace(
            type=text.split(":", 1)[0],
            index=int(text.split(":", 1)[1]) if ":" in text else None,
        )


class _FakeCpuCuda(_FakeCuda):
    @staticmethod
    def is_available() -> bool:
        return False


class _FakeCpuTorch(_FakeTorch):
    cuda = _FakeCpuCuda()


def _manifest(**updates):
    value = {
        "schema_version": 1,
        "protocol": "rwkv7-kernel-package-v1",
        "distribution": "rwkv7-kernels",
        "version": "0.8.0+cu124.torch25.sm89",
        "adapter_specifier": ">=0.8.0,<0.9",
        "python_series": kernel_package._runtime_environment(_FakeTorch())[  # noqa: SLF001
            "python_series"
        ],
        "python_abi": kernel_package._runtime_environment(_FakeTorch())[  # noqa: SLF001
            "python_abi"
        ],
        "platform_system": kernel_package._runtime_environment(_FakeTorch())[  # noqa: SLF001
            "platform_system"
        ],
        "platform_machine": kernel_package._runtime_environment(_FakeTorch())[  # noqa: SLF001
            "platform_machine"
        ],
        "torch_series": "2.5",
        "torch_cuda_version": "12.4",
        "torch_cxx11_abi": False,
        "architectures": ["sm89"],
        "extensions": {"ada_lora": "rwkv7_kernels._C.ada_lora"},
    }
    value.update(updates)
    return value


def test_missing_package_recommends_exact_runtime_distribution(monkeypatch) -> None:
    monkeypatch.delenv(kernel_package.KERNEL_MODE_ENV, raising=False)
    monkeypatch.setattr(kernel_package, "_load_package_module", lambda: (None, None))
    report = kernel_package.inspect_kernel_package(torch_module=_FakeTorch())

    assert report["status"] == "missing"
    assert report["recommended_distribution"] == "rwkv7-kernels"
    assert report["recommended_build"] == "cu124-torch25-sm89"


def test_compatible_package_loads_binary_and_records_selection(monkeypatch) -> None:
    monkeypatch.delenv(kernel_package.KERNEL_MODE_ENV, raising=False)
    monkeypatch.setattr(kernel_package, "_distribution_version", lambda _name: "0.8.0")
    extension = SimpleNamespace(__name__="rwkv7_kernels._C.ada_lora")
    package = SimpleNamespace(
        manifest=lambda: _manifest(),
        load_extension=lambda name: extension if name == "ada_lora" else None,
    )
    monkeypatch.setattr(kernel_package, "_load_package_module", lambda: (package, None))
    kernel_package.reset_kernel_runtime_events()

    loaded = kernel_package.load_prebuilt_extension(
        "ada_lora", torch_module=_FakeTorch(), device="cuda:0"
    )

    assert loaded is extension
    runtime = kernel_package.kernel_runtime_report(
        torch_module=_FakeTorch(), device="cuda:0"
    )
    assert runtime["package"]["status"] == "ready"
    assert runtime["extensions"]["ada_lora"]["source"] == "prebuilt"
    assert runtime["extensions"]["ada_lora"]["status"] == "selected"


def test_incompatible_package_falls_back_in_auto_and_fails_in_prebuilt(
    monkeypatch,
) -> None:
    package = SimpleNamespace(manifest=lambda: _manifest(torch_series="2.6"))
    monkeypatch.setattr(kernel_package, "_load_package_module", lambda: (package, None))
    monkeypatch.setattr(kernel_package, "_distribution_version", lambda _name: "0.8.0")
    monkeypatch.delenv(kernel_package.KERNEL_MODE_ENV, raising=False)

    assert (
        kernel_package.load_prebuilt_extension("ada_lora", torch_module=_FakeTorch())
        is None
    )
    monkeypatch.setenv(kernel_package.KERNEL_MODE_ENV, "prebuilt")
    with pytest.raises(RuntimeError, match="torch_series mismatch"):
        kernel_package.load_prebuilt_extension("ada_lora", torch_module=_FakeTorch())


def test_manifest_requires_a_visible_cuda_architecture(monkeypatch) -> None:
    monkeypatch.setattr(kernel_package, "_distribution_version", lambda _name: "0.8.0")
    report = kernel_package.inspect_kernel_package(
        torch_module=_FakeCpuTorch(),
        package_module=SimpleNamespace(manifest=lambda: _manifest()),
    )
    assert report["status"] == "incompatible"
    assert "runtime does not expose a CUDA GPU architecture" in report["reasons"]


def test_manifest_rejects_missing_required_abi_field(monkeypatch) -> None:
    monkeypatch.setattr(kernel_package, "_distribution_version", lambda _name: "0.8.0")
    manifest = _manifest()
    del manifest["torch_cxx11_abi"]
    report = kernel_package.inspect_kernel_package(
        torch_module=_FakeTorch(),
        package_module=SimpleNamespace(manifest=lambda: manifest),
    )
    assert report["status"] == "incompatible"
    assert (
        "kernel manifest is missing required field torch_cxx11_abi" in report["reasons"]
    )


def test_jit_and_portable_modes_disable_prebuilt(monkeypatch) -> None:
    for mode, jit_allowed in (("jit", True), ("portable", False)):
        monkeypatch.setenv(kernel_package.KERNEL_MODE_ENV, mode)
        report = kernel_package.inspect_kernel_package(torch_module=_FakeTorch())
        assert report["status"] == "disabled"
        assert kernel_package.jit_extensions_allowed() is jit_allowed
        assert kernel_package.prebuilt_extensions_allowed() is False


def test_invalid_mode_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv(kernel_package.KERNEL_MODE_ENV, "fastest")
    with pytest.raises(ValueError, match="RWKV7_KERNELS_MODE"):
        kernel_package.kernel_mode()
