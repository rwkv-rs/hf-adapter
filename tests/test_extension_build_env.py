from __future__ import annotations

import os

from rwkv7_hf.extension_build import cuda_extension_build_environment


MANAGED = (
    "PATH",
    "CUDA_HOME",
    "TORCH_CUDA_ARCH_LIST",
    "LIBRARY_PATH",
    "LD_LIBRARY_PATH",
)


def test_cuda_extension_build_environment_restores_every_managed_value(monkeypatch) -> None:
    original = {
        "PATH": "/caller/bin",
        "CUDA_HOME": "/caller/cuda",
        "TORCH_CUDA_ARCH_LIST": "8.9",
        "LIBRARY_PATH": "/caller/lib",
        "LD_LIBRARY_PATH": "/caller/ld",
    }
    for key, value in original.items():
        monkeypatch.setenv(key, value)

    with cuda_extension_build_environment(arch_list="12.0"):
        # This build is card-local; the caller value is restored afterward.
        assert os.environ["TORCH_CUDA_ARCH_LIST"] == "12.0"
        assert os.environ["CUDA_HOME"] == "/caller/cuda"

    assert {key: os.environ.get(key) for key in MANAGED} == original


def test_cuda_extension_build_environment_removes_temporary_arch(monkeypatch) -> None:
    before = {key: os.environ.get(key) for key in MANAGED}
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    with cuda_extension_build_environment(arch_list="7.0;7.5"):
        assert os.environ["TORCH_CUDA_ARCH_LIST"] == "7.0;7.5"
    assert "TORCH_CUDA_ARCH_LIST" not in os.environ
    for key in MANAGED:
        if key != "TORCH_CUDA_ARCH_LIST":
            assert os.environ.get(key) == before[key]
