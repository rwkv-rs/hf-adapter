from __future__ import annotations

import os
from pathlib import Path

from rwkv7_hf import extension_build
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


def test_cuda_extension_build_environment_finds_thin_venv_base_tools(
    monkeypatch,
    tmp_path: Path,
) -> None:
    venv_bin = tmp_path / "venv" / "bin"
    base_bin = tmp_path / "base" / "bin"
    venv_bin.mkdir(parents=True)
    base_bin.mkdir(parents=True)
    for tool in ("ninja", "nvcc"):
        (base_bin / tool).touch()

    monkeypatch.setattr(extension_build.sys, "executable", str(venv_bin / "python"))
    monkeypatch.setattr(extension_build.sys, "base_prefix", str(base_bin.parent))
    monkeypatch.setenv("PATH", "/caller/bin")
    monkeypatch.delenv("CUDA_HOME", raising=False)

    with cuda_extension_build_environment(arch_list="7.0"):
        paths = os.environ["PATH"].split(os.pathsep)
        assert paths[:2] == [str(venv_bin), str(base_bin)]
        assert os.environ["CUDA_HOME"] == str(base_bin.parent)

    assert os.environ["PATH"] == "/caller/bin"
    assert "CUDA_HOME" not in os.environ
