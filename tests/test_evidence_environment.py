from __future__ import annotations

import hashlib
from pathlib import Path

from evaluation.common import cuda_toolkit_provenance


def test_cuda_toolkit_provenance_binds_compiler_environment(
    tmp_path: Path, monkeypatch
):
    cuda = tmp_path / "cuda-13.0.88"
    nvcc = cuda / "bin/nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text(
        "#!/bin/sh\necho 'Cuda compilation tools, release 13.0, V13.0.88'\n"
    )
    nvcc.chmod(0o755)
    provenance = cuda / "PROVENANCE.txt"
    provenance.write_text("nvcc_wheel_sha256=" + "a" * 64 + "\n")
    monkeypatch.setenv("CUDA_HOME", str(cuda))
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "extensions"))

    report = cuda_toolkit_provenance()
    assert report["cuda_home"] == str(cuda.resolve())
    assert report["nvcc"] == str(nvcc)
    assert report["nvcc_version"] == ["Cuda compilation tools, release 13.0, V13.0.88"]
    assert report["provenance"] == {
        "path": str(provenance),
        "sha256": hashlib.sha256(provenance.read_bytes()).hexdigest(),
    }


def test_cuda_toolkit_provenance_is_explicit_when_compiler_is_absent(monkeypatch):
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("TORCH_EXTENSIONS_DIR", raising=False)
    assert cuda_toolkit_provenance() == {
        "cuda_home": None,
        "torch_extensions_dir": None,
        "nvcc": None,
        "nvcc_version": None,
        "provenance": None,
    }
