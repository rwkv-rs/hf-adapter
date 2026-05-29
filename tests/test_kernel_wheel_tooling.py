from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from scripts.inspect_kernel_wheel import inspect_wheel


def _write_wheel(path: Path, *, include_binary: bool = True) -> None:
    manifest = {
        "schema_version": 1,
        "protocol": "rwkv7-kernel-package-v1",
        "distribution": "rwkv7-kernels",
        "version": "0.8.0+cu124.torch25.sm70",
        "adapter_specifier": ">=0.8.0,<0.9",
        "python_series": "3.11",
        "python_abi": "cpython-311",
        "platform_system": "Linux",
        "platform_machine": "x86_64",
        "torch_series": "2.5",
        "torch_cuda_version": "12.4",
        "torch_cxx11_abi": False,
        "architectures": ["sm70"],
        "extensions": {"native_wkv_fp16": "rwkv7_kernels._C.native_wkv_fp16"},
        "source_commit": "test",
        "source_tree_sha256": "0" * 64,
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("rwkv7_kernels/_manifest.json", json.dumps(manifest))
        if include_binary:
            archive.writestr(
                "rwkv7_kernels/_C/native_wkv_fp16.cpython-311-x86_64-linux-gnu.so",
                b"binary",
            )


def test_inspect_kernel_wheel_validates_manifest_binary_closure(tmp_path) -> None:
    wheel = tmp_path / "rwkv7_kernels_test-0.8.0-cp311-cp311-linux_x86_64.whl"
    _write_wheel(wheel)

    report = inspect_wheel(wheel)

    assert report["filename"] == wheel.name
    assert len(report["sha256"]) == 64
    assert report["manifest"]["protocol"] == "rwkv7-kernel-package-v1"
    assert len(report["binary_modules"]) == 1


def test_inspect_kernel_wheel_rejects_missing_binary(tmp_path) -> None:
    wheel = tmp_path / "broken.whl"
    _write_wheel(wheel, include_binary=False)
    with pytest.raises(RuntimeError, match="binary module mismatch"):
        inspect_wheel(wheel)
