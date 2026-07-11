from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import write_valid_hf_wheel, write_valid_kernel_wheel
from scripts.audit_release_wheels import (
    MIGRATION_MANIFEST,
    audit_hf_wheel,
    audit_kernel_wheel,
)


def first_migrated_member() -> str:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "kernels/rwkv7_kernels/nvidia/MIGRATION_MANIFEST.json").read_text()
    )
    destination = Path(manifest["files"][0]["destination"])
    return destination.relative_to("kernels").as_posix()


def test_release_wheel_audit_accepts_clean_hf_and_all_102_sources(tmp_path: Path):
    hf = tmp_path / "rwkv7_hf-1.0.0-py3-none-any.whl"
    kernel = tmp_path / "rwkv7_kernels-1.0.0-py3-none-any.whl"
    write_valid_hf_wheel(hf)
    write_valid_kernel_wheel(kernel)
    assert audit_hf_wheel(hf)["status"] == "passed"
    report = audit_kernel_wheel(kernel)
    assert report["status"] == "passed"
    assert report["migrated_files"] == 102


def test_kernel_wheel_audit_rejects_omitted_migrated_source(tmp_path: Path):
    kernel = tmp_path / "rwkv7_kernels-1.0.0-py3-none-any.whl"
    member = first_migrated_member()
    write_valid_kernel_wheel(kernel, omit=member)
    with pytest.raises(ValueError, match="omitted migrated source"):
        audit_kernel_wheel(kernel)


def test_kernel_wheel_audit_rejects_changed_migrated_bytes(tmp_path: Path):
    kernel = tmp_path / "rwkv7_kernels-1.0.0-py3-none-any.whl"
    member = first_migrated_member()
    write_valid_kernel_wheel(kernel, tamper=member)
    with pytest.raises(ValueError, match="source hash mismatch"):
        audit_kernel_wheel(kernel)


def test_wheel_audit_rejects_cross_package_ownership(tmp_path: Path):
    hf = tmp_path / "rwkv7_hf-1.0.0-py3-none-any.whl"
    kernel = tmp_path / "rwkv7_kernels-1.0.0-py3-none-any.whl"
    write_valid_hf_wheel(hf, extra={"rwkv7_kernels/protocol.py": b"bad"})
    write_valid_kernel_wheel(kernel, extra={"rwkv7_hf/modeling_rwkv7.py": b"bad"})
    with pytest.raises(ValueError, match="optional kernel package"):
        audit_hf_wheel(hf)
    with pytest.raises(ValueError, match="Hugging Face model package"):
        audit_kernel_wheel(kernel)


def test_kernel_wheel_audit_rejects_missing_manifest(tmp_path: Path):
    kernel = tmp_path / "rwkv7_kernels-1.0.0-py3-none-any.whl"
    write_valid_kernel_wheel(kernel, omit=MIGRATION_MANIFEST)
    with pytest.raises(ValueError, match="missing NVIDIA migration manifest"):
        audit_kernel_wheel(kernel)
