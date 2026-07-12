from __future__ import annotations

import json
from pathlib import Path

import pytest

from conftest import write_valid_hf_wheel, write_valid_kernel_wheel
from scripts.audit_release_wheels import (
    CAPABILITY_INVENTORY,
    MIGRATION_MANIFEST,
    RECURRENT_SOURCE_SCOPE,
    SOURCE_SCOPE,
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
    assert report["transfers"] == {
        "adapted_clean_boundary": 2,
        "byte_identical": 100,
    }
    assert report["capability_inventory"]["capabilities"] == 16
    assert report["capability_inventory"]["mapped_migration_files"] == 102
    assert report["source_scope"]["historical_files"] == 153
    assert report["source_scope"]["dispositions"]["byte_migrated_nvidia"] == 100
    assert report["source_scope"]["dispositions"]["adapted_protocol"] == 12
    assert report["recurrent_source_scope"]["historical_files"] == 3
    assert report["recurrent_source_scope"]["byte_identical_implementations"] == 2


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


def test_hf_wheel_audit_rejects_unpinned_kernel_extra(tmp_path: Path):
    hf = tmp_path / "rwkv7_hf-1.0.0-py3-none-any.whl"
    write_valid_hf_wheel(
        hf,
        metadata=(
            "Metadata-Version: 2.4\n"
            "Name: rwkv7-hf\n"
            "Version: 1.0.0\n"
            "Provides-Extra: kernels\n"
            'Requires-Dist: rwkv7-kernels>=1; extra == "kernels"\n'
        ).encode(),
    )
    with pytest.raises(ValueError, match="extra is not pinned to 1.0.0"):
        audit_hf_wheel(hf)


def test_kernel_wheel_audit_rejects_false_migrated_git_blob(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "kernels/rwkv7_kernels/nvidia/MIGRATION_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][0]["git_blob"] = "0" * 40
    kernel = tmp_path / "rwkv7_kernels-1.0.0-py3-none-any.whl"
    write_valid_kernel_wheel(
        kernel,
        extra={MIGRATION_MANIFEST: json.dumps(manifest).encode()},
    )
    with pytest.raises(ValueError, match="source Git blob mismatch"):
        audit_kernel_wheel(kernel)


def test_kernel_wheel_audit_rejects_undeclared_clean_boundary_adaptation(
    tmp_path: Path,
):
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "kernels/rwkv7_kernels/nvidia/MIGRATION_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    exact = next(
        row for row in manifest["files"] if row["transfer"] == "byte_identical"
    )
    exact["transfer"] = "adapted_clean_boundary"
    exact["adaptation"] = "undeclared adaptation"
    kernel = tmp_path / "rwkv7_kernels-1.0.0-py3-none-any.whl"
    write_valid_kernel_wheel(
        kernel,
        extra={MIGRATION_MANIFEST: json.dumps(manifest).encode()},
    )
    with pytest.raises(ValueError, match="unexpected clean-boundary adaptation"):
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


def test_kernel_wheel_audit_rejects_missing_capability_inventory(tmp_path: Path):
    kernel = tmp_path / "rwkv7_kernels-1.0.0-py3-none-any.whl"
    write_valid_kernel_wheel(kernel, omit=CAPABILITY_INVENTORY)
    with pytest.raises(ValueError, match="missing NVIDIA capability inventory"):
        audit_kernel_wheel(kernel)


def test_kernel_wheel_audit_rejects_missing_historical_source_scope(tmp_path: Path):
    kernel = tmp_path / "rwkv7_kernels-1.0.0-py3-none-any.whl"
    write_valid_kernel_wheel(kernel, omit=SOURCE_SCOPE)
    with pytest.raises(ValueError, match="missing historical source-scope"):
        audit_kernel_wheel(kernel)


def test_kernel_wheel_audit_rejects_missing_recurrent_source_scope(tmp_path: Path):
    kernel = tmp_path / "rwkv7_kernels-1.0.0-py3-none-any.whl"
    write_valid_kernel_wheel(kernel, omit=RECURRENT_SOURCE_SCOPE)
    with pytest.raises(ValueError, match="missing recurrent source-scope"):
        audit_kernel_wheel(kernel)


def test_kernel_wheel_audit_rejects_unmapped_migrated_source(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    inventory_path = root / "kernels/rwkv7_kernels/nvidia/CAPABILITY_INVENTORY.json"
    inventory = json.loads(inventory_path.read_text())
    inventory["capabilities"][0]["migration_files"].pop()
    kernel = tmp_path / "rwkv7_kernels-1.0.0-py3-none-any.whl"
    write_valid_kernel_wheel(
        kernel,
        extra={CAPABILITY_INVENTORY: json.dumps(inventory).encode()},
    )
    with pytest.raises(ValueError, match="capability migration coverage differs"):
        audit_kernel_wheel(kernel)


def test_kernel_wheel_audit_rejects_unknown_policy_flag(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    inventory_path = root / "kernels/rwkv7_kernels/nvidia/CAPABILITY_INVENTORY.json"
    inventory = json.loads(inventory_path.read_text())
    inventory["capabilities"][0]["policy_flags"].append("imaginary_kernel_flag")
    kernel = tmp_path / "rwkv7_kernels-1.0.0-py3-none-any.whl"
    write_valid_kernel_wheel(
        kernel,
        extra={CAPABILITY_INVENTORY: json.dumps(inventory).encode()},
    )
    with pytest.raises(ValueError, match="unknown policy flags"):
        audit_kernel_wheel(kernel)


def test_kernel_wheel_audit_rejects_unclassified_historical_source(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    scope_path = root / "kernels/rwkv7_kernels/nvidia/SOURCE_SCOPE.json"
    scope = json.loads(scope_path.read_text())
    scope["entries"][0]["disposition"] = "unclassified"
    kernel = tmp_path / "rwkv7_kernels-1.0.0-py3-none-any.whl"
    write_valid_kernel_wheel(
        kernel,
        extra={SOURCE_SCOPE: json.dumps(scope).encode()},
    )
    with pytest.raises(ValueError, match="unknown dispositions"):
        audit_kernel_wheel(kernel)


def test_kernel_wheel_audit_rejects_changed_historical_tree_identity(
    tmp_path: Path,
):
    root = Path(__file__).resolve().parents[1]
    scope_path = root / "kernels/rwkv7_kernels/nvidia/SOURCE_SCOPE.json"
    scope = json.loads(scope_path.read_text())
    scope["entries"][0]["git_blob"] = "0" * 40
    kernel = tmp_path / "rwkv7_kernels-1.0.0-py3-none-any.whl"
    write_valid_kernel_wheel(
        kernel,
        extra={SOURCE_SCOPE: json.dumps(scope).encode()},
    )
    with pytest.raises(ValueError, match="do not reconstruct the frozen Git tree"):
        audit_kernel_wheel(kernel)


def test_kernel_wheel_audit_rejects_changed_recurrent_blob_identity(
    tmp_path: Path,
):
    root = Path(__file__).resolve().parents[1]
    scope_path = root / "kernels/rwkv7_kernels/nvidia/RECURRENT_SOURCE_SCOPE.json"
    scope = json.loads(scope_path.read_text())
    scope["entries"][1]["git_blob"] = "0" * 40
    kernel = tmp_path / "rwkv7_kernels-1.0.0-py3-none-any.whl"
    write_valid_kernel_wheel(
        kernel,
        extra={RECURRENT_SOURCE_SCOPE: json.dumps(scope).encode()},
    )
    with pytest.raises(ValueError, match="do not reconstruct the frozen Git tree"):
        audit_kernel_wheel(kernel)


def test_kernel_wheel_audit_rejects_mismatched_adaptation_rationale(
    tmp_path: Path,
):
    root = Path(__file__).resolve().parents[1]
    scope_path = root / "kernels/rwkv7_kernels/nvidia/SOURCE_SCOPE.json"
    scope = json.loads(scope_path.read_text())
    adapted = next(
        row
        for row in scope["entries"]
        if row.get("destination") and row["disposition"] == "adapted_protocol"
    )
    adapted["adaptation"] = "different rationale"
    kernel = tmp_path / "rwkv7_kernels-1.0.0-py3-none-any.whl"
    write_valid_kernel_wheel(
        kernel,
        extra={SOURCE_SCOPE: json.dumps(scope).encode()},
    )
    with pytest.raises(ValueError, match="historical NVIDIA migration scope differs"):
        audit_kernel_wheel(kernel)
