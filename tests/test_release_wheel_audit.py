from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import zipfile

import pytest

from conftest import _wheel_metadata, write_valid_hf_wheel, write_valid_kernel_wheel
from scripts.audit_release_wheels import (
    CAPABILITY_INVENTORY,
    EXPECTED_MIGRATION_TRANSFERS,
    EXPECTED_SOURCE_SCOPE_DISPOSITIONS,
    MIGRATION_MANIFEST,
    RECURRENT_SOURCE_SCOPE,
    SOURCE_SCOPE,
    audit_hf_wheel,
    audit_kernel_wheel,
)


def rewrite_wheel_member(path: Path, member: str, payload: bytes) -> None:
    with zipfile.ZipFile(path) as archive:
        members = {
            name: archive.read(name)
            for name in archive.namelist()
            if not name.endswith("/")
        }
    members[member] = payload
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in sorted(members.items()):
            archive.writestr(name, value)


def changed_record(
    path: Path,
    *,
    target: str,
    field: int | None,
    value: str = "",
) -> bytes:
    with zipfile.ZipFile(path) as archive:
        record_member = next(
            name for name in archive.namelist() if name.endswith(".dist-info/RECORD")
        )
        rows = list(csv.reader(io.StringIO(archive.read(record_member).decode())))
    if field is None:
        rows = [row for row in rows if row[0] != target]
    else:
        row = next(row for row in rows if row[0] == target)
        row[field] = value
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerows(rows)
    return stream.getvalue().encode()


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
    assert audit_hf_wheel(hf)["tool_files"] == 5
    report = audit_kernel_wheel(kernel)
    assert report["status"] == "passed"
    assert report["public_protocol"] == {
        "status": "passed",
        "api_version": 4,
        "contract_schema": "rwkv7-kernel-plugin-api-v1",
        "operations": [
            "training_program",
            "model_forward",
            "linear_training",
            "mix6_training",
            "recurrent",
        ],
        "optional_backend_entrypoint": "execute_optional_v4",
    }
    assert report["migrated_files"] == 102
    assert report["transfers"] == EXPECTED_MIGRATION_TRANSFERS
    assert sum(report["transfers"].values()) == 102
    assert report["capability_inventory"]["capabilities"] == 16
    assert report["capability_inventory"]["mapped_migration_files"] == 102
    assert report["source_scope"]["historical_files"] == 153
    assert report["source_scope"]["dispositions"] == EXPECTED_SOURCE_SCOPE_DISPOSITIONS
    assert report["recurrent_source_scope"]["historical_files"] == 3
    assert report["recurrent_source_scope"]["byte_identical_implementations"] == 2
    assert report["dependencies"] == ["ninja", "numpy", "packaging", "torch"]
    assert report["license"] == {
        "expression": "MIT",
        "file": "LICENSE",
        "member": "rwkv7_kernels-1.0.0.dist-info/licenses/LICENSE",
    }


def test_hf_wheel_audit_rejects_unrecorded_payload(tmp_path: Path):
    hf = tmp_path / "rwkv7_hf-1.0.0-py3-none-any.whl"
    write_valid_hf_wheel(hf)
    record = "rwkv7_hf-1.0.0.dist-info/RECORD"
    rewrite_wheel_member(
        hf,
        record,
        changed_record(
            hf,
            target="rwkv7_hf/modeling_rwkv7.py",
            field=None,
        ),
    )
    with pytest.raises(ValueError, match="RECORD coverage differs"):
        audit_hf_wheel(hf)


def test_hf_wheel_audit_rejects_record_hash_mismatch(tmp_path: Path):
    hf = tmp_path / "rwkv7_hf-1.0.0-py3-none-any.whl"
    write_valid_hf_wheel(hf)
    record = "rwkv7_hf-1.0.0.dist-info/RECORD"
    rewrite_wheel_member(
        hf,
        record,
        changed_record(
            hf,
            target="rwkv7_hf/modeling_rwkv7.py",
            field=1,
            value="sha256=" + "A" * 43,
        ),
    )
    with pytest.raises(ValueError, match="RECORD hash differs"):
        audit_hf_wheel(hf)


def test_hf_wheel_audit_rejects_record_size_mismatch(tmp_path: Path):
    hf = tmp_path / "rwkv7_hf-1.0.0-py3-none-any.whl"
    write_valid_hf_wheel(hf)
    record = "rwkv7_hf-1.0.0.dist-info/RECORD"
    rewrite_wheel_member(
        hf,
        record,
        changed_record(
            hf,
            target="rwkv7_hf/modeling_rwkv7.py",
            field=2,
            value="0",
        ),
    )
    with pytest.raises(ValueError, match="RECORD size differs"):
        audit_hf_wheel(hf)


def test_hf_wheel_audit_rejects_non_universal_wheel_tag(tmp_path: Path):
    hf = tmp_path / "rwkv7_hf-1.0.0-py3-none-any.whl"
    write_valid_hf_wheel(
        hf,
        extra={
            "rwkv7_hf-1.0.0.dist-info/WHEEL": (
                b"Wheel-Version: 1.0\n"
                b"Root-Is-Purelib: true\n"
                b"Tag: cp312-cp312-manylinux_2_28_x86_64\n"
            )
        },
    )
    with pytest.raises(ValueError, match="tag must be exactly py3-none-any"):
        audit_hf_wheel(hf)


def test_hf_wheel_audit_rejects_extra_top_level_import(tmp_path: Path):
    hf = tmp_path / "rwkv7_hf-1.0.0-py3-none-any.whl"
    write_valid_hf_wheel(
        hf,
        extra={
            "rwkv7_hf-1.0.0.dist-info/top_level.txt": (
                b"rwkv7_hf\nrwkv7_hf_tools\nsitecustomize\n"
            )
        },
    )
    with pytest.raises(ValueError, match="top_level.txt differs"):
        audit_hf_wheel(hf)


def test_hf_wheel_audit_rejects_malicious_console_entry_point(tmp_path: Path):
    hf = tmp_path / "rwkv7_hf-1.0.0-py3-none-any.whl"
    write_valid_hf_wheel(
        hf,
        extra={
            "rwkv7_hf-1.0.0.dist-info/entry_points.txt": (
                b"[console_scripts]\nrwkv7-hf = os:system\n"
            )
        },
    )
    with pytest.raises(ValueError, match="console entry points differ"):
        audit_hf_wheel(hf)


def test_hf_wheel_audit_rejects_added_dependency_marker(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    metadata = _wheel_metadata(
        distribution="rwkv7-hf",
        project_path=root / "pyproject.toml",
    ).replace(
        b"Requires-Dist: torch\n",
        b'Requires-Dist: torch; python_version >= "3.10"\n',
    )
    hf = tmp_path / "rwkv7_hf-1.0.0-py3-none-any.whl"
    write_valid_hf_wheel(hf, metadata=metadata)
    with pytest.raises(ValueError, match="Requires-Dist contract differs"):
        audit_hf_wheel(hf)


def test_kernel_wheel_audit_rejects_direct_url_dependency(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    metadata = _wheel_metadata(
        distribution="rwkv7-kernels",
        project_path=root / "kernels" / "pyproject.toml",
        license_expression="MIT",
        license_file="LICENSE",
    ).replace(
        b"Requires-Dist: torch\n",
        b"Requires-Dist: torch @ https://example.invalid/torch.whl\n",
    )
    kernel = tmp_path / "rwkv7_kernels-1.0.0-py3-none-any.whl"
    write_valid_kernel_wheel(kernel, metadata=metadata)
    with pytest.raises(ValueError, match="Requires-Dist contract differs"):
        audit_kernel_wheel(kernel)


def test_kernel_wheel_audit_rejects_extra_dependency(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    metadata = (
        _wheel_metadata(
            distribution="rwkv7-kernels",
            project_path=root / "kernels" / "pyproject.toml",
            license_expression="MIT",
            license_file="LICENSE",
        )
        + b"Requires-Dist: requests\n"
    )
    kernel = tmp_path / "rwkv7_kernels-1.0.0-py3-none-any.whl"
    write_valid_kernel_wheel(kernel, metadata=metadata)
    with pytest.raises(ValueError, match="Requires-Dist contract differs"):
        audit_kernel_wheel(kernel)


def test_kernel_wheel_audit_rejects_inventory_protocol_api_mismatch(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    protocol = root / "kernels/rwkv7_kernels/protocol.py"
    source = protocol.read_text(encoding="utf-8")
    payload = source.replace(
        "RWKV7_KERNEL_API_VERSION = 4",
        "RWKV7_KERNEL_API_VERSION = 3",
    )
    assert payload != source
    kernel = tmp_path / "rwkv7_kernels-1.0.0-py3-none-any.whl"
    write_valid_kernel_wheel(
        kernel,
        extra={"rwkv7_kernels/protocol.py": payload.encode()},
    )
    with pytest.raises(ValueError, match="kernel protocol API version must be 4"):
        audit_kernel_wheel(kernel)


def test_hf_wheel_audit_rejects_kernel_api_mismatch(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    ops = root / "rwkv7_hf/ops_rwkv7.py"
    source = ops.read_text(encoding="utf-8")
    payload = source.replace("_KERNEL_API_VERSION = 4", "_KERNEL_API_VERSION = 3")
    assert payload != source
    hf = tmp_path / "rwkv7_hf-1.0.0-py3-none-any.whl"
    write_valid_hf_wheel(hf, extra={"rwkv7_hf/ops_rwkv7.py": payload.encode()})
    with pytest.raises(ValueError, match="HF optional boundary kernel API version"):
        audit_hf_wheel(hf)


def test_kernel_wheel_audit_requires_public_optional_backend_entrypoint(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    init_path = root / "kernels/rwkv7_kernels/__init__.py"
    source = init_path.read_text(encoding="utf-8")
    payload = source.replace(
        '    "execute_optional_v4",\n',
        "",
    )
    assert payload != source
    kernel = tmp_path / "rwkv7_kernels-1.0.0-py3-none-any.whl"
    write_valid_kernel_wheel(
        kernel,
        extra={"rwkv7_kernels/__init__.py": payload.encode()},
    )
    with pytest.raises(ValueError, match="exactly the API-v4 public surface"):
        audit_kernel_wheel(kernel)


def test_kernel_wheel_audit_rejects_modified_plugin_contract(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    contract_path = root / "kernels/rwkv7_kernels/KERNEL_PLUGIN_API.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["public_cache_layout"] = "V,K"
    kernel = tmp_path / "rwkv7_kernels-1.0.0-py3-none-any.whl"
    write_valid_kernel_wheel(
        kernel,
        extra={
            "rwkv7_kernels/KERNEL_PLUGIN_API.json": (
                json.dumps(contract, indent=2) + "\n"
            ).encode()
        },
    )
    with pytest.raises(ValueError, match="contract JSON differs"):
        audit_kernel_wheel(kernel)


def test_kernel_wheel_audit_rejects_extra_legacy_root_export(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    init_path = root / "kernels/rwkv7_kernels/__init__.py"
    source = init_path.read_text(encoding="utf-8")
    payload = source.replace(
        '    "execute_optional_v4",\n',
        '    "execute_optional_v4",\n    "recurrent_v1",\n',
    )
    kernel = tmp_path / "rwkv7_kernels-1.0.0-py3-none-any.whl"
    write_valid_kernel_wheel(
        kernel,
        extra={"rwkv7_kernels/__init__.py": payload.encode()},
    )
    with pytest.raises(ValueError, match="exactly the API-v4 public surface"):
        audit_kernel_wheel(kernel)


def test_kernel_wheel_audit_requires_backend_entrypoint_definition(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    backend_path = root / "kernels/rwkv7_kernels/backend.py"
    source = backend_path.read_text(encoding="utf-8")
    payload = source.replace(
        "def execute_optional_v4(",
        "def removed_execute_optional_v4(",
        1,
    )
    assert payload != source
    kernel = tmp_path / "rwkv7_kernels-1.0.0-py3-none-any.whl"
    write_valid_kernel_wheel(
        kernel,
        extra={"rwkv7_kernels/backend.py": payload.encode()},
    )
    with pytest.raises(ValueError, match="does not define execute_optional_v4"):
        audit_kernel_wheel(kernel)


def test_kernel_wheel_audit_requires_runtime_preflight(tmp_path: Path):
    kernel = tmp_path / "rwkv7_kernels-1.0.0-py3-none-any.whl"
    write_valid_kernel_wheel(
        kernel,
        omit="rwkv7_kernels/_runtime_preflight.py",
    )
    with pytest.raises(ValueError, match="missing runtime files"):
        audit_kernel_wheel(kernel)


def test_kernel_wheel_audit_rejects_incomplete_direct_dependencies(tmp_path: Path):
    kernel = tmp_path / "rwkv7_kernels-1.0.0-py3-none-any.whl"
    write_valid_kernel_wheel(
        kernel,
        metadata=(
            b"Metadata-Version: 2.4\n"
            b"Name: rwkv7-kernels\n"
            b"Version: 1.0.0\n"
            b"License-Expression: MIT\n"
            b"License-File: LICENSE\n"
            b"Requires-Dist: torch\n"
        ),
    )
    with pytest.raises(ValueError, match="Requires-Dist contract differs"):
        audit_kernel_wheel(kernel)


def test_kernel_wheel_audit_rejects_missing_pep639_license_metadata(tmp_path: Path):
    kernel = tmp_path / "rwkv7_kernels-1.0.0-py3-none-any.whl"
    write_valid_kernel_wheel(
        kernel,
        metadata=(
            b"Metadata-Version: 2.4\n"
            b"Name: rwkv7-kernels\n"
            b"Version: 1.0.0\n"
            b"Requires-Dist: torch\n"
            b"Requires-Dist: numpy\n"
            b"Requires-Dist: packaging\n"
            b"Requires-Dist: ninja\n"
        ),
    )
    with pytest.raises(ValueError, match="MIT License-Expression"):
        audit_kernel_wheel(kernel)


def test_kernel_wheel_audit_rejects_missing_license_payload(tmp_path: Path):
    kernel = tmp_path / "rwkv7_kernels-1.0.0-py3-none-any.whl"
    write_valid_kernel_wheel(
        kernel,
        omit="rwkv7_kernels-1.0.0.dist-info/licenses/LICENSE",
    )
    with pytest.raises(ValueError, match="missing its declared LICENSE payload"):
        audit_kernel_wheel(kernel)


def test_kernel_wheel_audit_rejects_changed_license_payload(tmp_path: Path):
    kernel = tmp_path / "rwkv7_kernels-1.0.0-py3-none-any.whl"
    write_valid_kernel_wheel(
        kernel,
        extra={
            "rwkv7_kernels-1.0.0.dist-info/licenses/LICENSE": b"not the project license\n"
        },
    )
    with pytest.raises(ValueError, match="LICENSE payload differs from checkout"):
        audit_kernel_wheel(kernel)


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
    metadata = _wheel_metadata(
        distribution="rwkv7-hf",
        project_path=Path(__file__).resolve().parents[1] / "pyproject.toml",
    ).replace(b"rwkv7-kernels==1.0.0", b"rwkv7-kernels>=1")
    write_valid_hf_wheel(
        hf,
        metadata=metadata,
    )
    with pytest.raises(ValueError, match="Requires-Dist contract differs"):
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


def test_kernel_wheel_audit_rejects_relabelled_source_scope_counts(
    tmp_path: Path,
):
    root = Path(__file__).resolve().parents[1]
    scope_path = root / "kernels/rwkv7_kernels/nvidia/SOURCE_SCOPE.json"
    scope = json.loads(scope_path.read_text())
    row = next(
        entry
        for entry in scope["entries"]
        if entry["disposition"] == "canonical_reference"
    )
    row["disposition"] = "tooling_relocated_or_retired"
    scope["counts"]["canonical_reference"] -= 1
    scope["counts"]["tooling_relocated_or_retired"] += 1
    kernel = tmp_path / "rwkv7_kernels-1.0.0-py3-none-any.whl"
    write_valid_kernel_wheel(
        kernel,
        extra={SOURCE_SCOPE: json.dumps(scope).encode()},
    )
    with pytest.raises(ValueError, match="canonical disposition counts differ"):
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
