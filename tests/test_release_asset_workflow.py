from __future__ import annotations

from argparse import Namespace
import hashlib
import json
from pathlib import Path
import tarfile
import zipfile

import pytest

from conftest import write_valid_hf_wheel, write_valid_kernel_wheel, write_valid_sdist
from scripts.verify_release_assets import (
    DEVICES,
    FLA_COMMIT,
    audit_sdist,
    audit_wheel_against_checkout,
    device_evidence_archive_name,
    expected_artifacts,
    verify,
)
from scripts.release_route_contract import (
    FORMAL_REFERENCE_BACKEND_ENVIRONMENT,
    REQUIRED_REFERENCE_TRAINING_ROUTES,
)


def write_release(tmp_path: Path, *, mismatch_device: str | None = None) -> Namespace:
    version = "1.0.0"
    source_sha = "a" * 40
    artifacts = {}
    sums = []
    for index, name in enumerate(expected_artifacts(version)):
        path = tmp_path / name
        if name.endswith(".whl") and name.startswith("rwkv7_hf-"):
            write_valid_hf_wheel(path)
        elif name.endswith(".whl") and name.startswith("rwkv7_kernels-"):
            write_valid_kernel_wheel(path)
        elif name == f"rwkv7_hf-{version}.tar.gz":
            write_valid_sdist(
                path,
                wheel=tmp_path / f"rwkv7_hf-{version}-py3-none-any.whl",
                distribution="rwkv7-hf",
                version=version,
            )
        elif name == f"rwkv7_kernels-{version}.tar.gz":
            write_valid_sdist(
                path,
                wheel=tmp_path / f"rwkv7_kernels-{version}-py3-none-any.whl",
                distribution="rwkv7-kernels",
                version=version,
            )
        else:
            path.write_bytes(f"artifact-{index}".encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        artifacts[name] = {"sha256": digest, "size": path.stat().st_size}
        sums.append(f"{digest}  {name}")
    hf_wheel = artifacts[f"rwkv7_hf-{version}-py3-none-any.whl"]["sha256"]
    kernel_wheel = artifacts[f"rwkv7_kernels-{version}-py3-none-any.whl"]["sha256"]
    harness_sha = "b" * 40
    devices = {
        device: {
            "status": "passed",
            "hf_wheel_sha256": hf_wheel,
            "kernel_wheel_sha256": kernel_wheel,
            "harness_sha": harness_sha,
            "lm_eval_units": 144,
            "lm_eval_status": "passed",
            "training_policy": "reference",
            "training_backend_environment": dict(FORMAL_REFERENCE_BACKEND_ENVIRONMENT),
            "correctness_status": "passed",
            "hf_ecosystem_status": "passed",
            "training_status": "passed",
            "quantization_status": "passed",
            "fla_status": "passed",
            "speed_status": "passed",
            "sft_status": "passed",
            "dpo_status": "passed",
            "grpo_status": "passed",
            "compact_bundle_manifest_sha256": hashlib.sha256(
                device.encode()
            ).hexdigest(),
            "acceptance_started_at": {
                "rtx-4080": "2026-08-28T00:00:00+00:00",
                "rtx-4090": "2026-08-28T01:00:00+00:00",
            }[device],
            "acceptance_completed_at": {
                "rtx-4080": "2026-08-28T01:00:00+00:00",
                "rtx-4090": "2026-08-28T02:00:00+00:00",
            }[device],
            "actual_routes": {
                "prefill": ["native-self-chunk-prefill-v2"],
                "decode": ["native-fused-token-decode-v2"],
                "training": sorted(REQUIRED_REFERENCE_TRAINING_ROUTES),
                "quantization": ["native-w8-linear-v1"],
            },
        }
        for device in DEVICES
    }
    if mismatch_device:
        devices[mismatch_device]["kernel_wheel_sha256"] = "0" * 64
    evidence = {}
    for device in sorted(DEVICES):
        name = device_evidence_archive_name(device, version)
        path = tmp_path / name
        path.write_bytes(f"compact-evidence-{device}".encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        sums.append(f"{digest}  {name}")
        evidence[device] = {
            "archive": name,
            "sha256": digest,
            "size": path.stat().st_size,
            "compact_bundle_manifest_sha256": devices[device][
                "compact_bundle_manifest_sha256"
            ],
        }
    provenance = {
        "schema": "rwkv7-release-provenance-v2",
        "version": version,
        "source_sha": source_sha,
        "fla_commit": FLA_COMMIT,
        "harness_sha": harness_sha,
        "artifacts": artifacts,
        "evidence": evidence,
        "validation": {"status": "passed", "devices": devices},
    }
    provenance_path = tmp_path / "release-provenance.json"
    provenance_path.write_text(json.dumps(provenance) + "\n")
    sums.append(
        f"{hashlib.sha256(provenance_path.read_bytes()).hexdigest()}  {provenance_path.name}"
    )
    (tmp_path / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    return Namespace(
        directory=tmp_path,
        version=version,
        source_sha=source_sha,
        require_validation_passed=True,
    )


def test_release_asset_verifier_accepts_exact_required_device_artifacts(tmp_path: Path):
    report = verify(write_release(tmp_path))
    assert report["status"] == "passed"
    assert set(report["devices"]) == DEVICES
    assert set(report["evidence"]) == DEVICES


def test_release_asset_verifier_rejects_a_device_using_another_wheel(tmp_path: Path):
    args = write_release(tmp_path, mismatch_device="rtx-4090")
    with pytest.raises(ValueError, match="kernel wheel mismatch"):
        verify(args)


def test_release_asset_verifier_rejects_overlapping_device_order(tmp_path: Path):
    args = write_release(tmp_path)
    path = tmp_path / "release-provenance.json"
    provenance = json.loads(path.read_text())
    provenance["validation"]["devices"]["rtx-4090"]["acceptance_started_at"] = (
        "2026-08-28T00:30:00+00:00"
    )
    path.write_text(json.dumps(provenance) + "\n")
    sums = (tmp_path / "SHA256SUMS").read_text().splitlines()
    sums[-1] = f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
    (tmp_path / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    with pytest.raises(ValueError, match="overlap or violate required order"):
        verify(args)


def test_checkout_audit_rejects_wheel_payload_from_another_source(tmp_path: Path):
    wheel = tmp_path / "bad.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("rwkv7_hf/modeling_rwkv7.py", b"not tagged source")
    with pytest.raises(ValueError, match="differs from checkout"):
        audit_wheel_against_checkout(
            wheel,
            mappings=(("rwkv7_hf/", Path(__file__).parents[1] / "rwkv7_hf"),),
        )


def test_checkout_audit_rejects_unowned_top_level_wheel_payload(tmp_path: Path):
    wheel = tmp_path / "rwkv7_hf-1.0.0-py3-none-any.whl"
    write_valid_hf_wheel(wheel, extra={"sitecustomize.py": b"raise RuntimeError\n"})

    with pytest.raises(ValueError, match="unowned payload.*sitecustomize.py"):
        audit_wheel_against_checkout(
            wheel,
            mappings=(
                ("rwkv7_hf/", Path(__file__).parents[1] / "rwkv7_hf"),
                ("rwkv7_hf_tools/", Path(__file__).parents[1] / "rwkv7_hf_tools"),
            ),
        )


def test_sdist_audit_rejects_payload_that_differs_from_wheel(tmp_path: Path):
    wheel = tmp_path / "rwkv7_hf-1.0.0-py3-none-any.whl"
    sdist = tmp_path / "rwkv7_hf-1.0.0.tar.gz"
    write_valid_hf_wheel(wheel)
    write_valid_sdist(
        sdist,
        wheel=wheel,
        distribution="rwkv7-hf",
        replace={"rwkv7_hf/modeling_rwkv7.py": b"different"},
    )
    with pytest.raises(ValueError, match="payload differs from wheel"):
        audit_sdist(
            sdist,
            distribution="rwkv7-hf",
            version="1.0.0",
            wheel=wheel,
            package_prefixes=("rwkv7_hf/", "rwkv7_hf_tools/"),
            forbidden_prefix="rwkv7_kernels/",
        )


def test_sdist_audit_rejects_unowned_build_hook(tmp_path: Path):
    wheel = tmp_path / "rwkv7_hf-1.0.0-py3-none-any.whl"
    sdist = tmp_path / "rwkv7_hf-1.0.0.tar.gz"
    write_valid_hf_wheel(wheel)
    write_valid_sdist(
        sdist,
        wheel=wheel,
        distribution="rwkv7-hf",
        replace={"setup.py": b'raise RuntimeError("unvalidated build hook")\n'},
    )

    with pytest.raises(ValueError, match="unowned payload.*setup.py"):
        audit_sdist(
            sdist,
            distribution="rwkv7-hf",
            version="1.0.0",
            wheel=wheel,
            package_prefixes=("rwkv7_hf/", "rwkv7_hf_tools/"),
            forbidden_prefix="rwkv7_kernels/",
        )


def test_sdist_audit_rejects_modified_pyproject(tmp_path: Path):
    wheel = tmp_path / "rwkv7_hf-1.0.0-py3-none-any.whl"
    sdist = tmp_path / "rwkv7_hf-1.0.0.tar.gz"
    write_valid_hf_wheel(wheel)
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    write_valid_sdist(
        sdist,
        wheel=wheel,
        distribution="rwkv7-hf",
        replace={"pyproject.toml": (pyproject + "\n# unowned change\n").encode()},
    )

    with pytest.raises(ValueError, match="differs from checkout.*pyproject.toml"):
        audit_sdist(
            sdist,
            distribution="rwkv7-hf",
            version="1.0.0",
            wheel=wheel,
            package_prefixes=("rwkv7_hf/", "rwkv7_hf_tools/"),
            forbidden_prefix="rwkv7_kernels/",
        )


def test_sdist_audit_rejects_symbolic_link_member(tmp_path: Path):
    wheel = tmp_path / "rwkv7_hf-1.0.0-py3-none-any.whl"
    sdist = tmp_path / "rwkv7_hf-1.0.0.tar.gz"
    write_valid_hf_wheel(wheel)
    with tarfile.open(sdist, "w:gz") as archive:
        member = tarfile.TarInfo("rwkv7_hf-1.0.0/rwkv7_hf/modeling_rwkv7.py")
        member.type = tarfile.SYMTYPE
        member.linkname = "/etc/passwd"
        archive.addfile(member)
    with pytest.raises(ValueError, match="non-regular source-distribution member"):
        audit_sdist(
            sdist,
            distribution="rwkv7-hf",
            version="1.0.0",
            wheel=wheel,
            package_prefixes=("rwkv7_hf/", "rwkv7_hf_tools/"),
            forbidden_prefix="rwkv7_kernels/",
        )


def test_kernel_sdist_audit_rejects_hf_tooling_ownership(tmp_path: Path):
    wheel = tmp_path / "rwkv7_kernels-1.0.0-py3-none-any.whl"
    sdist = tmp_path / "rwkv7_kernels-1.0.0.tar.gz"
    write_valid_kernel_wheel(wheel)
    write_valid_sdist(
        sdist,
        wheel=wheel,
        distribution="rwkv7-kernels",
        replace={"rwkv7_hf_tools/cli.py": b"wrong owner"},
    )
    with pytest.raises(ValueError, match="crosses package ownership"):
        audit_sdist(
            sdist,
            distribution="rwkv7-kernels",
            version="1.0.0",
            wheel=wheel,
            package_prefixes=("rwkv7_kernels/",),
            forbidden_prefix="rwkv7_hf",
        )


def test_kernel_sdist_audit_rejects_missing_declared_license(tmp_path: Path):
    wheel = tmp_path / "rwkv7_kernels-1.0.0-py3-none-any.whl"
    sdist = tmp_path / "rwkv7_kernels-1.0.0.tar.gz"
    write_valid_kernel_wheel(wheel)
    write_valid_sdist(
        sdist,
        wheel=wheel,
        distribution="rwkv7-kernels",
        omit="LICENSE",
    )
    with pytest.raises(ValueError, match="omitted checkout-owned payload.*LICENSE"):
        audit_sdist(
            sdist,
            distribution="rwkv7-kernels",
            version="1.0.0",
            wheel=wheel,
            package_prefixes=("rwkv7_kernels/",),
            forbidden_prefix="rwkv7_hf",
        )


def test_publish_workflow_never_rebuilds_validated_artifacts():
    workflow = (Path(__file__).parents[1] / ".github/workflows/publish.yml").read_text()
    assert "gh release download" in workflow
    assert "sha256sum -c SHA256SUMS" in workflow
    assert "scripts/verify_release_assets.py" in workflow
    assert "scripts/build_release_provenance.py" in workflow
    assert "--verify-existing" in workflow
    assert "rwkv7-evidence-rtx-4080-${VERSION}.tar.gz" in workflow
    assert "rwkv7-evidence-rtx-4090-${VERSION}.tar.gz" in workflow
    assert "python -m build" not in workflow
    assert "needs: [verify-assets, publish-kernels]" in workflow
    assert 'python -m pip install "twine==7.0.0"' in workflow
    for action_sha in (
        "actions/checkout@11d5960a326750d5838078e36cf38b85af677262",
        "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065",
        "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02",
        "actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093",
        "pypa/gh-action-pypi-publish@dc37677b2e1c63e2034f94d8a5b11f265b73ba33",
    ):
        assert action_sha in workflow
