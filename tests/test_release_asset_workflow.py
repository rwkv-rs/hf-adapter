from __future__ import annotations

from argparse import Namespace
import hashlib
import json
from pathlib import Path

import pytest

from scripts.verify_release_assets import (
    DEVICES,
    FLA_COMMIT,
    expected_artifacts,
    verify,
)


def write_release(tmp_path: Path, *, mismatch_device: str | None = None) -> Namespace:
    version = "1.0.0"
    source_sha = "a" * 40
    artifacts = {}
    sums = []
    for index, name in enumerate(expected_artifacts(version)):
        path = tmp_path / name
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
        }
        for device in DEVICES
    }
    if mismatch_device:
        devices[mismatch_device]["kernel_wheel_sha256"] = "0" * 64
    provenance = {
        "schema": "rwkv7-release-provenance-v1",
        "version": version,
        "source_sha": source_sha,
        "fla_commit": FLA_COMMIT,
        "harness_sha": harness_sha,
        "artifacts": artifacts,
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


def test_release_asset_verifier_accepts_exact_three_device_artifacts(tmp_path: Path):
    report = verify(write_release(tmp_path))
    assert report["status"] == "passed"
    assert set(report["devices"]) == DEVICES


def test_release_asset_verifier_rejects_a_device_using_another_wheel(tmp_path: Path):
    args = write_release(tmp_path, mismatch_device="rtx-4090")
    with pytest.raises(ValueError, match="kernel wheel mismatch"):
        verify(args)


def test_publish_workflow_never_rebuilds_validated_artifacts():
    workflow = (Path(__file__).parents[1] / ".github/workflows/publish.yml").read_text()
    assert "gh release download" in workflow
    assert "sha256sum -c SHA256SUMS" in workflow
    assert "scripts/verify_release_assets.py" in workflow
    assert "python -m build" not in workflow
    assert "needs: [verify-assets, publish-kernels]" in workflow
