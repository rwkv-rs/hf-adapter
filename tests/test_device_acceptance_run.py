from __future__ import annotations

from argparse import Namespace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest

from evaluation.record_device_acceptance import finish_report, start_report


def args_start(tmp_path: Path) -> Namespace:
    hf = tmp_path / "rwkv7_hf-1.0.0-py3-none-any.whl"
    kernels = tmp_path / "rwkv7_kernels-1.0.0-py3-none-any.whl"
    hf.write_bytes(b"hf")
    kernels.write_bytes(b"kernels")
    return Namespace(
        device="rtx-4080",
        source_sha="a" * 40,
        harness_sha="b" * 40,
        hf_wheel=hf,
        kernel_wheel=kernels,
        output=tmp_path / "device-acceptance.json",
    )


def write_validation(path: Path, run: dict) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "rwkv7-device-release-validation-v1",
                "status": "passed",
                **{
                    field: run[field]
                    for field in (
                        "device",
                        "source_sha",
                        "harness_sha",
                        "hf_wheel_sha256",
                        "kernel_wheel_sha256",
                    )
                },
            }
        )
        + "\n"
    )


def test_device_acceptance_binds_passed_validation_and_wheel_lifetime(tmp_path: Path):
    start_args = args_start(tmp_path)
    running = start_report(
        start_args, now=datetime(2026, 8, 28, tzinfo=timezone.utc)
    )
    validation = tmp_path / "release-validation.json"
    write_validation(validation, running)
    passed = finish_report(
        Namespace(run_report=start_args.output, release_validation=validation),
        now=datetime(2026, 8, 28, 1, tzinfo=timezone.utc),
    )
    assert passed["status"] == "passed"
    assert passed["release_validation_sha256"] == hashlib.sha256(
        validation.read_bytes()
    ).hexdigest()


def test_device_acceptance_rejects_validation_from_another_wheel(tmp_path: Path):
    start_args = args_start(tmp_path)
    running = start_report(
        start_args, now=datetime(2026, 8, 28, tzinfo=timezone.utc)
    )
    validation = tmp_path / "release-validation.json"
    write_validation(validation, running)
    payload = json.loads(validation.read_text())
    payload["kernel_wheel_sha256"] = "0" * 64
    validation.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="kernel_wheel_sha256"):
        finish_report(
            Namespace(run_report=start_args.output, release_validation=validation),
            now=datetime(2026, 8, 28, 1, tzinfo=timezone.utc),
        )
