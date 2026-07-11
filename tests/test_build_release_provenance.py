from __future__ import annotations

from argparse import Namespace
import hashlib
import json
from pathlib import Path

import pytest

from evaluation.build_backend_v2_compact_bundle import build_bundle
from scripts.build_release_provenance import DEVICE_REPORT, REQUIRED_GATES, build
from scripts.verify_release_assets import (
    DEVICES,
    FLA_COMMIT,
    expected_artifacts,
    verify,
)


VERSION = "1.0.0"
SOURCE_SHA = "a" * 40
HARNESS_SHA = "b" * 40


def create_artifacts(root: Path) -> dict[str, str]:
    identities = {}
    for index, name in enumerate(expected_artifacts(VERSION)):
        path = root / name
        path.write_bytes(f"release-artifact-{index}".encode())
        identities[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return identities


def device_report(device: str, identities: dict[str, str]) -> dict:
    return {
        "schema": "rwkv7-device-release-validation-v1",
        "device": device,
        "status": "passed",
        "source_sha": SOURCE_SHA,
        "harness_sha": HARNESS_SHA,
        "fla_commit": FLA_COMMIT,
        "hf_wheel_sha256": identities[f"rwkv7_hf-{VERSION}-py3-none-any.whl"],
        "kernel_wheel_sha256": identities[f"rwkv7_kernels-{VERSION}-py3-none-any.whl"],
        "lm_eval_units": 144,
        "lm_eval_status": "passed",
        **{f"{gate}_status": "passed" for gate in REQUIRED_GATES},
        "actual_routes": {
            "prefill": ["native-self-chunk-prefill-v2"],
            "decode": ["native-fused-token-decode-v2"],
            "training": ["native-nvidia-train-temp-autograd-v2"],
            "quantization": ["native-w8-linear-v1", "torchao-int4-v1"],
        },
    }


def compact_args(source: Path, output: Path, device: str) -> Namespace:
    return Namespace(
        input_dir=source,
        output_dir=output,
        device=device,
        harness_sha=HARNESS_SHA,
        max_file_mib=1.0,
    )


def setup_release(tmp_path: Path) -> tuple[Namespace, dict[str, Path], dict[str, str]]:
    release = tmp_path / "release"
    release.mkdir()
    identities = create_artifacts(release)
    bundles = {}
    for device in sorted(DEVICES):
        source = tmp_path / f"raw-{device}"
        source.mkdir()
        (source / DEVICE_REPORT).write_text(
            json.dumps(device_report(device, identities)) + "\n", encoding="utf-8"
        )
        bundles[device] = build_bundle(
            compact_args(source, tmp_path / f"bundle-{device}", device)
        )
    args = Namespace(
        directory=release,
        version=VERSION,
        source_sha=SOURCE_SHA,
        harness_sha=HARNESS_SHA,
        device_evidence=[f"{device}={path}" for device, path in bundles.items()],
    )
    return args, bundles, identities


def test_builder_generates_verifiable_deterministic_release(tmp_path: Path):
    args, _, _ = setup_release(tmp_path)
    first = build(args)
    first_provenance = (args.directory / "release-provenance.json").read_bytes()
    first_sums = (args.directory / "SHA256SUMS").read_bytes()
    second = build(args)
    assert first == second
    assert (args.directory / "release-provenance.json").read_bytes() == first_provenance
    assert (args.directory / "SHA256SUMS").read_bytes() == first_sums
    report = verify(
        Namespace(
            directory=args.directory,
            version=VERSION,
            source_sha=SOURCE_SHA,
            require_validation_passed=True,
        )
    )
    assert report["status"] == "passed"
    assert set(report["devices"]) == DEVICES


def rewrite_bundle(
    tmp_path: Path,
    *,
    device: str,
    bundle: Path,
    mutate,
) -> Path:
    report = json.loads((bundle / DEVICE_REPORT).read_text())
    mutate(report)
    source = tmp_path / f"rewritten-{device}"
    source.mkdir()
    (source / DEVICE_REPORT).write_text(json.dumps(report) + "\n")
    replacement = tmp_path / f"replacement-{device}"
    return build_bundle(compact_args(source, replacement, device))


def replace_arg(args: Namespace, device: str, bundle: Path) -> None:
    args.device_evidence = [
        f"{name}={bundle if name == device else path}"
        for name, raw in (row.split("=", 1) for row in args.device_evidence)
        for path in [Path(raw)]
    ]


def test_builder_rejects_missing_gate(tmp_path: Path):
    args, bundles, _ = setup_release(tmp_path)
    device = "rtx-4080"
    replacement = rewrite_bundle(
        tmp_path,
        device=device,
        bundle=bundles[device],
        mutate=lambda report: report.pop("training_status"),
    )
    replace_arg(args, device, replacement)
    with pytest.raises(ValueError, match="training gate did not pass"):
        build(args)


def test_builder_rejects_different_wheel(tmp_path: Path):
    args, bundles, _ = setup_release(tmp_path)
    device = "tesla-v100"
    replacement = rewrite_bundle(
        tmp_path,
        device=device,
        bundle=bundles[device],
        mutate=lambda report: report.__setitem__("kernel_wheel_sha256", "0" * 64),
    )
    replace_arg(args, device, replacement)
    with pytest.raises(ValueError, match="kernel wheel mismatch"):
        build(args)


def test_builder_rejects_wrong_harness(tmp_path: Path):
    args, _, _ = setup_release(tmp_path)
    args.harness_sha = "c" * 40
    with pytest.raises(ValueError, match="compact evidence harness SHA mismatch"):
        build(args)


def test_builder_rejects_invalid_compact_manifest(tmp_path: Path):
    args, bundles, _ = setup_release(tmp_path)
    (bundles["rtx-4090"] / DEVICE_REPORT).write_text("{}\n")
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        build(args)


def test_builder_rejects_requested_selector_as_actual_route(tmp_path: Path):
    args, bundles, _ = setup_release(tmp_path)
    device = "rtx-4090"
    replacement = rewrite_bundle(
        tmp_path,
        device=device,
        bundle=bundles[device],
        mutate=lambda report: report["actual_routes"].__setitem__("prefill", "auto"),
    )
    replace_arg(args, device, replacement)
    with pytest.raises(ValueError, match="requested selector"):
        build(args)
