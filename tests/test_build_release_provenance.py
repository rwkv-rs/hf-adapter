from __future__ import annotations

from argparse import Namespace
import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest

from conftest import write_valid_hf_wheel, write_valid_kernel_wheel, write_valid_sdist
from evaluation.build_backend_v2_compact_bundle import build_bundle
from scripts.build_release_provenance import (
    DEVICE_REPORT,
    DEVICE_RUN_REPORT,
    REQUIRED_GATES,
    build,
    verify_existing,
)
from scripts.release_route_contract import (
    ADAPTIVE_TRAINING_PROGRAM_ROUTE,
    FORMAL_REFERENCE_BACKEND_ENVIRONMENT,
    HISTORICAL_WHOLE_MODEL_TRAINING_ROUTE,
    READABLE_TRAINING_MODEL_ROUTE,
    REQUIRED_REFERENCE_TRAINING_ROUTES,
)
from scripts.verify_release_assets import (
    DEVICES,
    FLA_COMMIT,
    device_evidence_archive_name,
    expected_artifacts,
    verify,
)


VERSION = "1.0.0"
SOURCE_SHA = "a" * 40
HARNESS_SHA = "b" * 40
DEVICE_TIMES = {
    "rtx-4080": ("2026-08-28T00:00:00+00:00", "2026-08-28T01:00:00+00:00"),
    "rtx-4090": ("2026-08-28T01:00:00+00:00", "2026-08-28T02:00:00+00:00"),
}


def create_artifacts(root: Path) -> dict[str, str]:
    identities = {}
    for index, name in enumerate(expected_artifacts(VERSION)):
        path = root / name
        if name.endswith(".whl") and name.startswith("rwkv7_hf-"):
            write_valid_hf_wheel(path)
        elif name.endswith(".whl") and name.startswith("rwkv7_kernels-"):
            write_valid_kernel_wheel(path)
        elif name == f"rwkv7_hf-{VERSION}.tar.gz":
            write_valid_sdist(
                path,
                wheel=root / f"rwkv7_hf-{VERSION}-py3-none-any.whl",
                distribution="rwkv7-hf",
                version=VERSION,
            )
        elif name == f"rwkv7_kernels-{VERSION}.tar.gz":
            write_valid_sdist(
                path,
                wheel=root / f"rwkv7_kernels-{VERSION}-py3-none-any.whl",
                distribution="rwkv7-kernels",
                version=VERSION,
            )
        else:
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
        "training_policy": "reference",
        "training_backend_environment": dict(FORMAL_REFERENCE_BACKEND_ENVIRONMENT),
        **{f"{gate}_status": "passed" for gate in REQUIRED_GATES},
        "actual_routes": {
            "prefill": ["native-self-chunk-prefill-v2"],
            "decode": ["native-fused-token-decode-v2"],
            "training": sorted(REQUIRED_REFERENCE_TRAINING_ROUTES),
            "quantization": ["native-w8-linear-v1", "torchao-int4-v1"],
        },
    }


def write_device_run(source: Path, device: str, report_path: Path) -> None:
    started_at, completed_at = DEVICE_TIMES[device]
    report = json.loads(report_path.read_text())
    (source / DEVICE_RUN_REPORT).write_text(
        json.dumps(
            {
                "schema": "rwkv7-device-acceptance-run-v1",
                "device": device,
                "status": "passed",
                "source_sha": report["source_sha"],
                "harness_sha": report["harness_sha"],
                "hf_wheel_sha256": report["hf_wheel_sha256"],
                "kernel_wheel_sha256": report["kernel_wheel_sha256"],
                "started_at": started_at,
                "completed_at": completed_at,
                "release_validation_sha256": hashlib.sha256(
                    report_path.read_bytes()
                ).hexdigest(),
            }
        )
        + "\n",
        encoding="utf-8",
    )


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
        report_path = source / DEVICE_REPORT
        report_path.write_text(
            json.dumps(device_report(device, identities)) + "\n", encoding="utf-8"
        )
        write_device_run(source, device, report_path)
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
    for device in DEVICES:
        assert (
            args.directory / device_evidence_archive_name(device, VERSION)
        ).is_file()
    rebuilt = verify_existing(
        Namespace(
            directory=args.directory,
            version=VERSION,
            source_sha=SOURCE_SHA,
            harness_sha=None,
            device_evidence=[],
        )
    )
    assert rebuilt == first
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


def test_final_verifier_rejects_extra_release_asset(tmp_path: Path):
    args, _, _ = setup_release(tmp_path)
    build(args)
    (args.directory / "unreviewed-wheel.whl").write_bytes(b"unreviewed")
    with pytest.raises(ValueError, match="release asset set differs.*unreviewed-wheel"):
        verify(
            Namespace(
                directory=args.directory,
                version=VERSION,
                source_sha=SOURCE_SHA,
                require_validation_passed=True,
            )
        )


def test_final_verifier_rejects_extra_checksum_row(tmp_path: Path):
    args, _, _ = setup_release(tmp_path)
    build(args)
    with (args.directory / "SHA256SUMS").open("a", encoding="utf-8") as stream:
        stream.write(f"{'0' * 64}  undeclared-artifact.bin\n")
    with pytest.raises(ValueError, match="SHA256SUMS entry set differs.*undeclared"):
        verify(
            Namespace(
                directory=args.directory,
                version=VERSION,
                source_sha=SOURCE_SHA,
                require_validation_passed=True,
            )
        )


def test_existing_verifier_rejects_rewritten_summary_without_bundle_evidence(
    tmp_path: Path,
):
    args, _, _ = setup_release(tmp_path)
    build(args)
    path = args.directory / "release-provenance.json"
    declared = json.loads(path.read_text())
    declared["validation"]["devices"]["rtx-4080"]["speed_status"] = "failed"
    path.write_text(json.dumps(declared, indent=2, sort_keys=True) + "\n")
    sums = (args.directory / "SHA256SUMS").read_text().splitlines()
    sums[-1] = f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
    (args.directory / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    with pytest.raises(ValueError, match="differs from validated compact evidence"):
        verify_existing(
            Namespace(
                directory=args.directory,
                version=VERSION,
                source_sha=SOURCE_SHA,
                harness_sha=None,
                device_evidence=[],
            )
        )


def test_existing_verifier_rejects_unsafe_compact_archive_member(tmp_path: Path):
    args, _, _ = setup_release(tmp_path)
    build(args)
    archive = args.directory / device_evidence_archive_name("rtx-4080", VERSION)
    with tarfile.open(archive, "w:gz") as tar:
        payload = b"escape\n"
        member = tarfile.TarInfo("../escape.txt")
        member.size = len(payload)
        tar.addfile(member, io.BytesIO(payload))
    with pytest.raises(ValueError, match="unsafe compact evidence archive member"):
        verify_existing(
            Namespace(
                directory=args.directory,
                version=VERSION,
                source_sha=SOURCE_SHA,
                harness_sha=None,
                device_evidence=[],
            )
        )


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
    write_device_run(source, device, source / DEVICE_REPORT)
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


def test_builder_rejects_non_reference_training_provenance(tmp_path: Path):
    args, bundles, _ = setup_release(tmp_path)
    device = "rtx-4080"
    replacement = rewrite_bundle(
        tmp_path,
        device=device,
        bundle=bundles[device],
        mutate=lambda report: report["training_backend_environment"].__setitem__(
            "RWKV7_TRAINING_KERNEL_IMPL", "adaptive"
        ),
    )
    replace_arg(args, device, replacement)
    with pytest.raises(ValueError, match="reference training environment differs"):
        build(args)


def test_builder_rejects_different_wheel(tmp_path: Path):
    args, bundles, _ = setup_release(tmp_path)
    device = "rtx-4090"
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


def test_builder_rejects_historical_whole_model_training_route(tmp_path: Path):
    args, bundles, _ = setup_release(tmp_path)
    device = "rtx-4080"
    replacement = rewrite_bundle(
        tmp_path,
        device=device,
        bundle=bundles[device],
        mutate=lambda report: report["actual_routes"].__setitem__(
            "training", [HISTORICAL_WHOLE_MODEL_TRAINING_ROUTE]
        ),
    )
    replace_arg(args, device, replacement)
    with pytest.raises(ValueError, match="historical whole-model train-temp"):
        build(args)


def test_builder_requires_all_clean_training_boundaries(tmp_path: Path):
    args, bundles, _ = setup_release(tmp_path)
    device = "rtx-4080"
    replacement = rewrite_bundle(
        tmp_path,
        device=device,
        bundle=bundles[device],
        mutate=lambda report: report["actual_routes"].__setitem__(
            "training", [READABLE_TRAINING_MODEL_ROUTE]
        ),
    )
    replace_arg(args, device, replacement)
    with pytest.raises(ValueError, match="complete reference program"):
        build(args)


def test_builder_requires_readable_model_training_boundary(tmp_path: Path):
    args, bundles, _ = setup_release(tmp_path)
    device = "rtx-4080"
    replacement = rewrite_bundle(
        tmp_path,
        device=device,
        bundle=bundles[device],
        mutate=lambda report: report["actual_routes"].__setitem__(
            "training",
            sorted(
                REQUIRED_REFERENCE_TRAINING_ROUTES
                - {READABLE_TRAINING_MODEL_ROUTE}
            ),
        ),
    )
    replace_arg(args, device, replacement)
    with pytest.raises(ValueError, match="complete reference program"):
        build(args)


def test_builder_rejects_obsolete_adaptive_training_program(tmp_path: Path):
    args, bundles, _ = setup_release(tmp_path)
    device = "rtx-4080"
    replacement = rewrite_bundle(
        tmp_path,
        device=device,
        bundle=bundles[device],
        mutate=lambda report: report["actual_routes"].__setitem__(
            "training",
            [*sorted(REQUIRED_REFERENCE_TRAINING_ROUTES), ADAPTIVE_TRAINING_PROGRAM_ROUTE],
        ),
    )
    replace_arg(args, device, replacement)
    with pytest.raises(ValueError, match="optional diagnostic routes"):
        build(args)


def test_builder_rejects_overlapping_device_acceptance(tmp_path: Path):
    args, bundles, _ = setup_release(tmp_path)
    device = "rtx-4090"
    source = tmp_path / "overlap-4090"
    source.mkdir()
    report_path = source / DEVICE_REPORT
    report_path.write_bytes((bundles[device] / DEVICE_REPORT).read_bytes())
    write_device_run(source, device, report_path)
    run = json.loads((source / DEVICE_RUN_REPORT).read_text())
    run["started_at"] = "2026-08-28T00:30:00+00:00"
    (source / DEVICE_RUN_REPORT).write_text(json.dumps(run) + "\n")
    replacement = build_bundle(
        compact_args(source, tmp_path / "overlap-bundle", device)
    )
    replace_arg(args, device, replacement)
    with pytest.raises(ValueError, match="overlap or violate required order"):
        build(args)
