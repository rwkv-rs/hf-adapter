#!/usr/bin/env python3
"""Build final release provenance from the required compact device bundles."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.build_backend_v2_compact_bundle import validate_bundle  # noqa: E402
from scripts.release_route_contract import (  # noqa: E402
    FORMAL_REFERENCE_BACKEND_ENVIRONMENT,
    validate_actual_routes,
)
from scripts.verify_release_assets import (  # noqa: E402
    DEVICE_ORDER,
    DEVICES,
    FLA_COMMIT,
    expected_artifacts,
)


DEVICE_REPORT = "release-validation.json"
DEVICE_RUN_REPORT = "device-acceptance.json"
REPORT_SCHEMA = "rwkv7-device-release-validation-v1"
PROVENANCE_SCHEMA = "rwkv7-release-provenance-v1"
REQUIRED_GATES = (
    "correctness",
    "hf_ecosystem",
    "training",
    "quantization",
    "fla",
    "speed",
    "sft",
    "dpo",
    "grpo",
)


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--harness-sha", required=True)
    parser.add_argument(
        "--device-evidence",
        action="append",
        default=[],
        metavar="DEVICE=COMPACT_BUNDLE",
        help="repeat exactly once for rtx-4080 and rtx-4090",
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_device_evidence(rows: list[str]) -> dict[str, Path]:
    devices: dict[str, Path] = {}
    for row in rows:
        if "=" not in row:
            raise ValueError(f"invalid --device-evidence value: {row}")
        device, raw_path = row.split("=", 1)
        if device not in DEVICES:
            raise ValueError(f"unexpected release device: {device}")
        if device in devices:
            raise ValueError(f"duplicate release device: {device}")
        devices[device] = Path(raw_path).expanduser().resolve()
    if set(devices) != DEVICES:
        missing = sorted(DEVICES - set(devices))
        extra = sorted(set(devices) - DEVICES)
        raise ValueError(
            f"release devices do not match; missing={missing}, extra={extra}"
        )
    return devices


def artifact_identities(root: Path, version: str) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for name in expected_artifacts(version):
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing or unsafe release artifact: {name}")
        artifacts[name] = {"sha256": sha256_file(path), "size": path.stat().st_size}
    return artifacts


def aware_datetime(value: Any, *, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"invalid {label} timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} timestamp is not timezone-aware")
    return parsed


def read_device_report(
    *,
    device: str,
    bundle: Path,
    source_sha: str,
    harness_sha: str,
    hf_wheel_sha256: str,
    kernel_wheel_sha256: str,
) -> dict[str, Any]:
    validate_bundle(bundle)
    metadata_path = bundle / "BUNDLE.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != "rwkv7-backend-v2-compact-evidence-v1":
        raise ValueError(f"unexpected compact evidence schema: {device}")
    if metadata.get("device") != device:
        raise ValueError(f"compact evidence device mismatch: {device}")
    if metadata.get("harness_sha") != harness_sha:
        raise ValueError(f"compact evidence harness SHA mismatch: {device}")

    report_path = bundle / DEVICE_REPORT
    if not report_path.is_file() or report_path.is_symlink():
        raise ValueError(f"compact evidence is missing {DEVICE_REPORT}: {device}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError(f"unexpected device validation schema: {device}")
    if report.get("device") != device:
        raise ValueError(f"device validation identity mismatch: {device}")
    if report.get("status") != "passed":
        raise ValueError(f"device validation did not pass: {device}")
    if report.get("source_sha") != source_sha:
        raise ValueError(f"device validation source SHA mismatch: {device}")
    if report.get("harness_sha") != harness_sha:
        raise ValueError(f"device validation harness SHA mismatch: {device}")
    if report.get("fla_commit") != FLA_COMMIT:
        raise ValueError(f"device validation FLA commit mismatch: {device}")
    if report.get("hf_wheel_sha256") != hf_wheel_sha256:
        raise ValueError(f"device validation HF wheel mismatch: {device}")
    if report.get("kernel_wheel_sha256") != kernel_wheel_sha256:
        raise ValueError(f"device validation kernel wheel mismatch: {device}")
    if report.get("lm_eval_units") != 144 or report.get("lm_eval_status") != "passed":
        raise ValueError(f"formal lm_eval gate did not pass: {device}")
    for gate in REQUIRED_GATES:
        if report.get(f"{gate}_status") != "passed":
            raise ValueError(f"{gate} gate did not pass: {device}")
    if report.get("training_policy") != "reference":
        raise ValueError(f"formal reference training policy is missing: {device}")
    if (
        report.get("training_backend_environment")
        != FORMAL_REFERENCE_BACKEND_ENVIRONMENT
    ):
        raise ValueError(f"formal reference training environment differs: {device}")

    try:
        routes = validate_actual_routes(report.get("actual_routes"))
    except ValueError as exc:
        raise ValueError(f"invalid actual route evidence for {device}: {exc}") from exc

    run_path = bundle / DEVICE_RUN_REPORT
    if not run_path.is_file() or run_path.is_symlink():
        raise ValueError(f"compact evidence is missing {DEVICE_RUN_REPORT}: {device}")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if (
        run.get("schema") != "rwkv7-device-acceptance-run-v1"
        or run.get("status") != "passed"
        or run.get("device") != device
    ):
        raise ValueError(f"device acceptance run did not pass: {device}")
    for field, expected in (
        ("source_sha", source_sha),
        ("harness_sha", harness_sha),
        ("hf_wheel_sha256", hf_wheel_sha256),
        ("kernel_wheel_sha256", kernel_wheel_sha256),
        ("release_validation_sha256", sha256_file(report_path)),
    ):
        if run.get(field) != expected:
            raise ValueError(
                f"device acceptance run identity mismatch: {device}/{field}"
            )
    started_at = aware_datetime(run.get("started_at"), label=f"{device} start")
    completed_at = aware_datetime(run.get("completed_at"), label=f"{device} completion")
    if completed_at <= started_at:
        raise ValueError(f"device acceptance completion precedes start: {device}")

    manifest_sha = sha256_file(bundle / "MANIFEST.sha256")
    return {
        "status": "passed",
        "hf_wheel_sha256": hf_wheel_sha256,
        "kernel_wheel_sha256": kernel_wheel_sha256,
        "harness_sha": harness_sha,
        "lm_eval_units": 144,
        "lm_eval_status": "passed",
        "training_policy": "reference",
        "training_backend_environment": dict(FORMAL_REFERENCE_BACKEND_ENVIRONMENT),
        **{f"{gate}_status": "passed" for gate in REQUIRED_GATES},
        "compact_bundle_manifest_sha256": manifest_sha,
        "acceptance_started_at": started_at.isoformat(),
        "acceptance_completed_at": completed_at.isoformat(),
        "actual_routes": routes,
    }


def write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
    temporary.chmod(0o644)
    temporary.replace(path)


def build(args: argparse.Namespace) -> dict[str, Any]:
    root = args.directory.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"release directory does not exist: {root}")
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_sha):
        raise ValueError("source SHA must be a lowercase 40-character Git SHA")
    if not re.fullmatch(r"[0-9a-f]{40}", args.harness_sha):
        raise ValueError("harness SHA must be a lowercase 40-character Git SHA")

    artifacts = artifact_identities(root, args.version)
    hf_wheel_sha256 = artifacts[f"rwkv7_hf-{args.version}-py3-none-any.whl"]["sha256"]
    kernel_wheel_sha256 = artifacts[f"rwkv7_kernels-{args.version}-py3-none-any.whl"][
        "sha256"
    ]
    devices = {
        device: read_device_report(
            device=device,
            bundle=bundle,
            source_sha=args.source_sha,
            harness_sha=args.harness_sha,
            hf_wheel_sha256=hf_wheel_sha256,
            kernel_wheel_sha256=kernel_wheel_sha256,
        )
        for device, bundle in sorted(
            parse_device_evidence(args.device_evidence).items()
        )
    }
    for previous, following in zip(DEVICE_ORDER, DEVICE_ORDER[1:]):
        previous_completed = aware_datetime(
            devices[previous]["acceptance_completed_at"],
            label=f"{previous} completion",
        )
        following_started = aware_datetime(
            devices[following]["acceptance_started_at"],
            label=f"{following} start",
        )
        if following_started < previous_completed:
            raise ValueError(
                "device acceptance runs overlap or violate required order: "
                f"{previous} -> {following}"
            )
    provenance = {
        "schema": PROVENANCE_SCHEMA,
        "version": args.version,
        "source_sha": args.source_sha,
        "fla_commit": FLA_COMMIT,
        "harness_sha": args.harness_sha,
        "artifacts": artifacts,
        "validation": {"status": "passed", "devices": devices},
    }
    provenance_payload = (
        json.dumps(provenance, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    write_atomic(root / "release-provenance.json", provenance_payload)

    sums = [f"{row['sha256']}  {name}" for name, row in artifacts.items()]
    sums.append(
        f"{hashlib.sha256(provenance_payload).hexdigest()}  release-provenance.json"
    )
    write_atomic(root / "SHA256SUMS", ("\n".join(sums) + "\n").encode())
    return provenance


def main(argv: list[str] | None = None) -> int:
    provenance = build(arguments(argv))
    print(
        json.dumps(
            {
                "devices": sorted(provenance["validation"]["devices"]),
                "status": "passed",
                "version": provenance["version"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
