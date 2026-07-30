#!/usr/bin/env python3
"""Build final release provenance from the required compact device bundles."""

from __future__ import annotations

import argparse
from datetime import datetime
import gzip
import hashlib
import io
import json
from pathlib import Path
from pathlib import PurePosixPath
import re
import sys
import tarfile
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
    device_evidence_archive_name,
    expected_artifacts,
)


DEVICE_REPORT = "release-validation.json"
DEVICE_RUN_REPORT = "device-acceptance.json"
REPORT_SCHEMA = "rwkv7-device-release-validation-v1"
PROVENANCE_SCHEMA = "rwkv7-release-provenance-v2"
MAX_EVIDENCE_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_EVIDENCE_MEMBER_BYTES = 8 * 1024 * 1024
MAX_EVIDENCE_MEMBERS = 10_000
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
    parser.add_argument("--harness-sha")
    parser.add_argument(
        "--device-evidence",
        action="append",
        default=[],
        metavar="DEVICE=COMPACT_BUNDLE",
        help="repeat exactly once for rtx-4080 and rtx-4090",
    )
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help=(
            "validate the two archived compact bundles and byte-compare rebuilt "
            "provenance/checksums instead of writing release metadata"
        ),
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


def evidence_archive_root(device: str, version: str) -> str:
    return device_evidence_archive_name(device, version).removesuffix(".tar.gz")


def write_device_evidence_archive(
    *,
    bundle: Path,
    archive: Path,
    device: str,
    version: str,
) -> None:
    """Write one deterministic, regular-file-only compact evidence archive."""

    validate_bundle(bundle)
    files = sorted(path for path in bundle.rglob("*") if path.is_file())
    if len(files) > MAX_EVIDENCE_MEMBERS:
        raise ValueError(f"compact evidence contains too many files: {device}")
    root_name = evidence_archive_root(device, version)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=archive.parent, delete=False) as stream:
        temporary = Path(stream.name)
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=9,
            fileobj=stream,
            mtime=0,
        ) as compressed:
            with tarfile.open(
                fileobj=compressed,
                mode="w",
                format=tarfile.PAX_FORMAT,
            ) as tar:
                for path in files:
                    relative = path.relative_to(bundle)
                    if path.is_symlink() or not path.is_file():
                        raise ValueError(
                            f"compact evidence path is not a regular file: {relative}"
                        )
                    payload = path.read_bytes()
                    if len(payload) > MAX_EVIDENCE_MEMBER_BYTES:
                        raise ValueError(
                            f"compact evidence file is too large: {device}/{relative}"
                        )
                    info = tarfile.TarInfo(
                        f"{root_name}/{relative.as_posix()}"
                    )
                    info.size = len(payload)
                    info.mode = 0o644
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    tar.addfile(info, io.BytesIO(payload))
        stream.flush()
    if temporary.stat().st_size > MAX_EVIDENCE_ARCHIVE_BYTES:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"compact evidence archive is too large: {device}")
    temporary.chmod(0o644)
    temporary.replace(archive)


def extract_device_evidence_archive(
    *,
    archive: Path,
    output: Path,
    device: str,
    version: str,
) -> Path:
    """Safely unpack a release evidence archive and validate its manifest."""

    if not archive.is_file() or archive.is_symlink():
        raise ValueError(f"missing or unsafe compact evidence archive: {device}")
    if archive.stat().st_size > MAX_EVIDENCE_ARCHIVE_BYTES:
        raise ValueError(f"compact evidence archive is too large: {device}")
    expected_root = evidence_archive_root(device, version)
    seen: set[str] = set()
    member_count = 0
    with tarfile.open(archive, mode="r:gz") as tar:
        for member in tar:
            member_count += 1
            if member_count > MAX_EVIDENCE_MEMBERS:
                raise ValueError(f"compact evidence archive has too many members: {device}")
            name = PurePosixPath(member.name)
            if (
                name.is_absolute()
                or ".." in name.parts
                or len(name.parts) < 2
                or name.parts[0] != expected_root
            ):
                raise ValueError(
                    f"unsafe compact evidence archive member: {device}/{member.name}"
                )
            if not member.isfile():
                raise ValueError(
                    f"non-regular compact evidence archive member: "
                    f"{device}/{member.name}"
                )
            relative = PurePosixPath(*name.parts[1:]).as_posix()
            if relative in seen:
                raise ValueError(
                    f"duplicate compact evidence archive member: {device}/{relative}"
                )
            seen.add(relative)
            if member.size > MAX_EVIDENCE_MEMBER_BYTES:
                raise ValueError(
                    f"compact evidence archive member is too large: {device}/{relative}"
                )
            source = tar.extractfile(member)
            if source is None:
                raise ValueError(
                    f"unreadable compact evidence archive member: {device}/{relative}"
                )
            payload = source.read(MAX_EVIDENCE_MEMBER_BYTES + 1)
            if len(payload) != member.size or len(payload) > MAX_EVIDENCE_MEMBER_BYTES:
                raise ValueError(
                    f"invalid compact evidence archive member size: {device}/{relative}"
                )
            destination = output / expected_root / Path(*PurePosixPath(relative).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
            destination.chmod(0o644)
    bundle = output / expected_root
    validate_bundle(bundle)
    return bundle


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


def validate_release_identity(
    *, root: Path, source_sha: str, harness_sha: str
) -> None:
    if not root.is_dir():
        raise ValueError(f"release directory does not exist: {root}")
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("source SHA must be a lowercase 40-character Git SHA")
    if not re.fullmatch(r"[0-9a-f]{40}", harness_sha):
        raise ValueError("harness SHA must be a lowercase 40-character Git SHA")


def compose_provenance(
    *,
    root: Path,
    version: str,
    source_sha: str,
    harness_sha: str,
    bundles: dict[str, Path],
) -> dict[str, Any]:
    validate_release_identity(
        root=root, source_sha=source_sha, harness_sha=harness_sha
    )
    if set(bundles) != DEVICES:
        raise ValueError("release evidence does not cover the required devices")

    artifacts = artifact_identities(root, version)
    hf_wheel_sha256 = artifacts[f"rwkv7_hf-{version}-py3-none-any.whl"]["sha256"]
    kernel_wheel_sha256 = artifacts[f"rwkv7_kernels-{version}-py3-none-any.whl"][
        "sha256"
    ]
    devices = {
        device: read_device_report(
            device=device,
            bundle=bundle,
            source_sha=source_sha,
            harness_sha=harness_sha,
            hf_wheel_sha256=hf_wheel_sha256,
            kernel_wheel_sha256=kernel_wheel_sha256,
        )
        for device, bundle in sorted(bundles.items())
    }
    evidence = {}
    for device in DEVICE_ORDER:
        archive_name = device_evidence_archive_name(device, version)
        archive = root / archive_name
        if not archive.is_file() or archive.is_symlink():
            raise ValueError(f"missing or unsafe compact evidence archive: {device}")
        manifest_sha = sha256_file(bundles[device] / "MANIFEST.sha256")
        if devices[device]["compact_bundle_manifest_sha256"] != manifest_sha:
            raise ValueError(f"compact evidence manifest identity mismatch: {device}")
        evidence[device] = {
            "archive": archive_name,
            "sha256": sha256_file(archive),
            "size": archive.stat().st_size,
            "compact_bundle_manifest_sha256": manifest_sha,
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
        "version": version,
        "source_sha": source_sha,
        "fla_commit": FLA_COMMIT,
        "harness_sha": harness_sha,
        "artifacts": artifacts,
        "evidence": evidence,
        "validation": {"status": "passed", "devices": devices},
    }
    return provenance


def release_metadata_payloads(
    provenance: dict[str, Any],
) -> tuple[bytes, bytes]:
    provenance_payload = (
        json.dumps(provenance, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    sums = [
        f"{row['sha256']}  {name}"
        for name, row in provenance["artifacts"].items()
    ]
    for device in DEVICE_ORDER:
        row = provenance["evidence"][device]
        sums.append(f"{row['sha256']}  {row['archive']}")
    sums.append(
        f"{hashlib.sha256(provenance_payload).hexdigest()}  release-provenance.json"
    )
    return provenance_payload, ("\n".join(sums) + "\n").encode()


def build(args: argparse.Namespace) -> dict[str, Any]:
    root = args.directory.expanduser().resolve()
    harness_sha = str(args.harness_sha or "")
    validate_release_identity(
        root=root, source_sha=args.source_sha, harness_sha=harness_sha
    )
    bundles = parse_device_evidence(args.device_evidence)
    for device in DEVICE_ORDER:
        write_device_evidence_archive(
            bundle=bundles[device],
            archive=root / device_evidence_archive_name(device, args.version),
            device=device,
            version=args.version,
        )
    provenance = compose_provenance(
        root=root,
        version=args.version,
        source_sha=args.source_sha,
        harness_sha=harness_sha,
        bundles=bundles,
    )
    provenance_payload, sums_payload = release_metadata_payloads(provenance)
    write_atomic(root / "release-provenance.json", provenance_payload)
    write_atomic(root / "SHA256SUMS", sums_payload)
    return provenance


def verify_existing(args: argparse.Namespace) -> dict[str, Any]:
    """Rebuild release metadata from archived compact evidence without writing."""

    root = args.directory.expanduser().resolve()
    if args.device_evidence:
        raise ValueError("--verify-existing reads only release evidence archives")
    provenance_path = root / "release-provenance.json"
    sums_path = root / "SHA256SUMS"
    if not provenance_path.is_file() or provenance_path.is_symlink():
        raise ValueError("missing or unsafe release-provenance.json")
    if not sums_path.is_file() or sums_path.is_symlink():
        raise ValueError("missing or unsafe SHA256SUMS")
    declared_payload = provenance_path.read_bytes()
    declared = json.loads(declared_payload)
    if declared.get("schema") != PROVENANCE_SCHEMA:
        raise ValueError("unexpected release provenance schema")
    if declared.get("version") != args.version:
        raise ValueError("release provenance version mismatch")
    if declared.get("source_sha") != args.source_sha:
        raise ValueError("release provenance source SHA mismatch")
    harness_sha = str(declared.get("harness_sha", ""))
    if args.harness_sha is not None and args.harness_sha != harness_sha:
        raise ValueError("release provenance harness SHA mismatch")
    validate_release_identity(
        root=root, source_sha=args.source_sha, harness_sha=harness_sha
    )

    with tempfile.TemporaryDirectory(prefix="rwkv7-release-evidence-") as temp_name:
        temp = Path(temp_name)
        bundles = {
            device: extract_device_evidence_archive(
                archive=root / device_evidence_archive_name(device, args.version),
                output=temp,
                device=device,
                version=args.version,
            )
            for device in DEVICE_ORDER
        }
        rebuilt = compose_provenance(
            root=root,
            version=args.version,
            source_sha=args.source_sha,
            harness_sha=harness_sha,
            bundles=bundles,
        )
    rebuilt_payload, rebuilt_sums = release_metadata_payloads(rebuilt)
    if declared_payload != rebuilt_payload:
        raise ValueError(
            "release provenance differs from validated compact evidence rebuild"
        )
    if sums_path.read_bytes() != rebuilt_sums:
        raise ValueError("SHA256SUMS differs from validated provenance rebuild")
    return rebuilt


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    provenance = verify_existing(args) if args.verify_existing else build(args)
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
