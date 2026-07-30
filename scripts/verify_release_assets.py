#!/usr/bin/env python3
"""Verify release artifacts against the required RTX acceptance devices."""

from __future__ import annotations

import argparse
from datetime import datetime
from email.parser import BytesParser
from email.policy import default as email_policy
import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath
import re
import sys
import tarfile
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_release_wheels import (  # noqa: E402
    audit_hf_wheel,
    audit_kernel_wheel,
    open_wheel,
)
from scripts.release_route_contract import (  # noqa: E402
    FORMAL_REFERENCE_BACKEND_ENVIRONMENT,
    validate_actual_routes,
)


FLA_COMMIT = "80e494f6c588e091fc8316b612870df29375c5b8"
DEVICE_ORDER = ("rtx-4080", "rtx-4090")
DEVICES = frozenset(DEVICE_ORDER)


def device_evidence_archive_name(device: str, version: str) -> str:
    if device not in DEVICES:
        raise ValueError(f"unexpected release device: {device}")
    return f"rwkv7-evidence-{device}-{version}.tar.gz"


def expected_device_evidence_archives(version: str) -> tuple[str, ...]:
    return tuple(
        device_evidence_archive_name(device, version) for device in DEVICE_ORDER
    )


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--require-validation-passed", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aware_datetime(value: Any, *, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"invalid {label} timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} timestamp is not timezone-aware")
    return parsed


def expected_artifacts(version: str) -> tuple[str, ...]:
    return (
        f"rwkv7_hf-{version}-py3-none-any.whl",
        f"rwkv7_hf-{version}.tar.gz",
        f"rwkv7_kernels-{version}-py3-none-any.whl",
        f"rwkv7_kernels-{version}.tar.gz",
    )


def audit_wheel_against_checkout(
    wheel: Path,
    *,
    mappings: tuple[tuple[str, Path], ...],
) -> dict[str, Any]:
    """Require the complete wheel payload to be owned by the tagged checkout.

    Package files must be a byte-for-byte, complete copy of the supplied source
    roots.  The only other accepted members are files in the wheel's single
    ``.dist-info`` directory.  In particular, top-level modules and ``.data``
    payloads are not silently ignored: those locations can install executable
    Python outside the audited package roots.
    """

    archive, members = open_wheel(wheel)
    matched: list[str] = []
    try:
        expected: dict[str, Path] = {}
        for prefix, source_root in mappings:
            resolved_root = source_root.resolve()
            if not resolved_root.is_dir() or resolved_root.is_symlink():
                raise ValueError(f"wheel checkout root is missing or unsafe: {source_root}")
            for source in sorted(resolved_root.rglob("*")):
                if not source.is_file() or source.is_symlink():
                    continue
                relative = source.relative_to(resolved_root)
                if "__pycache__" in relative.parts or source.suffix in {
                    ".pyc",
                    ".pyo",
                }:
                    continue
                member = f"{prefix}{relative.as_posix()}"
                if member in expected:
                    raise ValueError(f"duplicate wheel checkout owner: {member}")
                expected[member] = source

        for member in sorted(members):
            mapping = next((row for row in mappings if member.startswith(row[0])), None)
            if mapping is None:
                continue
            prefix, source_root = mapping
            relative = PurePosixPath(member).relative_to(PurePosixPath(prefix))
            source = source_root.joinpath(*relative.parts).resolve()
            resolved_root = source_root.resolve()
            if resolved_root != source and resolved_root not in source.parents:
                raise ValueError(f"wheel payload escaped checkout root: {member}")
            if not source.is_file() or source.is_symlink():
                raise ValueError(f"wheel payload is absent from checkout: {member}")
            if archive.read(member) != source.read_bytes():
                raise ValueError(f"wheel payload differs from checkout: {member}")
            matched.append(member)

        missing = sorted(set(expected) - set(matched))
        if missing:
            raise ValueError(f"wheel omitted checkout-owned payload: {missing}")

        dist_info_roots = {
            PurePosixPath(member).parts[0]
            for member in members
            if PurePosixPath(member).parts
            and PurePosixPath(member).parts[0].endswith(".dist-info")
        }
        if len(dist_info_roots) != 1:
            raise ValueError("wheel must contain exactly one .dist-info payload root")
        dist_info = next(iter(dist_info_roots))
        unowned = sorted(
            member
            for member in members
            if member not in matched and not member.startswith(f"{dist_info}/")
        )
        if unowned:
            raise ValueError(f"wheel contains unowned payload: {unowned}")
    finally:
        archive.close()
    if not matched:
        raise ValueError("wheel has no checkout-owned package payload")
    return {"status": "passed", "matched_files": len(matched)}


def read_sums(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or Path(name).name != name:
            raise ValueError(f"unsafe SHA256SUMS row: {line}")
        if name in rows:
            raise ValueError(f"duplicate SHA256SUMS row: {name}")
        rows[name] = digest
    return rows


def read_sdist(path: Path, expected_root: str) -> dict[str, bytes]:
    """Read a source distribution without trusting or extracting tar paths."""

    path = path.expanduser().resolve()
    if not path.is_file() or path.is_symlink() or not path.name.endswith(".tar.gz"):
        raise ValueError(f"missing or unsafe source distribution: {path}")
    files: dict[str, bytes] = {}
    with tarfile.open(path, mode="r:gz") as archive:
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            if (
                name.is_absolute()
                or ".." in name.parts
                or not name.parts
                or name.parts[0] != expected_root
            ):
                raise ValueError(f"unsafe source-distribution member: {member.name}")
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(
                    f"non-regular source-distribution member: {member.name}"
                )
            normalized = str(name)
            if normalized in files:
                raise ValueError(f"duplicate source-distribution member: {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(
                    f"unreadable source-distribution member: {member.name}"
                )
            files[normalized] = stream.read()
    return files


def audit_sdist(
    path: Path,
    *,
    distribution: str,
    version: str,
    wheel: Path,
    package_prefixes: tuple[str, ...],
    forbidden_prefix: str,
) -> dict[str, Any]:
    """Bind every install-relevant sdist file to the tagged checkout or wheel."""

    root = f"{distribution.replace('-', '_')}-{version}"
    files = read_sdist(path, root)
    pkg_info_name = f"{root}/PKG-INFO"
    pyproject_name = f"{root}/pyproject.toml"
    if pkg_info_name not in files or pyproject_name not in files:
        raise ValueError(f"source distribution is missing metadata: {distribution}")
    metadata = BytesParser(policy=email_policy).parsebytes(files[pkg_info_name])
    if metadata.get("Name") != distribution or metadata.get("Version") != version:
        raise ValueError(f"source-distribution metadata differs: {distribution}")
    project = tomllib.loads(files[pyproject_name].decode())["project"]
    if project.get("name") != distribution or project.get("version") != version:
        raise ValueError(f"source-distribution pyproject differs: {distribution}")

    relative_files = {
        str(PurePosixPath(*PurePosixPath(name).parts[1:])): payload
        for name, payload in files.items()
    }
    if any(name.startswith(forbidden_prefix) for name in relative_files):
        raise ValueError(
            f"source distribution crosses package ownership: {distribution}"
        )
    wheel_archive, wheel_members = open_wheel(wheel)
    try:
        metadata_members = sorted(
            name for name in wheel_members if name.endswith(".dist-info/METADATA")
        )
        if len(metadata_members) != 1:
            raise ValueError(f"wheel metadata is ambiguous: {distribution}")
        if relative_files[pkg_info_name.removeprefix(f"{root}/")] != wheel_archive.read(
            metadata_members[0]
        ):
            raise ValueError(
                f"source-distribution PKG-INFO differs from wheel: {distribution}"
            )
        package_members = sorted(
            name
            for name in wheel_members
            if any(name.startswith(prefix) for prefix in package_prefixes)
        )
        if not package_members:
            raise ValueError(f"wheel contains no package payload: {distribution}")
        for member in package_members:
            if member not in relative_files:
                raise ValueError(f"source distribution omitted wheel payload: {member}")
            if relative_files[member] != wheel_archive.read(member):
                raise ValueError(
                    f"source distribution payload differs from wheel: {member}"
                )
    finally:
        wheel_archive.close()

    checkout_root = ROOT if distribution == "rwkv7-hf" else ROOT / "kernels"
    exact_files = {
        "pyproject.toml": checkout_root / "pyproject.toml",
        "README.md": checkout_root / "README.md",
        "LICENSE": checkout_root / "LICENSE",
    }
    if distribution == "rwkv7-hf":
        checkout_mappings = (
            ("rwkv7_hf/", ROOT / "rwkv7_hf"),
            ("rwkv7_hf_tools/", ROOT / "rwkv7_hf_tools"),
            ("tests/", ROOT / "tests"),
        )
        egg_info = "rwkv7_hf.egg-info/"
    else:
        checkout_mappings = (
            ("rwkv7_kernels/", ROOT / "kernels" / "rwkv7_kernels"),
        )
        egg_info = "rwkv7_kernels.egg-info/"

    missing_checkout_files = sorted(set(exact_files) - set(relative_files))
    if missing_checkout_files:
        raise ValueError(
            "source distribution omitted checkout-owned payload: "
            f"{distribution}: {missing_checkout_files}"
        )

    generated_egg_info = {
        f"{egg_info}{name}"
        for name in (
            "PKG-INFO",
            "SOURCES.txt",
            "dependency_links.txt",
            "entry_points.txt",
            "requires.txt",
            "top_level.txt",
        )
    }
    generated = {"PKG-INFO", "setup.cfg", *generated_egg_info}
    expected_setup_cfg = b"[egg_info]\ntag_build = \ntag_date = 0\n\n"
    unowned: list[str] = []
    for name, payload in sorted(relative_files.items()):
        if name in exact_files:
            source = exact_files[name]
            if not source.is_file() or source.is_symlink() or payload != source.read_bytes():
                raise ValueError(
                    f"source distribution differs from checkout: {distribution}/{name}"
                )
            continue
        mapping = next(
            (row for row in checkout_mappings if name.startswith(row[0])), None
        )
        if mapping is not None:
            prefix, source_root = mapping
            relative = PurePosixPath(name).relative_to(PurePosixPath(prefix))
            source = source_root.joinpath(*relative.parts).resolve()
            resolved_root = source_root.resolve()
            if resolved_root != source and resolved_root not in source.parents:
                raise ValueError(f"source-distribution payload escaped checkout: {name}")
            if not source.is_file() or source.is_symlink() or payload != source.read_bytes():
                raise ValueError(
                    f"source distribution differs from checkout: {distribution}/{name}"
                )
            continue
        if name in generated:
            if name == "setup.cfg" and payload != expected_setup_cfg:
                raise ValueError(
                    f"source-distribution generated setup.cfg differs: {distribution}"
                )
            if name == f"{egg_info}PKG-INFO" and payload != relative_files["PKG-INFO"]:
                raise ValueError(
                    f"source-distribution egg-info metadata differs: {distribution}"
                )
            continue
        unowned.append(name)
    if unowned:
        raise ValueError(f"source distribution contains unowned payload: {unowned}")
    return {
        "status": "passed",
        "distribution": distribution,
        "version": version,
        "matched_package_files": len(package_members),
        "members": len(files),
    }


def verify(args: argparse.Namespace) -> dict[str, Any]:
    root = args.directory.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"release directory does not exist: {root}")
    names = expected_artifacts(args.version)
    summed_names = {
        *names,
        *expected_device_evidence_archives(args.version),
        "release-provenance.json",
    }
    expected_directory_names = {*summed_names, "SHA256SUMS"}
    actual_directory_names = {path.name for path in root.iterdir()}
    if actual_directory_names != expected_directory_names:
        missing = sorted(expected_directory_names - actual_directory_names)
        extra = sorted(actual_directory_names - expected_directory_names)
        raise ValueError(
            f"release asset set differs: missing={missing}, extra={extra}"
        )
    sums = read_sums(root / "SHA256SUMS")
    if set(sums) != summed_names:
        missing = sorted(summed_names - set(sums))
        extra = sorted(set(sums) - summed_names)
        raise ValueError(
            f"SHA256SUMS entry set differs: missing={missing}, extra={extra}"
        )
    artifacts = {}
    for name in names:
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing or unsafe release artifact: {name}")
        digest = sha256_file(path)
        if sums.get(name) != digest:
            raise ValueError(f"SHA256SUMS mismatch: {name}")
        artifacts[name] = {"sha256": digest, "size": path.stat().st_size}
    hf_wheel_path = root / f"rwkv7_hf-{args.version}-py3-none-any.whl"
    kernel_wheel_path = root / f"rwkv7_kernels-{args.version}-py3-none-any.whl"
    audit_hf_wheel(hf_wheel_path)
    audit_kernel_wheel(kernel_wheel_path)
    checkout_payloads = {
        "rwkv7-hf": audit_wheel_against_checkout(
            hf_wheel_path,
            mappings=(
                ("rwkv7_hf/", ROOT / "rwkv7_hf"),
                ("rwkv7_hf_tools/", ROOT / "rwkv7_hf_tools"),
            ),
        ),
        "rwkv7-kernels": audit_wheel_against_checkout(
            kernel_wheel_path,
            mappings=(("rwkv7_kernels/", ROOT / "kernels" / "rwkv7_kernels"),),
        ),
    }
    sdists = {
        "rwkv7-hf": audit_sdist(
            root / f"rwkv7_hf-{args.version}.tar.gz",
            distribution="rwkv7-hf",
            version=args.version,
            wheel=hf_wheel_path,
            package_prefixes=("rwkv7_hf/", "rwkv7_hf_tools/"),
            forbidden_prefix="rwkv7_kernels/",
        ),
        "rwkv7-kernels": audit_sdist(
            root / f"rwkv7_kernels-{args.version}.tar.gz",
            distribution="rwkv7-kernels",
            version=args.version,
            wheel=kernel_wheel_path,
            package_prefixes=("rwkv7_kernels/",),
            forbidden_prefix="rwkv7_hf",
        ),
    }

    evidence_archives = {}
    for device in DEVICE_ORDER:
        name = device_evidence_archive_name(device, args.version)
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing or unsafe compact evidence archive: {device}")
        digest = sha256_file(path)
        if sums.get(name) != digest:
            raise ValueError(f"SHA256SUMS mismatch: {name}")
        evidence_archives[device] = {
            "archive": name,
            "sha256": digest,
            "size": path.stat().st_size,
        }

    provenance_path = root / "release-provenance.json"
    if sums.get(provenance_path.name) != sha256_file(provenance_path):
        raise ValueError("release provenance is not covered by SHA256SUMS")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("schema") != "rwkv7-release-provenance-v2":
        raise ValueError("unexpected release provenance schema")
    if provenance.get("version") != args.version:
        raise ValueError("release provenance version mismatch")
    if provenance.get("source_sha") != args.source_sha:
        raise ValueError("release provenance source SHA mismatch")
    if provenance.get("fla_commit") != FLA_COMMIT:
        raise ValueError("release provenance FLA commit mismatch")
    if provenance.get("artifacts") != artifacts:
        raise ValueError("release provenance artifact identities do not match")
    harness_sha = str(provenance.get("harness_sha", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", harness_sha):
        raise ValueError("release provenance harness SHA is missing")

    validation = provenance.get("validation") or {}
    if args.require_validation_passed and validation.get("status") != "passed":
        raise ValueError("required-device release validation has not passed")
    devices = validation.get("devices") or {}
    if set(devices) != DEVICES:
        raise ValueError("release provenance does not cover the required devices")
    declared_evidence = provenance.get("evidence") or {}
    if set(declared_evidence) != DEVICES:
        raise ValueError("release provenance does not cover device evidence archives")
    hf_wheel = artifacts[f"rwkv7_hf-{args.version}-py3-none-any.whl"]["sha256"]
    kernel_wheel = artifacts[f"rwkv7_kernels-{args.version}-py3-none-any.whl"]["sha256"]
    for device, row in devices.items():
        if row.get("status") != "passed":
            raise ValueError(f"device validation did not pass: {device}")
        if row.get("hf_wheel_sha256") != hf_wheel:
            raise ValueError(f"HF wheel mismatch in device evidence: {device}")
        if row.get("kernel_wheel_sha256") != kernel_wheel:
            raise ValueError(f"kernel wheel mismatch in device evidence: {device}")
        if row.get("harness_sha") != harness_sha:
            raise ValueError(f"harness SHA mismatch in device evidence: {device}")
        if row.get("lm_eval_units") != 144 or row.get("lm_eval_status") != "passed":
            raise ValueError(f"formal lm_eval gate did not pass: {device}")
        for gate in (
            "correctness",
            "hf_ecosystem",
            "training",
            "quantization",
            "fla",
            "speed",
            "sft",
            "dpo",
            "grpo",
        ):
            if row.get(f"{gate}_status") != "passed":
                raise ValueError(f"{gate} gate did not pass: {device}")
        if row.get("training_policy") != "reference":
            raise ValueError(f"formal reference training policy is missing: {device}")
        if (
            row.get("training_backend_environment")
            != FORMAL_REFERENCE_BACKEND_ENVIRONMENT
        ):
            raise ValueError(f"formal reference training environment differs: {device}")
        bundle_sha = str(row.get("compact_bundle_manifest_sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", bundle_sha):
            raise ValueError(f"compact evidence identity is missing: {device}")
        expected_evidence = {
            **evidence_archives[device],
            "compact_bundle_manifest_sha256": bundle_sha,
        }
        if declared_evidence.get(device) != expected_evidence:
            raise ValueError(f"compact evidence archive identity mismatch: {device}")
        started_at = aware_datetime(
            row.get("acceptance_started_at"), label=f"{device} start"
        )
        completed_at = aware_datetime(
            row.get("acceptance_completed_at"), label=f"{device} completion"
        )
        if completed_at <= started_at:
            raise ValueError(f"device acceptance completion precedes start: {device}")
        try:
            validate_actual_routes(row.get("actual_routes"))
        except ValueError as exc:
            raise ValueError(
                f"actual route evidence is invalid: {device}: {exc}"
            ) from exc
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
    return {
        "status": "passed",
        "version": args.version,
        "source_sha": args.source_sha,
        "harness_sha": harness_sha,
        "artifacts": artifacts,
        "sdists": sdists,
        "checkout_payloads": checkout_payloads,
        "devices": sorted(devices),
        "evidence": declared_evidence,
    }


def main(argv: list[str] | None = None) -> int:
    report = verify(arguments(argv))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
