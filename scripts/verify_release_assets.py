#!/usr/bin/env python3
"""Verify that a GitHub release contains the exact three-device validated artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


FLA_COMMIT = "80e494f6c588e091fc8316b612870df29375c5b8"
DEVICES = {"rtx-4080", "tesla-v100", "rtx-4090"}


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


def expected_artifacts(version: str) -> tuple[str, ...]:
    return (
        f"rwkv7_hf-{version}-py3-none-any.whl",
        f"rwkv7_hf-{version}.tar.gz",
        f"rwkv7_kernels-{version}-py3-none-any.whl",
        f"rwkv7_kernels-{version}.tar.gz",
    )


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


def verify(args: argparse.Namespace) -> dict[str, Any]:
    root = args.directory.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"release directory does not exist: {root}")
    names = expected_artifacts(args.version)
    sums = read_sums(root / "SHA256SUMS")
    artifacts = {}
    for name in names:
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing or unsafe release artifact: {name}")
        digest = sha256_file(path)
        if sums.get(name) != digest:
            raise ValueError(f"SHA256SUMS mismatch: {name}")
        artifacts[name] = {"sha256": digest, "size": path.stat().st_size}

    provenance_path = root / "release-provenance.json"
    if sums.get(provenance_path.name) != sha256_file(provenance_path):
        raise ValueError("release provenance is not covered by SHA256SUMS")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("schema") != "rwkv7-release-provenance-v1":
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
        raise ValueError("three-device release validation has not passed")
    devices = validation.get("devices") or {}
    if set(devices) != DEVICES:
        raise ValueError("release provenance does not cover the three required devices")
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
        bundle_sha = str(row.get("compact_bundle_manifest_sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", bundle_sha):
            raise ValueError(f"compact evidence identity is missing: {device}")
    return {
        "status": "passed",
        "version": args.version,
        "source_sha": args.source_sha,
        "harness_sha": harness_sha,
        "artifacts": artifacts,
        "devices": sorted(devices),
    }


def main(argv: list[str] | None = None) -> int:
    report = verify(arguments(argv))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
