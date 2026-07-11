#!/usr/bin/env python3
"""Audit exact PyPI versions and immutable release artifact hashes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any
import urllib.error
import urllib.request


PROJECT = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--distribution",
        action="append",
        required=True,
        help="project=version; repeat for every distribution",
    )
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="project=/path/to/exact/wheel-or-sdist",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--index-url", default="https://pypi.org/pypi")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--harness-sha")
    return parser.parse_args(argv)


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_pairs(values: list[str], label: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} must use project=value: {value}")
        project, selected = value.split("=", 1)
        project = canonical_name(project.strip())
        selected = selected.strip()
        if not PROJECT.fullmatch(project) or not selected:
            raise ValueError(f"invalid {label}: {value}")
        if project in result:
            raise ValueError(f"duplicate {label} project: {project}")
        result[project] = selected
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_json(url: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "rwkv7-release-audit/1"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.load(response)


def audit_distribution(
    project: str,
    version: str,
    artifact: Path | None,
    *,
    index_url: str,
    timeout: float,
) -> dict[str, Any]:
    url = f"{index_url.rstrip('/')}/{project}/{version}/json"
    try:
        payload = fetch_json(url, timeout)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        return {
            "project": project,
            "version": version,
            "url": url,
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }
    info = payload.get("info") or {}
    files = []
    for row in payload.get("urls") or []:
        digests = row.get("digests") or {}
        files.append(
            {
                "filename": row.get("filename"),
                "packagetype": row.get("packagetype"),
                "python_version": row.get("python_version"),
                "size": row.get("size"),
                "sha256": digests.get("sha256"),
                "upload_time_iso_8601": row.get("upload_time_iso_8601"),
                "yanked": bool(row.get("yanked")),
            }
        )
    reasons = []
    if canonical_name(str(info.get("name", ""))) != project:
        reasons.append("project name mismatch")
    if str(info.get("version", "")) != version:
        reasons.append("version mismatch")
    if not files:
        reasons.append("release has no files")
    if not any(row["packagetype"] == "bdist_wheel" for row in files):
        reasons.append("release has no wheel")
    if any(row["yanked"] for row in files):
        reasons.append("release contains a yanked file")
    if any(not re.fullmatch(r"[0-9a-f]{64}", str(row["sha256"])) for row in files):
        reasons.append("release file has no valid SHA256")

    expected_artifact = None
    if artifact is not None:
        artifact = artifact.expanduser().resolve()
        if not artifact.is_file():
            reasons.append(f"expected artifact is missing: {artifact}")
        else:
            expected_artifact = {
                "path": str(artifact),
                "filename": artifact.name,
                "size": artifact.stat().st_size,
                "sha256": sha256_file(artifact),
            }
            matches = [row for row in files if row["filename"] == artifact.name]
            if len(matches) != 1:
                reasons.append("expected artifact filename is not unique on PyPI")
            elif (
                matches[0]["sha256"] != expected_artifact["sha256"]
                or matches[0]["size"] != expected_artifact["size"]
            ):
                reasons.append(
                    "published artifact bytes differ from the local release file"
                )
    return {
        "project": project,
        "version": version,
        "url": url,
        "status": "passed" if not reasons else "failed",
        "reasons": reasons,
        "requires_python": info.get("requires_python"),
        "requires_dist": info.get("requires_dist"),
        "files": sorted(files, key=lambda row: str(row["filename"])),
        "expected_artifact": expected_artifact,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    distributions = parse_pairs(args.distribution, "distribution")
    artifacts = {
        project: Path(path)
        for project, path in parse_pairs(args.artifact, "artifact").items()
    }
    unknown = set(artifacts) - set(distributions)
    if unknown:
        raise ValueError(f"artifact without a distribution: {sorted(unknown)}")
    rows = [
        audit_distribution(
            project,
            version,
            artifacts.get(project),
            index_url=args.index_url,
            timeout=args.timeout,
        )
        for project, version in distributions.items()
    ]
    return {
        "schema": "rwkv7-pypi-release-audit-v1",
        "status": "passed"
        if all(row["status"] == "passed" for row in rows)
        else "failed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "harness_sha": args.harness_sha,
        "index_url": args.index_url,
        "command": sys.argv,
        "python": sys.version,
        "distributions": rows,
    }


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "status": report["status"]}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
