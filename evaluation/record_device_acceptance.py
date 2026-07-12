#!/usr/bin/env python3
"""Record the ordered lifetime of one immutable-wheel device acceptance run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import socket
import sys
import tempfile
from typing import Any


DEVICES = {"rtx-4080", "tesla-v100", "rtx-4090"}
SCHEMA = "rwkv7-device-acceptance-run-v1"
DEVICE_REPORT_SCHEMA = "rwkv7-device-release-validation-v1"


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    start = subparsers.add_parser("start")
    start.add_argument("--device", choices=sorted(DEVICES), required=True)
    start.add_argument("--source-sha", required=True)
    start.add_argument("--harness-sha", required=True)
    start.add_argument("--hf-wheel", type=Path, required=True)
    start.add_argument("--kernel-wheel", type=Path, required=True)
    start.add_argument("--output", type=Path, required=True)

    finish = subparsers.add_parser("finish")
    finish.add_argument("--run-report", type=Path, required=True)
    finish.add_argument("--release-validation", type=Path, required=True)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_file(path: Path, *, suffix: str | None = None) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"missing or unsafe file: {resolved}")
    if suffix is not None and not resolved.name.endswith(suffix):
        raise ValueError(f"unexpected file type: {resolved}")
    return resolved


def require_sha(label: str, value: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError(f"{label} SHA must be a lowercase 40-character Git SHA")
    return value


def now_iso(now: datetime | None = None) -> str:
    value = datetime.now(timezone.utc) if now is None else now
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("acceptance timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat()


def write_atomic(path: Path, report: dict[str, Any]) -> None:
    payload = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
    temporary.chmod(0o644)
    temporary.replace(path)


def start_report(args: argparse.Namespace, *, now: datetime | None = None) -> dict[str, Any]:
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ValueError(f"acceptance run report already exists: {output}")
    hf_wheel = safe_file(args.hf_wheel, suffix=".whl")
    kernel_wheel = safe_file(args.kernel_wheel, suffix=".whl")
    report = {
        "schema": SCHEMA,
        "device": args.device,
        "status": "running",
        "source_sha": require_sha("source", args.source_sha),
        "harness_sha": require_sha("harness", args.harness_sha),
        "hf_wheel_sha256": sha256_file(hf_wheel),
        "kernel_wheel_sha256": sha256_file(kernel_wheel),
        "started_at": now_iso(now),
        "host": socket.gethostname(),
        "command": sys.argv,
    }
    write_atomic(output, report)
    return report


def finish_report(
    args: argparse.Namespace, *, now: datetime | None = None
) -> dict[str, Any]:
    run_path = safe_file(args.run_report, suffix=".json")
    validation_path = safe_file(args.release_validation, suffix=".json")
    report = json.loads(run_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if report.get("schema") != SCHEMA or report.get("status") != "running":
        raise ValueError("device acceptance run is not an active v1 report")
    if (
        validation.get("schema") != DEVICE_REPORT_SCHEMA
        or validation.get("status") != "passed"
    ):
        raise ValueError("device release validation has not passed")
    for field in (
        "device",
        "source_sha",
        "harness_sha",
        "hf_wheel_sha256",
        "kernel_wheel_sha256",
    ):
        if report.get(field) != validation.get(field):
            raise ValueError(f"device acceptance identity mismatch: {field}")
    started = datetime.fromisoformat(str(report["started_at"]))
    completed = datetime.fromisoformat(now_iso(now))
    if completed <= started:
        raise ValueError("device acceptance completion must follow its start")
    report.update(
        {
            "status": "passed",
            "completed_at": completed.isoformat(),
            "release_validation_sha256": sha256_file(validation_path),
            "finish_command": sys.argv,
        }
    )
    write_atomic(run_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    report = start_report(args) if args.action == "start" else finish_report(args)
    print(
        json.dumps(
            {
                "device": report["device"],
                "status": report["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
