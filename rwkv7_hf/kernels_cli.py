# coding=utf-8
"""Inspect, recommend, or install an exact-runtime RWKV-7 kernel wheel."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any, Sequence

from .kernel_package import (
    _runtime_environment,
    _validate_manifest,
    inspect_kernel_package,
)


DEFAULT_INDEX_URL = (
    "https://github.com/rwkv-rs/hf-adapter/releases/latest/download/"
    "rwkv7-kernel-index-v1.json"
)
INDEX_URL_ENV = "RWKV7_KERNEL_INDEX_URL"


def _read_index(location: str) -> dict[str, Any]:
    if location.startswith(("https://", "http://")):
        request = urllib.request.Request(
            location,
            headers={"User-Agent": "rwkv7-hf-kernels/1"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read()
    else:
        payload = Path(location).expanduser().read_bytes()
    data = json.loads(payload)
    if data.get("protocol") != "rwkv7-kernel-index-v1":
        raise RuntimeError(
            f"unsupported kernel index protocol: {data.get('protocol')!r}"
        )
    return data


def _matching_wheels(
    index: dict[str, Any], torch_module: Any, device: Any
) -> list[dict[str, Any]]:
    environment = _runtime_environment(torch_module, device)
    matches = []
    for wheel in index.get("wheels", []):
        manifest = wheel.get("manifest")
        if not isinstance(manifest, dict):
            continue
        if not _validate_manifest(manifest, environment):
            matches.append(wheel)
    return matches


def _render_status(report: dict[str, Any]) -> str:
    package = report.get("manifest") or {}
    lines = [
        "RWKV-7 prebuilt kernel package",
        f"Status: {report['status'].upper()}",
        f"Mode: {report['mode']}",
        f"Installed: {package.get('distribution', 'none')}",
        f"Recommended: {report.get('recommended_distribution') or 'none'}",
        f"Build lane: {report.get('recommended_build') or 'none'}",
    ]
    lines.extend(f"Reason: {reason}" for reason in report.get("reasons", []))
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        default="status",
        choices=("status", "recommend", "list", "install"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--index-url",
        default=os.environ.get(INDEX_URL_ENV, DEFAULT_INDEX_URL),
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    import torch

    report = inspect_kernel_package(torch_module=torch, device=args.device)
    if args.command in {"status", "recommend"}:
        print(json.dumps(report, indent=2) if args.json else _render_status(report))
        return 0 if report["status"] in {"ready", "missing"} else 1

    index = _read_index(args.index_url)
    matches = _matching_wheels(index, torch, args.device)
    if args.command == "list":
        payload = {
            "index": args.index_url,
            "compatible_wheels": matches,
        }
        print(
            json.dumps(payload, indent=2)
            if args.json
            else "\n".join(item["filename"] for item in matches)
        )
        return 0 if matches else 1

    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one compatible kernel wheel, found {len(matches)}; "
            f"recommended distribution is {report.get('recommended_distribution')!r}"
        )
    wheel = matches[0]
    requirement = f"{wheel['url']}#sha256={wheel['sha256']}"
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--no-deps",
        requirement,
    ]
    if args.dry_run:
        print(json.dumps({"command": command, "wheel": wheel}, indent=2))
        return 0
    subprocess.check_call(command)
    print(
        "Kernel wheel installed. Run `rwkv7-hf doctor` in a new process to verify it."
    )
    return 0


def cli() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli()
