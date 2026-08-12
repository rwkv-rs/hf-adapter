#!/usr/bin/env python3
"""Capture and optionally enforce the cross-card HF fast-path v1 runtime lock."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, distribution, version
from pathlib import Path
from typing import Any


PACKAGES = {
    "torch": "torch",
    "triton": "triton",
    "transformers": "transformers",
    "fla": "flash-linear-attention",
    "causal_conv1d": "causal-conv1d",
}


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def package_source(name: str) -> dict[str, Any] | None:
    try:
        direct_url = distribution(name).read_text("direct_url.json")
    except PackageNotFoundError:
        return None
    if not direct_url:
        return None
    value = json.loads(direct_url)
    vcs = value.get("vcs_info") or {}
    return {
        "url": value.get("url"),
        "vcs": vcs.get("vcs"),
        "commit_id": vcs.get("commit_id"),
        "requested_revision": vcs.get("requested_revision"),
    }


def pip_freeze() -> str:
    output = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        text=True,
        stderr=subprocess.STDOUT,
    )
    lines = sorted(line.strip() for line in output.splitlines() if line.strip())
    return "\n".join(lines) + "\n"


def git_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def capture(root: Path) -> tuple[dict[str, Any], str]:
    import torch

    freeze = pip_freeze()
    sources = {label: package_source(package) for label, package in PACKAGES.items()}
    runtime_lock = {
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "triton_version": package_version("triton"),
        "transformers_version": package_version("transformers"),
        "fla_version": package_version("flash-linear-attention"),
        "causal_conv1d_version": package_version("causal-conv1d"),
        "package_source_commits": {
            label: source.get("commit_id") if source else None for label, source in sources.items()
        },
        "pip_freeze_sha256": hashlib.sha256(freeze.encode("utf-8")).hexdigest(),
        "repository_commit": git_commit(root),
        "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        "docker_image_digest": os.environ.get("DOCKER_IMAGE_DIGEST"),
    }
    gpu = None
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "device_count_visible": torch.cuda.device_count(),
        }
    manifest = {
        "schema_version": 1,
        "protocol": "hf_fast_path_v1",
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "runtime_lock": runtime_lock,
        "package_sources": sources,
        "gpu": gpu,
    }
    return manifest, freeze


def compare_runtime_lock(actual: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    keys = sorted(set(actual) | set(expected))
    return [
        f"{key}: actual={actual.get(key)!r}, expected={expected.get(key)!r}"
        for key in keys
        if actual.get(key) != expected.get(key)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pip-freeze-output", type=Path, required=True)
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--write-lock", type=Path)
    parser.add_argument("--require-python", default="3.10")
    args = parser.parse_args()
    if args.lock and args.write_lock:
        parser.error("--lock and --write-lock are mutually exclusive")
    return args


def main() -> int:
    args = parse_args()
    manifest, freeze = capture(args.root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    args.pip_freeze_output.write_text(freeze, encoding="utf-8")

    errors: list[str] = []
    python_version = str(manifest["runtime_lock"]["python_version"])
    if not python_version.startswith(args.require_python + "."):
        errors.append(
            f"python_version: actual={python_version!r}, required major.minor={args.require_python!r}"
        )
    if args.lock:
        expected_document = json.loads(args.lock.read_text(encoding="utf-8"))
        expected = expected_document.get("runtime_lock", expected_document)
        errors.extend(compare_runtime_lock(manifest["runtime_lock"], expected))
    if args.write_lock:
        args.write_lock.parent.mkdir(parents=True, exist_ok=True)
        args.write_lock.write_text(
            json.dumps({"schema_version": 1, "runtime_lock": manifest["runtime_lock"]}, indent=2)
            + "\n",
            encoding="utf-8",
        )
    result = {
        "status": "pass" if not errors else "fail",
        "manifest": str(args.output),
        "lock": str(args.lock or args.write_lock or ""),
        "errors": errors,
    }
    print("HF_FAST_PATH_V1_ENVIRONMENT " + json.dumps(result), flush=True)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
