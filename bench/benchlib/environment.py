"""Machine-readable benchmark environment capture."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from typing import Any, Iterable


DEFAULT_PACKAGES = (
    "torch",
    "transformers",
    "safetensors",
    "accelerate",
    "triton",
    "flash-linear-attention",
)


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ("git", *args), stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def collect_environment(
    *,
    device: str = "cuda",
    packages: Iterable[str] = DEFAULT_PACKAGES,
) -> dict[str, Any]:
    """Collect reproducibility metadata without requiring Torch at import time."""

    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None

    gpu: dict[str, Any] | None = None
    if device.startswith("cuda"):
        try:
            import torch

            index = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(index)
            gpu = {
                "index": index,
                "name": properties.name,
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_memory_bytes": int(properties.total_memory),
                "torch_cuda": torch.version.cuda,
            }
        except (ImportError, RuntimeError):
            gpu = None

    return {
        "schema_version": 1,
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
        "packages": versions,
        "device": device,
        "gpu": gpu,
    }
