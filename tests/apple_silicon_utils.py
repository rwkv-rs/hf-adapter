# coding=utf-8
"""Shared helpers for Apple Silicon hardware smoke scripts.

The Apple smoke tests are also run directly as scripts from ``tests/`` and on
non-Apple CI hosts, so keep this module dependency-light and safe to import
without torch, MLX, or macOS-only libraries.
"""
from __future__ import annotations

import json
import platform
import re
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def append_result(path: str, row: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def emit(path: str, row: dict[str, Any]) -> None:
    print(json.dumps(row, ensure_ascii=False))
    append_result(path, row)


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "missing"


def darwin_sysctl(name: str) -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", name], text=True).strip()
    except Exception:
        return "unknown"


def apple_memory_gb() -> int | str:
    raw = darwin_sysctl("hw.memsize")
    try:
        return round(int(raw) / 1024 / 1024 / 1024)
    except Exception:
        return "unknown"


def infer_model_size_label(model_path: str, explicit: str = "") -> str:
    if explicit:
        return explicit.lower()
    match = re.search(r"(\d+(?:\.\d+)?b)", Path(model_path).name.lower())
    return match.group(1) if match else "unknown"


def mps_backend(torch: Any) -> Any | None:
    return getattr(getattr(torch, "backends", None), "mps", None)


def mps_is_available(torch: Any) -> bool:
    mps = mps_backend(torch)
    if mps is None or not hasattr(mps, "is_available"):
        return False
    try:
        return bool(mps.is_available())
    except Exception:
        return False


def mps_is_built(torch: Any) -> bool:
    mps = mps_backend(torch)
    if mps is None or not hasattr(mps, "is_built"):
        return False
    try:
        return bool(mps.is_built())
    except Exception:
        return False


def mps_memory_stats(torch: Any) -> dict[str, int]:
    if not hasattr(torch, "mps"):
        return {}
    stats: dict[str, int] = {}
    for key, func_name in (
        ("mps_current_allocated_memory_bytes", "current_allocated_memory"),
        ("mps_driver_allocated_memory_bytes", "driver_allocated_memory"),
        ("mps_recommended_max_memory_bytes", "recommended_max_memory"),
    ):
        try:
            stats[key] = int(getattr(torch.mps, func_name)())
        except Exception:
            pass
    return stats


# Backward-compatible name used by the real-model training smoke.
mps_memory = mps_memory_stats


def choose_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        if requested == "mps" and not mps_is_available(torch):
            raise RuntimeError("requested --device mps but MPS is unavailable")
        return requested
    return "mps" if mps_is_available(torch) else "cpu"


def dtype_for(torch: Any, name: str) -> Any:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def tensor_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}


def parse_ints(raw: str) -> list[int]:
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    if any(v <= 0 for v in values):
        raise ValueError(f"all values must be positive: {values}")
    return values


def sync(torch: Any, device: str) -> None:
    if device == "mps" and getattr(torch, "mps", None) is not None:
        try:
            torch.mps.synchronize()
        except Exception:
            pass
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
