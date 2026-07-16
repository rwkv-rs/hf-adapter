# coding=utf-8
"""Strict loader for optional offline Marlin schedule profiles.

Production defaults remain evidence-gated in :mod:`kernel_policy`.  This
module only consumes a profile when ``RWKV7_MARLIN_AUTOTUNE_PROFILE`` is set
explicitly.  Profiles are tied to the exact GPU, compute capability, PyTorch
and CUDA runtime that produced them; any mismatch fails closed to Marlin's
built-in automatic scheduler.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


PROFILE_ENV = "RWKV7_MARLIN_AUTOTUNE_PROFILE"
SCHEMA_VERSION = 1


def _runtime_identity(device, torch_module) -> dict[str, Any]:
    index = torch_module.device(device).index
    if index is None:
        index = torch_module.cuda.current_device()
    return {
        "device": str(torch_module.cuda.get_device_name(index)),
        "compute_capability": [
            int(value) for value in torch_module.cuda.get_device_capability(index)
        ],
        "torch_version": str(torch_module.__version__),
        "cuda_version": str(torch_module.version.cuda),
    }


def _normalize_schedule(raw) -> tuple[int, int, int, int, int]:
    values = tuple(int(value) for value in raw)
    if len(values) != 5:
        raise ValueError("autotune schedule must have five integers")
    tile_k, block_n, threads, sms, stages = values
    if tile_k not in (64, 128) or block_n not in (64, 128, 256):
        raise ValueError("autotune profile contains an unsupported Marlin tile")
    if threads not in (128, 256) or sms == 0 or sms < -1:
        raise ValueError("autotune profile contains an invalid thread/SM count")
    if stages not in (-1, 2, 4):
        raise ValueError("autotune profile contains an invalid stage count")
    return values


def schedules_for_linear(
    *,
    device,
    in_features: int,
    out_features: int,
    group_size: int,
    torch_module,
    profile_path: str | os.PathLike[str] | None = None,
) -> dict[int, tuple[int, int, int, int, int]]:
    """Return exact-row schedules from a matching explicit profile.

    Missing files, malformed profiles and identity mismatches deliberately
    return an empty mapping.  The caller then retains the built-in scheduler.
    """

    raw_path = profile_path if profile_path is not None else os.environ.get(PROFILE_ENV)
    if not raw_path:
        return {}
    try:
        payload = json.loads(Path(raw_path).expanduser().read_text(encoding="utf-8"))
        if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
            return {}
        identity = _runtime_identity(device, torch_module)
        if any(payload.get(key) != value for key, value in identity.items()):
            return {}
        selected: dict[int, tuple[int, int, int, int, int]] = {}
        for entry in payload.get("entries", ()):
            if (
                int(entry.get("k", -1)) != int(in_features)
                or int(entry.get("n", -1)) != int(out_features)
                or int(entry.get("group_size", -1)) != int(group_size)
            ):
                continue
            rows = int(entry["rows"])
            if rows <= 0:
                continue
            selected[rows] = _normalize_schedule(entry["schedule"])
        return selected
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return {}


__all__ = ["PROFILE_ENV", "SCHEMA_VERSION", "schedules_for_linear"]
