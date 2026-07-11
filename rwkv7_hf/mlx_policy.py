# coding=utf-8
"""Environment parsing and backend policy helpers for the optional MLX runtime.

This module deliberately contains no model/runtime imports so policy can be
shared by sessions, benchmarks, and the model without creating import cycles.
"""
from __future__ import annotations

import os


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def env_int(name: str, default: int, *, lower: int = 1, upper: int = 4096) -> int:
    raw = os.environ.get(name)
    try:
        value = int(raw) if raw not in {None, ""} else int(default)
    except ValueError:
        value = int(default)
    return max(int(lower), min(int(upper), value))


def env_scan_prefill_mode(name: str, default: str = "off") -> str:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        raw = default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on", "force", "forced"}:
        return "on"
    if value in {"0", "false", "no", "off", "disable", "disabled"}:
        return "off"
    if value == "auto":
        return "auto"
    return default


def env_choice(name: str, default: str, choices: set[str]) -> str:
    raw = os.environ.get(name)
    value = (raw if raw is not None and raw != "" else default).strip().lower()
    return value if value in choices else default


__all__ = [
    "env_choice",
    "env_flag",
    "env_float",
    "env_int",
    "env_scan_prefill_mode",
]
