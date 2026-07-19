# coding=utf-8
"""Module-selection policies for native MM8/MM4 quantization."""
from __future__ import annotations
import os
import re

NATIVE_MM_POLICIES = ("memory", "speed", "balanced")

_FFN_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.ffn\.(?:key|value)$")


def normalize_native_mm_policy(policy: str | None) -> str:
    """Return a canonical native quantization module-selection policy."""

    value = (policy or "memory").strip().lower().replace("-", "_")
    aliases = {
        "default": "memory",
        "all": "memory",
        "size": "memory",
        "size_gated": "memory",
        "fast": "speed",
        "head": "speed",
        "head_only": "speed",
        "lm_head": "speed",
        "lm_head_only": "speed",
        "hybrid": "balanced",
        "speed_balanced": "balanced",
    }
    value = aliases.get(value, value)
    if value not in NATIVE_MM_POLICIES:
        allowed = ", ".join(NATIVE_MM_POLICIES)
        raise ValueError(f"unsupported native MM quantization policy {policy!r}; expected one of: {allowed}")
    return value


def should_quantize_linear(name: str, weight_numel: int, *, min_params: int, policy: str | None = "memory") -> bool:
    """Return whether a Linear module should be replaced by MM8/MM4.

    ``memory`` keeps the historical size-gated behavior. ``speed`` only swaps
    ``lm_head``. ``balanced`` additionally packs the first FFN layer pair by
    default, which is the measured old-card production compromise: lower
    weight memory while retaining the paired prefill/decode speed gate. The layer
    count can be overridden with ``RWKV7_NATIVE_MM_BALANCED_FFN_LAYERS``.
    """

    if int(weight_numel) < int(min_params):
        return False
    policy = normalize_native_mm_policy(policy)
    if policy == "memory":
        return True
    if name == "lm_head" or name.endswith(".lm_head"):
        return True
    if policy == "speed":
        return False
    try:
        layer_count = max(
            0, int(os.environ.get("RWKV7_NATIVE_MM_BALANCED_FFN_LAYERS", "1"))
        )
    except ValueError:
        layer_count = 1
    match = _FFN_LAYER_RE.search(name)
    return bool(match is not None and int(match.group(1)) < layer_count)
