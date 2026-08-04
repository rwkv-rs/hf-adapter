# coding=utf-8
"""Module-selection policies for native MM8/MM4 quantization."""
from __future__ import annotations

NATIVE_MM_POLICIES = ("memory", "speed")


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
    }
    value = aliases.get(value, value)
    if value not in NATIVE_MM_POLICIES:
        allowed = ", ".join(NATIVE_MM_POLICIES)
        raise ValueError(f"unsupported native MM quantization policy {policy!r}; expected one of: {allowed}")
    return value


def should_quantize_linear(name: str, weight_numel: int, *, min_params: int, policy: str | None = "memory") -> bool:
    """Return whether a Linear module should be replaced by MM8/MM4.

    ``memory`` keeps the historical size-gated behavior. ``speed`` only swaps
    ``lm_head`` so cached decode stays dense through the recurrent/FFN path.
    """

    if int(weight_numel) < int(min_params):
        return False
    policy = normalize_native_mm_policy(policy)
    if policy == "memory":
        return True
    return name == "lm_head" or name.endswith(".lm_head")
