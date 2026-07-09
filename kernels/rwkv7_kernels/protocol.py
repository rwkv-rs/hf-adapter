"""Versioned public protocol shared by RWKV-7 optional kernels."""
from __future__ import annotations

from typing import Any, TypedDict


RWKV7_KERNEL_API_VERSION = 1


class RecurrentSupport(TypedDict):
    """Capability decision returned by :func:`probe_recurrent_v1`."""

    supported: bool
    implementation: str
    reason: str


def support_result(
    *, supported: bool, implementation: str, reason: str
) -> RecurrentSupport:
    """Build a normalized recurrent-v1 capability result."""

    return {
        "supported": bool(supported),
        "implementation": str(implementation),
        "reason": str(reason),
    }


def validate_support_result(value: Any) -> RecurrentSupport:
    """Validate a recurrent-v1 probe response before dispatch."""

    if not isinstance(value, dict):
        raise TypeError("probe_recurrent_v1() must return a dict")
    missing = {"supported", "implementation", "reason"} - set(value)
    if missing:
        names = ", ".join(sorted(missing))
        raise TypeError(f"probe_recurrent_v1() result is missing: {names}")
    return support_result(
        supported=bool(value["supported"]),
        implementation=str(value["implementation"]),
        reason=str(value["reason"]),
    )


__all__ = [
    "RWKV7_KERNEL_API_VERSION",
    "RecurrentSupport",
    "support_result",
    "validate_support_result",
]
