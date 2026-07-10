"""Versioned public protocol shared by RWKV-7 optional kernels.

The protocol intentionally exposes capabilities rather than implementation
classes.  The Hugging Face package owns model objects and public cache/output
semantics; this companion package may execute an accepted request and returns
plain tensors/mappings only.
"""
from __future__ import annotations

from typing import Any, TypedDict


RWKV7_KERNEL_API_VERSION = 2


class KernelSupport(TypedDict):
    """Capability decision returned by every public probe."""

    supported: bool
    implementation: str
    reason: str


RecurrentSupport = KernelSupport


class ModelForwardSupport(KernelSupport, total=False):
    phase: str


class ModelForwardResult(TypedDict, total=False):
    """Tensor-only result returned by :func:`model_forward_v1`.

    ``output_kind`` is mandatory and is either ``"base"`` or ``"causal_lm"``.
    The remaining fields mirror the corresponding Transformers output without
    importing Transformers into the optional kernel package.
    """

    output_kind: str
    last_hidden_state: Any
    logits: Any
    loss: Any
    past_key_values: Any
    hidden_states: Any


def support_result(
    *, supported: bool, implementation: str, reason: str, phase: str | None = None
) -> KernelSupport:
    """Build a normalized capability result."""

    result: dict[str, Any] = {
        "supported": bool(supported),
        "implementation": str(implementation),
        "reason": str(reason),
    }
    if phase is not None:
        result["phase"] = str(phase)
    return result  # type: ignore[return-value]


def validate_support_result(value: Any, *, probe_name: str = "kernel probe") -> KernelSupport:
    """Validate a public probe response before dispatch."""

    if not isinstance(value, dict):
        raise TypeError(f"{probe_name}() must return a dict")
    missing = {"supported", "implementation", "reason"} - set(value)
    if missing:
        names = ", ".join(sorted(missing))
        raise TypeError(f"{probe_name}() result is missing: {names}")
    return support_result(
        supported=bool(value["supported"]),
        implementation=str(value["implementation"]),
        reason=str(value["reason"]),
        phase=None if "phase" not in value else str(value["phase"]),
    )


def validate_model_request(value: Any) -> dict[str, Any]:
    """Validate the stable model-forward request envelope.

    Field-level shape/dtype capability belongs to the selected implementation;
    this function only protects the public ABI from malformed callers.
    """

    if not isinstance(value, dict):
        raise TypeError("model-forward request must be a dict")
    missing = {"model_kind", "training", "use_cache"} - set(value)
    if missing:
        names = ", ".join(sorted(missing))
        raise TypeError(f"model-forward request is missing: {names}")
    if value["model_kind"] not in ("base", "causal_lm"):
        raise ValueError("model_kind must be 'base' or 'causal_lm'")
    return value


def validate_model_result(value: Any, *, expected_kind: str) -> ModelForwardResult:
    """Validate a model-forward response without importing HF output types."""

    if not isinstance(value, dict):
        raise TypeError("model_forward_v1() must return a dict")
    kind = value.get("output_kind")
    if kind != expected_kind:
        raise ValueError(
            f"model_forward_v1() output_kind mismatch: expected {expected_kind!r}, got {kind!r}"
        )
    required = {"last_hidden_state"} if kind == "base" else {"logits"}
    missing = required - set(value)
    if missing:
        names = ", ".join(sorted(missing))
        raise TypeError(f"model_forward_v1() result is missing: {names}")
    return value


__all__ = [
    "RWKV7_KERNEL_API_VERSION",
    "KernelSupport",
    "ModelForwardResult",
    "ModelForwardSupport",
    "RecurrentSupport",
    "support_result",
    "validate_model_request",
    "validate_model_result",
    "validate_support_result",
]
