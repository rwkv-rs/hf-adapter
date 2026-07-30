"""Versioned public protocol shared by RWKV-7 optional kernels.

The protocol intentionally exposes capabilities rather than implementation
classes.  The Hugging Face package owns model objects and public cache/output
semantics; this companion package may execute an accepted request and returns
plain tensors/mappings only.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict


RWKV7_KERNEL_API_VERSION = 4

# Public API-v4 operation names are frozen for the 1.0 line.  New backends
# implement one or more of these operations behind ``execute_optional_v4``;
# model repositories never import an implementation module directly.
RWKV7_OPTIONAL_OPERATIONS = (
    "training_program",
    "model_forward",
    "linear_training",
    "mix6_training",
    "recurrent",
)


def _require_exact_type(name: str, value: Any, expected: type) -> None:
    if type(value) is not expected:
        raise TypeError(
            f"{name} must be exactly {expected.__name__}; got {type(value).__name__}"
        )


OptionalKernelKind = Literal[
    "training_program",
    "model_forward",
    "linear_training",
    "mix6_training",
    "recurrent",
]


class OptionalKernelEnvelope(TypedDict):
    """Normalized response returned by the API-v4 backend facade.

    Every operation has the same outer shape.  ``result`` is deliberately
    opaque here because tensor/result validation remains operation-specific at
    the clean Hugging Face boundary.
    """

    api_version: int
    kind: OptionalKernelKind
    supported: bool
    implementation: str
    reason: str
    result: Any
    phase: str


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

    _require_exact_type("supported", supported, bool)
    _require_exact_type("implementation", implementation, str)
    _require_exact_type("reason", reason, str)
    if phase is not None:
        _require_exact_type("phase", phase, str)
    result: dict[str, Any] = {
        "supported": supported,
        "implementation": implementation,
        "reason": reason,
    }
    if phase is not None:
        result["phase"] = phase
    return result  # type: ignore[return-value]


def optional_kernel_result(
    *,
    kind: OptionalKernelKind,
    supported: bool,
    implementation: str,
    reason: str,
    result: Any,
    phase: str,
) -> OptionalKernelEnvelope:
    """Build the single stable execution envelope exposed by API v4."""

    _require_exact_type("kind", kind, str)
    if kind not in RWKV7_OPTIONAL_OPERATIONS:
        raise ValueError(f"unknown optional-kernel kind {kind!r}")
    _require_exact_type("supported", supported, bool)
    _require_exact_type("implementation", implementation, str)
    _require_exact_type("reason", reason, str)
    _require_exact_type("phase", phase, str)
    if not supported and result is not None:
        raise ValueError("unsupported optional-kernel results must be None")
    return {
        "api_version": RWKV7_KERNEL_API_VERSION,
        "kind": kind,
        "supported": supported,
        "implementation": implementation,
        "reason": reason,
        "result": result,
        "phase": phase,
    }


def validate_optional_kernel_result(
    value: Any, *, expected_kind: OptionalKernelKind | None = None
) -> OptionalKernelEnvelope:
    """Validate the common API-v4 envelope without inspecting its payload."""

    if not isinstance(value, dict):
        raise TypeError("execute_optional_v4() must return a dict")
    required = {
        "api_version",
        "kind",
        "supported",
        "implementation",
        "reason",
        "result",
        "phase",
    }
    missing = required - set(value)
    if missing:
        names = ", ".join(sorted(missing))
        raise TypeError(f"execute_optional_v4() result is missing: {names}")
    _require_exact_type("api_version", value["api_version"], int)
    if value["api_version"] != RWKV7_KERNEL_API_VERSION:
        raise ValueError(
            "execute_optional_v4() API version mismatch: "
            f"expected {RWKV7_KERNEL_API_VERSION}, got {value['api_version']!r}"
        )
    kind = value["kind"]
    _require_exact_type("kind", kind, str)
    _require_exact_type("supported", value["supported"], bool)
    _require_exact_type("implementation", value["implementation"], str)
    _require_exact_type("reason", value["reason"], str)
    _require_exact_type("phase", value["phase"], str)
    if expected_kind is not None and kind != expected_kind:
        raise ValueError(
            "execute_optional_v4() kind mismatch: "
            f"expected {expected_kind!r}, got {kind!r}"
        )
    if kind not in RWKV7_OPTIONAL_OPERATIONS:
        raise ValueError(f"execute_optional_v4() returned unknown kind {kind!r}")
    if not value["supported"] and value["result"] is not None:
        raise ValueError("unsupported optional-kernel results must be None")
    return optional_kernel_result(
        kind=kind,
        supported=value["supported"],
        implementation=value["implementation"],
        reason=value["reason"],
        result=value["result"],
        phase=value["phase"],
    )


def validate_support_result(
    value: Any, *, probe_name: str = "kernel probe"
) -> KernelSupport:
    """Validate a public probe response before dispatch."""

    if not isinstance(value, dict):
        raise TypeError(f"{probe_name}() must return a dict")
    missing = {"supported", "implementation", "reason"} - set(value)
    if missing:
        names = ", ".join(sorted(missing))
        raise TypeError(f"{probe_name}() result is missing: {names}")
    _require_exact_type("supported", value["supported"], bool)
    _require_exact_type("implementation", value["implementation"], str)
    _require_exact_type("reason", value["reason"], str)
    if "phase" in value:
        _require_exact_type("phase", value["phase"], str)
    return support_result(
        supported=value["supported"],
        implementation=value["implementation"],
        reason=value["reason"],
        phase=None if "phase" not in value else value["phase"],
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
    "RWKV7_OPTIONAL_OPERATIONS",
    "KernelSupport",
    "ModelForwardResult",
    "ModelForwardSupport",
    "OptionalKernelEnvelope",
    "OptionalKernelKind",
    "RecurrentSupport",
    "optional_kernel_result",
    "support_result",
    "validate_model_request",
    "validate_model_result",
    "validate_optional_kernel_result",
    "validate_support_result",
]
