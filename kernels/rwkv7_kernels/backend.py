"""Single API-v4 facade for every optional RWKV-7 backend operation.

The clean Hugging Face package calls only :func:`execute_optional_v4`.  Policy,
capability probing, execution, and result-envelope normalization stay in this
optional package.  The existing v1 dispatchers remain the implementation
adapters so the v4 boundary does not change established kernel behavior.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from .dispatcher import probe_recurrent_v1, recurrent_v1
from .model_dispatcher import model_forward_v1, probe_model_forward_v1
from .protocol import (
    OptionalKernelEnvelope,
    OptionalKernelKind,
    optional_kernel_result,
    validate_support_result,
)
from .recurrent.training_factorized import TOKEN_CHUNK_LENGTH
from .training_dispatcher import (
    execute_linear_training_v1,
    execute_mix6_training_v1,
    execute_recurrent_training_v1,
    probe_training_program_v1,
)


_TRAINING_PROGRAM_ID = "native-nvidia-rwkv7-adaptive-training-program-v1"
_REFERENCE_PROGRAM_ID = "torch-reference-training-program-v1"
_KINDS = (
    "training_program",
    "model_forward",
    "linear_training",
    "mix6_training",
    "recurrent",
)
_FACT_NAMES = frozenset(
    {
        "fully_active",
        "initial_state_zero",
        "token_aligned",
        "autograd_leaf_eligible",
        "force_reference_program",
    }
)


def _envelope(
    kind: OptionalKernelKind,
    support: Mapping[str, Any],
    *,
    result: Any,
    phase: str,
    implementation: str | None = None,
) -> OptionalKernelEnvelope:
    normalized = validate_support_result(support, probe_name=f"{kind} backend")
    return optional_kernel_result(
        kind=kind,
        supported=normalized["supported"],
        implementation=(
            normalized["implementation"]
            if implementation is None
            else implementation
        ),
        reason=normalized["reason"],
        result=result,
        phase=phase,
    )


def _unsupported_program(
    kind: OptionalKernelKind,
    *,
    program_id: Any,
    phase: str = "training",
) -> OptionalKernelEnvelope:
    return optional_kernel_result(
        kind=kind,
        supported=False,
        implementation=_REFERENCE_PROGRAM_ID,
        reason=f"unknown optional training program_id {program_id!r}",
        result=None,
        phase=phase,
    )


def _leaf_request(
    kind: OptionalKernelKind,
    kwargs: dict[str, Any],
    *,
    accepted_facts: frozenset[str],
) -> tuple[dict[str, Any] | None, OptionalKernelEnvelope | None]:
    """Translate the v4 program certificate and fact mapping to v1 hints."""

    program_id = kwargs.pop("program_id", None)
    if program_id == _TRAINING_PROGRAM_ID:
        return None, optional_kernel_result(
            kind=kind,
            supported=False,
            implementation=_REFERENCE_PROGRAM_ID,
            reason=(
                "the legacy adaptive training program certificate is disabled "
                "until every concrete recurrent, linear, and Mix6 leaf can be "
                "preflighted before execution"
            ),
            result=None,
            phase="training",
        )
    if program_id is not None:
        return None, _unsupported_program(kind, program_id=program_id)

    facts = kwargs.pop("facts", None)
    if facts is None:
        facts = {}
    if not isinstance(facts, Mapping):
        raise TypeError("facts must be a mapping or None")
    unknown = set(facts) - _FACT_NAMES
    if unknown:
        names = ", ".join(sorted(str(name) for name in unknown))
        raise TypeError(f"unknown training facts: {names}")
    force_reference = facts.get("force_reference_program", False)
    if not isinstance(force_reference, bool):
        raise TypeError("force_reference_program fact must be a bool")
    if force_reference:
        return None, optional_kernel_result(
            kind=kind,
            supported=False,
            implementation=_REFERENCE_PROGRAM_ID,
            reason="the model selected one readable reference training program",
            result=None,
            phase="training",
        )

    hints = {
        name: facts[name]
        for name in accepted_facts
        if name in facts
    }
    for name in accepted_facts:
        if name in kwargs:
            hints[name] = kwargs.pop(name)
    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected {kind} options: {names}")
    if kind in ("linear_training", "recurrent"):
        hints["adaptive_fast_program"] = None
    return hints, None


def _execute_training_program(
    *args: Any, **kwargs: Any
) -> OptionalKernelEnvelope:
    hidden_states = args[0] if args else kwargs.get("hidden_states")
    attention_mask = args[1] if len(args) > 1 else kwargs.get("attention_mask")
    sequence = None
    if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
        sequence = int(attention_mask.shape[1])
    elif isinstance(hidden_states, torch.Tensor) and hidden_states.ndim == 3:
        sequence = int(hidden_states.shape[1])
    token_aligned = bool(
        sequence is not None and sequence % TOKEN_CHUNK_LENGTH == 0
    )

    requested_training = kwargs.pop("training", True)
    if requested_training is not True:
        raise ValueError("training_program is available only for training=True")
    # API v4 owns this shape policy. Ignore a same-valued v3 migration hint but
    # never allow callers to forge the decision.
    supplied_alignment = kwargs.pop("token_aligned", token_aligned)
    if not isinstance(supplied_alignment, bool):
        raise TypeError("token_aligned must be a bool when supplied")
    if supplied_alignment != token_aligned:
        raise ValueError("token_aligned does not match the API-v4 shape decision")
    force_reference = kwargs.pop("force_reference_program", False)
    if not isinstance(force_reference, bool):
        raise TypeError("force_reference_program must be a bool")
    if force_reference:
        return optional_kernel_result(
            kind="training_program",
            supported=False,
            implementation=_REFERENCE_PROGRAM_ID,
            reason="the model requested one readable reference training program",
            result=None,
            phase="training",
        )

    support = probe_training_program_v1(
        *args,
        **kwargs,
        training=True,
        token_aligned=token_aligned,
    )
    normalized = validate_support_result(
        support, probe_name="probe_training_program_v1"
    )
    # API v4 cannot yet describe the complete concrete leaf plan.  Keep the
    # public facade fail-closed even if a private/experimental probe is
    # monkeypatched or accidentally regresses to a partial positive result.
    if normalized["supported"]:
        normalized = {
            "supported": False,
            "implementation": normalized["implementation"],
            "reason": (
                "the private training probe cannot issue a public certificate "
                "until API v4 binds every concrete recurrent, linear, and Mix6 "
                "leaf before execution"
            ),
        }
    return _envelope(
        "training_program",
        normalized,
        result=None,
        phase="training",
        implementation=_REFERENCE_PROGRAM_ID,
    )


def _execute_model_forward(
    owner: Any, request: dict[str, Any]
) -> OptionalKernelEnvelope:
    support = validate_support_result(
        probe_model_forward_v1(owner, request),
        probe_name="probe_model_forward_v1",
    )
    phase = support.get("phase", "training" if request["training"] else "prefill")
    if not support["supported"]:
        return _envelope("model_forward", support, result=None, phase=phase)
    result = model_forward_v1(owner, request)
    implementation = result.get("implementation", support["implementation"])
    phase = result.get("phase", phase)
    return _envelope(
        "model_forward",
        support,
        result=result,
        phase=phase,
        implementation=implementation,
    )


def _execute_linear_training(*args: Any, **kwargs: Any) -> OptionalKernelEnvelope:
    hints, declined = _leaf_request(
        "linear_training",
        kwargs,
        accepted_facts=frozenset(
            {"fully_active", "initial_state_zero", "token_aligned"}
        ),
    )
    if declined is not None:
        return declined
    assert hints is not None
    execution = execute_linear_training_v1(*args, **hints)
    return _envelope(
        "linear_training",
        execution,
        result=execution.get("output"),
        phase="training",
    )


def _execute_mix6_training(*args: Any, **kwargs: Any) -> OptionalKernelEnvelope:
    hints, declined = _leaf_request(
        "mix6_training",
        kwargs,
        accepted_facts=frozenset({"fully_active", "token_aligned"}),
    )
    if declined is not None:
        return declined
    assert hints is not None
    execution = execute_mix6_training_v1(*args, **hints)
    return _envelope(
        "mix6_training",
        execution,
        result=execution.get("result"),
        phase="training",
    )


def _execute_recurrent(*args: Any, **kwargs: Any) -> OptionalKernelEnvelope:
    training = kwargs.pop("training", None)
    if not isinstance(training, bool):
        raise TypeError("recurrent requires a boolean training option")
    if training:
        hints, declined = _leaf_request(
            "recurrent",
            kwargs,
            accepted_facts=frozenset(
                {"fully_active", "initial_state_zero", "token_aligned"}
            ),
        )
        if declined is not None:
            return declined
        assert hints is not None
        execution = execute_recurrent_training_v1(*args, **hints)
        return _envelope(
            "recurrent",
            execution,
            result=execution.get("result"),
            phase="training",
        )

    if "program_id" in kwargs or "facts" in kwargs:
        raise TypeError("inference recurrent does not accept training program metadata")
    support = validate_support_result(
        probe_recurrent_v1(*args, **kwargs),
        probe_name="probe_recurrent_v1",
    )
    receptance = args[0] if args else kwargs.get("receptance")
    phase = (
        "decode"
        if isinstance(receptance, torch.Tensor)
        and receptance.ndim >= 2
        and int(receptance.shape[1]) == 1
        else "prefill"
    )
    result = recurrent_v1(*args, **kwargs) if support["supported"] else None
    return _envelope("recurrent", support, result=result, phase=phase)


def execute_optional_v4(
    kind: OptionalKernelKind, *args: Any, **kwargs: Any
) -> OptionalKernelEnvelope:
    """Probe and execute one optional operation through the stable v4 ABI.

    Unsupported requests return a normalized envelope with ``result=None``.
    Backend execution errors intentionally propagate so the HF caller can
    apply its requested auto/strict fallback policy without losing evidence.
    """

    if type(kind) is not str:
        raise TypeError(f"kind must be exactly str; got {type(kind).__name__}")
    if kind not in _KINDS:
        choices = ", ".join(_KINDS)
        raise ValueError(f"kind must be one of {choices}; got {kind!r}")
    if kind == "training_program":
        return _execute_training_program(*args, **kwargs)
    if kind == "model_forward":
        return _execute_model_forward(*args, **kwargs)
    if kind == "linear_training":
        return _execute_linear_training(*args, **kwargs)
    if kind == "mix6_training":
        return _execute_mix6_training(*args, **kwargs)
    return _execute_recurrent(*args, **kwargs)


__all__ = ["execute_optional_v4"]
