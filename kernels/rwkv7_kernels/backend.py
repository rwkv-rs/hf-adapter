"""Single API-v4 facade for every optional RWKV-7 backend operation.

The clean Hugging Face package calls only :func:`execute_optional_v4`.  Policy,
capability probing, execution, and result-envelope normalization stay in this
optional package.  The existing v1 dispatchers remain the implementation
adapters so the v4 boundary does not change established kernel behavior.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import secrets
import threading
from typing import Any

import torch

from .dispatcher import probe_recurrent_v1, recurrent_v1
from .model_dispatcher import model_forward_v1, probe_model_forward_v1
from .protocol import (
    OptionalKernelEnvelope,
    OptionalKernelKind,
    RWKV7_OPTIONAL_OPERATIONS,
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
_MAX_PROGRAM_CERTIFICATES = 4096
_PROGRAM_CERTIFICATES: OrderedDict[str, dict[str, Any]] = OrderedDict()
_PROGRAM_CERTIFICATE_LOCK = threading.Lock()
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
            normalized["implementation"] if implementation is None else implementation
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


def _issue_program_certificate(
    hidden_states: torch.Tensor,
    *,
    fully_active: bool,
    initial_state_zero: bool | None,
    token_aligned: bool,
    autograd_leaf_eligible: bool,
) -> str:
    """Create one opaque process-local certificate for a model call."""

    token = f"{_TRAINING_PROGRAM_ID}:{secrets.token_hex(16)}"
    certificate = {
        "shape": tuple(hidden_states.shape),
        "device_type": hidden_states.device.type,
        "device_index": hidden_states.device.index,
        "dtype": hidden_states.dtype,
        "facts": {
            "fully_active": fully_active,
            "initial_state_zero": initial_state_zero,
            "token_aligned": token_aligned,
            "autograd_leaf_eligible": autograd_leaf_eligible,
        },
    }
    with _PROGRAM_CERTIFICATE_LOCK:
        _PROGRAM_CERTIFICATES[token] = certificate
        _PROGRAM_CERTIFICATES.move_to_end(token)
        while len(_PROGRAM_CERTIFICATES) > _MAX_PROGRAM_CERTIFICATES:
            _PROGRAM_CERTIFICATES.popitem(last=False)
    return token


def _program_certificate(program_id: str) -> dict[str, Any] | None:
    with _PROGRAM_CERTIFICATE_LOCK:
        certificate = _PROGRAM_CERTIFICATES.get(program_id)
        if certificate is None:
            return None
        _PROGRAM_CERTIFICATES.move_to_end(program_id)
        return certificate


def _certificate_matches_leaf(
    certificate: Mapping[str, Any],
    args: tuple[Any, ...],
    facts: Mapping[str, Any],
) -> str | None:
    """Return a reason when a certified leaf no longer matches its model call."""

    value = args[0] if args else None
    if not isinstance(value, torch.Tensor) or value.ndim not in (3, 4):
        return "certified training leaves require a rank-three or rank-four tensor"
    shape = certificate.get("shape")
    if not isinstance(shape, tuple) or len(shape) != 3:
        return "the program certificate has an invalid hidden-state shape"
    if tuple(value.shape[:2]) != tuple(shape[:2]):
        return "leaf batch/token shape does not match the program certificate"
    if (
        value.device.type != certificate.get("device_type")
        or value.device.index != certificate.get("device_index")
        or value.dtype != certificate.get("dtype")
    ):
        return "leaf device or dtype does not match the program certificate"
    expected_facts = certificate.get("facts")
    if not isinstance(expected_facts, Mapping):
        return "the program certificate has invalid model facts"
    for name, expected in expected_facts.items():
        if facts.get(name) != expected:
            return f"leaf fact {name!r} does not match the program certificate"
    return None


def _leaf_request(
    kind: OptionalKernelKind,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    accepted_facts: frozenset[str],
) -> tuple[dict[str, Any] | None, OptionalKernelEnvelope | None]:
    """Translate the v4 program certificate and fact mapping to v1 hints."""

    program_id = kwargs.pop("program_id", None)
    certificate = None
    if program_id is not None:
        if not isinstance(program_id, str):
            return None, _unsupported_program(kind, program_id=program_id)
        certificate = _program_certificate(program_id)
        if certificate is None:
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

    if certificate is not None:
        mismatch = _certificate_matches_leaf(certificate, args, facts)
        if mismatch is not None:
            return None, optional_kernel_result(
                kind=kind,
                supported=False,
                implementation=_REFERENCE_PROGRAM_ID,
                reason=mismatch,
                result=None,
                phase="training",
            )

    hints = {name: facts[name] for name in accepted_facts if name in facts}
    for name in accepted_facts:
        if name in kwargs:
            hints[name] = kwargs.pop(name)
    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise TypeError(f"unexpected {kind} options: {names}")
    if kind in ("linear_training", "recurrent"):
        hints["adaptive_fast_program"] = True if certificate is not None else None
    return hints, None


def _execute_training_program(*args: Any, **kwargs: Any) -> OptionalKernelEnvelope:
    hidden_states = args[0] if args else kwargs.get("hidden_states")
    attention_mask = args[1] if len(args) > 1 else kwargs.get("attention_mask")
    sequence = None
    if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
        sequence = int(attention_mask.shape[1])
    elif isinstance(hidden_states, torch.Tensor) and hidden_states.ndim == 3:
        sequence = int(hidden_states.shape[1])
    token_aligned = bool(sequence is not None and sequence % TOKEN_CHUNK_LENGTH == 0)

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
    result = None
    if normalized["supported"]:
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError("supported training preflight requires hidden_states")
        program_id = _issue_program_certificate(
            hidden_states,
            fully_active=bool(kwargs["fully_active"]),
            initial_state_zero=kwargs.get("initial_state_zero"),
            token_aligned=token_aligned,
            autograd_leaf_eligible=bool(kwargs["autograd_leaf_eligible"]),
        )
        result = {"program_id": program_id, "token_aligned": token_aligned}
    return _envelope(
        "training_program",
        normalized,
        result=result,
        phase="training",
        implementation=normalized["implementation"],
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
        args,
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
        args,
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
            args,
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
    if kind not in RWKV7_OPTIONAL_OPERATIONS:
        choices = ", ".join(RWKV7_OPTIONAL_OPERATIONS)
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
