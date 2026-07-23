"""Fail-closed selection for optional RWKV-7 training leaf protocols."""

from __future__ import annotations

import os
from typing import Any

import torch

from .linear.training_flattened import flattened_linear as _run_flattened
from .linear.training_flattened import (
    probe_linear_training_v1 as _probe_flattened,
)
from .protocol import validate_support_result
from .recurrent.training_factorized import (
    _run_factorized_recurrent as _run_factorized,
)
from .recurrent.training_factorized import (
    TOKEN_CHUNK_LENGTH,
    probe_recurrent_training_v1 as _probe_factorized,
)
from .recurrent.training_matrix import _batched_matrix_recurrence as _run_matrix
from .recurrent.training_matrix import (
    probe_recurrent_training_v1 as _probe_matrix,
)
from .trace import record_linear as _record_linear_trace
from .trace import record_mix6 as _record_mix6_trace
from .trace import record_recurrent as _record_recurrent_trace
from .time_mix.training_mix6 import _run_mix6_training as _run_mix6
from .time_mix.training_mix6 import (
    probe_mix6_training_v1 as _probe_mix6,
)


_TRAINING_IMPL_ENV = "RWKV7_TRAINING_KERNEL_IMPL"
_TRAINING_IMPLS = ("auto", "adaptive", "matrix", "factorized")
_MATRIX_IMPLEMENTATION = "torch-cuda-rwkv7-batched-matrix-recurrent-training-v1"
_FACTORIZED_IMPLEMENTATION = "native-nvidia-rwkv7-factorized-recurrent-training-v1"
_FLATTENED_IMPLEMENTATION = "torch-cuda-rwkv7-flattened-linear-training-v1"
_MIX6_IMPLEMENTATION = "native-nvidia-rwkv7-mix6-training-v1"
_RECURRENT_HINT_NAMES = frozenset(
    ("fully_active", "initial_state_zero", "token_aligned")
)


def _requested_implementation() -> str:
    name = os.environ.get(_TRAINING_IMPL_ENV, "auto").strip().lower()
    if name not in _TRAINING_IMPLS:
        choices = ", ".join(_TRAINING_IMPLS)
        raise ValueError(f"{_TRAINING_IMPL_ENV} must be one of {choices}; got {name!r}")
    return name


def _attention_mask(args: tuple[Any, ...], kwargs: dict[str, Any]):
    if "attention_mask" in kwargs:
        return kwargs["attention_mask"]
    return args[7] if len(args) > 7 else None


def _recurrent_shape(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[int, ...] | None:
    receptance = kwargs.get("receptance", args[0] if args else None)
    if not isinstance(receptance, torch.Tensor):
        return None
    return tuple(receptance.shape)


def _recurrent_request_is_fully_active(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> bool:
    fully_active = kwargs.get("fully_active")
    if isinstance(fully_active, bool):
        return fully_active
    attention_mask = _attention_mask(args, kwargs)
    if attention_mask is None:
        return True
    if not isinstance(attention_mask, torch.Tensor):
        return False
    return bool(attention_mask.to(dtype=torch.bool).all().detach().cpu())


def _recurrent_request_is_token_aligned(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> bool:
    token_aligned = kwargs.get("token_aligned")
    if isinstance(token_aligned, bool):
        return token_aligned
    shape = _recurrent_shape(args, kwargs)
    return bool(
        shape is not None
        and len(shape) == 4
        and shape[1] > 0
        and shape[1] % TOKEN_CHUNK_LENGTH == 0
    )


def _recurrent_leaf_kwargs(
    kwargs: dict[str, Any], *, include_initial_state_zero: bool = False
) -> dict[str, Any]:
    """Remove dispatcher-only request hints before calling a leaf probe."""

    leaf_kwargs = {
        name: value
        for name, value in kwargs.items()
        if name not in _RECURRENT_HINT_NAMES
    }
    if include_initial_state_zero and "initial_state_zero" in kwargs:
        leaf_kwargs["initial_state_zero"] = kwargs["initial_state_zero"]
    return leaf_kwargs


def _invalid_recurrent_hint(kwargs: dict[str, Any]) -> str | None:
    for name in sorted(_RECURRENT_HINT_NAMES):
        value = kwargs.get(name)
        if value is not None and not isinstance(value, bool):
            return f"{name} must be a bool or None"
    return None


def _validated_recurrent_probe(probe, *args: Any, **kwargs: Any) -> dict[str, Any]:
    result = validate_support_result(
        probe(
            *args,
            **_recurrent_leaf_kwargs(
                kwargs,
                include_initial_state_zero=(probe is _probe_factorized),
            ),
        ),
        probe_name="probe_recurrent_training_v1",
    )
    expected = (
        _FACTORIZED_IMPLEMENTATION
        if probe is _probe_factorized
        else _MATRIX_IMPLEMENTATION
    )
    if result["implementation"] != expected:
        raise TypeError(
            "recurrent leaf probe returned an unexpected implementation: "
            f"expected {expected!r}, got {result['implementation']!r}"
        )
    return result


def _adaptive_recurrent_probe(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Use the fast dense leaf only where its accepted contract applies."""

    initial_state_zero = kwargs.get("initial_state_zero") is True
    # Without model-owned cache provenance, adaptive must select the exact
    # matrix leaf.  Check this before inspecting the mask so standalone calls
    # do not pay a device-to-host scalar copy merely to fail closed.
    fully_active = (
        _recurrent_request_is_fully_active(args, kwargs) if initial_state_zero else True
    )
    token_aligned = (
        _recurrent_request_is_token_aligned(args, kwargs)
        if initial_state_zero
        else True
    )
    if fully_active and token_aligned and initial_state_zero:
        factorized = _validated_recurrent_probe(
            _probe_factorized,
            *args,
            **kwargs,
        )
        if factorized["supported"]:
            return factorized
        matrix = _validated_recurrent_probe(_probe_matrix, *args, **kwargs)
        if matrix["supported"]:
            matrix = dict(matrix)
            matrix["reason"] = (
                "adaptive exact fallback after the factorized route declined: "
                f"{factorized['reason']}; {matrix['reason']}"
            )
        return matrix

    matrix = _validated_recurrent_probe(_probe_matrix, *args, **kwargs)
    if matrix["supported"]:
        matrix = dict(matrix)
        if not initial_state_zero:
            request_kind = (
                "a recurrent request without model-proven zero initial-state provenance"
            )
        elif not fully_active:
            request_kind = "a masked recurrent request"
        else:
            request_kind = (
                "an unaligned recurrent request; the factorized leaf requires "
                f"token lengths divisible by {TOKEN_CHUNK_LENGTH}"
            )
        matrix["reason"] = (
            f"adaptive exact route for {request_kind}; {matrix['reason']}"
        )
    return matrix


def probe_recurrent_training_v1(*args: Any, **kwargs: Any):
    """Report one explicit training leaf while production auto stays reference."""

    requested = _requested_implementation()
    invalid_hint = _invalid_recurrent_hint(kwargs)
    if invalid_hint is not None:
        implementation = (
            _FACTORIZED_IMPLEMENTATION
            if requested == "factorized"
            else _MATRIX_IMPLEMENTATION
        )
        return {
            "supported": False,
            "implementation": implementation,
            "reason": f"invalid recurrent request hint: {invalid_hint}",
        }
    if requested == "auto":
        return {
            "supported": False,
            "implementation": _MATRIX_IMPLEMENTATION,
            "reason": (
                "production auto keeps training on reference until the adaptive "
                "full-model release gate passes"
            ),
        }
    if requested == "adaptive":
        return _adaptive_recurrent_probe(*args, **kwargs)
    probe = _probe_matrix if requested == "matrix" else _probe_factorized
    return _validated_recurrent_probe(probe, *args, **kwargs)


def execute_recurrent_training_v1(
    receptance: torch.Tensor,
    decay: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    fully_active: bool | None = None,
    initial_state_zero: bool | None = None,
    token_aligned: bool | None = None,
) -> dict[str, Any]:
    """Validate and execute one recurrent request as an atomic transaction."""

    args = (
        receptance,
        decay,
        key,
        value,
        a,
        b,
        initial_state,
        attention_mask,
    )
    hints = {
        "fully_active": fully_active,
        "initial_state_zero": initial_state_zero,
        "token_aligned": token_aligned,
    }
    support = probe_recurrent_training_v1(*args, **hints)
    if not support["supported"]:
        return {**support, "result": None}
    implementation = str(support["implementation"])
    if implementation == _MATRIX_IMPLEMENTATION:
        result = _run_matrix(*args)
    elif implementation == _FACTORIZED_IMPLEMENTATION:
        result = _run_factorized(
            *args,
            fully_active=fully_active,
            initial_state_zero=initial_state_zero,
            token_aligned=token_aligned,
        )
    else:
        raise RuntimeError(
            "recurrent training probe selected an unknown implementation: "
            f"{implementation}"
        )
    _record_recurrent_trace(implementation)
    return {**support, "result": result}


def recurrent_training_v1(*args: Any, **kwargs: Any):
    """Execute one fully validated recurrent training request."""

    execution = execute_recurrent_training_v1(*args, **kwargs)
    if not execution["supported"]:
        raise RuntimeError(str(execution["reason"]))
    return execution["result"]


def probe_linear_training_v1(*args: Any, **kwargs: Any):
    """Report an exact or flattened projection for the selected candidate."""

    requested = _requested_implementation()
    fully_active = kwargs.get("fully_active")
    token_aligned = kwargs.get("token_aligned")
    if requested == "auto":
        return {
            "supported": False,
            "implementation": _FLATTENED_IMPLEMENTATION,
            "reason": (
                "production auto keeps training linears on reference until the "
                "full-model precision and performance release gates pass"
            ),
        }
    if requested == "matrix":
        return {
            "supported": False,
            "implementation": "torch-reference-linear-v1",
            "reason": (
                "the exact matrix candidate accelerates only the recurrent leaf; "
                "linears retain the readable reference accumulation contract"
            ),
        }
    if requested == "adaptive" and (
        fully_active is not True or token_aligned is not True
    ):
        request_kind = (
            "masked or standalone"
            if fully_active is not True
            else "token-length-unaligned"
        )
        return {
            "supported": False,
            "implementation": "torch-reference-linear-v1",
            "reason": (
                "the adaptive candidate retains reference linears for "
                f"{request_kind} requests"
            ),
        }
    return validate_support_result(
        _probe_flattened(*args, **kwargs),
        probe_name="probe_linear_training_v1",
    )


def execute_linear_training_v1(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    fully_active: bool | None = None,
    token_aligned: bool | None = None,
) -> dict[str, Any]:
    """Validate and execute one projection as one call-local transaction.

    The returned support envelope is valid only for this execution.  It is not
    a reusable capability token, so tensors cannot change between validation
    and execution and callers have no trusted bypass to retain across calls.
    """

    support = probe_linear_training_v1(
        value,
        weight,
        bias,
        fully_active=fully_active,
        token_aligned=token_aligned,
    )
    if not support["supported"]:
        return {**support, "output": None}
    implementation = str(support["implementation"])
    if implementation != _FLATTENED_IMPLEMENTATION:
        raise RuntimeError(
            "linear training probe selected an unknown implementation: "
            f"{implementation}"
        )
    output = _run_flattened(value, weight, bias)
    _record_linear_trace(implementation)
    return {**support, "output": output}


def linear_training_v1(*args: Any, **kwargs: Any):
    """Execute a fully validated stateless flattened projection."""

    result = execute_linear_training_v1(*args, **kwargs)
    if not result["supported"]:
        raise RuntimeError(str(result["reason"]))
    return result["output"]


def probe_mix6_training_v1(
    value: torch.Tensor,
    shifted: torch.Tensor,
    *mixes: torch.Tensor,
    fully_active: bool | None = None,
    token_aligned: bool | None = None,
):
    """Report support for the stateless native six-way token-mix leaf."""

    # Masking and shift-state semantics have already been resolved into the
    # explicit ``shifted`` tensor by the readable model.  Unlike the recurrent
    # and flattened-linear candidates, this leaf therefore has no padding or
    # token-chunk alignment restriction.
    del fully_active, token_aligned

    requested = _requested_implementation()
    if requested == "auto":
        return {
            "supported": False,
            "implementation": _MIX6_IMPLEMENTATION,
            "reason": (
                "production auto keeps Mix6 on reference until the adaptive "
                "full-model release gate passes"
            ),
        }
    if requested == "matrix":
        return {
            "supported": False,
            "implementation": "torch-reference-mix6-v1",
            "reason": "the matrix candidate accelerates only the recurrent leaf",
        }
    return validate_support_result(
        _probe_mix6(value, shifted, *mixes),
        probe_name="probe_mix6_training_v1",
    )


def execute_mix6_training_v1(
    value: torch.Tensor,
    shifted: torch.Tensor,
    *mixes: torch.Tensor,
    fully_active: bool | None = None,
    token_aligned: bool | None = None,
) -> dict[str, Any]:
    """Validate and execute one explicit-shift Mix6 request atomically."""

    support = probe_mix6_training_v1(
        value,
        shifted,
        *mixes,
        fully_active=fully_active,
        token_aligned=token_aligned,
    )
    if not support["supported"]:
        return {**support, "result": None}
    implementation = str(support["implementation"])
    if implementation != _MIX6_IMPLEMENTATION:
        raise RuntimeError(
            f"Mix6 training probe selected an unknown implementation: {implementation}"
        )
    result = _run_mix6(value, shifted, *mixes)
    _record_mix6_trace(implementation)
    return {**support, "result": result}


def mix6_training_v1(
    value: torch.Tensor,
    shifted: torch.Tensor,
    *mixes: torch.Tensor,
    fully_active: bool | None = None,
    token_aligned: bool | None = None,
) -> tuple[torch.Tensor, ...]:
    """Execute one fully validated explicit-shift Mix6 request."""

    execution = execute_mix6_training_v1(
        value,
        shifted,
        *mixes,
        fully_active=fully_active,
        token_aligned=token_aligned,
    )
    if not execution["supported"]:
        raise RuntimeError(str(execution["reason"]))
    return execution["result"]


__all__ = [
    "execute_linear_training_v1",
    "execute_mix6_training_v1",
    "execute_recurrent_training_v1",
    "linear_training_v1",
    "mix6_training_v1",
    "probe_linear_training_v1",
    "probe_mix6_training_v1",
    "probe_recurrent_training_v1",
    "recurrent_training_v1",
]
