"""Fail-closed selection for optional RWKV-7 training leaf protocols."""

from __future__ import annotations

import os
from typing import Any

import torch

from .linear.training_flattened import linear_training_v1 as _run_flattened
from .linear.training_flattened import (
    probe_linear_training_v1 as _probe_flattened,
)
from .protocol import validate_support_result
from .recurrent.training_factorized import recurrent_training_v1 as _run_factorized
from .recurrent.training_factorized import (
    TOKEN_CHUNK_LENGTH,
    probe_recurrent_training_v1 as _probe_factorized,
)
from .recurrent.training_matrix import recurrent_training_v1 as _run_matrix
from .recurrent.training_matrix import (
    probe_recurrent_training_v1 as _probe_matrix,
)
from .trace import record_linear as _record_linear_trace
from .trace import record_recurrent as _record_recurrent_trace


_TRAINING_IMPL_ENV = "RWKV7_TRAINING_KERNEL_IMPL"
_TRAINING_IMPLS = ("auto", "adaptive", "matrix", "factorized")
_MATRIX_IMPLEMENTATION = "torch-cuda-rwkv7-batched-matrix-recurrent-training-v1"
_FACTORIZED_IMPLEMENTATION = "native-nvidia-rwkv7-factorized-recurrent-training-v1"
_FLATTENED_IMPLEMENTATION = "torch-cuda-rwkv7-flattened-linear-training-v1"


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
    attention_mask = _attention_mask(args, kwargs)
    if attention_mask is None:
        return True
    if not isinstance(attention_mask, torch.Tensor):
        return False
    return bool(attention_mask.to(dtype=torch.bool).all().detach().cpu())


def _recurrent_request_is_chunk_aligned(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> bool:
    shape = _recurrent_shape(args, kwargs)
    return bool(
        shape is not None
        and len(shape) == 4
        and shape[1] > 0
        and shape[1] % TOKEN_CHUNK_LENGTH == 0
    )


def _validated_recurrent_probe(
    probe, *args: Any, **kwargs: Any
) -> dict[str, Any]:
    return validate_support_result(
        probe(*args, **kwargs),
        probe_name="probe_recurrent_training_v1",
    )


def _adaptive_recurrent_probe(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Use the fast dense leaf only where its accepted contract applies."""

    fully_active = _recurrent_request_is_fully_active(args, kwargs)
    chunk_aligned = _recurrent_request_is_chunk_aligned(args, kwargs)
    if fully_active and chunk_aligned:
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
        request_kind = (
            "a masked recurrent request"
            if not fully_active
            else (
                "an unaligned recurrent request; the factorized leaf requires "
                f"token lengths divisible by {TOKEN_CHUNK_LENGTH}"
            )
        )
        matrix["reason"] = f"adaptive exact route for {request_kind}; {matrix['reason']}"
    return matrix


def probe_recurrent_training_v1(*args: Any, **kwargs: Any):
    """Report one explicit training leaf while production auto stays reference."""

    requested = _requested_implementation()
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


def recurrent_training_v1(*args: Any, **kwargs: Any):
    """Execute the selected, capability-checked recurrent training leaf."""

    support = probe_recurrent_training_v1(*args, **kwargs)
    if not support["supported"]:
        raise RuntimeError(str(support["reason"]))
    implementation = str(support["implementation"])
    _record_recurrent_trace(implementation)
    if implementation == _MATRIX_IMPLEMENTATION:
        return _run_matrix(*args, **kwargs)
    if implementation == _FACTORIZED_IMPLEMENTATION:
        return _run_factorized(*args, **kwargs)
    raise RuntimeError(
        "recurrent training probe selected an unknown implementation: "
        f"{implementation}"
    )


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


def linear_training_v1(*args: Any, **kwargs: Any):
    """Execute an explicitly selected stateless flattened projection."""

    support = probe_linear_training_v1(*args, **kwargs)
    if not support["supported"]:
        raise RuntimeError(str(support["reason"]))
    implementation = str(support["implementation"])
    if implementation != _FLATTENED_IMPLEMENTATION:
        raise RuntimeError(
            "linear training probe selected an unknown implementation: "
            f"{implementation}"
        )
    _record_linear_trace(implementation)
    return _run_flattened(*args, **kwargs)


__all__ = [
    "linear_training_v1",
    "probe_linear_training_v1",
    "probe_recurrent_training_v1",
    "recurrent_training_v1",
]
