"""Fail-closed selection for optional RWKV-7 training leaf protocols."""

from __future__ import annotations

import os
from typing import Any

import torch

from ._runtime_preflight import _certify_recurrent_runtime
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
    load_mix6_training_cuda_extension,
    probe_mix6_training_v1 as _probe_mix6,
)


_TRAINING_IMPL_ENV = "RWKV7_TRAINING_KERNEL_IMPL"
_TRAINING_IMPLS = ("auto", "adaptive", "matrix", "factorized")
_MATRIX_IMPLEMENTATION = "torch-cuda-rwkv7-batched-matrix-recurrent-training-v1"
_FACTORIZED_IMPLEMENTATION = "native-nvidia-rwkv7-factorized-recurrent-training-v1"
_FLATTENED_IMPLEMENTATION = "torch-cuda-rwkv7-flattened-linear-training-v1"
_MIX6_IMPLEMENTATION = "native-nvidia-rwkv7-mix6-training-v1"
_PROGRAM_IMPLEMENTATION = "native-nvidia-rwkv7-adaptive-training-program-v1"
_REFERENCE_PROGRAM_IMPLEMENTATION = "torch-reference-training-program-v1"
_RECURRENT_HINT_NAMES = frozenset(
    (
        "adaptive_fast_program",
        "fully_active",
        "initial_state_zero",
        "token_aligned",
        "force_reference_recurrent",
    )
)
_ADAPTIVE_FAST_DOMAIN_BATCH = 4
_ADAPTIVE_FAST_DOMAIN_TOKENS = 128


def adaptive_training_fast_domain_v1(
    *,
    batch: int,
    tokens: int,
    fully_active: bool,
    initial_state_zero: bool,
    token_aligned: bool,
) -> bool:
    """Return the currently certified dense adaptive training envelope.

    This policy is shared by recurrent and flattened-linear dispatch so a
    model forward cannot combine an exact recurrent fallback with a different
    projection accumulation program.  It is intentionally limited to the
    one large dense shape that has passed the strict multi-lane model gate;
    explicit ``factorized`` remains available for isolated experimentation.
    """

    return bool(
        fully_active
        and initial_state_zero
        and token_aligned
        and batch == _ADAPTIVE_FAST_DOMAIN_BATCH
        and tokens == _ADAPTIVE_FAST_DOMAIN_TOKENS
    )


def probe_training_program_v1(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    training: bool,
    fully_active: bool,
    initial_state_zero: bool | None,
    token_aligned: bool,
    autograd_leaf_eligible: bool,
    head_dim: int | None,
) -> dict[str, Any]:
    """Preflight the dense B4/T128 adaptive training program.

    Model-owned shape, mask, cache, and autograd facts define the only fast
    domain.  The linear leaf is ordinary cuBLAS with no lazy dependency; this
    preflight loads both native CUDA extensions before a certificate is
    issued.  Every later leaf still validates its concrete tensors, and an
    unexpected decline fails the certified model call closed.
    """

    if _requested_implementation() != "adaptive":
        return {
            "supported": False,
            "implementation": _REFERENCE_PROGRAM_IMPLEMENTATION,
            "reason": "the coupled program is selected only by adaptive training",
        }
    if not all(
        isinstance(value, bool)
        for value in (
            training,
            fully_active,
            token_aligned,
            autograd_leaf_eligible,
        )
    ):
        return {
            "supported": False,
            "implementation": _PROGRAM_IMPLEMENTATION,
            "reason": (
                "training, fully_active, token_aligned, and "
                "autograd_leaf_eligible must be booleans"
            ),
        }
    if initial_state_zero is not None and not isinstance(initial_state_zero, bool):
        return {
            "supported": False,
            "implementation": _PROGRAM_IMPLEMENTATION,
            "reason": "initial_state_zero must be a bool or None",
        }
    if not training or not torch.is_grad_enabled():
        return {
            "supported": False,
            "implementation": _PROGRAM_IMPLEMENTATION,
            "reason": "the coupled program requires enabled autograd training",
        }
    if not autograd_leaf_eligible:
        return {
            "supported": False,
            "implementation": _PROGRAM_IMPLEMENTATION,
            "reason": (
                "the coupled program requires gradient-bearing inputs and a "
                "non-reentrant checkpoint forward"
            ),
        }
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
        return {
            "supported": False,
            "implementation": _PROGRAM_IMPLEMENTATION,
            "reason": "hidden_states must be a [B,T,C] tensor",
        }
    batch, tokens, _channels = tuple(hidden_states.shape)
    if not adaptive_training_fast_domain_v1(
        batch=batch,
        tokens=tokens,
        fully_active=fully_active,
        initial_state_zero=initial_state_zero is True,
        token_aligned=token_aligned,
    ):
        return {
            "supported": False,
            "implementation": _PROGRAM_IMPLEMENTATION,
            "reason": "request is outside the certified adaptive fast domain",
        }
    if head_dim != 64:
        return {
            "supported": False,
            "implementation": _PROGRAM_IMPLEMENTATION,
            "reason": "the factorized recurrent program requires head_dim=64",
        }
    if not isinstance(attention_mask, torch.Tensor) or tuple(attention_mask.shape) != (
        batch,
        tokens,
    ):
        return {
            "supported": False,
            "implementation": _PROGRAM_IMPLEMENTATION,
            "reason": "attention_mask must be a [B,T] tensor",
        }
    if (
        not torch.cuda.is_available()
        or not hidden_states.is_cuda
        or hidden_states.dtype != torch.bfloat16
    ):
        return {
            "supported": False,
            "implementation": _PROGRAM_IMPLEMENTATION,
            "reason": "the coupled program requires BF16 CUDA hidden states",
        }
    if not attention_mask.is_cuda or attention_mask.device != hidden_states.device:
        return {
            "supported": False,
            "implementation": _PROGRAM_IMPLEMENTATION,
            "reason": "attention_mask must share the hidden-state CUDA device",
        }
    if torch.cuda.get_device_capability(hidden_states.device) < (8, 0):
        return {
            "supported": False,
            "implementation": _PROGRAM_IMPLEMENTATION,
            "reason": "the BF16 training program requires sm80 or newer",
        }

    from .nvidia.official_training_cuda import recurrent_training_cuda_available

    if not recurrent_training_cuda_available(build=True):
        return {
            "supported": False,
            "implementation": _PROGRAM_IMPLEMENTATION,
            "reason": "the native recurrent extension is not loaded",
        }
    try:
        load_mix6_training_cuda_extension(device=hidden_states.device)
    except Exception as exc:
        return {
            "supported": False,
            "implementation": _PROGRAM_IMPLEMENTATION,
            "reason": f"the native Mix6 extension is not loaded: {exc}",
        }

    # Recurrent leaf probes may skip runtime/toolchain discovery only after
    # both native dependencies above have completed successfully.
    _certify_recurrent_runtime(hidden_states.device)
    return {
        "supported": True,
        "implementation": _PROGRAM_IMPLEMENTATION,
        "reason": (
            "dense B4/T128 recurrent, flattened-linear, and Mix6 program "
            "passed atomic runtime preflight"
        ),
    }


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


def _linear_leaf_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Remove model-owned coupled-program hints from leaf calls."""

    return {
        name: value
        for name, value in kwargs.items()
        if name not in {"adaptive_fast_program", "initial_state_zero"}
    }


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
    adaptive_fast_program = kwargs.get("adaptive_fast_program")
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
    shape = _recurrent_shape(args, kwargs)
    fast_domain = bool(
        initial_state_zero
        and adaptive_fast_program is not False
        and shape is not None
        and len(shape) == 4
        and adaptive_training_fast_domain_v1(
            batch=shape[0],
            tokens=shape[1],
            fully_active=fully_active,
            initial_state_zero=initial_state_zero,
            token_aligned=token_aligned,
        )
    )
    if fast_domain:
        factorized = _validated_recurrent_probe(
            _probe_factorized,
            *args,
            **kwargs,
        )
        if factorized["supported"]:
            return factorized
        if adaptive_fast_program is True:
            factorized = dict(factorized)
            factorized["reason"] = (
                "the preflight-certified adaptive program declined during "
                f"atomic recurrent execution: {factorized['reason']}"
            )
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
        elif not token_aligned:
            request_kind = (
                "an unaligned recurrent request; the factorized leaf requires "
                f"token lengths divisible by {TOKEN_CHUNK_LENGTH}"
            )
        elif adaptive_fast_program is False:
            request_kind = "a request whose coupled fast-program preflight declined"
        else:
            request_kind = (
                "a dense request outside the certified adaptive fast domain "
                f"B={_ADAPTIVE_FAST_DOMAIN_BATCH}, "
                f"T={_ADAPTIVE_FAST_DOMAIN_TOKENS}"
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
    if kwargs.get("force_reference_recurrent") is True:
        return {
            "supported": False,
            "implementation": _MATRIX_IMPLEMENTATION,
            "reason": (
                "the model pinned checkpoint forward and replay to reference recurrence"
            ),
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
    adaptive_fast_program: bool | None = None,
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
        "adaptive_fast_program": adaptive_fast_program,
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
    for name in (
        "adaptive_fast_program",
        "fully_active",
        "initial_state_zero",
        "token_aligned",
    ):
        hint = kwargs.get(name)
        if hint is not None and not isinstance(hint, bool):
            return {
                "supported": False,
                "implementation": _FLATTENED_IMPLEMENTATION,
                "reason": f"{name} must be a bool or None",
            }
    fully_active = kwargs.get("fully_active")
    adaptive_fast_program = kwargs.get("adaptive_fast_program")
    initial_state_zero = kwargs.get("initial_state_zero")
    token_aligned = kwargs.get("token_aligned")
    value = kwargs.get("value", args[0] if args else None)
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
    if requested == "adaptive":
        shape = tuple(value.shape) if isinstance(value, torch.Tensor) else ()
        fast_domain = bool(
            len(shape) == 3
            and adaptive_fast_program is True
            and adaptive_training_fast_domain_v1(
                batch=shape[0],
                tokens=shape[1],
                fully_active=fully_active is True,
                initial_state_zero=initial_state_zero is True,
                token_aligned=token_aligned is True,
            )
        )
        if fast_domain:
            return validate_support_result(
                _probe_flattened(*args, **_linear_leaf_kwargs(kwargs)),
                probe_name="probe_linear_training_v1",
            )
        if adaptive_fast_program is not True:
            request_kind = "request without a coupled fast-program certificate"
        elif initial_state_zero is not True:
            request_kind = "stateful or standalone"
        elif fully_active is not True:
            request_kind = "masked or standalone"
        elif token_aligned is not True:
            request_kind = "token-length-unaligned"
        else:
            request_kind = (
                "dense request outside the certified adaptive fast domain "
                f"B={_ADAPTIVE_FAST_DOMAIN_BATCH}, "
                f"T={_ADAPTIVE_FAST_DOMAIN_TOKENS}"
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
        _probe_flattened(*args, **_linear_leaf_kwargs(kwargs)),
        probe_name="probe_linear_training_v1",
    )


def execute_linear_training_v1(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    adaptive_fast_program: bool | None = None,
    fully_active: bool | None = None,
    initial_state_zero: bool | None = None,
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
        adaptive_fast_program=adaptive_fast_program,
        fully_active=fully_active,
        initial_state_zero=initial_state_zero,
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
    "adaptive_training_fast_domain_v1",
    "execute_linear_training_v1",
    "execute_mix6_training_v1",
    "execute_recurrent_training_v1",
    "linear_training_v1",
    "mix6_training_v1",
    "probe_linear_training_v1",
    "probe_mix6_training_v1",
    "probe_recurrent_training_v1",
    "probe_training_program_v1",
    "recurrent_training_v1",
]
