# coding=utf-8
"""Readable RWKV-7 math with versioned optional operator boundaries.

The reference implementation below is the source of truth.  An independently
installed :mod:`rwkv7_kernels` wheel may replace recurrence or the complete
layer loop; model structure, cache semantics, and Hugging Face APIs stay here.
"""
from __future__ import annotations

from contextvars import ContextVar
import importlib
import os
from types import ModuleType
from typing import Any

import torch


_KERNEL_API_VERSION = 2
_BACKEND_ENV = "RWKV7_BACKEND"
_BACKEND_MODES = ("auto", "reference", "optimized")
_kernel_module: ModuleType | None = None
_kernel_import_attempted = False
_kernel_import_error: str | None = None
_last_recurrent_route: ContextVar[dict[str, str] | None] = ContextVar(
    "rwkv7_last_recurrent_route", default=None
)
_last_model_route: ContextVar[dict[str, str] | None] = ContextVar(
    "rwkv7_last_model_route", default=None
)


def rwkv7_recurrent_reference(
    receptance: torch.Tensor,
    decay: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the RWKV-7 recurrent update in canonical [K,V] layout.

    Args:
        receptance, decay, key, a, b:
            Tensors shaped [batch, time, heads, key_dim].
        value:
            Tensor shaped [batch, time, heads, value_dim].
        initial_state:
            Tensor shaped [batch, heads, key_dim, value_dim].
        attention_mask:
            Optional boolean tensor shaped [batch, time]. A false position
            produces a zero output and leaves that batch row's state unchanged.

    Returns:
        Sequence outputs [batch, time, heads, value_dim] and the final
        recurrent state [batch, heads, key_dim, value_dim].

    This function deliberately contains no dispatch, compilation,
    environment-variable, device, or layout policy.
    """

    if receptance.ndim != 4:
        raise ValueError("RWKV7 recurrent inputs must be shaped [B,T,H,D]")
    if any(tensor.ndim != 4 for tensor in (decay, key, value, a, b)):
        raise ValueError("RWKV7 recurrent inputs must be shaped [B,T,H,D]")
    batch, time, heads, key_dim = receptance.shape
    value_dim = int(value.shape[-1])
    expected_key_shape = (batch, time, heads, key_dim)
    if any(tuple(tensor.shape) != expected_key_shape for tensor in (decay, key, a, b)):
        raise ValueError("receptance, decay, key, a, and b must have identical shapes")
    if tuple(value.shape[:3]) != (batch, time, heads):
        raise ValueError("value must share the [B,T,H] dimensions")
    if tuple(initial_state.shape) != (batch, heads, key_dim, value_dim):
        raise ValueError(
            "initial_state must be shaped [batch, heads, key_dim, value_dim]"
        )
    if attention_mask is not None:
        if tuple(attention_mask.shape) != (batch, time):
            raise ValueError("attention_mask must be shaped [batch, time]")
        attention_mask = attention_mask.to(
            device=initial_state.device, dtype=torch.bool
        )

    # Evaluate samples independently so the batched-matmul shape cannot change
    # FP16 rounding when a framework regroups the same examples. This remains
    # the direct recurrence below, not an alternate kernel or dispatch route.
    def run_sample(batch_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        state = initial_state[batch_idx : batch_idx + 1]
        outputs: list[torch.Tensor] = []
        sample_mask = (
            None
            if attention_mask is None
            else attention_mask[batch_idx : batch_idx + 1]
        )
        for token_idx in range(time):
            # Match the official reference's mixed-precision contract exactly:
            # projections and outer products stay in the model dtype, while
            # the accumulated recurrent state and decay are FP32. Casting every
            # operand to the state dtype would define a different FP16 model.
            r_t = receptance[batch_idx : batch_idx + 1, token_idx]
            w_t = decay[batch_idx : batch_idx + 1, token_idx].to(
                dtype=state.dtype
            )
            k_t = key[batch_idx : batch_idx + 1, token_idx]
            v_t = value[batch_idx : batch_idx + 1, token_idx]
            a_t = a[batch_idx : batch_idx + 1, token_idx]
            b_t = b[batch_idx : batch_idx + 1, token_idx]

            # Evaluate the canonical [K,V] state in the official [V,K]
            # presentation, then transpose it back. Multiplication order is
            # important for long-sequence numerical parity.
            state_vk = state.transpose(-1, -2)
            ab = a_t.unsqueeze(-1) @ b_t.unsqueeze(-2)
            vk = v_t.unsqueeze(-1) @ k_t.unsqueeze(-2)
            candidate_vk = (
                state_vk * w_t.unsqueeze(-2)
                + state_vk @ ab.to(dtype=state.dtype)
                + vk.to(dtype=state.dtype)
            )
            candidate = candidate_vk.transpose(-1, -2)
            output = (
                candidate_vk.to(dtype=r_t.dtype) @ r_t.unsqueeze(-1)
            ).squeeze(-1)

            if sample_mask is not None:
                active = sample_mask[:, token_idx]
                state = torch.where(active.view(1, 1, 1, 1), candidate, state)
                output = torch.where(
                    active.view(1, 1, 1), output, torch.zeros_like(output)
                )
            else:
                state = candidate
            outputs.append(output.to(dtype=value.dtype))

        return torch.stack(outputs, dim=1), state

    samples = [run_sample(batch_idx) for batch_idx in range(batch)]
    return (
        torch.cat([sample[0] for sample in samples], dim=0),
        torch.cat([sample[1] for sample in samples], dim=0),
    )


def _backend_mode(value: str | None) -> str:
    normalized = (
        os.environ.get(_BACKEND_ENV, "auto") if value is None else value
    ).strip().lower()
    if normalized not in _BACKEND_MODES:
        choices = ", ".join(_BACKEND_MODES)
        raise ValueError(f"RWKV7 backend must be one of {choices}; got {value!r}")
    return normalized


def _record_route(
    *, requested: str, selected: str, implementation: str, reason: str
) -> None:
    _last_recurrent_route.set(
        {
            "requested": requested,
            "selected": selected,
            "implementation": implementation,
            "reason": reason,
        }
    )


def get_last_recurrent_route() -> dict[str, str] | None:
    """Return the actual route taken by the most recent call in this context."""

    route = _last_recurrent_route.get()
    return None if route is None else dict(route)


def get_last_model_route() -> dict[str, str] | None:
    """Return the actual whole-model route taken by the latest call."""

    route = _last_model_route.get()
    return None if route is None else dict(route)


def _load_kernel_module() -> ModuleType | None:
    global _kernel_import_attempted, _kernel_import_error, _kernel_module
    if _kernel_import_attempted:
        return _kernel_module
    _kernel_import_attempted = True
    try:
        module = importlib.import_module("rwkv7_kernels")
    except Exception as exc:  # optional companion package
        _kernel_import_error = f"{type(exc).__name__}: {exc}"
        return None
    version = getattr(module, "RWKV7_KERNEL_API_VERSION", None)
    if version != _KERNEL_API_VERSION:
        _kernel_import_error = (
            "kernel API mismatch: "
            f"package={version!r}, adapter={_KERNEL_API_VERSION}"
        )
        return None
    _kernel_module = module
    _kernel_import_error = None
    return module


def _reset_kernel_discovery_for_tests() -> None:
    """Reset optional-package discovery for isolated protocol tests."""

    global _kernel_import_attempted, _kernel_import_error, _kernel_module
    _kernel_module = None
    _kernel_import_attempted = False
    _kernel_import_error = None
    _last_recurrent_route.set(None)
    _last_model_route.set(None)


def _probe_fields(value: Any) -> tuple[bool, str, str]:
    if not isinstance(value, dict):
        raise TypeError("rwkv7_kernels.probe_recurrent_v1() must return a dict")
    missing = {"supported", "implementation", "reason"} - set(value)
    if missing:
        names = ", ".join(sorted(missing))
        raise TypeError(f"kernel probe result is missing: {names}")
    return (
        bool(value["supported"]),
        str(value["implementation"]),
        str(value["reason"]),
    )


def _model_probe_fields(value: Any) -> tuple[bool, str, str, str]:
    supported, implementation, reason = _probe_fields(value)
    if not isinstance(value, dict) or "phase" not in value:
        raise TypeError("model-forward probe result is missing: phase")
    phase = str(value["phase"])
    if phase not in ("prefill", "decode", "training"):
        raise ValueError(
            "model-forward probe phase must be prefill, decode, or training"
        )
    return supported, implementation, reason, phase


def _validate_kernel_result(
    result: Any,
    *,
    value: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("rwkv7_kernels.recurrent_v1() must return two tensors")
    output, final_state = result
    if not isinstance(output, torch.Tensor) or not isinstance(
        final_state, torch.Tensor
    ):
        raise TypeError("rwkv7_kernels.recurrent_v1() must return two tensors")
    if tuple(output.shape) != tuple(value.shape):
        raise ValueError(
            "kernel output shape mismatch: "
            f"expected {tuple(value.shape)}, got {tuple(output.shape)}"
        )
    if tuple(final_state.shape) != tuple(initial_state.shape):
        raise ValueError(
            "kernel state shape mismatch: "
            f"expected {tuple(initial_state.shape)}, got {tuple(final_state.shape)}"
        )
    return output, final_state


def _validate_model_result(result: Any, *, expected_kind: str) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise TypeError("rwkv7_kernels.model_forward_v1() must return a dict")
    kind = result.get("output_kind")
    if kind != expected_kind:
        raise ValueError(
            "kernel model output kind mismatch: "
            f"expected {expected_kind!r}, got {kind!r}"
        )
    required = {"last_hidden_state"} if kind == "base" else {"logits"}
    missing = required - set(result)
    if missing:
        names = ", ".join(sorted(missing))
        raise TypeError(f"kernel model result is missing: {names}")
    tensor_name = "last_hidden_state" if kind == "base" else "logits"
    if not isinstance(result[tensor_name], torch.Tensor):
        raise TypeError(f"kernel model result {tensor_name} must be a tensor")
    return result


def maybe_model_forward(
    owner: Any,
    request: dict[str, Any],
    *,
    backend: str | None = None,
) -> dict[str, Any] | None:
    """Try the optional whole-model protocol and otherwise request fallback.

    Returning ``None`` means that the readable layer loop must run.  ``auto``
    contains missing/unsupported/failing optional implementations, whereas
    ``optimized`` is strict and exposes the exact failure.  The request and
    result are plain mappings so the optional wheel never owns HF output types.
    """

    requested = _backend_mode(backend)
    if not isinstance(request, dict):
        raise TypeError("RWKV7 model-forward request must be a dict")
    model_kind = request.get("model_kind")
    if model_kind not in ("base", "causal_lm"):
        raise ValueError("model_kind must be 'base' or 'causal_lm'")

    hidden = request.get("hidden_states")
    fallback_phase = "training" if bool(request.get("training")) else "prefill"
    if (
        fallback_phase != "training"
        and isinstance(hidden, torch.Tensor)
        and hidden.ndim >= 2
        and int(hidden.shape[1]) == 1
    ):
        fallback_phase = "decode"

    def record(
        selected: str, implementation: str, reason: str, phase: str = fallback_phase
    ) -> None:
        _last_model_route.set(
            {
                "requested": requested,
                "selected": selected,
                "implementation": implementation,
                "reason": reason,
                "phase": phase,
            }
        )

    if requested == "reference":
        record(
            "reference",
            "torch-reference-model-v1",
            "reference backend was explicitly requested",
        )
        return None

    module = _load_kernel_module()
    if module is None:
        reason = _kernel_import_error or "rwkv7_kernels is not installed"
        if requested == "optimized":
            raise RuntimeError(f"optimized RWKV7 backend is unavailable: {reason}")
        record("reference", "torch-reference-model-v1", reason)
        return None

    probe = getattr(module, "probe_model_forward_v1", None)
    run = getattr(module, "model_forward_v1", None)
    if not callable(probe) or not callable(run):
        failure: Exception = RuntimeError(
            "rwkv7_kernels does not implement model-forward protocol v1"
        )
    else:
        try:
            supported, implementation, reason, phase = _model_probe_fields(
                probe(owner, request)
            )
            if not supported:
                if requested == "optimized":
                    raise RuntimeError(
                        "optimized RWKV7 backend does not support this request: "
                        f"{reason}"
                    )
                record("reference", "torch-reference-model-v1", reason, phase)
                return None
            result = _validate_model_result(
                run(owner, request), expected_kind=str(model_kind)
            )
        except Exception as exc:
            if requested == "optimized":
                raise RuntimeError(
                    "optimized RWKV7 model-forward-v1 execution failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            failure = exc
        else:
            record("optimized", implementation, reason, phase)
            return result

    reason = f"optional kernel failure: {type(failure).__name__}: {failure}"
    if requested == "optimized":
        raise RuntimeError(reason) from failure
    record("reference", "torch-reference-model-v1", reason)
    return None


def rwkv7_recurrent(
    receptance: torch.Tensor,
    decay: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    backend: str | None = None,
    training: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run recurrent semantics through reference or optional protocol v1.

    ``auto`` is failure-contained and always retains the readable reference
    path. ``optimized`` is a validation mode: missing, unsupported, malformed,
    or failing kernels raise instead of silently changing the claimed route.
    The optional v1 protocol is inference-only, so training always uses the
    differentiable reference implementation.
    """

    requested = _backend_mode(backend)
    if requested == "reference":
        reason = "reference backend was explicitly requested"
        _record_route(
            requested=requested,
            selected="reference",
            implementation="torch-reference-v1",
            reason=reason,
        )
        return rwkv7_recurrent_reference(
            receptance,
            decay,
            key,
            value,
            a,
            b,
            initial_state,
            attention_mask,
        )

    if training:
        reason = "the recurrent-v1 optional kernel protocol is inference-only"
        if requested == "optimized":
            raise RuntimeError(
                f"optimized RWKV7 backend does not support this request: {reason}"
            )
        _record_route(
            requested=requested,
            selected="reference",
            implementation="torch-reference-v1",
            reason=reason,
        )
        return rwkv7_recurrent_reference(
            receptance,
            decay,
            key,
            value,
            a,
            b,
            initial_state,
            attention_mask,
        )

    module = _load_kernel_module()
    if module is None:
        reason = _kernel_import_error or "rwkv7_kernels is not installed"
        if requested == "optimized":
            raise RuntimeError(f"optimized RWKV7 backend is unavailable: {reason}")
        _record_route(
            requested=requested,
            selected="reference",
            implementation="torch-reference-v1",
            reason=reason,
        )
        return rwkv7_recurrent_reference(
            receptance,
            decay,
            key,
            value,
            a,
            b,
            initial_state,
            attention_mask,
        )

    probe = getattr(module, "probe_recurrent_v1", None)
    run = getattr(module, "recurrent_v1", None)
    if not callable(probe) or not callable(run):
        failure: Exception = RuntimeError(
            "rwkv7_kernels does not implement recurrent protocol v1"
        )
    else:
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
        try:
            supported, implementation, reason = _probe_fields(probe(*args))
            if not supported:
                if requested == "optimized":
                    raise RuntimeError(
                        "optimized RWKV7 backend does not support this request: "
                        f"{reason}"
                    )
                _record_route(
                    requested=requested,
                    selected="reference",
                    implementation="torch-reference-v1",
                    reason=reason,
                )
                return rwkv7_recurrent_reference(*args)
            result = _validate_kernel_result(
                run(*args), value=value, initial_state=initial_state
            )
        except Exception as exc:
            if requested == "optimized":
                raise RuntimeError(
                    "optimized RWKV7 recurrent-v1 execution failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            failure = exc
        else:
            _record_route(
                requested=requested,
                selected="optimized",
                implementation=implementation,
                reason=reason,
            )
            return result

    reason = f"optional kernel failure: {type(failure).__name__}: {failure}"
    if requested == "optimized":
        raise RuntimeError(reason) from failure
    _record_route(
        requested=requested,
        selected="reference",
        implementation="torch-reference-v1",
        reason=reason,
    )
    return rwkv7_recurrent_reference(
        receptance,
        decay,
        key,
        value,
        a,
        b,
        initial_state,
        attention_mask,
    )


__all__ = [
    "get_last_model_route",
    "get_last_recurrent_route",
    "maybe_model_forward",
    "rwkv7_recurrent",
    "rwkv7_recurrent_reference",
]
