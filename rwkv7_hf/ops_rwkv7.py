# coding=utf-8
"""Readable RWKV-7 math with versioned optional operator boundaries.

The reference implementation below is the source of truth.  An independently
installed :mod:`rwkv7_kernels` wheel may replace stateless training linears,
recurrence, or the complete layer loop; model structure, cache semantics, and
Hugging Face APIs stay here.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import importlib
import os
from types import ModuleType
from typing import Any, Iterator

import torch


_KERNEL_API_VERSION = 3
_BACKEND_ENV = "RWKV7_BACKEND"
_BACKEND_MODES = ("auto", "reference", "optimized")
# The optional training protocol defines a 16-token chunk. Keeping the value
# here avoids importing the optional package into the clean HF implementation;
# it describes request semantics only and never selects a device or kernel.
_TRAINING_TOKEN_CHUNK_LENGTH = 16
_kernel_module: ModuleType | None = None
_kernel_import_attempted = False
_kernel_import_error: str | None = None
_last_recurrent_route: ContextVar[dict[str, str] | None] = ContextVar(
    "rwkv7_last_recurrent_route", default=None
)
_last_model_route: ContextVar[dict[str, str] | None] = ContextVar(
    "rwkv7_last_model_route", default=None
)
_last_linear_route: ContextVar[dict[str, str] | None] = ContextVar(
    "rwkv7_last_linear_route", default=None
)
_last_mix6_route: ContextVar[dict[str, str] | None] = ContextVar(
    "rwkv7_last_mix6_route", default=None
)
_training_batch_fully_active: ContextVar[bool | None] = ContextVar(
    "rwkv7_training_batch_fully_active", default=None
)
_training_batch_token_aligned: ContextVar[bool | None] = ContextVar(
    "rwkv7_training_batch_token_aligned", default=None
)
_training_batch_initial_state_zero: ContextVar[bool | None] = ContextVar(
    "rwkv7_training_batch_initial_state_zero", default=None
)
_training_batch_adaptive_fast_program: ContextVar[bool | None] = ContextVar(
    "rwkv7_training_batch_adaptive_fast_program", default=None
)
_training_batch_force_reference_recurrent: ContextVar[bool] = ContextVar(
    "rwkv7_training_batch_force_reference_recurrent", default=False
)


@dataclass(frozen=True)
class RWKV7TrainingBatchContext:
    """Immutable model-owned facts shared by independent training leaves."""

    fully_active: bool
    token_aligned: bool | None
    initial_state_zero: bool | None
    autograd_leaf_eligible: bool | None
    force_reference_recurrent: bool
    adaptive_fast_program: bool | None
    program_implementation: str
    program_reason: str


_last_training_batch_context: ContextVar[RWKV7TrainingBatchContext | None] = ContextVar(
    "rwkv7_last_training_batch_context", default=None
)


def _is_checkpoint_control_flow(exc: Exception) -> bool:
    """Return whether *exc* belongs to PyTorch checkpoint control flow.

    PyTorch stops a non-reentrant checkpoint replay by raising the private
    ``_StopRecomputationError`` after it has recreated every saved tensor.
    Optional backend boundaries normally contain arbitrary implementation
    failures, but treating that signal as a kernel failure makes execution
    continue into the reference fallback. The checkpoint pack hook then sees
    an extra saved tensor and raises ``target_frame.early_stop is set``.

    Detect checkpoint-owned exceptions without importing private PyTorch
    symbols so this package remains importable across supported Torch
    releases. User-visible ``CheckpointError`` instances must escape for the
    same reason: they describe replay correctness, not an optional backend
    failure.
    """

    return type(exc).__module__.startswith("torch.utils.checkpoint")


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
            w_t = decay[batch_idx : batch_idx + 1, token_idx].to(dtype=state.dtype)
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
            output = (candidate_vk.to(dtype=r_t.dtype) @ r_t.unsqueeze(-1)).squeeze(-1)

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
        (os.environ.get(_BACKEND_ENV, "auto") if value is None else value)
        .strip()
        .lower()
    )
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


def get_last_linear_route() -> dict[str, str] | None:
    """Return the route taken by the latest training linear projection."""

    route = _last_linear_route.get()
    return None if route is None else dict(route)


def get_last_mix6_route() -> dict[str, str] | None:
    """Return the route taken by the latest six-way token-mix leaf."""

    route = _last_mix6_route.get()
    return None if route is None else dict(route)


def get_last_training_batch_context() -> RWKV7TrainingBatchContext | None:
    """Return the latest resolved context for the enclosing causal-LM head."""

    return _last_training_batch_context.get()


def get_last_training_program_route() -> dict[str, Any] | None:
    """Return the coupled adaptive-program decision and its model-owned facts."""

    context = _last_training_batch_context.get()
    if context is None:
        return None
    if context.adaptive_fast_program is None:
        selected = "not-applicable"
    else:
        selected = "optimized" if context.adaptive_fast_program else "reference"
    return {
        "selected": selected,
        "implementation": context.program_implementation,
        "reason": context.program_reason,
        "facts": {
            "fully_active": context.fully_active,
            "token_aligned": context.token_aligned,
            "initial_state_zero": context.initial_state_zero,
            "autograd_leaf_eligible": context.autograd_leaf_eligible,
            "force_reference_recurrent": context.force_reference_recurrent,
        },
    }


def resolve_training_batch_context(
    attention_mask: torch.Tensor,
    *,
    training: bool,
    fully_active: bool | None = None,
    initial_state_zero: bool | None = None,
    autograd_leaf_eligible: bool | None = None,
    force_reference_recurrent: bool = False,
    hidden_states: torch.Tensor | None = None,
    head_dim: int | None = None,
) -> RWKV7TrainingBatchContext:
    """Resolve one immutable training program before any layer leaf executes.

    The clean model owns padding and cache semantics.  The returned value says
    only whether
    every position is active, whether its sequence length matches the optional
    protocol's 16-token chunk, and whether the model created a fresh empty
    cache for this call.  The optional package may additionally certify the
    coupled adaptive recurrent/linear program before the first projection.
    No model, parameter, cache, or optimizer object crosses that probe.
    """

    if fully_active is None:
        fully_active = bool(attention_mask.to(dtype=torch.bool).all().detach().cpu())
    elif not isinstance(fully_active, bool):
        raise TypeError("fully_active must be a bool or None")
    if initial_state_zero is not None and not isinstance(initial_state_zero, bool):
        raise TypeError("initial_state_zero must be a bool or None")
    if not isinstance(force_reference_recurrent, bool):
        raise TypeError("force_reference_recurrent must be a bool")

    token_aligned = None
    adaptive_fast_program = None
    program_implementation = "torch-reference-training-program-v1"
    program_reason = "model is not executing a training program"
    if training:
        if autograd_leaf_eligible is None:
            autograd_leaf_eligible = bool(
                torch.is_grad_enabled()
                and isinstance(hidden_states, torch.Tensor)
                and hidden_states.requires_grad
            )
        elif not isinstance(autograd_leaf_eligible, bool):
            raise TypeError("autograd_leaf_eligible must be a bool or None")
        token_aligned = bool(
            attention_mask.ndim == 2
            and int(attention_mask.shape[1]) % _TRAINING_TOKEN_CHUNK_LENGTH == 0
        )
        adaptive_fast_program = False
        backend_mode = _backend_mode(None)
        module = _load_kernel_module() if backend_mode != "reference" else None
        if backend_mode == "reference":
            program_reason = "reference backend was explicitly requested"
        elif module is None:
            program_reason = _kernel_import_error or "rwkv7_kernels is not installed"
        probe = (
            getattr(module, "probe_training_program_v1", None)
            if module is not None
            else None
        )
        if callable(probe) and isinstance(hidden_states, torch.Tensor):
            try:
                supported, implementation, reason = _probe_fields(
                    probe(
                        hidden_states,
                        attention_mask,
                        training=True,
                        fully_active=fully_active,
                        initial_state_zero=initial_state_zero,
                        token_aligned=token_aligned,
                        autograd_leaf_eligible=autograd_leaf_eligible,
                        head_dim=head_dim,
                    ),
                    probe_name="probe_training_program_v1",
                )
            except Exception as exc:
                if _is_checkpoint_control_flow(exc):
                    raise
                program_reason = (
                    "coupled adaptive-program probe failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            else:
                adaptive_fast_program = supported
                program_implementation = implementation
                program_reason = reason
        elif module is not None and not callable(probe):
            program_reason = (
                "rwkv7_kernels does not implement probe_training_program_v1"
            )
        elif module is not None:
            program_reason = "hidden_states is unavailable for coupled preflight"
    context = RWKV7TrainingBatchContext(
        fully_active=fully_active,
        token_aligned=token_aligned,
        initial_state_zero=(initial_state_zero if training else None),
        autograd_leaf_eligible=(autograd_leaf_eligible if training else None),
        force_reference_recurrent=bool(training and force_reference_recurrent),
        adaptive_fast_program=adaptive_fast_program,
        program_implementation=program_implementation,
        program_reason=program_reason,
    )
    _last_training_batch_context.set(context)
    return context


def _publish_training_batch_context(context: RWKV7TrainingBatchContext):
    if not isinstance(context, RWKV7TrainingBatchContext):
        raise TypeError("context must be an RWKV7TrainingBatchContext")
    return (
        _training_batch_fully_active.set(context.fully_active),
        _training_batch_token_aligned.set(context.token_aligned),
        _training_batch_initial_state_zero.set(context.initial_state_zero),
        _training_batch_adaptive_fast_program.set(context.adaptive_fast_program),
        _training_batch_force_reference_recurrent.set(
            context.force_reference_recurrent
        ),
    )


@contextmanager
def training_batch_context(
    context: RWKV7TrainingBatchContext,
) -> Iterator[RWKV7TrainingBatchContext]:
    """Publish one batch context for a layer and restore the caller exactly."""

    tokens = _publish_training_batch_context(context)
    try:
        yield context
    finally:
        _training_batch_force_reference_recurrent.reset(tokens[4])
        _training_batch_adaptive_fast_program.reset(tokens[3])
        _training_batch_initial_state_zero.reset(tokens[2])
        _training_batch_token_aligned.reset(tokens[1])
        _training_batch_fully_active.reset(tokens[0])


def set_training_batch_context(
    attention_mask: torch.Tensor,
    *,
    training: bool,
    fully_active: bool | None = None,
    initial_state_zero: bool | None = None,
    autograd_leaf_eligible: bool | None = None,
    force_reference_recurrent: bool = False,
    hidden_states: torch.Tensor | None = None,
    head_dim: int | None = None,
) -> bool:
    """Publish a context for standalone protocol tests and return mask status.

    Model execution uses :func:`training_batch_context`, whose lexical scope
    restores every ContextVar.  This setter is intentionally low level for
    direct leaf callers that own the surrounding context lifetime.
    """

    context = resolve_training_batch_context(
        attention_mask,
        training=training,
        fully_active=fully_active,
        initial_state_zero=initial_state_zero,
        autograd_leaf_eligible=autograd_leaf_eligible,
        force_reference_recurrent=force_reference_recurrent,
        hidden_states=hidden_states,
        head_dim=head_dim,
    )
    _publish_training_batch_context(context)
    return context.fully_active


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
            f"kernel API mismatch: package={version!r}, adapter={_KERNEL_API_VERSION}"
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
    _last_linear_route.set(None)
    _last_mix6_route.set(None)
    _last_training_batch_context.set(None)
    _training_batch_fully_active.set(None)
    _training_batch_token_aligned.set(None)
    _training_batch_initial_state_zero.set(None)
    _training_batch_adaptive_fast_program.set(None)
    _training_batch_force_reference_recurrent.set(False)


def _probe_fields(
    value: Any, *, probe_name: str = "probe_recurrent_v1"
) -> tuple[bool, str, str]:
    if not isinstance(value, dict):
        raise TypeError(f"rwkv7_kernels.{probe_name}() must return a dict")
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
    supported, implementation, reason = _probe_fields(
        value, probe_name="probe_model_forward_v1"
    )
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
    fallback_phase = (
        "training"
        if bool(request.get("training")) or bool(request.get("grad_enabled"))
        else "prefill"
    )
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

    if fallback_phase == "training":
        # Training always keeps the readable HF block/layer/loss program. The
        # strict optional recurrent, linear, and Mix6 tensor leaves are probed
        # inside that loop. Returning before package discovery also prevents a
        # whole-model probe, model traversal, and device-to-host mask sync on
        # every ordinary Trainer/TRL step.
        record(
            "reference",
            "torch-reference-model-v1",
            "readable HF training loop owns structure; optional tensor leaves "
            "dispatch independently",
            "training",
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
            actual_implementation = str(result.get("implementation", implementation))
            actual_phase = str(result.get("phase", phase))
            if actual_phase not in ("prefill", "decode", "training"):
                raise ValueError(
                    "kernel model result phase must be prefill, decode, or training"
                )
        except Exception as exc:
            if _is_checkpoint_control_flow(exc):
                raise
            if requested == "optimized":
                raise RuntimeError(
                    "optimized RWKV7 model-forward-v1 execution failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            failure = exc
        else:
            record(
                "optimized",
                actual_implementation,
                reason,
                actual_phase,
            )
            return result

    reason = f"optional kernel failure: {type(failure).__name__}: {failure}"
    if requested == "optimized":
        raise RuntimeError(reason) from failure
    record("reference", "torch-reference-model-v1", reason)
    return None


def maybe_linear_training(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    training: bool,
    backend: str | None = None,
) -> torch.Tensor | None:
    """Return an optional CUDA training projection or request reference math.

    ``None`` means that :class:`RWKV7Linear` must retain its fixed-row
    reference GEMM.  The boundary owns no module or parameter state, so PEFT
    wrappers and ordinary Hugging Face autograd remain outside the kernel
    package.
    """

    if not training:
        return None
    requested = _backend_mode(backend)
    atomic_fast_program = _training_batch_adaptive_fast_program.get() is True

    def record(selected: str, implementation: str, reason: str) -> None:
        _last_linear_route.set(
            {
                "requested": requested,
                "selected": selected,
                "implementation": implementation,
                "reason": reason,
            }
        )

    if requested == "reference":
        record(
            "reference",
            "torch-reference-linear-v1",
            "reference backend was explicitly requested",
        )
        return None

    module = _load_kernel_module()
    if module is None:
        reason = _kernel_import_error or "rwkv7_kernels is not installed"
        if requested == "optimized" or atomic_fast_program:
            raise RuntimeError(f"optimized RWKV7 backend is unavailable: {reason}")
        record("reference", "torch-reference-linear-v1", reason)
        return None

    execute = getattr(module, "execute_linear_training_v1", None)
    probe = getattr(module, "probe_linear_training_v1", None)
    run = getattr(module, "linear_training_v1", None)
    if not callable(execute) and (not callable(probe) or not callable(run)):
        failure: Exception = RuntimeError(
            "rwkv7_kernels does not implement linear-training-v1"
        )
    else:
        try:
            hints = {
                "adaptive_fast_program": (_training_batch_adaptive_fast_program.get()),
                "fully_active": _training_batch_fully_active.get(),
                "initial_state_zero": _training_batch_initial_state_zero.get(),
                "token_aligned": _training_batch_token_aligned.get(),
            }
            if callable(execute):
                execution = execute(
                    value,
                    weight,
                    bias,
                    **hints,
                )
                supported, implementation, reason = _probe_fields(
                    execution,
                    probe_name="execute_linear_training_v1",
                )
                if not isinstance(execution, dict) or "output" not in execution:
                    raise TypeError(
                        "execute_linear_training_v1() result is missing: output"
                    )
                result = execution["output"]
            else:
                supported, implementation, reason = _probe_fields(
                    probe(value, weight, bias, **hints),
                    probe_name="probe_linear_training_v1",
                )
                result = None
            if not supported:
                if requested == "optimized" or atomic_fast_program:
                    raise RuntimeError(
                        "atomic adaptive RWKV7 training program does not support "
                        "this projection request: "
                        f"{reason}"
                    )
                record("reference", "torch-reference-linear-v1", reason)
                return None
            if not callable(execute):
                result = run(value, weight, bias, **hints)
            if not isinstance(result, torch.Tensor):
                function_name = (
                    "execute_linear_training_v1() output"
                    if callable(execute)
                    else "linear_training_v1()"
                )
                raise TypeError(f"{function_name} must be a tensor")
            expected = (*value.shape[:-1], int(weight.shape[0]))
            if tuple(result.shape) != expected:
                raise ValueError(
                    "training linear output shape mismatch: "
                    f"expected {expected}, got {tuple(result.shape)}"
                )
        except Exception as exc:
            if _is_checkpoint_control_flow(exc):
                raise
            if requested == "optimized" or atomic_fast_program:
                raise RuntimeError(
                    "atomic adaptive RWKV7 linear-training-v1 execution failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            failure = exc
        else:
            record("optimized", implementation, reason)
            return result

    reason = f"optional kernel failure: {type(failure).__name__}: {failure}"
    if requested == "optimized" or atomic_fast_program:
        raise RuntimeError(reason) from failure
    record("reference", "torch-reference-linear-v1", reason)
    return None


def maybe_mix6_training(
    value: torch.Tensor,
    shifted: torch.Tensor,
    mixes: tuple[torch.Tensor, ...],
    *,
    training: bool,
    backend: str | None = None,
) -> tuple[torch.Tensor, ...] | None:
    """Try the stateless six-way token-shift leaf used during training.

    The readable model continues to own the shift state and the six parameter
    tensors.  The optional package receives only tensors and returns six mixed
    tensors; unsupported shapes, masks, adapters, devices, or missing wheels
    fall back to the explicit PyTorch equations in :class:`RWKV7TimeMix`.
    """

    if not training:
        return None
    if len(mixes) != 6:
        raise ValueError("RWKV7 Mix6 requires exactly six parameter tensors")
    requested = _backend_mode(backend)
    atomic_fast_program = _training_batch_adaptive_fast_program.get() is True

    def record(selected: str, implementation: str, reason: str) -> None:
        _last_mix6_route.set(
            {
                "requested": requested,
                "selected": selected,
                "implementation": implementation,
                "reason": reason,
            }
        )

    if requested == "reference":
        record(
            "reference",
            "torch-reference-mix6-v1",
            "reference backend was explicitly requested",
        )
        return None

    module = _load_kernel_module()
    if module is None:
        reason = _kernel_import_error or "rwkv7_kernels is not installed"
        if requested == "optimized" or atomic_fast_program:
            raise RuntimeError(f"optimized RWKV7 backend is unavailable: {reason}")
        record("reference", "torch-reference-mix6-v1", reason)
        return None

    execute = getattr(module, "execute_mix6_training_v1", None)
    probe = getattr(module, "probe_mix6_training_v1", None)
    run = getattr(module, "mix6_training_v1", None)
    if not callable(execute) and (not callable(probe) or not callable(run)):
        failure: Exception = RuntimeError(
            "rwkv7_kernels does not implement mix6-training-v1"
        )
    else:
        try:
            hints = {
                "fully_active": _training_batch_fully_active.get(),
                "token_aligned": _training_batch_token_aligned.get(),
            }
            if callable(execute):
                execution = execute(
                    value,
                    shifted,
                    *mixes,
                    **hints,
                )
                supported, implementation, reason = _probe_fields(
                    execution,
                    probe_name="execute_mix6_training_v1",
                )
                if not isinstance(execution, dict) or "result" not in execution:
                    raise TypeError(
                        "execute_mix6_training_v1() result is missing: result"
                    )
                result = execution["result"]
            else:
                supported, implementation, reason = _probe_fields(
                    probe(value, shifted, *mixes, **hints),
                    probe_name="probe_mix6_training_v1",
                )
                result = None
            if not supported:
                if requested == "optimized" or atomic_fast_program:
                    raise RuntimeError(
                        "atomic adaptive RWKV7 training program does not support "
                        "this Mix6 request: "
                        f"{reason}"
                    )
                record("reference", "torch-reference-mix6-v1", reason)
                return None
            if not callable(execute):
                result = run(value, shifted, *mixes, **hints)
            if not isinstance(result, tuple) or len(result) != 6:
                raise TypeError("mix6_training_v1() must return six tensors")
            if any(
                not isinstance(item, torch.Tensor)
                or tuple(item.shape) != tuple(value.shape)
                or item.dtype != value.dtype
                or item.device != value.device
                for item in result
            ):
                raise ValueError(
                    "mix6_training_v1() outputs must match input shape, dtype, and device"
                )
        except Exception as exc:
            if _is_checkpoint_control_flow(exc):
                raise
            if requested == "optimized" or atomic_fast_program:
                raise RuntimeError(
                    "atomic adaptive RWKV7 mix6-training-v1 execution failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            failure = exc
        else:
            record("optimized", implementation, reason)
            return result

    reason = f"optional kernel failure: {type(failure).__name__}: {failure}"
    if requested == "optimized" or atomic_fast_program:
        raise RuntimeError(reason) from failure
    record("reference", "torch-reference-mix6-v1", reason)
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
    initial_state_zero: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run recurrent semantics through reference or optional protocol v1.

    ``auto`` is failure-contained and always retains the readable reference
    path. ``optimized`` is a validation mode: missing, unsupported, malformed,
    or failing kernels raise instead of silently changing the claimed route.
    Training uses a separate leaf-autograd protocol. Production ``auto`` may
    keep it on the readable path while an explicitly selected candidate is
    compared against the same canonical recurrent boundary.
    """

    if initial_state_zero is not None and not isinstance(initial_state_zero, bool):
        raise TypeError("initial_state_zero must be a bool or None")

    requested = _backend_mode(backend)
    atomic_fast_program = bool(
        training and _training_batch_adaptive_fast_program.get() is True
    )
    if training and _training_batch_force_reference_recurrent.get():
        reason = (
            "reentrant checkpoint forward and replay are pinned to the readable "
            "recurrent program"
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

    module = _load_kernel_module()
    if module is None:
        reason = _kernel_import_error or "rwkv7_kernels is not installed"
        if requested == "optimized" or atomic_fast_program:
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

    protocol = "recurrent-training-v1" if training else "recurrent-v1"
    probe_name = "probe_recurrent_training_v1" if training else "probe_recurrent_v1"
    run_name = "recurrent_training_v1" if training else "recurrent_v1"
    execute_name = "execute_recurrent_training_v1" if training else None
    execute = getattr(module, execute_name, None) if execute_name is not None else None
    probe = getattr(module, probe_name, None)
    run = getattr(module, run_name, None)
    if not callable(execute) and (not callable(probe) or not callable(run)):
        failure: Exception = RuntimeError(
            f"rwkv7_kernels does not implement {protocol}"
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
        # The readable model resolves these request semantics once per model
        # call.  Forwarding the immutable booleans keeps an optional training
        # dispatcher from synchronizing the CUDA mask back to the host in
        # every layer.  Standalone calls have no context and retain the
        # original positional-only protocol behavior.
        training_hints: dict[str, bool] = {}
        if training:
            fully_active = _training_batch_fully_active.get()
            token_aligned = _training_batch_token_aligned.get()
            batch_initial_state_zero = _training_batch_initial_state_zero.get()
            adaptive_fast_program = _training_batch_adaptive_fast_program.get()
            if fully_active is not None:
                training_hints["fully_active"] = fully_active
            if token_aligned is not None:
                training_hints["token_aligned"] = token_aligned
            if adaptive_fast_program is not None:
                training_hints["adaptive_fast_program"] = adaptive_fast_program
            if initial_state_zero is not None:
                training_hints["initial_state_zero"] = bool(
                    initial_state_zero
                    and (
                        batch_initial_state_zero
                        if batch_initial_state_zero is not None
                        else True
                    )
                )
        try:
            if callable(execute):
                execution = execute(*args, **training_hints)
                supported, implementation, reason = _probe_fields(
                    execution,
                    probe_name="execute_recurrent_training_v1",
                )
                if not isinstance(execution, dict) or "result" not in execution:
                    raise TypeError(
                        "execute_recurrent_training_v1() result is missing: result"
                    )
                candidate = execution["result"]
            else:
                supported, implementation, reason = _probe_fields(
                    probe(*args, **training_hints), probe_name=probe_name
                )
                candidate = None
            if not supported:
                if requested == "optimized" or atomic_fast_program:
                    raise RuntimeError(
                        "atomic adaptive RWKV7 recurrent program does not support "
                        "this request: "
                        f"{reason}"
                    )
                _record_route(
                    requested=requested,
                    selected="reference",
                    implementation="torch-reference-v1",
                    reason=reason,
                )
                return rwkv7_recurrent_reference(*args)
            if not callable(execute):
                candidate = run(*args, **training_hints)
            result = _validate_kernel_result(
                candidate,
                value=value,
                initial_state=initial_state,
            )
        except Exception as exc:
            if _is_checkpoint_control_flow(exc):
                raise
            if requested == "optimized" or atomic_fast_program:
                raise RuntimeError(
                    f"atomic adaptive RWKV7 {protocol} execution failed: "
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
    if requested == "optimized" or atomic_fast_program:
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
    "RWKV7TrainingBatchContext",
    "get_last_linear_route",
    "get_last_mix6_route",
    "get_last_model_route",
    "get_last_recurrent_route",
    "get_last_training_batch_context",
    "get_last_training_program_route",
    "maybe_linear_training",
    "maybe_mix6_training",
    "maybe_model_forward",
    "rwkv7_recurrent",
    "rwkv7_recurrent_reference",
    "resolve_training_batch_context",
    "set_training_batch_context",
    "training_batch_context",
]
