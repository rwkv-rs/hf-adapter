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
from typing import Any, Callable, Iterator

import torch


_KERNEL_API_VERSION = 4
_BACKEND_ENV = "RWKV7_BACKEND"
_BACKEND_MODES = ("auto", "reference", "optimized")
_kernel_module: ModuleType | None = None
_kernel_import_attempted = False
_kernel_import_error: str | None = None


# ---------------------------------------------------------------------------
# Readable source-of-truth recurrence
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Frozen API-v4 plug-in boundary
# ---------------------------------------------------------------------------


def _backend_mode(value: str | None) -> str:
    """Normalize the user-facing optional-backend mode."""

    normalized = (
        (os.environ.get(_BACKEND_ENV, "auto") if value is None else value)
        .strip()
        .lower()
    )
    if normalized not in _BACKEND_MODES:
        choices = ", ".join(_BACKEND_MODES)
        raise ValueError(f"RWKV7 backend must be one of {choices}; got {value!r}")
    return normalized


@dataclass(frozen=True)
class RWKV7ExecutionContext:
    """Model-owned facts and one optional training-program certificate.

    The context is resolved once per model call and passed explicitly through
    the readable layer loop.  It contains no hardware policy, cache tensors,
    model objects, or optimizer state.
    """

    training: bool
    fully_active: bool
    initial_state_zero: bool | None
    autograd_leaf_eligible: bool | None
    force_reference_program: bool
    optimized_program: bool | None
    program_id: str | None
    token_aligned: bool | None
    program_implementation: str
    program_reason: str


_last_routes: ContextVar[dict[str, dict[str, str]] | None] = ContextVar(
    "rwkv7_last_routes", default=None
)
_last_execution_context: ContextVar[RWKV7ExecutionContext | None] = ContextVar(
    "rwkv7_last_execution_context", default=None
)
# Third-party PEFT and quantized linear wrappers preserve ``forward(x)`` but
# cannot accept adapter-private kwargs.  One lexical bridge lets the owned
# RWKV7Linear leaf read the already-resolved context without changing the
# standard nn.Linear call contract.
_linear_execution_context: ContextVar[RWKV7ExecutionContext | None] = ContextVar(
    "rwkv7_linear_execution_context", default=None
)


@contextmanager
def linear_execution_context(
    context: RWKV7ExecutionContext,
) -> Iterator[RWKV7ExecutionContext]:
    if not isinstance(context, RWKV7ExecutionContext):
        raise TypeError("context must be an RWKV7ExecutionContext")
    token = _linear_execution_context.set(context)
    try:
        yield context
    finally:
        _linear_execution_context.reset(token)


def _record_route(kind: str, **route: str) -> None:
    routes = dict(_last_routes.get() or {})
    routes[kind] = dict(route)
    _last_routes.set(routes)


def _get_route(kind: str) -> dict[str, str] | None:
    route = (_last_routes.get() or {}).get(kind)
    return None if route is None else dict(route)


def get_last_recurrent_route() -> dict[str, str] | None:
    return _get_route("recurrent")


def get_last_model_route() -> dict[str, str] | None:
    return _get_route("model")


def get_last_linear_route() -> dict[str, str] | None:
    return _get_route("linear")


def get_last_mix6_route() -> dict[str, str] | None:
    return _get_route("mix6")


def get_last_training_program_route() -> dict[str, Any] | None:
    context = _last_execution_context.get()
    if context is None:
        return None
    selected = "not-applicable"
    if context.optimized_program is not None:
        selected = "optimized" if context.optimized_program else "reference"
    return {
        "selected": selected,
        "implementation": context.program_implementation,
        "reason": context.program_reason,
        "program_id": context.program_id,
        "facts": {
            "fully_active": context.fully_active,
            "token_aligned": context.token_aligned,
            "initial_state_zero": context.initial_state_zero,
            "autograd_leaf_eligible": context.autograd_leaf_eligible,
            "force_reference_program": context.force_reference_program,
        },
    }


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
    if type(version) is not int or version != _KERNEL_API_VERSION:
        _kernel_import_error = (
            f"kernel API mismatch: package={version!r}, adapter={_KERNEL_API_VERSION}"
        )
        return None
    execute = getattr(module, "execute_optional_v4", None)
    if not callable(execute):
        _kernel_import_error = "rwkv7_kernels does not implement execute_optional_v4"
        return None
    _kernel_module = module
    _kernel_import_error = None
    return module


def _kernel_envelope(kind: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
    module = _load_kernel_module()
    if module is None:
        raise RuntimeError(_kernel_import_error or "rwkv7_kernels is not installed")
    value = module.execute_optional_v4(kind, *args, **kwargs)
    if not isinstance(value, dict):
        raise TypeError("rwkv7_kernels.execute_optional_v4() must return a dict")
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
        raise TypeError(
            "execute_optional_v4() result is missing: " + ", ".join(sorted(missing))
        )
    extra = set(value) - required
    if extra:
        raise TypeError(
            "execute_optional_v4() result has unknown fields: "
            + ", ".join(sorted(str(name) for name in extra))
        )
    if type(value["api_version"]) is not int:
        raise TypeError("execute_optional_v4() api_version must be exactly int")
    if value["api_version"] != _KERNEL_API_VERSION:
        raise ValueError(
            "execute_optional_v4() API version mismatch: "
            f"expected {_KERNEL_API_VERSION}, got {value['api_version']!r}"
        )
    for name in ("kind", "implementation", "reason", "phase"):
        if type(value[name]) is not str:
            raise TypeError(f"execute_optional_v4() {name} must be exactly str")
    if type(value["supported"]) is not bool:
        raise TypeError("execute_optional_v4() supported must be exactly bool")
    if value["kind"] != kind:
        raise ValueError(
            f"execute_optional_v4() kind mismatch: expected {kind!r}, "
            f"got {value['kind']!r}"
        )
    envelope = {
        "api_version": _KERNEL_API_VERSION,
        "kind": kind,
        "supported": value["supported"],
        "implementation": value["implementation"],
        "reason": value["reason"],
        "result": value["result"],
        "phase": value["phase"],
    }
    if not envelope["supported"] and envelope["result"] is not None:
        raise ValueError("unsupported optional operation must return result=None")
    return envelope


def _reference_implementation(kind: str) -> str:
    return {
        "model_forward": "torch-reference-model-v1",
        "linear_training": "torch-reference-linear-v1",
        "mix6_training": "torch-reference-mix6-v1",
        "recurrent": "torch-reference-v1",
    }[kind]


def _route_key(kind: str) -> str:
    return {
        "model_forward": "model",
        "linear_training": "linear",
        "mix6_training": "mix6",
        "recurrent": "recurrent",
    }[kind]


def _optional_result(
    kind: str,
    *args: Any,
    backend: str | None,
    execution_context: RWKV7ExecutionContext | None = None,
    phase: str | None = None,
    validate: Callable[[Any], Any] | None = None,
    fail_closed_on_error: bool = False,
    **kwargs: Any,
) -> Any | None:
    """Execute one v4 operation, or return ``None`` for readable fallback."""

    requested = _backend_mode(backend)
    context = execution_context
    atomic = bool(context is not None and context.optimized_program is True)
    force_reference = bool(
        context is not None
        and context.training
        and (context.force_reference_program or context.autograd_leaf_eligible is False)
    )
    reference = _reference_implementation(kind)
    route_key = _route_key(kind)

    def record(selected: str, implementation: str, reason: str, route_phase=None):
        route = {
            "requested": requested,
            "selected": selected,
            "implementation": implementation,
            "reason": reason,
        }
        if route_phase is not None:
            route["phase"] = str(route_phase)
        _record_route(route_key, **route)

    if requested == "reference":
        record(
            "reference", reference, "reference backend was explicitly requested", phase
        )
        return None
    if force_reference:
        reason = (
            "the model selected one readable training program because this "
            "request has no atomic fast-program certificate"
        )
        if requested == "optimized":
            raise RuntimeError(
                f"optimized RWKV7 {kind.replace('_', ' ')} requires an atomic "
                f"fast-program certificate: {reason}"
            )
        record("reference", reference, reason, phase)
        return None

    if _load_kernel_module() is None:
        reason = _kernel_import_error or "rwkv7_kernels is not installed"
        if requested == "optimized" or atomic:
            raise RuntimeError(f"optimized RWKV7 backend is unavailable: {reason}")
        record("reference", reference, reason, phase)
        return None

    try:
        envelope = _kernel_envelope(kind, *args, **kwargs)
        actual_phase = envelope["phase"] or phase
        if not envelope["supported"]:
            if requested == "optimized" or atomic:
                raise RuntimeError(envelope["reason"])
            record("reference", reference, envelope["reason"], actual_phase)
            return None
        result = envelope["result"]
        if validate is not None:
            result = validate(result)
    except Exception as exc:
        if _is_checkpoint_control_flow(exc):
            raise
        if requested == "optimized" or atomic or fail_closed_on_error:
            failure_mode = (
                "fail-closed"
                if fail_closed_on_error
                else ("atomic" if atomic else "strict")
            )
            raise RuntimeError(
                f"{failure_mode} optional RWKV7 "
                f"{kind.replace('_', ' ')} execution failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        reason = f"optional kernel failure: {type(exc).__name__}: {exc}"
        record("reference", reference, reason, phase)
        return None

    record(
        "optimized",
        envelope["implementation"],
        envelope["reason"],
        actual_phase,
    )
    return result


# ---------------------------------------------------------------------------
# Model-owned execution facts
# ---------------------------------------------------------------------------


def resolve_execution_context(
    attention_mask: torch.Tensor,
    *,
    training: bool,
    fully_active: bool | None = None,
    initial_state_zero: bool | None = None,
    autograd_leaf_eligible: bool | None = None,
    force_reference_program: bool = False,
    hidden_states: torch.Tensor | None = None,
    head_dim: int | None = None,
) -> RWKV7ExecutionContext:
    """Resolve model facts and an optional atomic training certificate once."""

    if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
        raise TypeError("attention_mask must be a rank-2 tensor")
    if fully_active is None:
        fully_active = bool(attention_mask.to(dtype=torch.bool).all().detach().cpu())
    if not isinstance(fully_active, bool):
        raise TypeError("fully_active must be a bool or None")
    if initial_state_zero is not None and not isinstance(initial_state_zero, bool):
        raise TypeError("initial_state_zero must be a bool or None")
    if not isinstance(force_reference_program, bool):
        raise TypeError("force_reference_program must be a bool")

    eligible: bool | None = None
    optimized: bool | None = None
    program_id: str | None = None
    token_aligned: bool | None = None
    implementation = "torch-reference-training-program-v1"
    reason = "model is not executing a training program"
    force_reference = False

    if training:
        if autograd_leaf_eligible is None:
            eligible = bool(
                torch.is_grad_enabled()
                and isinstance(hidden_states, torch.Tensor)
                and hidden_states.requires_grad
            )
        elif not isinstance(autograd_leaf_eligible, bool):
            raise TypeError("autograd_leaf_eligible must be a bool or None")
        else:
            eligible = autograd_leaf_eligible
        force_reference = bool(
            force_reference_program
            or (eligible is False and isinstance(hidden_states, torch.Tensor))
        )
        optimized = False
        mode = _backend_mode(None)
        if mode == "reference":
            reason = "reference backend was explicitly requested"
        elif force_reference:
            reason = "the request cannot receive an atomic fast-program certificate"
        elif not isinstance(hidden_states, torch.Tensor):
            reason = "hidden_states is unavailable for training preflight"
        elif _load_kernel_module() is None:
            reason = _kernel_import_error or "rwkv7_kernels is not installed"
        else:
            try:
                envelope = _kernel_envelope(
                    "training_program",
                    hidden_states,
                    attention_mask,
                    training=True,
                    fully_active=fully_active,
                    initial_state_zero=initial_state_zero,
                    autograd_leaf_eligible=eligible,
                    force_reference_program=force_reference,
                    head_dim=head_dim,
                )
                if envelope["supported"]:
                    result = envelope["result"]
                    if not isinstance(result, dict):
                        raise TypeError("training_program result must be a dict")
                    program_id = result.get("program_id")
                    token_aligned = result.get("token_aligned")
                    if not isinstance(program_id, str) or not isinstance(
                        token_aligned, bool
                    ):
                        raise TypeError(
                            "training_program result requires string program_id "
                            "and boolean token_aligned"
                        )
            except Exception as exc:
                if _is_checkpoint_control_flow(exc):
                    raise
                program_id = None
                token_aligned = None
                reason = (
                    f"optional training preflight failed: {type(exc).__name__}: {exc}"
                )
            else:
                implementation = envelope["implementation"]
                reason = envelope["reason"]
                optimized = envelope["supported"]
        if mode == "optimized" and optimized is not True:
            raise RuntimeError(
                "optimized RWKV7 training program is unavailable at the model "
                f"boundary: {reason}"
            )

    context = RWKV7ExecutionContext(
        training=bool(training),
        fully_active=fully_active,
        initial_state_zero=initial_state_zero if training else None,
        autograd_leaf_eligible=eligible,
        force_reference_program=force_reference,
        optimized_program=optimized,
        program_id=program_id,
        token_aligned=token_aligned,
        program_implementation=implementation,
        program_reason=reason,
    )
    _last_execution_context.set(context)
    return context


# ---------------------------------------------------------------------------
# Inference-only whole-model plug-in
# ---------------------------------------------------------------------------


def maybe_model_forward(
    owner: Any,
    request: dict[str, Any],
    *,
    backend: str | None = None,
    execution_context: RWKV7ExecutionContext | None = None,
) -> dict[str, Any] | None:
    """Try one inference-only whole-model call through the optional wheel."""

    if not isinstance(request, dict):
        raise TypeError("RWKV7 model-forward request must be a dict")
    kind = request.get("model_kind")
    if kind not in ("base", "causal_lm"):
        raise ValueError("model_kind must be 'base' or 'causal_lm'")
    training = bool(request.get("training")) or bool(request.get("grad_enabled"))
    hidden = request.get("hidden_states")
    phase = "training" if training else "prefill"
    if (
        not training
        and isinstance(hidden, torch.Tensor)
        and hidden.ndim >= 2
        and int(hidden.shape[1]) == 1
    ):
        phase = "decode"
    if training:
        _record_route(
            "model",
            requested=_backend_mode(backend),
            selected="reference",
            implementation="torch-reference-model-v1",
            reason=(
                "readable HF training loop owns structure; optional tensor "
                "leaves dispatch through one explicit execution context"
            ),
            phase="training",
        )
        return None

    # The explicit reference path and a missing optional wheel cannot execute
    # foreign code, so they do not need a defensive deep cache clone.  This
    # keeps the package-free reference implementation allocation-free at the
    # optional boundary while strict mode still fails before touching cache.
    if _backend_mode(backend) == "reference" or _load_kernel_module() is None:
        return _optional_result(
            "model_forward",
            owner,
            request,
            backend=backend,
            execution_context=execution_context,
            phase=phase,
        )

    # Native decode binds the canonical cache tensors directly to persistent
    # CUDA-Graph buffers.  Cloning here would copy every layer's recurrent
    # state on every token and destroy the runner's zero-copy fast path.  The
    # API-v4 contract therefore requires a negative probe to be side-effect
    # free; once a positive operation starts, any exception or malformed
    # payload is fail-closed even in ``auto`` mode and the caller discards the
    # failed cache instead of recomputing from partially updated state.
    original_cache = request.get("past_key_values")
    original_seen_tokens = (
        int(original_cache.get_seq_length()) if original_cache is not None else None
    )

    expected_shape: tuple[int, ...] | None = None
    expected_dtype: torch.dtype | None = None
    expected_device: torch.device | None = None
    sequence_length: int | None = None
    batch_size: int | None = None
    if kind == "base" and isinstance(hidden, torch.Tensor) and hidden.ndim == 3:
        expected_shape = tuple(hidden.shape)
        expected_dtype = hidden.dtype
        expected_device = hidden.device
        batch_size = int(hidden.shape[0])
        sequence_length = int(hidden.shape[1])
    elif kind == "causal_lm":
        model_input = request.get("inputs_embeds")
        if not isinstance(model_input, torch.Tensor):
            model_input = request.get("input_ids")
        if isinstance(model_input, torch.Tensor) and model_input.ndim in (1, 2, 3):
            if model_input.ndim == 1:
                batch_size, sequence_length = 1, int(model_input.shape[0])
            else:
                batch_size = int(model_input.shape[0])
                sequence_length = int(model_input.shape[1])
            keep = request.get("logits_to_keep")
            if isinstance(keep, torch.Tensor):
                output_length = int(keep.numel())
            elif keep is None or int(keep) <= 0:
                output_length = sequence_length
            else:
                output_length = min(int(keep), sequence_length)
            vocab_size = getattr(getattr(owner, "config", None), "vocab_size", None)
            if isinstance(vocab_size, int):
                expected_shape = (batch_size, output_length, vocab_size)
            output_embeddings = getattr(owner, "lm_head", None)
            output_weight = getattr(output_embeddings, "weight", None)
            if isinstance(output_weight, torch.Tensor):
                expected_dtype = output_weight.dtype
                expected_device = output_weight.device
            elif isinstance(model_input, torch.Tensor):
                expected_device = model_input.device

    def validate(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict) or value.get("output_kind") != kind:
            raise TypeError("optional model result has the wrong output_kind")
        tensor_name = "last_hidden_state" if kind == "base" else "logits"
        output_tensor = value.get(tensor_name)
        if not isinstance(output_tensor, torch.Tensor):
            raise TypeError(f"optional model result requires tensor {tensor_name}")
        if expected_shape is not None and tuple(output_tensor.shape) != expected_shape:
            raise ValueError(
                f"optional model {tensor_name} shape mismatch: expected "
                f"{expected_shape}, got {tuple(output_tensor.shape)}"
            )
        if (
            expected_dtype is not None
            and expected_device is not None
            and (
                output_tensor.dtype != expected_dtype
                or output_tensor.device != expected_device
            )
        ):
            raise ValueError(
                f"optional model {tensor_name} must share model dtype and device"
            )
        loss = value.get("loss")
        labels_present = request.get("labels") is not None
        if kind == "base" and loss is not None:
            raise TypeError("optional base-model result cannot contain loss")
        if kind == "causal_lm" and labels_present and loss is None:
            raise TypeError("optional causal-LM result requires loss when labels exist")
        if kind == "causal_lm" and not labels_present and loss is not None:
            raise TypeError("optional causal-LM result returned loss without labels")
        if loss is not None:
            if (
                not isinstance(loss, torch.Tensor)
                or loss.numel() != 1
                or loss.device != output_tensor.device
                or not loss.dtype.is_floating_point
            ):
                raise TypeError(
                    "optional model loss must be one floating scalar tensor"
                )
        history = value.get("hidden_states")
        history_requested = bool(request.get("output_hidden_states"))
        if history_requested and not isinstance(history, (tuple, list)):
            raise TypeError("optional model result requires hidden-state history")
        if not history_requested and history is not None:
            raise TypeError("optional model returned unrequested hidden-state history")
        if history is not None:
            expected_layers = getattr(
                getattr(owner, "config", None), "num_hidden_layers", None
            )
            if isinstance(expected_layers, int) and len(history) != expected_layers + 1:
                raise ValueError("optional model hidden-state history length mismatch")
            hidden_size = getattr(getattr(owner, "config", None), "hidden_size", None)
            history_shape = (
                (batch_size, sequence_length, hidden_size)
                if all(
                    isinstance(item, int)
                    for item in (batch_size, sequence_length, hidden_size)
                )
                else None
            )
            if any(
                not isinstance(item, torch.Tensor)
                or item.device != output_tensor.device
                or item.dtype != output_tensor.dtype
                or (history_shape is not None and tuple(item.shape) != history_shape)
                for item in history
            ):
                raise TypeError(
                    "optional model hidden_states must match model shape, dtype, "
                    "and device"
                )
        returned_cache = value.get("past_key_values")
        if bool(request.get("use_cache")):
            if original_cache is None or returned_cache is not original_cache:
                raise TypeError(
                    "optional model result must return the caller's canonical cache"
                )
            if sequence_length is not None and original_seen_tokens is not None:
                expected_seen = original_seen_tokens + sequence_length
                if int(original_cache.get_seq_length()) != expected_seen:
                    raise ValueError(
                        "optional model cache advanced by an unexpected token count"
                    )
        elif returned_cache is not None:
            raise TypeError("optional model result returned cache with use_cache=False")
        return value

    result = _optional_result(
        "model_forward",
        owner,
        request,
        backend=backend,
        execution_context=execution_context,
        phase=phase,
        validate=validate,
        fail_closed_on_error=True,
    )
    return result


# ---------------------------------------------------------------------------
# Training tensor-leaf plug-ins
# ---------------------------------------------------------------------------


def _training_facts(context: RWKV7ExecutionContext | None) -> dict[str, Any]:
    if context is None:
        return {}
    return {
        "fully_active": context.fully_active,
        "initial_state_zero": context.initial_state_zero,
        "token_aligned": context.token_aligned,
        "autograd_leaf_eligible": context.autograd_leaf_eligible,
    }


def maybe_linear_training(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    training: bool,
    backend: str | None = None,
    execution_context: RWKV7ExecutionContext | None = None,
) -> torch.Tensor | None:
    """Return an optional stateless training projection, otherwise ``None``."""

    if not training:
        return None
    context = execution_context or _linear_execution_context.get()

    def validate(result: Any) -> torch.Tensor:
        if not isinstance(result, torch.Tensor):
            raise TypeError("optional linear_training result must be a tensor")
        expected = (*value.shape[:-1], int(weight.shape[0]))
        if tuple(result.shape) != expected:
            raise ValueError(
                f"training linear output shape mismatch: expected {expected}, "
                f"got {tuple(result.shape)}"
            )
        if result.dtype != value.dtype or result.device != value.device:
            raise ValueError(
                "training linear output must share the input dtype and device"
            )
        return result

    return _optional_result(
        "linear_training",
        value,
        weight,
        bias,
        backend=backend,
        execution_context=context,
        phase="training",
        program_id=None if context is None else context.program_id,
        facts=_training_facts(context),
        validate=validate,
    )


def maybe_mix6_training(
    value: torch.Tensor,
    shifted: torch.Tensor,
    mixes: tuple[torch.Tensor, ...],
    *,
    training: bool,
    backend: str | None = None,
    execution_context: RWKV7ExecutionContext | None = None,
) -> tuple[torch.Tensor, ...] | None:
    """Return the optional six-way token-mix leaf, otherwise ``None``."""

    if not training:
        return None
    if len(mixes) != 6:
        raise ValueError("RWKV7 Mix6 requires exactly six parameter tensors")
    context = execution_context

    def validate(result: Any) -> tuple[torch.Tensor, ...]:
        if not isinstance(result, tuple) or len(result) != 6:
            raise TypeError("optional mix6_training result must contain six tensors")
        if any(
            not isinstance(item, torch.Tensor)
            or tuple(item.shape) != tuple(value.shape)
            or item.dtype != value.dtype
            or item.device != value.device
            for item in result
        ):
            raise ValueError("Mix6 outputs must match input shape, dtype, and device")
        return result

    return _optional_result(
        "mix6_training",
        value,
        shifted,
        *mixes,
        backend=backend,
        execution_context=context,
        phase="training",
        program_id=None if context is None else context.program_id,
        facts=_training_facts(context),
        validate=validate,
    )


# ---------------------------------------------------------------------------
# Public recurrence selector
# ---------------------------------------------------------------------------


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
    execution_context: RWKV7ExecutionContext | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the canonical recurrence through reference or API-v4 backend."""

    if initial_state_zero is not None and not isinstance(initial_state_zero, bool):
        raise TypeError("initial_state_zero must be a bool or None")
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
    context = execution_context
    facts = _training_facts(context)
    if training and initial_state_zero is not None:
        batch_zero = None if context is None else context.initial_state_zero
        facts["initial_state_zero"] = bool(
            initial_state_zero and (True if batch_zero is None else batch_zero)
        )
    kwargs: dict[str, Any] = {"training": bool(training)}
    if training:
        kwargs.update(
            program_id=None if context is None else context.program_id,
            facts=facts,
        )

    def validate(result: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(result, tuple) or len(result) != 2:
            raise TypeError("optional recurrent result must contain two tensors")
        output, final_state = result
        if not isinstance(output, torch.Tensor) or not isinstance(
            final_state, torch.Tensor
        ):
            raise TypeError("optional recurrent result must contain two tensors")
        if tuple(output.shape) != tuple(value.shape):
            raise ValueError("optional recurrent output shape mismatch")
        if tuple(final_state.shape) != tuple(initial_state.shape):
            raise ValueError("optional recurrent state shape mismatch")
        if output.dtype != value.dtype or output.device != value.device:
            raise ValueError(
                "optional recurrent output must share value dtype and device"
            )
        if (
            final_state.dtype != initial_state.dtype
            or final_state.device != initial_state.device
        ):
            raise ValueError(
                "optional recurrent state must share initial-state dtype and device"
            )
        return output, final_state

    result = _optional_result(
        "recurrent",
        *args,
        backend=backend,
        execution_context=context,
        phase="training" if training else None,
        validate=validate,
        **kwargs,
    )
    if result is None:
        return rwkv7_recurrent_reference(*args)
    return result


def _reset_kernel_discovery_for_tests() -> None:
    """Reset the lazy optional package and per-context diagnostics."""

    global _kernel_import_attempted, _kernel_import_error, _kernel_module
    _kernel_module = None
    _kernel_import_attempted = False
    _kernel_import_error = None
    _last_routes.set(None)
    _last_execution_context.set(None)
    _linear_execution_context.set(None)


__all__ = [
    "RWKV7ExecutionContext",
    "get_last_linear_route",
    "get_last_mix6_route",
    "get_last_model_route",
    "get_last_recurrent_route",
    "get_last_training_program_route",
    "linear_execution_context",
    "maybe_linear_training",
    "maybe_mix6_training",
    "maybe_model_forward",
    "resolve_execution_context",
    "rwkv7_recurrent",
    "rwkv7_recurrent_reference",
]
