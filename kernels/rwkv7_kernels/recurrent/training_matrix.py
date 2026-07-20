"""Exact batched-matrix training leaf for the RWKV-7 recurrence.

The readable Hugging Face reference evaluates each sample independently to
make batch regrouping invariance explicit.  This optional leaf preserves the
same mixed-precision matrix order while batching independent samples and
heads into each PyTorch CUDA matrix multiplication.  PyTorch still owns the
autograd graph; this module owns no model, parameter, cache, optimizer, or
checkpoint state.
"""

from __future__ import annotations

from typing import Any

import torch


IMPLEMENTATION = "torch-cuda-rwkv7-batched-matrix-recurrent-training-v1"


def _unsupported(reason: str) -> dict[str, Any]:
    return {
        "supported": False,
        "implementation": IMPLEMENTATION,
        "reason": reason,
    }


def probe_recurrent_training_v1(
    receptance: torch.Tensor,
    decay: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> dict[str, Any]:
    """Report support for the exact batched CUDA matrix recurrence."""

    vectors = (receptance, decay, key, value, a, b)
    if not all(isinstance(item, torch.Tensor) for item in (*vectors, initial_state)):
        return _unsupported("all recurrent inputs and state must be tensors")
    if receptance.ndim != 4 or initial_state.ndim != 4:
        return _unsupported("rank-four recurrent inputs and state are required")
    batch, tokens, heads, key_width = receptance.shape
    expected_key_shape = (batch, tokens, heads, key_width)
    if any(tuple(item.shape) != expected_key_shape for item in (decay, key, a, b)):
        return _unsupported(
            "receptance, decay, key, a, and b must have identical shapes"
        )
    if tuple(value.shape[:3]) != (batch, tokens, heads):
        return _unsupported("value must share the recurrent [B,T,H] dimensions")
    value_width = int(value.shape[-1])
    if tuple(initial_state.shape) != (batch, heads, key_width, value_width):
        return _unsupported("initial_state must use canonical [B,H,K,V] layout")
    if tokens <= 0:
        return _unsupported("the matrix training leaf requires a non-empty sequence")
    devices = {item.device for item in (*vectors, initial_state)}
    if len(devices) != 1 or not receptance.is_cuda:
        return _unsupported("the matrix training leaf requires one CUDA device")
    if receptance.dtype not in (torch.float16, torch.bfloat16):
        return _unsupported("recurrent vectors must use FP16 or BF16")
    if any(item.dtype != receptance.dtype for item in (key, value, a, b)):
        return _unsupported("r/k/v/a/b must share one model dtype")
    if decay.dtype != torch.float32 or initial_state.dtype != torch.float32:
        return _unsupported("decay and canonical recurrent state must use FP32")
    if attention_mask is not None:
        if not isinstance(attention_mask, torch.Tensor):
            return _unsupported("attention_mask must be a tensor or None")
        if tuple(attention_mask.shape) != (batch, tokens):
            return _unsupported("attention_mask must be shaped [B,T]")
        if attention_mask.device != receptance.device:
            return _unsupported("attention_mask must share the recurrent device")
    if not any(item.requires_grad for item in (*vectors, initial_state)):
        return _unsupported("the matrix training leaf requires an autograd request")
    return {
        "supported": True,
        "implementation": IMPLEMENTATION,
        "reason": (
            "exact mixed-precision RWKV-7 matrix recurrence is supported by "
            "batched PyTorch CUDA operations"
        ),
    }


def _batched_matrix_recurrence(
    receptance: torch.Tensor,
    decay: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the public matrix order while batching samples and heads."""

    state = initial_state
    outputs: list[torch.Tensor] = []
    mask = (
        None
        if attention_mask is None
        else attention_mask.to(device=state.device, dtype=torch.bool)
    )
    for token_index in range(int(receptance.shape[1])):
        r_token = receptance[:, token_index]
        decay_token = decay[:, token_index].to(dtype=state.dtype)
        key_token = key[:, token_index]
        value_token = value[:, token_index]
        a_token = a[:, token_index]
        b_token = b[:, token_index]

        # Match the public recurrence exactly: both outer products are formed
        # in the model dtype, then promoted before the FP32 state update.  The
        # multiplication order must not be replaced with a factorized rank-one
        # update because BF16/FP16 rounding makes those different programs.
        state_value_key = state.transpose(-1, -2)
        a_b = a_token.unsqueeze(-1) @ b_token.unsqueeze(-2)
        value_key = value_token.unsqueeze(-1) @ key_token.unsqueeze(-2)
        candidate_value_key = (
            state_value_key * decay_token.unsqueeze(-2)
            + state_value_key @ a_b.to(dtype=state.dtype)
            + value_key.to(dtype=state.dtype)
        )
        candidate_state = candidate_value_key.transpose(-1, -2)
        output = (
            candidate_value_key.to(dtype=r_token.dtype)
            @ r_token.unsqueeze(-1)
        ).squeeze(-1)

        if mask is not None:
            active = mask[:, token_index]
            state = torch.where(
                active[:, None, None, None], candidate_state, state
            )
            output = torch.where(
                active[:, None, None], output, torch.zeros_like(output)
            )
        else:
            state = candidate_state
        outputs.append(output.to(dtype=value.dtype))
    return torch.stack(outputs, dim=1), state


def recurrent_training_v1(
    receptance: torch.Tensor,
    decay: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the exact batched-matrix recurrence with ordinary PyTorch autograd."""

    support = probe_recurrent_training_v1(
        receptance,
        decay,
        key,
        value,
        a,
        b,
        initial_state,
        attention_mask,
    )
    if not support["supported"]:
        raise RuntimeError(str(support["reason"]))
    return _batched_matrix_recurrence(
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
    "IMPLEMENTATION",
    "probe_recurrent_training_v1",
    "recurrent_training_v1",
]
