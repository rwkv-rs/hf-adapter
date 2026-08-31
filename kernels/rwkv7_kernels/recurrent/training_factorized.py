"""Factorized CUDA implementation of the RWKV-7 training recurrence.

The Hugging Face model supplies canonical ``[B,T,H,D]`` vectors, positive
decay, and public ``[B,H,K,V]`` state.  The CUDA leaf consumes dense chunks of
16 tokens.  This adapter compacts masked samples into one padded batch, adds
only no-op recurrent updates, and scatters outputs back so left and right
padding retain the public state semantics. Unsupported dtype, layout or
autograd requests fail closed.
"""

from __future__ import annotations

from typing import Any

import torch

from .._runtime_preflight import recurrent_runtime_certified


IMPLEMENTATION = "native-nvidia-rwkv7-factorized-recurrent-training-v1"
TOKEN_CHUNK_LENGTH = 16


def _unsupported(reason: str) -> dict[str, Any]:
    return {
        "supported": False,
        "implementation": IMPLEMENTATION,
        "reason": reason,
    }


def probe_recurrent_training_v1(
    receptance,
    decay,
    key,
    value,
    a,
    b,
    initial_state,
    attention_mask,
    *,
    initial_state_zero: bool | None = None,
) -> dict[str, Any]:
    tensors = (receptance, decay, key, value, a, b, initial_state)
    if not all(isinstance(item, torch.Tensor) for item in tensors):
        return _unsupported("all recurrent inputs and state must be tensors")
    if attention_mask is not None and not isinstance(attention_mask, torch.Tensor):
        return _unsupported("attention_mask must be a tensor or None")
    if not torch.cuda.is_available() or not all(item.is_cuda for item in tensors):
        return _unsupported("the native training kernel requires CUDA tensors")
    if receptance.ndim != 4 or initial_state.ndim != 4:
        return _unsupported("rank-four recurrent inputs and state are required")
    expected = tuple(receptance.shape)
    if any(tuple(item.shape) != expected for item in (decay, key, value, a, b)):
        return _unsupported("all recurrent vectors must have identical shapes")
    batch, tokens, heads, width = expected
    if width != 64 or tuple(initial_state.shape) != (batch, heads, 64, 64):
        return _unsupported("the native training kernel requires K=V=64")
    if tokens <= 0:
        return _unsupported("the native training kernel requires a non-empty sequence")
    if receptance.dtype != torch.bfloat16:
        return _unsupported("the native training kernel requires BF16 r/k/v/a/b")
    if any(item.dtype != torch.bfloat16 for item in (key, value, a, b)):
        return _unsupported("the native training kernel requires BF16 r/k/v/a/b")
    if decay.dtype != torch.float32:
        return _unsupported("the native training kernel requires translated FP32 decay")
    if initial_state.dtype != torch.float32:
        return _unsupported("the public recurrent state must be FP32")
    if initial_state_zero is not None and not isinstance(initial_state_zero, bool):
        return _unsupported("initial_state_zero must be a bool or None")
    if initial_state_zero is False:
        return _unsupported(
            "native training requires a model-proven zero initial state"
        )
    if initial_state_zero is None and bool(
        torch.count_nonzero(initial_state).detach().cpu()
    ):
        return _unsupported("native training requires a zero initial state")
    if attention_mask is not None:
        if tuple(attention_mask.shape) != (batch, tokens):
            return _unsupported("attention_mask must be shaped [B,T]")
        if not attention_mask.is_cuda:
            return _unsupported("attention_mask must share the CUDA device")
        if attention_mask.device != receptance.device:
            return _unsupported("attention_mask must share the recurrent CUDA device")
    if any(item.device != receptance.device for item in tensors):
        return _unsupported("all recurrent tensors must share one CUDA device")
    if not any(item.requires_grad for item in tensors):
        return _unsupported("the native training kernel requires an autograd request")

    if recurrent_runtime_certified(receptance.device):
        return {
            "supported": True,
            "implementation": IMPLEMENTATION,
            "reason": "coupled runtime preflight and tensor-local checks passed",
        }
    capability = torch.cuda.get_device_capability(receptance.device)
    if capability < (8, 0):
        return _unsupported("the BF16 training kernel requires sm80 or newer")
    from ..nvidia.official_training_cuda import recurrent_training_cuda_available

    # Only the explicit ``factorized`` policy or the dense branch of
    # ``adaptive`` reaches this probe. Lazy compilation is therefore expected
    # here; production ``auto`` returns before importing or building a
    # training extension.
    if not recurrent_training_cuda_available(build=True):
        return _unsupported("the native CUDA training extension is not loaded")
    return {
        "supported": True,
        "implementation": IMPLEMENTATION,
        "reason": "zero-state BF16 CUDA autograd request is supported",
    }


def _run_masked_training(
    recurrent_inputs,
    initial_state,
    attention_mask,
    *,
    runner,
    fully_active: bool | None = None,
    token_aligned: bool | None = None,
):
    """Pack one dense CUDA batch while preserving per-sample padding."""

    if fully_active is not None and not isinstance(fully_active, bool):
        raise TypeError("fully_active must be a bool or None")
    if token_aligned is not None and not isinstance(token_aligned, bool):
        raise TypeError("token_aligned must be a bool or None")

    receptance, decay, key, value, a, b = recurrent_inputs
    batch, tokens, heads, width = receptance.shape
    mask = (
        torch.ones(batch, tokens, dtype=torch.bool, device=receptance.device)
        if attention_mask is None
        else attention_mask.to(device=receptance.device, dtype=torch.bool)
    )
    if token_aligned is None:
        token_aligned = tokens % TOKEN_CHUNK_LENGTH == 0
    if fully_active is None:
        fully_active = attention_mask is None or bool(mask.all().detach().cpu())
    if token_aligned and fully_active:
        return runner(*recurrent_inputs, initial_state)

    active_by_sample = [
        torch.nonzero(mask[batch_idx], as_tuple=False).flatten()
        for batch_idx in range(batch)
    ]
    valid_lengths = [int(active.numel()) for active in active_by_sample]
    longest = max(valid_lengths, default=0)
    padded_tokens = max(
        TOKEN_CHUNK_LENGTH,
        ((longest + TOKEN_CHUNK_LENGTH - 1) // TOKEN_CHUNK_LENGTH * TOKEN_CHUNK_LENGTH),
    )

    packed_inputs = []
    for input_index, item in enumerate(recurrent_inputs):
        rows = []
        for batch_idx, active in enumerate(active_by_sample):
            selected = item[batch_idx : batch_idx + 1].index_select(1, active)
            pad_tokens = padded_tokens - int(active.numel())
            if pad_tokens:
                # A zero r/k/v/a/b update with decay=1 is an exact recurrent
                # no-op.  The empty selection remains in the graph for an
                # all-padding sample, so its public input gradient is a tensor
                # of explicit zeros rather than ``None``.
                fill = 1.0 if input_index == 1 else 0.0
                padding = torch.full(
                    (1, pad_tokens, heads, width),
                    fill,
                    device=item.device,
                    dtype=item.dtype,
                )
                selected = torch.cat((selected, padding), dim=1)
            rows.append(selected)
        packed_inputs.append(torch.cat(rows, dim=0))

    compact_output, final_state = runner(*packed_inputs, initial_state)
    restored_rows = []
    for batch_idx, (active, valid_tokens) in enumerate(
        zip(active_by_sample, valid_lengths, strict=True)
    ):
        restored = torch.zeros_like(value[batch_idx : batch_idx + 1]).index_copy(
            1,
            active,
            compact_output[batch_idx : batch_idx + 1, :valid_tokens],
        )
        restored_rows.append(restored)
    return torch.cat(restored_rows, dim=0), final_state


def _run_factorized_recurrent(
    receptance,
    decay,
    key,
    value,
    a,
    b,
    initial_state,
    attention_mask,
    *,
    fully_active: bool | None = None,
    initial_state_zero: bool | None = None,
    token_aligned: bool | None = None,
):
    # Every public caller probes before execution.  That probe loads the
    # extension once; call the custom-autograd edge directly so each model
    # layer does not repeat runtime/toolchain discovery after certification.
    from ..nvidia.official_training_cuda import _ClampW

    return _run_masked_training(
        (receptance, decay, key, value, a, b),
        initial_state,
        attention_mask,
        runner=_ClampW.apply,
        fully_active=fully_active,
        token_aligned=token_aligned,
    )


def recurrent_training_v1(
    receptance,
    decay,
    key,
    value,
    a,
    b,
    initial_state,
    attention_mask,
    *,
    fully_active: bool | None = None,
    initial_state_zero: bool | None = None,
    token_aligned: bool | None = None,
):
    support = probe_recurrent_training_v1(
        receptance,
        decay,
        key,
        value,
        a,
        b,
        initial_state,
        attention_mask,
        initial_state_zero=initial_state_zero,
    )
    if not support["supported"]:
        raise RuntimeError(str(support["reason"]))
    return _run_factorized_recurrent(
        receptance,
        decay,
        key,
        value,
        a,
        b,
        initial_state,
        attention_mask,
        fully_active=fully_active,
        initial_state_zero=initial_state_zero,
        token_aligned=token_aligned,
    )


__all__ = [
    "IMPLEMENTATION",
    "TOKEN_CHUNK_LENGTH",
    "probe_recurrent_training_v1",
    "recurrent_training_v1",
]
