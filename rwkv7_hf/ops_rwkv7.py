# coding=utf-8
"""Small, hardware-neutral PyTorch operator boundary for RWKV-7."""
from __future__ import annotations

import torch


def rwkv7_recurrent(
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

    This deliberately contains no dispatch, compilation, environment-variable,
    device, or layout policy. A performance branch can replace this one
    boundary while the surrounding HF model remains unchanged.
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

    state = initial_state
    state_dtype = state.dtype
    outputs: list[torch.Tensor] = []
    for token_idx in range(time):
        r_t = receptance[:, token_idx].to(dtype=state_dtype)
        w_t = decay[:, token_idx].to(dtype=state_dtype)
        k_t = key[:, token_idx].to(dtype=state_dtype)
        v_t = value[:, token_idx].to(dtype=state_dtype)
        a_t = a[:, token_idx].to(dtype=state_dtype)
        b_t = b[:, token_idx].to(dtype=state_dtype)

        # The public/cache layout is [K,V]. Evaluate the equation in the
        # official [V,K] presentation, then transpose the result back. Keeping
        # this multiplication order is important for numerical parity with the
        # official RWKV implementation and FLA over long recurrent sequences.
        state_vk = state.transpose(-1, -2)
        ab = a_t.unsqueeze(-1) @ b_t.unsqueeze(-2)
        vk = v_t.unsqueeze(-1) @ k_t.unsqueeze(-2)
        candidate_vk = (
            state_vk * w_t.unsqueeze(-2)
            + state_vk @ ab
            + vk
        )
        candidate = candidate_vk.transpose(-1, -2)
        output = (candidate_vk @ r_t.unsqueeze(-1)).squeeze(-1)

        if attention_mask is not None:
            active = attention_mask[:, token_idx]
            state_active = active.view(batch, 1, 1, 1)
            output_active = active.view(batch, 1, 1)
            state = torch.where(state_active, candidate, state)
            output = torch.where(output_active, output, torch.zeros_like(output))
        else:
            state = candidate
        outputs.append(output.to(dtype=value.dtype))

    return torch.stack(outputs, dim=1), state


__all__ = ["rwkv7_recurrent"]
