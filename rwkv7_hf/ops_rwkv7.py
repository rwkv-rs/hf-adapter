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


__all__ = ["rwkv7_recurrent"]
