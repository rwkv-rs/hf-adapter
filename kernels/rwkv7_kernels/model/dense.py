"""Dense sequential whole-model executor migrated from the native JIT line.

This is an internal correctness bridge for the final backend-v2 dispatcher. It
already executes the complete clean RWKV-7 layer stack and canonical cache, but
is not advertised by the public probe until the fused prefill, decode, quant and
training phase set is migrated and accepted as one immutable wheel.
"""
from __future__ import annotations

from typing import Any

import torch

from .dense_step import block_step_batched
from .packing import extract_dense_packs


IMPLEMENTATION = "native-torchscript-dense-sequential-v2"


def _active_where(active: torch.Tensor, candidate: torch.Tensor, old: torch.Tensor):
    shape = (int(active.shape[0]),) + (1,) * (candidate.ndim - 1)
    return torch.where(active.view(shape), candidate, old)


def run_base_model(owner: Any, request: dict[str, Any]) -> dict[str, Any]:
    """Execute a normalized base-model request and return a plain mapping.

    Required request fields are produced by the clean model immediately before
    its readable layer loop. ``past_key_values`` therefore remains the exact HF
    cache instance, while the historical internal [V,K] presentation is used
    only for the token-step math and transposed back before publication.
    """

    hidden_states = request["hidden_states"]
    attention_mask = request["attention_mask"]
    working_cache = request["past_key_values"]
    use_cache = bool(request["use_cache"])
    output_hidden_states = bool(request["output_hidden_states"])

    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
        raise TypeError("dense model backend requires [B,T,D] hidden_states")
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
        raise TypeError("dense model backend requires a [B,T] attention_mask")
    batch, sequence, hidden = hidden_states.shape
    if tuple(attention_mask.shape) != (batch, sequence):
        raise ValueError("attention_mask shape does not match hidden_states")
    if int(hidden) != int(owner.config.hidden_size):
        raise ValueError("hidden dimension does not match RWKV7 config")

    packs, heads, head_dim = extract_dense_packs(owner)
    recurrent_vk: list[torch.Tensor] = []
    attention_shift: list[torch.Tensor] = []
    ffn_shift: list[torch.Tensor] = []
    for layer_idx in range(len(packs)):
        recurrent, attn_previous, ffn_previous = owner._layer_state(
            working_cache, layer_idx, hidden_states
        )
        # Public cache is canonical [K,V]. The historical dense token equation
        # evaluates exactly the same state in [V,K] presentation.
        recurrent_vk.append(recurrent.transpose(-1, -2).contiguous())
        attention_shift.append(attn_previous)
        ffn_shift.append(ffn_previous)

    hidden_levels: list[list[torch.Tensor]] | None = None
    if output_hidden_states:
        hidden_levels = [[] for _ in range(len(packs) + 1)]
    final_tokens: list[torch.Tensor] = []

    for token_idx in range(sequence):
        active = attention_mask[:, token_idx].to(
            device=hidden_states.device, dtype=torch.bool
        )
        token = hidden_states[:, token_idx]
        if hidden_levels is not None:
            hidden_levels[0].append(token)
        v_first = torch.zeros(
            batch,
            heads * head_dim,
            device=token.device,
            dtype=token.dtype,
        )

        for layer_idx, pack in enumerate(packs):
            old_token = token
            old_attention = attention_shift[layer_idx]
            old_ffn = ffn_shift[layer_idx]
            old_v_first = v_first
            old_recurrent = recurrent_vk[layer_idx]
            candidate = block_step_batched(
                token,
                old_attention,
                old_ffn,
                v_first,
                old_recurrent,
                *pack,
            )
            candidate_token, candidate_attention, candidate_ffn, candidate_v, candidate_state = candidate
            zero_token = torch.zeros_like(old_token)
            token = _active_where(active, candidate_token, zero_token)
            attention_shift[layer_idx] = _active_where(
                active, candidate_attention, old_attention
            )
            ffn_shift[layer_idx] = _active_where(active, candidate_ffn, old_ffn)
            v_first = _active_where(active, candidate_v, old_v_first)
            recurrent_vk[layer_idx] = _active_where(
                active, candidate_state, old_recurrent
            )
            if hidden_levels is not None and layer_idx + 1 < len(packs):
                hidden_levels[layer_idx + 1].append(token)

        token = owner.norm(token)
        token = torch.where(active.unsqueeze(-1), token, torch.zeros_like(token))
        final_tokens.append(token)
        if hidden_levels is not None:
            hidden_levels[-1].append(token)

    output = torch.stack(final_tokens, dim=1)
    if use_cache:
        for layer_idx in range(len(packs)):
            working_cache.set_layer(
                layer_idx,
                recurrent_vk[layer_idx].transpose(-1, -2).contiguous(),
                attention_shift[layer_idx],
                ffn_shift[layer_idx],
            )
        working_cache.seen_tokens += int(sequence)
        output_cache = working_cache
    else:
        output_cache = None

    collected_hidden = None
    if hidden_levels is not None:
        collected_hidden = tuple(torch.stack(level, dim=1) for level in hidden_levels)

    return {
        "output_kind": "base",
        "last_hidden_state": output,
        "past_key_values": output_cache,
        "hidden_states": collected_hidden,
    }


__all__ = ["IMPLEMENTATION", "run_base_model"]
