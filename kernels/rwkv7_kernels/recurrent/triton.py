"""Experimental fused FP16 recurrent scan behind the v1 kernel protocol.

The public model and cache retain canonical ``[B,H,K,V]`` state. This module
only replaces the recurrent operator and deliberately has no model, cache, or
hardware-routing policy. It is opt-in until the full model and lm_eval gates
match the exact CUDA-graph backend on every release device.
"""
from __future__ import annotations

from typing import Any

import torch

try:  # optional companion dependency
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - hosts without Triton
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


IMPLEMENTATION = "native-triton-rank1-scan-v1"
_HAS_TRITON = triton is not None and tl is not None


if _HAS_TRITON:

    @triton.jit
    def _rank1_scan_kernel(
        r_ptr,
        w_ptr,
        k_ptr,
        v_ptr,
        a_ptr,
        b_ptr,
        state_ptr,
        mask_ptr,
        output_ptr,
        final_state_ptr,
        T,
        H: tl.constexpr,
        N: tl.constexpr,
        ROW_BLOCKS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        HAS_MASK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        row_block = pid % ROW_BLOCKS
        bh = pid // ROW_BLOCKS
        batch = bh // H
        head = bh % H

        # Work in the official [V,K] presentation while loading and storing
        # the public cache's physical [K,V] layout.
        offs_v = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, N)
        mask_v = offs_v < N
        state_base = bh * N * N
        state_offsets = state_base + offs_k[None, :] * N + offs_v[:, None]
        state_vk = tl.load(
            state_ptr + state_offsets,
            mask=mask_v[:, None],
            other=0.0,
        ).to(tl.float32)

        token = 0
        while token < T:
            vector_base = ((batch * T + token) * H + head) * N
            r = tl.load(r_ptr + vector_base + offs_k).to(tl.float32)
            w = tl.load(w_ptr + vector_base + offs_k).to(tl.float32)
            key = tl.load(k_ptr + vector_base + offs_k).to(tl.float32)
            value = tl.load(
                v_ptr + vector_base + offs_v, mask=mask_v, other=0.0
            ).to(tl.float32)
            factor_a = tl.load(a_ptr + vector_base + offs_k).to(tl.float32)
            factor_b = tl.load(b_ptr + vector_base + offs_k).to(tl.float32)

            state_dot_a = tl.sum(state_vk * factor_a[None, :], axis=1)
            candidate = (
                state_vk * w[None, :]
                + state_dot_a[:, None] * factor_b[None, :]
                + value[:, None] * key[None, :]
            )
            active = True
            if HAS_MASK:
                active = tl.load(mask_ptr + batch * T + token).to(tl.int1)
            state_vk = tl.where(active, candidate, state_vk)

            # The official path casts the recurrent candidate back to model
            # dtype before its readout matmul. The v1 Triton lane is FP16.
            recurrent = tl.sum(
                candidate.to(tl.float16) * r.to(tl.float16)[None, :], axis=1
            )
            recurrent = tl.where(active & mask_v, recurrent, 0.0)
            tl.store(
                output_ptr + vector_base + offs_v,
                recurrent,
                mask=mask_v,
            )
            token += 1

        tl.store(
            final_state_ptr + state_offsets,
            state_vk,
            mask=mask_v[:, None],
        )


def _unsupported(reason: str) -> dict[str, Any]:
    return {
        "supported": False,
        "implementation": IMPLEMENTATION,
        "reason": reason,
    }


def probe_recurrent_v1(
    receptance,
    decay,
    key,
    value,
    a,
    b,
    initial_state,
    attention_mask,
) -> dict[str, Any]:
    tensors = (receptance, decay, key, value, a, b, initial_state)
    if not _HAS_TRITON:
        return _unsupported("Triton is unavailable")
    if not torch.cuda.is_available() or not all(item.is_cuda for item in tensors):
        return _unsupported("the Triton scan requires CUDA tensors")
    if any(item.requires_grad for item in tensors):
        return _unsupported("the Triton scan is inference-only")
    if receptance.ndim != 4 or initial_state.ndim != 4:
        return _unsupported("rank-four recurrent inputs and state are required")
    expected = tuple(receptance.shape)
    if any(tuple(item.shape) != expected for item in (decay, key, value, a, b)):
        return _unsupported("all recurrent vectors must have identical shapes")
    batch, _, heads, key_dim = expected
    if key_dim != 64 or tuple(initial_state.shape) != (batch, heads, 64, 64):
        return _unsupported("the promoted Triton lane requires K=V=64")
    if receptance.dtype != torch.float16:
        return _unsupported(f"unsupported input dtype {receptance.dtype}")
    if any(item.dtype != torch.float16 for item in (key, value, a, b)):
        return _unsupported("r/k/v/a/b must all use FP16")
    if decay.dtype not in (torch.float16, torch.float32):
        return _unsupported("decay must use FP16 or FP32")
    if initial_state.dtype != torch.float32:
        return _unsupported("the canonical recurrent state must use FP32")
    if attention_mask is not None:
        if tuple(attention_mask.shape) != expected[:2]:
            return _unsupported("attention_mask must be shaped [B,T]")
        if not attention_mask.is_cuda:
            return _unsupported("attention_mask must be on CUDA")
    return {
        "supported": True,
        "implementation": IMPLEMENTATION,
        "reason": "FP16 K=V=64 inference request is supported",
    }


def recurrent_v1(
    receptance,
    decay,
    key,
    value,
    a,
    b,
    initial_state,
    attention_mask,
):
    support = probe_recurrent_v1(
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
    batch, tokens, heads, head_dim = receptance.shape
    block_m = 16
    row_blocks = triton.cdiv(head_dim, block_m)
    output = torch.empty_like(value)
    final_state = torch.empty_like(initial_state)
    contiguous = tuple(
        item.contiguous()
        for item in (receptance, decay, key, value, a, b, initial_state)
    )
    mask = None if attention_mask is None else attention_mask.bool().contiguous()
    mask_pointer = initial_state if mask is None else mask
    _rank1_scan_kernel[(batch * heads * row_blocks,)](
        *contiguous,
        mask_pointer,
        output,
        final_state,
        tokens,
        heads,
        head_dim,
        ROW_BLOCKS=row_blocks,
        BLOCK_M=block_m,
        HAS_MASK=mask is not None,
        num_warps=4,
    )
    return output, final_state


__all__ = ["probe_recurrent_v1", "recurrent_v1"]
