# coding=utf-8
"""Optional fused time-mix prototypes for RWKV-7 decode.

The HF/native_graph fast-token path currently materializes six time-mixed
attention inputs with separate pointwise torch ops::

    xr = x + (prev - x) * x_r
    xw = x + (prev - x) * x_w
    xk = x + (prev - x) * x_k
    xv = x + (prev - x) * x_v
    xa = x + (prev - x) * x_a
    xg = x + (prev - x) * x_g

For decode batch sizes this is launch-bound.  This module keeps the fused
variant optional and dependency-light: it uses one Triton elementwise launch on
CUDA hosts and falls back to the exact torch expression otherwise.  It is a
building block for the fused fp16 backend ladder, not a required runtime
import path for plain HF usage.
"""
from __future__ import annotations

from typing import Any

try:  # pragma: no cover - optional dependency in local no-CUDA tests
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - exercised on CUDA/Triton hosts
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


_HAS_TRITON = triton is not None and tl is not None


if _HAS_TRITON:

    @triton.jit
    def _attn_shift_mix_kernel(
        x_ptr,
        prev_ptr,
        xr_mix_ptr,
        xw_mix_ptr,
        xk_mix_ptr,
        xv_mix_ptr,
        xa_mix_ptr,
        xg_mix_ptr,
        out_r_ptr,
        out_w_ptr,
        out_k_ptr,
        out_v_ptr,
        out_a_ptr,
        out_g_ptr,
        hidden: tl.constexpr,
        total: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total
        h_offsets = offsets % hidden

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        prev = tl.load(prev_ptr + offsets, mask=mask, other=0.0)
        delta = prev - x

        xr = tl.load(xr_mix_ptr + h_offsets, mask=mask, other=0.0)
        xw = tl.load(xw_mix_ptr + h_offsets, mask=mask, other=0.0)
        xk = tl.load(xk_mix_ptr + h_offsets, mask=mask, other=0.0)
        xv = tl.load(xv_mix_ptr + h_offsets, mask=mask, other=0.0)
        xa = tl.load(xa_mix_ptr + h_offsets, mask=mask, other=0.0)
        xg = tl.load(xg_mix_ptr + h_offsets, mask=mask, other=0.0)

        tl.store(out_r_ptr + offsets, x + delta * xr, mask=mask)
        tl.store(out_w_ptr + offsets, x + delta * xw, mask=mask)
        tl.store(out_k_ptr + offsets, x + delta * xk, mask=mask)
        tl.store(out_v_ptr + offsets, x + delta * xv, mask=mask)
        tl.store(out_a_ptr + offsets, x + delta * xa, mask=mask)
        tl.store(out_g_ptr + offsets, x + delta * xg, mask=mask)


def fused_attn_shift_mix_available() -> bool:
    """Return whether the optional Triton attention shift-mix prototype can run."""

    return bool(_HAS_TRITON and torch is not None)


def _flatten_hidden_input(x: Any, *, name: str):
    if torch is None:
        raise RuntimeError("fused_attn_shift_mix requires torch")
    if x.dim() == 3:
        if int(x.shape[1]) != 1:
            raise ValueError(f"{name} must be shaped [batch, 1, hidden] or [batch, hidden]")
        return x.reshape(int(x.shape[0]), int(x.shape[2])), True
    if x.dim() == 2:
        return x, False
    raise ValueError(f"{name} must be shaped [batch, 1, hidden] or [batch, hidden]")


def _flatten_mix(mix: Any, hidden: int, *, name: str):
    if torch is None:
        raise RuntimeError("fused_attn_shift_mix requires torch")
    if int(mix.numel()) != int(hidden):
        raise ValueError(f"{name} must contain hidden={hidden} values; got shape {tuple(mix.shape)}")
    return mix.reshape(hidden)


def _torch_attn_shift_mix(x, prev, x_r, x_w, x_k, x_v, x_a, x_g):
    delta = prev - x
    return (
        torch.addcmul(x, delta, x_r),
        torch.addcmul(x, delta, x_w),
        torch.addcmul(x, delta, x_k),
        torch.addcmul(x, delta, x_v),
        torch.addcmul(x, delta, x_a),
        torch.addcmul(x, delta, x_g),
    )


def fused_attn_shift_mix(
    x: Any,
    prev: Any,
    x_r: Any,
    x_w: Any,
    x_k: Any,
    x_v: Any,
    x_a: Any,
    x_g: Any,
    *,
    block_size: int = 256,
    force_fallback: bool = False,
):
    """Compute all six RWKV-7 attention time-mix inputs in one optional launch.

    Inputs may be shaped ``[batch, hidden]`` or ``[batch, 1, hidden]``. Mix
    vectors may use any shape with ``hidden`` elements, matching FLA weights such
    as ``[1, 1, hidden]``. Returned tensors preserve the input rank.
    """

    if torch is None:
        raise RuntimeError("fused_attn_shift_mix requires torch")
    x2, had_seq = _flatten_hidden_input(x, name="x")
    prev2, _ = _flatten_hidden_input(prev, name="prev")
    if tuple(x2.shape) != tuple(prev2.shape):
        raise ValueError("x and prev must have identical flattened shapes")
    batch, hidden = int(x2.shape[0]), int(x2.shape[1])
    mixes = tuple(
        _flatten_mix(m, hidden, name=n)
        for m, n in (
            (x_r, "x_r"),
            (x_w, "x_w"),
            (x_k, "x_k"),
            (x_v, "x_v"),
            (x_a, "x_a"),
            (x_g, "x_g"),
        )
    )

    use_triton = (
        not force_fallback
        and fused_attn_shift_mix_available()
        and x2.is_cuda
        and prev2.is_cuda
        and all(m.is_cuda for m in mixes)
        and x2.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and prev2.dtype == x2.dtype
        and all(m.dtype == x2.dtype for m in mixes)
    )
    if not use_triton:
        outs = _torch_attn_shift_mix(x2, prev2, *mixes)
    else:
        x_c = x2.contiguous()
        prev_c = prev2.contiguous()
        mixes_c = tuple(m.contiguous() for m in mixes)
        outs = tuple(torch.empty((batch, hidden), device=x2.device, dtype=x2.dtype) for _ in range(6))
        total = int(batch * hidden)
        grid = (triton.cdiv(total, int(block_size)),)
        _attn_shift_mix_kernel[grid](
            x_c,
            prev_c,
            *mixes_c,
            *outs,
            hidden,
            total,
            BLOCK_SIZE=int(block_size),
            num_warps=4,
        )
    if had_seq:
        return tuple(out.unsqueeze(1) for out in outs)
    return outs
