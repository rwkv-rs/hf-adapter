# coding=utf-8
"""Optional fused time-mix prototypes for RWKV-7 decode and prefill.

The HF/native_graph fast-token and native-prefill paths currently materialize
six time-mixed attention inputs with separate pointwise torch ops::

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
    def _sequence_shift_mix_fp16_rn(x, previous, mix):
        # PyTorch eager materialises fp16 subtraction, multiplication, and
        # addition separately. Use explicit half2 instructions so LLVM cannot
        # contract the expression into a wider/FMA operation. Keep this helper
        # single-output: Triton 3.2 can abort while lowering multi-output inline
        # assembly on otherwise valid CUDA shapes.
        return tl.inline_asm_elementwise(
            asm="""
            {
                .reg .b32 delta;
                .reg .b32 product;
                sub.rn.f16x2 delta, $2, $1;
                mul.rn.f16x2 product, delta, $3;
                add.rn.f16x2 $0, $1, product;
            }
            """,
            constraints="=r,r,r,r",
            args=[x, previous, mix],
            dtype=tl.float16,
            is_pure=True,
            pack=2,
        )

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

    @triton.jit
    def _attn_sequence_shift_mix_kernel(
        x_ptr,
        initial_ptr,
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
        next_ptr,
        hidden: tl.constexpr,
        tokens: tl.constexpr,
        total: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        STRICT_FP16_ROUNDING: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total
        columns = offsets % hidden
        rows = offsets // hidden
        token_ids = rows % tokens
        batch_ids = rows // tokens
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        seq_prev = tl.load(x_ptr + offsets - hidden, mask=mask & (token_ids > 0), other=0.0)
        initial_prev = tl.load(
            initial_ptr + batch_ids * hidden + columns,
            mask=mask & (token_ids == 0),
            other=0.0,
        )
        previous = seq_prev + initial_prev
        xr = tl.load(xr_mix_ptr + columns, mask=mask, other=0.0)
        xw = tl.load(xw_mix_ptr + columns, mask=mask, other=0.0)
        xk = tl.load(xk_mix_ptr + columns, mask=mask, other=0.0)
        xv = tl.load(xv_mix_ptr + columns, mask=mask, other=0.0)
        xa = tl.load(xa_mix_ptr + columns, mask=mask, other=0.0)
        xg = tl.load(xg_mix_ptr + columns, mask=mask, other=0.0)
        if STRICT_FP16_ROUNDING:
            out_r = _sequence_shift_mix_fp16_rn(x, previous, xr)
            out_w = _sequence_shift_mix_fp16_rn(x, previous, xw)
            out_k = _sequence_shift_mix_fp16_rn(x, previous, xk)
            out_v = _sequence_shift_mix_fp16_rn(x, previous, xv)
            out_a = _sequence_shift_mix_fp16_rn(x, previous, xa)
            out_g = _sequence_shift_mix_fp16_rn(x, previous, xg)
            tl.store(out_r_ptr + offsets, out_r, mask=mask)
            tl.store(out_w_ptr + offsets, out_w, mask=mask)
            tl.store(out_k_ptr + offsets, out_k, mask=mask)
            tl.store(out_v_ptr + offsets, out_v, mask=mask)
            tl.store(out_a_ptr + offsets, out_a, mask=mask)
            tl.store(out_g_ptr + offsets, out_g, mask=mask)
        else:
            delta = previous - x
            tl.store(out_r_ptr + offsets, x + delta * xr, mask=mask)
            tl.store(out_w_ptr + offsets, x + delta * xw, mask=mask)
            tl.store(out_k_ptr + offsets, x + delta * xk, mask=mask)
            tl.store(out_v_ptr + offsets, x + delta * xv, mask=mask)
            tl.store(out_a_ptr + offsets, x + delta * xa, mask=mask)
            tl.store(out_g_ptr + offsets, x + delta * xg, mask=mask)
        tl.store(
            next_ptr + batch_ids * hidden + columns,
            x,
            mask=mask & (token_ids == tokens - 1),
        )

    @triton.jit
    def _ffn_sequence_shift_mix_kernel(
        x_ptr,
        initial_ptr,
        mix_ptr,
        out_ptr,
        next_ptr,
        hidden: tl.constexpr,
        tokens: tl.constexpr,
        total: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        STRICT_FP16_ROUNDING: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total
        columns = offsets % hidden
        rows = offsets // hidden
        token_ids = rows % tokens
        batch_ids = rows // tokens
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        seq_prev = tl.load(x_ptr + offsets - hidden, mask=mask & (token_ids > 0), other=0.0)
        initial_prev = tl.load(
            initial_ptr + batch_ids * hidden + columns,
            mask=mask & (token_ids == 0),
            other=0.0,
        )
        mix = tl.load(mix_ptr + columns, mask=mask, other=0.0)
        previous = seq_prev + initial_prev
        if STRICT_FP16_ROUNDING:
            output = _sequence_shift_mix_fp16_rn(x, previous, mix)
        else:
            output = x + (previous - x) * mix
        tl.store(out_ptr + offsets, output, mask=mask)
        tl.store(
            next_ptr + batch_ids * hidden + columns,
            x,
            mask=mask & (token_ids == tokens - 1),
        )


def fused_attn_shift_mix_available() -> bool:
    """Return whether the optional Triton attention shift-mix prototype can run."""

    return bool(_HAS_TRITON and torch is not None)


def _flatten_hidden_input(x: Any, *, name: str):
    if torch is None:
        raise RuntimeError("fused_attn_shift_mix requires torch")
    if x.dim() == 3:
        return x.reshape(int(x.shape[0]) * int(x.shape[1]), int(x.shape[2])), tuple(x.shape)
    if x.dim() == 2:
        return x, None
    raise ValueError(f"{name} must be shaped [batch, tokens, hidden] or [batch, hidden]")


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

    Inputs may be shaped ``[batch, hidden]`` or ``[batch, tokens, hidden]``. Mix
    vectors may use any shape with ``hidden`` elements, matching FLA weights such
    as ``[1, 1, hidden]``. Returned tensors preserve the input rank/shape.
    """

    if torch is None:
        raise RuntimeError("fused_attn_shift_mix requires torch")
    x2, restore_shape = _flatten_hidden_input(x, name="x")
    prev2, prev_shape = _flatten_hidden_input(prev, name="prev")
    if tuple(x2.shape) != tuple(prev2.shape) or restore_shape != prev_shape:
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
    if restore_shape is not None:
        return tuple(out.reshape(restore_shape) for out in outs)
    return outs


def fused_attn_sequence_shift_mix(
    x: Any,
    initial_prev: Any,
    *mixes: Any,
    block_size: int = 256,
    num_warps: int = 4,
    workspace: Any | None = None,
    strict_fp16_rounding: bool = False,
):
    """Shift-mix a full ``[B,T,C]`` sequence without materialising ``cat``."""

    if torch is None:
        raise RuntimeError("fused_attn_sequence_shift_mix requires torch")
    if not fused_attn_shift_mix_available() or x.dim() != 3 or not x.is_cuda:
        prev = torch.cat([initial_prev.reshape(int(x.shape[0]), 1, int(x.shape[2])), x[:, :-1]], dim=1)
        return (*fused_attn_shift_mix(x, prev, *mixes, force_fallback=True), x[:, -1].contiguous())
    batch, tokens, hidden = map(int, x.shape)
    initial = initial_prev.reshape(batch, hidden)
    flat_mixes = tuple(_flatten_mix(m, hidden, name="mix") for m in mixes)
    if len(flat_mixes) != 6:
        raise ValueError("fused_attn_sequence_shift_mix requires six mix vectors")
    if initial.dtype != x.dtype or any(m.dtype != x.dtype for m in flat_mixes):
        prev = torch.cat([initial.reshape(batch, 1, hidden), x[:, :-1]], dim=1)
        return (*fused_attn_shift_mix(x, prev, *flat_mixes, force_fallback=True), x[:, -1].contiguous())
    source = x.contiguous()
    # Keep the six mixed streams in one allocation.  This reduces allocator
    # traffic and lets the prefill loop reuse one stable sequence workspace.
    expected = (6, batch, tokens, hidden)
    if (
        workspace is not None
        and tuple(workspace.shape) == expected
        and workspace.device == x.device
        and workspace.dtype == x.dtype
        and workspace.is_contiguous()
    ):
        out_storage = workspace
    else:
        out_storage = torch.empty(expected, device=x.device, dtype=x.dtype)
    # Physical order keeps R/K/V adjacent for the optional strided-batched
    # projection, while the returned logical order remains R/W/K/V/A/G.
    slots = tuple(out_storage.unbind(0))
    outs = (slots[0], slots[3], slots[1], slots[2], slots[4], slots[5])
    next_state = torch.empty((batch, hidden), device=x.device, dtype=x.dtype)
    total = int(source.numel())
    _attn_sequence_shift_mix_kernel[(triton.cdiv(total, int(block_size)),)](
        source, initial.contiguous(), *flat_mixes, *outs, next_state,
        hidden,
        tokens,
        total,
        BLOCK_SIZE=int(block_size),
        STRICT_FP16_ROUNDING=bool(strict_fp16_rounding and source.dtype == torch.float16),
        num_warps=int(num_warps),
    )
    return (*outs, next_state)


def fused_ffn_sequence_shift_mix(
    x: Any,
    initial_prev: Any,
    mix: Any,
    block_size: int = 256,
    num_warps: int = 4,
    workspace: Any | None = None,
    strict_fp16_rounding: bool = False,
):
    """Compute the full FFN shift-mix and next state in one launch."""

    if torch is None:
        raise RuntimeError("fused_ffn_sequence_shift_mix requires torch")
    if not fused_attn_shift_mix_available() or x.dim() != 3 or not x.is_cuda:
        batch, _, hidden = map(int, x.shape)
        prev = torch.cat([initial_prev.reshape(batch, 1, hidden), x[:, :-1]], dim=1)
        return x + (prev - x) * mix.reshape(1, 1, hidden), x[:, -1].contiguous()
    batch, tokens, hidden = map(int, x.shape)
    initial = initial_prev.reshape(batch, hidden)
    flat_mix = _flatten_mix(mix, hidden, name="mix")
    if initial.dtype != x.dtype or flat_mix.dtype != x.dtype:
        prev = torch.cat([initial.reshape(batch, 1, hidden), x[:, :-1]], dim=1)
        return x + (prev - x) * flat_mix.reshape(1, 1, hidden), x[:, -1].contiguous()
    source = x.contiguous()
    if (
        workspace is not None
        and tuple(workspace.shape) == tuple(source.shape)
        and workspace.device == source.device
        and workspace.dtype == source.dtype
        and workspace.is_contiguous()
    ):
        out = workspace
    else:
        out = torch.empty_like(source)
    next_state = torch.empty((batch, hidden), device=x.device, dtype=x.dtype)
    total = int(source.numel())
    _ffn_sequence_shift_mix_kernel[(triton.cdiv(total, int(block_size)),)](
        source, initial.contiguous(), flat_mix, out, next_state,
        hidden,
        tokens,
        total,
        BLOCK_SIZE=int(block_size),
        STRICT_FP16_ROUNDING=bool(strict_fp16_rounding and source.dtype == torch.float16),
        num_warps=int(num_warps),
    )
    return out, next_state
