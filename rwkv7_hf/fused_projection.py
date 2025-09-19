# coding=utf-8
"""Optional fused projection prototypes for RWKV-7 decode.

This module is deliberately optional: importing it must not make Triton a hard
runtime dependency for the HF adapter.  The first prototype fuses the three
attention dense projections used in one-token decode:

    xr @ r_proj.T, xk @ k_proj.T, xv @ v_proj.T

into one Triton launch for CUDA tensors.  It is a benchmark/prototype building
block for the FUSED_BACKEND.md ladder, not yet the default model path.

The split-K variant is inspired by Albatross' small-B/T linear kernels: for
decode-sized batches it parallelizes the input dimension into independent
partial sums and then reduces them.  It stays a separate prototype so
GPU/layout-specific tuning can happen without changing the default HF path.
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
    def _rkv_gemv_kernel(
        xr_ptr,
        xk_ptr,
        xv_ptr,
        wr_ptr,
        wk_ptr,
        wv_ptr,
        out_r_ptr,
        out_k_ptr,
        out_v_ptr,
        hidden: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        block_id = tl.program_id(1)
        offs_m = block_id * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_K)
        mask_m = offs_m < hidden

        acc_r = tl.zeros((BLOCK_M,), tl.float32)
        acc_k = tl.zeros((BLOCK_M,), tl.float32)
        acc_v = tl.zeros((BLOCK_M,), tl.float32)

        for start in range(0, hidden, BLOCK_K):
            kidx = start + offs_k
            mask_k = kidx < hidden
            xr = tl.load(xr_ptr + batch_id * hidden + kidx, mask=mask_k, other=0.0)
            xk = tl.load(xk_ptr + batch_id * hidden + kidx, mask=mask_k, other=0.0)
            xv = tl.load(xv_ptr + batch_id * hidden + kidx, mask=mask_k, other=0.0)

            weight_offsets = offs_m[:, None] * hidden + kidx[None, :]
            mask_w = mask_m[:, None] & mask_k[None, :]
            wr = tl.load(wr_ptr + weight_offsets, mask=mask_w, other=0.0)
            wk = tl.load(wk_ptr + weight_offsets, mask=mask_w, other=0.0)
            wv = tl.load(wv_ptr + weight_offsets, mask=mask_w, other=0.0)

            acc_r += tl.sum(wr * xr[None, :], axis=1)
            acc_k += tl.sum(wk * xk[None, :], axis=1)
            acc_v += tl.sum(wv * xv[None, :], axis=1)

        out_base = batch_id * hidden + offs_m
        tl.store(out_r_ptr + out_base, acc_r, mask=mask_m)
        tl.store(out_k_ptr + out_base, acc_k, mask=mask_m)
        tl.store(out_v_ptr + out_base, acc_v, mask=mask_m)

    @triton.jit
    def _rkv_gemv_splitk_partial_kernel(
        xr_ptr,
        xk_ptr,
        xv_ptr,
        wr_ptr,
        wk_ptr,
        wv_ptr,
        partial_ptr,
        hidden: tl.constexpr,
        N_M_BLOCKS: tl.constexpr,
        SPLIT_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        block_id = tl.program_id(1)
        split_id = tl.program_id(2)
        offs_local = tl.arange(0, BLOCK_M)
        offs_m = block_id * BLOCK_M + offs_local
        offs_k = split_id * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_m = offs_m < hidden
        mask_k = offs_k < hidden

        xr = tl.load(xr_ptr + batch_id * hidden + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        xk = tl.load(xk_ptr + batch_id * hidden + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        xv = tl.load(xv_ptr + batch_id * hidden + offs_k, mask=mask_k, other=0.0).to(tl.float32)
        weight_offsets = offs_m[:, None] * hidden + offs_k[None, :]
        mask_w = mask_m[:, None] & mask_k[None, :]
        wr = tl.load(wr_ptr + weight_offsets, mask=mask_w, other=0.0).to(tl.float32)
        wk = tl.load(wk_ptr + weight_offsets, mask=mask_w, other=0.0).to(tl.float32)
        wv = tl.load(wv_ptr + weight_offsets, mask=mask_w, other=0.0).to(tl.float32)

        acc_r = tl.sum(wr * xr[None, :], axis=1)
        acc_k = tl.sum(wk * xk[None, :], axis=1)
        acc_v = tl.sum(wv * xv[None, :], axis=1)

        proj_stride = N_M_BLOCKS * SPLIT_K * BLOCK_M
        base = (batch_id * 3 * proj_stride) + (block_id * SPLIT_K + split_id) * BLOCK_M + offs_local
        tl.store(partial_ptr + base, acc_r, mask=mask_m)
        tl.store(partial_ptr + base + proj_stride, acc_k, mask=mask_m)
        tl.store(partial_ptr + base + 2 * proj_stride, acc_v, mask=mask_m)

    @triton.jit
    def _rkv_gemv_splitk_reduce_kernel(
        partial_ptr,
        out_r_ptr,
        out_k_ptr,
        out_v_ptr,
        hidden: tl.constexpr,
        N_M_BLOCKS: tl.constexpr,
        SPLIT_K: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        block_id = tl.program_id(1)
        offs_local = tl.arange(0, BLOCK_M)
        offs_m = block_id * BLOCK_M + offs_local
        mask_m = offs_m < hidden
        proj_stride = N_M_BLOCKS * SPLIT_K * BLOCK_M
        base = (batch_id * 3 * proj_stride) + block_id * SPLIT_K * BLOCK_M + offs_local

        acc_r = tl.zeros((BLOCK_M,), tl.float32)
        acc_k = tl.zeros((BLOCK_M,), tl.float32)
        acc_v = tl.zeros((BLOCK_M,), tl.float32)
        for split_id in range(0, SPLIT_K):
            split_off = split_id * BLOCK_M
            acc_r += tl.load(partial_ptr + base + split_off, mask=mask_m, other=0.0).to(tl.float32)
            acc_k += tl.load(partial_ptr + base + proj_stride + split_off, mask=mask_m, other=0.0).to(tl.float32)
            acc_v += tl.load(partial_ptr + base + 2 * proj_stride + split_off, mask=mask_m, other=0.0).to(tl.float32)

        out_base = batch_id * hidden + offs_m
        tl.store(out_r_ptr + out_base, acc_r, mask=mask_m)
        tl.store(out_k_ptr + out_base, acc_k, mask=mask_m)
        tl.store(out_v_ptr + out_base, acc_v, mask=mask_m)


def fused_rkv_available() -> bool:
    """Return whether the optional Triton R/K/V projection prototype can run."""

    return bool(_HAS_TRITON and torch is not None)


def fused_rkv_splitk_available() -> bool:
    """Return whether the optional split-K R/K/V projection prototype can run."""

    return bool(_HAS_TRITON and torch is not None)


def _flatten_projection_input(x: Any, *, name: str):
    if torch is None:
        raise RuntimeError("fused_rkv_projection requires torch")
    if x.dim() == 3:
        if int(x.shape[1]) != 1:
            raise ValueError(f"{name} must be shaped [batch, 1, hidden] or [batch, hidden]")
        return x.reshape(int(x.shape[0]), int(x.shape[2])), True
    if x.dim() == 2:
        return x, False
    raise ValueError(f"{name} must be shaped [batch, 1, hidden] or [batch, hidden]")


def _validate_weight(w: Any, hidden: int, *, name: str) -> None:
    if w.dim() != 2 or int(w.shape[0]) != hidden or int(w.shape[1]) != hidden:
        raise ValueError(f"{name} must be a square [hidden, hidden] matrix; got {tuple(w.shape)} expected {hidden}x{hidden}")


def fused_rkv_projection(
    xr: Any,
    xk: Any,
    xv: Any,
    r_weight: Any,
    k_weight: Any,
    v_weight: Any,
    *,
    block_m: int = 16,
    block_k: int = 64,
    force_fallback: bool = False,
):
    """Compute R/K/V projections with one optional Triton launch.

    Args mirror ``torch.nn.functional.linear`` without bias. Inputs may be
    shaped ``[batch, hidden]`` or ``[batch, 1, hidden]``. Outputs preserve the
    same rank as the inputs.

    The function falls back to three torch linear calls when Triton/CUDA is not
    available.  That keeps the prototype safe to import on CPU-only machines and
    old/unsupported GPU stacks.
    """

    if torch is None:
        raise RuntimeError("fused_rkv_projection requires torch")
    xr2, had_seq = _flatten_projection_input(xr, name="xr")
    xk2, _ = _flatten_projection_input(xk, name="xk")
    xv2, _ = _flatten_projection_input(xv, name="xv")
    if tuple(xr2.shape) != tuple(xk2.shape) or tuple(xr2.shape) != tuple(xv2.shape):
        raise ValueError("xr, xk and xv must have identical flattened shapes")
    batch, hidden = int(xr2.shape[0]), int(xr2.shape[1])
    _validate_weight(r_weight, hidden, name="r_weight")
    _validate_weight(k_weight, hidden, name="k_weight")
    _validate_weight(v_weight, hidden, name="v_weight")

    use_triton = (
        not force_fallback
        and fused_rkv_available()
        and xr2.is_cuda
        and xk2.is_cuda
        and xv2.is_cuda
        and r_weight.is_cuda
        and k_weight.is_cuda
        and v_weight.is_cuda
        and xr2.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and r_weight.dtype == xr2.dtype
        and k_weight.dtype == xr2.dtype
        and v_weight.dtype == xr2.dtype
    )
    if not use_triton:
        r = torch.nn.functional.linear(xr2, r_weight)
        k = torch.nn.functional.linear(xk2, k_weight)
        v = torch.nn.functional.linear(xv2, v_weight)
    else:
        xr_c = xr2.contiguous()
        xk_c = xk2.contiguous()
        xv_c = xv2.contiguous()
        wr_c = r_weight.contiguous()
        wk_c = k_weight.contiguous()
        wv_c = v_weight.contiguous()
        r = torch.empty((batch, hidden), device=xr2.device, dtype=xr2.dtype)
        k = torch.empty_like(r)
        v = torch.empty_like(r)
        grid = (batch, triton.cdiv(hidden, int(block_m)))
        _rkv_gemv_kernel[grid](
            xr_c,
            xk_c,
            xv_c,
            wr_c,
            wk_c,
            wv_c,
            r,
            k,
            v,
            hidden,
            BLOCK_M=int(block_m),
            BLOCK_K=int(block_k),
            num_warps=4,
        )
    if had_seq:
        return r.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)
    return r, k, v


def fused_rkv_projection_splitk(
    xr: Any,
    xk: Any,
    xv: Any,
    r_weight: Any,
    k_weight: Any,
    v_weight: Any,
    *,
    block_m: int = 32,
    block_k: int = 128,
    force_fallback: bool = False,
):
    """Compute R/K/V projections with a split-K small-B Triton prototype.

    This mirrors ``fused_rkv_projection`` but uses two launches: one computes
    fp32 partial sums over K chunks and one reduces them.  The extra launch is
    meant for decode-like small batches where Albatross-style K parallelism can
    expose more GPU work than a single serial-K GEMV block.
    """

    if torch is None:
        raise RuntimeError("fused_rkv_projection_splitk requires torch")
    xr2, had_seq = _flatten_projection_input(xr, name="xr")
    xk2, _ = _flatten_projection_input(xk, name="xk")
    xv2, _ = _flatten_projection_input(xv, name="xv")
    if tuple(xr2.shape) != tuple(xk2.shape) or tuple(xr2.shape) != tuple(xv2.shape):
        raise ValueError("xr, xk and xv must have identical flattened shapes")
    batch, hidden = int(xr2.shape[0]), int(xr2.shape[1])
    _validate_weight(r_weight, hidden, name="r_weight")
    _validate_weight(k_weight, hidden, name="k_weight")
    _validate_weight(v_weight, hidden, name="v_weight")
    block_m_i = int(block_m)
    block_k_i = int(block_k)
    if block_m_i <= 0 or block_k_i <= 0:
        raise ValueError(f"block_m and block_k must be positive; got {block_m}, {block_k}")

    use_triton = (
        not force_fallback
        and fused_rkv_splitk_available()
        and xr2.is_cuda
        and xk2.is_cuda
        and xv2.is_cuda
        and r_weight.is_cuda
        and k_weight.is_cuda
        and v_weight.is_cuda
        and xr2.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and r_weight.dtype == xr2.dtype
        and k_weight.dtype == xr2.dtype
        and v_weight.dtype == xr2.dtype
    )
    if not use_triton:
        r = torch.nn.functional.linear(xr2, r_weight)
        k = torch.nn.functional.linear(xk2, k_weight)
        v = torch.nn.functional.linear(xv2, v_weight)
    else:
        xr_c = xr2.contiguous()
        xk_c = xk2.contiguous()
        xv_c = xv2.contiguous()
        wr_c = r_weight.contiguous()
        wk_c = k_weight.contiguous()
        wv_c = v_weight.contiguous()
        r = torch.empty((batch, hidden), device=xr2.device, dtype=xr2.dtype)
        k = torch.empty_like(r)
        v = torch.empty_like(r)
        n_m_blocks = triton.cdiv(hidden, block_m_i)
        split_k = triton.cdiv(hidden, block_k_i)
        partial = torch.empty((batch, 3, n_m_blocks, split_k, block_m_i), device=xr2.device, dtype=torch.float32)
        _rkv_gemv_splitk_partial_kernel[(batch, n_m_blocks, split_k)](
            xr_c,
            xk_c,
            xv_c,
            wr_c,
            wk_c,
            wv_c,
            partial,
            hidden,
            N_M_BLOCKS=n_m_blocks,
            SPLIT_K=split_k,
            BLOCK_M=block_m_i,
            BLOCK_K=block_k_i,
            num_warps=4,
        )
        _rkv_gemv_splitk_reduce_kernel[(batch, n_m_blocks)](
            partial,
            r,
            k,
            v,
            hidden,
            N_M_BLOCKS=n_m_blocks,
            SPLIT_K=split_k,
            BLOCK_M=block_m_i,
            num_warps=1,
        )
    if had_seq:
        return r.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)
    return r, k, v
