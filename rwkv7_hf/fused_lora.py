# coding=utf-8
"""Optional fused LoRA projection prototypes for RWKV-7 decode.

The first target is the W/A LoRA pair in RWKV-7 attention.  Both modules share
rank and shape on current checkpoints and sit in the largest decode component
(`attn_linears_lora`).  The prototype keeps the HF path unchanged and provides
telemetry for replacing multiple small GEMV launches with two grouped Triton
kernels: a fused down/activation pass and a fused up/bias pass.
"""
from __future__ import annotations

from typing import Any

try:  # pragma: no cover - optional dependency in local no-CUDA tests
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

try:  # pragma: no cover - exercised on CUDA/Triton hosts
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


_HAS_TRITON = triton is not None and tl is not None


if _HAS_TRITON:

    @triton.jit
    def _wa_lora_down_kernel(
        xw_ptr,
        xa_ptr,
        w_down_ptr,
        a_down_ptr,
        w_mid_ptr,
        a_mid_ptr,
        hidden: tl.constexpr,
        rank: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        block_id = tl.program_id(1)
        offs_r = block_id * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_k = tl.arange(0, BLOCK_K)
        mask_r = offs_r < rank

        acc_w = tl.zeros((BLOCK_R,), tl.float32)
        acc_a = tl.zeros((BLOCK_R,), tl.float32)
        for start in range(0, hidden, BLOCK_K):
            kidx = start + offs_k
            mask_k = kidx < hidden
            xw = tl.load(xw_ptr + batch_id * hidden + kidx, mask=mask_k, other=0.0).to(tl.float32)
            xa = tl.load(xa_ptr + batch_id * hidden + kidx, mask=mask_k, other=0.0).to(tl.float32)
            w_offsets = offs_r[:, None] * hidden + kidx[None, :]
            a_offsets = offs_r[:, None] * hidden + kidx[None, :]
            wd = tl.load(w_down_ptr + w_offsets, mask=mask_r[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
            ad = tl.load(a_down_ptr + a_offsets, mask=mask_r[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
            acc_w += tl.sum(wd * xw[None, :], axis=1)
            acc_a += tl.sum(ad * xa[None, :], axis=1)

        # W LoRA activation is tanh.  Use sigmoid identity to avoid relying on
        # backend-specific tl.tanh availability.
        w_act = 2.0 * tl.sigmoid(2.0 * acc_w) - 1.0
        tl.store(w_mid_ptr + batch_id * rank + offs_r, w_act, mask=mask_r)
        tl.store(a_mid_ptr + batch_id * rank + offs_r, acc_a, mask=mask_r)

    @triton.jit
    def _wa_lora_up_kernel(
        w_mid_ptr,
        a_mid_ptr,
        w_up_ptr,
        a_up_ptr,
        w_bias_ptr,
        a_bias_ptr,
        w_out_ptr,
        a_out_ptr,
        hidden: tl.constexpr,
        rank: tl.constexpr,
        HAS_W_BIAS: tl.constexpr,
        HAS_A_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        block_id = tl.program_id(1)
        offs_m = block_id * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_r = tl.arange(0, BLOCK_R)
        mask_m = offs_m < hidden

        acc_w = tl.zeros((BLOCK_M,), tl.float32)
        acc_a = tl.zeros((BLOCK_M,), tl.float32)
        for start in range(0, rank, BLOCK_R):
            ridx = start + offs_r
            mask_r = ridx < rank
            wm = tl.load(w_mid_ptr + batch_id * rank + ridx, mask=mask_r, other=0.0).to(tl.float32)
            am = tl.load(a_mid_ptr + batch_id * rank + ridx, mask=mask_r, other=0.0).to(tl.float32)
            w_offsets = offs_m[:, None] * rank + ridx[None, :]
            a_offsets = offs_m[:, None] * rank + ridx[None, :]
            wu = tl.load(w_up_ptr + w_offsets, mask=mask_m[:, None] & mask_r[None, :], other=0.0).to(tl.float32)
            au = tl.load(a_up_ptr + a_offsets, mask=mask_m[:, None] & mask_r[None, :], other=0.0).to(tl.float32)
            acc_w += tl.sum(wu * wm[None, :], axis=1)
            acc_a += tl.sum(au * am[None, :], axis=1)

        if HAS_W_BIAS:
            wb = tl.load(w_bias_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
            acc_w += wb
        if HAS_A_BIAS:
            ab = tl.load(a_bias_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
            acc_a += ab
        out_base = batch_id * hidden + offs_m
        tl.store(w_out_ptr + out_base, acc_w, mask=mask_m)
        tl.store(a_out_ptr + out_base, acc_a, mask=mask_m)


def fused_wa_lora_available() -> bool:
    """Return whether the optional Triton W/A LoRA prototype can run."""

    return bool(_HAS_TRITON and torch is not None)


def _flatten_lora_input(x: Any, hidden: int | None = None, *, name: str):
    if torch is None:
        raise RuntimeError("fused_wa_lora requires torch")
    if x.dim() == 3:
        if int(x.shape[1]) != 1:
            raise ValueError(f"{name} must be [batch, 1, hidden] or [batch, hidden], got {tuple(x.shape)}")
        if hidden is not None and int(x.shape[2]) != hidden:
            raise ValueError(f"{name} hidden mismatch: got {int(x.shape[2])}, expected {hidden}")
        return x.reshape(int(x.shape[0]), int(x.shape[2])), True
    if x.dim() == 2:
        if hidden is not None and int(x.shape[1]) != hidden:
            raise ValueError(f"{name} hidden mismatch: got {int(x.shape[1])}, expected {hidden}")
        return x, False
    raise ValueError(f"{name} must be [batch, 1, hidden] or [batch, hidden]")


def fused_wa_lora(
    xw: Any,
    xa: Any,
    w_down_weight: Any,
    a_down_weight: Any,
    w_up_weight: Any,
    a_up_weight: Any,
    w_up_bias: Any | None = None,
    a_up_bias: Any | None = None,
    *,
    block_m: int = 16,
    block_r: int = 64,
    block_k: int = 64,
    force_fallback: bool = False,
):
    """Compute RWKV W/A LoRA outputs with grouped optional Triton kernels.

    W LoRA uses tanh after the down projection.  A LoRA uses identity after the
    down projection.  The returned tensors match the raw module outputs; caller
    still applies the outer sigmoid/scaling used by RWKV attention.
    """

    if torch is None or F is None:
        raise RuntimeError("fused_wa_lora requires torch")
    xw2, had_seq = _flatten_lora_input(xw, name="xw")
    hidden = int(xw2.shape[1])
    xa2, _ = _flatten_lora_input(xa, hidden, name="xa")
    if tuple(xw2.shape) != tuple(xa2.shape):
        raise ValueError("xw and xa must have identical flattened shapes")
    if w_down_weight.dim() != 2 or a_down_weight.dim() != 2:
        raise ValueError("down weights must be [rank, hidden]")
    rank = int(w_down_weight.shape[0])
    if int(w_down_weight.shape[1]) != hidden or int(a_down_weight.shape[0]) != rank or int(a_down_weight.shape[1]) != hidden:
        raise ValueError("w/a down weights must share [rank, hidden] shape")
    if w_up_weight.dim() != 2 or a_up_weight.dim() != 2:
        raise ValueError("up weights must be [hidden, rank]")
    if int(w_up_weight.shape[0]) != hidden or int(w_up_weight.shape[1]) != rank:
        raise ValueError(f"w_up_weight must be [{hidden}, {rank}], got {tuple(w_up_weight.shape)}")
    if int(a_up_weight.shape[0]) != hidden or int(a_up_weight.shape[1]) != rank:
        raise ValueError(f"a_up_weight must be [{hidden}, {rank}], got {tuple(a_up_weight.shape)}")
    if w_up_bias is not None and (w_up_bias.dim() != 1 or int(w_up_bias.shape[0]) != hidden):
        raise ValueError(f"w_up_bias must be [{hidden}], got {tuple(w_up_bias.shape)}")
    if a_up_bias is not None and (a_up_bias.dim() != 1 or int(a_up_bias.shape[0]) != hidden):
        raise ValueError(f"a_up_bias must be [{hidden}], got {tuple(a_up_bias.shape)}")

    use_triton = (
        not force_fallback
        and fused_wa_lora_available()
        and xw2.is_cuda
        and xa2.is_cuda
        and w_down_weight.is_cuda
        and a_down_weight.is_cuda
        and w_up_weight.is_cuda
        and a_up_weight.is_cuda
        and (w_up_bias is None or w_up_bias.is_cuda)
        and (a_up_bias is None or a_up_bias.is_cuda)
        and xw2.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and w_down_weight.dtype == xw2.dtype
        and a_down_weight.dtype == xw2.dtype
        and w_up_weight.dtype == xw2.dtype
        and a_up_weight.dtype == xw2.dtype
    )
    if not use_triton:
        wh = torch.tanh(F.linear(xw2, w_down_weight))
        ah = F.linear(xa2, a_down_weight)
        w_out = F.linear(wh, w_up_weight, w_up_bias)
        a_out = F.linear(ah, a_up_weight, a_up_bias)
    else:
        batch = int(xw2.shape[0])
        xw_c = xw2.contiguous()
        xa_c = xa2.contiguous()
        wd_c = w_down_weight.contiguous()
        ad_c = a_down_weight.contiguous()
        wu_c = w_up_weight.contiguous()
        au_c = a_up_weight.contiguous()
        wb_c = w_up_bias.contiguous() if w_up_bias is not None else wu_c
        ab_c = a_up_bias.contiguous() if a_up_bias is not None else au_c
        w_mid = torch.empty((batch, rank), device=xw2.device, dtype=xw2.dtype)
        a_mid = torch.empty_like(w_mid)
        w_out = torch.empty((batch, hidden), device=xw2.device, dtype=xw2.dtype)
        a_out = torch.empty_like(w_out)
        _wa_lora_down_kernel[(batch, triton.cdiv(rank, int(block_r)))](
            xw_c,
            xa_c,
            wd_c,
            ad_c,
            w_mid,
            a_mid,
            hidden,
            rank,
            BLOCK_R=int(block_r),
            BLOCK_K=int(block_k),
            num_warps=4,
        )
        _wa_lora_up_kernel[(batch, triton.cdiv(hidden, int(block_m)))](
            w_mid,
            a_mid,
            wu_c,
            au_c,
            wb_c,
            ab_c,
            w_out,
            a_out,
            hidden,
            rank,
            HAS_W_BIAS=w_up_bias is not None,
            HAS_A_BIAS=a_up_bias is not None,
            BLOCK_M=int(block_m),
            BLOCK_R=int(block_r),
            num_warps=4,
        )
    if had_seq:
        return w_out.unsqueeze(1), a_out.unsqueeze(1)
    return w_out, a_out
