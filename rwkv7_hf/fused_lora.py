# coding=utf-8
"""Optional fused LoRA projection prototypes for RWKV-7 decode.

The first target is the W/A LoRA pair in RWKV-7 attention.  Both modules share
rank and shape on current checkpoints and sit in the largest decode component
(`attn_linears_lora`).  The prototype keeps the HF path unchanged and provides
telemetry for replacing multiple small GEMV launches with two grouped Triton
kernels: a fused down/activation pass and a fused up/bias pass.

Later probes extend the group to W/A/G and W/A/G/V-gate.  They intentionally
remain optional telemetry until end-to-end native_graph measurements show they
help the production HF fast-token path.
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

    @triton.jit
    def _wag_lora_down_kernel(
        xw_ptr,
        xa_ptr,
        xg_ptr,
        w_down_ptr,
        a_down_ptr,
        g_down_ptr,
        w_mid_ptr,
        a_mid_ptr,
        g_mid_ptr,
        hidden: tl.constexpr,
        w_rank: tl.constexpr,
        a_rank: tl.constexpr,
        g_rank: tl.constexpr,
        max_rank: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        block_id = tl.program_id(1)
        offs_r = block_id * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_k = tl.arange(0, BLOCK_K)
        mask_w_r = offs_r < w_rank
        mask_a_r = offs_r < a_rank
        mask_g_r = offs_r < g_rank

        acc_w = tl.zeros((BLOCK_R,), tl.float32)
        acc_a = tl.zeros((BLOCK_R,), tl.float32)
        acc_g = tl.zeros((BLOCK_R,), tl.float32)
        for start in range(0, hidden, BLOCK_K):
            kidx = start + offs_k
            mask_k = kidx < hidden
            xw = tl.load(xw_ptr + batch_id * hidden + kidx, mask=mask_k, other=0.0).to(tl.float32)
            xa = tl.load(xa_ptr + batch_id * hidden + kidx, mask=mask_k, other=0.0).to(tl.float32)
            xg = tl.load(xg_ptr + batch_id * hidden + kidx, mask=mask_k, other=0.0).to(tl.float32)
            w_offsets = offs_r[:, None] * hidden + kidx[None, :]
            a_offsets = offs_r[:, None] * hidden + kidx[None, :]
            g_offsets = offs_r[:, None] * hidden + kidx[None, :]
            wd = tl.load(w_down_ptr + w_offsets, mask=mask_w_r[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
            ad = tl.load(a_down_ptr + a_offsets, mask=mask_a_r[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
            gd = tl.load(g_down_ptr + g_offsets, mask=mask_g_r[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
            acc_w += tl.sum(wd * xw[None, :], axis=1)
            acc_a += tl.sum(ad * xa[None, :], axis=1)
            acc_g += tl.sum(gd * xg[None, :], axis=1)

        # W uses tanh, A uses identity, G uses sigmoid in current RWKV-7 LoRA.
        w_act = 2.0 * tl.sigmoid(2.0 * acc_w) - 1.0
        g_act = tl.sigmoid(acc_g)
        tl.store(w_mid_ptr + batch_id * w_rank + offs_r, w_act, mask=mask_w_r)
        tl.store(a_mid_ptr + batch_id * a_rank + offs_r, acc_a, mask=mask_a_r)
        tl.store(g_mid_ptr + batch_id * g_rank + offs_r, g_act, mask=mask_g_r)

    @triton.jit
    def _wag_lora_up_kernel(
        w_mid_ptr,
        a_mid_ptr,
        g_mid_ptr,
        w_up_ptr,
        a_up_ptr,
        g_up_ptr,
        w_bias_ptr,
        a_bias_ptr,
        g_bias_ptr,
        w_out_ptr,
        a_out_ptr,
        g_out_ptr,
        hidden: tl.constexpr,
        w_rank: tl.constexpr,
        a_rank: tl.constexpr,
        g_rank: tl.constexpr,
        max_rank: tl.constexpr,
        HAS_W_BIAS: tl.constexpr,
        HAS_A_BIAS: tl.constexpr,
        HAS_G_BIAS: tl.constexpr,
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
        acc_g = tl.zeros((BLOCK_M,), tl.float32)
        for start in range(0, max_rank, BLOCK_R):
            ridx = start + offs_r
            mask_w_r = ridx < w_rank
            mask_a_r = ridx < a_rank
            mask_g_r = ridx < g_rank
            wm = tl.load(w_mid_ptr + batch_id * w_rank + ridx, mask=mask_w_r, other=0.0).to(tl.float32)
            am = tl.load(a_mid_ptr + batch_id * a_rank + ridx, mask=mask_a_r, other=0.0).to(tl.float32)
            gm = tl.load(g_mid_ptr + batch_id * g_rank + ridx, mask=mask_g_r, other=0.0).to(tl.float32)
            w_offsets = offs_m[:, None] * w_rank + ridx[None, :]
            a_offsets = offs_m[:, None] * a_rank + ridx[None, :]
            g_offsets = offs_m[:, None] * g_rank + ridx[None, :]
            wu = tl.load(w_up_ptr + w_offsets, mask=mask_m[:, None] & mask_w_r[None, :], other=0.0).to(tl.float32)
            au = tl.load(a_up_ptr + a_offsets, mask=mask_m[:, None] & mask_a_r[None, :], other=0.0).to(tl.float32)
            gu = tl.load(g_up_ptr + g_offsets, mask=mask_m[:, None] & mask_g_r[None, :], other=0.0).to(tl.float32)
            acc_w += tl.sum(wu * wm[None, :], axis=1)
            acc_a += tl.sum(au * am[None, :], axis=1)
            acc_g += tl.sum(gu * gm[None, :], axis=1)

        if HAS_W_BIAS:
            wb = tl.load(w_bias_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
            acc_w += wb
        if HAS_A_BIAS:
            ab = tl.load(a_bias_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
            acc_a += ab
        if HAS_G_BIAS:
            gb = tl.load(g_bias_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
            acc_g += gb
        out_base = batch_id * hidden + offs_m
        tl.store(w_out_ptr + out_base, acc_w, mask=mask_m)
        tl.store(a_out_ptr + out_base, acc_a, mask=mask_m)
        tl.store(g_out_ptr + out_base, acc_g, mask=mask_m)

    @triton.jit
    def _wavg_lora_down_kernel(
        xw_ptr,
        xa_ptr,
        xg_ptr,
        xv_ptr,
        w_down_ptr,
        a_down_ptr,
        g_down_ptr,
        v_down_ptr,
        w_mid_ptr,
        a_mid_ptr,
        g_mid_ptr,
        v_mid_ptr,
        hidden: tl.constexpr,
        w_rank: tl.constexpr,
        a_rank: tl.constexpr,
        g_rank: tl.constexpr,
        v_rank: tl.constexpr,
        max_rank: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        block_id = tl.program_id(1)
        offs_r = block_id * BLOCK_R + tl.arange(0, BLOCK_R)
        offs_k = tl.arange(0, BLOCK_K)
        mask_w_r = offs_r < w_rank
        mask_a_r = offs_r < a_rank
        mask_g_r = offs_r < g_rank
        mask_v_r = offs_r < v_rank

        acc_w = tl.zeros((BLOCK_R,), tl.float32)
        acc_a = tl.zeros((BLOCK_R,), tl.float32)
        acc_g = tl.zeros((BLOCK_R,), tl.float32)
        acc_v = tl.zeros((BLOCK_R,), tl.float32)
        for start in range(0, hidden, BLOCK_K):
            kidx = start + offs_k
            mask_k = kidx < hidden
            xw = tl.load(xw_ptr + batch_id * hidden + kidx, mask=mask_k, other=0.0).to(tl.float32)
            xa = tl.load(xa_ptr + batch_id * hidden + kidx, mask=mask_k, other=0.0).to(tl.float32)
            xg = tl.load(xg_ptr + batch_id * hidden + kidx, mask=mask_k, other=0.0).to(tl.float32)
            xv = tl.load(xv_ptr + batch_id * hidden + kidx, mask=mask_k, other=0.0).to(tl.float32)
            offsets = offs_r[:, None] * hidden + kidx[None, :]
            wd = tl.load(w_down_ptr + offsets, mask=mask_w_r[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
            ad = tl.load(a_down_ptr + offsets, mask=mask_a_r[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
            gd = tl.load(g_down_ptr + offsets, mask=mask_g_r[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
            vd = tl.load(v_down_ptr + offsets, mask=mask_v_r[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
            acc_w += tl.sum(wd * xw[None, :], axis=1)
            acc_a += tl.sum(ad * xa[None, :], axis=1)
            acc_g += tl.sum(gd * xg[None, :], axis=1)
            acc_v += tl.sum(vd * xv[None, :], axis=1)

        # W uses tanh, A and V-gate down use identity, G uses sigmoid.
        w_act = 2.0 * tl.sigmoid(2.0 * acc_w) - 1.0
        g_act = tl.sigmoid(acc_g)
        tl.store(w_mid_ptr + batch_id * w_rank + offs_r, w_act, mask=mask_w_r)
        tl.store(a_mid_ptr + batch_id * a_rank + offs_r, acc_a, mask=mask_a_r)
        tl.store(g_mid_ptr + batch_id * g_rank + offs_r, g_act, mask=mask_g_r)
        tl.store(v_mid_ptr + batch_id * v_rank + offs_r, acc_v, mask=mask_v_r)

    @triton.jit
    def _wavg_lora_up_kernel(
        w_mid_ptr,
        a_mid_ptr,
        g_mid_ptr,
        v_mid_ptr,
        w_up_ptr,
        a_up_ptr,
        g_up_ptr,
        v_up_ptr,
        w_bias_ptr,
        a_bias_ptr,
        g_bias_ptr,
        v_bias_ptr,
        w_out_ptr,
        a_out_ptr,
        g_out_ptr,
        v_gate_ptr,
        hidden: tl.constexpr,
        w_rank: tl.constexpr,
        a_rank: tl.constexpr,
        g_rank: tl.constexpr,
        v_rank: tl.constexpr,
        max_rank: tl.constexpr,
        HAS_W_BIAS: tl.constexpr,
        HAS_A_BIAS: tl.constexpr,
        HAS_G_BIAS: tl.constexpr,
        HAS_V_BIAS: tl.constexpr,
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
        acc_g = tl.zeros((BLOCK_M,), tl.float32)
        acc_v = tl.zeros((BLOCK_M,), tl.float32)
        for start in range(0, max_rank, BLOCK_R):
            ridx = start + offs_r
            mask_w_r = ridx < w_rank
            mask_a_r = ridx < a_rank
            mask_g_r = ridx < g_rank
            mask_v_r = ridx < v_rank
            wm = tl.load(w_mid_ptr + batch_id * w_rank + ridx, mask=mask_w_r, other=0.0).to(tl.float32)
            am = tl.load(a_mid_ptr + batch_id * a_rank + ridx, mask=mask_a_r, other=0.0).to(tl.float32)
            gm = tl.load(g_mid_ptr + batch_id * g_rank + ridx, mask=mask_g_r, other=0.0).to(tl.float32)
            vm = tl.load(v_mid_ptr + batch_id * v_rank + ridx, mask=mask_v_r, other=0.0).to(tl.float32)
            w_offsets = offs_m[:, None] * w_rank + ridx[None, :]
            a_offsets = offs_m[:, None] * a_rank + ridx[None, :]
            g_offsets = offs_m[:, None] * g_rank + ridx[None, :]
            v_offsets = offs_m[:, None] * v_rank + ridx[None, :]
            wu = tl.load(w_up_ptr + w_offsets, mask=mask_m[:, None] & mask_w_r[None, :], other=0.0).to(tl.float32)
            au = tl.load(a_up_ptr + a_offsets, mask=mask_m[:, None] & mask_a_r[None, :], other=0.0).to(tl.float32)
            gu = tl.load(g_up_ptr + g_offsets, mask=mask_m[:, None] & mask_g_r[None, :], other=0.0).to(tl.float32)
            vu = tl.load(v_up_ptr + v_offsets, mask=mask_m[:, None] & mask_v_r[None, :], other=0.0).to(tl.float32)
            acc_w += tl.sum(wu * wm[None, :], axis=1)
            acc_a += tl.sum(au * am[None, :], axis=1)
            acc_g += tl.sum(gu * gm[None, :], axis=1)
            acc_v += tl.sum(vu * vm[None, :], axis=1)

        if HAS_W_BIAS:
            wb = tl.load(w_bias_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
            acc_w += wb
        if HAS_A_BIAS:
            ab = tl.load(a_bias_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
            acc_a += ab
        if HAS_G_BIAS:
            gb = tl.load(g_bias_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
            acc_g += gb
        if HAS_V_BIAS:
            vb = tl.load(v_bias_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
            acc_v += vb
        out_base = batch_id * hidden + offs_m
        tl.store(w_out_ptr + out_base, acc_w, mask=mask_m)
        tl.store(a_out_ptr + out_base, acc_a, mask=mask_m)
        tl.store(g_out_ptr + out_base, acc_g, mask=mask_m)
        tl.store(v_gate_ptr + out_base, tl.sigmoid(acc_v), mask=mask_m)


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


def fused_wag_lora_available() -> bool:
    """Return whether the optional Triton W/A/G LoRA prototype can run."""

    return bool(_HAS_TRITON and torch is not None)


def fused_wavg_lora_available() -> bool:
    """Return whether the optional Triton W/A/G/V-gate LoRA prototype can run."""

    return bool(_HAS_TRITON and torch is not None)


def _validate_lora_weights(
    down_weight: Any,
    up_weight: Any,
    up_bias: Any | None,
    hidden: int,
    *,
    name: str,
) -> int:
    if down_weight.dim() != 2:
        raise ValueError(f"{name}_down_weight must be [rank, hidden], got {tuple(down_weight.shape)}")
    rank = int(down_weight.shape[0])
    if int(down_weight.shape[1]) != hidden:
        raise ValueError(f"{name}_down_weight hidden mismatch: got {int(down_weight.shape[1])}, expected {hidden}")
    if up_weight.dim() != 2 or int(up_weight.shape[0]) != hidden or int(up_weight.shape[1]) != rank:
        raise ValueError(f"{name}_up_weight must be [{hidden}, {rank}], got {tuple(up_weight.shape)}")
    if up_bias is not None and (up_bias.dim() != 1 or int(up_bias.shape[0]) != hidden):
        raise ValueError(f"{name}_up_bias must be [{hidden}], got {tuple(up_bias.shape)}")
    return rank


def fused_wag_lora(
    xw: Any,
    xa: Any,
    xg: Any,
    w_down_weight: Any,
    a_down_weight: Any,
    g_down_weight: Any,
    w_up_weight: Any,
    a_up_weight: Any,
    g_up_weight: Any,
    w_up_bias: Any | None = None,
    a_up_bias: Any | None = None,
    g_up_bias: Any | None = None,
    *,
    block_m: int = 16,
    block_r: int = 64,
    block_k: int = 64,
    force_fallback: bool = False,
):
    """Compute RWKV W/A/G LoRA outputs with grouped optional Triton kernels.

    The grouped prototype covers the next larger attention projection/LoRA
    bucket after the W/A-only probe.  It supports different low-rank sizes, as
    current checkpoints use W/A rank 64 and G rank 128 on 0.1B.  Activations
    match RWKV-7 modules: W uses tanh, A uses identity, and G uses sigmoid.
    Returned tensors are raw module outputs; caller still applies the outer W/A
    sigmoid/scaling used by RWKV attention.
    """

    if torch is None or F is None:
        raise RuntimeError("fused_wag_lora requires torch")
    xw2, had_seq = _flatten_lora_input(xw, name="xw")
    hidden = int(xw2.shape[1])
    xa2, xa_had_seq = _flatten_lora_input(xa, hidden, name="xa")
    xg2, xg_had_seq = _flatten_lora_input(xg, hidden, name="xg")
    if tuple(xw2.shape) != tuple(xa2.shape) or tuple(xw2.shape) != tuple(xg2.shape):
        raise ValueError("xw, xa and xg must have identical flattened shapes")
    if xa_had_seq != had_seq or xg_had_seq != had_seq:
        raise ValueError("xw, xa and xg must use the same rank/layout")

    w_rank = _validate_lora_weights(w_down_weight, w_up_weight, w_up_bias, hidden, name="w")
    a_rank = _validate_lora_weights(a_down_weight, a_up_weight, a_up_bias, hidden, name="a")
    g_rank = _validate_lora_weights(g_down_weight, g_up_weight, g_up_bias, hidden, name="g")
    max_rank = max(w_rank, a_rank, g_rank)

    use_triton = (
        not force_fallback
        and fused_wag_lora_available()
        and xw2.is_cuda
        and xa2.is_cuda
        and xg2.is_cuda
        and w_down_weight.is_cuda
        and a_down_weight.is_cuda
        and g_down_weight.is_cuda
        and w_up_weight.is_cuda
        and a_up_weight.is_cuda
        and g_up_weight.is_cuda
        and (w_up_bias is None or w_up_bias.is_cuda)
        and (a_up_bias is None or a_up_bias.is_cuda)
        and (g_up_bias is None or g_up_bias.is_cuda)
        and xw2.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and w_down_weight.dtype == xw2.dtype
        and a_down_weight.dtype == xw2.dtype
        and g_down_weight.dtype == xw2.dtype
        and w_up_weight.dtype == xw2.dtype
        and a_up_weight.dtype == xw2.dtype
        and g_up_weight.dtype == xw2.dtype
    )
    if not use_triton:
        wh = torch.tanh(F.linear(xw2, w_down_weight))
        ah = F.linear(xa2, a_down_weight)
        gh = torch.sigmoid(F.linear(xg2, g_down_weight))
        w_out = F.linear(wh, w_up_weight, w_up_bias)
        a_out = F.linear(ah, a_up_weight, a_up_bias)
        g_out = F.linear(gh, g_up_weight, g_up_bias)
    else:
        batch = int(xw2.shape[0])
        xw_c = xw2.contiguous()
        xa_c = xa2.contiguous()
        xg_c = xg2.contiguous()
        wd_c = w_down_weight.contiguous()
        ad_c = a_down_weight.contiguous()
        gd_c = g_down_weight.contiguous()
        wu_c = w_up_weight.contiguous()
        au_c = a_up_weight.contiguous()
        gu_c = g_up_weight.contiguous()
        wb_c = w_up_bias.contiguous() if w_up_bias is not None else wu_c
        ab_c = a_up_bias.contiguous() if a_up_bias is not None else au_c
        gb_c = g_up_bias.contiguous() if g_up_bias is not None else gu_c
        w_mid = torch.empty((batch, w_rank), device=xw2.device, dtype=xw2.dtype)
        a_mid = torch.empty((batch, a_rank), device=xw2.device, dtype=xw2.dtype)
        g_mid = torch.empty((batch, g_rank), device=xw2.device, dtype=xw2.dtype)
        w_out = torch.empty((batch, hidden), device=xw2.device, dtype=xw2.dtype)
        a_out = torch.empty_like(w_out)
        g_out = torch.empty_like(w_out)
        _wag_lora_down_kernel[(batch, triton.cdiv(max_rank, int(block_r)))](
            xw_c,
            xa_c,
            xg_c,
            wd_c,
            ad_c,
            gd_c,
            w_mid,
            a_mid,
            g_mid,
            hidden,
            w_rank,
            a_rank,
            g_rank,
            max_rank,
            BLOCK_R=int(block_r),
            BLOCK_K=int(block_k),
            num_warps=4,
        )
        _wag_lora_up_kernel[(batch, triton.cdiv(hidden, int(block_m)))](
            w_mid,
            a_mid,
            g_mid,
            wu_c,
            au_c,
            gu_c,
            wb_c,
            ab_c,
            gb_c,
            w_out,
            a_out,
            g_out,
            hidden,
            w_rank,
            a_rank,
            g_rank,
            max_rank,
            HAS_W_BIAS=w_up_bias is not None,
            HAS_A_BIAS=a_up_bias is not None,
            HAS_G_BIAS=g_up_bias is not None,
            BLOCK_M=int(block_m),
            BLOCK_R=int(block_r),
            num_warps=4,
        )
    if had_seq:
        return w_out.unsqueeze(1), a_out.unsqueeze(1), g_out.unsqueeze(1)
    return w_out, a_out, g_out


def fused_wavg_lora(
    xw: Any,
    xa: Any,
    xg: Any,
    xv: Any,
    w_down_weight: Any,
    a_down_weight: Any,
    g_down_weight: Any,
    v_down_weight: Any,
    w_up_weight: Any,
    a_up_weight: Any,
    g_up_weight: Any,
    v_up_weight: Any,
    w_up_bias: Any | None = None,
    a_up_bias: Any | None = None,
    g_up_bias: Any | None = None,
    v_up_bias: Any | None = None,
    *,
    block_m: int = 16,
    block_r: int = 64,
    block_k: int = 64,
    force_fallback: bool = False,
):
    """Compute RWKV W/A/G LoRA outputs plus V interpolation gate.

    This larger telemetry probe covers the layer>0 time-mix LoRA bucket that is
    still outside the current W/A/G-only fusion.  Activations match the HF
    fast-token math:

    * W: ``up(tanh(down(xw))) + bias``
    * A: ``up(down(xa)) + bias``; caller applies the outer sigmoid
    * G: ``up(sigmoid(down(xg)))`` (usually no bias)
    * V gate: ``sigmoid(up(down(xv)) + bias)``

    The returned ``v_gate`` is the already-sigmoided interpolation gate used by
    ``v = v + (v_first - v) * v_gate``.
    """

    if torch is None or F is None:
        raise RuntimeError("fused_wavg_lora requires torch")
    xw2, had_seq = _flatten_lora_input(xw, name="xw")
    hidden = int(xw2.shape[1])
    xa2, xa_had_seq = _flatten_lora_input(xa, hidden, name="xa")
    xg2, xg_had_seq = _flatten_lora_input(xg, hidden, name="xg")
    xv2, xv_had_seq = _flatten_lora_input(xv, hidden, name="xv")
    if tuple(xw2.shape) != tuple(xa2.shape) or tuple(xw2.shape) != tuple(xg2.shape) or tuple(xw2.shape) != tuple(xv2.shape):
        raise ValueError("xw, xa, xg and xv must have identical flattened shapes")
    if xa_had_seq != had_seq or xg_had_seq != had_seq or xv_had_seq != had_seq:
        raise ValueError("xw, xa, xg and xv must use the same rank/layout")

    w_rank = _validate_lora_weights(w_down_weight, w_up_weight, w_up_bias, hidden, name="w")
    a_rank = _validate_lora_weights(a_down_weight, a_up_weight, a_up_bias, hidden, name="a")
    g_rank = _validate_lora_weights(g_down_weight, g_up_weight, g_up_bias, hidden, name="g")
    v_rank = _validate_lora_weights(v_down_weight, v_up_weight, v_up_bias, hidden, name="v")
    max_rank = max(w_rank, a_rank, g_rank, v_rank)

    use_triton = (
        not force_fallback
        and fused_wavg_lora_available()
        and xw2.is_cuda
        and xa2.is_cuda
        and xg2.is_cuda
        and xv2.is_cuda
        and w_down_weight.is_cuda
        and a_down_weight.is_cuda
        and g_down_weight.is_cuda
        and v_down_weight.is_cuda
        and w_up_weight.is_cuda
        and a_up_weight.is_cuda
        and g_up_weight.is_cuda
        and v_up_weight.is_cuda
        and (w_up_bias is None or w_up_bias.is_cuda)
        and (a_up_bias is None or a_up_bias.is_cuda)
        and (g_up_bias is None or g_up_bias.is_cuda)
        and (v_up_bias is None or v_up_bias.is_cuda)
        and xw2.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and w_down_weight.dtype == xw2.dtype
        and a_down_weight.dtype == xw2.dtype
        and g_down_weight.dtype == xw2.dtype
        and v_down_weight.dtype == xw2.dtype
        and w_up_weight.dtype == xw2.dtype
        and a_up_weight.dtype == xw2.dtype
        and g_up_weight.dtype == xw2.dtype
        and v_up_weight.dtype == xw2.dtype
    )
    if not use_triton:
        wh = torch.tanh(F.linear(xw2, w_down_weight))
        ah = F.linear(xa2, a_down_weight)
        gh = torch.sigmoid(F.linear(xg2, g_down_weight))
        vh = F.linear(xv2, v_down_weight)
        w_out = F.linear(wh, w_up_weight, w_up_bias)
        a_out = F.linear(ah, a_up_weight, a_up_bias)
        g_out = F.linear(gh, g_up_weight, g_up_bias)
        v_gate = torch.sigmoid(F.linear(vh, v_up_weight, v_up_bias))
    else:
        batch = int(xw2.shape[0])
        xw_c = xw2.contiguous()
        xa_c = xa2.contiguous()
        xg_c = xg2.contiguous()
        xv_c = xv2.contiguous()
        wd_c = w_down_weight.contiguous()
        ad_c = a_down_weight.contiguous()
        gd_c = g_down_weight.contiguous()
        vd_c = v_down_weight.contiguous()
        wu_c = w_up_weight.contiguous()
        au_c = a_up_weight.contiguous()
        gu_c = g_up_weight.contiguous()
        vu_c = v_up_weight.contiguous()
        wb_c = w_up_bias.contiguous() if w_up_bias is not None else wu_c
        ab_c = a_up_bias.contiguous() if a_up_bias is not None else au_c
        gb_c = g_up_bias.contiguous() if g_up_bias is not None else gu_c
        vb_c = v_up_bias.contiguous() if v_up_bias is not None else vu_c
        w_mid = torch.empty((batch, w_rank), device=xw2.device, dtype=xw2.dtype)
        a_mid = torch.empty((batch, a_rank), device=xw2.device, dtype=xw2.dtype)
        g_mid = torch.empty((batch, g_rank), device=xw2.device, dtype=xw2.dtype)
        v_mid = torch.empty((batch, v_rank), device=xw2.device, dtype=xw2.dtype)
        w_out = torch.empty((batch, hidden), device=xw2.device, dtype=xw2.dtype)
        a_out = torch.empty_like(w_out)
        g_out = torch.empty_like(w_out)
        v_gate = torch.empty_like(w_out)
        _wavg_lora_down_kernel[(batch, triton.cdiv(max_rank, int(block_r)))](
            xw_c,
            xa_c,
            xg_c,
            xv_c,
            wd_c,
            ad_c,
            gd_c,
            vd_c,
            w_mid,
            a_mid,
            g_mid,
            v_mid,
            hidden,
            w_rank,
            a_rank,
            g_rank,
            v_rank,
            max_rank,
            BLOCK_R=int(block_r),
            BLOCK_K=int(block_k),
            num_warps=4,
        )
        _wavg_lora_up_kernel[(batch, triton.cdiv(hidden, int(block_m)))](
            w_mid,
            a_mid,
            g_mid,
            v_mid,
            wu_c,
            au_c,
            gu_c,
            vu_c,
            wb_c,
            ab_c,
            gb_c,
            vb_c,
            w_out,
            a_out,
            g_out,
            v_gate,
            hidden,
            w_rank,
            a_rank,
            g_rank,
            v_rank,
            max_rank,
            HAS_W_BIAS=w_up_bias is not None,
            HAS_A_BIAS=a_up_bias is not None,
            HAS_G_BIAS=g_up_bias is not None,
            HAS_V_BIAS=v_up_bias is not None,
            BLOCK_M=int(block_m),
            BLOCK_R=int(block_r),
            num_warps=4,
        )
    if had_seq:
        return w_out.unsqueeze(1), a_out.unsqueeze(1), g_out.unsqueeze(1), v_gate.unsqueeze(1)
    return w_out, a_out, g_out, v_gate
