# coding=utf-8
"""Optional combined attention projection prototypes for RWKV-7 decode.

This module is a telemetry-only stepping stone toward the fused fp16 backend.
It combines the R/K/V dense projections with the W/A/G LoRA group in a two
launch prototype:

1. one Triton launch computes R/K/V plus W/A/G low-rank down activations;
2. one Triton launch computes W/A/G low-rank up projections.

The HF model path remains unchanged.  Benchmarks decide whether/when this
building block should be integrated behind ``rwkv7_forward_token``.
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
    def _rkv_wag_down_kernel(
        xr_ptr,
        xk_ptr,
        xv_ptr,
        xw_ptr,
        xa_ptr,
        xg_ptr,
        r_weight_ptr,
        k_weight_ptr,
        v_weight_ptr,
        w_down_ptr,
        a_down_ptr,
        g_down_ptr,
        r_out_ptr,
        k_out_ptr,
        v_out_ptr,
        w_mid_ptr,
        a_mid_ptr,
        g_mid_ptr,
        hidden: tl.constexpr,
        w_rank: tl.constexpr,
        a_rank: tl.constexpr,
        g_rank: tl.constexpr,
        max_rank: tl.constexpr,
        H_BLOCKS: tl.constexpr,
        R_BLOCKS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        block_id = tl.program_id(1)
        offs_k = tl.arange(0, BLOCK_K)

        if block_id < H_BLOCKS:
            offs_m = block_id * BLOCK_M + tl.arange(0, BLOCK_M)
            mask_m = offs_m < hidden
            acc_r = tl.zeros((BLOCK_M,), tl.float32)
            acc_k = tl.zeros((BLOCK_M,), tl.float32)
            acc_v = tl.zeros((BLOCK_M,), tl.float32)
            for start in range(0, hidden, BLOCK_K):
                kidx = start + offs_k
                mask_k = kidx < hidden
                xr = tl.load(xr_ptr + batch_id * hidden + kidx, mask=mask_k, other=0.0).to(tl.float32)
                xk = tl.load(xk_ptr + batch_id * hidden + kidx, mask=mask_k, other=0.0).to(tl.float32)
                xv = tl.load(xv_ptr + batch_id * hidden + kidx, mask=mask_k, other=0.0).to(tl.float32)
                weight_offsets = offs_m[:, None] * hidden + kidx[None, :]
                mask_w = mask_m[:, None] & mask_k[None, :]
                rw = tl.load(r_weight_ptr + weight_offsets, mask=mask_w, other=0.0).to(tl.float32)
                kw = tl.load(k_weight_ptr + weight_offsets, mask=mask_w, other=0.0).to(tl.float32)
                vw = tl.load(v_weight_ptr + weight_offsets, mask=mask_w, other=0.0).to(tl.float32)
                acc_r += tl.sum(rw * xr[None, :], axis=1)
                acc_k += tl.sum(kw * xk[None, :], axis=1)
                acc_v += tl.sum(vw * xv[None, :], axis=1)
            out_base = batch_id * hidden + offs_m
            tl.store(r_out_ptr + out_base, acc_r, mask=mask_m)
            tl.store(k_out_ptr + out_base, acc_k, mask=mask_m)
            tl.store(v_out_ptr + out_base, acc_v, mask=mask_m)

        if block_id < R_BLOCKS:
            offs_r = block_id * BLOCK_R + tl.arange(0, BLOCK_R)
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
            w_act = 2.0 * tl.sigmoid(2.0 * acc_w) - 1.0
            g_act = tl.sigmoid(acc_g)
            tl.store(w_mid_ptr + batch_id * w_rank + offs_r, w_act, mask=mask_w_r)
            tl.store(a_mid_ptr + batch_id * a_rank + offs_r, acc_a, mask=mask_a_r)
            tl.store(g_mid_ptr + batch_id * g_rank + offs_r, g_act, mask=mask_g_r)

    @triton.jit
    def _wag_up_kernel(
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
    def _rkv_wavg_down_kernel(
        xr_ptr,
        xk_ptr,
        xv_ptr,
        xw_ptr,
        xa_ptr,
        xg_ptr,
        r_weight_ptr,
        k_weight_ptr,
        v_weight_ptr,
        w_down_ptr,
        a_down_ptr,
        g_down_ptr,
        vg_down_ptr,
        r_out_ptr,
        k_out_ptr,
        v_out_ptr,
        w_mid_ptr,
        a_mid_ptr,
        g_mid_ptr,
        vg_mid_ptr,
        hidden: tl.constexpr,
        w_rank: tl.constexpr,
        a_rank: tl.constexpr,
        g_rank: tl.constexpr,
        vg_rank: tl.constexpr,
        max_rank: tl.constexpr,
        H_BLOCKS: tl.constexpr,
        R_BLOCKS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_R: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        block_id = tl.program_id(1)
        offs_k = tl.arange(0, BLOCK_K)

        if block_id < H_BLOCKS:
            offs_m = block_id * BLOCK_M + tl.arange(0, BLOCK_M)
            mask_m = offs_m < hidden
            acc_r = tl.zeros((BLOCK_M,), tl.float32)
            acc_k = tl.zeros((BLOCK_M,), tl.float32)
            acc_v = tl.zeros((BLOCK_M,), tl.float32)
            for start in range(0, hidden, BLOCK_K):
                kidx = start + offs_k
                mask_k = kidx < hidden
                xr = tl.load(xr_ptr + batch_id * hidden + kidx, mask=mask_k, other=0.0).to(tl.float32)
                xk = tl.load(xk_ptr + batch_id * hidden + kidx, mask=mask_k, other=0.0).to(tl.float32)
                xv = tl.load(xv_ptr + batch_id * hidden + kidx, mask=mask_k, other=0.0).to(tl.float32)
                weight_offsets = offs_m[:, None] * hidden + kidx[None, :]
                mask_w = mask_m[:, None] & mask_k[None, :]
                rw = tl.load(r_weight_ptr + weight_offsets, mask=mask_w, other=0.0).to(tl.float32)
                kw = tl.load(k_weight_ptr + weight_offsets, mask=mask_w, other=0.0).to(tl.float32)
                vw = tl.load(v_weight_ptr + weight_offsets, mask=mask_w, other=0.0).to(tl.float32)
                acc_r += tl.sum(rw * xr[None, :], axis=1)
                acc_k += tl.sum(kw * xk[None, :], axis=1)
                acc_v += tl.sum(vw * xv[None, :], axis=1)
            out_base = batch_id * hidden + offs_m
            tl.store(r_out_ptr + out_base, acc_r, mask=mask_m)
            tl.store(k_out_ptr + out_base, acc_k, mask=mask_m)
            tl.store(v_out_ptr + out_base, acc_v, mask=mask_m)

        if block_id < R_BLOCKS:
            offs_r = block_id * BLOCK_R + tl.arange(0, BLOCK_R)
            mask_w_r = offs_r < w_rank
            mask_a_r = offs_r < a_rank
            mask_g_r = offs_r < g_rank
            mask_vg_r = offs_r < vg_rank
            acc_w = tl.zeros((BLOCK_R,), tl.float32)
            acc_a = tl.zeros((BLOCK_R,), tl.float32)
            acc_g = tl.zeros((BLOCK_R,), tl.float32)
            acc_vg = tl.zeros((BLOCK_R,), tl.float32)
            for start in range(0, hidden, BLOCK_K):
                kidx = start + offs_k
                mask_k = kidx < hidden
                xw = tl.load(xw_ptr + batch_id * hidden + kidx, mask=mask_k, other=0.0).to(tl.float32)
                xa = tl.load(xa_ptr + batch_id * hidden + kidx, mask=mask_k, other=0.0).to(tl.float32)
                xg = tl.load(xg_ptr + batch_id * hidden + kidx, mask=mask_k, other=0.0).to(tl.float32)
                xv_gate = tl.load(xv_ptr + batch_id * hidden + kidx, mask=mask_k, other=0.0).to(tl.float32)
                w_offsets = offs_r[:, None] * hidden + kidx[None, :]
                a_offsets = offs_r[:, None] * hidden + kidx[None, :]
                g_offsets = offs_r[:, None] * hidden + kidx[None, :]
                vg_offsets = offs_r[:, None] * hidden + kidx[None, :]
                wd = tl.load(w_down_ptr + w_offsets, mask=mask_w_r[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
                ad = tl.load(a_down_ptr + a_offsets, mask=mask_a_r[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
                gd = tl.load(g_down_ptr + g_offsets, mask=mask_g_r[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
                vgd = tl.load(vg_down_ptr + vg_offsets, mask=mask_vg_r[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
                acc_w += tl.sum(wd * xw[None, :], axis=1)
                acc_a += tl.sum(ad * xa[None, :], axis=1)
                acc_g += tl.sum(gd * xg[None, :], axis=1)
                acc_vg += tl.sum(vgd * xv_gate[None, :], axis=1)
            w_act = 2.0 * tl.sigmoid(2.0 * acc_w) - 1.0
            g_act = tl.sigmoid(acc_g)
            tl.store(w_mid_ptr + batch_id * w_rank + offs_r, w_act, mask=mask_w_r)
            tl.store(a_mid_ptr + batch_id * a_rank + offs_r, acc_a, mask=mask_a_r)
            tl.store(g_mid_ptr + batch_id * g_rank + offs_r, g_act, mask=mask_g_r)
            tl.store(vg_mid_ptr + batch_id * vg_rank + offs_r, acc_vg, mask=mask_vg_r)

    @triton.jit
    def _wavg_up_kernel(
        w_mid_ptr,
        a_mid_ptr,
        g_mid_ptr,
        vg_mid_ptr,
        w_up_ptr,
        a_up_ptr,
        g_up_ptr,
        vg_up_ptr,
        w_bias_ptr,
        a_bias_ptr,
        g_bias_ptr,
        vg_bias_ptr,
        w_out_ptr,
        a_out_ptr,
        g_out_ptr,
        vg_out_ptr,
        hidden: tl.constexpr,
        w_rank: tl.constexpr,
        a_rank: tl.constexpr,
        g_rank: tl.constexpr,
        vg_rank: tl.constexpr,
        max_rank: tl.constexpr,
        HAS_W_BIAS: tl.constexpr,
        HAS_A_BIAS: tl.constexpr,
        HAS_G_BIAS: tl.constexpr,
        HAS_VG_BIAS: tl.constexpr,
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
        acc_vg = tl.zeros((BLOCK_M,), tl.float32)
        for start in range(0, max_rank, BLOCK_R):
            ridx = start + offs_r
            mask_w_r = ridx < w_rank
            mask_a_r = ridx < a_rank
            mask_g_r = ridx < g_rank
            mask_vg_r = ridx < vg_rank
            wm = tl.load(w_mid_ptr + batch_id * w_rank + ridx, mask=mask_w_r, other=0.0).to(tl.float32)
            am = tl.load(a_mid_ptr + batch_id * a_rank + ridx, mask=mask_a_r, other=0.0).to(tl.float32)
            gm = tl.load(g_mid_ptr + batch_id * g_rank + ridx, mask=mask_g_r, other=0.0).to(tl.float32)
            vgm = tl.load(vg_mid_ptr + batch_id * vg_rank + ridx, mask=mask_vg_r, other=0.0).to(tl.float32)
            w_offsets = offs_m[:, None] * w_rank + ridx[None, :]
            a_offsets = offs_m[:, None] * a_rank + ridx[None, :]
            g_offsets = offs_m[:, None] * g_rank + ridx[None, :]
            vg_offsets = offs_m[:, None] * vg_rank + ridx[None, :]
            wu = tl.load(w_up_ptr + w_offsets, mask=mask_m[:, None] & mask_w_r[None, :], other=0.0).to(tl.float32)
            au = tl.load(a_up_ptr + a_offsets, mask=mask_m[:, None] & mask_a_r[None, :], other=0.0).to(tl.float32)
            gu = tl.load(g_up_ptr + g_offsets, mask=mask_m[:, None] & mask_g_r[None, :], other=0.0).to(tl.float32)
            vgu = tl.load(vg_up_ptr + vg_offsets, mask=mask_m[:, None] & mask_vg_r[None, :], other=0.0).to(tl.float32)
            acc_w += tl.sum(wu * wm[None, :], axis=1)
            acc_a += tl.sum(au * am[None, :], axis=1)
            acc_g += tl.sum(gu * gm[None, :], axis=1)
            acc_vg += tl.sum(vgu * vgm[None, :], axis=1)
        if HAS_W_BIAS:
            wb = tl.load(w_bias_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
            acc_w += wb
        if HAS_A_BIAS:
            ab = tl.load(a_bias_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
            acc_a += ab
        if HAS_G_BIAS:
            gb = tl.load(g_bias_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
            acc_g += gb
        if HAS_VG_BIAS:
            vgb = tl.load(vg_bias_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
            acc_vg += vgb
        out_base = batch_id * hidden + offs_m
        tl.store(w_out_ptr + out_base, acc_w, mask=mask_m)
        tl.store(a_out_ptr + out_base, acc_a, mask=mask_m)
        tl.store(g_out_ptr + out_base, acc_g, mask=mask_m)
        tl.store(vg_out_ptr + out_base, acc_vg, mask=mask_m)


def fused_rkv_wag_projection_available() -> bool:
    """Return whether the optional combined projection prototype can run."""

    return bool(_HAS_TRITON and torch is not None)


def fused_rkv_wavg_projection_available() -> bool:
    """Return whether the R/K/V + W/A/G/V-gate projection prototype can run."""

    return fused_rkv_wag_projection_available()


def _flatten(x: Any, hidden: int | None = None, *, name: str):
    if torch is None:
        raise RuntimeError("fused_rkv_wag_projection requires torch")
    if x.dim() == 3:
        if hidden is not None and int(x.shape[2]) != hidden:
            raise ValueError(f"{name} hidden mismatch: got {int(x.shape[2])}, expected {hidden}")
        return x.reshape(int(x.shape[0]) * int(x.shape[1]), int(x.shape[2])), tuple(x.shape)
    if x.dim() == 2:
        if hidden is not None and int(x.shape[1]) != hidden:
            raise ValueError(f"{name} hidden mismatch: got {int(x.shape[1])}, expected {hidden}")
        return x, None
    raise ValueError(f"{name} must be [batch, tokens, hidden] or [batch, hidden]")


def _validate_square_weight(w: Any, hidden: int, *, name: str) -> None:
    if w.dim() != 2 or int(w.shape[0]) != hidden or int(w.shape[1]) != hidden:
        raise ValueError(f"{name} must be [{hidden}, {hidden}], got {tuple(w.shape)}")


def _validate_lora_weight(down: Any, up: Any, bias: Any | None, hidden: int, *, name: str) -> int:
    if down.dim() != 2 or int(down.shape[1]) != hidden:
        raise ValueError(f"{name}_down must be [rank, {hidden}], got {tuple(down.shape)}")
    rank = int(down.shape[0])
    if up.dim() != 2 or int(up.shape[0]) != hidden or int(up.shape[1]) != rank:
        raise ValueError(f"{name}_up must be [{hidden}, {rank}], got {tuple(up.shape)}")
    if bias is not None and (bias.dim() != 1 or int(bias.shape[0]) != hidden):
        raise ValueError(f"{name}_bias must be [{hidden}], got {tuple(bias.shape)}")
    return rank


def fused_rkv_wag_projection(
    xr: Any,
    xk: Any,
    xv: Any,
    xw: Any,
    xa: Any,
    xg: Any,
    r_weight: Any,
    k_weight: Any,
    v_weight: Any,
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
    block_m: int = 64,
    block_r: int = 64,
    block_k: int = 64,
    force_fallback: bool = False,
):
    """Compute R/K/V dense projections plus W/A/G LoRA outputs.

    Outputs are returned as ``(r, k, v, w, a, g)`` and preserve the input shape
    (`[batch, hidden]` or `[batch, tokens, hidden]`).  W/A/G outputs are raw LoRA
    module outputs; callers still apply RWKV's outer sigmoid/scaling.
    """

    if torch is None or F is None:
        raise RuntimeError("fused_rkv_wag_projection requires torch")
    xr2, restore_shape = _flatten(xr, name="xr")
    hidden = int(xr2.shape[1])
    inputs = [xr2]
    for name, value in (("xk", xk), ("xv", xv), ("xw", xw), ("xa", xa), ("xg", xg)):
        flat, shape = _flatten(value, hidden, name=name)
        if shape != restore_shape or tuple(flat.shape) != tuple(xr2.shape):
            raise ValueError("all projection inputs must have identical flattened shape/layout")
        inputs.append(flat)
    xk2, xv2, xw2, xa2, xg2 = inputs[1:]
    for name, weight in (("r_weight", r_weight), ("k_weight", k_weight), ("v_weight", v_weight)):
        _validate_square_weight(weight, hidden, name=name)
    w_rank = _validate_lora_weight(w_down_weight, w_up_weight, w_up_bias, hidden, name="w")
    a_rank = _validate_lora_weight(a_down_weight, a_up_weight, a_up_bias, hidden, name="a")
    g_rank = _validate_lora_weight(g_down_weight, g_up_weight, g_up_bias, hidden, name="g")
    max_rank = max(w_rank, a_rank, g_rank)

    tensors = [
        xr2,
        xk2,
        xv2,
        xw2,
        xa2,
        xg2,
        r_weight,
        k_weight,
        v_weight,
        w_down_weight,
        a_down_weight,
        g_down_weight,
        w_up_weight,
        a_up_weight,
        g_up_weight,
    ]
    use_triton = (
        not force_fallback
        and fused_rkv_wag_projection_available()
        and all(t.is_cuda for t in tensors)
        and (w_up_bias is None or w_up_bias.is_cuda)
        and (a_up_bias is None or a_up_bias.is_cuda)
        and (g_up_bias is None or g_up_bias.is_cuda)
        and xr2.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and all(t.dtype == xr2.dtype for t in tensors)
    )
    if not use_triton:
        r = F.linear(xr2, r_weight)
        k = F.linear(xk2, k_weight)
        v = F.linear(xv2, v_weight)
        wh = torch.tanh(F.linear(xw2, w_down_weight))
        ah = F.linear(xa2, a_down_weight)
        gh = torch.sigmoid(F.linear(xg2, g_down_weight))
        w = F.linear(wh, w_up_weight, w_up_bias)
        a = F.linear(ah, a_up_weight, a_up_bias)
        g = F.linear(gh, g_up_weight, g_up_bias)
    else:
        batch = int(xr2.shape[0])
        xr_c, xk_c, xv_c, xw_c, xa_c, xg_c = [t.contiguous() for t in (xr2, xk2, xv2, xw2, xa2, xg2)]
        rw_c, kw_c, vw_c = [t.contiguous() for t in (r_weight, k_weight, v_weight)]
        wd_c, ad_c, gd_c = [t.contiguous() for t in (w_down_weight, a_down_weight, g_down_weight)]
        wu_c, au_c, gu_c = [t.contiguous() for t in (w_up_weight, a_up_weight, g_up_weight)]
        wb_c = w_up_bias.contiguous() if w_up_bias is not None else wu_c
        ab_c = a_up_bias.contiguous() if a_up_bias is not None else au_c
        gb_c = g_up_bias.contiguous() if g_up_bias is not None else gu_c
        r = torch.empty((batch, hidden), device=xr2.device, dtype=xr2.dtype)
        k = torch.empty_like(r)
        v = torch.empty_like(r)
        w_mid = torch.empty((batch, w_rank), device=xr2.device, dtype=xr2.dtype)
        a_mid = torch.empty((batch, a_rank), device=xr2.device, dtype=xr2.dtype)
        g_mid = torch.empty((batch, g_rank), device=xr2.device, dtype=xr2.dtype)
        w = torch.empty_like(r)
        a = torch.empty_like(r)
        g = torch.empty_like(r)
        h_blocks = triton.cdiv(hidden, int(block_m))
        r_blocks = triton.cdiv(max_rank, int(block_r))
        _rkv_wag_down_kernel[(batch, max(h_blocks, r_blocks))](
            xr_c,
            xk_c,
            xv_c,
            xw_c,
            xa_c,
            xg_c,
            rw_c,
            kw_c,
            vw_c,
            wd_c,
            ad_c,
            gd_c,
            r,
            k,
            v,
            w_mid,
            a_mid,
            g_mid,
            hidden,
            w_rank,
            a_rank,
            g_rank,
            max_rank,
            H_BLOCKS=h_blocks,
            R_BLOCKS=r_blocks,
            BLOCK_M=int(block_m),
            BLOCK_R=int(block_r),
            BLOCK_K=int(block_k),
            num_warps=4,
        )
        _wag_up_kernel[(batch, h_blocks)](
            w_mid,
            a_mid,
            g_mid,
            wu_c,
            au_c,
            gu_c,
            wb_c,
            ab_c,
            gb_c,
            w,
            a,
            g,
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
    if restore_shape is not None:
        return tuple(t.reshape(restore_shape) for t in (r, k, v, w, a, g))
    return r, k, v, w, a, g


def fused_rkv_wavg_projection(
    xr: Any,
    xk: Any,
    xv: Any,
    xw: Any,
    xa: Any,
    xg: Any,
    r_weight: Any,
    k_weight: Any,
    v_weight: Any,
    w_down_weight: Any,
    a_down_weight: Any,
    g_down_weight: Any,
    vg_down_weight: Any,
    w_up_weight: Any,
    a_up_weight: Any,
    g_up_weight: Any,
    vg_up_weight: Any,
    w_up_bias: Any | None = None,
    a_up_bias: Any | None = None,
    g_up_bias: Any | None = None,
    vg_up_bias: Any | None = None,
    *,
    block_m: int = 64,
    block_r: int = 64,
    block_k: int = 64,
    force_fallback: bool = False,
):
    """Compute R/K/V projections plus W/A/G/V-gate LoRA outputs.

    This is the deeper decode-fusion variant of :func:`fused_rkv_wag_projection`.
    It also groups the per-layer V-gate LoRA used after layer 0, removing one
    more small down/up GEMV pair from the native_graph decode path.  The
    returned tuple is ``(r, k, v, w, a, g, v_gate)`` with the same shape as the
    inputs.
    """

    if torch is None or F is None:
        raise RuntimeError("fused_rkv_wavg_projection requires torch")
    xr2, restore_shape = _flatten(xr, name="xr")
    hidden = int(xr2.shape[1])
    inputs = [xr2]
    for name, value in (("xk", xk), ("xv", xv), ("xw", xw), ("xa", xa), ("xg", xg)):
        flat, shape = _flatten(value, hidden, name=name)
        if shape != restore_shape or tuple(flat.shape) != tuple(xr2.shape):
            raise ValueError("all projection inputs must have identical flattened shape/layout")
        inputs.append(flat)
    xk2, xv2, xw2, xa2, xg2 = inputs[1:]
    for name, weight in (("r_weight", r_weight), ("k_weight", k_weight), ("v_weight", v_weight)):
        _validate_square_weight(weight, hidden, name=name)
    w_rank = _validate_lora_weight(w_down_weight, w_up_weight, w_up_bias, hidden, name="w")
    a_rank = _validate_lora_weight(a_down_weight, a_up_weight, a_up_bias, hidden, name="a")
    g_rank = _validate_lora_weight(g_down_weight, g_up_weight, g_up_bias, hidden, name="g")
    vg_rank = _validate_lora_weight(vg_down_weight, vg_up_weight, vg_up_bias, hidden, name="vg")
    max_rank = max(w_rank, a_rank, g_rank, vg_rank)

    tensors = [
        xr2,
        xk2,
        xv2,
        xw2,
        xa2,
        xg2,
        r_weight,
        k_weight,
        v_weight,
        w_down_weight,
        a_down_weight,
        g_down_weight,
        vg_down_weight,
        w_up_weight,
        a_up_weight,
        g_up_weight,
        vg_up_weight,
    ]
    use_triton = (
        not force_fallback
        and fused_rkv_wavg_projection_available()
        and all(t.is_cuda for t in tensors)
        and (w_up_bias is None or w_up_bias.is_cuda)
        and (a_up_bias is None or a_up_bias.is_cuda)
        and (g_up_bias is None or g_up_bias.is_cuda)
        and (vg_up_bias is None or vg_up_bias.is_cuda)
        and xr2.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and all(t.dtype == xr2.dtype for t in tensors)
    )
    if not use_triton:
        r = F.linear(xr2, r_weight)
        k = F.linear(xk2, k_weight)
        v = F.linear(xv2, v_weight)
        wh = torch.tanh(F.linear(xw2, w_down_weight))
        ah = F.linear(xa2, a_down_weight)
        gh = torch.sigmoid(F.linear(xg2, g_down_weight))
        vgh = F.linear(xv2, vg_down_weight)
        w = F.linear(wh, w_up_weight, w_up_bias)
        a = F.linear(ah, a_up_weight, a_up_bias)
        g = F.linear(gh, g_up_weight, g_up_bias)
        v_gate = F.linear(vgh, vg_up_weight, vg_up_bias)
    else:
        batch = int(xr2.shape[0])
        xr_c, xk_c, xv_c, xw_c, xa_c, xg_c = [t.contiguous() for t in (xr2, xk2, xv2, xw2, xa2, xg2)]
        rw_c, kw_c, vw_c = [t.contiguous() for t in (r_weight, k_weight, v_weight)]
        wd_c, ad_c, gd_c, vgd_c = [t.contiguous() for t in (w_down_weight, a_down_weight, g_down_weight, vg_down_weight)]
        wu_c, au_c, gu_c, vgu_c = [t.contiguous() for t in (w_up_weight, a_up_weight, g_up_weight, vg_up_weight)]
        wb_c = w_up_bias.contiguous() if w_up_bias is not None else wu_c
        ab_c = a_up_bias.contiguous() if a_up_bias is not None else au_c
        gb_c = g_up_bias.contiguous() if g_up_bias is not None else gu_c
        vgb_c = vg_up_bias.contiguous() if vg_up_bias is not None else vgu_c
        r = torch.empty((batch, hidden), device=xr2.device, dtype=xr2.dtype)
        k = torch.empty_like(r)
        v = torch.empty_like(r)
        w_mid = torch.empty((batch, w_rank), device=xr2.device, dtype=xr2.dtype)
        a_mid = torch.empty((batch, a_rank), device=xr2.device, dtype=xr2.dtype)
        g_mid = torch.empty((batch, g_rank), device=xr2.device, dtype=xr2.dtype)
        vg_mid = torch.empty((batch, vg_rank), device=xr2.device, dtype=xr2.dtype)
        w = torch.empty_like(r)
        a = torch.empty_like(r)
        g = torch.empty_like(r)
        v_gate = torch.empty_like(r)
        h_blocks = triton.cdiv(hidden, int(block_m))
        r_blocks = triton.cdiv(max_rank, int(block_r))
        _rkv_wavg_down_kernel[(batch, max(h_blocks, r_blocks))](
            xr_c,
            xk_c,
            xv_c,
            xw_c,
            xa_c,
            xg_c,
            rw_c,
            kw_c,
            vw_c,
            wd_c,
            ad_c,
            gd_c,
            vgd_c,
            r,
            k,
            v,
            w_mid,
            a_mid,
            g_mid,
            vg_mid,
            hidden,
            w_rank,
            a_rank,
            g_rank,
            vg_rank,
            max_rank,
            H_BLOCKS=h_blocks,
            R_BLOCKS=r_blocks,
            BLOCK_M=int(block_m),
            BLOCK_R=int(block_r),
            BLOCK_K=int(block_k),
            num_warps=4,
        )
        _wavg_up_kernel[(batch, h_blocks)](
            w_mid,
            a_mid,
            g_mid,
            vg_mid,
            wu_c,
            au_c,
            gu_c,
            vgu_c,
            wb_c,
            ab_c,
            gb_c,
            vgb_c,
            w,
            a,
            g,
            v_gate,
            hidden,
            w_rank,
            a_rank,
            g_rank,
            vg_rank,
            max_rank,
            HAS_W_BIAS=w_up_bias is not None,
            HAS_A_BIAS=a_up_bias is not None,
            HAS_G_BIAS=g_up_bias is not None,
            HAS_VG_BIAS=vg_up_bias is not None,
            BLOCK_M=int(block_m),
            BLOCK_R=int(block_r),
            num_warps=4,
        )
    if restore_shape is not None:
        return tuple(t.reshape(restore_shape) for t in (r, k, v, w, a, g, v_gate))
    return r, k, v, w, a, g, v_gate
