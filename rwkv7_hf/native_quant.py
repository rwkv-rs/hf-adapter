# coding=utf-8
"""Native quantization prototypes for RWKV-7 serving.

This module starts the RWKV-native W8/W4 path with simple row-wise packed
weights plus fused dequant GEMV.  It is intentionally optional and safe to
import without Triton/CUDA: CPU-only or unsupported hosts fall back to a torch
reference that reconstructs the dequantized weight.

The first target is decode-hot linear layers where generic bitsandbytes kernels
are currently much slower than fp16 on V100.  This prototype is telemetry-first;
it is not wired into the HF model path until correctness and speed are both
validated by benchmark rows.
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
    def _int8_rowwise_gemv_kernel(
        x_ptr,
        q_weight_ptr,
        scale_ptr,
        bias_ptr,
        out_ptr,
        in_features: tl.constexpr,
        out_features: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        block_id = tl.program_id(1)
        offs_m = block_id * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_K)
        mask_m = offs_m < out_features

        acc = tl.zeros((BLOCK_M,), tl.float32)
        for start in range(0, in_features, BLOCK_K):
            kidx = start + offs_k
            mask_k = kidx < in_features
            x = tl.load(x_ptr + batch_id * in_features + kidx, mask=mask_k, other=0.0).to(tl.float32)
            q_offsets = offs_m[:, None] * in_features + kidx[None, :]
            q = tl.load(q_weight_ptr + q_offsets, mask=mask_m[:, None] & mask_k[None, :], other=0).to(tl.float32)
            acc += tl.sum(q * x[None, :], axis=1)

        scale = tl.load(scale_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
        acc = acc * scale
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
            acc += bias
        tl.store(out_ptr + batch_id * out_features + offs_m, acc, mask=mask_m)

    @triton.jit
    def _int4_rowwise_gemv_kernel(
        x_ptr,
        q_weight_ptr,
        scale_ptr,
        bias_ptr,
        out_ptr,
        in_features: tl.constexpr,
        packed_in_features: tl.constexpr,
        out_features: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        block_id = tl.program_id(1)
        offs_m = block_id * BLOCK_M + tl.arange(0, BLOCK_M)
        # Iterate over packed int4 bytes, not logical input features.  The
        # earlier prototype iterated over every feature and therefore loaded
        # each packed byte twice (once for the low nibble and once for the high
        # nibble).  One program now consumes BLOCK_K packed bytes / 2*BLOCK_K
        # dense features, preserving row-wise scales while halving q-weight
        # traffic for W4 decode GEMV.
        offs_p = tl.arange(0, BLOCK_K)
        mask_m = offs_m < out_features

        acc = tl.zeros((BLOCK_M,), tl.float32)
        for start in range(0, packed_in_features, BLOCK_K):
            pidx = start + offs_p
            mask_p = pidx < packed_in_features
            k0 = pidx * 2
            k1 = k0 + 1
            mask_k0 = mask_p & (k0 < in_features)
            mask_k1 = mask_p & (k1 < in_features)
            x0 = tl.load(x_ptr + batch_id * in_features + k0, mask=mask_k0, other=0.0).to(tl.float32)
            x1 = tl.load(x_ptr + batch_id * in_features + k1, mask=mask_k1, other=0.0).to(tl.float32)

            q_offsets = offs_m[:, None] * packed_in_features + pidx[None, :]
            packed = tl.load(q_weight_ptr + q_offsets, mask=mask_m[:, None] & mask_p[None, :], other=0).to(tl.int32)
            q0_u4 = packed & 0xF
            q1_u4 = (packed >> 4) & 0xF
            q0 = tl.where(q0_u4 >= 8, q0_u4 - 16, q0_u4).to(tl.float32)
            q1 = tl.where(q1_u4 >= 8, q1_u4 - 16, q1_u4).to(tl.float32)
            acc += tl.sum(q0 * x0[None, :] + q1 * x1[None, :], axis=1)

        scale = tl.load(scale_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
        acc = acc * scale
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
            acc += bias
        tl.store(out_ptr + batch_id * out_features + offs_m, acc, mask=mask_m)

    @triton.jit
    def _int8_fused_rkv_gemv_kernel(
        xr_ptr,
        xk_ptr,
        xv_ptr,
        qr_ptr,
        qk_ptr,
        qv_ptr,
        sr_ptr,
        sk_ptr,
        sv_ptr,
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
            xr = tl.load(xr_ptr + batch_id * hidden + kidx, mask=mask_k, other=0.0).to(tl.float32)
            xk = tl.load(xk_ptr + batch_id * hidden + kidx, mask=mask_k, other=0.0).to(tl.float32)
            xv = tl.load(xv_ptr + batch_id * hidden + kidx, mask=mask_k, other=0.0).to(tl.float32)

            q_offsets = offs_m[:, None] * hidden + kidx[None, :]
            mask_q = mask_m[:, None] & mask_k[None, :]
            qr = tl.load(qr_ptr + q_offsets, mask=mask_q, other=0).to(tl.float32)
            qk = tl.load(qk_ptr + q_offsets, mask=mask_q, other=0).to(tl.float32)
            qv = tl.load(qv_ptr + q_offsets, mask=mask_q, other=0).to(tl.float32)
            acc_r += tl.sum(qr * xr[None, :], axis=1)
            acc_k += tl.sum(qk * xk[None, :], axis=1)
            acc_v += tl.sum(qv * xv[None, :], axis=1)

        scale_r = tl.load(sr_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
        scale_k = tl.load(sk_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
        scale_v = tl.load(sv_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
        out_base = batch_id * hidden + offs_m
        tl.store(out_r_ptr + out_base, acc_r * scale_r, mask=mask_m)
        tl.store(out_k_ptr + out_base, acc_k * scale_k, mask=mask_m)
        tl.store(out_v_ptr + out_base, acc_v * scale_v, mask=mask_m)

    @triton.jit
    def _int4_fused_rkv_gemv_kernel(
        xr_ptr,
        xk_ptr,
        xv_ptr,
        qr_ptr,
        qk_ptr,
        qv_ptr,
        sr_ptr,
        sk_ptr,
        sv_ptr,
        out_r_ptr,
        out_k_ptr,
        out_v_ptr,
        hidden: tl.constexpr,
        packed_hidden: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        block_id = tl.program_id(1)
        offs_m = block_id * BLOCK_M + tl.arange(0, BLOCK_M)
        # Iterate over packed int4 bytes so each weight byte is loaded once and
        # contributes both low/high nibbles to the dot product.  This is the W4
        # equivalent of fusing dequant with GEMV, and it avoids the duplicated
        # byte loads from the original logical-feature loop.
        offs_p = tl.arange(0, BLOCK_K)
        mask_m = offs_m < hidden

        acc_r = tl.zeros((BLOCK_M,), tl.float32)
        acc_k = tl.zeros((BLOCK_M,), tl.float32)
        acc_v = tl.zeros((BLOCK_M,), tl.float32)
        for start in range(0, packed_hidden, BLOCK_K):
            pidx = start + offs_p
            mask_p = pidx < packed_hidden
            k0 = pidx * 2
            k1 = k0 + 1
            mask_k0 = mask_p & (k0 < hidden)
            mask_k1 = mask_p & (k1 < hidden)
            xr0 = tl.load(xr_ptr + batch_id * hidden + k0, mask=mask_k0, other=0.0).to(tl.float32)
            xr1 = tl.load(xr_ptr + batch_id * hidden + k1, mask=mask_k1, other=0.0).to(tl.float32)
            xk0 = tl.load(xk_ptr + batch_id * hidden + k0, mask=mask_k0, other=0.0).to(tl.float32)
            xk1 = tl.load(xk_ptr + batch_id * hidden + k1, mask=mask_k1, other=0.0).to(tl.float32)
            xv0 = tl.load(xv_ptr + batch_id * hidden + k0, mask=mask_k0, other=0.0).to(tl.float32)
            xv1 = tl.load(xv_ptr + batch_id * hidden + k1, mask=mask_k1, other=0.0).to(tl.float32)

            q_offsets = offs_m[:, None] * packed_hidden + pidx[None, :]
            mask_q = mask_m[:, None] & mask_p[None, :]

            qr_packed = tl.load(qr_ptr + q_offsets, mask=mask_q, other=0).to(tl.int32)
            qk_packed = tl.load(qk_ptr + q_offsets, mask=mask_q, other=0).to(tl.int32)
            qv_packed = tl.load(qv_ptr + q_offsets, mask=mask_q, other=0).to(tl.int32)
            qr0_u4 = qr_packed & 0xF
            qk0_u4 = qk_packed & 0xF
            qv0_u4 = qv_packed & 0xF
            qr1_u4 = (qr_packed >> 4) & 0xF
            qk1_u4 = (qk_packed >> 4) & 0xF
            qv1_u4 = (qv_packed >> 4) & 0xF
            qr0 = tl.where(qr0_u4 >= 8, qr0_u4 - 16, qr0_u4).to(tl.float32)
            qk0 = tl.where(qk0_u4 >= 8, qk0_u4 - 16, qk0_u4).to(tl.float32)
            qv0 = tl.where(qv0_u4 >= 8, qv0_u4 - 16, qv0_u4).to(tl.float32)
            qr1 = tl.where(qr1_u4 >= 8, qr1_u4 - 16, qr1_u4).to(tl.float32)
            qk1 = tl.where(qk1_u4 >= 8, qk1_u4 - 16, qk1_u4).to(tl.float32)
            qv1 = tl.where(qv1_u4 >= 8, qv1_u4 - 16, qv1_u4).to(tl.float32)

            acc_r += tl.sum(qr0 * xr0[None, :] + qr1 * xr1[None, :], axis=1)
            acc_k += tl.sum(qk0 * xk0[None, :] + qk1 * xk1[None, :], axis=1)
            acc_v += tl.sum(qv0 * xv0[None, :] + qv1 * xv1[None, :], axis=1)

        scale_r = tl.load(sr_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
        scale_k = tl.load(sk_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
        scale_v = tl.load(sv_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
        out_base = batch_id * hidden + offs_m
        tl.store(out_r_ptr + out_base, acc_r * scale_r, mask=mask_m)
        tl.store(out_k_ptr + out_base, acc_k * scale_k, mask=mask_m)
        tl.store(out_v_ptr + out_base, acc_v * scale_v, mask=mask_m)

    @triton.jit
    def _int4_stacked_rkv_gemv_kernel(
        x_ptr,
        q_weight_ptr,
        scale_ptr,
        out_ptr,
        batch: tl.constexpr,
        projections: tl.constexpr,
        hidden: tl.constexpr,
        packed_hidden: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Projection-axis W4A16 GEMV with one accumulator per program.

        The original fused R/K/V prototype keeps three activation tiles, three
        packed-weight tiles, and three fp32 accumulators live in every program.
        That saves launches but creates substantial register pressure.  This
        KernelBench-Mega-style layout keeps the launch fused while making R/K/V
        a grid dimension.  Inputs and packed weights are pre-stacked as
        ``[batch, 3, hidden]`` and ``[3, hidden, packed_hidden]`` so each program
        follows a single projection and only needs one accumulator.
        """

        batch_id = tl.program_id(0)
        projection_id = tl.program_id(1)
        block_id = tl.program_id(2)
        offs_m = block_id * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_p = tl.arange(0, BLOCK_K)
        mask_m = offs_m < hidden

        x_base = (batch_id * projections + projection_id) * hidden
        q_base = projection_id * hidden * packed_hidden
        scale_base = projection_id * hidden
        out_base = (batch_id * projections + projection_id) * hidden

        acc = tl.zeros((BLOCK_M,), tl.float32)
        for start in range(0, packed_hidden, BLOCK_K):
            pidx = start + offs_p
            mask_p = pidx < packed_hidden
            k0 = pidx * 2
            k1 = k0 + 1
            mask_k0 = mask_p & (k0 < hidden)
            mask_k1 = mask_p & (k1 < hidden)
            x0 = tl.load(x_ptr + x_base + k0, mask=mask_k0, other=0.0).to(tl.float32)
            x1 = tl.load(x_ptr + x_base + k1, mask=mask_k1, other=0.0).to(tl.float32)

            q_offsets = q_base + offs_m[:, None] * packed_hidden + pidx[None, :]
            packed = tl.load(q_weight_ptr + q_offsets, mask=mask_m[:, None] & mask_p[None, :], other=0).to(tl.int32)
            q0_u4 = packed & 0xF
            q1_u4 = (packed >> 4) & 0xF
            q0 = tl.where(q0_u4 >= 8, q0_u4 - 16, q0_u4).to(tl.float32)
            q1 = tl.where(q1_u4 >= 8, q1_u4 - 16, q1_u4).to(tl.float32)
            acc += tl.sum(q0 * x0[None, :] + q1 * x1[None, :], axis=1)

        scale = tl.load(scale_ptr + scale_base + offs_m, mask=mask_m, other=0.0).to(tl.float32)
        tl.store(out_ptr + out_base + offs_m, acc * scale, mask=mask_m)

    @triton.jit
    def _int4_groupwise_stacked_rkv_gemv_kernel(
        x_ptr,
        q_weight_ptr,
        scale_ptr,
        out_ptr,
        batch: tl.constexpr,
        projections: tl.constexpr,
        hidden: tl.constexpr,
        packed_hidden: tl.constexpr,
        num_groups: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Projection-axis W4A16 GEMV with one scale per K group.

        ``BLOCK_K`` is required to cover exactly one packed group. This keeps
        the scale load outside the packed-value reduction instead of expanding
        a scale tensor across every nibble.
        """

        batch_id = tl.program_id(0)
        projection_id = tl.program_id(1)
        block_id = tl.program_id(2)
        offs_m = block_id * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_p = tl.arange(0, BLOCK_K)
        mask_m = offs_m < hidden

        x_base = (batch_id * projections + projection_id) * hidden
        q_base = projection_id * hidden * packed_hidden
        scale_base = projection_id * hidden * num_groups
        out_base = (batch_id * projections + projection_id) * hidden

        acc = tl.zeros((BLOCK_M,), tl.float32)
        for start in range(0, packed_hidden, BLOCK_K):
            pidx = start + offs_p
            mask_p = pidx < packed_hidden
            k0 = pidx * 2
            k1 = k0 + 1
            mask_k0 = mask_p & (k0 < hidden)
            mask_k1 = mask_p & (k1 < hidden)
            x0 = tl.load(x_ptr + x_base + k0, mask=mask_k0, other=0.0).to(tl.float32)
            x1 = tl.load(x_ptr + x_base + k1, mask=mask_k1, other=0.0).to(tl.float32)

            q_offsets = q_base + offs_m[:, None] * packed_hidden + pidx[None, :]
            packed = tl.load(q_weight_ptr + q_offsets, mask=mask_m[:, None] & mask_p[None, :], other=0).to(tl.int32)
            q0_u4 = packed & 0xF
            q1_u4 = (packed >> 4) & 0xF
            q0 = tl.where(q0_u4 >= 8, q0_u4 - 16, q0_u4).to(tl.float32)
            q1 = tl.where(q1_u4 >= 8, q1_u4 - 16, q1_u4).to(tl.float32)
            partial = tl.sum(q0 * x0[None, :] + q1 * x1[None, :], axis=1)
            group_id = (start * 2) // GROUP_SIZE
            scale = tl.load(
                scale_ptr + scale_base + offs_m * num_groups + group_id,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
            acc += partial * scale

        tl.store(out_ptr + out_base + offs_m, acc, mask=mask_m)



def native_int8_gemv_available() -> bool:
    """Return whether the optional Triton int8 dequant-GEMV prototype can run."""

    return bool(_HAS_TRITON and torch is not None)


def native_int4_gemv_available() -> bool:
    """Return whether the optional Triton int4 dequant-GEMV prototype can run."""

    return bool(_HAS_TRITON and torch is not None)


def native_int8_fused_rkv_available() -> bool:
    """Return whether the optional fused int8 R/K/V prototype can run."""

    return bool(_HAS_TRITON and torch is not None)


def native_int4_fused_rkv_available() -> bool:
    """Return whether the optional fused int4 R/K/V prototype can run."""

    return bool(_HAS_TRITON and torch is not None)


def native_int4_stacked_rkv_available() -> bool:
    """Return whether the projection-axis W4A16 R/K/V kernel can run."""

    return bool(_HAS_TRITON and torch is not None)


def quantize_int8_rowwise(weight: Any, *, eps: float = 1e-8):
    """Pack a dense weight matrix into signed row-wise int8 plus fp32 scales.

    Args:
        weight: ``[out_features, in_features]`` tensor.

    Returns:
        ``(q_weight, scales)`` where ``q_weight`` is int8 and ``scales`` is
        fp32 with one scale per output row. The dequantized weight is
        approximately ``q_weight.float() * scales[:, None]``.
    """

    if torch is None:
        raise RuntimeError("quantize_int8_rowwise requires torch")
    if weight.dim() != 2:
        raise ValueError(f"weight must be [out_features, in_features], got {tuple(weight.shape)}")
    w = weight.detach().float()
    scales = (w.abs().amax(dim=1).clamp_min(float(eps)) / 127.0).to(torch.float32)
    q = torch.round(w / scales[:, None]).clamp(-127, 127).to(torch.int8)
    return q.contiguous(), scales.contiguous()


def quantize_int4_rowwise(weight: Any, *, eps: float = 1e-8):
    """Pack a dense weight matrix into signed row-wise int4 plus fp32 scales.

    The packed layout stores two signed 4-bit values per byte.  For each row,
    the even input index is stored in the low nibble and the odd input index is
    stored in the high nibble.  Values use two's-complement nibbles, so the
    dequantized value is approximately ``sign_extend(nibble) * scale[row]``.

    Args:
        weight: ``[out_features, in_features]`` tensor.

    Returns:
        ``(q_weight, scales)`` where ``q_weight`` is uint8 with shape
        ``[out_features, ceil(in_features / 2)]`` and ``scales`` is fp32 with
        one scale per output row.  The original ``in_features`` is recovered
        from the activation passed to :func:`int4_rowwise_gemv`; pass it
        explicitly to :func:`dequantize_int4_rowwise` for standalone unpacking.
    """

    if torch is None:
        raise RuntimeError("quantize_int4_rowwise requires torch")
    if weight.dim() != 2:
        raise ValueError(f"weight must be [out_features, in_features], got {tuple(weight.shape)}")
    w = weight.detach().float()
    out_features, in_features = int(w.shape[0]), int(w.shape[1])
    scales = (w.abs().amax(dim=1).clamp_min(float(eps)) / 7.0).to(torch.float32)
    q = torch.round(w / scales[:, None]).clamp(-7, 7).to(torch.int16)
    q_u4 = torch.bitwise_and(q, 0xF).to(torch.uint8)
    if in_features % 2:
        pad = torch.zeros((out_features, 1), device=q_u4.device, dtype=torch.uint8)
        q_u4 = torch.cat([q_u4, pad], dim=1)
    low = q_u4[:, 0::2]
    high = q_u4[:, 1::2] << 4
    packed = torch.bitwise_or(low, high)
    return packed.contiguous(), scales.contiguous()


def quantize_int4_groupwise(
    weight: Any,
    *,
    group_size: int = 128,
    mse_search: bool = True,
    eps: float = 1e-8,
):
    """Pack signed W4 with independent scales along the input-feature axis.

    Group-local scaling avoids a single outlier setting the quantization step
    for an entire output row. Optional clipping-factor search minimizes weight
    reconstruction MSE per group while retaining the same signed-nibble layout
    consumed by the fast projection-axis kernel.
    """

    if torch is None:
        raise RuntimeError("quantize_int4_groupwise requires torch")
    if weight.dim() != 2:
        raise ValueError(f"weight must be [out_features, in_features], got {tuple(weight.shape)}")
    group_size = int(group_size)
    if group_size <= 0 or group_size % 2:
        raise ValueError(f"group_size must be a positive even integer, got {group_size}")

    w = weight.detach().float()
    out_features, in_features = int(w.shape[0]), int(w.shape[1])
    num_groups = (in_features + group_size - 1) // group_size
    padded_features = num_groups * group_size
    if padded_features != in_features:
        padded = torch.zeros((out_features, padded_features), device=w.device, dtype=w.dtype)
        padded[:, :in_features] = w
        w = padded
    grouped = w.reshape(out_features, num_groups, group_size)
    base_scales = grouped.abs().amax(dim=-1).clamp_min(float(eps)) / 7.0

    if mse_search:
        best_scales = base_scales
        best_q = torch.round(grouped / best_scales[..., None]).clamp(-7, 7)
        best_error = ((grouped - best_q * best_scales[..., None]) ** 2).mean(dim=-1)
        # Clipping a small fraction of outliers usually lowers aggregate W4
        # reconstruction error. Keep this deterministic and calibration-free.
        for factor in (0.98, 0.95, 0.92, 0.89, 0.86, 0.83, 0.80):
            candidate_scales = base_scales * factor
            candidate_q = torch.round(grouped / candidate_scales[..., None]).clamp(-7, 7)
            error = ((grouped - candidate_q * candidate_scales[..., None]) ** 2).mean(dim=-1)
            better = error < best_error
            best_error = torch.where(better, error, best_error)
            best_scales = torch.where(better, candidate_scales, best_scales)
            best_q = torch.where(better[..., None], candidate_q, best_q)
        q = best_q
        scales = best_scales
    else:
        scales = base_scales
        q = torch.round(grouped / scales[..., None]).clamp(-7, 7)

    q = q.reshape(out_features, padded_features)[:, :in_features].to(torch.int16)
    q_u4 = torch.bitwise_and(q, 0xF).to(torch.uint8)
    if in_features % 2:
        q_u4 = torch.cat(
            [q_u4, torch.zeros((out_features, 1), device=q_u4.device, dtype=torch.uint8)],
            dim=1,
        )
    packed = torch.bitwise_or(q_u4[:, 0::2], q_u4[:, 1::2] << 4)
    return packed.contiguous(), scales.to(torch.float16).contiguous()


def _flatten_input(x: Any, in_features: int, *, name: str):
    if torch is None:
        raise RuntimeError("int8_rowwise_gemv requires torch")
    if x.dim() == 3:
        if int(x.shape[1]) != 1 or int(x.shape[2]) != in_features:
            raise ValueError(f"{name} must be [batch, 1, {in_features}] or [batch, {in_features}], got {tuple(x.shape)}")
        return x.reshape(int(x.shape[0]), in_features), True
    if x.dim() == 2:
        if int(x.shape[1]) != in_features:
            raise ValueError(f"{name} must be [batch, 1, {in_features}] or [batch, {in_features}], got {tuple(x.shape)}")
        return x, False
    raise ValueError(f"{name} must be [batch, 1, in_features] or [batch, in_features]")


def dequantize_int8_rowwise(q_weight: Any, scales: Any):
    if torch is None:
        raise RuntimeError("dequantize_int8_rowwise requires torch")
    return q_weight.float() * scales.float().reshape(-1, 1)


def dequantize_int4_rowwise(q_weight: Any, scales: Any, in_features: int):
    """Unpack row-wise signed int4 weights into a dense dequantized matrix."""

    if torch is None:
        raise RuntimeError("dequantize_int4_rowwise requires torch")
    if q_weight.dim() != 2:
        raise ValueError("q_weight must be [out_features, ceil(in_features / 2)]")
    if q_weight.dtype != torch.uint8:
        raise ValueError(f"q_weight must be torch.uint8, got {q_weight.dtype}")
    out_features, packed_in_features = int(q_weight.shape[0]), int(q_weight.shape[1])
    if in_features <= 0:
        raise ValueError(f"in_features must be positive, got {in_features}")
    if (in_features + 1) // 2 > packed_in_features:
        raise ValueError(
            f"packed q_weight only covers {packed_in_features * 2} input features, "
            f"but in_features={in_features}"
        )
    if scales.dim() != 1 or int(scales.shape[0]) != out_features:
        raise ValueError(f"scales must be [{out_features}], got {tuple(scales.shape)}")

    packed = q_weight.to(torch.int16)
    low = torch.bitwise_and(packed, 0xF)
    high = torch.bitwise_and(packed >> 4, 0xF)
    low = torch.where(low >= 8, low - 16, low).to(torch.int8)
    high = torch.where(high >= 8, high - 16, high).to(torch.int8)
    q = torch.empty((out_features, packed_in_features * 2), device=q_weight.device, dtype=torch.int8)
    q[:, 0::2] = low
    q[:, 1::2] = high
    q = q[:, :in_features].float()
    return q * scales.float().reshape(-1, 1)


def dequantize_int4_groupwise(q_weight: Any, scales: Any, in_features: int, group_size: int):
    """Unpack signed W4 weights with ``[out_features, num_groups]`` scales."""

    if scales.dim() != 2:
        raise ValueError("groupwise scales must be [out_features, num_groups]")
    unit_scales = torch.ones(int(q_weight.shape[0]), device=scales.device, dtype=torch.float32)
    q = dequantize_int4_rowwise(q_weight, unit_scales, in_features)
    expected_groups = (int(in_features) + int(group_size) - 1) // int(group_size)
    if tuple(int(v) for v in scales.shape) != (int(q_weight.shape[0]), expected_groups):
        raise ValueError(
            f"scales must be [{int(q_weight.shape[0])}, {expected_groups}], got {tuple(scales.shape)}"
        )
    expanded = scales.float().repeat_interleave(int(group_size), dim=1)[:, :in_features]
    return q * expanded


def int8_rowwise_gemv(
    x: Any,
    q_weight: Any,
    scales: Any,
    bias: Any | None = None,
    *,
    block_m: int = 16,
    block_k: int = 64,
    force_fallback: bool = False,
):
    """Run row-wise int8 dequant GEMV/GEMM for decode-sized batches.

    Inputs may be shaped ``[batch, in_features]`` or ``[batch, 1, in_features]``.
    Outputs preserve the input rank. This prototype assumes int8 weights and
    fp32/fp16/bf16 activations; unsupported hosts fall back to torch.
    """

    if torch is None or F is None:
        raise RuntimeError("int8_rowwise_gemv requires torch")
    if q_weight.dim() != 2:
        raise ValueError("q_weight must be [out_features, in_features]")
    out_features, in_features = int(q_weight.shape[0]), int(q_weight.shape[1])
    if scales.dim() != 1 or int(scales.shape[0]) != out_features:
        raise ValueError(f"scales must be [{out_features}], got {tuple(scales.shape)}")
    x2, had_seq = _flatten_input(x, in_features, name="x")
    if q_weight.dtype != torch.int8:
        raise ValueError(f"q_weight must be torch.int8, got {q_weight.dtype}")
    if bias is not None and (bias.dim() != 1 or int(bias.shape[0]) != out_features):
        raise ValueError(f"bias must be [{out_features}], got {tuple(bias.shape)}")

    use_triton = (
        not force_fallback
        and native_int8_gemv_available()
        and x2.is_cuda
        and q_weight.is_cuda
        and scales.is_cuda
        and (bias is None or bias.is_cuda)
        and x2.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )
    if not use_triton:
        weight = dequantize_int8_rowwise(q_weight, scales).to(dtype=x2.dtype, device=x2.device)
        out = F.linear(x2, weight, bias.to(device=x2.device, dtype=x2.dtype) if bias is not None else None)
    else:
        x_c = x2.contiguous()
        q_c = q_weight.contiguous()
        s_c = scales.contiguous()
        b_c = bias.contiguous() if bias is not None else scales  # unused when HAS_BIAS=False
        out = torch.empty((int(x2.shape[0]), out_features), device=x2.device, dtype=x2.dtype)
        grid = (int(x2.shape[0]), triton.cdiv(out_features, int(block_m)))
        _int8_rowwise_gemv_kernel[grid](
            x_c,
            q_c,
            s_c,
            b_c,
            out,
            in_features,
            out_features,
            HAS_BIAS=bias is not None,
            BLOCK_M=int(block_m),
            BLOCK_K=int(block_k),
            num_warps=4,
        )
    if had_seq:
        return out.unsqueeze(1)
    return out


def _infer_input_features(x: Any, *, name: str) -> int:
    if x.dim() == 3:
        if int(x.shape[1]) != 1:
            raise ValueError(f"{name} must be [batch, 1, in_features] or [batch, in_features], got {tuple(x.shape)}")
        return int(x.shape[2])
    if x.dim() == 2:
        return int(x.shape[1])
    raise ValueError(f"{name} must be [batch, 1, in_features] or [batch, in_features]")


def _validate_int8_square_projection(q_weight: Any, scales: Any, hidden: int, *, name: str) -> None:
    if q_weight.dim() != 2 or int(q_weight.shape[0]) != hidden or int(q_weight.shape[1]) != hidden:
        raise ValueError(f"{name} q_weight must be [{hidden}, {hidden}], got {tuple(q_weight.shape)}")
    if q_weight.dtype != torch.int8:
        raise ValueError(f"{name} q_weight must be torch.int8, got {q_weight.dtype}")
    if scales.dim() != 1 or int(scales.shape[0]) != hidden:
        raise ValueError(f"{name} scales must be [{hidden}], got {tuple(scales.shape)}")


def _validate_int4_square_projection(q_weight: Any, scales: Any, hidden: int, *, name: str) -> None:
    packed_hidden = (int(hidden) + 1) // 2
    if q_weight.dim() != 2 or int(q_weight.shape[0]) != hidden or int(q_weight.shape[1]) != packed_hidden:
        raise ValueError(f"{name} q_weight must be [{hidden}, {packed_hidden}], got {tuple(q_weight.shape)}")
    if q_weight.dtype != torch.uint8:
        raise ValueError(f"{name} q_weight must be torch.uint8, got {q_weight.dtype}")
    if scales.dim() != 1 or int(scales.shape[0]) != hidden:
        raise ValueError(f"{name} scales must be [{hidden}], got {tuple(scales.shape)}")


def int8_fused_rkv_gemv(
    xr: Any,
    xk: Any,
    xv: Any,
    q_r_weight: Any,
    q_k_weight: Any,
    q_v_weight: Any,
    r_scales: Any,
    k_scales: Any,
    v_scales: Any,
    *,
    block_m: int = 16,
    block_k: int = 64,
    force_fallback: bool = False,
):
    """Compute row-wise int8 R/K/V projections with one optional Triton launch.

    This prototype targets the decode-hot RWKV attention projection group:
    ``xr @ r_proj.T``, ``xk @ k_proj.T`` and ``xv @ v_proj.T``.  It keeps the
    normal HF model path untouched and is intended for benchmark telemetry
    before integrating a native fused quant backend.
    """

    if torch is None:
        raise RuntimeError("int8_fused_rkv_gemv requires torch")
    xr2, had_seq = _flatten_input(xr, _infer_input_features(xr, name="xr"), name="xr")
    hidden = int(xr2.shape[1])
    xk2, _ = _flatten_input(xk, hidden, name="xk")
    xv2, _ = _flatten_input(xv, hidden, name="xv")
    if tuple(xr2.shape) != tuple(xk2.shape) or tuple(xr2.shape) != tuple(xv2.shape):
        raise ValueError("xr, xk and xv must have identical flattened shapes")
    _validate_int8_square_projection(q_r_weight, r_scales, hidden, name="r")
    _validate_int8_square_projection(q_k_weight, k_scales, hidden, name="k")
    _validate_int8_square_projection(q_v_weight, v_scales, hidden, name="v")

    use_triton = (
        not force_fallback
        and native_int8_fused_rkv_available()
        and xr2.is_cuda
        and xk2.is_cuda
        and xv2.is_cuda
        and q_r_weight.is_cuda
        and q_k_weight.is_cuda
        and q_v_weight.is_cuda
        and r_scales.is_cuda
        and k_scales.is_cuda
        and v_scales.is_cuda
        and xr2.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )
    if not use_triton:
        r = int8_rowwise_gemv(xr2, q_r_weight, r_scales, force_fallback=True)
        k = int8_rowwise_gemv(xk2, q_k_weight, k_scales, force_fallback=True)
        v = int8_rowwise_gemv(xv2, q_v_weight, v_scales, force_fallback=True)
    else:
        batch = int(xr2.shape[0])
        xr_c = xr2.contiguous()
        xk_c = xk2.contiguous()
        xv_c = xv2.contiguous()
        qr_c = q_r_weight.contiguous()
        qk_c = q_k_weight.contiguous()
        qv_c = q_v_weight.contiguous()
        sr_c = r_scales.contiguous()
        sk_c = k_scales.contiguous()
        sv_c = v_scales.contiguous()
        r = torch.empty((batch, hidden), device=xr2.device, dtype=xr2.dtype)
        k = torch.empty_like(r)
        v = torch.empty_like(r)
        grid = (batch, triton.cdiv(hidden, int(block_m)))
        _int8_fused_rkv_gemv_kernel[grid](
            xr_c,
            xk_c,
            xv_c,
            qr_c,
            qk_c,
            qv_c,
            sr_c,
            sk_c,
            sv_c,
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


def int4_fused_rkv_gemv(
    xr: Any,
    xk: Any,
    xv: Any,
    q_r_weight: Any,
    q_k_weight: Any,
    q_v_weight: Any,
    r_scales: Any,
    k_scales: Any,
    v_scales: Any,
    *,
    block_m: int = 16,
    block_k: int = 64,
    force_fallback: bool = False,
):
    """Compute row-wise int4 R/K/V projections with one optional Triton launch.

    ``block_k`` is interpreted as packed int4 bytes for the Triton path, so one
    loop tile consumes up to ``2 * block_k`` logical input features.
    """

    if torch is None:
        raise RuntimeError("int4_fused_rkv_gemv requires torch")
    xr2, had_seq = _flatten_input(xr, _infer_input_features(xr, name="xr"), name="xr")
    hidden = int(xr2.shape[1])
    packed_hidden = (hidden + 1) // 2
    xk2, _ = _flatten_input(xk, hidden, name="xk")
    xv2, _ = _flatten_input(xv, hidden, name="xv")
    if tuple(xr2.shape) != tuple(xk2.shape) or tuple(xr2.shape) != tuple(xv2.shape):
        raise ValueError("xr, xk and xv must have identical flattened shapes")
    _validate_int4_square_projection(q_r_weight, r_scales, hidden, name="r")
    _validate_int4_square_projection(q_k_weight, k_scales, hidden, name="k")
    _validate_int4_square_projection(q_v_weight, v_scales, hidden, name="v")

    use_triton = (
        not force_fallback
        and native_int4_fused_rkv_available()
        and xr2.is_cuda
        and xk2.is_cuda
        and xv2.is_cuda
        and q_r_weight.is_cuda
        and q_k_weight.is_cuda
        and q_v_weight.is_cuda
        and r_scales.is_cuda
        and k_scales.is_cuda
        and v_scales.is_cuda
        and xr2.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )
    if not use_triton:
        r = int4_rowwise_gemv(xr2, q_r_weight, r_scales, force_fallback=True)
        k = int4_rowwise_gemv(xk2, q_k_weight, k_scales, force_fallback=True)
        v = int4_rowwise_gemv(xv2, q_v_weight, v_scales, force_fallback=True)
    else:
        batch = int(xr2.shape[0])
        xr_c = xr2.contiguous()
        xk_c = xk2.contiguous()
        xv_c = xv2.contiguous()
        qr_c = q_r_weight.contiguous()
        qk_c = q_k_weight.contiguous()
        qv_c = q_v_weight.contiguous()
        sr_c = r_scales.contiguous()
        sk_c = k_scales.contiguous()
        sv_c = v_scales.contiguous()
        r = torch.empty((batch, hidden), device=xr2.device, dtype=xr2.dtype)
        k = torch.empty_like(r)
        v = torch.empty_like(r)
        grid = (batch, triton.cdiv(hidden, int(block_m)))
        _int4_fused_rkv_gemv_kernel[grid](
            xr_c,
            xk_c,
            xv_c,
            qr_c,
            qk_c,
            qv_c,
            sr_c,
            sk_c,
            sv_c,
            r,
            k,
            v,
            hidden,
            packed_hidden,
            BLOCK_M=int(block_m),
            BLOCK_K=int(block_k),
            num_warps=4,
        )
    if had_seq:
        return r.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)
    return r, k, v


def int4_stacked_rkv_gemv(
    x: Any,
    q_weight: Any,
    scales: Any,
    *,
    block_m: int = 16,
    block_k: int = 64,
    num_warps: int = 4,
    force_fallback: bool = False,
):
    """Run a pre-stacked projection-axis W4A16 R/K/V GEMV.

    Args:
        x: Activations with shape ``[batch, 3, hidden]``.
        q_weight: Packed signed-int4 weights with shape
            ``[3, hidden, ceil(hidden / 2)]``.
        scales: Symmetric row scales with shape ``[3, hidden]`` or groupwise
            scales with shape ``[3, hidden, num_groups]``.

    The pre-stacked contract is deliberate: a native RWKV decode path can have
    its shift/mix kernel write R/K/V activations directly into this layout and
    can pack the three projection weights once at load time.  Keeping stack
    copies out of the timed kernel exposes whether projection-axis decomposition
    removes enough register pressure to justify that producer/weight layout.
    """

    if torch is None or F is None:
        raise RuntimeError("int4_stacked_rkv_gemv requires torch")
    if x.dim() != 3 or int(x.shape[1]) != 3:
        raise ValueError(f"x must be [batch, 3, hidden], got {tuple(x.shape)}")
    batch, projections, hidden = (int(x.shape[0]), int(x.shape[1]), int(x.shape[2]))
    packed_hidden = (hidden + 1) // 2
    expected_q_shape = (projections, hidden, packed_hidden)
    if q_weight.dim() != 3 or tuple(int(v) for v in q_weight.shape) != expected_q_shape:
        raise ValueError(f"q_weight must be {expected_q_shape}, got {tuple(q_weight.shape)}")
    if q_weight.dtype != torch.uint8:
        raise ValueError(f"q_weight must be torch.uint8, got {q_weight.dtype}")
    groupwise = scales.dim() == 3
    if groupwise:
        num_groups = int(scales.shape[2])
        if tuple(int(v) for v in scales.shape[:2]) != (projections, hidden) or num_groups <= 0:
            raise ValueError(f"groupwise scales must be [3, {hidden}, num_groups], got {tuple(scales.shape)}")
        group_size = (hidden + num_groups - 1) // num_groups
        if group_size % 2 or (group_size + 1) // 2 != int(block_k):
            raise ValueError(
                f"groupwise W4 requires block_k=group_size/2; got group_size={group_size}, block_k={block_k}"
            )
    else:
        num_groups = 1
        group_size = hidden
        if scales.dim() != 2 or tuple(int(v) for v in scales.shape) != (projections, hidden):
            raise ValueError(f"scales must be [{projections}, {hidden}], got {tuple(scales.shape)}")
    if int(block_m) <= 0 or int(block_k) <= 0:
        raise ValueError("block_m and block_k must be positive")
    if int(num_warps) not in (1, 2, 4, 8):
        raise ValueError("num_warps must be one of 1, 2, 4, 8")

    use_triton = (
        not force_fallback
        and native_int4_stacked_rkv_available()
        and x.is_cuda
        and q_weight.is_cuda
        and scales.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )
    if not use_triton:
        outputs = []
        for projection in range(projections):
            if groupwise:
                weight = dequantize_int4_groupwise(
                    q_weight[projection], scales[projection], hidden, group_size
                ).to(dtype=x.dtype, device=x.device)
            else:
                weight = dequantize_int4_rowwise(q_weight[projection], scales[projection], hidden).to(
                    dtype=x.dtype, device=x.device
                )
            outputs.append(F.linear(x[:, projection, :], weight))
        return torch.stack(outputs, dim=1)

    x_c = x.contiguous()
    q_c = q_weight.contiguous()
    s_c = scales.contiguous()
    out = torch.empty((batch, projections, hidden), device=x.device, dtype=x.dtype)
    grid = (batch, projections, triton.cdiv(hidden, int(block_m)))
    if groupwise:
        _int4_groupwise_stacked_rkv_gemv_kernel[grid](
            x_c, q_c, s_c, out, batch, projections, hidden, packed_hidden,
            num_groups, GROUP_SIZE=int(group_size), BLOCK_M=int(block_m), BLOCK_K=int(block_k),
            num_warps=int(num_warps),
        )
    else:
        _int4_stacked_rkv_gemv_kernel[grid](
            x_c, q_c, s_c, out, batch, projections, hidden, packed_hidden,
            BLOCK_M=int(block_m), BLOCK_K=int(block_k), num_warps=int(num_warps),
        )
    return out


def int4_rowwise_gemv(
    x: Any,
    q_weight: Any,
    scales: Any,
    bias: Any | None = None,
    *,
    block_m: int = 16,
    block_k: int = 64,
    force_fallback: bool = False,
):
    """Run row-wise int4 dequant GEMV/GEMM for decode-sized batches.

    Inputs may be shaped ``[batch, in_features]`` or ``[batch, 1, in_features]``.
    Outputs preserve the input rank.  The original dense input dimension is
    inferred from ``x``; the packed weight must have at least
    ``ceil(in_features / 2)`` bytes per row.  ``block_k`` is interpreted as
    packed int4 bytes for the Triton path, so one loop tile consumes up to
    ``2 * block_k`` logical input features.
    """

    if torch is None or F is None:
        raise RuntimeError("int4_rowwise_gemv requires torch")
    if q_weight.dim() != 2:
        raise ValueError("q_weight must be [out_features, ceil(in_features / 2)]")
    if q_weight.dtype != torch.uint8:
        raise ValueError(f"q_weight must be torch.uint8, got {q_weight.dtype}")
    out_features, packed_in_features = int(q_weight.shape[0]), int(q_weight.shape[1])
    in_features = _infer_input_features(x, name="x")
    if (in_features + 1) // 2 > packed_in_features:
        raise ValueError(
            f"packed q_weight only covers {packed_in_features * 2} input features, "
            f"but x has in_features={in_features}"
        )
    if scales.dim() != 1 or int(scales.shape[0]) != out_features:
        raise ValueError(f"scales must be [{out_features}], got {tuple(scales.shape)}")
    x2, had_seq = _flatten_input(x, in_features, name="x")
    if bias is not None and (bias.dim() != 1 or int(bias.shape[0]) != out_features):
        raise ValueError(f"bias must be [{out_features}], got {tuple(bias.shape)}")

    use_triton = (
        not force_fallback
        and native_int4_gemv_available()
        and x2.is_cuda
        and q_weight.is_cuda
        and scales.is_cuda
        and (bias is None or bias.is_cuda)
        and x2.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )
    if not use_triton:
        weight = dequantize_int4_rowwise(q_weight, scales, in_features).to(dtype=x2.dtype, device=x2.device)
        out = F.linear(x2, weight, bias.to(device=x2.device, dtype=x2.dtype) if bias is not None else None)
    else:
        x_c = x2.contiguous()
        q_c = q_weight.contiguous()
        s_c = scales.contiguous()
        b_c = bias.contiguous() if bias is not None else scales  # unused when HAS_BIAS=False
        out = torch.empty((int(x2.shape[0]), out_features), device=x2.device, dtype=x2.dtype)
        grid = (int(x2.shape[0]), triton.cdiv(out_features, int(block_m)))
        _int4_rowwise_gemv_kernel[grid](
            x_c,
            q_c,
            s_c,
            b_c,
            out,
            in_features,
            packed_in_features,
            out_features,
            HAS_BIAS=bias is not None,
            BLOCK_M=int(block_m),
            BLOCK_K=int(block_k),
            num_warps=4,
        )
    if had_seq:
        return out.unsqueeze(1)
    return out


def int8_weight_footprint_bytes(q_weight: Any, scales: Any, bias: Any | None = None) -> int:
    """Approximate packed weight footprint for telemetry."""

    total = int(q_weight.numel()) + int(scales.numel()) * 4
    if bias is not None:
        total += int(bias.numel()) * int(bias.element_size())
    return total


def int4_weight_footprint_bytes(q_weight: Any, scales: Any, bias: Any | None = None) -> int:
    """Approximate packed W4 weight footprint for telemetry."""

    total = int(q_weight.numel()) + int(scales.numel()) * 4
    if bias is not None:
        total += int(bias.numel()) * int(bias.element_size())
    return total
