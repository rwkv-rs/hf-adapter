# coding=utf-8
"""Official-rwkv-style int4 weight quantization (4-bit affine, packed 2/byte).

The 4-bit sibling of :mod:`rwkv7_hf.native_quant_mm8`: same per-row + per-column
affine scheme (mx/rx per-col, my/ry per-row) but 16 levels instead of 256, with
two 4-bit weights packed per uint8 along the output (M) dimension. Reads 4x less
weight bandwidth than fp16 (vs 2x for int8) -> a higher decode-speedup ceiling
on memory-bound layers, at the cost of more quantization error.

Layout: ``weight W: [N, M]`` used as ``y = x @ W``. For an ``nn.Linear`` with
``weight [out, in]`` quantize ``weight.t().contiguous()`` (N=in, M=out) and call
:func:`mm4_linear`.

Packing (along M): ``byte[n, b] = u4[n, 2b] | (u4[n, 2b+1] << 4)``, so two
adjacent output columns share one byte; ``M`` is padded to even.

Dequant (``+0.5`` rounding center, scales stored as ``rx/4``, ``ry/4`` so the
product absorbs the 16-level factor)::

    u4 = (packed[n, m//2] >> (4*(m & 1))) & 0xF
    W_approx = (u4 + 0.5) * ry_s * rx_s + my + mx
"""
from __future__ import annotations

import os

try:  # pragma: no cover
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    from .kernel_policy import current_kernel_policy
except Exception:  # pragma: no cover - remote-code fallback
    current_kernel_policy = None  # type: ignore[assignment]

from .native_quant_policy import normalize_native_mm_policy, should_quantize_linear

try:
    from .sm70_quant import (
        is_sm70,
        quantize_w4_groupwise,
        quantize_w4_row,
        w4_groupwise_linear as sm70_w4_groupwise_linear,
        w4_linear_add as sm70_w4_linear_add,
        w4_linear_relu2 as sm70_w4_linear_relu2,
        w4_linear as sm70_w4_linear,
    )
except Exception:  # pragma: no cover
    is_sm70 = lambda _device=None: False  # type: ignore[assignment]
    quantize_w4_groupwise = None  # type: ignore[assignment]
    quantize_w4_row = None  # type: ignore[assignment]
    sm70_w4_groupwise_linear = None  # type: ignore[assignment]
    sm70_w4_linear_add = None  # type: ignore[assignment]
    sm70_w4_linear_relu2 = None  # type: ignore[assignment]
    sm70_w4_linear = None  # type: ignore[assignment]


def quantize_mm4(weight):
    """Quantize ``weight: [N, M]`` to the 4-bit affine (mm4) format.

    Returns ``(packed_u8, mx, rx_s, my, ry_s, M_orig, M_padded)`` where
    ``packed_u8`` is ``uint8 [N, M_padded//2]`` and scales are in ``weight.dtype``
    (``mx, rx_s`` are ``[M_padded]``; ``my, ry_s`` are ``[N, 1]``).
    """
    if torch is None:
        raise RuntimeError("quantize_mm4 requires torch")
    w = weight.float()
    n, m = w.shape
    m_orig = m
    if m % 2:  # pad output dim to even so it packs cleanly
        w = torch.nn.functional.pad(w, (0, 1))
        m = w.shape[1]
    eps = 1e-8
    if n > m:
        my = w.amin(dim=1, keepdim=True); w = w - my
        mx = w.amin(dim=0); w = w - mx
    else:
        mx = w.amin(dim=0); w = w - mx
        my = w.amin(dim=1, keepdim=True); w = w - my
    rx = w.amax(dim=0).clamp(min=eps); w = w / rx
    ry = w.amax(dim=1, keepdim=True).clamp(min=eps); w = w / ry
    u4 = torch.clamp(torch.floor(w * 16.0), 0, 15).to(torch.uint8)  # [N, M]
    lo = u4[:, 0::2]
    hi = u4[:, 1::2]
    packed = (lo | (hi << 4)).to(torch.uint8).contiguous()  # [N, M//2]
    out = weight.dtype
    return (packed, mx.to(out), (rx / 4.0).to(out), my.to(out), (ry / 4.0).to(out), m_orig, m)


def dequantize_mm4(packed, mx, rx_s, my, ry_s, m_orig, out_dtype=None):
    """Materialize the dequantized weight ``[N, M_orig]`` (reference, not fused)."""
    if torch is None:
        raise RuntimeError("dequantize_mm4 requires torch")
    dtype = out_dtype if out_dtype is not None else mx.dtype
    n, mh = packed.shape
    m_padded = mh * 2
    lo = (packed & 0x0F).to(dtype)
    hi = ((packed >> 4) & 0x0F).to(dtype)
    u4 = torch.empty(n, m_padded, dtype=dtype, device=packed.device)
    u4[:, 0::2] = lo
    u4[:, 1::2] = hi
    deq = (u4 + 0.5) * ry_s * rx_s + my + mx  # [N, M_padded]
    return deq[:, :m_orig]


def mm4_matmul(x, packed, mx, rx_s, my, ry_s, m_orig):
    """``y = x @ dequant(W)`` (reference path, materializes the full weight)."""
    if torch is None:
        raise RuntimeError("mm4_matmul requires torch")
    deq = dequantize_mm4(packed, mx, rx_s, my, ry_s, m_orig, out_dtype=x.dtype)
    return x @ deq


def mm4_linear(x, packed, mx, rx_s, my, ry_s, m_orig):
    """Drop-in for ``F.linear(x, weight)`` with pre-quantized ``weight``."""
    return mm4_matmul(x, packed, mx, rx_s, my, ry_s, m_orig)


# --------------------------------------------------------------------------- #
# Fused Triton dequant-matmul (the speed path). Reads packed uint8 + scales,
# unpacks the two 4-bit nibbles per byte in registers, dequantizes, accumulates
# in fp32 -- never materializes the fp16 weight. Mirrors native_quant_mm8's
# hardening: int64 addresses, CUDA-only fused path, prefill/large-batch fallback.
# --------------------------------------------------------------------------- #

try:  # pragma: no cover
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]

_HAS_TRITON = triton is not None and tl is not None


if _HAS_TRITON:

    @triton.jit
    def _mm4_gemv_kernel(
        x_ptr, p_ptr, mx_ptr, rx_ptr, my_ptr, ry_ptr, y_ptr,
        N, M, MH,
        BLOCK_PAIRS: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        """Paired-nibble GEMV. Each program owns BLOCK_PAIRS packed bytes, i.e.
        ``2*BLOCK_PAIRS`` output cols. Loads every packed byte once and extracts
        both the low (even col) and high (odd col) nibble, so the 4-bit bandwidth
        advantage is not wasted on redundant byte loads."""
        pid = tl.program_id(0)
        offs_b = pid * BLOCK_PAIRS + tl.arange(0, BLOCK_PAIRS)   # packed col index
        mask_b = offs_b < MH
        m0 = offs_b * 2                                            # even output cols
        m1 = m0 + 1                                                # odd output cols
        mask0 = m0 < M
        mask1 = m1 < M
        rx0 = tl.load(rx_ptr + m0, mask=mask0, other=0.0).to(tl.float32)
        rx1 = tl.load(rx_ptr + m1, mask=mask1, other=0.0).to(tl.float32)
        mx0 = tl.load(mx_ptr + m0, mask=mask0, other=0.0).to(tl.float32)
        mx1 = tl.load(mx_ptr + m1, mask=mask1, other=0.0).to(tl.float32)
        acc0 = tl.zeros((BLOCK_PAIRS,), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_PAIRS,), dtype=tl.float32)
        offs_n = tl.arange(0, BLOCK_N)
        for n0 in range(0, N, BLOCK_N):
            n = n0 + offs_n
            mask_n = n < N
            x = tl.load(x_ptr + n, mask=mask_n, other=0.0).to(tl.float32)
            ry_n = tl.load(ry_ptr + n, mask=mask_n, other=0.0).to(tl.float32)
            my_n = tl.load(my_ptr + n, mask=mask_n, other=0.0).to(tl.float32)
            addr = n.to(tl.int64)[:, None] * MH + offs_b.to(tl.int64)[None, :]
            byte = tl.load(p_ptr + addr, mask=mask_n[:, None] & mask_b[None, :], other=0).to(tl.int32)
            lo = (byte & 0xF).to(tl.float32)                     # u4 for even cols
            hi = ((byte >> 4) & 0xF).to(tl.float32)              # u4 for odd cols
            deq0 = (lo + 0.5) * ry_n[:, None] * rx0[None, :] + my_n[:, None] + mx0[None, :]
            deq1 = (hi + 0.5) * ry_n[:, None] * rx1[None, :] + my_n[:, None] + mx1[None, :]
            acc0 += tl.sum(x[:, None] * deq0, axis=0)
            acc1 += tl.sum(x[:, None] * deq1, axis=0)
        tl.store(y_ptr + m0, acc0, mask=mask0)
        tl.store(y_ptr + m1, acc1, mask=mask1)

    @triton.jit
    def _mm4_batched_gemv_kernel(
        x_ptr, p_ptr, mx_ptr, rx_ptr, my_ptr, ry_ptr, y_ptr,
        B, N, M, MH,
        BLOCK_PAIRS: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        """Paired-nibble decode GEMV with one launch for all batch rows."""
        pid_b = tl.program_id(0)
        pid_m = tl.program_id(1)
        offs_b = pid_m * BLOCK_PAIRS + tl.arange(0, BLOCK_PAIRS)
        mask_b = offs_b < MH
        m0 = offs_b * 2
        m1 = m0 + 1
        mask0 = m0 < M
        mask1 = m1 < M
        rx0 = tl.load(rx_ptr + m0, mask=mask0, other=0.0).to(tl.float32)
        rx1 = tl.load(rx_ptr + m1, mask=mask1, other=0.0).to(tl.float32)
        mx0 = tl.load(mx_ptr + m0, mask=mask0, other=0.0).to(tl.float32)
        mx1 = tl.load(mx_ptr + m1, mask=mask1, other=0.0).to(tl.float32)
        acc0 = tl.zeros((BLOCK_PAIRS,), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_PAIRS,), dtype=tl.float32)
        offs_n = tl.arange(0, BLOCK_N)
        for n0 in range(0, N, BLOCK_N):
            n = n0 + offs_n
            mask_n = n < N
            x = tl.load(x_ptr + pid_b * N + n, mask=mask_n, other=0.0).to(tl.float32)
            ry_n = tl.load(ry_ptr + n, mask=mask_n, other=0.0).to(tl.float32)
            my_n = tl.load(my_ptr + n, mask=mask_n, other=0.0).to(tl.float32)
            addr = n.to(tl.int64)[:, None] * MH + offs_b.to(tl.int64)[None, :]
            byte = tl.load(p_ptr + addr, mask=mask_n[:, None] & mask_b[None, :], other=0).to(tl.int32)
            lo = (byte & 0xF).to(tl.float32)
            hi = ((byte >> 4) & 0xF).to(tl.float32)
            deq0 = (lo + 0.5) * ry_n[:, None] * rx0[None, :] + my_n[:, None] + mx0[None, :]
            deq1 = (hi + 0.5) * ry_n[:, None] * rx1[None, :] + my_n[:, None] + mx1[None, :]
            acc0 += tl.sum(x[:, None] * deq0, axis=0)
            acc1 += tl.sum(x[:, None] * deq1, axis=0)
        base = pid_b * M
        tl.store(y_ptr + base + m0, acc0, mask=mask0)
        tl.store(y_ptr + base + m1, acc1, mask=mask1)

    @triton.jit
    def _mm4_batched_dot_kernel(
        x_ptr, p_ptr, mx_ptr, rx_ptr, my_ptr, ry_ptr, y_ptr,
        B, N, M, MH,
        BLOCK_B: tl.constexpr, BLOCK_PAIRS: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        """Tensor-core paired-nibble kernel; each packed byte is loaded once."""
        pid_b = tl.program_id(0)
        pid_m = tl.program_id(1)
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        offs_p = pid_m * BLOCK_PAIRS + tl.arange(0, BLOCK_PAIRS)
        m0 = offs_p * 2
        m1 = m0 + 1
        mask_b = offs_b < B
        mask_p = offs_p < MH
        mask0 = m0 < M
        mask1 = m1 < M
        acc0 = tl.zeros((BLOCK_B, BLOCK_PAIRS), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_B, BLOCK_PAIRS), dtype=tl.float32)
        sum_x = tl.zeros((BLOCK_B,), dtype=tl.float32)
        sum_x_my = tl.zeros((BLOCK_B,), dtype=tl.float32)
        offs_n = tl.arange(0, BLOCK_N)
        for n0 in range(0, N, BLOCK_N):
            n = n0 + offs_n
            mask_n = n < N
            x = tl.load(
                x_ptr + offs_b[:, None] * N + n[None, :],
                mask=mask_b[:, None] & mask_n[None, :], other=0.0,
            ).to(tl.float32)
            ry_n = tl.load(ry_ptr + n, mask=mask_n, other=0.0).to(tl.float32)
            my_n = tl.load(my_ptr + n, mask=mask_n, other=0.0).to(tl.float32)
            byte = tl.load(
                p_ptr + n[:, None].to(tl.int64) * MH + offs_p[None, :].to(tl.int64),
                mask=mask_n[:, None] & mask_p[None, :], other=0,
            ).to(tl.int32)
            lo = (byte & 0xF).to(tl.float32)
            hi = ((byte >> 4) & 0xF).to(tl.float32)
            xr = (x * ry_n[None, :]).to(tl.float16)
            acc0 += tl.dot(xr, (lo + 0.5).to(tl.float16))
            acc1 += tl.dot(xr, (hi + 0.5).to(tl.float16))
            sum_x += tl.sum(x, axis=1)
            sum_x_my += tl.sum(x * my_n[None, :], axis=1)
        rx0 = tl.load(rx_ptr + m0, mask=mask0, other=0.0).to(tl.float32)
        rx1 = tl.load(rx_ptr + m1, mask=mask1, other=0.0).to(tl.float32)
        mx0 = tl.load(mx_ptr + m0, mask=mask0, other=0.0).to(tl.float32)
        mx1 = tl.load(mx_ptr + m1, mask=mask1, other=0.0).to(tl.float32)
        base = offs_b[:, None] * M
        out0 = acc0 * rx0[None, :] + sum_x_my[:, None] + sum_x[:, None] * mx0[None, :]
        out1 = acc1 * rx1[None, :] + sum_x_my[:, None] + sum_x[:, None] * mx1[None, :]
        tl.store(y_ptr + base + m0[None, :], out0, mask=mask_b[:, None] & mask0[None, :])
        tl.store(y_ptr + base + m1[None, :], out1, mask=mask_b[:, None] & mask1[None, :])


def mm4_gemv_available(device=None) -> bool:
    if not (_HAS_TRITON and torch is not None and torch.cuda.is_available()):
        return False
    if device is None:
        return True
    return torch.device(device).type == "cuda"


def _mm4_policy_int(x, name: str, default: int) -> int:
    if current_kernel_policy is None or torch is None:
        return int(default)
    try:
        policy = current_kernel_policy(device=x.device, torch_module=torch)
        value = getattr(policy, name, None)
        return int(default if value is None else value)
    except Exception:
        return int(default)


def _mm4_decode_blocks(x, block_pairs, block_n):
    if block_pairs is not None and block_n is not None:
        return int(block_pairs), int(block_n)
    blackwell = bool(x.is_cuda and torch.cuda.get_device_capability(x.device)[0] >= 12)
    default = 128 if blackwell else 64
    policy_pairs = _mm4_policy_int(x, "mm4_gemv_block_pairs", default)
    policy_n = _mm4_policy_int(x, "mm4_gemv_block_n", default)
    pairs = int(block_pairs or os.environ.get("RWKV7_MM4_GEMV_BLOCK_PAIRS", policy_pairs))
    width = int(block_n or os.environ.get("RWKV7_MM4_GEMV_BLOCK_N", policy_n))
    if pairs not in {16, 32, 64, 128} or width not in {16, 32, 64, 128}:
        raise ValueError("RWKV7_MM4_GEMV_BLOCK_PAIRS/BLOCK_N must be 16, 32, 64, or 128")
    return pairs, width


def mm4_effective_launch_config(device=None) -> dict[str, int]:
    """Return the exact policy/env-resolved W4 launch signature."""

    if torch is None:
        return {}
    if device is None:
        resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved = torch.device(device)
    probe = torch.empty((), device=resolved)
    blackwell = bool(
        resolved.type == "cuda"
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability(resolved)[0] >= 12
    )
    pairs, block_n = _mm4_decode_blocks(probe, None, None)
    defaults = {
        "dot_min_rows": _mm4_policy_int(probe, "mm4_dot_min_rows", 4),
        "dot_block_b": _mm4_policy_int(probe, "mm4_dot_block_b", 16),
        "dot_block_pairs": _mm4_policy_int(probe, "mm4_dot_block_pairs", 128),
        "dot_block_n": _mm4_policy_int(probe, "mm4_dot_block_n", 32),
        "dot_warps": _mm4_policy_int(probe, "mm4_dot_warps", 8),
        "fused_max_rows": _mm4_policy_int(
            probe,
            "mm4_fused_max_rows",
            16 if blackwell or mm4_batched_dot_enabled(resolved) else 4,
        ),
    }
    return {
        "gemv_block_pairs": pairs,
        "gemv_block_n": block_n,
        "dot_min_rows": int(os.environ.get("RWKV7_MM4_DOT_MIN_ROWS", defaults["dot_min_rows"])),
        "dot_block_b": int(os.environ.get("RWKV7_MM4_DOT_BLOCK_B", defaults["dot_block_b"])),
        "dot_block_pairs": int(
            os.environ.get("RWKV7_MM4_DOT_BLOCK_PAIRS", defaults["dot_block_pairs"])
        ),
        "dot_block_n": int(os.environ.get("RWKV7_MM4_DOT_BLOCK_N", defaults["dot_block_n"])),
        "dot_warps": int(os.environ.get("RWKV7_MM4_DOT_WARPS", defaults["dot_warps"])),
        "fused_max_rows": int(
            os.environ.get("RWKV7_MM4_FUSED_MAX_ROWS", defaults["fused_max_rows"])
        ),
    }


def _mm4_batched_dot_device_supported(major: int, minor: int, name: str) -> bool:
    """Return whether exact-device tensor-core W4 batch evidence exists."""

    return bool(
        int(major) >= 12
        or (int(major), int(minor)) == (8, 6)
        or (
            (int(major), int(minor)) == (8, 9)
            and "4090" in str(name).lower()
        )
    )


def mm4_batched_dot_enabled(device=None) -> bool:
    """Whether the measured tensor-core W4 batch kernel is enabled.

    The measured ``sm_120+``, consumer/workstation ``sm_86``, and exact RTX
    4090 routes provide the fp16 tensor-core primitive used by
    :func:`mm4_batched_dot_triton`.
    Older code dispatched every ``sm_86`` batch row as a separate GEMV (or
    materialized the dequantized weight above four rows), which made the 2.9B
    lm-head speed-policy lane about 0.65x fp16 at bsz8.

    Keep ``sm_80`` and non-4090 ``sm_89`` devices on their existing routes;
    capability policy must not silently generalize one measured card to every
    architecture peer.
    """
    if torch is None or not torch.cuda.is_available():
        return False
    dev = torch.device("cuda" if device is None else device)
    if dev.type != "cuda":
        return False
    major, minor = torch.cuda.get_device_capability(dev)
    try:
        name = str(torch.cuda.get_device_name(dev))
    except Exception:
        name = ""
    return _mm4_batched_dot_device_supported(major, minor, name)


def mm4_gemv_triton(x, packed, mx, rx_s, my, ry_s, m_orig, *, block_pairs=None, block_n=None):
    """Fused int4 dequant GEMV: ``x: [N]`` -> ``[M_orig]``."""
    if not (x.is_cuda and mm4_gemv_available(x.device)):
        return mm4_matmul(x, packed, mx, rx_s, my, ry_s, m_orig)
    block_pairs, block_n = _mm4_decode_blocks(x, block_pairs, block_n)
    n = packed.shape[0]
    mh = packed.shape[1]
    m_padded = mh * 2
    y = torch.empty(m_padded, device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(mh, block_pairs),)
    _mm4_gemv_kernel[grid](
        x, packed, mx.reshape(-1), rx_s.reshape(-1), my.reshape(-1), ry_s.reshape(-1), y,
        n, m_padded, mh, BLOCK_PAIRS=block_pairs, BLOCK_N=block_n, num_warps=4)
    return y[:m_orig]


def mm4_batched_gemv_triton(x, packed, mx, rx_s, my, ry_s, m_orig, *, block_pairs=None, block_n=None):
    """Fused int4 dequant GEMV for a small contiguous ``[B, N]`` decode batch."""
    if not (x.is_cuda and x.dim() == 2 and mm4_gemv_available(x.device)):
        raise RuntimeError("mm4_batched_gemv_triton requires a 2-D CUDA input")
    block_pairs, block_n = _mm4_decode_blocks(x, block_pairs, block_n)
    x = x.contiguous()
    b, n = x.shape
    if int(n) != int(packed.shape[0]):
        raise ValueError(f"input width {n} does not match quantized weight width {packed.shape[0]}")
    mh = packed.shape[1]
    m_padded = mh * 2
    y = torch.empty((b, m_padded), device=x.device, dtype=x.dtype)
    grid = (b, triton.cdiv(mh, block_pairs))
    _mm4_batched_gemv_kernel[grid](
        x, packed, mx.reshape(-1), rx_s.reshape(-1), my.reshape(-1), ry_s.reshape(-1), y,
        b, n, m_padded, mh, BLOCK_PAIRS=block_pairs, BLOCK_N=block_n, num_warps=4,
    )
    return y[:, :m_orig]


def mm4_batched_dot_triton(
    x,
    packed,
    mx,
    rx_s,
    my,
    ry_s,
    m_orig,
    *,
    block_b=None,
    block_pairs=None,
    block_n=None,
    num_warps=None,
):
    """Tensor-core fp16 W4 path for decode batches large enough to reuse weights."""
    if not (x.is_cuda and x.dim() == 2 and mm4_gemv_available(x.device)):
        raise RuntimeError("mm4_batched_dot_triton requires a 2-D CUDA input")
    if x.dtype != torch.float16:
        raise TypeError("mm4_batched_dot_triton currently requires fp16 input")
    x = x.contiguous()
    b, n = x.shape
    if int(n) != int(packed.shape[0]):
        raise ValueError(f"input width {n} does not match quantized weight width {packed.shape[0]}")
    launch = mm4_effective_launch_config(x.device)
    block_b = int(block_b or launch["dot_block_b"])
    block_pairs = int(block_pairs or launch["dot_block_pairs"])
    block_n = int(block_n or launch["dot_block_n"])
    num_warps = int(num_warps or launch["dot_warps"])
    if block_b not in {16, 32, 64}:
        raise ValueError("RWKV7_MM4_DOT_BLOCK_B must be 16, 32, or 64")
    if block_pairs not in {16, 32, 64, 128}:
        raise ValueError("RWKV7_MM4_DOT_BLOCK_PAIRS must be 16, 32, 64, or 128")
    if block_n not in {16, 32, 64, 128}:
        raise ValueError("RWKV7_MM4_DOT_BLOCK_N must be 16, 32, 64, or 128")
    if num_warps not in {1, 2, 4, 8}:
        raise ValueError("RWKV7_MM4_DOT_WARPS must be 1, 2, 4, or 8")
    mh = packed.shape[1]
    m_padded = mh * 2
    y = torch.empty((b, m_padded), device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(b, block_b), triton.cdiv(mh, block_pairs))
    _mm4_batched_dot_kernel[grid](
        x, packed, mx.reshape(-1), rx_s.reshape(-1), my.reshape(-1), ry_s.reshape(-1), y,
        b, n, m_padded, mh,
        BLOCK_B=block_b,
        BLOCK_PAIRS=block_pairs,
        BLOCK_N=block_n,
        num_warps=num_warps,
    )
    return y[:, :m_orig]


def mm4_matmul_triton(x, packed, mx, rx_s, my, ry_s, m_orig, *, max_gemv_rows: int | None = None):
    """Fused int4 dequant matmul with safe fallbacks (see native_quant_mm8)."""
    if not (x.is_cuda and mm4_gemv_available(x.device)):
        return mm4_matmul(x, packed, mx, rx_s, my, ry_s, m_orig)
    if x.dim() == 1:
        return mm4_gemv_triton(x, packed, mx, rx_s, my, ry_s, m_orig)
    if x.dim() != 2:
        return mm4_matmul(x, packed, mx, rx_s, my, ry_s, m_orig)
    if int(x.shape[0]) == 1:
        return mm4_gemv_triton(x[0], packed, mx, rx_s, my, ry_s, m_orig).unsqueeze(0)
    blackwell = torch.cuda.get_device_capability(x.device)[0] >= 12
    batched_dot = mm4_batched_dot_enabled(x.device)
    launch = mm4_effective_launch_config(x.device)
    if max_gemv_rows is not None:
        row_limit = int(max_gemv_rows)
    else:
        row_limit = int(launch["fused_max_rows"])
        row_limit = min(max(1, row_limit), 65536)
    if int(x.shape[0]) > row_limit:
        return mm4_matmul(x, packed, mx, rx_s, my, ry_s, m_orig)
    if (
        batched_dot
        and int(x.shape[0]) >= int(launch["dot_min_rows"])
        and x.dtype == torch.float16
    ):
        return mm4_batched_dot_triton(x, packed, mx, rx_s, my, ry_s, m_orig)
    if blackwell or batched_dot:
        return mm4_batched_gemv_triton(x, packed, mx, rx_s, my, ry_s, m_orig)
    return torch.stack(
        [mm4_gemv_triton(row, packed, mx, rx_s, my, ry_s, m_orig) for row in x], dim=0,
    )


class MM4Linear(torch.nn.Module):
    """Drop-in for ``nn.Linear`` storing int4 (mm4) packed weights + dequant on forward."""

    def __init__(self, linear, *, fused=True, group_size=0):
        super().__init__()
        self.in_features, self.out_features = linear.weight.shape[1], linear.weight.shape[0]
        self.group_size = int(group_size)
        if self.group_size not in {0, 128, 256}:
            raise ValueError("native MM4 group_size must be 0, 128, or 256")
        self.groupwise = bool(
            self.group_size == 128 and quantize_w4_groupwise is not None
        )
        self.sm70_rowwise = bool(
            not self.groupwise
            and is_sm70(linear.weight.device)
            and quantize_w4_row is not None
        )
        if self.groupwise:
            packed_group, group_scales, packed_inputs = quantize_w4_groupwise(
                linear.weight.data, group_size=self.group_size
            )
            self.register_buffer("packed_group", packed_group)
            self.register_buffer("group_scales", group_scales)
            self.packed_inputs = int(packed_inputs)
            self.m_orig = self.out_features
        elif self.sm70_rowwise:
            packed_row, row_scale, packed_inputs = quantize_w4_row(linear.weight.data)
            self.register_buffer("packed_row", packed_row)
            self.register_buffer("row_scale", row_scale)
            self.packed_inputs = int(packed_inputs)
            self.m_orig = self.out_features
        else:
            packed, mx, rx_s, my, ry_s, m_orig, m_padded = quantize_mm4(linear.weight.data.t().contiguous())
            self.m_orig = m_orig
            self.register_buffer("packed", packed)
            self.register_buffer("mx", mx)
            self.register_buffer("rx_s", rx_s)
            self.register_buffer("my", my)
            self.register_buffer("ry_s", ry_s)
        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.data.clone())
        else:
            self.bias = None
        self.fused = bool(fused)

    def forward(self, x):
        if self.groupwise and sm70_w4_groupwise_linear is not None:
            y = sm70_w4_groupwise_linear(
                x,
                self.packed_group,
                self.group_scales,
                self.out_features,
                self.in_features,
                group_size=self.group_size,
            )
            return y if self.bias is None else y + self.bias
        if self.sm70_rowwise and sm70_w4_linear is not None:
            y = sm70_w4_linear(x, self.packed_row, self.row_scale, self.out_features, self.in_features)
            return y if self.bias is None else y + self.bias
        if x.dim() == 1:
            if self.fused and x.is_cuda and mm4_gemv_available(x.device):
                y = mm4_gemv_triton(x, self.packed, self.mx, self.rx_s, self.my, self.ry_s, self.m_orig)
            else:
                y = mm4_matmul(x, self.packed, self.mx, self.rx_s, self.my, self.ry_s, self.m_orig)
            if self.bias is not None:
                y = y + self.bias
            return y
        leading = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features)
        if self.fused and x2.is_cuda and mm4_gemv_available(x2.device):
            y = mm4_matmul_triton(x2, self.packed, self.mx, self.rx_s, self.my, self.ry_s, self.m_orig)
        else:
            y = mm4_matmul(x2, self.packed, self.mx, self.rx_s, self.my, self.ry_s, self.m_orig)
        y = y.reshape(*leading, self.out_features)
        if self.bias is not None:
            y = y + self.bias
        return y

    def rwkv7_forward_into(self, x, out):
        if self.groupwise and sm70_w4_groupwise_linear is not None and self.bias is None:
            return sm70_w4_groupwise_linear(
                x,
                self.packed_group,
                self.group_scales,
                self.out_features,
                self.in_features,
                group_size=self.group_size,
                out=out,
            )
        if self.sm70_rowwise and sm70_w4_linear is not None and self.bias is None:
            return sm70_w4_linear(x, self.packed_row, self.row_scale, self.out_features, self.in_features, out=out)
        result = self.forward(x)
        out.copy_(result)
        return out

    def rwkv7_forward_relu2(self, x):
        if (
            self.sm70_rowwise
            and sm70_w4_linear_relu2 is not None
            and self.bias is None
            and os.environ.get("RWKV7_SM70_W4_FUSED_EPILOGUE", "0").strip().lower()
            not in {"", "0", "false", "no", "off"}
        ):
            return sm70_w4_linear_relu2(
                x,
                self.packed_row,
                self.row_scale,
                self.out_features,
                self.in_features,
            )
        return torch.relu(self.forward(x)) ** 2

    def rwkv7_forward_add(self, x, residual):
        if (
            self.sm70_rowwise
            and sm70_w4_linear_add is not None
            and self.bias is None
            and os.environ.get("RWKV7_SM70_W4_FUSED_EPILOGUE", "0").strip().lower()
            not in {"", "0", "false", "no", "off"}
        ):
            return sm70_w4_linear_add(
                x,
                self.packed_row,
                self.row_scale,
                residual,
                self.out_features,
                self.in_features,
            )
        return self.forward(x) + residual

    def extra_repr(self):
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"mm4(fused={self.fused}, group_size={self.group_size})"
        )


NATIVE_MM4_GROUP_POLICIES = (
    "all",
    "lm_head",
    "ffn_key",
    "ffn_value",
    "lm_head_and_key",
    "lm_head_and_value",
)


def normalize_native_mm4_group_policy(policy: str | None) -> str:
    value = (policy or "all").strip().lower().replace("-", "_")
    if value not in NATIVE_MM4_GROUP_POLICIES:
        allowed = ", ".join(NATIVE_MM4_GROUP_POLICIES)
        raise ValueError(
            f"unsupported native MM4 group policy {policy!r}; expected: {allowed}"
        )
    return value


def native_mm4_group_size_for_module(
    name: str, group_size: int, group_policy: str
) -> int:
    if int(group_size) == 0:
        return 0
    policy = normalize_native_mm4_group_policy(group_policy)
    is_head = name == "lm_head" or name.endswith(".lm_head")
    is_key = name.endswith(".ffn.key")
    is_value = name.endswith(".ffn.value")
    enabled = {
        "all": True,
        "lm_head": is_head,
        "ffn_key": is_key,
        "ffn_value": is_value,
        "lm_head_and_key": is_head or is_key,
        "lm_head_and_value": is_head or is_value,
    }[policy]
    return int(group_size) if enabled else 0


def quantize_model_mm4(
    model,
    *,
    min_params: int = 8_000_000,
    fused: bool = True,
    policy: str = "memory",
    group_size: int = 0,
    group_policy: str = "all",
) -> int:
    """Swap eligible ``nn.Linear`` modules for :class:`MM4Linear`.

    ``policy="memory"`` quantizes every size-gated Linear. ``policy="speed"``
    quantizes only ``lm_head`` so cached decode stays dense through per-layer
    FFN/recurrent projections until fused quantized block kernels are available.
    """
    if torch is None:
        raise RuntimeError("quantize_model_mm4 requires torch")
    policy = normalize_native_mm_policy(policy)
    group_policy = normalize_native_mm4_group_policy(group_policy)
    targets = [
        n
        for n, m in model.named_modules()
        if isinstance(m, torch.nn.Linear)
        and should_quantize_linear(n, int(m.weight.numel()), min_params=min_params, policy=policy)
    ]
    for full_name in targets:
        parent_name, _, attr = full_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(
            parent,
            attr,
            MM4Linear(
                getattr(parent, attr),
                fused=fused,
                group_size=native_mm4_group_size_for_module(
                    full_name, group_size, group_policy
                ),
            ),
        )
    setattr(model, "_rwkv7_native_mm_quantization", "mm4")
    setattr(model, "_rwkv7_native_mm_replaced_modules", len(targets))
    setattr(model, "_rwkv7_native_mm4_group_size", int(group_size))
    setattr(model, "_rwkv7_native_mm4_group_policy", group_policy)
    setattr(
        model,
        "_rwkv7_native_mm_block_replaced_modules",
        sum(name.startswith("model.layers.") for name in targets),
    )
    for cache_attr in (
        "_rwkv7_native_jit_pack_cache",
        "_rwkv7_native_graph_pack_cache",
        "_rwkv7_native_graph_runner_cache",
        "_rwkv7_native_prefill_graph_runner_cache",
        "_rwkv7_native_prefill_graph_hot_runner",
    ):
        if hasattr(model, cache_attr):
            delattr(model, cache_attr)
    return len(targets)
