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

try:  # pragma: no cover
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from .native_quant_policy import normalize_native_mm_policy, should_quantize_linear

try:
    from .sm70_quant import is_sm70, quantize_w4_row, w4_linear as sm70_w4_linear
except Exception:  # pragma: no cover
    is_sm70 = lambda _device=None: False  # type: ignore[assignment]
    quantize_w4_row = None  # type: ignore[assignment]
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


def mm4_gemv_available(device=None) -> bool:
    if not (_HAS_TRITON and torch is not None and torch.cuda.is_available()):
        return False
    if device is None:
        return True
    return torch.device(device).type == "cuda"


def mm4_gemv_triton(x, packed, mx, rx_s, my, ry_s, m_orig, *, block_pairs=64, block_n=64):
    """Fused int4 dequant GEMV: ``x: [N]`` -> ``[M_orig]``."""
    if not (x.is_cuda and mm4_gemv_available(x.device)):
        return mm4_matmul(x, packed, mx, rx_s, my, ry_s, m_orig)
    n = packed.shape[0]
    mh = packed.shape[1]
    m_padded = mh * 2
    y = torch.empty(m_padded, device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(mh, block_pairs),)
    _mm4_gemv_kernel[grid](
        x, packed, mx.reshape(-1), rx_s.reshape(-1), my.reshape(-1), ry_s.reshape(-1), y,
        n, m_padded, mh, BLOCK_PAIRS=block_pairs, BLOCK_N=block_n, num_warps=4)
    return y[:m_orig]


def mm4_matmul_triton(x, packed, mx, rx_s, my, ry_s, m_orig, *, max_gemv_rows: int = 4):
    """Fused int4 dequant matmul with safe fallbacks (see native_quant_mm8)."""
    if not (x.is_cuda and mm4_gemv_available(x.device)):
        return mm4_matmul(x, packed, mx, rx_s, my, ry_s, m_orig)
    if x.dim() == 1:
        return mm4_gemv_triton(x, packed, mx, rx_s, my, ry_s, m_orig)
    if x.dim() != 2 or int(x.shape[0]) > int(max_gemv_rows):
        return mm4_matmul(x, packed, mx, rx_s, my, ry_s, m_orig)
    if int(x.shape[0]) == 1:
        return mm4_gemv_triton(x[0], packed, mx, rx_s, my, ry_s, m_orig).unsqueeze(0)
    outs = [mm4_gemv_triton(x[i], packed, mx, rx_s, my, ry_s, m_orig) for i in range(x.shape[0])]
    return torch.stack(outs, dim=0)


class MM4Linear(torch.nn.Module):
    """Drop-in for ``nn.Linear`` storing int4 (mm4) packed weights + dequant on forward."""

    def __init__(self, linear, *, fused=True):
        super().__init__()
        self.in_features, self.out_features = linear.weight.shape[1], linear.weight.shape[0]
        self.sm70_rowwise = bool(is_sm70(linear.weight.device) and quantize_w4_row is not None)
        if self.sm70_rowwise:
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
        if self.sm70_rowwise and sm70_w4_linear is not None and self.bias is None:
            return sm70_w4_linear(x, self.packed_row, self.row_scale, self.out_features, self.in_features, out=out)
        result = self.forward(x)
        out.copy_(result)
        return out

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, mm4(fused={self.fused})"


def quantize_model_mm4(
    model,
    *,
    min_params: int = 8_000_000,
    fused: bool = True,
    policy: str = "memory",
) -> int:
    """Swap eligible ``nn.Linear`` modules for :class:`MM4Linear`.

    ``policy="memory"`` quantizes every size-gated Linear. ``policy="speed"``
    quantizes only ``lm_head`` so cached decode stays dense through per-layer
    FFN/recurrent projections until fused quantized block kernels are available.
    """
    if torch is None:
        raise RuntimeError("quantize_model_mm4 requires torch")
    policy = normalize_native_mm_policy(policy)
    targets = [
        n
        for n, m in model.named_modules()
        if isinstance(m, torch.nn.Linear)
        and should_quantize_linear(n, int(m.weight.numel()), min_params=min_params, policy=policy)
    ]
    for full_name in targets:
        parent_name, _, attr = full_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, MM4Linear(getattr(parent, attr), fused=fused))
    setattr(model, "_rwkv7_native_mm_quantization", "mm4")
    setattr(model, "_rwkv7_native_mm_replaced_modules", len(targets))
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
