# coding=utf-8
"""MLX packed W8/W4 affine dequant-matmul helpers for RWKV-7.

This is the Apple-side sibling of :mod:`rwkv7_hf.native_quant_mm8` and
:mod:`rwkv7_hf.native_quant_mm4`.  It intentionally exposes two execution
styles:

``reference``
    Materialize the approximate dense dequantized matrix, then call MLX matmul.
    This is useful for tests and formula validation.

``affine``
    Compute the same affine quantized matmul without materializing the fp16/fp32
    dequantized weight matrix.  It is still written in portable MLX ops rather
    than a custom Metal kernel, but it is the stable speed-path seam that a
    future fused Metal W8/W4 projection kernel can replace.

``metal``
    Use an optional custom MLX/Metal kernel that fuses dequantization and the
    projection dot product.  Quantized weights are stored in a Metal-friendly
    transposed packed layout so each output column reads contiguous bytes.  This
    is the first Apple W8/W4 fused-kernel seam; it remains opt-in while the
    production speed path is tuned across model sizes and Apple GPUs.

Weights use the same layout as the native Torch/CUDA helpers: quantize
``W: [N, M]`` used as ``y = x @ W``.  For an HF/torch Linear weight shaped
``[out, in]``, pass ``weight.T`` so ``N=in`` and ``M=out``.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from .mlx_bridge import mlx_array_nbytes, mlx_available, require_mlx


_EPS = 1e-8


def _mx():
    return require_mlx()


def _weight_dtype(weight: Any):
    return getattr(weight, "dtype", None)


def metal_quant_available() -> bool:
    """Return whether MLX custom Metal kernels are available for quant matmul."""

    if not mlx_available():
        return False
    try:
        mx = require_mlx()
        return bool(hasattr(mx, "fast") and hasattr(mx.fast, "metal_kernel"))
    except Exception:
        return False


def _affine_minmax(weight: Any):
    """Return ``(w_norm, mx, rx, my, ry)`` for the RWKV affine quantizer."""

    mx = _mx()
    w = weight.astype(mx.float32)
    n, m = int(w.shape[0]), int(w.shape[1])
    if n > m:
        my = mx.min(w, axis=1, keepdims=True)
        w = w - my
        mx_col = mx.min(w, axis=0)
        w = w - mx_col
    else:
        mx_col = mx.min(w, axis=0)
        w = w - mx_col
        my = mx.min(w, axis=1, keepdims=True)
        w = w - my
    rx = mx.maximum(mx.max(w, axis=0), _EPS)
    w = w / rx
    ry = mx.maximum(mx.max(w, axis=1, keepdims=True), _EPS)
    w = w / ry
    return w, mx_col, rx, my, ry


@dataclass
class MLXMM8Weight:
    """Packed int8 affine weight for ``y = x @ W``."""

    w_u8: Any | None
    mx: Any
    rx: Any
    my: Any
    ry: Any
    n: int
    m: int
    dense_dtype: Any
    w_u8_t: Any | None = None

    @property
    def bits(self) -> int:
        return 8

    @property
    def storage_bytes(self) -> int:
        return sum(
            mlx_array_nbytes(x)
            for x in (self.w_u8, self.w_u8_t, self.mx, self.rx, self.my, self.ry)
            if x is not None
        )


@dataclass
class MLXMM4Weight:
    """Packed int4 affine weight for ``y = x @ W``."""

    packed: Any | None
    mx: Any
    rx_s: Any
    my: Any
    ry_s: Any
    n: int
    m_orig: int
    m_padded: int
    dense_dtype: Any
    packed_t: Any | None = None

    @property
    def bits(self) -> int:
        return 4

    @property
    def storage_bytes(self) -> int:
        return sum(
            mlx_array_nbytes(x)
            for x in (self.packed, self.packed_t, self.mx, self.rx_s, self.my, self.ry_s)
            if x is not None
        )


@dataclass
class MLXQuantizedLinear:
    """Quantized MLX Linear weight with reference and affine matmul backends."""

    weight: MLXMM8Weight | MLXMM4Weight
    backend: str = "affine"

    @property
    def bits(self) -> int:
        return int(self.weight.bits)

    @property
    def in_features(self) -> int:
        return int(self.weight.n)

    @property
    def out_features(self) -> int:
        return int(self.weight.m if isinstance(self.weight, MLXMM8Weight) else self.weight.m_orig)

    @property
    def storage_bytes(self) -> int:
        return int(self.weight.storage_bytes)

    @classmethod
    def from_linear_weight(cls, dense_weight: Any, *, bits: int, backend: str = "affine") -> "MLXQuantizedLinear":
        """Quantize an MLX Linear ``weight [out, in]`` for ``linear(x, weight)``."""

        layout = "metal" if backend == "metal" else "standard"
        if bits == 8:
            return cls(quantize_mlx_mm8(dense_weight.T, layout=layout), backend=backend)
        if bits == 4:
            return cls(quantize_mlx_mm4(dense_weight.T, layout=layout), backend=backend)
        raise ValueError(f"unsupported MLX quant bits {bits}; expected 8 or 4")

    def __call__(self, x: Any) -> Any:
        if isinstance(self.weight, MLXMM8Weight):
            return mm8_matmul_mlx(x, self.weight, backend=self.backend)
        return mm4_matmul_mlx(x, self.weight, backend=self.backend)

    def telemetry(self) -> dict[str, Any]:
        return {
            "bits": self.bits,
            "backend": self.backend,
            "in_features": self.in_features,
            "out_features": self.out_features,
            "storage_bytes": self.storage_bytes,
        }


def _mm8_u8(q: MLXMM8Weight) -> Any:
    mx = _mx()
    if q.w_u8 is not None:
        return q.w_u8
    if q.w_u8_t is None:
        raise ValueError("MLXMM8Weight has neither standard nor transposed uint8 storage")
    return mx.transpose(q.w_u8_t)


def _mm8_u8_t(q: MLXMM8Weight) -> Any:
    mx = _mx()
    if q.w_u8_t is not None:
        return q.w_u8_t
    if q.w_u8 is None:
        raise ValueError("MLXMM8Weight has neither standard nor transposed uint8 storage")
    return mx.contiguous(mx.transpose(q.w_u8))


def quantize_mlx_mm8(weight: Any, *, layout: str = "standard") -> MLXMM8Weight:
    """Quantize ``weight [N, M]`` into the RWKV affine int8 layout."""

    mx = _mx()
    layout = (layout or "standard").lower().strip()
    if layout not in {"standard", "metal"}:
        raise ValueError(f"unsupported MLX mm8 layout {layout!r}; expected standard or metal")
    dense_dtype = _weight_dtype(weight)
    w, mx_col, rx, my, ry = _affine_minmax(weight)
    w_u8 = mx.clip(mx.floor(w * 256.0), 0, 255).astype(mx.uint8)
    w_u8_t = mx.contiguous(mx.transpose(w_u8)) if layout == "metal" else None
    out_dtype = dense_dtype or mx.float16
    q = MLXMM8Weight(
        w_u8=None if layout == "metal" else w_u8,
        mx=mx_col.astype(out_dtype),
        rx=(rx / 16.0).astype(out_dtype),
        my=my.astype(out_dtype),
        ry=(ry / 16.0).astype(out_dtype),
        n=int(weight.shape[0]),
        m=int(weight.shape[1]),
        dense_dtype=out_dtype,
        w_u8_t=w_u8_t,
    )
    mx.eval(*[x for x in (q.w_u8, q.w_u8_t, q.mx, q.rx, q.my, q.ry) if x is not None])
    return q


def dequantize_mlx_mm8(q: MLXMM8Weight, *, out_dtype: Any | None = None) -> Any:
    mx = _mx()
    dtype = out_dtype or q.dense_dtype
    return (_mm8_u8(q).astype(dtype) + 0.5) * q.ry * q.rx + q.my + q.mx


@lru_cache(maxsize=1)
def _metal_mm8_kernel():
    mx = require_mlx()
    if not metal_quant_available():
        raise RuntimeError("MLX custom Metal kernels are not available in this runtime")

    source = r'''
        uint row_id = thread_position_in_grid.x;
        uint R = uint(dims[0]);
        uint N = uint(dims[1]);
        uint M = uint(dims[2]);
        uint total = R * M;
        if (row_id >= total) {
            return;
        }

        uint r_id = row_id / M;
        uint m_id = row_id - r_id * M;
        uint x_base = r_id * N;
        uint q_base = m_id * N;

        float rx_m = float(rx[m_id]);
        float mx_m = float(mx_col[m_id]);
        float acc = 0.0f;
        for (uint n = 0; n < N; ++n) {
            float xv = float(x[x_base + n]);
            float qv = float(q_t[q_base + n]) + 0.5f;
            float deq = qv * float(ry[n]) * rx_m + float(my[n]) + mx_m;
            acc += xv * deq;
        }
        out[row_id] = acc;
    '''
    return mx.fast.metal_kernel(
        name="rwkv7_mm8_affine_matmul",
        input_names=["x", "q_t", "mx_col", "rx", "my", "ry", "dims"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


def mm8_matmul_metal(x: Any, q: MLXMM8Weight) -> Any:
    """Run fused Metal ``x @ dequant(q)`` for MM8 affine weights."""

    mx = _mx()
    x2 = x.reshape(-1, q.n)
    rows = int(x2.shape[0])
    dims = mx.array([rows, q.n, q.m], dtype=mx.uint32)
    out = _metal_mm8_kernel()(
        inputs=[
            x2,
            _mm8_u8_t(q),
            q.mx.reshape(q.m),
            q.rx.reshape(q.m),
            q.my.reshape(q.n),
            q.ry.reshape(q.n),
            dims,
        ],
        grid=(rows * q.m, 1, 1),
        threadgroup=(min(256, max(1, q.m)), 1, 1),
        output_shapes=[(rows, q.m)],
        output_dtypes=[x.dtype],
    )[0]
    return out.reshape(*x.shape[:-1], q.m)


def mm8_matmul_mlx(x: Any, q: MLXMM8Weight, *, backend: str = "affine") -> Any:
    """Run ``x @ dequant(q)`` with a reference, affine, or Metal MLX backend."""

    mx = _mx()
    if backend == "reference":
        return x @ dequantize_mlx_mm8(q, out_dtype=x.dtype)
    if backend == "metal":
        return mm8_matmul_metal(x, q)
    if backend != "affine":
        raise ValueError(f"unsupported MLX mm8 backend {backend!r}")
    x2 = x.reshape(-1, q.n).astype(mx.float32)
    qf = _mm8_u8(q).astype(mx.float32) + 0.5
    # y = (x*ry) @ q_u8 * rx + (x @ my) + sum(x)*mx
    term_q = (x2 * q.ry.reshape(1, q.n)) @ qf
    term_q = term_q * q.rx.reshape(1, q.m)
    term_my = x2 @ q.my.reshape(q.n, 1)
    term_mx = mx.sum(x2, axis=-1, keepdims=True) * q.mx.reshape(1, q.m)
    y = term_q + term_my + term_mx
    return y.astype(x.dtype).reshape(*x.shape[:-1], q.m)


def _mm4_packed(q: MLXMM4Weight) -> Any:
    mx = _mx()
    if q.packed is not None:
        return q.packed
    if q.packed_t is None:
        raise ValueError("MLXMM4Weight has neither standard nor transposed packed storage")
    return mx.transpose(q.packed_t)


def _mm4_packed_t(q: MLXMM4Weight) -> Any:
    mx = _mx()
    if q.packed_t is not None:
        return q.packed_t
    if q.packed is None:
        raise ValueError("MLXMM4Weight has neither standard nor transposed packed storage")
    return mx.contiguous(mx.transpose(q.packed))


def quantize_mlx_mm4(weight: Any, *, layout: str = "standard") -> MLXMM4Weight:
    """Quantize ``weight [N, M]`` into packed affine int4 layout."""

    mx = _mx()
    layout = (layout or "standard").lower().strip()
    if layout not in {"standard", "metal"}:
        raise ValueError(f"unsupported MLX mm4 layout {layout!r}; expected standard or metal")
    dense_dtype = _weight_dtype(weight)
    w = weight
    n, m_orig = int(w.shape[0]), int(w.shape[1])
    if m_orig % 2:
        pad = mx.zeros((n, 1), dtype=w.dtype)
        w = mx.concatenate([w, pad], axis=1)
    m_padded = int(w.shape[1])
    w_norm, mx_col, rx, my, ry = _affine_minmax(w)
    u4 = mx.clip(mx.floor(w_norm * 16.0), 0, 15).astype(mx.uint8)
    lo = u4[:, 0::2]
    hi = u4[:, 1::2]
    packed = mx.bitwise_or(lo, mx.left_shift(hi, 4)).astype(mx.uint8)
    packed_t = mx.contiguous(mx.transpose(packed)) if layout == "metal" else None
    out_dtype = dense_dtype or mx.float16
    q = MLXMM4Weight(
        packed=None if layout == "metal" else packed,
        mx=mx_col.astype(out_dtype),
        rx_s=(rx / 4.0).astype(out_dtype),
        my=my.astype(out_dtype),
        ry_s=(ry / 4.0).astype(out_dtype),
        n=n,
        m_orig=m_orig,
        m_padded=m_padded,
        dense_dtype=out_dtype,
        packed_t=packed_t,
    )
    mx.eval(*[x for x in (q.packed, q.packed_t, q.mx, q.rx_s, q.my, q.ry_s) if x is not None])
    return q


def unpack_mlx_mm4(q: MLXMM4Weight, *, out_dtype: Any | None = None) -> Any:
    """Unpack int4 nibbles into a dense uint/float matrix ``[N, M_padded]``."""

    mx = _mx()
    dtype = out_dtype or q.dense_dtype
    packed = _mm4_packed(q)
    lo = mx.bitwise_and(packed, 0x0F).astype(dtype)
    hi = mx.bitwise_and(mx.right_shift(packed, 4), 0x0F).astype(dtype)
    # Stack pairs then flatten the pair dimension: [N, M/2, 2] -> [N, M].
    return mx.stack([lo, hi], axis=-1).reshape(q.n, q.m_padded)


def dequantize_mlx_mm4(q: MLXMM4Weight, *, out_dtype: Any | None = None) -> Any:
    dtype = out_dtype or q.dense_dtype
    u4 = unpack_mlx_mm4(q, out_dtype=dtype)
    deq = (u4 + 0.5) * q.ry_s * q.rx_s + q.my + q.mx
    return deq[:, : q.m_orig]


@lru_cache(maxsize=1)
def _metal_mm4_kernel():
    mx = require_mlx()
    if not metal_quant_available():
        raise RuntimeError("MLX custom Metal kernels are not available in this runtime")

    source = r'''
        uint row_id = thread_position_in_grid.x;
        uint R = uint(dims[0]);
        uint N = uint(dims[1]);
        uint M = uint(dims[2]);
        uint total = R * M;
        if (row_id >= total) {
            return;
        }

        uint r_id = row_id / M;
        uint m_id = row_id - r_id * M;
        uint packed_col = m_id >> 1;
        bool high = (m_id & 1) != 0;
        uint x_base = r_id * N;
        uint q_base = packed_col * N;

        float rx_m = float(rx_s[m_id]);
        float mx_m = float(mx_col[m_id]);
        float acc = 0.0f;
        for (uint n = 0; n < N; ++n) {
            uint byte_v = uint(packed_t[q_base + n]);
            uint q_u4 = high ? ((byte_v >> 4) & 0x0Fu) : (byte_v & 0x0Fu);
            float xv = float(x[x_base + n]);
            float qv = float(q_u4) + 0.5f;
            float deq = qv * float(ry_s[n]) * rx_m + float(my[n]) + mx_m;
            acc += xv * deq;
        }
        out[row_id] = acc;
    '''
    return mx.fast.metal_kernel(
        name="rwkv7_mm4_affine_matmul",
        input_names=["x", "packed_t", "mx_col", "rx_s", "my", "ry_s", "dims"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


def mm4_matmul_metal(x: Any, q: MLXMM4Weight) -> Any:
    """Run fused Metal ``x @ dequant(q)`` for packed MM4 affine weights."""

    mx = _mx()
    x2 = x.reshape(-1, q.n)
    rows = int(x2.shape[0])
    dims = mx.array([rows, q.n, q.m_orig], dtype=mx.uint32)
    out = _metal_mm4_kernel()(
        inputs=[
            x2,
            _mm4_packed_t(q),
            q.mx.reshape(q.m_padded),
            q.rx_s.reshape(q.m_padded),
            q.my.reshape(q.n),
            q.ry_s.reshape(q.n),
            dims,
        ],
        grid=(rows * q.m_orig, 1, 1),
        threadgroup=(min(256, max(1, q.m_orig)), 1, 1),
        output_shapes=[(rows, q.m_orig)],
        output_dtypes=[x.dtype],
    )[0]
    return out.reshape(*x.shape[:-1], q.m_orig)


def mm4_matmul_mlx(x: Any, q: MLXMM4Weight, *, backend: str = "affine") -> Any:
    """Run ``x @ dequant(q)`` with a reference, affine, or Metal MLX backend."""

    mx = _mx()
    if backend == "reference":
        return x @ dequantize_mlx_mm4(q, out_dtype=x.dtype)
    if backend == "metal":
        return mm4_matmul_metal(x, q)
    if backend != "affine":
        raise ValueError(f"unsupported MLX mm4 backend {backend!r}")
    x2 = x.reshape(-1, q.n).astype(mx.float32)
    u4 = unpack_mlx_mm4(q, out_dtype=mx.float32) + 0.5
    term_q = (x2 * q.ry_s.reshape(1, q.n)) @ u4
    term_q = term_q * q.rx_s.reshape(1, q.m_padded)
    term_my = x2 @ q.my.reshape(q.n, 1)
    term_mx = mx.sum(x2, axis=-1, keepdims=True) * q.mx.reshape(1, q.m_padded)
    y = term_q + term_my + term_mx
    return y[:, : q.m_orig].astype(x.dtype).reshape(*x.shape[:-1], q.m_orig)
