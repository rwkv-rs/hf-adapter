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

Weights use the same layout as the native Torch/CUDA helpers: quantize
``W: [N, M]`` used as ``y = x @ W``.  For an HF/torch Linear weight shaped
``[out, in]``, pass ``weight.T`` so ``N=in`` and ``M=out``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .mlx_bridge import mlx_array_nbytes, require_mlx


_EPS = 1e-8


def _mx():
    return require_mlx()


def _weight_dtype(weight: Any):
    return getattr(weight, "dtype", None)


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

    w_u8: Any
    mx: Any
    rx: Any
    my: Any
    ry: Any
    n: int
    m: int
    dense_dtype: Any

    @property
    def bits(self) -> int:
        return 8

    @property
    def storage_bytes(self) -> int:
        return sum(mlx_array_nbytes(x) for x in (self.w_u8, self.mx, self.rx, self.my, self.ry))


@dataclass
class MLXMM4Weight:
    """Packed int4 affine weight for ``y = x @ W``."""

    packed: Any
    mx: Any
    rx_s: Any
    my: Any
    ry_s: Any
    n: int
    m_orig: int
    m_padded: int
    dense_dtype: Any

    @property
    def bits(self) -> int:
        return 4

    @property
    def storage_bytes(self) -> int:
        return sum(mlx_array_nbytes(x) for x in (self.packed, self.mx, self.rx_s, self.my, self.ry_s))


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

        if bits == 8:
            return cls(quantize_mlx_mm8(dense_weight.T), backend=backend)
        if bits == 4:
            return cls(quantize_mlx_mm4(dense_weight.T), backend=backend)
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


def quantize_mlx_mm8(weight: Any) -> MLXMM8Weight:
    """Quantize ``weight [N, M]`` into the RWKV affine int8 layout."""

    mx = _mx()
    dense_dtype = _weight_dtype(weight)
    w, mx_col, rx, my, ry = _affine_minmax(weight)
    w_u8 = mx.clip(mx.floor(w * 256.0), 0, 255).astype(mx.uint8)
    out_dtype = dense_dtype or mx.float16
    q = MLXMM8Weight(
        w_u8=w_u8,
        mx=mx_col.astype(out_dtype),
        rx=(rx / 16.0).astype(out_dtype),
        my=my.astype(out_dtype),
        ry=(ry / 16.0).astype(out_dtype),
        n=int(weight.shape[0]),
        m=int(weight.shape[1]),
        dense_dtype=out_dtype,
    )
    mx.eval(q.w_u8, q.mx, q.rx, q.my, q.ry)
    return q


def dequantize_mlx_mm8(q: MLXMM8Weight, *, out_dtype: Any | None = None) -> Any:
    mx = _mx()
    dtype = out_dtype or q.dense_dtype
    return (q.w_u8.astype(dtype) + 0.5) * q.ry * q.rx + q.my + q.mx


def mm8_matmul_mlx(x: Any, q: MLXMM8Weight, *, backend: str = "affine") -> Any:
    """Run ``x @ dequant(q)`` with a reference or affine MLX backend."""

    mx = _mx()
    if backend == "reference":
        return x @ dequantize_mlx_mm8(q, out_dtype=x.dtype)
    if backend != "affine":
        raise ValueError(f"unsupported MLX mm8 backend {backend!r}")
    x2 = x.reshape(-1, q.n).astype(mx.float32)
    qf = q.w_u8.astype(mx.float32) + 0.5
    # y = (x*ry) @ q_u8 * rx + (x @ my) + sum(x)*mx
    term_q = (x2 * q.ry.reshape(1, q.n)) @ qf
    term_q = term_q * q.rx.reshape(1, q.m)
    term_my = x2 @ q.my.reshape(q.n, 1)
    term_mx = mx.sum(x2, axis=-1, keepdims=True) * q.mx.reshape(1, q.m)
    y = term_q + term_my + term_mx
    return y.astype(x.dtype).reshape(*x.shape[:-1], q.m)


def quantize_mlx_mm4(weight: Any) -> MLXMM4Weight:
    """Quantize ``weight [N, M]`` into packed affine int4 layout."""

    mx = _mx()
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
    out_dtype = dense_dtype or mx.float16
    q = MLXMM4Weight(
        packed=packed,
        mx=mx_col.astype(out_dtype),
        rx_s=(rx / 4.0).astype(out_dtype),
        my=my.astype(out_dtype),
        ry_s=(ry / 4.0).astype(out_dtype),
        n=n,
        m_orig=m_orig,
        m_padded=m_padded,
        dense_dtype=out_dtype,
    )
    mx.eval(q.packed, q.mx, q.rx_s, q.my, q.ry_s)
    return q


def unpack_mlx_mm4(q: MLXMM4Weight, *, out_dtype: Any | None = None) -> Any:
    """Unpack int4 nibbles into a dense uint/float matrix ``[N, M_padded]``."""

    mx = _mx()
    dtype = out_dtype or q.dense_dtype
    lo = mx.bitwise_and(q.packed, 0x0F).astype(dtype)
    hi = mx.bitwise_and(mx.right_shift(q.packed, 4), 0x0F).astype(dtype)
    # Stack pairs then flatten the pair dimension: [N, M/2, 2] -> [N, M].
    return mx.stack([lo, hi], axis=-1).reshape(q.n, q.m_padded)


def dequantize_mlx_mm4(q: MLXMM4Weight, *, out_dtype: Any | None = None) -> Any:
    dtype = out_dtype or q.dense_dtype
    u4 = unpack_mlx_mm4(q, out_dtype=dtype)
    deq = (u4 + 0.5) * q.ry_s * q.rx_s + q.my + q.mx
    return deq[:, : q.m_orig]


def mm4_matmul_mlx(x: Any, q: MLXMM4Weight, *, backend: str = "affine") -> Any:
    """Run ``x @ dequant(q)`` with a reference or affine MLX backend."""

    mx = _mx()
    if backend == "reference":
        return x @ dequantize_mlx_mm4(q, out_dtype=x.dtype)
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
