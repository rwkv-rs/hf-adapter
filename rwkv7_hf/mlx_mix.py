# coding=utf-8
"""Optional MLX/Metal fused elementwise mix helpers for RWKV-7.

RWKV-7 attention computes six token-mix inputs per layer::

    xx = x_prev - x
    xr = x + xx * x_r
    xw = x + xx * x_w
    xk = x + xx * x_k
    xv = x + xx * x_v
    xa = x + xx * x_a
    xg = x + xx * x_g

In the correctness-first MLX backend those are six high-level elementwise
expressions.  On Apple long-context prefill this contributes to the
per-token/per-layer dispatch count.  This module provides a small optional
Metal seam that computes all six outputs in one custom kernel while keeping a
portable MLX reference path for tests and non-Apple hosts.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from .mlx_bridge import mlx_available, require_mlx


def metal_attn_mix_available() -> bool:
    """Return whether MLX custom Metal kernels are available for mix fusion."""

    if not mlx_available():
        return False
    try:
        mx = require_mlx()
        return bool(hasattr(mx, "fast") and hasattr(mx.fast, "metal_kernel"))
    except Exception:
        return False


@lru_cache(maxsize=1)
def _metal_attn_mix_kernel():
    mx = require_mlx()
    if not metal_attn_mix_available():
        raise RuntimeError("MLX custom Metal kernels are not available in this runtime")

    source = r'''
        uint row_id = thread_position_in_grid.x;
        uint rows = uint(dims[0]);
        uint hidden = uint(dims[1]);
        uint total = rows * hidden;
        if (row_id >= total) {
            return;
        }

        uint h = row_id % hidden;
        float xv0 = float(x[row_id]);
        float xx = float(x_prev[row_id]) - xv0;
        xr[row_id] = xv0 + xx * float(mix_r[h]);
        xw[row_id] = xv0 + xx * float(mix_w[h]);
        xk[row_id] = xv0 + xx * float(mix_k[h]);
        xv[row_id] = xv0 + xx * float(mix_v[h]);
        xa[row_id] = xv0 + xx * float(mix_a[h]);
        xg[row_id] = xv0 + xx * float(mix_g[h]);
    '''
    return mx.fast.metal_kernel(
        name="rwkv7_attn_mix6",
        input_names=["x", "x_prev", "mix_r", "mix_w", "mix_k", "mix_v", "mix_a", "mix_g", "dims"],
        output_names=["xr", "xw", "xk", "xv", "xa", "xg"],
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=1)
def _metal_sequence_mix_kernel():
    """Fuse sequence shifting with the four materialized attention mixes.

    W/A are already handled algebraically by the packed LoRA-down path.  The
    sequence prefill graph still needs ``xx`` for that packed GEMM plus the
    R/K/V/G mixed inputs.  Producing all five tensors directly avoids a
    concatenate, the standalone subtraction, and four independent
    elementwise launches per layer.
    """

    mx = require_mlx()
    if not metal_attn_mix_available():
        raise RuntimeError("MLX custom Metal kernels are not available in this runtime")
    source = r'''
        uint index16 = thread_position_in_grid.x;
        uint B = uint(dims[0]);
        uint T = uint(dims[1]);
        uint hidden = uint(dims[2]);
        uint hidden4 = hidden / 4;
        uint hidden16 = hidden / 16;
        uint total16 = B * T * hidden16;
        if (index16 >= total16) {
            return;
        }

        uint h16 = index16 % hidden16;
        uint h4 = h16 * 4;
        uint row = index16 / hidden16;
        uint t = row % T;
        uint b = row / T;
        uint index4 = index16 * 4;
        const device half4* x4 = reinterpret_cast<const device half4*>(x);
        const device half4* x_prev4 = reinterpret_cast<const device half4*>(x_prev);
        const device half4* mix_r4 = reinterpret_cast<const device half4*>(mix_r);
        const device half4* mix_k4 = reinterpret_cast<const device half4*>(mix_k);
        const device half4* mix_v4 = reinterpret_cast<const device half4*>(mix_v);
        const device half4* mix_g4 = reinterpret_cast<const device half4*>(mix_g);
        device half4* xx4 = reinterpret_cast<device half4*>(xx);
        device half4* xr4 = reinterpret_cast<device half4*>(xr);
        device half4* xk4 = reinterpret_cast<device half4*>(xk);
        device half4* xv4 = reinterpret_cast<device half4*>(xv);
        device half4* xg4 = reinterpret_cast<device half4*>(xg);
        for (uint part = 0; part < 4; ++part) {
            uint q = index4 + part;
            uint mh = h4 + part;
            half4 cur = x4[q];
            half4 prev = t == 0
                ? x_prev4[b * hidden4 + mh]
                : x4[q - hidden4];
            half4 delta = prev - cur;
            xx4[q] = delta;
            xr4[q] = cur + delta * mix_r4[mh];
            xk4[q] = cur + delta * mix_k4[mh];
            xv4[q] = cur + delta * mix_v4[mh];
            xg4[q] = cur + delta * mix_g4[mh];
        }
    '''
    return mx.fast.metal_kernel(
        name="rwkv7_attn_sequence_shift_mix4",
        input_names=["x", "x_prev", "mix_r", "mix_k", "mix_v", "mix_g", "dims"],
        output_names=["xx", "xr", "xk", "xv", "xg"],
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=1)
def _metal_ffn_sequence_mix_kernel():
    mx = require_mlx()
    if not metal_attn_mix_available():
        raise RuntimeError("MLX custom Metal kernels are not available in this runtime")
    source = r'''
        uint index16 = thread_position_in_grid.x;
        uint B = uint(dims[0]);
        uint T = uint(dims[1]);
        uint hidden = uint(dims[2]);
        uint hidden4 = hidden / 4;
        uint hidden16 = hidden / 16;
        uint total16 = B * T * hidden16;
        if (index16 >= total16) {
            return;
        }

        uint h16 = index16 % hidden16;
        uint h4 = h16 * 4;
        uint row = index16 / hidden16;
        uint t = row % T;
        uint b = row / T;
        uint index4 = index16 * 4;
        const device half4* x4 = reinterpret_cast<const device half4*>(x);
        const device half4* x_prev4 = reinterpret_cast<const device half4*>(x_prev);
        const device half4* mix4 = reinterpret_cast<const device half4*>(mix);
        device half4* out4 = reinterpret_cast<device half4*>(out);
        for (uint part = 0; part < 4; ++part) {
            uint q = index4 + part;
            uint mh = h4 + part;
            half4 cur = x4[q];
            half4 prev = t == 0
                ? x_prev4[b * hidden4 + mh]
                : x4[q - hidden4];
            out4[q] = cur + (prev - cur) * mix4[mh];
        }
    '''
    return mx.fast.metal_kernel(
        name="rwkv7_ffn_sequence_shift_mix",
        input_names=["x", "x_prev", "mix", "dims"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


def attn_sequence_mix_metal(
    x: Any,
    x_prev: Any,
    mix_r: Any,
    mix_k: Any,
    mix_v: Any,
    mix_g: Any,
) -> tuple[Any, Any, Any, Any, Any]:
    """Return ``xx, xr, xk, xv, xg`` for a contiguous ``[B,T,H]`` chunk."""

    mx = require_mlx()
    B, T, hidden = (int(dim) for dim in x.shape)
    if hidden % 16:
        raise ValueError("fused attention sequence mix requires hidden size divisible by 16")
    dims = mx.array([B, T, hidden], dtype=mx.uint32)
    outputs = _metal_sequence_mix_kernel()(
        inputs=[
            x,
            x_prev.reshape(B, hidden),
            mix_r.reshape(hidden),
            mix_k.reshape(hidden),
            mix_v.reshape(hidden),
            mix_g.reshape(hidden),
            dims,
        ],
        grid=(B * T * hidden // 16, 1, 1),
        threadgroup=(min(256, max(1, hidden)), 1, 1),
        output_shapes=[x.shape, x.shape, x.shape, x.shape, x.shape],
        output_dtypes=[x.dtype, x.dtype, x.dtype, x.dtype, x.dtype],
    )
    return tuple(outputs)  # type: ignore[return-value]


def ffn_sequence_mix_metal(x: Any, x_prev: Any, mix: Any) -> Any:
    """Fuse the FFN sequence shift and token mix for ``[B,T,H]`` input."""

    mx = require_mlx()
    B, T, hidden = (int(dim) for dim in x.shape)
    if hidden % 16:
        raise ValueError("fused FFN sequence mix requires hidden size divisible by 16")
    dims = mx.array([B, T, hidden], dtype=mx.uint32)
    (out,) = _metal_ffn_sequence_mix_kernel()(
        inputs=[x, x_prev.reshape(B, hidden), mix.reshape(hidden), dims],
        grid=(B * T * hidden // 16, 1, 1),
        threadgroup=(min(256, max(1, hidden)), 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )
    return out


def attn_mix_reference(
    x: Any,
    x_prev: Any,
    mix_r: Any,
    mix_w: Any,
    mix_k: Any,
    mix_v: Any,
    mix_a: Any,
    mix_g: Any,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Portable MLX reference for the six RWKV-7 attention mix tensors."""

    hidden = int(x.shape[-1])
    xx = x_prev - x
    return (
        x + xx * mix_r.reshape(1, hidden),
        x + xx * mix_w.reshape(1, hidden),
        x + xx * mix_k.reshape(1, hidden),
        x + xx * mix_v.reshape(1, hidden),
        x + xx * mix_a.reshape(1, hidden),
        x + xx * mix_g.reshape(1, hidden),
    )


def attn_mix_metal(
    x: Any,
    x_prev: Any,
    mix_r: Any,
    mix_w: Any,
    mix_k: Any,
    mix_v: Any,
    mix_a: Any,
    mix_g: Any,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Run the fused Metal six-way attention mix kernel."""

    mx = require_mlx()
    hidden = int(x.shape[-1])
    x2 = x.reshape(-1, hidden)
    x_prev2 = x_prev.reshape(-1, hidden)
    rows = int(x2.shape[0])
    dims = mx.array([rows, hidden], dtype=mx.uint32)
    outputs = _metal_attn_mix_kernel()(
        inputs=[
            x2,
            x_prev2,
            mix_r.reshape(hidden),
            mix_w.reshape(hidden),
            mix_k.reshape(hidden),
            mix_v.reshape(hidden),
            mix_a.reshape(hidden),
            mix_g.reshape(hidden),
            dims,
        ],
        grid=(rows * hidden, 1, 1),
        threadgroup=(min(256, max(1, hidden)), 1, 1),
        output_shapes=[x2.shape, x2.shape, x2.shape, x2.shape, x2.shape, x2.shape],
        output_dtypes=[x.dtype, x.dtype, x.dtype, x.dtype, x.dtype, x.dtype],
    )
    return tuple(out.reshape(*x.shape) for out in outputs)  # type: ignore[return-value]


def attn_mix(
    x: Any,
    x_prev: Any,
    mix_r: Any,
    mix_w: Any,
    mix_k: Any,
    mix_v: Any,
    mix_a: Any,
    mix_g: Any,
    *,
    backend: str = "reference",
) -> tuple[tuple[Any, Any, Any, Any, Any, Any], str]:
    """Dispatch the six-way RWKV-7 attention mix to reference or Metal.

    Returns ``((xr, xw, xk, xv, xa, xg), backend_used)``.  ``backend=auto``
    uses Metal when available and falls back to the portable MLX expressions.
    """

    choice = (backend or "reference").lower().strip()
    if choice in {"reference", "mlx", "portable"}:
        return attn_mix_reference(x, x_prev, mix_r, mix_w, mix_k, mix_v, mix_a, mix_g), "reference"
    if choice == "auto":
        if metal_attn_mix_available():
            return attn_mix_metal(x, x_prev, mix_r, mix_w, mix_k, mix_v, mix_a, mix_g), "metal"
        return attn_mix_reference(x, x_prev, mix_r, mix_w, mix_k, mix_v, mix_a, mix_g), "reference"
    if choice == "metal":
        return attn_mix_metal(x, x_prev, mix_r, mix_w, mix_k, mix_v, mix_a, mix_g), "metal"
    raise ValueError(f"unsupported MLX attention mix backend {backend!r}; expected reference, metal, or auto")
