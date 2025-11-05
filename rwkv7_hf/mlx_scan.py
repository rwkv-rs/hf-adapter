# coding=utf-8
"""Optional MLX/Metal multi-token RWKV-7 recurrent scan helpers.

This is the first Apple-side "big fused" WKV seam.  The existing
``mlx_wkv.wkv_update_metal`` fuses one token/layer state update.  This module
fuses the recurrent WKV update over a whole sequence for one layer once the
layer projections have already produced ``r/w/v/k/kk/a`` shaped ``[B,T,H,N]``.

It is intentionally standalone before being wired into the full MLX model
prefill path: correctness and kernel shape are validated here first, then the
next step is converting MLX prefill from token-major to layer-major so each
layer can call this scan once per chunk instead of one WKV kernel per token.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from .mlx_bridge import mlx_available, require_mlx
from .mlx_wkv import wkv_update_reference


def metal_wkv_scan_available() -> bool:
    """Return whether MLX custom Metal kernels are available for WKV scan."""

    if not mlx_available():
        return False
    try:
        mx = require_mlx()
        return bool(hasattr(mx, "fast") and hasattr(mx.fast, "metal_kernel"))
    except Exception:
        return False


@lru_cache(maxsize=1)
def _metal_wkv_scan_kernel():
    mx = require_mlx()
    if not metal_wkv_scan_available():
        raise RuntimeError("MLX custom Metal kernels are not available in this runtime")

    source = r'''
        uint row_id = thread_position_in_grid.x;
        uint B = uint(dims[0]);
        uint T = uint(dims[1]);
        uint H = uint(dims[2]);
        uint N = uint(dims[3]);
        uint rows = B * H * N;
        if (row_id >= rows) {
            return;
        }

        uint i = row_id % N;
        uint bh = row_id / N;
        uint h = bh % H;
        uint b = bh / H;
        uint row_base = ((b * H + h) * N + i) * N;
        uint vec_base0 = ((b * T) * H + h) * N;
        uint out_row_base = ((b * T) * H + h) * N + i;

        for (uint t = 0; t < T; ++t) {
            uint vec_base = vec_base0 + t * H * N;
            float dot_kk = 0.0f;
            for (uint l = 0; l < N; ++l) {
                float s_val = (t == 0) ? float(state[row_base + l]) : float(state_out[row_base + l]);
                dot_kk += s_val * float(kk[vec_base + l]);
            }

            float acc = 0.0f;
            for (uint j = 0; j < N; ++j) {
                float s_val = (t == 0) ? float(state[row_base + j]) : float(state_out[row_base + j]);
                float new_s = s_val * float(w[vec_base + j])
                            - dot_kk * float(kka[vec_base + j])
                            + float(v[vec_base + i]) * float(k[vec_base + j]);
                state_out[row_base + j] = new_s;
                acc += new_s * float(r[vec_base + j]);
            }
            out[out_row_base + t * H * N] = acc;
        }
    '''
    return mx.fast.metal_kernel(
        name="rwkv7_wkv_scan",
        input_names=["state", "w", "v", "k", "kk", "kka", "r", "dims"],
        output_names=["state_out", "out"],
        source=source,
        ensure_row_contiguous=True,
    )


def wkv_scan_reference(state: Any, w: Any, v: Any, k: Any, kk: Any, a: Any, r: Any) -> tuple[Any, Any]:
    """Portable MLX sequence scan for one RWKV-7 layer.

    Inputs are ``state [B,H,N,N]`` and per-token tensors ``[B,T,H,N]``.
    Returns ``(out [B,T,H,N], final_state [B,H,N,N])``.
    """

    mx = require_mlx()
    if int(w.ndim) != 4:
        raise ValueError(f"w must be [B,T,H,N], got {tuple(w.shape)}")
    B, T, H, N = (int(x) for x in w.shape)
    cur = state
    outs = []
    for t in range(T):
        out_t, cur = wkv_update_reference(
            cur,
            w[:, t],
            v[:, t],
            k[:, t],
            kk[:, t],
            a[:, t],
            r[:, t],
        )
        outs.append(out_t)
    out = mx.stack(outs, axis=1) if outs else mx.zeros((B, 0, H, N), dtype=r.dtype)
    mx.eval(out, cur)
    return out, cur


def wkv_scan_metal(state: Any, w: Any, v: Any, k: Any, kk: Any, a: Any, r: Any) -> tuple[Any, Any]:
    """Run the fused Metal recurrent WKV scan over ``T`` tokens."""

    mx = require_mlx()
    if int(w.ndim) != 4:
        raise ValueError(f"w must be [B,T,H,N], got {tuple(w.shape)}")
    B, T, H, N = (int(x) for x in w.shape)
    if tuple(int(x) for x in state.shape) != (B, H, N, N):
        raise ValueError(f"state must be [B,H,N,N] matching w, got {tuple(state.shape)} vs {(B,H,N,N)}")
    dims = mx.array([B, T, H, N], dtype=mx.uint32)
    rows = B * H * N
    state_out, out = _metal_wkv_scan_kernel()(
        inputs=[
            state,
            w.reshape(B, T, H, N),
            v.reshape(B, T, H, N),
            k.reshape(B, T, H, N),
            kk.reshape(B, T, H, N),
            (kk * a).reshape(B, T, H, N),
            r.reshape(B, T, H, N),
            dims,
        ],
        grid=(rows, 1, 1),
        threadgroup=(min(256, max(1, rows)), 1, 1),
        output_shapes=[state.shape, (B, T, H, N)],
        output_dtypes=[mx.float32, r.dtype],
    )
    return out, state_out


def wkv_scan(
    state: Any,
    w: Any,
    v: Any,
    k: Any,
    kk: Any,
    a: Any,
    r: Any,
    *,
    backend: str = "reference",
) -> tuple[Any, Any, str]:
    """Dispatch multi-token WKV scan to reference or Metal backend."""

    choice = (backend or "reference").lower().strip()
    if choice in {"reference", "mlx", "portable"}:
        out, new_state = wkv_scan_reference(state, w, v, k, kk, a, r)
        return out, new_state, "reference"
    if choice == "auto":
        if metal_wkv_scan_available():
            out, new_state = wkv_scan_metal(state, w, v, k, kk, a, r)
            return out, new_state, "metal"
        out, new_state = wkv_scan_reference(state, w, v, k, kk, a, r)
        return out, new_state, "reference"
    if choice == "metal":
        out, new_state = wkv_scan_metal(state, w, v, k, kk, a, r)
        return out, new_state, "metal"
    raise ValueError(f"unsupported MLX WKV scan backend {backend!r}; expected reference, metal, or auto")
