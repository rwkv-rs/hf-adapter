# coding=utf-8
"""Optional MLX/Metal RWKV-7 recurrent WKV update helpers.

The correctness-first MLX backend originally updates the RWKV-7 recurrent state
with three high-level operations per token/layer::

    vk = v[:, :, :, None] @ k[:, :, None, :]
    ab = (-kk)[:, :, :, None] @ (kk * a)[:, :, None, :]
    state = state * w[:, :, None, :] + state @ ab + vk
    out = state @ r[:, :, :, None]

For Apple production work the expensive part is the WKV state update.  This
module exposes a real custom Metal kernel seam that computes the same update
without materializing ``vk`` or ``ab`` and without launching a separate matrix
multiply for ``state @ ab``.  It stays optional/import-safe on non-Apple hosts;
callers opt in with ``backend=\"metal\"`` or ``backend=\"auto\"``.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from .mlx_bridge import mlx_available, require_mlx


def metal_wkv_available() -> bool:
    """Return whether MLX custom Metal kernels are available in this runtime."""

    if not mlx_available():
        return False
    try:
        mx = require_mlx()
        return bool(hasattr(mx, "fast") and hasattr(mx.fast, "metal_kernel"))
    except Exception:
        return False


@lru_cache(maxsize=1)
def _metal_wkv_kernel():
    mx = require_mlx()
    if not metal_wkv_available():
        raise RuntimeError("MLX custom Metal kernels are not available in this runtime")

    source = r'''
        uint row_id = thread_position_in_grid.x;
        uint B = uint(dims[0]);
        uint H = uint(dims[1]);
        uint N = uint(dims[2]);
        uint rows = B * H * N;
        if (row_id >= rows) {
            return;
        }

        uint i = row_id % N;
        uint bh = row_id / N;
        uint hbase = bh * N;
        uint sbase = row_id * N;

        float dot_kk = 0.0f;
        for (uint l = 0; l < N; ++l) {
            dot_kk += float(state[sbase + l]) * float(kk[hbase + l]);
        }

        float acc = 0.0f;
        for (uint j = 0; j < N; ++j) {
            float new_s = float(state[sbase + j]) * float(w[hbase + j])
                        - dot_kk * float(kka[hbase + j])
                        + float(v[hbase + i] * k[hbase + j]);
            state_out[sbase + j] = new_s;
            acc += new_s * float(r[hbase + j]);
        }
        out[row_id] = acc;
    '''
    return mx.fast.metal_kernel(
        name="rwkv7_wkv_update",
        input_names=["state", "w", "v", "k", "kk", "kka", "r", "dims"],
        output_names=["state_out", "out"],
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=1)
def _metal_wkv_b1_n64_kernel():
    """One-simdgroup-per-state-row WKV update for B1/Hx64 decode.

    The generic kernel assigns one thread to a complete 64-element state row.
    That is a good low-launch-overhead fallback, but adjacent SIMD lanes then
    read different rows with a 64-float stride.  B1 decode is latency-bound, so
    use one 32-lane SIMD group per row: every lane handles two contiguous
    columns and both row reductions stay inside ``simd_sum``.
    """

    mx = require_mlx()
    if not metal_wkv_available():
        raise RuntimeError("MLX custom Metal kernels are not available in this runtime")

    source = r'''
        constexpr uint rows_per_threadgroup = 2;
        uint row_id = threadgroup_position_in_grid.x * rows_per_threadgroup
                    + simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint B = uint(dims[0]);
        uint H = uint(dims[1]);
        uint N = uint(dims[2]);
        uint rows = B * H * N;
        if (row_id >= rows || lane >= 32 || N != 64) {
            return;
        }

        uint i = row_id % N;
        uint bh = row_id / N;
        uint hbase = bh * N;
        uint sbase = row_id * N;
        uint j0 = lane;
        uint j1 = lane + 32;

        float s0 = float(state[sbase + j0]);
        float s1 = float(state[sbase + j1]);
        float dot_kk = simd_sum(
            s0 * float(kk[hbase + j0])
            + s1 * float(kk[hbase + j1])
        );

        float vi = float(v[hbase + i]);
        float new0 = s0 * float(w[hbase + j0])
                   - dot_kk * float(kka[hbase + j0])
                   + vi * float(k[hbase + j0]);
        float new1 = s1 * float(w[hbase + j1])
                   - dot_kk * float(kka[hbase + j1])
                   + vi * float(k[hbase + j1]);
        state_out[sbase + j0] = new0;
        state_out[sbase + j1] = new1;

        float acc = simd_sum(
            new0 * float(r[hbase + j0])
            + new1 * float(r[hbase + j1])
        );
        if (lane == 0) {
            out[row_id] = acc;
        }
    '''
    return mx.fast.metal_kernel(
        name="rwkv7_wkv_update_b1_n64_simd",
        input_names=["state", "w", "v", "k", "kk", "kka", "r", "dims"],
        output_names=["state_out", "out"],
        source=source,
        ensure_row_contiguous=True,
    )


def wkv_update_reference(state: Any, w: Any, v: Any, k: Any, kk: Any, a: Any, r: Any) -> tuple[Any, Any]:
    """Portable MLX reference update for ``state [B,H,N,N]``.

    Returns ``(out_heads, new_state)`` where ``out_heads`` is shaped
    ``[B,H,N]`` and ``new_state`` is shaped ``[B,H,N,N]``.  This intentionally
    mirrors the original correctness-first MLX formula, including dtype/order of
    the intermediate ``vk`` and ``ab`` materialization.  Keeping the reference
    path exact prevents the optional Metal seam from changing default behavior.
    """

    mx = require_mlx()
    b, h, n = int(state.shape[0]), int(state.shape[1]), int(state.shape[2])
    vk = v.reshape(b, h, n, 1) @ k.reshape(b, h, 1, n)
    ab = (-kk).reshape(b, h, n, 1) @ (kk * a).reshape(b, h, 1, n)
    new_state = state * w.reshape(b, h, 1, n) + state @ ab.astype(mx.float32) + vk.astype(mx.float32)
    out = (new_state.astype(r.dtype) @ r.reshape(b, h, n, 1)).reshape(b, h, n)
    return out, new_state


def wkv_update_metal(state: Any, w: Any, v: Any, k: Any, kk: Any, a: Any, r: Any) -> tuple[Any, Any]:
    """Run the custom Metal WKV update kernel.

    The kernel computes ``new_state`` in float32 and ``out_heads`` in the dtype
    of ``r``.  Inputs must represent one recurrent decode/prefill token and use
    shapes compatible with ``[B,H,N]`` / ``[B,H,N,N]``.
    """

    mx = require_mlx()
    b, h, n = int(state.shape[0]), int(state.shape[1]), int(state.shape[2])
    if int(state.shape[3]) != n:
        raise ValueError(f"state must be [B,H,N,N], got {tuple(state.shape)}")
    rows = b * h * n
    dims = mx.array([b, h, n], dtype=mx.uint32)
    b1_n64 = b == 1 and n == 64
    kernel = _metal_wkv_b1_n64_kernel() if b1_n64 else _metal_wkv_kernel()
    state_out, out = kernel(
        inputs=[state, w.reshape(b, h, n), v.reshape(b, h, n), k.reshape(b, h, n), kk.reshape(b, h, n), (kk * a).reshape(b, h, n), r.reshape(b, h, n), dims],
        grid=(((((rows + 1) // 2) * 64) if b1_n64 else rows), 1, 1),
        threadgroup=((64 if b1_n64 else min(256, max(1, rows))), 1, 1),
        output_shapes=[state.shape, (b, h, n)],
        output_dtypes=[mx.float32, r.dtype],
    )
    return out, state_out


def wkv_update(state: Any, w: Any, v: Any, k: Any, kk: Any, a: Any, r: Any, *, backend: str = "reference") -> tuple[Any, Any, str]:
    """Dispatch RWKV-7 WKV update to reference or Metal backend.

    Returns ``(out_heads, new_state, backend_used)``.  ``backend=\"auto\"`` uses
    Metal when available and falls back to the portable reference path.
    """

    choice = (backend or "reference").lower().strip()
    if choice in {"reference", "mlx", "portable"}:
        out, new_state = wkv_update_reference(state, w, v, k, kk, a, r)
        return out, new_state, "reference"
    if choice == "auto":
        if metal_wkv_available():
            out, new_state = wkv_update_metal(state, w, v, k, kk, a, r)
            return out, new_state, "metal"
        out, new_state = wkv_update_reference(state, w, v, k, kk, a, r)
        return out, new_state, "reference"
    if choice == "metal":
        out, new_state = wkv_update_metal(state, w, v, k, kk, a, r)
        return out, new_state, "metal"
    raise ValueError(f"unsupported MLX WKV backend {backend!r}; expected reference, metal, or auto")
