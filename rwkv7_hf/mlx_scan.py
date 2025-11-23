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
        uint lane = thread_index_in_threadgroup;
        uint B = uint(dims[0]);
        uint T = uint(dims[1]);
        uint H = uint(dims[2]);
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

        // Keep the complete state row thread-local for the whole sequence.
        // The previous implementation round-tripped every element through
        // ``state_out`` twice per token.  At serving batch sizes that turned
        // the scan into a global-memory bandwidth kernel even though the
        // recurrent row is only 64 fp32 values on released RWKV-7 models.
        float state_row[N];
        threadgroup float4 shared_update[N];
        threadgroup float shared_kk[N];
        for (uint j = 0; j < N; ++j) {
            state_row[j] = float(state[row_base + j]);
        }

        for (uint t = 0; t < T; ++t) {
            uint vec_base = vec_base0 + t * H * N;
            // One threadgroup covers exactly one [b,h] state matrix.  Load
            // the six token vectors once into threadgroup memory instead of
            // having all N row threads reread them from global memory.
            shared_update[lane] = float4(
                float(w[vec_base + lane]),
                float(k[vec_base + lane]),
                float(kka[vec_base + lane]),
                float(r[vec_base + lane])
            );
            shared_kk[lane] = float(kk[vec_base + lane]);
            float v_i = float(v[vec_base + i]);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float dot_kk = 0.0f;
            for (uint l = 0; l < N; ++l) {
                dot_kk += state_row[l] * shared_kk[l];
            }

            float acc = 0.0f;
            for (uint j = 0; j < N; ++j) {
                float4 update = shared_update[j];
                float new_s = state_row[j] * update.x
                            - dot_kk * update.z
                            + v_i * update.y;
                state_row[j] = new_s;
                acc += new_s * update.w;
            }
            out[out_row_base + t * H * N] = acc;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (uint j = 0; j < N; ++j) {
            state_out[row_base + j] = state_row[j];
        }
    '''
    return mx.fast.metal_kernel(
        name="rwkv7_wkv_scan",
        input_names=["state", "w", "v", "k", "kk", "kka", "r", "dims"],
        output_names=["state_out", "out"],
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=1)
def _metal_wkv_scan_post_kernel():
    """FP16 scan with optional recurrence prep and the post epilogue fused."""

    mx = require_mlx()
    if not metal_wkv_scan_available():
        raise RuntimeError("MLX custom Metal kernels are not available in this runtime")

    source = r'''
        uint row_id = thread_position_in_grid.x;
        uint lane = thread_index_in_threadgroup;
        uint simd_lane = thread_index_in_simdgroup;
        uint simd_group = simdgroup_index_in_threadgroup;
        uint B = uint(dims[0]);
        uint T = uint(dims[1]);
        uint H = uint(dims[2]);
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
        uint affine_base = h * N;
        constexpr uint Q = N / 4;

        float4 state_row[Q];
        threadgroup float4 shared_w[Q];
        // K/R/A inputs already cross an fp16 graph boundary.  Keep them in
        // half4 threadgroup storage and widen only in registers, halving the
        // dominant per-token shared-memory traffic of the recurrent loop.
        threadgroup half4 shared_k[Q];
        threadgroup half4 shared_kka[Q];
        threadgroup half4 shared_r[Q];
        threadgroup half4 shared_kk[Q];
        threadgroup float partial_sum[(N + 31) / 32];
        threadgroup float partial_square_sum[(N + 31) / 32];
        threadgroup half partial_sk[(N + 31) / 32];
        threadgroup float shared_kk_inv_norm[1];
        for (uint q = 0; q < Q; ++q) {
            uint j = q * 4;
            state_row[q] = float4(
                float(state[row_base + j]),
                float(state[row_base + j + 1]),
                float(state[row_base + j + 2]),
                float(state[row_base + j + 3])
            );
        }

        for (uint t = 0; t < T; ++t) {
            uint vec_base = vec_base0 + t * H * N;
            if (FUSE_PREP) {
                float kk_square_sum = 0.0f;
                if (lane < Q) {
                    uint j = lane * 4;
                    half4 raw_k = half4(
                        half(k[vec_base + j]), half(k[vec_base + j + 1]),
                        half(k[vec_base + j + 2]), half(k[vec_base + j + 3])
                    );
                    half4 kk_weight = half4(
                        half(kk[affine_base + j]), half(kk[affine_base + j + 1]),
                        half(kk[affine_base + j + 2]), half(kk[affine_base + j + 3])
                    );
                    float4 kk_pre = float4(half4(raw_k * kk_weight));
                    shared_kk[lane] = half4(kk_pre);
                    kk_square_sum = dot(kk_pre, kk_pre);
                }
                float kk_simd_sum = simd_sum(kk_square_sum);
                if (lane == 0) {
                    shared_kk_inv_norm[0] = metal::fast::rsqrt(
                        max(kk_simd_sum, 1.0e-12f)
                    );
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (lane < Q) {
                    uint j = lane * 4;
                    half4 raw_k = half4(
                        half(k[vec_base + j]), half(k[vec_base + j + 1]),
                        half(k[vec_base + j + 2]), half(k[vec_base + j + 3])
                    );
                    float4 raw_a = float4(
                        float(a[vec_base + j]), float(a[vec_base + j + 1]),
                        float(a[vec_base + j + 2]), float(a[vec_base + j + 3])
                    );
                    half4 av = half4(
                        1.0f / (1.0f + metal::fast::exp(-raw_a))
                    );
                    half4 ka = half4(
                        half(k_a[affine_base + j]), half(k_a[affine_base + j + 1]),
                        half(k_a[affine_base + j + 2]), half(k_a[affine_base + j + 3])
                    );
                    half4 normalized_kk = half4(
                        float4(shared_kk[lane]) * shared_kk_inv_norm[0]
                    );
                    half4 adjusted_k = half4(raw_k * half4(half4(1.0h) + (av - half4(1.0h)) * ka));
                    float4 raw_w = float4(
                        float(w[vec_base + j]), float(w[vec_base + j + 1]),
                        float(w[vec_base + j + 2]), float(w[vec_base + j + 3])
                    );
                    float4 sigmoid_w = 1.0f / (1.0f + metal::fast::exp(-raw_w));
                    shared_w[lane] = metal::fast::exp(-0.606531f * sigmoid_w);
                    shared_k[lane] = adjusted_k;
                    shared_kk[lane] = normalized_kk;
                    shared_kka[lane] = half4(normalized_kk * av);
                    shared_r[lane] = half4(
                        half(r[vec_base + j]), half(r[vec_base + j + 1]),
                        half(r[vec_base + j + 2]), half(r[vec_base + j + 3])
                    );
                }
            } else if (lane < Q) {
                uint j = lane * 4;
                shared_w[lane] = float4(
                    float(w[vec_base + j]), float(w[vec_base + j + 1]),
                    float(w[vec_base + j + 2]), float(w[vec_base + j + 3])
                );
                shared_k[lane] = half4(
                    half(k[vec_base + j]), half(k[vec_base + j + 1]),
                    half(k[vec_base + j + 2]), half(k[vec_base + j + 3])
                );
                shared_r[lane] = half4(
                    half(r[vec_base + j]), half(r[vec_base + j + 1]),
                    half(r[vec_base + j + 2]), half(r[vec_base + j + 3])
                );
                shared_kk[lane] = half4(
                    half(kk[vec_base + j]), half(kk[vec_base + j + 1]),
                    half(kk[vec_base + j + 2]), half(kk[vec_base + j + 3])
                );
                half4 av = half4(
                    half(a[vec_base + j]), half(a[vec_base + j + 1]),
                    half(a[vec_base + j + 2]), half(a[vec_base + j + 3])
                );
                shared_kka[lane] = half4(shared_kk[lane] * av);
            }
            float v_i;
            if (FUSE_V) {
                half raw_v_i = half(v[vec_base + i]);
                half first_v_i = half(v_first[vec_base + i]);
                float raw_v_mix_i = float(v_mix[vec_base + i]);
                half v_mix_i = half(
                    1.0f / (1.0f + metal::fast::exp(-raw_v_mix_i))
                );
                v_i = float(half(raw_v_i + half(first_v_i - raw_v_i) * v_mix_i));
            } else {
                v_i = float(v[vec_base + i]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float dot_kk = 0.0f;
            for (uint q = 0; q < Q; ++q) {
                dot_kk += dot(state_row[q], float4(shared_kk[q]));
            }

            float acc = 0.0f;
            for (uint q = 0; q < Q; ++q) {
                float4 new_s = state_row[q] * shared_w[q]
                             - dot_kk * float4(shared_kka[q])
                             + v_i * float4(shared_k[q]);
                state_row[q] = new_s;
                acc += dot(new_s, float4(shared_r[q]));
            }
            // The standalone scan stores FP16 before GroupNorm.  Round here
            // at the same boundary so the fused epilogue sees the same data.
            float lane_value = float(half(acc));
            float lane_sum = simd_sum(lane_value);
            float lane_square_sum = simd_sum(lane_value * lane_value);
            uint lane_q = lane / 4;
            uint lane_component = lane % 4;
            half lane_sk = half(shared_r[lane_q][lane_component])
                         * half(shared_k[lane_q][lane_component])
                         * half(r_k[affine_base + lane]);
            half lane_sk_sum = simd_sum(lane_sk);
            if (simd_lane == 0) {
                partial_sum[simd_group] = lane_sum;
                partial_square_sum[simd_group] = lane_square_sum;
                partial_sk[simd_group] = lane_sk_sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Every lane consumes the same two SIMD-group partials for N=64.
            // Recomputing this tiny scalar epilogue avoids publishing through
            // lane 0 and removes a full threadgroup barrier per token.
            constexpr uint simd_groups = (N + 31) / 32;
            float sum = 0.0f;
            float square_sum = 0.0f;
            half sk = half(0.0h);
            for (uint j = 0; j < simd_groups; ++j) {
                sum += partial_sum[j];
                square_sum += partial_square_sum[j];
                sk += partial_sk[j];
            }
            float mean = sum / float(N);
            float variance = max(square_sum / float(N) - mean * mean, 0.0f);
            float inv_std = metal::fast::rsqrt(variance + float(N) * 1.0e-5f);

            half normalized = half(
                (lane_value - mean) * inv_std
            );
            half y = normalized * half(norm_weight[affine_base + i])
                   + half(norm_bias[affine_base + i]);
            half bonus = sk * half(v_i);
            half gated = (y + bonus) * half(g[vec_base + i]);
            out[out_row_base + t * H * N] = gated;
        }

        for (uint q = 0; q < Q; ++q) {
            uint j = q * 4;
            state_out[row_base + j] = state_row[q][0];
            state_out[row_base + j + 1] = state_row[q][1];
            state_out[row_base + j + 2] = state_row[q][2];
            state_out[row_base + j + 3] = state_row[q][3];
        }
    '''
    return mx.fast.metal_kernel(
        name="rwkv7_wkv_scan_post_fp16",
        input_names=[
            "state",
            "w",
            "v",
            "k",
            "kk",
            "a",
            "r",
            "k_a",
            "v_first",
            "v_mix",
            "norm_weight",
            "norm_bias",
            "r_k",
            "g",
            "dims",
        ],
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
        template=[("N", N)],
        grid=(rows, 1, 1),
        # One head row per threadgroup keeps all lanes on the same input
        # vectors and avoids the register pressure of the old 256-lane group.
        threadgroup=(min(256, max(1, N)), 1, 1),
        output_shapes=[state.shape, (B, T, H, N)],
        output_dtypes=[mx.float32, r.dtype],
    )
    return out, state_out


def wkv_scan_post_metal_fp16(
    state: Any,
    w: Any,
    v: Any,
    k: Any,
    kk: Any,
    a: Any,
    r: Any,
    norm_weight: Any,
    norm_bias: Any,
    r_k: Any,
    g: Any,
    *,
    preprocess: bool = False,
    k_k: Any | None = None,
    k_a: Any | None = None,
    v_first: Any | None = None,
    v_mix: Any | None = None,
) -> tuple[Any, Any]:
    """Fuse scan, per-head GroupNorm, RWKV bonus, and output gate for FP16.

    The returned activations are ready for ``o_proj`` and retain shape
    ``[B,T,H,N]``.  Other dtypes deliberately stay on the generic path.
    """

    mx = require_mlx()
    B, T, H, N = (int(x) for x in w.shape)
    if N % 4:
        raise ValueError(f"fused WKV scan post epilogue requires head_dim divisible by 4, got {N}")
    if tuple(int(x) for x in state.shape) != (B, H, N, N):
        raise ValueError(f"state must be [B,H,N,N] matching w, got {tuple(state.shape)}")
    if r.dtype != mx.float16 or g.dtype != mx.float16:
        raise ValueError("fused WKV scan post epilogue currently requires FP16 r/g inputs")
    if preprocess and (k_k is None or k_a is None):
        raise ValueError("fused WKV scan prep requires both k_k and k_a")
    preprocess_v = bool(preprocess and v_first is not None and v_mix is not None)
    expected_elements = H * N
    for name, value in (
        ("norm_weight", norm_weight),
        ("norm_bias", norm_bias),
        ("r_k", r_k),
    ):
        elements = 1
        for dim in value.shape:
            elements *= int(dim)
        if elements != expected_elements:
            raise ValueError(
                f"{name} must contain {expected_elements} values, got shape {tuple(value.shape)}"
            )
    dims = mx.array([B, T, H, N], dtype=mx.uint32)
    rows = B * H * N
    state_out, out = _metal_wkv_scan_post_kernel()(
        inputs=[
            state,
            w.reshape(B, T, H, N),
            v.reshape(B, T, H, N),
            k.reshape(B, T, H, N),
            (
                k_k.reshape(H * N)
                if preprocess
                else kk.reshape(B, T, H, N)
            ),
            a.reshape(B, T, H, N),
            r.reshape(B, T, H, N),
            (
                k_a.reshape(H * N)
                if preprocess
                else r_k.reshape(H * N)
            ),
            (
                v_first.reshape(B, T, H, N)
                if preprocess_v
                else v.reshape(B, T, H, N)
            ),
            (
                v_mix.reshape(B, T, H, N)
                if preprocess_v
                else v.reshape(B, T, H, N)
            ),
            norm_weight.reshape(H * N),
            norm_bias.reshape(H * N),
            r_k.reshape(H * N),
            g.reshape(B, T, H, N),
            dims,
        ],
        template=[
            ("N", N),
            ("FUSE_PREP", bool(preprocess)),
            ("FUSE_V", preprocess_v),
        ],
        grid=(rows, 1, 1),
        threadgroup=(min(256, max(1, N)), 1, 1),
        output_shapes=[state.shape, (B, T, H, N)],
        output_dtypes=[mx.float32, mx.float16],
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
