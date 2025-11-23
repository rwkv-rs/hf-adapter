# coding=utf-8
"""Optional Apple Metal residual-add plus LayerNorm fusion."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from .mlx_bridge import mlx_available, require_mlx


def metal_add_layer_norm_available() -> bool:
    if not mlx_available():
        return False
    try:
        mx = require_mlx()
        return bool(hasattr(mx, "fast") and hasattr(mx.fast, "metal_kernel"))
    except Exception:
        return False


@lru_cache(maxsize=1)
def _metal_add_layer_norm_kernel():
    mx = require_mlx()
    if not metal_add_layer_norm_available():
        raise RuntimeError("MLX custom Metal kernels are not available in this runtime")
    source = r'''
        uint row = threadgroup_position_in_grid.x;
        uint lane = thread_index_in_threadgroup;
        uint simd_lane = thread_index_in_simdgroup;
        uint simd_group = simdgroup_index_in_threadgroup;
        uint rows = uint(dims[0]);
        if (row >= rows) {
            return;
        }
        constexpr uint ITEMS = HIDDEN / TG;
        constexpr uint SIMD_GROUPS = TG / 32;
        uint base = row * HIDDEN;
        half values[ITEMS];
        float local_sum = 0.0f;
        float local_square_sum = 0.0f;
        for (uint item = 0; item < ITEMS; ++item) {
            uint col = lane + item * TG;
            half z = half(half(residual[base + col]) + half(update[base + col]));
            values[item] = z;
            float zf = float(z);
            local_sum += zf;
            local_square_sum += zf * zf;
        }
        float group_sum = simd_sum(local_sum);
        float group_square_sum = simd_sum(local_square_sum);
        threadgroup float partial_sum[SIMD_GROUPS];
        threadgroup float partial_square_sum[SIMD_GROUPS];
        if (simd_lane == 0) {
            partial_sum[simd_group] = group_sum;
            partial_square_sum[simd_group] = group_square_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float sum = 0.0f;
        float square_sum = 0.0f;
        for (uint group = 0; group < SIMD_GROUPS; ++group) {
            sum += partial_sum[group];
            square_sum += partial_square_sum[group];
        }
        float mean = sum / float(HIDDEN);
        float variance = max(square_sum / float(HIDDEN) - mean * mean, 0.0f);
        float inv_std = metal::fast::rsqrt(variance + 1.0e-5f);
        for (uint item = 0; item < ITEMS; ++item) {
            uint col = lane + item * TG;
            half z = values[item];
            residual_out[base + col] = z;
            half normalized = half((float(z) - mean) * inv_std);
            norm_out[base + col] = half(
                normalized * half(weight[col]) + half(bias[col])
            );
        }
    '''
    return mx.fast.metal_kernel(
        name="rwkv7_add_layer_norm_fp16",
        input_names=["residual", "update", "weight", "bias", "dims"],
        output_names=["residual_out", "norm_out"],
        source=source,
        ensure_row_contiguous=True,
    )


def add_layer_norm_metal_fp16(
    residual: Any,
    update: Any,
    weight: Any,
    bias: Any,
    eps: float,
) -> tuple[Any, Any]:
    """Return ``(residual + update, layer_norm(residual + update))``."""

    mx = require_mlx()
    hidden = int(residual.shape[-1])
    if residual.dtype != mx.float16 or update.dtype != mx.float16:
        raise ValueError("fused add+LayerNorm requires fp16 inputs")
    if hidden % 256:
        raise ValueError("fused add+LayerNorm requires hidden size divisible by 256")
    if abs(float(eps) - 1.0e-5) > 1.0e-12:
        raise ValueError("fused add+LayerNorm currently requires eps=1e-5")
    rows = int(residual.size) // hidden
    dims = mx.array([rows, hidden], dtype=mx.uint32)
    outputs = _metal_add_layer_norm_kernel()(
        inputs=[residual, update, weight.reshape(hidden), bias.reshape(hidden), dims],
        template=[("HIDDEN", hidden), ("TG", 256)],
        grid=(rows * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[residual.shape, residual.shape],
        output_dtypes=[mx.float16, mx.float16],
    )
    return outputs[0], outputs[1]
