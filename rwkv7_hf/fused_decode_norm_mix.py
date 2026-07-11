# coding=utf-8
"""Optional decode-time layer-norm, residual, and time-mix fusions.

These kernels target the one-token native-graph path.  They preserve cuBLAS
for the large dense projections and fuse only the memory-bound boundaries that
Albatross also treats as one operation:

* attention layer norm + six time mixes + attention shift-state update;
* attention residual add + FFN layer norm + FFN mix + FFN state update.

The functions have pure PyTorch fallbacks and update the recurrent shift-state
buffers in place, matching ``native_jit._block_ip`` semantics.
"""
from __future__ import annotations

from typing import Any

try:  # pragma: no cover - optional in lightweight local environments
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
    def _attn_norm_mix6_kernel(
        x_ptr,
        prev_ptr,
        norm_w_ptr,
        norm_b_ptr,
        mix_r_ptr,
        mix_w_ptr,
        mix_k_ptr,
        mix_v_ptr,
        mix_a_ptr,
        mix_g_ptr,
        xr_ptr,
        xw_ptr,
        xk_ptr,
        xv_ptr,
        xa_ptr,
        xg_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < hidden
        base = row * hidden
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / hidden
        centered = tl.where(mask, x - mean, 0.0)
        var = tl.sum(centered * centered, axis=0) / hidden
        weight = tl.load(norm_w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(norm_b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        h = centered * tl.rsqrt(var + eps) * weight + bias
        prev = tl.load(prev_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        delta = prev - h
        tl.store(prev_ptr + base + offs, h, mask=mask)

        mix = tl.load(mix_r_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(xr_ptr + base + offs, h + delta * mix, mask=mask)
        mix = tl.load(mix_w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(xw_ptr + base + offs, h + delta * mix, mask=mask)
        mix = tl.load(mix_k_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(xk_ptr + base + offs, h + delta * mix, mask=mask)
        mix = tl.load(mix_v_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(xv_ptr + base + offs, h + delta * mix, mask=mask)
        mix = tl.load(mix_a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(xa_ptr + base + offs, h + delta * mix, mask=mask)
        mix = tl.load(mix_g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(xg_ptr + base + offs, h + delta * mix, mask=mask)

    @triton.jit
    def _ffn_add_norm_mix_kernel(
        residual_ptr,
        attn_out_ptr,
        prev_ptr,
        norm_w_ptr,
        norm_b_ptr,
        mix_ptr,
        residual_out_ptr,
        mixed_ptr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < hidden
        base = row * hidden
        residual = tl.load(residual_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        attn_out = tl.load(attn_out_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        x = residual + attn_out
        mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / hidden
        centered = tl.where(mask, x - mean, 0.0)
        var = tl.sum(centered * centered, axis=0) / hidden
        weight = tl.load(norm_w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        bias = tl.load(norm_b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        h = centered * tl.rsqrt(var + eps) * weight + bias
        prev = tl.load(prev_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        mix = tl.load(mix_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(residual_out_ptr + base + offs, x, mask=mask)
        tl.store(mixed_ptr + base + offs, h + (prev - h) * mix, mask=mask)
        tl.store(prev_ptr + base + offs, h, mask=mask)


def fused_decode_norm_mix_available() -> bool:
    return bool(_HAS_TRITON and torch is not None)


def _flatten_rows(x: Any, *, name: str) -> tuple[Any, tuple[int, ...]]:
    if torch is None:
        raise RuntimeError("fused decode norm-mix requires torch")
    if x.dim() == 1:
        return x.reshape(1, -1), tuple(x.shape)
    if x.dim() == 2:
        return x, tuple(x.shape)
    raise ValueError(f"{name} must be [hidden] or [batch, hidden], got {tuple(x.shape)}")


def _restore_shape(x: Any, shape: tuple[int, ...]) -> Any:
    return x.reshape(shape)


def _validate_vector(value: Any, hidden: int, *, name: str) -> Any:
    if value is None or int(value.numel()) != hidden:
        shape = None if value is None else tuple(value.shape)
        raise ValueError(f"{name} must contain hidden={hidden} values; got {shape}")
    return value.reshape(hidden)


def _can_use_triton(*values: Any) -> bool:
    if not fused_decode_norm_mix_available():
        return False
    first = values[0]
    return bool(
        first.is_cuda
        and first.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and all(
            value.is_cuda and value.dtype == first.dtype and value.is_contiguous()
            for value in values
        )
    )


def fused_attn_norm_mix6_decode(
    x: Any,
    previous: Any,
    norm_weight: Any,
    norm_bias: Any,
    mix_r: Any,
    mix_w: Any,
    mix_k: Any,
    mix_v: Any,
    mix_a: Any,
    mix_g: Any,
    *,
    eps: float = 1e-5,
    num_warps: int = 4,
    force_fallback: bool = False,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Layer-normalize ``x``, update ``previous``, and emit six time mixes."""

    if torch is None or F is None:
        raise RuntimeError("fused_attn_norm_mix6_decode requires torch")
    x2, shape = _flatten_rows(x, name="x")
    prev2, prev_shape = _flatten_rows(previous, name="previous")
    if prev_shape != shape:
        raise ValueError("x and previous must have identical shapes")
    hidden = int(x2.shape[1])
    params = tuple(
        _validate_vector(value, hidden, name=name)
        for value, name in (
            (norm_weight, "norm_weight"),
            (norm_bias, "norm_bias"),
            (mix_r, "mix_r"),
            (mix_w, "mix_w"),
            (mix_k, "mix_k"),
            (mix_v, "mix_v"),
            (mix_a, "mix_a"),
            (mix_g, "mix_g"),
        )
    )
    use_triton = not force_fallback and _can_use_triton(x2, prev2, *params)
    if not use_triton:
        h = F.layer_norm(x2, (hidden,), params[0], params[1], float(eps))
        delta = prev2 - h
        outputs = tuple(h + delta * mix for mix in params[2:])
        previous.copy_(_restore_shape(h, shape))
        return tuple(_restore_shape(value, shape) for value in outputs)  # type: ignore[return-value]

    x_c = x2.contiguous()
    if x_c.data_ptr() == prev2.data_ptr():
        raise ValueError("x and previous must not alias")
    outputs = tuple(torch.empty_like(x_c) for _ in range(6))
    block = triton.next_power_of_2(hidden)
    _attn_norm_mix6_kernel[(int(x_c.shape[0]),)](
        x_c,
        prev2,
        *params,
        *outputs,
        hidden,
        eps=float(eps),
        BLOCK=int(block),
        num_warps=int(num_warps),
    )
    return tuple(_restore_shape(value, shape) for value in outputs)  # type: ignore[return-value]


def fused_ffn_add_norm_mix_decode(
    residual: Any,
    attn_out: Any,
    previous: Any,
    norm_weight: Any,
    norm_bias: Any,
    mix: Any,
    *,
    eps: float = 1e-5,
    num_warps: int = 4,
    force_fallback: bool = False,
) -> tuple[Any, Any]:
    """Fuse attention residual add, FFN layer norm/mix, and state update."""

    if torch is None or F is None:
        raise RuntimeError("fused_ffn_add_norm_mix_decode requires torch")
    residual2, shape = _flatten_rows(residual, name="residual")
    attn2, attn_shape = _flatten_rows(attn_out, name="attn_out")
    prev2, prev_shape = _flatten_rows(previous, name="previous")
    if attn_shape != shape or prev_shape != shape:
        raise ValueError("residual, attn_out, and previous must have identical shapes")
    hidden = int(residual2.shape[1])
    weight = _validate_vector(norm_weight, hidden, name="norm_weight")
    bias = _validate_vector(norm_bias, hidden, name="norm_bias")
    mix1 = _validate_vector(mix, hidden, name="mix")
    use_triton = not force_fallback and _can_use_triton(residual2, attn2, prev2, weight, bias, mix1)
    if not use_triton:
        x = residual2 + attn2
        h = F.layer_norm(x, (hidden,), weight, bias, float(eps))
        mixed = h + (prev2 - h) * mix1
        previous.copy_(_restore_shape(h, shape))
        return _restore_shape(x, shape), _restore_shape(mixed, shape)

    residual_c = residual2.contiguous()
    attn_c = attn2.contiguous()
    residual_out = torch.empty_like(residual_c)
    mixed = torch.empty_like(residual_c)
    block = triton.next_power_of_2(hidden)
    _ffn_add_norm_mix_kernel[(int(residual_c.shape[0]),)](
        residual_c,
        attn_c,
        prev2,
        weight,
        bias,
        mix1,
        residual_out,
        mixed,
        hidden,
        eps=float(eps),
        BLOCK=int(block),
        num_warps=int(num_warps),
    )
    return _restore_shape(residual_out, shape), _restore_shape(mixed, shape)


__all__ = [
    "fused_attn_norm_mix6_decode",
    "fused_decode_norm_mix_available",
    "fused_ffn_add_norm_mix_decode",
]
