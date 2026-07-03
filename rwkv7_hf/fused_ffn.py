# coding=utf-8
"""Optional fused FFN prototypes for RWKV-7 one-token decode.

The prototype keeps the HF model path unchanged.  It combines FFN shift-mix,
key projection, and relu² activation in one Triton launch, then computes the
value projection in a second launch.  Benchmarks decide whether this should be
integrated behind ``rwkv7_forward_token`` later.
"""
from __future__ import annotations

from typing import Any

try:  # pragma: no cover - optional dependency in local no-CUDA tests
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
    def _ffn_norm_shift_prefill_kernel(
        x_ptr,
        cached_prev_h_ptr,
        norm_weight_ptr,
        norm_bias_ptr,
        mix_ptr,
        fk_ptr,
        h_last_ptr,
        tokens: tl.constexpr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        batch = row // tokens
        token = row - batch * tokens
        offs = tl.arange(0, BLOCK_H)
        mask = offs < hidden
        base = row * hidden + offs

        x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / hidden
        centered = tl.where(mask, x - mean, 0.0)
        var = tl.sum(centered * centered, axis=0) / hidden
        weight = tl.load(norm_weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        bias = tl.load(norm_bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        h = centered * tl.rsqrt(var + eps) * weight + bias

        prev_base = ((batch * tokens + token - 1) * hidden + offs)
        prev_x = tl.load(x_ptr + prev_base, mask=mask & (token > 0), other=0.0).to(tl.float32)
        prev_mean = tl.sum(tl.where(mask, prev_x, 0.0), axis=0) / hidden
        prev_centered = tl.where(mask, prev_x - prev_mean, 0.0)
        prev_var = tl.sum(prev_centered * prev_centered, axis=0) / hidden
        prev_calc = prev_centered * tl.rsqrt(prev_var + eps) * weight + bias
        cached_prev = tl.load(cached_prev_h_ptr + batch * hidden + offs, mask=mask, other=0.0).to(tl.float32)
        prev_h = tl.where(token > 0, prev_calc, cached_prev)

        mix = tl.load(mix_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(fk_ptr + base, h + (prev_h - h) * mix, mask=mask)
        tl.store(h_last_ptr + batch * hidden + offs, h, mask=mask & (token == (tokens - 1)))

    @triton.jit
    def _ffn_layernorm_prefill_kernel(
        x_ptr,
        norm_weight_ptr,
        norm_bias_ptr,
        h_ptr,
        h_last_ptr,
        tokens: tl.constexpr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        batch = row // tokens
        token = row - batch * tokens
        offs = tl.arange(0, BLOCK_H)
        mask = offs < hidden
        base = row * hidden + offs

        x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / hidden
        centered = tl.where(mask, x - mean, 0.0)
        var = tl.sum(centered * centered, axis=0) / hidden
        weight = tl.load(norm_weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        bias = tl.load(norm_bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        h = centered * tl.rsqrt(var + eps) * weight + bias

        tl.store(h_ptr + base, h, mask=mask)
        tl.store(h_last_ptr + batch * hidden + offs, h, mask=mask & (token == (tokens - 1)))

    @triton.jit
    def _ffn_shift_from_norm_prefill_kernel(
        h_ptr,
        cached_prev_h_ptr,
        mix_ptr,
        fk_ptr,
        tokens: tl.constexpr,
        hidden: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        batch = row // tokens
        token = row - batch * tokens
        offs = tl.arange(0, BLOCK_H)
        mask = offs < hidden
        base = row * hidden + offs

        h = tl.load(h_ptr + base, mask=mask, other=0.0).to(tl.float32)
        prev_base = ((batch * tokens + token - 1) * hidden + offs)
        prev_h = tl.load(h_ptr + prev_base, mask=mask & (token > 0), other=0.0).to(tl.float32)
        cached_prev = tl.load(cached_prev_h_ptr + batch * hidden + offs, mask=mask, other=0.0).to(tl.float32)
        prev_h = tl.where(token > 0, prev_h, cached_prev)
        mix = tl.load(mix_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(fk_ptr + base, h + (prev_h - h) * mix, mask=mask)

    @triton.jit
    def _ffn_key_relu_kernel(
        x_ptr,
        prev_ptr,
        mix_ptr,
        key_weight_ptr,
        mid_ptr,
        hidden: tl.constexpr,
        intermediate: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        block_id = tl.program_id(1)
        offs_m = block_id * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_K)
        mask_m = offs_m < intermediate
        acc = tl.zeros((BLOCK_M,), tl.float32)
        for start in range(0, hidden, BLOCK_K):
            kidx = start + offs_k
            mask_k = kidx < hidden
            x = tl.load(x_ptr + batch_id * hidden + kidx, mask=mask_k, other=0.0).to(tl.float32)
            prev = tl.load(prev_ptr + batch_id * hidden + kidx, mask=mask_k, other=0.0).to(tl.float32)
            mix = tl.load(mix_ptr + kidx, mask=mask_k, other=0.0).to(tl.float32)
            shifted = x + (prev - x) * mix
            w = tl.load(key_weight_ptr + offs_m[:, None] * hidden + kidx[None, :], mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
            acc += tl.sum(w * shifted[None, :], axis=1)
        relu = tl.maximum(acc, 0.0)
        tl.store(mid_ptr + batch_id * intermediate + offs_m, relu * relu, mask=mask_m)

    @triton.jit
    def _ffn_value_kernel(
        mid_ptr,
        value_weight_ptr,
        out_ptr,
        hidden: tl.constexpr,
        intermediate: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        block_id = tl.program_id(1)
        offs_m = block_id * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_K)
        mask_m = offs_m < hidden
        acc = tl.zeros((BLOCK_M,), tl.float32)
        for start in range(0, intermediate, BLOCK_K):
            kidx = start + offs_k
            mask_k = kidx < intermediate
            h = tl.load(mid_ptr + batch_id * intermediate + kidx, mask=mask_k, other=0.0).to(tl.float32)
            w = tl.load(value_weight_ptr + offs_m[:, None] * intermediate + kidx[None, :], mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
            acc += tl.sum(w * h[None, :], axis=1)
        tl.store(out_ptr + batch_id * hidden + offs_m, acc, mask=mask_m)

    @triton.jit
    def _relu_square_inplace_kernel(
        x_ptr,
        total: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < total
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = tl.maximum(x, 0.0)
        tl.store(x_ptr + offs, y * y, mask=mask)


def fused_ffn_available() -> bool:
    """Return whether the optional Triton FFN prototype can run."""

    return bool(_HAS_TRITON and torch is not None)


def fused_relu_square_available() -> bool:
    """Return whether the single-kernel FFN activation helper can run."""

    return bool(_HAS_TRITON and torch is not None)


def fused_ffn_norm_shift_prefill_available() -> bool:
    """Return whether the optional prefill FFN norm+shift kernel can run."""

    return bool(_HAS_TRITON and torch is not None and F is not None)


def fused_relu_square_inplace(x: Any, *, block_size: int = 1024, force_fallback: bool = False):
    """Apply ``relu(x) ** 2`` in-place.

    This is a prefill FFN micro-boundary: keep the two large FFN GEMMs on
    cuBLAS, but replace the default PyTorch ``relu`` plus square elementwise
    pair with one Triton pass over the intermediate activation.  The helper is
    intentionally in-place and opt-in from ``native_jit`` so the default HF path
    stays unchanged until end-to-end rows prove it is profitable.
    """

    if torch is None:
        raise RuntimeError("fused_relu_square_inplace requires torch")
    if not force_fallback and fused_relu_square_available() and getattr(x, "is_cuda", False):
        x_c = x.contiguous()
        total = int(x_c.numel())
        if total > 0:
            block = int(block_size)
            if block <= 0:
                block = 1024
            _relu_square_inplace_kernel[(triton.cdiv(total, block),)](
                x_c,
                total,
                BLOCK=block,
                num_warps=4,
            )
        return x_c.reshape_as(x)
    x.relu_()
    x.mul_(x)
    return x


def fused_ffn_norm_shift_prefill(
    hidden_states: Any,
    cached_prev_h: Any,
    mix_x: Any,
    norm_weight: Any,
    norm_bias: Any,
    *,
    eps: float = 1e-5,
    block_h: int | None = None,
    mode: str = "recompute",
    force_fallback: bool = False,
):
    """Compute prefill FFN layer norm plus previous-token shift/mix.

    This is the larger FFN boundary used by native prefill before the FFN key
    GEMM:

    ``h = layer_norm(x)``
    ``prev = cat([cached_prev_h, h[:, :-1]])``
    ``fk = h + (prev - h) * mix_x``

    The default Triton path avoids materializing the full ``prev`` tensor and
    stores only ``fk`` plus the final ``h`` cache.  To keep programs independent
    it recomputes the previous token's layer norm for ``token > 0``.

    ``mode="two_pass"`` is a bounded FFN-memory-boundary probe: first write
    the normalized ``h`` sequence once, then build ``fk`` from adjacent rows in
    a second Triton pass.  It avoids the recompute work and the PyTorch
    ``cat``/pointwise temporaries while preserving the cuBLAS FFN GEMMs.
    Therefore both Triton modes stay opt-in until exact-card benchmark rows
    prove they are profitable.
    """

    if torch is None or F is None:
        raise RuntimeError("fused_ffn_norm_shift_prefill requires torch")
    if hidden_states.dim() != 3:
        raise ValueError(f"hidden_states must be [batch, tokens, hidden], got {tuple(hidden_states.shape)}")
    batch, tokens, hidden = (int(v) for v in hidden_states.shape)
    if cached_prev_h.dim() != 2 or int(cached_prev_h.shape[0]) != batch or int(cached_prev_h.shape[1]) != hidden:
        raise ValueError(f"cached_prev_h must be [{batch}, {hidden}], got {tuple(cached_prev_h.shape)}")
    mix = mix_x.reshape(-1)
    if int(mix.shape[0]) != hidden:
        raise ValueError(f"mix_x must contain {hidden} elements, got {tuple(mix_x.shape)}")
    norm_w = norm_weight.reshape(-1)
    norm_b = norm_bias.reshape(-1)
    if int(norm_w.shape[0]) != hidden or int(norm_b.shape[0]) != hidden:
        raise ValueError("norm_weight and norm_bias must match hidden size")

    use_triton = (
        not force_fallback
        and fused_ffn_norm_shift_prefill_available()
        and hidden_states.is_cuda
        and cached_prev_h.is_cuda
        and mix.is_cuda
        and norm_w.is_cuda
        and norm_b.is_cuda
        and hidden_states.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and cached_prev_h.dtype == hidden_states.dtype
        and mix.dtype == hidden_states.dtype
        and norm_w.dtype == hidden_states.dtype
        and norm_b.dtype == hidden_states.dtype
        and hidden <= 4096
    )
    mode_name = str(mode).strip().lower().replace("-", "_")
    if mode_name in {"", "0", "default", "recompute", "recompute_prev", "single", "single_pass"}:
        mode_name = "recompute"
    elif mode_name in {"1", "two", "two_pass", "twopass", "materialize_h", "norm_then_shift"}:
        mode_name = "two_pass"
    else:
        raise ValueError("fused_ffn_norm_shift_prefill mode must be 'recompute' or 'two_pass'")
    if not use_triton:
        h = F.layer_norm(hidden_states, (hidden,), norm_w, norm_b, float(eps))
        prev_h = torch.cat([cached_prev_h.view(batch, 1, hidden), h[:, :-1, :]], dim=1)
        fk = torch.addcmul(h, prev_h - h, mix.view(1, 1, hidden))
        return fk, h[:, -1, :].contiguous()

    x_c = hidden_states.contiguous()
    cached_c = cached_prev_h.contiguous()
    mix_c = mix.contiguous()
    norm_w_c = norm_w.contiguous()
    norm_b_c = norm_b.contiguous()
    fk = torch.empty_like(x_c)
    h_last = torch.empty((batch, hidden), device=x_c.device, dtype=x_c.dtype)
    block = int(block_h) if block_h is not None else int(triton.next_power_of_2(hidden))
    if block < hidden:
        block = int(triton.next_power_of_2(hidden))
    warps = 8 if block >= 1024 else 4
    if mode_name == "two_pass":
        h_full = torch.empty_like(x_c)
        _ffn_layernorm_prefill_kernel[(batch * tokens,)](
            x_c,
            norm_w_c,
            norm_b_c,
            h_full,
            h_last,
            tokens,
            hidden,
            float(eps),
            BLOCK_H=block,
            num_warps=warps,
        )
        _ffn_shift_from_norm_prefill_kernel[(batch * tokens,)](
            h_full,
            cached_c,
            mix_c,
            fk,
            tokens,
            hidden,
            BLOCK_H=block,
            num_warps=warps,
        )
        return fk, h_last
    _ffn_norm_shift_prefill_kernel[(batch * tokens,)](
        x_c,
        cached_c,
        norm_w_c,
        norm_b_c,
        mix_c,
        fk,
        h_last,
        tokens,
        hidden,
        float(eps),
        BLOCK_H=block,
        num_warps=warps,
    )
    return fk, h_last


def _flatten(x: Any, hidden: int | None = None, *, name: str):
    if torch is None:
        raise RuntimeError("fused_ffn requires torch")
    if x.dim() == 3:
        if int(x.shape[1]) != 1:
            raise ValueError(f"{name} must be [batch, 1, hidden] or [batch, hidden], got {tuple(x.shape)}")
        if hidden is not None and int(x.shape[2]) != hidden:
            raise ValueError(f"{name} hidden mismatch: got {int(x.shape[2])}, expected {hidden}")
        return x.reshape(int(x.shape[0]), int(x.shape[2])), True
    if x.dim() == 2:
        if hidden is not None and int(x.shape[1]) != hidden:
            raise ValueError(f"{name} hidden mismatch: got {int(x.shape[1])}, expected {hidden}")
        return x, False
    raise ValueError(f"{name} must be [batch, 1, hidden] or [batch, hidden]")


def fused_ffn(
    hidden_states: Any,
    prev_states: Any,
    mix_x: Any,
    key_weight: Any,
    value_weight: Any,
    *,
    block_m: int = 64,
    block_k: int = 64,
    force_fallback: bool = False,
):
    """Compute RWKV-7 FFN one-token output and next FFN state.

    Args mirror the FFN decode expression:

    ``k = hidden + (prev - hidden) * mix_x``
    ``out = value(relu(key(k)) ** 2)``

    Returns ``(out, next_state)``.  ``next_state`` is the original hidden input,
    preserving the same rank/layout as ``hidden_states``.
    """

    if torch is None or F is None:
        raise RuntimeError("fused_ffn requires torch")
    x2, had_seq = _flatten(hidden_states, name="hidden_states")
    hidden = int(x2.shape[1])
    prev2, prev_had_seq = _flatten(prev_states, hidden, name="prev_states")
    if prev_had_seq != had_seq or tuple(prev2.shape) != tuple(x2.shape):
        raise ValueError("hidden_states and prev_states must have identical flattened shape/layout")
    if mix_x.dim() not in (1, 2, 3):
        raise ValueError("mix_x must be broadcastable to hidden")
    mix = mix_x.reshape(-1)
    if int(mix.shape[0]) != hidden:
        raise ValueError(f"mix_x must have {hidden} elements, got {int(mix.shape[0])}")
    if key_weight.dim() != 2 or int(key_weight.shape[1]) != hidden:
        raise ValueError(f"key_weight must be [intermediate, {hidden}], got {tuple(key_weight.shape)}")
    intermediate = int(key_weight.shape[0])
    if value_weight.dim() != 2 or int(value_weight.shape[0]) != hidden or int(value_weight.shape[1]) != intermediate:
        raise ValueError(f"value_weight must be [{hidden}, {intermediate}], got {tuple(value_weight.shape)}")

    use_triton = (
        not force_fallback
        and fused_ffn_available()
        and x2.is_cuda
        and prev2.is_cuda
        and mix.is_cuda
        and key_weight.is_cuda
        and value_weight.is_cuda
        and x2.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and prev2.dtype == x2.dtype
        and mix.dtype == x2.dtype
        and key_weight.dtype == x2.dtype
        and value_weight.dtype == x2.dtype
    )
    if not use_triton:
        shifted = x2 + (prev2 - x2) * mix.view(1, -1)
        mid = torch.relu(F.linear(shifted, key_weight)) ** 2
        out = F.linear(mid, value_weight)
    else:
        batch = int(x2.shape[0])
        x_c = x2.contiguous()
        prev_c = prev2.contiguous()
        mix_c = mix.contiguous()
        key_c = key_weight.contiguous()
        value_c = value_weight.contiguous()
        mid = torch.empty((batch, intermediate), device=x2.device, dtype=x2.dtype)
        out = torch.empty((batch, hidden), device=x2.device, dtype=x2.dtype)
        _ffn_key_relu_kernel[(batch, triton.cdiv(intermediate, int(block_m)))](
            x_c,
            prev_c,
            mix_c,
            key_c,
            mid,
            hidden,
            intermediate,
            BLOCK_M=int(block_m),
            BLOCK_K=int(block_k),
            num_warps=4,
        )
        _ffn_value_kernel[(batch, triton.cdiv(hidden, int(block_m)))](
            mid,
            value_c,
            out,
            hidden,
            intermediate,
            BLOCK_M=int(block_m),
            BLOCK_K=int(block_k),
            num_warps=4,
        )
    if had_seq:
        return out.unsqueeze(1), hidden_states[:, -1:].contiguous() if hidden_states.dim() == 3 else hidden_states.unsqueeze(1)
    return out, hidden_states.contiguous()
