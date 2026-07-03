# coding=utf-8
"""Correctness-first RWKV-7 attention norm + time-mix6 helper.

This module is an isolated prototype for the native fused backend ladder.  It
mirrors the norm/shift/mix boundary used by :mod:`rwkv7_hf.native_jit.prefill`
without wiring itself into the production path or requiring CUDA/Triton.

Native prefill computes the attention input as::

    residual = layer_norm(x, pre_w, pre_b) if has_pre_norm else x
    h = layer_norm(residual, an_w, an_b)
    prev_h = cat([cached_previous_h, h[:, :-1, :]], dim=1)
    xr = h + (prev_h - h) * x_r
    ... xw/xk/xv/xa/xg ...

``fused_attn_norm_shift_mix`` expects ``prev_x`` to already be aligned with
``h`` in that same way (that is, the previous attention-normalized stream, not
raw previous residual activations).  The first version intentionally stays pure
PyTorch so it is CPU-testable and safe to use as a reference for future optional
kernels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

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
    def _attn_norm_shift_mix_prefill_kernel(
        x_ptr,
        cached_prev_h_ptr,
        pre_weight_ptr,
        pre_bias_ptr,
        norm_weight_ptr,
        norm_bias_ptr,
        xr_mix_ptr,
        xw_mix_ptr,
        xk_mix_ptr,
        xv_mix_ptr,
        xa_mix_ptr,
        xg_mix_ptr,
        residual_ptr,
        h_ptr,
        out_r_ptr,
        out_w_ptr,
        out_k_ptr,
        out_v_ptr,
        out_a_ptr,
        out_g_ptr,
        tokens: tl.constexpr,
        hidden: tl.constexpr,
        eps: tl.constexpr,
        HAS_PRE_NORM: tl.constexpr,
        STORE_RESIDUAL: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        row = tl.program_id(0)
        batch = row // tokens
        token = row - batch * tokens
        offs = tl.arange(0, BLOCK_H)
        mask = offs < hidden
        base = row * hidden + offs

        x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)
        if HAS_PRE_NORM:
            pre_mean = tl.sum(tl.where(mask, x, 0.0), axis=0) / hidden
            pre_centered = tl.where(mask, x - pre_mean, 0.0)
            pre_var = tl.sum(pre_centered * pre_centered, axis=0) / hidden
            pre_w = tl.load(pre_weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
            pre_b = tl.load(pre_bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            residual = pre_centered * tl.rsqrt(pre_var + eps) * pre_w + pre_b
            if STORE_RESIDUAL:
                tl.store(residual_ptr + base, residual, mask=mask)
        else:
            residual = x

        norm_mean = tl.sum(tl.where(mask, residual, 0.0), axis=0) / hidden
        norm_centered = tl.where(mask, residual - norm_mean, 0.0)
        norm_var = tl.sum(norm_centered * norm_centered, axis=0) / hidden
        norm_w = tl.load(norm_weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
        norm_b = tl.load(norm_bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        h = norm_centered * tl.rsqrt(norm_var + eps) * norm_w + norm_b
        tl.store(h_ptr + base, h, mask=mask)

        prev_base = ((batch * tokens + token - 1) * hidden + offs)
        prev_x = tl.load(x_ptr + prev_base, mask=mask & (token > 0), other=0.0).to(tl.float32)
        if HAS_PRE_NORM:
            prev_pre_mean = tl.sum(tl.where(mask, prev_x, 0.0), axis=0) / hidden
            prev_pre_centered = tl.where(mask, prev_x - prev_pre_mean, 0.0)
            prev_pre_var = tl.sum(prev_pre_centered * prev_pre_centered, axis=0) / hidden
            pre_w2 = tl.load(pre_weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
            pre_b2 = tl.load(pre_bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            prev_residual = prev_pre_centered * tl.rsqrt(prev_pre_var + eps) * pre_w2 + pre_b2
        else:
            prev_residual = prev_x
        prev_norm_mean = tl.sum(tl.where(mask, prev_residual, 0.0), axis=0) / hidden
        prev_norm_centered = tl.where(mask, prev_residual - prev_norm_mean, 0.0)
        prev_norm_var = tl.sum(prev_norm_centered * prev_norm_centered, axis=0) / hidden
        prev_h_calc = prev_norm_centered * tl.rsqrt(prev_norm_var + eps) * norm_w + norm_b
        cached_prev = tl.load(cached_prev_h_ptr + batch * hidden + offs, mask=mask, other=0.0).to(tl.float32)
        prev_h = tl.where(token > 0, prev_h_calc, cached_prev)

        delta = prev_h - h
        mix_r = tl.load(xr_mix_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        mix_w = tl.load(xw_mix_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        mix_k = tl.load(xk_mix_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        mix_v = tl.load(xv_mix_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        mix_a = tl.load(xa_mix_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        mix_g = tl.load(xg_mix_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_r_ptr + base, h + delta * mix_r, mask=mask)
        tl.store(out_w_ptr + base, h + delta * mix_w, mask=mask)
        tl.store(out_k_ptr + base, h + delta * mix_k, mask=mask)
        tl.store(out_v_ptr + base, h + delta * mix_v, mask=mask)
        tl.store(out_a_ptr + base, h + delta * mix_a, mask=mask)
        tl.store(out_g_ptr + base, h + delta * mix_g, mask=mask)


@dataclass(frozen=True)
class FusedNormMixOutput:
    """Return bundle for the norm + six-way time-mix prototype.

    ``backend`` is telemetry-only.  It is currently always ``"torch"`` because
    this file does not enable any CUDA/Triton fast path by default.
    """

    residual: Any
    h: Any
    xr: Any
    xw: Any
    xk: Any
    xv: Any
    xa: Any
    xg: Any
    backend: str = "torch"

    def mix_tuple(self) -> tuple[Any, Any, Any, Any, Any, Any]:
        """Return ``(xr, xw, xk, xv, xa, xg)``."""

        return (self.xr, self.xw, self.xk, self.xv, self.xa, self.xg)

    def as_tuple(self) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
        """Return ``(residual, h, xr, xw, xk, xv, xa, xg)``."""

        return (self.residual, self.h, *self.mix_tuple())

    def __iter__(self) -> Iterator[Any]:
        return iter(self.as_tuple())


def fused_attn_norm_shift_mix_available() -> bool:
    """Return whether the reference implementation can run in this process."""

    return torch is not None and F is not None


def fused_attn_norm_shift_mix_prefill_available() -> bool:
    """Return whether the optional prefill Triton norm+shift-mix path can run."""

    return bool(_HAS_TRITON and torch is not None and F is not None)


def _require_torch() -> None:
    if torch is None or F is None:
        raise RuntimeError("fused_attn_norm_shift_mix requires torch")


def _check_input_pair(x: Any, prev_x: Any) -> tuple[int, tuple[int, ...]]:
    _require_torch()
    if x.dim() not in (2, 3):
        raise ValueError("x must be shaped [batch, hidden] or [batch, tokens, hidden]")
    if tuple(x.shape) != tuple(prev_x.shape):
        raise ValueError(f"x and prev_x must have identical shapes; got {tuple(x.shape)} and {tuple(prev_x.shape)}")
    hidden = int(x.shape[-1])
    if hidden <= 0:
        raise ValueError("hidden dimension must be non-zero")
    return hidden, tuple(x.shape)


def _flatten_hidden_param(param: Any, hidden: int, *, name: str) -> Any:
    _require_torch()
    if param is None:
        return None
    if int(param.numel()) != hidden:
        raise ValueError(f"{name} must contain hidden={hidden} values; got shape {tuple(param.shape)}")
    return param.reshape(hidden)


def _flatten_mix(mix: Any, hidden: int, *, name: str) -> Any:
    _require_torch()
    if mix is None:
        raise ValueError(f"{name} is required")
    if int(mix.numel()) != hidden:
        raise ValueError(f"{name} must contain hidden={hidden} values; got shape {tuple(mix.shape)}")
    return mix.reshape(*((1,) * 2), hidden)


def _maybe_layer_norm(x: Any, weight: Any, bias: Any, *, apply: bool, eps: float) -> Any:
    _require_torch()
    if not apply:
        return x
    hidden = int(x.shape[-1])
    norm_weight = _flatten_hidden_param(weight, hidden, name="layer_norm_weight")
    norm_bias = _flatten_hidden_param(bias, hidden, name="layer_norm_bias")
    return F.layer_norm(x, (hidden,), norm_weight, norm_bias, eps)


def fused_attn_norm_shift_mix(
    x: Any,
    prev_x: Any,
    x_r: Any,
    x_w: Any,
    x_k: Any,
    x_v: Any,
    x_a: Any,
    x_g: Any,
    *,
    norm_weight: Any | None = None,
    norm_bias: Any | None = None,
    pre_norm_weight: Any | None = None,
    pre_norm_bias: Any | None = None,
    has_pre_norm: bool = False,
    has_attn_norm: bool | None = None,
    eps: float = 1e-5,
    force_fallback: bool = False,
) -> FusedNormMixOutput:
    """Compute RWKV-7 attention residual, normed ``h``, and six time-mixes.

    Args:
        x: Current layer input, shaped ``[B, T, hidden]`` for prefill or
            ``[B, hidden]`` for small decode-style probes.
        prev_x: Previous attention-normalized values aligned to ``x``.  For
            native-prefill equivalence this should be
            ``cat([xpa[:, None, :], h[:, :-1, :]], dim=1)``.
        x_r/x_w/x_k/x_v/x_a/x_g: RWKV-7 mix vectors.  Any shape with exactly
            ``hidden`` elements is accepted.
        norm_weight/norm_bias: Attention layer-norm parameters.  If
            ``has_attn_norm`` is ``None``, the norm is applied when either of
            these is provided and skipped when both are absent.
        pre_norm_weight/pre_norm_bias: Optional pre-norm parameters used only
            when ``has_pre_norm`` is true, matching ``native_jit.prefill``.
        has_pre_norm: Whether to apply the pre-attention layer norm to ``x``.
        has_attn_norm: Override for applying attention layer norm.  Set true to
            request unweighted layer norm even when ``norm_weight`` and
            ``norm_bias`` are ``None``.
        eps: LayerNorm epsilon.  ``native_jit.prefill`` uses ``1e-5``.
        force_fallback: Reserved for API symmetry with optional fused helpers;
            ignored because this prototype is intentionally pure torch.

    Returns:
        :class:`FusedNormMixOutput` with ``residual``, ``h``, and
        ``xr/xw/xk/xv/xa/xg``.  The ``backend`` field is ``"torch"``.
    """

    del force_fallback  # Explicitly no optional kernel in the first prototype.
    hidden, shape = _check_input_pair(x, prev_x)
    if len(shape) == 2:
        mix_shape = (1, hidden)
    else:
        mix_shape = (1, 1, hidden)

    residual = _maybe_layer_norm(
        x,
        pre_norm_weight,
        pre_norm_bias,
        apply=bool(has_pre_norm),
        eps=float(eps),
    )
    apply_attn_norm = (norm_weight is not None or norm_bias is not None) if has_attn_norm is None else bool(has_attn_norm)
    h = _maybe_layer_norm(
        residual,
        norm_weight,
        norm_bias,
        apply=apply_attn_norm,
        eps=float(eps),
    )

    mixes = tuple(
        _flatten_mix(m, hidden, name=n).reshape(mix_shape)
        for m, n in (
            (x_r, "x_r"),
            (x_w, "x_w"),
            (x_k, "x_k"),
            (x_v, "x_v"),
            (x_a, "x_a"),
            (x_g, "x_g"),
        )
    )
    delta = prev_x - h
    xr, xw, xk, xv, xa, xg = (torch.addcmul(h, delta, mix) for mix in mixes)
    return FusedNormMixOutput(residual=residual, h=h, xr=xr, xw=xw, xk=xk, xv=xv, xa=xa, xg=xg)


def fused_attn_norm_shift_mix_prefill(
    x: Any,
    cached_prev_h: Any,
    x_r: Any,
    x_w: Any,
    x_k: Any,
    x_v: Any,
    x_a: Any,
    x_g: Any,
    *,
    norm_weight: Any,
    norm_bias: Any,
    pre_norm_weight: Any | None = None,
    pre_norm_bias: Any | None = None,
    has_pre_norm: bool = False,
    eps: float = 1e-5,
    block_h: int | None = None,
    force_fallback: bool = False,
) -> FusedNormMixOutput:
    """Compute native-prefill attention norm plus six time-mixes.

    This optional CUDA/Triton boundary keeps the large R/K/V dense projections
    on cuBLAS, but fuses the pre-attention layer norm, attention layer norm,
    previous-token alignment, and six elementwise time-mixes into one kernel.
    To avoid a separate ``prev_h = cat(...)`` materialization, the Triton path
    recomputes the previous token's normalized attention input inside the same
    program and uses ``cached_prev_h`` for token 0.  The default path remains a
    pure PyTorch reference.
    """

    if torch is None or F is None:
        raise RuntimeError("fused_attn_norm_shift_mix_prefill requires torch")
    if x.dim() != 3:
        raise ValueError(f"x must be [batch, tokens, hidden], got {tuple(x.shape)}")
    batch, tokens, hidden = (int(v) for v in x.shape)
    if cached_prev_h.dim() != 2 or int(cached_prev_h.shape[0]) != batch or int(cached_prev_h.shape[1]) != hidden:
        raise ValueError(f"cached_prev_h must be [{batch}, {hidden}], got {tuple(cached_prev_h.shape)}")
    if norm_weight is None or norm_bias is None:
        raise ValueError("norm_weight and norm_bias are required for prefill norm+shift-mix")
    mixes = tuple(
        _flatten_mix(m, hidden, name=n).reshape(hidden)
        for m, n in (
            (x_r, "x_r"),
            (x_w, "x_w"),
            (x_k, "x_k"),
            (x_v, "x_v"),
            (x_a, "x_a"),
            (x_g, "x_g"),
        )
    )
    norm_w = _flatten_hidden_param(norm_weight, hidden, name="norm_weight")
    norm_b = _flatten_hidden_param(norm_bias, hidden, name="norm_bias")
    pre_w = _flatten_hidden_param(pre_norm_weight, hidden, name="pre_norm_weight") if has_pre_norm else norm_w
    pre_b = _flatten_hidden_param(pre_norm_bias, hidden, name="pre_norm_bias") if has_pre_norm else norm_b

    use_triton = (
        not force_fallback
        and fused_attn_norm_shift_mix_prefill_available()
        and x.is_cuda
        and cached_prev_h.is_cuda
        and norm_w.is_cuda
        and norm_b.is_cuda
        and pre_w is not None
        and pre_b is not None
        and pre_w.is_cuda
        and pre_b.is_cuda
        and all(m.is_cuda for m in mixes)
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and cached_prev_h.dtype == x.dtype
        and norm_w.dtype == x.dtype
        and norm_b.dtype == x.dtype
        and pre_w.dtype == x.dtype
        and pre_b.dtype == x.dtype
        and all(m.dtype == x.dtype for m in mixes)
        and hidden <= 4096
    )
    if not use_triton:
        residual = F.layer_norm(x, (hidden,), pre_w, pre_b, float(eps)) if has_pre_norm else x
        h = F.layer_norm(residual, (hidden,), norm_w, norm_b, float(eps))
        prev_h = torch.cat([cached_prev_h.view(batch, 1, hidden), h[:, :-1, :]], dim=1)
        delta = prev_h - h
        xr, xw, xk, xv, xa, xg = (torch.addcmul(h, delta, mix.view(1, 1, hidden)) for mix in mixes)
        return FusedNormMixOutput(residual=residual, h=h, xr=xr, xw=xw, xk=xk, xv=xv, xa=xa, xg=xg)

    x_c = x.contiguous()
    cached_c = cached_prev_h.contiguous()
    pre_w_c = pre_w.contiguous()
    pre_b_c = pre_b.contiguous()
    norm_w_c = norm_w.contiguous()
    norm_b_c = norm_b.contiguous()
    mixes_c = tuple(m.contiguous() for m in mixes)
    residual = torch.empty_like(x_c) if has_pre_norm else x
    h = torch.empty_like(x_c)
    outs = tuple(torch.empty_like(x_c) for _ in range(6))
    block = int(block_h) if block_h is not None else int(triton.next_power_of_2(hidden))
    if block < hidden:
        block = int(triton.next_power_of_2(hidden))
    grid = (batch * tokens,)
    _attn_norm_shift_mix_prefill_kernel[grid](
        x_c,
        cached_c,
        pre_w_c,
        pre_b_c,
        norm_w_c,
        norm_b_c,
        *mixes_c,
        residual if has_pre_norm else x_c,
        h,
        *outs,
        tokens,
        hidden,
        float(eps),
        HAS_PRE_NORM=bool(has_pre_norm),
        STORE_RESIDUAL=bool(has_pre_norm),
        BLOCK_H=block,
        num_warps=8 if block >= 1024 else 4,
    )
    return FusedNormMixOutput(residual=residual, h=h, xr=outs[0], xw=outs[1], xk=outs[2], xv=outs[3], xa=outs[4], xg=outs[5], backend="triton")


# Shorter aliases for experiments/benchmarks without implying production use.
norm_mix6 = fused_attn_norm_shift_mix
attn_norm_shift_mix = fused_attn_norm_shift_mix


__all__ = [
    "FusedNormMixOutput",
    "attn_norm_shift_mix",
    "fused_attn_norm_shift_mix",
    "fused_attn_norm_shift_mix_available",
    "fused_attn_norm_shift_mix_prefill",
    "fused_attn_norm_shift_mix_prefill_available",
    "norm_mix6",
]
