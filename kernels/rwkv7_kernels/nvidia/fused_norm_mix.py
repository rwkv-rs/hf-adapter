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


# Shorter aliases for experiments/benchmarks without implying production use.
norm_mix6 = fused_attn_norm_shift_mix
attn_norm_shift_mix = fused_attn_norm_shift_mix


__all__ = [
    "FusedNormMixOutput",
    "attn_norm_shift_mix",
    "fused_attn_norm_shift_mix",
    "fused_attn_norm_shift_mix_available",
    "norm_mix6",
]
