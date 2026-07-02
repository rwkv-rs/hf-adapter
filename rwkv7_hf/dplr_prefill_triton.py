#!/usr/bin/env python3
# coding=utf-8
"""Opt-in compiled DPLR/WY prefill scan prototype.

This module is the first compiled-backend hook for the DPLR/WY prefill line.
It intentionally keeps the public boundary identical to
``dplr_prefill.dplr_chunk_scan`` so the synthetic benchmark can switch between
pure torch and compiled implementations without touching HF model code.

P0 implementation note
----------------------
The current kernel target is a correctness/performance bridge, not the final
three-stage WY factor scan.  It delegates the per-token rank-1 DPLR recurrence
to the existing Triton recurrent scan kernel from ``fused_recurrent_update``.
That kernel is still mathematically the RWKV-7 DPLR update

    S_t = S_{t-1} (diag(w_t) + (-kk_t)(kk_t*a_t)^T) + v_t k_t^T

and uses fp32 state accumulation, but it does not yet materialize chunk-level
WY summaries or prefix-combine metadata.  Keeping this in a separate module lets
benchmarks label it as ``triton_wy``/``cuda_wy`` while the next iteration
replaces the delegate with explicit chunk-summary, prefix-combine, and
chunk-apply kernels.
"""
from __future__ import annotations

import os
from typing import Any

try:  # pragma: no cover - optional on lightweight hosts
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - CUDA/Triton hosts only
    from .fused_recurrent_update import (
        fused_recurrent_scan,
        fused_recurrent_scan_available,
        torch_recurrent_scan,
    )
except Exception:  # pragma: no cover - direct script fallback
    try:
        from fused_recurrent_update import (  # type: ignore[no-redef]
            fused_recurrent_scan,
            fused_recurrent_scan_available,
            torch_recurrent_scan,
        )
    except Exception:  # pragma: no cover
        fused_recurrent_scan = None  # type: ignore[assignment]
        fused_recurrent_scan_available = None  # type: ignore[assignment]
        torch_recurrent_scan = None  # type: ignore[assignment]


__all__ = [
    "dplr_chunk_scan_triton",
    "dplr_chunk_scan_triton_available",
]


def dplr_chunk_scan_triton_available() -> bool:
    """Return whether the opt-in compiled scan can run on this host."""

    return bool(
        torch is not None
        and fused_recurrent_scan is not None
        and fused_recurrent_scan_available is not None
        and fused_recurrent_scan_available()
    )


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _as_bthn(x: Any, H: int, N: int, *, name: str):
    if torch is None:
        raise RuntimeError("dplr_chunk_scan_triton requires torch")
    if not hasattr(x, "dim"):
        raise TypeError(f"{name} must be a torch.Tensor")
    if x.dim() == 4:
        if int(x.shape[2]) != H or int(x.shape[3]) != N:
            raise ValueError(
                f"{name} must be shaped [batch,tokens,{H},{N}] or "
                f"[batch,tokens,{H * N}]; got {tuple(x.shape)}"
            )
        return x.contiguous(), False
    if x.dim() == 3:
        if int(x.shape[2]) != H * N:
            raise ValueError(
                f"{name} must be shaped [batch,tokens,{H},{N}] or "
                f"[batch,tokens,{H * N}]; got {tuple(x.shape)}"
            )
        return x.reshape(int(x.shape[0]), int(x.shape[1]), H, N).contiguous(), True
    raise ValueError(f"{name} must be shaped [batch,tokens,{H},{N}] or [batch,tokens,{H * N}]")


def _validate_chunk_size(chunk_size: int) -> int:
    try:
        value = int(chunk_size)
    except Exception as exc:  # pragma: no cover - defensive
        raise TypeError("chunk_size must be an integer") from exc
    if value <= 0:
        raise ValueError("chunk_size must be a positive integer")
    return value


def _fallback_scan(r4: Any, w4: Any, k4: Any, v4: Any, kk4: Any, a4: Any, state: Any, *, flat: bool):
    if torch_recurrent_scan is None:
        raise RuntimeError("torch_recurrent_scan fallback is unavailable")
    B, T, H, N = (int(vv) for vv in r4.shape)
    out, final_state = torch_recurrent_scan(
        r4.reshape(B, T, H * N) if flat else r4,
        w4,
        k4,
        v4,
        kk4,
        a4,
        state,
    )
    return out, final_state


def dplr_chunk_scan_triton(
    r: Any,
    w: Any,
    k: Any,
    v: Any,
    kk: Any,
    a: Any,
    state: Any,
    *,
    chunk_size: int = 64,
    block_m: int | None = None,
    num_warps: int | None = None,
    force_fallback: bool = False,
):
    """Run the opt-in compiled DPLR scan prototype.

    Inputs mirror :func:`rwkv7_hf.dplr_prefill.dplr_chunk_scan`: vectors may be
    ``[B,T,H,N]`` or flattened ``[B,T,H*N]`` and state must be native VxK
    ``[B,H,N,N]``.  The current P0 backend uses the existing Triton recurrent
    scan kernel with split-row execution by default.  ``chunk_size`` is accepted
    to keep the future WY chunk API stable; P0 does not yet use it internally.

    Environment knobs for synthetic benchmarking:

    - ``RWKV7_DPLR_TRITON_BLOCK_M``: row block for split-row scan, default 8.
    - ``RWKV7_DPLR_TRITON_NUM_WARPS``: optional Triton num_warps override.
    """

    if torch is None:
        raise RuntimeError("dplr_chunk_scan_triton requires torch")
    if state.dim() != 4:
        raise ValueError("state must be shaped [batch, heads, head_dim, head_dim]")
    B, H, N, N2 = (int(vv) for vv in state.shape)
    if N != N2:
        raise ValueError("state must be square in the last two dimensions")
    _validate_chunk_size(chunk_size)

    r4, flat = _as_bthn(r, H, N, name="r")
    w4, _ = _as_bthn(w, H, N, name="w")
    k4, _ = _as_bthn(k, H, N, name="k")
    v4, _ = _as_bthn(v, H, N, name="v")
    kk4, _ = _as_bthn(kk, H, N, name="kk")
    a4, _ = _as_bthn(a, H, N, name="a")
    if int(r4.shape[0]) != B:
        raise ValueError("r/w/k/v/kk/a batch size must match state")
    T = int(r4.shape[1])
    for name, x in (("w", w4), ("k", k4), ("v", v4), ("kk", kk4), ("a", a4)):
        if int(x.shape[0]) != B or int(x.shape[1]) != T:
            raise ValueError(f"{name} batch/time shape must match r; got {tuple(x.shape)}")
        if x.device != r4.device:
            raise ValueError(f"{name} must be on the same device as r")
    if state.device != r4.device:
        raise ValueError("state must be on the same device as r")

    if block_m is None:
        block_m = _env_int("RWKV7_DPLR_TRITON_BLOCK_M", 8)
    block_m = int(block_m)
    if block_m <= 0:
        raise ValueError("block_m must be positive")
    if num_warps is None and os.environ.get("RWKV7_DPLR_TRITON_NUM_WARPS"):
        num_warps = _env_int("RWKV7_DPLR_TRITON_NUM_WARPS", 4)

    # HF native prefill can hand us fp32 auxiliaries (especially after state
    # prep) while the recurrent vectors are fp16.  The underlying Triton kernel
    # expects k/v/kk/a to share r's dtype, with fp32 accumulation for state, so
    # cast only those per-token inputs at the compiled-boundary.  w may remain
    # fp32 because the existing fused scan accepts fp32 decay vectors.
    k_kernel = k4 if k4.dtype == r4.dtype else k4.to(dtype=r4.dtype)
    v_kernel = v4 if v4.dtype == r4.dtype else v4.to(dtype=r4.dtype)
    kk_kernel = kk4 if kk4.dtype == r4.dtype else kk4.to(dtype=r4.dtype)
    a_kernel = a4 if a4.dtype == r4.dtype else a4.to(dtype=r4.dtype)

    fallback_reasons = []
    if force_fallback:
        fallback_reasons.append("force_fallback")
    if not dplr_chunk_scan_triton_available() or fused_recurrent_scan is None:
        fallback_reasons.append("triton_unavailable")
    if not (r4.is_cuda and w4.is_cuda and k_kernel.is_cuda and v_kernel.is_cuda and kk_kernel.is_cuda and a_kernel.is_cuda and state.is_cuda):
        fallback_reasons.append("non_cuda_tensor")
    if state.dtype != torch.float32:
        fallback_reasons.append(f"state_dtype={state.dtype}")
    if r4.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        fallback_reasons.append(f"r_dtype={r4.dtype}")
    if w4.dtype not in (r4.dtype, torch.float32):
        fallback_reasons.append(f"w_dtype={w4.dtype}")

    use_triton = not fallback_reasons
    if not use_triton and os.environ.get("RWKV7_DPLR_TRITON_STRICT", "0").lower() not in {"0", "false", "no", "off"}:
        raise RuntimeError("dplr_chunk_scan_triton strict mode fallback: " + ",".join(fallback_reasons))
    if not use_triton:
        return _fallback_scan(r4, w4, k4, v4, kk4, a4, state, flat=flat)

    out, final_state = fused_recurrent_scan(
        r4,
        w4,
        k_kernel,
        v_kernel,
        kk_kernel,
        a_kernel,
        state,
        block_n=N,
        block_m=block_m,
        num_warps=num_warps,
        force_fallback=False,
    )
    if flat and out.dim() == 4:
        return out.reshape(B, T, H * N), final_state
    return out, final_state
