#!/usr/bin/env python3
# coding=utf-8
"""Pure torch RWKV-7 DPLR/chunked prefill reference scan.

This module intentionally does not call ``native_jit`` and does not use Triton.
It is a small, correctness-first prototype for the RWKV-7 recurrent prefill
recurrence in the native VxK state orientation used by this repository.

The recurrence is affine over a chunk boundary::

    S_t = S_{t-1} A_t + B_t
    A_t = diag(w_t) + (-kk_t) (kk_t * a_t)^T
    B_t = v_t k_t^T

A future optimized implementation can compose the per-token ``A_t`` matrices
inside each chunk (or use a WY-style representation for the diagonal plus
rank-1 factors) and apply the resulting chunk transform to the incoming state.
For now, ``dplr_chunk_scan`` exposes the chunked interface while scanning
sequentially within each chunk so it remains easy to validate against the
existing recurrent reference.
"""
from __future__ import annotations

from typing import Any

try:  # pragma: no cover - exercised in lightweight environments without torch
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


__all__ = ["dplr_chunk_scan"]


def _require_torch():
    if torch is None:  # pragma: no cover - depends on local environment
        raise RuntimeError("dplr_chunk_scan requires torch")


def _as_bthn(x: Any, H: int, N: int, *, name: str):
    """Return ``x`` as contiguous [B,T,H,N] plus whether it was flat."""

    _require_torch()
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
        chunk_size_i = int(chunk_size)
    except Exception as exc:  # pragma: no cover - defensive
        raise TypeError("chunk_size must be an integer") from exc
    if chunk_size_i <= 0:
        raise ValueError("chunk_size must be a positive integer")
    return chunk_size_i


def _check_tensor_compat(name: str, x: Any, *, B: int, T: int, device: Any) -> None:
    if int(x.shape[0]) != B or int(x.shape[1]) != T:
        raise ValueError(f"{name} batch/time shape must match r; got {tuple(x.shape)}")
    if x.device != device:
        raise ValueError(f"{name} must be on the same device as r/state")


def _dplr_step(r_t: Any, w_t: Any, k_t: Any, v_t: Any, kk_t: Any, a_t: Any, state_t: Any):
    """One RWKV-7 DPLR recurrent update in fp32 compute.

    Shapes are all normalized to [B,H,N] except ``state_t`` [B,H,N,N].  State is
    native VxK: rows are value channels, columns are key channels.
    """

    B, H, N = (int(r_t.shape[0]), int(r_t.shape[1]), int(r_t.shape[2]))

    # B_t = v_t k_t^T stores values as rows and keys as columns.
    vk = v_t.view(B, H, N, 1) @ k_t.view(B, H, 1, N)

    # A_t = diag(w_t) + (-kk_t) (kk_t * a_t)^T.  ``state @ ab`` applies the
    # rank-1 DPLR correction on the K/column side, matching native VxK layout.
    ab = (-kk_t).view(B, H, N, 1) @ (kk_t * a_t).view(B, H, 1, N)
    new_state = state_t * w_t.view(B, H, 1, N) + state_t @ ab + vk
    out = new_state @ r_t.view(B, H, N, 1)
    return out.view(B, H, N), new_state


def dplr_chunk_scan(
    r: Any,
    w: Any,
    k: Any,
    v: Any,
    kk: Any,
    a: Any,
    state: Any,
    *,
    chunk_size: int = 64,
    force_fallback: bool = False,
):
    """Reference RWKV-7 DPLR/chunked prefill scan using only torch.

    Parameters
    ----------
    r, w, k, v, kk, a:
        Post-projection per-token tensors shaped either ``[B, T, H, N]`` or
        flattened ``[B, T, H*N]``.  The returned recurrent output follows the
        representation of ``r``.
    state:
        Initial recurrent state shaped ``[B, H, N, N]`` in the native VxK
        orientation used by ``rwkv7_hf.fused_recurrent_update``.
    chunk_size:
        Positive chunk length.  The current reference implementation scans
        sequentially inside every chunk; chunk boundaries are explicit so a
        later affine/WY composer can replace the inner loop without changing the
        caller contract.
    force_fallback:
        Reserved switch for future optimized paths.  Today both values use the
        same correctness-first torch fallback.

    Returns
    -------
    (out, final_state):
        ``out`` has the same shape style and dtype as ``r``. ``final_state`` has
        shape ``[B, H, N, N]`` and is cast back to the input state's dtype after
        fp32 accumulation for fp16/bf16 inputs.
    """

    _require_torch()
    chunk_size_i = _validate_chunk_size(chunk_size)
    _ = force_fallback  # Kept in the public signature for future fast paths.

    if not hasattr(state, "dim"):
        raise TypeError("state must be a torch.Tensor")
    if state.dim() != 4:
        raise ValueError("state must be shaped [batch, heads, head_dim, head_dim]")
    B, H, N, N2 = (int(vv) for vv in state.shape)
    if N != N2:
        raise ValueError("state must be square in the last two dimensions")
    if not getattr(state, "is_floating_point", lambda: False)():
        raise TypeError("state must be a floating point tensor")

    r4, flat = _as_bthn(r, H, N, name="r")
    w4, _ = _as_bthn(w, H, N, name="w")
    k4, _ = _as_bthn(k, H, N, name="k")
    v4, _ = _as_bthn(v, H, N, name="v")
    kk4, _ = _as_bthn(kk, H, N, name="kk")
    a4, _ = _as_bthn(a, H, N, name="a")

    if int(r4.shape[0]) != B:
        raise ValueError("r/w/k/v/kk/a batch size must match state")
    T = int(r4.shape[1])
    device = r4.device
    if state.device != device:
        raise ValueError("state must be on the same device as r")
    for name, x in (("w", w4), ("k", k4), ("v", v4), ("kk", kk4), ("a", a4)):
        _check_tensor_compat(name, x, B=B, T=T, device=device)

    tensors = (r4, w4, k4, v4, kk4, a4)
    if not all(x.is_floating_point() for x in tensors):
        raise TypeError("r/w/k/v/kk/a must be floating point tensors")

    out_dtype = r4.dtype
    state_dtype = state.dtype

    # Correctness/stability policy for the reference prototype: fp16/bf16 (and
    # fp32) scan in fp32, then cast public outputs back.  This mirrors the
    # native reference formula's explicit float accumulation for rank-1 terms.
    r32, w32, k32, v32, kk32, a32 = (x.to(dtype=torch.float32) for x in tensors)
    cur_state = state.to(dtype=torch.float32)

    if T == 0:
        empty = r4.new_empty((B, 0, H, N), dtype=out_dtype)
        return (empty.reshape(B, 0, H * N) if flat else empty), cur_state.to(dtype=state_dtype)

    outs = []
    for start in range(0, T, chunk_size_i):
        end = min(start + chunk_size_i, T)

        # Chunk boundary contract:
        #   incoming state: cur_state
        #   per-token affine terms: A_t and B_t from _dplr_step comments
        #   outgoing state after this loop: S_end = S_start Phi + Psi
        # Future fast path location: compose Phi/Psi (or WY factors) for
        # [start:end] here, while still materializing per-token outputs.
        for t in range(start, end):
            out_t, cur_state = _dplr_step(
                r32[:, t],
                w32[:, t],
                k32[:, t],
                v32[:, t],
                kk32[:, t],
                a32[:, t],
                cur_state,
            )
            outs.append(out_t)

    stacked = torch.stack(outs, dim=1).to(dtype=out_dtype)
    if flat:
        stacked = stacked.reshape(B, T, H * N)
    return stacked, cur_state.to(dtype=state_dtype)
