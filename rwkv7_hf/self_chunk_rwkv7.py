# coding=utf-8
"""FLA-independent, inference-only RWKV-7 DPLR chunk prefill.

The forward kernels in the adjacent ``self_chunk_*`` modules are vendored/adapted from
Flash Linear Attention under its MIT license.  This wrapper intentionally has
no ``fla`` import and exposes the native adapter state layout.
"""
from __future__ import annotations

import math

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - CUDA/Triton hosts
    from .self_chunk_A_fwd import chunk_dplr_fwd_intra
    from .self_chunk_h_fwd import chunk_dplr_fwd_h
    from .self_chunk_o_fwd import chunk_dplr_fwd_o
    from .self_chunk_cumsum import chunk_rwkv6_fwd_cumsum
    from .self_chunk_wy_fwd import prepare_wy_repr_fwd
    _AVAILABLE = True
except Exception:  # pragma: no cover
    _AVAILABLE = False


def self_chunk_rwkv7_available() -> bool:
    return bool(_AVAILABLE and torch is not None and torch.cuda.is_available())


def self_chunk_rwkv7(
    r,
    w_decay,
    k,
    v,
    kk,
    a_gate,
    state_native,
    *,
    chunk_size: int = 16,
    w_is_log: bool = False,
):
    """Return ``(recurrent_output, final_native_state)`` for equal lengths.

    Inputs are ``[B,T,H,N]`` and native state is ``[B,H,V,K]``.  The vendored
    DPLR kernels use ``[B,H,K,V]``, so only the small boundary state is
    transposed; sequence tensors stay in-place.
    """

    if not self_chunk_rwkv7_available():
        raise RuntimeError("self chunk RWKV-7 requires CUDA, Triton, and torch")
    if int(chunk_size) != 16:
        raise ValueError("self chunk RWKV-7 currently uses chunk_size=16")
    if r.dim() != 4 or any(tuple(x.shape) != tuple(r.shape) for x in (w_decay, k, v, kk, a_gate)):
        raise ValueError("self chunk inputs must share [B,T,H,N]")
    B, T, H, N = (int(x) for x in r.shape)
    if N != 64 or T % 16:
        raise ValueError("self chunk RWKV-7 currently requires head_dim=64 and T divisible by 16")

    # FLA's generalized DPLR convention for RWKV-7 is a=-kk, b=kk*a.
    dplr_a = -kk
    dplr_b = kk * a_gate
    log_decay = w_decay.float() if w_is_log else torch.log(w_decay.float())
    gi, ge = chunk_rwkv6_fwd_cumsum(log_decay, 16, scale=1.0 / math.log(2.0))
    A_ab, A_qk, A_ak, A_qb, qg, kg, ag, bg = chunk_dplr_fwd_intra(
        q=r,
        k=k,
        a=dplr_a,
        b=dplr_b,
        gi=gi,
        ge=ge,
        scale=1.0,
        chunk_size=16,
        safe_gate=True,
    )
    wy_w, wy_u, _ = prepare_wy_repr_fwd(
        ag=ag,
        v=v,
        A_ak=A_ak,
        A_ab=A_ab,
        cu_seqlens=None,
        chunk_size=16,
    )
    initial = state_native.transpose(-1, -2).contiguous()
    h, v_new, final = chunk_dplr_fwd_h(
        kg=kg,
        v=v,
        w=wy_w,
        u=wy_u,
        bg=bg,
        gk=gi,
        initial_state=initial,
        output_final_state=True,
        chunk_size=16,
    )
    out = chunk_dplr_fwd_o(
        qg=qg,
        v=v,
        v_new=v_new,
        A_qk=A_qk,
        A_qb=A_qb,
        h=h,
        chunk_size=16,
    )
    return out, final.transpose(-1, -2).contiguous()


__all__ = ["self_chunk_rwkv7", "self_chunk_rwkv7_available"]
