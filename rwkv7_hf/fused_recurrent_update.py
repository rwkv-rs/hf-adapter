# coding=utf-8
"""Optional fused recurrent state-update prototypes for RWKV-7 decode.

This module prototypes the most promising fp16 fusion target after projection
and shift-mix microbenchmarks: the one-token RWKV-7 recurrent update itself.
For each batch/head row it fuses the rank-1 state update and readout:

    ab = (-kk)[:, None] @ (kk * a)[None, :]
    new_state = state * w[None, :] + state @ ab + v[:, None] @ k[None, :]
    out = new_state @ r[:, None]

Using the rank-1 structure, ``state @ ab`` is computed without materializing
``ab``.  The implementation is optional: imports must work on CPU-only hosts
and fallback to the torch reference when Triton/CUDA is unavailable.
"""
from __future__ import annotations

from typing import Any

try:  # pragma: no cover - optional dependency in local no-CUDA tests
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:  # pragma: no cover - exercised on CUDA/Triton hosts
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


_HAS_TRITON = triton is not None and tl is not None


if _HAS_TRITON:

    @triton.jit
    def _recurrent_rank1_update_kernel(
        r_ptr,
        w_ptr,
        k_ptr,
        v_ptr,
        kk_ptr,
        a_ptr,
        state_ptr,
        out_ptr,
        new_state_ptr,
        n_rows: tl.constexpr,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row_id = tl.program_id(0)
        row_in_head = row_id % N
        bh_id = row_id // N
        offs = tl.arange(0, BLOCK_N)
        mask = offs < N

        vec_base = bh_id * N
        state_base = row_id * N
        st = tl.load(state_ptr + state_base + offs, mask=mask, other=0.0).to(tl.float32)
        kk = tl.load(kk_ptr + vec_base + offs, mask=mask, other=0.0).to(tl.float32)

        # Native formula: state @ ((-kk)[:, None] @ (kk*a)[None, :]).
        # For each row i this is -dot(state[i, :], kk) * kk[j] * a[j].
        state_dot_kk = tl.sum(st * kk, axis=0)

        w = tl.load(w_ptr + vec_base + offs, mask=mask, other=0.0).to(tl.float32)
        k = tl.load(k_ptr + vec_base + offs, mask=mask, other=0.0).to(tl.float32)
        a = tl.load(a_ptr + vec_base + offs, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(r_ptr + vec_base + offs, mask=mask, other=0.0).to(tl.float32)
        vi = tl.load(v_ptr + vec_base + row_in_head).to(tl.float32)

        new_st = st * w + vi * k - state_dot_kk * kk * a
        tl.store(new_state_ptr + state_base + offs, new_st, mask=mask)

        out_i = tl.sum(new_st * r, axis=0)
        tl.store(out_ptr + vec_base + row_in_head, out_i)


def fused_recurrent_update_available() -> bool:
    """Return whether the optional Triton recurrent update prototype can run."""

    return bool(_HAS_TRITON and torch is not None)


def _as_bhn(x: Any, H: int, N: int, *, name: str):
    if torch is None:
        raise RuntimeError("fused_recurrent_update requires torch")
    if x.dim() == 3:
        if int(x.shape[1]) != H or int(x.shape[2]) != N:
            raise ValueError(f"{name} must be shaped [batch,{H},{N}] or [batch,{H * N}]; got {tuple(x.shape)}")
        return x.contiguous(), False
    if x.dim() == 2:
        if int(x.shape[1]) != H * N:
            raise ValueError(f"{name} must be shaped [batch,{H},{N}] or [batch,{H * N}]; got {tuple(x.shape)}")
        return x.reshape(int(x.shape[0]), H, N).contiguous(), True
    raise ValueError(f"{name} must be shaped [batch,{H},{N}] or [batch,{H * N}]")


def torch_recurrent_update(r: Any, w: Any, k: Any, v: Any, kk: Any, a: Any, state: Any):
    """Reference one-token recurrent update matching the native_jit formula."""

    if torch is None:
        raise RuntimeError("torch_recurrent_update requires torch")
    if state.dim() != 4:
        raise ValueError("state must be shaped [batch, heads, head_dim, head_dim]")
    B, H, N, N2 = (int(vv) for vv in state.shape)
    if N != N2:
        raise ValueError("state must be square in the last two dimensions")
    r3, flat = _as_bhn(r, H, N, name="r")
    w3, _ = _as_bhn(w, H, N, name="w")
    k3, _ = _as_bhn(k, H, N, name="k")
    v3, _ = _as_bhn(v, H, N, name="v")
    kk3, _ = _as_bhn(kk, H, N, name="kk")
    a3, _ = _as_bhn(a, H, N, name="a")
    if int(r3.shape[0]) != B:
        raise ValueError("r/w/k/v/kk/a batch size must match state")

    vk = v3.view(B, H, N, 1) @ k3.view(B, H, 1, N)
    ab = (-kk3).view(B, H, N, 1) @ (kk3 * a3).view(B, H, 1, N)
    new_state = state * w3.view(B, H, 1, N) + state @ ab.float() + vk.float()
    out = new_state.to(r3.dtype) @ r3.view(B, H, N, 1)
    out = out.view(B, H, N)
    if flat:
        return out.reshape(B, H * N), new_state
    return out, new_state


def fused_recurrent_update(
    r: Any,
    w: Any,
    k: Any,
    v: Any,
    kk: Any,
    a: Any,
    state: Any,
    *,
    block_n: int = 64,
    force_fallback: bool = False,
):
    """Compute RWKV-7 one-token recurrent update with an optional Triton kernel.

    ``r,w,k,v,kk,a`` may be shaped ``[batch, heads, head_dim]`` or flattened as
    ``[batch, hidden]``. ``state`` must be ``[batch, heads, head_dim, head_dim]``
    and is not modified in place. The output shape follows the vector input rank.
    """

    if torch is None:
        raise RuntimeError("fused_recurrent_update requires torch")
    if state.dim() != 4:
        raise ValueError("state must be shaped [batch, heads, head_dim, head_dim]")
    B, H, N, N2 = (int(vv) for vv in state.shape)
    if N != N2:
        raise ValueError("state must be square in the last two dimensions")
    if int(block_n) < N:
        raise ValueError(f"block_n must be >= head_dim={N}; got {block_n}")
    r3, flat = _as_bhn(r, H, N, name="r")
    w3, _ = _as_bhn(w, H, N, name="w")
    k3, _ = _as_bhn(k, H, N, name="k")
    v3, _ = _as_bhn(v, H, N, name="v")
    kk3, _ = _as_bhn(kk, H, N, name="kk")
    a3, _ = _as_bhn(a, H, N, name="a")
    if int(r3.shape[0]) != B:
        raise ValueError("r/w/k/v/kk/a batch size must match state")

    use_triton = (
        not force_fallback
        and fused_recurrent_update_available()
        and r3.is_cuda
        and w3.is_cuda
        and k3.is_cuda
        and v3.is_cuda
        and kk3.is_cuda
        and a3.is_cuda
        and state.is_cuda
        and state.dtype == torch.float32
        and r3.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and all(t.dtype == r3.dtype for t in (w3, k3, v3, kk3, a3))
    )
    if not use_triton:
        return torch_recurrent_update(r3.reshape(B, H * N) if flat else r3, w3, k3, v3, kk3, a3, state)

    r_c = r3.contiguous()
    w_c = w3.contiguous()
    k_c = k3.contiguous()
    v_c = v3.contiguous()
    kk_c = kk3.contiguous()
    a_c = a3.contiguous()
    state_c = state.contiguous()
    out = torch.empty((B, H, N), device=r3.device, dtype=r3.dtype)
    new_state = torch.empty_like(state_c)
    grid = (B * H * N,)
    _recurrent_rank1_update_kernel[grid](
        r_c,
        w_c,
        k_c,
        v_c,
        kk_c,
        a_c,
        state_c,
        out,
        new_state,
        B * H * N,
        N,
        BLOCK_N=int(block_n),
        num_warps=2,
    )
    if flat:
        return out.reshape(B, H * N), new_state
    return out, new_state
