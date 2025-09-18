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

try:  # pragma: no cover - exercised on CUDA/Triton hosts
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]

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
    "dplr_dense_chunk_summary_triton",
    "dplr_dense_chunk_summary_triton_available",
    "dplr_dense_chunk_summary_torch",
]


_HAS_TRITON = triton is not None and tl is not None


if _HAS_TRITON:

    @triton.jit
    def _dense_chunk_summary_kernel(
        w_ptr,
        k_ptr,
        v_ptr,
        kk_ptr,
        a_ptr,
        transition_ptr,
        additive_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        N: tl.constexpr,
        CHUNKS: tl.constexpr,
        C: tl.constexpr,
        ROW_BLOCKS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Build dense chunk affine summaries using the DPLR rank-1 update.

        For one `(batch, chunk, head, row_block)` this computes row blocks of
        `P` and `Q` satisfying `S_end = S_start @ P + Q` for the chunk.  The
        kernel is intentionally dense-summary scaffolding for the three-stage
        WY backend: it pins down the chunk-summary boundary while the next
        iteration swaps dense `P/Q` for compact WY factors.
        """

        pid = tl.program_id(0)
        row_block = pid % ROW_BLOCKS
        tmp = pid // ROW_BLOCKS
        head_id = tmp % H
        chunk_id = (tmp // H) % CHUNKS
        batch_id = tmp // (H * CHUNKS)

        offs_i = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_j = tl.arange(0, BLOCK_N)
        mask_i = offs_i < N
        mask_j = offs_j < N

        # Row block of identity transition and zero additive term.
        transition = tl.where(offs_i[:, None] == offs_j[None, :], 1.0, 0.0).to(tl.float32)
        additive = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        i = 0
        while i < C:
            t = chunk_id * C + i
            vec_base = ((batch_id * T + t) * H + head_id) * N
            w = tl.load(w_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            key = tl.load(k_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            kk = tl.load(kk_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            a = tl.load(a_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            v_rows = tl.load(v_ptr + vec_base + offs_i, mask=mask_i, other=0.0).to(tl.float32)

            # A_i = diag(w_i) + p_i q_i^T, p_i=-kk_i, q_i=kk_i*a_i.
            p = -kk
            q = kk * a
            transition_dot_p = tl.sum(transition * p[None, :], axis=1)
            additive_dot_p = tl.sum(additive * p[None, :], axis=1)
            transition = transition * w[None, :] + transition_dot_p[:, None] * q[None, :]
            additive = additive * w[None, :] + additive_dot_p[:, None] * q[None, :] + v_rows[:, None] * key[None, :]
            i += 1

        summary_base = (((batch_id * CHUNKS + chunk_id) * H + head_id) * N + offs_i[:, None]) * N + offs_j[None, :]
        mask = mask_i[:, None] & mask_j[None, :]
        tl.store(transition_ptr + summary_base, transition, mask=mask)
        tl.store(additive_ptr + summary_base, additive, mask=mask)


def dplr_chunk_scan_triton_available() -> bool:
    """Return whether the opt-in compiled scan can run on this host."""

    return bool(
        torch is not None
        and fused_recurrent_scan is not None
        and fused_recurrent_scan_available is not None
        and fused_recurrent_scan_available()
    )


def dplr_dense_chunk_summary_triton_available() -> bool:
    """Return whether the dense chunk-summary Triton probe can run."""

    return bool(_HAS_TRITON and torch is not None)


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


def _check_same_bthn(name: str, x: Any, *, shape: tuple[int, int, int, int], device: Any) -> None:
    if tuple(int(v) for v in x.shape) != shape:
        raise ValueError(f"{name} shape must match w; got {tuple(x.shape)}, expected {shape}")
    if x.device != device:
        raise ValueError(f"{name} must be on the same device as w")
    if not x.is_floating_point():
        raise TypeError(f"{name} must be a floating point tensor")


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


def dplr_dense_chunk_summary_torch(w: Any, k: Any, v: Any, kk: Any, a: Any, *, chunk_size: int = 64):
    """Reference dense affine chunk summaries.

    Returns `transition` and `additive` shaped `[B, chunks, H, N, N]` where each
    chunk satisfies `S_end = S_start @ transition + additive`.  This is a
    correctness oracle for the compiled summary kernel and a bridge toward the
    future compact WY summary.
    """

    if torch is None:
        raise RuntimeError("dplr_dense_chunk_summary_torch requires torch")
    chunk_size_i = _validate_chunk_size(chunk_size)
    if not hasattr(w, "dim"):
        raise TypeError("w must be a torch.Tensor")
    if w.dim() != 4:
        raise ValueError("summary inputs must be shaped [batch,tokens,heads,head_dim]")
    B, T, H, N = (int(vv) for vv in w.shape)
    if T % chunk_size_i != 0:
        raise ValueError(f"T={T} must be divisible by chunk_size={chunk_size_i} for the summary prototype")
    shape = (B, T, H, N)
    for name, x in (("k", k), ("v", v), ("kk", kk), ("a", a)):
        _check_same_bthn(name, x, shape=shape, device=w.device)

    chunks = T // chunk_size_i
    transition_rows = []
    additive_rows = []
    eye = torch.eye(N, device=w.device, dtype=torch.float32).view(1, 1, N, N).expand(B, H, N, N)
    for chunk in range(chunks):
        trans = eye.clone()
        add = torch.zeros((B, H, N, N), device=w.device, dtype=torch.float32)
        start = chunk * chunk_size_i
        for local_i in range(chunk_size_i):
            t = start + local_i
            w_i = w[:, t].float()
            k_i = k[:, t].float()
            v_i = v[:, t].float()
            kk_i = kk[:, t].float()
            a_i = a[:, t].float()
            p_i = -kk_i
            q_i = kk_i * a_i
            trans_dot_p = torch.sum(trans * p_i.unsqueeze(-2), dim=-1)
            add_dot_p = torch.sum(add * p_i.unsqueeze(-2), dim=-1)
            trans = trans * w_i.unsqueeze(-2) + trans_dot_p.unsqueeze(-1) * q_i.unsqueeze(-2)
            add = add * w_i.unsqueeze(-2) + add_dot_p.unsqueeze(-1) * q_i.unsqueeze(-2)
            add = add + v_i.unsqueeze(-1) * k_i.unsqueeze(-2)
        transition_rows.append(trans)
        additive_rows.append(add)
    return {
        "algorithm": "torch_dense_dplr_summary",
        "chunk_size": chunk_size_i,
        "transition": torch.stack(transition_rows, dim=1),
        "additive": torch.stack(additive_rows, dim=1),
    }


def dplr_dense_chunk_summary_triton(
    w: Any,
    k: Any,
    v: Any,
    kk: Any,
    a: Any,
    *,
    chunk_size: int = 64,
    block_m: int | None = None,
    block_n: int | None = None,
    force_fallback: bool = False,
):
    """Build dense DPLR chunk summaries with a Triton row-block kernel.

    This is the first explicit chunk-summary kernel boundary for the future
    three-stage WY backend.  It intentionally returns dense `P/Q` summaries as
    a correctness scaffold; production WY work should replace those dense
    tensors with compact factors and then add prefix-combine and chunk-apply
    kernels.
    """

    if torch is None:
        raise RuntimeError("dplr_dense_chunk_summary_triton requires torch")
    chunk_size_i = _validate_chunk_size(chunk_size)
    if not hasattr(w, "dim"):
        raise TypeError("w must be a torch.Tensor")
    if w.dim() != 4:
        raise ValueError("summary inputs must be shaped [batch,tokens,heads,head_dim]")
    B, T, H, N = (int(vv) for vv in w.shape)
    if T % chunk_size_i != 0:
        raise ValueError(f"T={T} must be divisible by chunk_size={chunk_size_i} for the summary prototype")
    shape = (B, T, H, N)
    for name, x in (("k", k), ("v", v), ("kk", kk), ("a", a)):
        _check_same_bthn(name, x, shape=shape, device=w.device)
    if not w.is_floating_point():
        raise TypeError("w must be a floating point tensor")
    if block_m is None:
        block_m = _env_int("RWKV7_DPLR_TRITON_SUMMARY_BLOCK_M", 8)
    if block_n is None:
        block_n = N
    block_m = int(block_m)
    block_n = int(block_n)
    if block_m <= 0:
        raise ValueError("block_m must be positive")
    if block_n < N:
        raise ValueError(f"block_n must be >= head_dim={N}; got {block_n}")

    use_triton = (
        not force_fallback
        and dplr_dense_chunk_summary_triton_available()
        and w.is_cuda
        and k.is_cuda
        and v.is_cuda
        and kk.is_cuda
        and a.is_cuda
    )
    if not use_triton:
        return dplr_dense_chunk_summary_torch(w, k, v, kk, a, chunk_size=chunk_size_i)

    chunks = T // chunk_size_i
    w_c = w.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    kk_c = kk.contiguous()
    a_c = a.contiguous()
    transition = torch.empty((B, chunks, H, N, N), device=w.device, dtype=torch.float32)
    additive = torch.empty_like(transition)
    row_blocks = triton.cdiv(N, block_m)
    _dense_chunk_summary_kernel[(B * chunks * H * row_blocks,)](
        w_c,
        k_c,
        v_c,
        kk_c,
        a_c,
        transition,
        additive,
        T,
        H,
        N,
        chunks,
        chunk_size_i,
        ROW_BLOCKS=int(row_blocks),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4 if block_m < N else 8,
    )
    return {
        "algorithm": "triton_dense_dplr_summary",
        "chunk_size": chunk_size_i,
        "transition": transition,
        "additive": additive,
    }


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
