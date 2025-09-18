#!/usr/bin/env python3
# coding=utf-8
"""Opt-in compiled DPLR/WY prefill scan prototype.

This module is the first compiled-backend hook for the DPLR/WY prefill line.
It intentionally keeps the public boundary identical to
``dplr_prefill.dplr_chunk_scan`` so the synthetic benchmark can switch between
pure torch and compiled implementations without touching HF model code.

Backend notes
-------------
The ``triton_wy`` P0 path is a correctness/performance bridge, not the final
compact WY factor scan.  It delegates the per-token rank-1 DPLR recurrence to
the existing Triton recurrent scan kernel from ``fused_recurrent_update``.  That
kernel is still mathematically the RWKV-7 DPLR update

    S_t = S_{t-1} (diag(w_t) + (-kk_t)(kk_t*a_t)^T) + v_t k_t^T

and uses fp32 state accumulation.

The ``triton_dense3`` path below is the explicit three-stage scaffold:
chunk-summary, chunk-level prefix combine, and chunk apply/output.  It
materializes dense ``[N,N]`` summaries to prove the kernel boundaries before the
next iteration replaces them with compact WY factors.
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
    "dplr_compact_wy_chunk_summary_torch",
    "dplr_compact_wy_chunk_summary_triton",
    "dplr_compact_wy_chunk_summary_triton_available",
    "dplr_compact_wy_summary_to_dense",
    "dplr_compact_wy_apply_summaries_torch",
    "dplr_compact_wy_prefix_combine_torch",
    "dplr_compact_wy_prefix_combine_triton",
    "dplr_compact_wy_prefix_combine_triton_available",
    "dplr_dense_prefix_combine_torch",
    "dplr_dense_prefix_combine_triton",
    "dplr_dense_chunk_apply_torch",
    "dplr_dense_chunk_apply_triton",
    "dplr_dense_three_stage_triton",
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

    @triton.jit
    def _compact_wy_summary_kernel(
        w_ptr,
        k_ptr,
        v_ptr,
        kk_ptr,
        a_ptr,
        diag_ptr,
        trans_left_ptr,
        trans_right_ptr,
        add_left_ptr,
        add_right_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        N: tl.constexpr,
        CHUNKS: tl.constexpr,
        C: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        """Build compact WY/low-rank chunk factors for one chunk/head.

        This first target-shape kernel owns a full `(N, C)` factor tile for one
        `(batch, chunk, head)`.  It is intentionally constrained to
        `BLOCK_N >= N` and `BLOCK_R >= C` so the compact metadata contract can
        be validated before splitting the kernel for broader shapes.
        """

        pid = tl.program_id(0)
        head_id = pid % H
        chunk_id = (pid // H) % CHUNKS
        batch_id = pid // (H * CHUNKS)

        offs_n = tl.arange(0, BLOCK_N)
        offs_r = tl.arange(0, BLOCK_R)
        mask_n = offs_n < N
        mask_r = offs_r < C
        factor_mask = mask_n[:, None] & mask_r[None, :]

        diag = tl.full((BLOCK_N,), 1.0, dtype=tl.float32)
        trans_left = tl.zeros((BLOCK_N, BLOCK_R), dtype=tl.float32)
        trans_right = tl.zeros((BLOCK_N, BLOCK_R), dtype=tl.float32)
        add_left = tl.zeros((BLOCK_N, BLOCK_R), dtype=tl.float32)
        add_right = tl.zeros((BLOCK_N, BLOCK_R), dtype=tl.float32)

        i = 0
        while i < C:
            t = chunk_id * C + i
            vec_base = ((batch_id * T + t) * H + head_id) * N
            w = tl.load(w_ptr + vec_base + offs_n, mask=mask_n, other=0.0).to(tl.float32)
            key = tl.load(k_ptr + vec_base + offs_n, mask=mask_n, other=0.0).to(tl.float32)
            val = tl.load(v_ptr + vec_base + offs_n, mask=mask_n, other=0.0).to(tl.float32)
            kk = tl.load(kk_ptr + vec_base + offs_n, mask=mask_n, other=0.0).to(tl.float32)
            aval = tl.load(a_ptr + vec_base + offs_n, mask=mask_n, other=0.0).to(tl.float32)

            p = -kk
            q = kk * aval

            trans_coeff = tl.sum(trans_right * p[:, None], axis=0)
            new_left_col = diag * p + tl.sum(trans_left * trans_coeff[None, :], axis=1)
            trans_right = trans_right * w[:, None]
            diag = diag * w
            trans_left = tl.where(offs_r[None, :] == i, new_left_col[:, None], trans_left)
            trans_right = tl.where(offs_r[None, :] == i, q[:, None], trans_right)

            add_coeff = tl.sum(add_right * p[:, None], axis=0)
            add_right = add_right * w[:, None] + q[:, None] * add_coeff[None, :]
            add_left = tl.where(offs_r[None, :] == i, val[:, None], add_left)
            add_right = tl.where(offs_r[None, :] == i, key[:, None], add_right)
            i += 1

        diag_base = ((batch_id * CHUNKS + chunk_id) * H + head_id) * N
        tl.store(diag_ptr + diag_base + offs_n, diag, mask=mask_n)

        factor_base = (((batch_id * CHUNKS + chunk_id) * H + head_id) * N + offs_n[:, None]) * C + offs_r[None, :]
        tl.store(trans_left_ptr + factor_base, trans_left, mask=factor_mask)
        tl.store(trans_right_ptr + factor_base, trans_right, mask=factor_mask)
        tl.store(add_left_ptr + factor_base, add_left, mask=factor_mask)
        tl.store(add_right_ptr + factor_base, add_right, mask=factor_mask)

    @triton.jit
    def _compact_wy_prefix_combine_kernel(
        state_ptr,
        diag_ptr,
        trans_left_ptr,
        trans_right_ptr,
        add_left_ptr,
        add_right_ptr,
        start_state_ptr,
        final_state_ptr,
        H: tl.constexpr,
        N: tl.constexpr,
        CHUNKS: tl.constexpr,
        R: tl.constexpr,
        ROW_BLOCKS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        """Compute chunk start states directly from compact WY factors."""

        pid = tl.program_id(0)
        row_block = pid % ROW_BLOCKS
        bh_id = pid // ROW_BLOCKS
        head_id = bh_id % H
        batch_id = bh_id // H

        offs_i = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_j = tl.arange(0, BLOCK_N)
        offs_r = tl.arange(0, BLOCK_R)
        mask_i = offs_i < N
        mask_j = offs_j < N
        mask_r = offs_r < R
        row_mask = mask_i[:, None] & mask_j[None, :]

        state_base = (batch_id * H + head_id) * N * N
        cur = tl.load(
            state_ptr + state_base + offs_i[:, None] * N + offs_j[None, :],
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)

        chunk = 0
        while chunk < CHUNKS:
            summary_base = ((batch_id * CHUNKS + chunk) * H + head_id) * N
            factor_base = summary_base * R

            start_base = ((batch_id * CHUNKS + chunk) * H + head_id) * N * N
            tl.store(start_state_ptr + start_base + offs_i[:, None] * N + offs_j[None, :], cur, mask=row_mask)

            diag = tl.load(diag_ptr + summary_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            factor_mask = mask_j[:, None] & mask_r[None, :]
            trans_left = tl.load(
                trans_left_ptr + factor_base + offs_j[:, None] * R + offs_r[None, :],
                mask=factor_mask,
                other=0.0,
            ).to(tl.float32)
            trans_right = tl.load(
                trans_right_ptr + factor_base + offs_j[:, None] * R + offs_r[None, :],
                mask=factor_mask,
                other=0.0,
            ).to(tl.float32)
            add_left = tl.load(
                add_left_ptr + factor_base + offs_i[:, None] * R + offs_r[None, :],
                mask=mask_i[:, None] & mask_r[None, :],
                other=0.0,
            ).to(tl.float32)
            add_right = tl.load(
                add_right_ptr + factor_base + offs_j[:, None] * R + offs_r[None, :],
                mask=factor_mask,
                other=0.0,
            ).to(tl.float32)

            trans_coeff = tl.dot(cur, trans_left, input_precision="ieee")
            next_cur = cur * diag[None, :]
            next_cur += tl.dot(trans_coeff, tl.trans(trans_right), input_precision="ieee")
            next_cur += tl.dot(add_left, tl.trans(add_right), input_precision="ieee")

            cur = next_cur
            chunk += 1

        tl.store(final_state_ptr + state_base + offs_i[:, None] * N + offs_j[None, :], cur, mask=row_mask)

    @triton.jit
    def _dense_prefix_combine_kernel(
        state_ptr,
        transition_ptr,
        additive_ptr,
        start_state_ptr,
        final_state_ptr,
        H: tl.constexpr,
        N: tl.constexpr,
        CHUNKS: tl.constexpr,
        ROW_BLOCKS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Compute chunk start states from dense chunk summaries.

        This is stage 2 of the dense three-stage scaffold:

            start[c] = state before chunk c
            cur = cur @ transition[c] + additive[c]

        The implementation deliberately uses explicit column reductions rather
        than `tl.dot` to keep fp32 behavior stable for correctness probes.
        """

        pid = tl.program_id(0)
        row_block = pid % ROW_BLOCKS
        bh_id = pid // ROW_BLOCKS
        head_id = bh_id % H
        batch_id = bh_id // H

        offs_i = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_j = tl.arange(0, BLOCK_N)
        mask_i = offs_i < N
        mask_j = offs_j < N
        row_mask = mask_i[:, None] & mask_j[None, :]

        state_base = (batch_id * H + head_id) * N * N
        cur = tl.load(
            state_ptr + state_base + offs_i[:, None] * N + offs_j[None, :],
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)

        chunk = 0
        while chunk < CHUNKS:
            summary_base = ((batch_id * CHUNKS + chunk) * H + head_id) * N * N
            row_base = summary_base + offs_i[:, None] * N + offs_j[None, :]
            tl.store(start_state_ptr + row_base, cur, mask=row_mask)

            next_cur = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            col = 0
            while col < N:
                p_col = tl.load(
                    transition_ptr + summary_base + offs_j * N + col,
                    mask=mask_j,
                    other=0.0,
                ).to(tl.float32)
                dot = tl.sum(cur * p_col[None, :], axis=1)
                q_col = tl.load(
                    additive_ptr + summary_base + offs_i * N + col,
                    mask=mask_i,
                    other=0.0,
                ).to(tl.float32)
                val = dot + q_col
                next_cur = tl.where(offs_j[None, :] == col, val[:, None], next_cur)
                col += 1

            cur = next_cur
            chunk += 1

        tl.store(
            final_state_ptr + state_base + offs_i[:, None] * N + offs_j[None, :],
            cur,
            mask=row_mask,
        )

    @triton.jit
    def _dense_chunk_apply_kernel(
        r_ptr,
        w_ptr,
        k_ptr,
        v_ptr,
        kk_ptr,
        a_ptr,
        start_state_ptr,
        out_ptr,
        chunk_end_state_ptr,
        T: tl.constexpr,
        H: tl.constexpr,
        N: tl.constexpr,
        CHUNKS: tl.constexpr,
        C: tl.constexpr,
        ROW_BLOCKS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Apply per-token DPLR scan inside each chunk from chunk start states.

        This is stage 3 of the dense scaffold.  It mirrors the existing
        row-split recurrent scan, but the grid owns `(batch, chunk, head,
        row_block)` and starts from the prefix-combined chunk start state.
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
        row_mask = mask_i[:, None] & mask_j[None, :]
        summary_base = ((batch_id * CHUNKS + chunk_id) * H + head_id) * N * N
        st = tl.load(
            start_state_ptr + summary_base + offs_i[:, None] * N + offs_j[None, :],
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)

        local_i = 0
        while local_i < C:
            t = chunk_id * C + local_i
            vec_base = ((batch_id * T + t) * H + head_id) * N
            r = tl.load(r_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            key = tl.load(k_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            kk = tl.load(kk_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            a = tl.load(a_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            v_rows = tl.load(v_ptr + vec_base + offs_i, mask=mask_i, other=0.0).to(tl.float32)

            state_dot_kk = tl.sum(st * kk[None, :], axis=1)
            st = st * w[None, :] + v_rows[:, None] * key[None, :] - state_dot_kk[:, None] * kk[None, :] * a[None, :]
            recurrent = tl.sum(st * r[None, :], axis=1)
            tl.store(out_ptr + vec_base + offs_i, recurrent, mask=mask_i)
            local_i += 1

        tl.store(chunk_end_state_ptr + summary_base + offs_i[:, None] * N + offs_j[None, :], st, mask=row_mask)


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


def dplr_compact_wy_chunk_summary_triton_available() -> bool:
    """Return whether the compact WY summary Triton probe can run."""

    return bool(_HAS_TRITON and torch is not None)


def dplr_compact_wy_prefix_combine_triton_available() -> bool:
    """Return whether the compact WY prefix-combine Triton probe can run."""

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


def _compact_factor_dot(factors: Any, vec: Any):
    """Return ``factors^T vec`` for compact factors shaped ``[..., N, R]``."""

    if int(factors.shape[-1]) == 0:
        return vec.new_zeros((*vec.shape[:-1], 0))
    return torch.einsum("...nr,...n->...r", factors, vec)


def _compact_factor_weighted_sum(factors: Any, weights: Any):
    """Return ``factors weights`` for compact factors shaped ``[..., N, R]``."""

    if int(factors.shape[-1]) == 0:
        return factors.new_zeros(factors.shape[:-1])
    return torch.einsum("...nr,...r->...n", factors, weights)


def _compact_append_factor_column(factors: Any, col: Any):
    return torch.cat((factors, col.unsqueeze(-1)), dim=-1)


def _compact_outer_to_dense(left: Any, right: Any):
    if int(left.shape[-1]) == 0:
        return left.new_zeros((*left.shape[:-1], int(left.shape[-2])))
    return left @ right.transpose(-1, -2)


def _compact_apply_transition_to_state(state: Any, diag: Any, left: Any, right: Any):
    out = state.float() * diag.float().unsqueeze(-2)
    if int(left.shape[-1]) != 0:
        out = out + (state.float() @ left.float()) @ right.float().transpose(-1, -2)
    return out


def dplr_compact_wy_chunk_summary_torch(
    w: Any,
    k: Any,
    v: Any,
    kk: Any,
    a: Any,
    *,
    chunk_size: int = 64,
):
    """Reference compact WY/low-rank chunk summaries.

    Returns compact factors for every chunk, without materializing dense
    ``[N,N]`` transition/additive summaries:

    ``transition = diag_embed(transition_diag) + transition_left @ transition_right.T``
    ``additive   = additive_left @ additive_right.T``

    Rank grows by one per token in the chunk, so this is still a correctness
    reference, but it establishes the compact metadata contract that the next
    Triton kernels should write directly.
    """

    if torch is None:
        raise RuntimeError("dplr_compact_wy_chunk_summary_torch requires torch")
    chunk_size_i = _validate_chunk_size(chunk_size)
    if not hasattr(w, "dim"):
        raise TypeError("w must be a torch.Tensor")
    if w.dim() != 4:
        raise ValueError("summary inputs must be shaped [batch,tokens,heads,head_dim]")
    B, T, H, N = (int(vv) for vv in w.shape)
    if T % chunk_size_i != 0:
        raise ValueError(f"T={T} must be divisible by chunk_size={chunk_size_i} for the compact WY prototype")
    shape = (B, T, H, N)
    for name, x in (("k", k), ("v", v), ("kk", kk), ("a", a)):
        _check_same_bthn(name, x, shape=shape, device=w.device)
    if not w.is_floating_point():
        raise TypeError("w must be a floating point tensor")

    chunks = T // chunk_size_i
    diag_rows = []
    trans_left_rows = []
    trans_right_rows = []
    add_left_rows = []
    add_right_rows = []
    for chunk in range(chunks):
        diag = torch.ones((B, H, N), device=w.device, dtype=torch.float32)
        trans_left = torch.empty((B, H, N, 0), device=w.device, dtype=torch.float32)
        trans_right = torch.empty((B, H, N, 0), device=w.device, dtype=torch.float32)
        add_left = torch.empty((B, H, N, 0), device=w.device, dtype=torch.float32)
        add_right = torch.empty((B, H, N, 0), device=w.device, dtype=torch.float32)

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

            new_left_col = diag * p_i + _compact_factor_weighted_sum(
                trans_left,
                _compact_factor_dot(trans_right, p_i),
            )
            trans_right = trans_right * w_i.unsqueeze(-1)
            diag = diag * w_i
            trans_left = _compact_append_factor_column(trans_left, new_left_col)
            trans_right = _compact_append_factor_column(trans_right, q_i)

            if int(add_right.shape[-1]) != 0:
                add_coeff = _compact_factor_dot(add_right, p_i)
                add_right = add_right * w_i.unsqueeze(-1) + q_i.unsqueeze(-1) * add_coeff.unsqueeze(-2)
            add_left = _compact_append_factor_column(add_left, v_i)
            add_right = _compact_append_factor_column(add_right, k_i)

        diag_rows.append(diag)
        trans_left_rows.append(trans_left)
        trans_right_rows.append(trans_right)
        add_left_rows.append(add_left)
        add_right_rows.append(add_right)

    return {
        "algorithm": "torch_compact_wy_summary",
        "chunk_size": chunk_size_i,
        "rank": chunk_size_i,
        "transition_diag": torch.stack(diag_rows, dim=1),
        "transition_left": torch.stack(trans_left_rows, dim=1),
        "transition_right": torch.stack(trans_right_rows, dim=1),
        "additive_left": torch.stack(add_left_rows, dim=1),
        "additive_right": torch.stack(add_right_rows, dim=1),
    }


def dplr_compact_wy_chunk_summary_triton(
    w: Any,
    k: Any,
    v: Any,
    kk: Any,
    a: Any,
    *,
    chunk_size: int = 64,
    block_n: int | None = None,
    block_r: int | None = None,
    force_fallback: bool = False,
):
    """Build compact WY/low-rank chunk summaries with a target-shape Triton kernel."""

    if torch is None:
        raise RuntimeError("dplr_compact_wy_chunk_summary_triton requires torch")
    chunk_size_i = _validate_chunk_size(chunk_size)
    if not hasattr(w, "dim"):
        raise TypeError("w must be a torch.Tensor")
    if w.dim() != 4:
        raise ValueError("summary inputs must be shaped [batch,tokens,heads,head_dim]")
    B, T, H, N = (int(vv) for vv in w.shape)
    if T % chunk_size_i != 0:
        raise ValueError(f"T={T} must be divisible by chunk_size={chunk_size_i} for the compact WY prototype")
    shape = (B, T, H, N)
    for name, x in (("k", k), ("v", v), ("kk", kk), ("a", a)):
        _check_same_bthn(name, x, shape=shape, device=w.device)
    if not w.is_floating_point():
        raise TypeError("w must be a floating point tensor")
    if block_n is None:
        block_n = _env_int("RWKV7_DPLR_TRITON_COMPACT_BLOCK_N", N)
    if block_r is None:
        block_r = _env_int("RWKV7_DPLR_TRITON_COMPACT_BLOCK_R", chunk_size_i)
    block_n = int(block_n)
    block_r = int(block_r)
    if block_n < N:
        raise ValueError(f"block_n must be >= head_dim={N}; got {block_n}")
    if block_r < chunk_size_i:
        raise ValueError(f"block_r must be >= chunk_size={chunk_size_i}; got {block_r}")

    target_supported = N <= 64 and chunk_size_i <= 64 and block_n <= 64 and block_r <= 64
    use_triton = (
        not force_fallback
        and target_supported
        and dplr_compact_wy_chunk_summary_triton_available()
        and w.is_cuda
        and k.is_cuda
        and v.is_cuda
        and kk.is_cuda
        and a.is_cuda
    )
    if not use_triton:
        return dplr_compact_wy_chunk_summary_torch(w, k, v, kk, a, chunk_size=chunk_size_i)

    chunks = T // chunk_size_i
    w_c = w.contiguous()
    k_c = k.contiguous()
    v_c = v.contiguous()
    kk_c = kk.contiguous()
    a_c = a.contiguous()
    transition_diag = torch.empty((B, chunks, H, N), device=w.device, dtype=torch.float32)
    transition_left = torch.empty((B, chunks, H, N, chunk_size_i), device=w.device, dtype=torch.float32)
    transition_right = torch.empty_like(transition_left)
    additive_left = torch.empty_like(transition_left)
    additive_right = torch.empty_like(transition_left)
    _compact_wy_summary_kernel[(B * chunks * H,)](
        w_c,
        k_c,
        v_c,
        kk_c,
        a_c,
        transition_diag,
        transition_left,
        transition_right,
        additive_left,
        additive_right,
        T,
        H,
        N,
        chunks,
        chunk_size_i,
        BLOCK_N=block_n,
        BLOCK_R=block_r,
        num_warps=8 if max(block_n, block_r) >= 64 else 4,
    )
    return {
        "algorithm": "triton_compact_wy_summary",
        "chunk_size": chunk_size_i,
        "rank": chunk_size_i,
        "transition_diag": transition_diag,
        "transition_left": transition_left,
        "transition_right": transition_right,
        "additive_left": additive_left,
        "additive_right": additive_right,
    }


def dplr_compact_wy_summary_to_dense(summary: dict[str, Any]):
    """Materialize dense summaries from compact WY factors for correctness tests."""

    if torch is None:
        raise RuntimeError("dplr_compact_wy_summary_to_dense requires torch")
    diag = summary["transition_diag"].float()
    trans_left = summary["transition_left"].float()
    trans_right = summary["transition_right"].float()
    add_left = summary["additive_left"].float()
    add_right = summary["additive_right"].float()
    if diag.dim() != 4 or trans_left.dim() != 5 or trans_right.dim() != 5:
        raise ValueError("compact transition summary must be [B,chunks,H,N] plus [B,chunks,H,N,R] factors")
    if add_left.dim() != 5 or add_right.dim() != 5:
        raise ValueError("compact additive summary must be [B,chunks,H,N,R] factors")
    if tuple(trans_left.shape) != tuple(trans_right.shape):
        raise ValueError("transition_left and transition_right shapes must match")
    if tuple(add_left.shape) != tuple(add_right.shape):
        raise ValueError("additive_left and additive_right shapes must match")
    B, chunks, H, N = (int(vv) for vv in diag.shape)
    eye = torch.eye(N, device=diag.device, dtype=torch.float32).view(1, 1, 1, N, N)
    transition = eye * diag.unsqueeze(-2) + _compact_outer_to_dense(trans_left, trans_right)
    additive = _compact_outer_to_dense(add_left, add_right)
    return {
        "algorithm": "torch_compact_wy_summary_dense_oracle",
        "chunk_size": int(summary.get("chunk_size", 0)),
        "rank": int(summary.get("rank", int(trans_left.shape[-1]))),
        "transition": transition.reshape(B, chunks, H, N, N),
        "additive": additive.reshape(B, chunks, H, N, N),
    }


def dplr_compact_wy_apply_summaries_torch(state: Any, summary: dict[str, Any]):
    """Apply compact WY summaries to an initial native VxK state."""

    if torch is None:
        raise RuntimeError("dplr_compact_wy_apply_summaries_torch requires torch")
    if state.dim() != 4:
        raise ValueError("state must be [B,H,N,N]")
    diag = summary["transition_diag"]
    trans_left = summary["transition_left"]
    trans_right = summary["transition_right"]
    add_left = summary["additive_left"]
    add_right = summary["additive_right"]
    B, H, N, N2 = (int(vv) for vv in state.shape)
    if N != N2:
        raise ValueError("state must be square in the last two dimensions")
    if int(diag.shape[0]) != B or int(diag.shape[2]) != H or int(diag.shape[3]) != N:
        raise ValueError("compact summary shapes must match state")

    cur = state.float()
    chunks = int(diag.shape[1])
    for chunk in range(chunks):
        cur = _compact_apply_transition_to_state(
            cur,
            diag[:, chunk],
            trans_left[:, chunk],
            trans_right[:, chunk],
        )
        cur = cur + _compact_outer_to_dense(add_left[:, chunk].float(), add_right[:, chunk].float())
    return cur


def dplr_compact_wy_prefix_combine_torch(state: Any, summary: dict[str, Any]):
    """Reference chunk-prefix combine directly over compact WY factors.

    Returns `(start_states, final_state)` where `start_states[:, c]` is the
    native VxK state before chunk `c`.  Unlike the dense prefix-combine oracle,
    this never stores dense transition/additive summaries; it applies
    `diag + U V^T` and `X Y^T` factors directly to the dense recurrent state.
    """

    if torch is None:
        raise RuntimeError("dplr_compact_wy_prefix_combine_torch requires torch")
    if state.dim() != 4:
        raise ValueError("state must be [B,H,N,N]")
    diag = summary["transition_diag"]
    trans_left = summary["transition_left"]
    trans_right = summary["transition_right"]
    add_left = summary["additive_left"]
    add_right = summary["additive_right"]
    B, H, N, N2 = (int(vv) for vv in state.shape)
    if N != N2:
        raise ValueError("state must be square in the last two dimensions")
    if diag.dim() != 4:
        raise ValueError("transition_diag must be [B,chunks,H,N]")
    if tuple(trans_left.shape) != tuple(trans_right.shape):
        raise ValueError("transition_left and transition_right shapes must match")
    if tuple(add_left.shape) != tuple(add_right.shape):
        raise ValueError("additive_left and additive_right shapes must match")
    if trans_left.dim() != 5 or add_left.dim() != 5:
        raise ValueError("compact factors must be [B,chunks,H,N,R]")
    chunks = int(diag.shape[1])
    if (
        int(diag.shape[0]) != B
        or int(diag.shape[2]) != H
        or int(diag.shape[3]) != N
        or int(trans_left.shape[0]) != B
        or int(trans_left.shape[1]) != chunks
        or int(trans_left.shape[2]) != H
        or int(trans_left.shape[3]) != N
        or int(add_left.shape[0]) != B
        or int(add_left.shape[1]) != chunks
        or int(add_left.shape[2]) != H
        or int(add_left.shape[3]) != N
    ):
        raise ValueError("compact summary shapes must match state")

    cur = state.float()
    starts = []
    for chunk in range(chunks):
        starts.append(cur)
        cur = _compact_apply_transition_to_state(
            cur,
            diag[:, chunk],
            trans_left[:, chunk],
            trans_right[:, chunk],
        )
        cur = cur + _compact_outer_to_dense(add_left[:, chunk].float(), add_right[:, chunk].float())
    return torch.stack(starts, dim=1), cur


def dplr_compact_wy_prefix_combine_triton(
    state: Any,
    summary: dict[str, Any],
    *,
    block_m: int | None = None,
    block_n: int | None = None,
    force_fallback: bool = False,
):
    """Triton chunk-prefix combine directly over compact WY factors."""

    if torch is None:
        raise RuntimeError("dplr_compact_wy_prefix_combine_triton requires torch")
    if state.dim() != 4:
        raise ValueError("state must be [B,H,N,N]")
    diag = summary["transition_diag"]
    trans_left = summary["transition_left"]
    trans_right = summary["transition_right"]
    add_left = summary["additive_left"]
    add_right = summary["additive_right"]
    B, H, N, N2 = (int(vv) for vv in state.shape)
    if N != N2:
        raise ValueError("state must be square in the last two dimensions")
    if diag.dim() != 4 or trans_left.dim() != 5 or trans_right.dim() != 5 or add_left.dim() != 5 or add_right.dim() != 5:
        raise ValueError("compact summary must be [B,chunks,H,N] plus [B,chunks,H,N,R] factors")
    chunks = int(diag.shape[1])
    R = int(trans_left.shape[-1])
    if R <= 0:
        raise ValueError("compact factor rank must be positive")
    if tuple(trans_left.shape) != tuple(trans_right.shape):
        raise ValueError("transition_left and transition_right shapes must match")
    if tuple(add_left.shape) != tuple(add_right.shape):
        raise ValueError("additive_left and additive_right shapes must match")
    if (
        int(diag.shape[0]) != B
        or int(diag.shape[2]) != H
        or int(diag.shape[3]) != N
        or int(trans_left.shape[0]) != B
        or int(trans_left.shape[1]) != chunks
        or int(trans_left.shape[2]) != H
        or int(trans_left.shape[3]) != N
        or int(add_left.shape[0]) != B
        or int(add_left.shape[1]) != chunks
        or int(add_left.shape[2]) != H
        or int(add_left.shape[3]) != N
    ):
        raise ValueError("compact summary shapes must match state")
    if block_m is None:
        block_m = _env_int("RWKV7_DPLR_TRITON_COMPACT_PREFIX_BLOCK_M", 8)
    if block_n is None:
        block_n = N
    block_m = int(block_m)
    block_n = int(block_n)
    if block_m <= 0:
        raise ValueError("block_m must be positive")
    if block_n < N:
        raise ValueError(f"block_n must be >= head_dim={N}; got {block_n}")

    target_supported = 16 <= N <= 64 and 16 <= R <= 64 and block_n <= 64
    use_triton = (
        not force_fallback
        and target_supported
        and dplr_compact_wy_prefix_combine_triton_available()
        and state.is_cuda
        and diag.is_cuda
        and trans_left.is_cuda
        and trans_right.is_cuda
        and add_left.is_cuda
        and add_right.is_cuda
        and state.dtype == torch.float32
    )
    if not use_triton:
        return dplr_compact_wy_prefix_combine_torch(state, summary)

    state_c = state.contiguous()
    diag_c = diag.contiguous()
    trans_left_c = trans_left.contiguous()
    trans_right_c = trans_right.contiguous()
    add_left_c = add_left.contiguous()
    add_right_c = add_right.contiguous()
    start_states = torch.empty((B, chunks, H, N, N), device=state.device, dtype=torch.float32)
    final_state = torch.empty_like(state_c)
    row_blocks = triton.cdiv(N, block_m)
    _compact_wy_prefix_combine_kernel[(B * H * row_blocks,)](
        state_c,
        diag_c,
        trans_left_c,
        trans_right_c,
        add_left_c,
        add_right_c,
        start_states,
        final_state,
        H,
        N,
        chunks,
        R,
        ROW_BLOCKS=int(row_blocks),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_R=R,
        num_warps=4 if block_m < N else 8,
    )
    return start_states, final_state


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


def dplr_dense_prefix_combine_torch(state: Any, transition: Any, additive: Any):
    """Reference chunk-prefix combine over dense summaries.

    Returns `(start_states, final_state)` where `start_states[:, c]` is the
    native VxK state before chunk `c`.
    """

    if torch is None:
        raise RuntimeError("dplr_dense_prefix_combine_torch requires torch")
    if state.dim() != 4 or transition.dim() != 5 or additive.dim() != 5:
        raise ValueError("state must be [B,H,N,N] and summaries [B,chunks,H,N,N]")
    B, H, N, N2 = (int(vv) for vv in state.shape)
    if N != N2:
        raise ValueError("state must be square in the last two dimensions")
    if tuple(transition.shape) != tuple(additive.shape):
        raise ValueError("transition and additive shapes must match")
    if int(transition.shape[0]) != B or int(transition.shape[2]) != H or int(transition.shape[3]) != N or int(transition.shape[4]) != N:
        raise ValueError("summary shapes must be [B,chunks,H,N,N] matching state")

    chunks = int(transition.shape[1])
    cur = state.float()
    starts = []
    for chunk in range(chunks):
        starts.append(cur)
        cur = cur @ transition[:, chunk].float() + additive[:, chunk].float()
    return torch.stack(starts, dim=1), cur


def dplr_dense_prefix_combine_triton(
    state: Any,
    transition: Any,
    additive: Any,
    *,
    block_m: int | None = None,
    block_n: int | None = None,
    force_fallback: bool = False,
):
    """Triton chunk-prefix combine for dense summaries."""

    if torch is None:
        raise RuntimeError("dplr_dense_prefix_combine_triton requires torch")
    if state.dim() != 4 or transition.dim() != 5 or additive.dim() != 5:
        raise ValueError("state must be [B,H,N,N] and summaries [B,chunks,H,N,N]")
    B, H, N, N2 = (int(vv) for vv in state.shape)
    if N != N2:
        raise ValueError("state must be square in the last two dimensions")
    if tuple(transition.shape) != tuple(additive.shape):
        raise ValueError("transition and additive shapes must match")
    chunks = int(transition.shape[1])
    if int(transition.shape[0]) != B or int(transition.shape[2]) != H or int(transition.shape[3]) != N or int(transition.shape[4]) != N:
        raise ValueError("summary shapes must be [B,chunks,H,N,N] matching state")
    if block_m is None:
        block_m = _env_int("RWKV7_DPLR_TRITON_PREFIX_BLOCK_M", 8)
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
        and state.is_cuda
        and transition.is_cuda
        and additive.is_cuda
        and state.dtype == torch.float32
    )
    if not use_triton:
        return dplr_dense_prefix_combine_torch(state, transition, additive)

    state_c = state.contiguous()
    transition_c = transition.contiguous()
    additive_c = additive.contiguous()
    start_states = torch.empty((B, chunks, H, N, N), device=state.device, dtype=torch.float32)
    final_state = torch.empty_like(state_c)
    row_blocks = triton.cdiv(N, block_m)
    _dense_prefix_combine_kernel[(B * H * row_blocks,)](
        state_c,
        transition_c,
        additive_c,
        start_states,
        final_state,
        H,
        N,
        chunks,
        ROW_BLOCKS=int(row_blocks),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=4 if block_m < N else 8,
    )
    return start_states, final_state


def _torch_dplr_step_bthn(r_t: Any, w_t: Any, k_t: Any, v_t: Any, kk_t: Any, a_t: Any, state_t: Any):
    B, H, N = (int(r_t.shape[0]), int(r_t.shape[1]), int(r_t.shape[2]))
    vk = v_t.float().view(B, H, N, 1) @ k_t.float().view(B, H, 1, N)
    ab = (-kk_t.float()).view(B, H, N, 1) @ (kk_t.float() * a_t.float()).view(B, H, 1, N)
    new_state = state_t.float() * w_t.float().view(B, H, 1, N) + state_t.float() @ ab + vk
    out = new_state @ r_t.float().view(B, H, N, 1)
    return out.view(B, H, N), new_state


def dplr_dense_chunk_apply_torch(r: Any, w: Any, k: Any, v: Any, kk: Any, a: Any, start_states: Any, *, chunk_size: int = 64):
    """Reference chunk-apply stage from prefix-combined start states."""

    if torch is None:
        raise RuntimeError("dplr_dense_chunk_apply_torch requires torch")
    chunk_size_i = _validate_chunk_size(chunk_size)
    if start_states.dim() != 5:
        raise ValueError("start_states must be [B,chunks,H,N,N]")
    B, chunks, H, N, N2 = (int(vv) for vv in start_states.shape)
    if N != N2:
        raise ValueError("start_states must be square in the last two dimensions")
    r4, flat = _as_bthn(r, H, N, name="r")
    w4, _ = _as_bthn(w, H, N, name="w")
    k4, _ = _as_bthn(k, H, N, name="k")
    v4, _ = _as_bthn(v, H, N, name="v")
    kk4, _ = _as_bthn(kk, H, N, name="kk")
    a4, _ = _as_bthn(a, H, N, name="a")
    T = int(r4.shape[1])
    if T != chunks * chunk_size_i:
        raise ValueError("tokens must equal chunks * chunk_size")

    outs = []
    chunk_ends = []
    for chunk in range(chunks):
        cur = start_states[:, chunk].float()
        for local_i in range(chunk_size_i):
            t = chunk * chunk_size_i + local_i
            out_t, cur = _torch_dplr_step_bthn(r4[:, t], w4[:, t], k4[:, t], v4[:, t], kk4[:, t], a4[:, t], cur)
            outs.append(out_t.to(dtype=r4.dtype))
        chunk_ends.append(cur)
    out = torch.stack(outs, dim=1) if outs else r4.new_empty((B, 0, H, N))
    if flat:
        out = out.reshape(B, T, H * N)
    return out, torch.stack(chunk_ends, dim=1)


def dplr_dense_chunk_apply_triton(
    r: Any,
    w: Any,
    k: Any,
    v: Any,
    kk: Any,
    a: Any,
    start_states: Any,
    *,
    chunk_size: int = 64,
    block_m: int | None = None,
    block_n: int | None = None,
    force_fallback: bool = False,
):
    """Triton chunk-apply stage for prefix-combined dense summaries."""

    if torch is None:
        raise RuntimeError("dplr_dense_chunk_apply_triton requires torch")
    chunk_size_i = _validate_chunk_size(chunk_size)
    if start_states.dim() != 5:
        raise ValueError("start_states must be [B,chunks,H,N,N]")
    B, chunks, H, N, N2 = (int(vv) for vv in start_states.shape)
    if N != N2:
        raise ValueError("start_states must be square in the last two dimensions")
    r4, flat = _as_bthn(r, H, N, name="r")
    w4, _ = _as_bthn(w, H, N, name="w")
    k4, _ = _as_bthn(k, H, N, name="k")
    v4, _ = _as_bthn(v, H, N, name="v")
    kk4, _ = _as_bthn(kk, H, N, name="kk")
    a4, _ = _as_bthn(a, H, N, name="a")
    T = int(r4.shape[1])
    if T != chunks * chunk_size_i:
        raise ValueError("tokens must equal chunks * chunk_size")
    if block_m is None:
        block_m = _env_int("RWKV7_DPLR_TRITON_APPLY_BLOCK_M", 8)
    if block_n is None:
        block_n = N
    block_m = int(block_m)
    block_n = int(block_n)
    if block_m <= 0:
        raise ValueError("block_m must be positive")
    if block_n < N:
        raise ValueError(f"block_n must be >= head_dim={N}; got {block_n}")

    k_kernel = k4 if k4.dtype == r4.dtype else k4.to(dtype=r4.dtype)
    v_kernel = v4 if v4.dtype == r4.dtype else v4.to(dtype=r4.dtype)
    kk_kernel = kk4 if kk4.dtype == r4.dtype else kk4.to(dtype=r4.dtype)
    a_kernel = a4 if a4.dtype == r4.dtype else a4.to(dtype=r4.dtype)
    use_triton = (
        not force_fallback
        and dplr_dense_chunk_summary_triton_available()
        and r4.is_cuda
        and w4.is_cuda
        and k_kernel.is_cuda
        and v_kernel.is_cuda
        and kk_kernel.is_cuda
        and a_kernel.is_cuda
        and start_states.is_cuda
        and start_states.dtype == torch.float32
        and r4.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and w4.dtype in (r4.dtype, torch.float32)
    )
    if not use_triton:
        return dplr_dense_chunk_apply_torch(r4.reshape(B, T, H * N) if flat else r4, w4, k4, v4, kk4, a4, start_states, chunk_size=chunk_size_i)

    r_c = r4.contiguous()
    w_c = w4.contiguous()
    k_c = k_kernel.contiguous()
    v_c = v_kernel.contiguous()
    kk_c = kk_kernel.contiguous()
    a_c = a_kernel.contiguous()
    start_c = start_states.contiguous()
    out = torch.empty((B, T, H, N), device=r4.device, dtype=r4.dtype)
    chunk_ends = torch.empty_like(start_c)
    row_blocks = triton.cdiv(N, block_m)
    _dense_chunk_apply_kernel[(B * chunks * H * row_blocks,)](
        r_c,
        w_c,
        k_c,
        v_c,
        kk_c,
        a_c,
        start_c,
        out,
        chunk_ends,
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
    if flat:
        out = out.reshape(B, T, H * N)
    return out, chunk_ends


def dplr_dense_three_stage_triton(
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
    """Dense three-stage DPLR scaffold: summary -> prefix -> chunk apply."""

    if torch is None:
        raise RuntimeError("dplr_dense_three_stage_triton requires torch")
    if state.dim() != 4:
        raise ValueError("state must be [B,H,N,N]")
    B, H, N, _ = (int(vv) for vv in state.shape)
    r4, flat = _as_bthn(r, H, N, name="r")
    w4, _ = _as_bthn(w, H, N, name="w")
    k4, _ = _as_bthn(k, H, N, name="k")
    v4, _ = _as_bthn(v, H, N, name="v")
    kk4, _ = _as_bthn(kk, H, N, name="kk")
    a4, _ = _as_bthn(a, H, N, name="a")
    summary = dplr_dense_chunk_summary_triton(w4, k4, v4, kk4, a4, chunk_size=chunk_size, force_fallback=force_fallback)
    start_states, prefix_final = dplr_dense_prefix_combine_triton(
        state,
        summary["transition"],
        summary["additive"],
        force_fallback=force_fallback,
    )
    out, chunk_ends = dplr_dense_chunk_apply_triton(
        r4.reshape(B, int(r4.shape[1]), H * N) if flat else r4,
        w4,
        k4,
        v4,
        kk4,
        a4,
        start_states,
        chunk_size=chunk_size,
        force_fallback=force_fallback,
    )
    final_from_apply = chunk_ends[:, -1].to(dtype=state.dtype)
    # Prefer the chunk-apply final state because it follows the same token path
    # as the emitted recurrent outputs; prefix_final is kept as a consistency
    # check by callers/tests.
    _ = prefix_final
    return out, final_from_apply


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
