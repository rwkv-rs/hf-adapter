# coding=utf-8
"""Optional fused recurrent state-update prototypes for RWKV-7 decode/prefill.

This module prototypes the most promising fp16 fusion target after projection
and shift-mix microbenchmarks: the one-token RWKV-7 recurrent update itself.
For each batch/head row it fuses the rank-1 state update and readout:

    ab = (-kk)[:, None] @ (kk * a)[None, :]
    new_state = state * w[None, :] + state @ ab + v[:, None] @ k[None, :]
    out = new_state @ r[:, None]

Using the rank-1 structure, ``state @ ab`` is computed without materializing
``ab``.  The implementation is optional: imports must work on CPU-only hosts
and fallback to the torch reference when Triton/CUDA is unavailable.

The scan prototype extends the same rank-1 update across a prompt chunk.  It is
not wired into the HF forward path yet; it is a prefill-kernel development
target that can be benchmarked against FLA's ``chunk_rwkv7`` recurrent scan.
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

    @triton.jit
    def _recurrent_output_prepare_kernel(
        r_ptr,
        w_ptr,
        k_ptr,
        v_ptr,
        kk_ptr,
        a_ptr,
        state_ptr,
        g_ptr,
        rk_ptr,
        gn_weight_ptr,
        gn_bias_ptr,
        out_ptr,
        new_state_ptr,
        H: tl.constexpr,
        N: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        bh_id = tl.program_id(0)
        head_id = bh_id % H
        offs_i = tl.arange(0, BLOCK_N)
        offs_j = tl.arange(0, BLOCK_N)
        mask_i = offs_i < N
        mask_j = offs_j < N
        vec_base = bh_id * N
        state_base = bh_id * N * N

        st = tl.load(
            state_ptr + state_base + offs_i[:, None] * N + offs_j[None, :],
            mask=mask_i[:, None] & mask_j[None, :],
            other=0.0,
        ).to(tl.float32)
        r = tl.load(r_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
        k = tl.load(k_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
        kk = tl.load(kk_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
        a = tl.load(a_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
        v_cols = tl.load(v_ptr + vec_base + offs_i, mask=mask_i, other=0.0).to(tl.float32)

        # Native formula using the rank-1 structure:
        # state @ ((-kk)[:, None] @ (kk*a)[None, :])
        # = -dot(state[i, :], kk) * kk[j] * a[j].
        state_dot_kk = tl.sum(st * kk[None, :], axis=1)
        new_st = st * w[None, :] + v_cols[:, None] * k[None, :] - state_dot_kk[:, None] * kk[None, :] * a[None, :]
        tl.store(
            new_state_ptr + state_base + offs_i[:, None] * N + offs_j[None, :],
            new_st,
            mask=mask_i[:, None] & mask_j[None, :],
        )

        recurrent = tl.sum(new_st * r[None, :], axis=1)
        mean = tl.sum(tl.where(mask_i, recurrent, 0.0), axis=0) / N
        centered = tl.where(mask_i, recurrent - mean, 0.0)
        var = tl.sum(centered * centered, axis=0) / N
        normed = centered * tl.rsqrt(var + eps)

        r_rows = tl.load(r_ptr + vec_base + offs_i, mask=mask_i, other=0.0).to(tl.float32)
        k_rows = tl.load(k_ptr + vec_base + offs_i, mask=mask_i, other=0.0).to(tl.float32)
        rk = tl.load(rk_ptr + head_id * N + offs_i, mask=mask_i, other=0.0).to(tl.float32)
        corr_scale = tl.sum(r_rows * k_rows * rk, axis=0)
        gate = tl.load(g_ptr + vec_base + offs_i, mask=mask_i, other=0.0).to(tl.float32)
        weight = tl.load(gn_weight_ptr + head_id * N + offs_i, mask=mask_i, other=1.0).to(tl.float32)
        bias = tl.load(gn_bias_ptr + head_id * N + offs_i, mask=mask_i, other=0.0).to(tl.float32)
        prepared = (normed * weight + bias + corr_scale * v_cols) * gate
        tl.store(out_ptr + vec_base + offs_i, prepared, mask=mask_i)

    @triton.jit
    def _recurrent_scan_kernel(
        r_ptr,
        w_ptr,
        k_ptr,
        v_ptr,
        kk_ptr,
        a_ptr,
        state_ptr,
        out_ptr,
        final_state_ptr,
        T,
        H: tl.constexpr,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        bh_id = tl.program_id(0)
        head_id = bh_id % H
        batch_id = bh_id // H

        offs_i = tl.arange(0, BLOCK_N)
        offs_j = tl.arange(0, BLOCK_N)
        mask_i = offs_i < N
        mask_j = offs_j < N
        state_base = (batch_id * H + head_id) * N * N
        st = tl.load(
            state_ptr + state_base + offs_i[:, None] * N + offs_j[None, :],
            mask=mask_i[:, None] & mask_j[None, :],
            other=0.0,
        ).to(tl.float32)

        t = 0
        while t < T:
            vec_base = ((batch_id * T + t) * H + head_id) * N
            r = tl.load(r_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            k = tl.load(k_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            kk = tl.load(kk_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            a = tl.load(a_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            v_cols = tl.load(v_ptr + vec_base + offs_i, mask=mask_i, other=0.0).to(tl.float32)

            state_dot_kk = tl.sum(st * kk[None, :], axis=1)
            st = st * w[None, :] + v_cols[:, None] * k[None, :] - state_dot_kk[:, None] * kk[None, :] * a[None, :]

            recurrent = tl.sum(st * r[None, :], axis=1)
            tl.store(out_ptr + vec_base + offs_i, recurrent, mask=mask_i)
            t += 1

        tl.store(
            final_state_ptr + state_base + offs_i[:, None] * N + offs_j[None, :],
            st,
            mask=mask_i[:, None] & mask_j[None, :],
        )

    @triton.jit
    def _recurrent_scan_state_prep_kernel(
        r_ptr,
        w_raw_ptr,
        k_raw_ptr,
        v_raw_ptr,
        a_ptr,
        state_ptr,
        k_k_ptr,
        k_a_ptr,
        v_first_ptr,
        v_gate_ptr,
        out_ptr,
        final_state_ptr,
        k_out_ptr,
        v_out_ptr,
        T,
        H: tl.constexpr,
        N: tl.constexpr,
        HAS_V_GATE: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        bh_id = tl.program_id(0)
        head_id = bh_id % H
        batch_id = bh_id // H

        offs = tl.arange(0, BLOCK_N)
        mask = offs < N
        state_base = (batch_id * H + head_id) * N * N
        param_base = head_id * N
        st = tl.load(
            state_ptr + state_base + offs[:, None] * N + offs[None, :],
            mask=mask[:, None] & mask[None, :],
            other=0.0,
        ).to(tl.float32)
        kk_scale = tl.load(k_k_ptr + param_base + offs, mask=mask, other=0.0).to(tl.float32)
        ka_scale = tl.load(k_a_ptr + param_base + offs, mask=mask, other=0.0).to(tl.float32)

        t = 0
        while t < T:
            vec_base = ((batch_id * T + t) * H + head_id) * N
            r = tl.load(r_ptr + vec_base + offs, mask=mask, other=0.0).to(tl.float32)
            w_raw = tl.load(w_raw_ptr + vec_base + offs, mask=mask, other=0.0).to(tl.float32)
            k_raw = tl.load(k_raw_ptr + vec_base + offs, mask=mask, other=0.0).to(tl.float32)
            v_raw = tl.load(v_raw_ptr + vec_base + offs, mask=mask, other=0.0).to(tl.float32)
            a_val = tl.load(a_ptr + vec_base + offs, mask=mask, other=0.0).to(tl.float32)

            kk_raw = k_raw * kk_scale
            norm2 = tl.sum(tl.where(mask, kk_raw * kk_raw, 0.0), axis=0)
            kk = kk_raw * tl.rsqrt(tl.maximum(norm2, 1.0e-20))
            k_adj = k_raw * (1.0 + (a_val - 1.0) * ka_scale)
            v_adj = v_raw
            if HAS_V_GATE:
                vf = tl.load(v_first_ptr + vec_base + offs, mask=mask, other=0.0).to(tl.float32)
                vg = tl.load(v_gate_ptr + vec_base + offs, mask=mask, other=0.0).to(tl.float32)
                v_adj = v_raw + (vf - v_raw) * vg
            w = tl.exp(-0.606531 * tl.sigmoid(w_raw))

            state_dot_kk = tl.sum(st * kk[None, :], axis=1)
            st = st * w[None, :] + v_adj[:, None] * k_adj[None, :] - state_dot_kk[:, None] * kk[None, :] * a_val[None, :]

            recurrent = tl.sum(st * r[None, :], axis=1)
            tl.store(out_ptr + vec_base + offs, recurrent, mask=mask)
            tl.store(k_out_ptr + vec_base + offs, k_adj, mask=mask)
            tl.store(v_out_ptr + vec_base + offs, v_adj, mask=mask)
            t += 1

        tl.store(
            final_state_ptr + state_base + offs[:, None] * N + offs[None, :],
            st,
            mask=mask[:, None] & mask[None, :],
        )

    @triton.jit
    def _recurrent_scan_state_prep_correction_kernel(
        r_ptr,
        w_raw_ptr,
        k_raw_ptr,
        v_raw_ptr,
        a_ptr,
        state_ptr,
        k_k_ptr,
        k_a_ptr,
        r_k_ptr,
        v_first_ptr,
        v_gate_ptr,
        out_ptr,
        final_state_ptr,
        correction_ptr,
        T,
        H: tl.constexpr,
        N: tl.constexpr,
        HAS_V_GATE: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        bh_id = tl.program_id(0)
        head_id = bh_id % H
        batch_id = bh_id // H

        offs = tl.arange(0, BLOCK_N)
        mask = offs < N
        state_base = (batch_id * H + head_id) * N * N
        param_base = head_id * N
        st = tl.load(
            state_ptr + state_base + offs[:, None] * N + offs[None, :],
            mask=mask[:, None] & mask[None, :],
            other=0.0,
        ).to(tl.float32)
        kk_scale = tl.load(k_k_ptr + param_base + offs, mask=mask, other=0.0).to(tl.float32)
        ka_scale = tl.load(k_a_ptr + param_base + offs, mask=mask, other=0.0).to(tl.float32)
        rk = tl.load(r_k_ptr + param_base + offs, mask=mask, other=0.0).to(tl.float32)

        t = 0
        while t < T:
            vec_base = ((batch_id * T + t) * H + head_id) * N
            r = tl.load(r_ptr + vec_base + offs, mask=mask, other=0.0).to(tl.float32)
            w_raw = tl.load(w_raw_ptr + vec_base + offs, mask=mask, other=0.0).to(tl.float32)
            k_raw = tl.load(k_raw_ptr + vec_base + offs, mask=mask, other=0.0).to(tl.float32)
            v_raw = tl.load(v_raw_ptr + vec_base + offs, mask=mask, other=0.0).to(tl.float32)
            a_val = tl.load(a_ptr + vec_base + offs, mask=mask, other=0.0).to(tl.float32)

            kk_raw = k_raw * kk_scale
            norm2 = tl.sum(tl.where(mask, kk_raw * kk_raw, 0.0), axis=0)
            kk = kk_raw * tl.rsqrt(tl.maximum(norm2, 1.0e-20))
            k_adj = k_raw * (1.0 + (a_val - 1.0) * ka_scale)
            v_adj = v_raw
            if HAS_V_GATE:
                vf = tl.load(v_first_ptr + vec_base + offs, mask=mask, other=0.0).to(tl.float32)
                vg = tl.load(v_gate_ptr + vec_base + offs, mask=mask, other=0.0).to(tl.float32)
                v_adj = v_raw + (vf - v_raw) * vg
            w = tl.exp(-0.606531 * tl.sigmoid(w_raw))

            state_dot_kk = tl.sum(st * kk[None, :], axis=1)
            st = st * w[None, :] + v_adj[:, None] * k_adj[None, :] - state_dot_kk[:, None] * kk[None, :] * a_val[None, :]

            recurrent = tl.sum(st * r[None, :], axis=1)
            corr_scale = tl.sum(tl.where(mask, r * k_adj * rk, 0.0), axis=0)
            tl.store(out_ptr + vec_base + offs, recurrent, mask=mask)
            tl.store(correction_ptr + vec_base + offs, corr_scale * v_adj, mask=mask)
            t += 1

        tl.store(
            final_state_ptr + state_base + offs[:, None] * N + offs[None, :],
            st,
            mask=mask[:, None] & mask[None, :],
        )

    @triton.jit
    def _recurrent_scan_state_prep_rows_kernel(
        r_ptr,
        w_raw_ptr,
        k_raw_ptr,
        v_raw_ptr,
        a_ptr,
        state_ptr,
        k_k_ptr,
        k_a_ptr,
        v_first_ptr,
        v_gate_ptr,
        out_ptr,
        final_state_ptr,
        k_out_ptr,
        v_out_ptr,
        T,
        H: tl.constexpr,
        N: tl.constexpr,
        ROW_BLOCKS: tl.constexpr,
        HAS_V_GATE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        row_block = pid % ROW_BLOCKS
        bh_id = pid // ROW_BLOCKS
        head_id = bh_id % H
        batch_id = bh_id // H

        offs_i = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_j = tl.arange(0, BLOCK_N)
        mask_i = offs_i < N
        mask_j = offs_j < N
        state_base = (batch_id * H + head_id) * N * N
        param_base = head_id * N
        st = tl.load(
            state_ptr + state_base + offs_i[:, None] * N + offs_j[None, :],
            mask=mask_i[:, None] & mask_j[None, :],
            other=0.0,
        ).to(tl.float32)
        kk_scale_j = tl.load(k_k_ptr + param_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
        ka_scale_j = tl.load(k_a_ptr + param_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
        ka_scale_i = tl.load(k_a_ptr + param_base + offs_i, mask=mask_i, other=0.0).to(tl.float32)

        t = 0
        while t < T:
            vec_base = ((batch_id * T + t) * H + head_id) * N
            r = tl.load(r_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            w_raw = tl.load(w_raw_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            k_raw_j = tl.load(k_raw_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            k_raw_i = tl.load(k_raw_ptr + vec_base + offs_i, mask=mask_i, other=0.0).to(tl.float32)
            v_raw_i = tl.load(v_raw_ptr + vec_base + offs_i, mask=mask_i, other=0.0).to(tl.float32)
            a_j = tl.load(a_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            a_i = tl.load(a_ptr + vec_base + offs_i, mask=mask_i, other=0.0).to(tl.float32)

            kk_raw = k_raw_j * kk_scale_j
            norm2 = tl.sum(tl.where(mask_j, kk_raw * kk_raw, 0.0), axis=0)
            kk = kk_raw * tl.rsqrt(tl.maximum(norm2, 1.0e-20))
            k_adj_j = k_raw_j * (1.0 + (a_j - 1.0) * ka_scale_j)
            k_adj_i = k_raw_i * (1.0 + (a_i - 1.0) * ka_scale_i)
            v_adj_i = v_raw_i
            if HAS_V_GATE:
                vf_i = tl.load(v_first_ptr + vec_base + offs_i, mask=mask_i, other=0.0).to(tl.float32)
                vg_i = tl.load(v_gate_ptr + vec_base + offs_i, mask=mask_i, other=0.0).to(tl.float32)
                v_adj_i = v_raw_i + (vf_i - v_raw_i) * vg_i
            w = tl.exp(-0.606531 * tl.sigmoid(w_raw))

            state_dot_kk = tl.sum(st * kk[None, :], axis=1)
            st = st * w[None, :] + v_adj_i[:, None] * k_adj_j[None, :] - state_dot_kk[:, None] * kk[None, :] * a_j[None, :]

            recurrent = tl.sum(st * r[None, :], axis=1)
            tl.store(out_ptr + vec_base + offs_i, recurrent, mask=mask_i)
            tl.store(k_out_ptr + vec_base + offs_i, k_adj_i, mask=mask_i)
            tl.store(v_out_ptr + vec_base + offs_i, v_adj_i, mask=mask_i)
            t += 1

        tl.store(
            final_state_ptr + state_base + offs_i[:, None] * N + offs_j[None, :],
            st,
            mask=mask_i[:, None] & mask_j[None, :],
        )

    @triton.jit
    def _recurrent_scan_state_prep_output_prepare_kernel(
        r_ptr,
        w_raw_ptr,
        k_raw_ptr,
        v_raw_ptr,
        a_ptr,
        state_ptr,
        k_k_ptr,
        k_a_ptr,
        v_first_ptr,
        v_gate_ptr,
        g_ptr,
        rk_ptr,
        gn_weight_ptr,
        gn_bias_ptr,
        out_ptr,
        final_state_ptr,
        T,
        H: tl.constexpr,
        N: tl.constexpr,
        HAS_V_GATE: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        bh_id = tl.program_id(0)
        head_id = bh_id % H
        batch_id = bh_id // H

        offs = tl.arange(0, BLOCK_N)
        mask = offs < N
        state_base = (batch_id * H + head_id) * N * N
        param_base = head_id * N
        st = tl.load(
            state_ptr + state_base + offs[:, None] * N + offs[None, :],
            mask=mask[:, None] & mask[None, :],
            other=0.0,
        ).to(tl.float32)
        kk_scale = tl.load(k_k_ptr + param_base + offs, mask=mask, other=0.0).to(tl.float32)
        ka_scale = tl.load(k_a_ptr + param_base + offs, mask=mask, other=0.0).to(tl.float32)
        rk = tl.load(rk_ptr + param_base + offs, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(gn_weight_ptr + param_base + offs, mask=mask, other=1.0).to(tl.float32)
        bias = tl.load(gn_bias_ptr + param_base + offs, mask=mask, other=0.0).to(tl.float32)

        t = 0
        while t < T:
            vec_base = ((batch_id * T + t) * H + head_id) * N
            r = tl.load(r_ptr + vec_base + offs, mask=mask, other=0.0).to(tl.float32)
            w_raw = tl.load(w_raw_ptr + vec_base + offs, mask=mask, other=0.0).to(tl.float32)
            k_raw = tl.load(k_raw_ptr + vec_base + offs, mask=mask, other=0.0).to(tl.float32)
            v_raw = tl.load(v_raw_ptr + vec_base + offs, mask=mask, other=0.0).to(tl.float32)
            a_val = tl.load(a_ptr + vec_base + offs, mask=mask, other=0.0).to(tl.float32)
            g_val = tl.load(g_ptr + vec_base + offs, mask=mask, other=0.0).to(tl.float32)

            kk_raw = k_raw * kk_scale
            norm2 = tl.sum(tl.where(mask, kk_raw * kk_raw, 0.0), axis=0)
            kk = kk_raw * tl.rsqrt(tl.maximum(norm2, 1.0e-20))
            k_adj = k_raw * (1.0 + (a_val - 1.0) * ka_scale)
            v_adj = v_raw
            if HAS_V_GATE:
                vf = tl.load(v_first_ptr + vec_base + offs, mask=mask, other=0.0).to(tl.float32)
                vg = tl.load(v_gate_ptr + vec_base + offs, mask=mask, other=0.0).to(tl.float32)
                v_adj = v_raw + (vf - v_raw) * vg
            w = tl.exp(-0.606531 * tl.sigmoid(w_raw))

            state_dot_kk = tl.sum(st * kk[None, :], axis=1)
            st = st * w[None, :] + v_adj[:, None] * k_adj[None, :] - state_dot_kk[:, None] * kk[None, :] * a_val[None, :]

            recurrent = tl.sum(st * r[None, :], axis=1)
            mean = tl.sum(tl.where(mask, recurrent, 0.0), axis=0) / N
            centered = tl.where(mask, recurrent - mean, 0.0)
            var = tl.sum(centered * centered, axis=0) / N
            normed = centered * tl.rsqrt(var + eps)
            corr_scale = tl.sum(r * k_adj * rk, axis=0)
            prepared = (normed * weight + bias + corr_scale * v_adj) * g_val
            tl.store(out_ptr + vec_base + offs, prepared, mask=mask)
            t += 1

        tl.store(
            final_state_ptr + state_base + offs[:, None] * N + offs[None, :],
            st,
            mask=mask[:, None] & mask[None, :],
        )

    @triton.jit
    def _recurrent_scan_rows_kernel(
        r_ptr,
        w_ptr,
        k_ptr,
        v_ptr,
        kk_ptr,
        a_ptr,
        state_ptr,
        out_ptr,
        final_state_ptr,
        T,
        H: tl.constexpr,
        N: tl.constexpr,
        ROW_BLOCKS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        row_block = pid % ROW_BLOCKS
        bh_id = pid // ROW_BLOCKS
        head_id = bh_id % H
        batch_id = bh_id // H

        offs_i = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_j = tl.arange(0, BLOCK_N)
        mask_i = offs_i < N
        mask_j = offs_j < N
        state_base = (batch_id * H + head_id) * N * N
        st = tl.load(
            state_ptr + state_base + offs_i[:, None] * N + offs_j[None, :],
            mask=mask_i[:, None] & mask_j[None, :],
            other=0.0,
        ).to(tl.float32)

        t = 0
        while t < T:
            vec_base = ((batch_id * T + t) * H + head_id) * N
            r = tl.load(r_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            k = tl.load(k_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            kk = tl.load(kk_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            a = tl.load(a_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            v_rows = tl.load(v_ptr + vec_base + offs_i, mask=mask_i, other=0.0).to(tl.float32)

            state_dot_kk = tl.sum(st * kk[None, :], axis=1)
            st = st * w[None, :] + v_rows[:, None] * k[None, :] - state_dot_kk[:, None] * kk[None, :] * a[None, :]

            recurrent = tl.sum(st * r[None, :], axis=1)
            tl.store(out_ptr + vec_base + offs_i, recurrent, mask=mask_i)
            t += 1

        tl.store(
            final_state_ptr + state_base + offs_i[:, None] * N + offs_j[None, :],
            st,
            mask=mask_i[:, None] & mask_j[None, :],
        )

    @triton.jit
    def _recurrent_scan_clampw_kernel(
        r_ptr,
        w_raw_ptr,
        k_ptr,
        v_ptr,
        kk_ptr,
        a_ptr,
        state_ptr,
        out_ptr,
        final_state_ptr,
        T,
        H: tl.constexpr,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        bh_id = tl.program_id(0)
        head_id = bh_id % H
        batch_id = bh_id // H

        offs_i = tl.arange(0, BLOCK_N)
        offs_j = tl.arange(0, BLOCK_N)
        mask_i = offs_i < N
        mask_j = offs_j < N
        state_base = (batch_id * H + head_id) * N * N
        st = tl.load(
            state_ptr + state_base + offs_i[:, None] * N + offs_j[None, :],
            mask=mask_i[:, None] & mask_j[None, :],
            other=0.0,
        ).to(tl.float32)

        t = 0
        while t < T:
            vec_base = ((batch_id * T + t) * H + head_id) * N
            r = tl.load(r_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            w_raw = tl.load(w_raw_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            w = tl.exp(-0.606531 * tl.sigmoid(w_raw))
            k = tl.load(k_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            kk = tl.load(kk_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            a = tl.load(a_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            v_cols = tl.load(v_ptr + vec_base + offs_i, mask=mask_i, other=0.0).to(tl.float32)

            state_dot_kk = tl.sum(st * kk[None, :], axis=1)
            st = st * w[None, :] + v_cols[:, None] * k[None, :] - state_dot_kk[:, None] * kk[None, :] * a[None, :]

            recurrent = tl.sum(st * r[None, :], axis=1)
            tl.store(out_ptr + vec_base + offs_i, recurrent, mask=mask_i)
            t += 1

        tl.store(
            final_state_ptr + state_base + offs_i[:, None] * N + offs_j[None, :],
            st,
            mask=mask_i[:, None] & mask_j[None, :],
        )

    @triton.jit
    def _recurrent_scan_rows_clampw_kernel(
        r_ptr,
        w_raw_ptr,
        k_ptr,
        v_ptr,
        kk_ptr,
        a_ptr,
        state_ptr,
        out_ptr,
        final_state_ptr,
        T,
        H: tl.constexpr,
        N: tl.constexpr,
        ROW_BLOCKS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        row_block = pid % ROW_BLOCKS
        bh_id = pid // ROW_BLOCKS
        head_id = bh_id % H
        batch_id = bh_id // H

        offs_i = row_block * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_j = tl.arange(0, BLOCK_N)
        mask_i = offs_i < N
        mask_j = offs_j < N
        state_base = (batch_id * H + head_id) * N * N
        st = tl.load(
            state_ptr + state_base + offs_i[:, None] * N + offs_j[None, :],
            mask=mask_i[:, None] & mask_j[None, :],
            other=0.0,
        ).to(tl.float32)

        t = 0
        while t < T:
            vec_base = ((batch_id * T + t) * H + head_id) * N
            r = tl.load(r_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            w_raw = tl.load(w_raw_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            w = tl.exp(-0.606531 * tl.sigmoid(w_raw))
            k = tl.load(k_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            kk = tl.load(kk_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            a = tl.load(a_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            v_rows = tl.load(v_ptr + vec_base + offs_i, mask=mask_i, other=0.0).to(tl.float32)

            state_dot_kk = tl.sum(st * kk[None, :], axis=1)
            st = st * w[None, :] + v_rows[:, None] * k[None, :] - state_dot_kk[:, None] * kk[None, :] * a[None, :]

            recurrent = tl.sum(st * r[None, :], axis=1)
            tl.store(out_ptr + vec_base + offs_i, recurrent, mask=mask_i)
            t += 1

        tl.store(
            final_state_ptr + state_base + offs_i[:, None] * N + offs_j[None, :],
            st,
            mask=mask_i[:, None] & mask_j[None, :],
        )

    @triton.jit
    def _recurrent_scan_output_prepare_kernel(
        r_ptr,
        w_ptr,
        k_ptr,
        v_ptr,
        kk_ptr,
        a_ptr,
        state_ptr,
        g_ptr,
        rk_ptr,
        gn_weight_ptr,
        gn_bias_ptr,
        out_ptr,
        final_state_ptr,
        T,
        H: tl.constexpr,
        N: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        bh_id = tl.program_id(0)
        head_id = bh_id % H
        batch_id = bh_id // H

        offs_i = tl.arange(0, BLOCK_N)
        offs_j = tl.arange(0, BLOCK_N)
        mask_i = offs_i < N
        mask_j = offs_j < N
        state_base = (batch_id * H + head_id) * N * N
        st = tl.load(
            state_ptr + state_base + offs_i[:, None] * N + offs_j[None, :],
            mask=mask_i[:, None] & mask_j[None, :],
            other=0.0,
        ).to(tl.float32)

        t = 0
        while t < T:
            vec_base = ((batch_id * T + t) * H + head_id) * N
            r = tl.load(r_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            k = tl.load(k_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            kk = tl.load(kk_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            a = tl.load(a_ptr + vec_base + offs_j, mask=mask_j, other=0.0).to(tl.float32)
            v_cols = tl.load(v_ptr + vec_base + offs_i, mask=mask_i, other=0.0).to(tl.float32)

            state_dot_kk = tl.sum(st * kk[None, :], axis=1)
            st = st * w[None, :] + v_cols[:, None] * k[None, :] - state_dot_kk[:, None] * kk[None, :] * a[None, :]

            recurrent = tl.sum(st * r[None, :], axis=1)
            mean = tl.sum(tl.where(mask_i, recurrent, 0.0), axis=0) / N
            centered = tl.where(mask_i, recurrent - mean, 0.0)
            var = tl.sum(centered * centered, axis=0) / N
            normed = centered * tl.rsqrt(var + eps)

            r_rows = tl.load(r_ptr + vec_base + offs_i, mask=mask_i, other=0.0).to(tl.float32)
            k_rows = tl.load(k_ptr + vec_base + offs_i, mask=mask_i, other=0.0).to(tl.float32)
            g_rows = tl.load(g_ptr + vec_base + offs_i, mask=mask_i, other=0.0).to(tl.float32)
            rk = tl.load(rk_ptr + head_id * N + offs_i, mask=mask_i, other=0.0).to(tl.float32)
            weight = tl.load(gn_weight_ptr + head_id * N + offs_i, mask=mask_i, other=1.0).to(tl.float32)
            bias = tl.load(gn_bias_ptr + head_id * N + offs_i, mask=mask_i, other=0.0).to(tl.float32)
            corr_scale = tl.sum(r_rows * k_rows * rk, axis=0)
            prepared = (normed * weight + bias + corr_scale * v_cols) * g_rows
            tl.store(out_ptr + vec_base + offs_i, prepared, mask=mask_i)
            t += 1

        tl.store(
            final_state_ptr + state_base + offs_i[:, None] * N + offs_j[None, :],
            st,
            mask=mask_i[:, None] & mask_j[None, :],
        )


def fused_recurrent_update_available() -> bool:
    """Return whether the optional Triton recurrent update prototype can run."""

    return bool(_HAS_TRITON and torch is not None)


def fused_recurrent_output_prepare_available() -> bool:
    """Return whether fused recurrent-update-plus-output-prep can run."""

    return bool(_HAS_TRITON and torch is not None)


def fused_recurrent_scan_available() -> bool:
    """Return whether the optional Triton recurrent scan prototype can run."""

    return bool(_HAS_TRITON and torch is not None)


def fused_recurrent_scan_clampw_available() -> bool:
    """Return whether the raw-W clampw recurrent scan prototype can run."""

    return bool(_HAS_TRITON and torch is not None)


def fused_recurrent_scan_state_prep_available() -> bool:
    """Return whether fused state-prep plus recurrent scan can run."""

    return bool(_HAS_TRITON and torch is not None)


def fused_recurrent_scan_state_prep_correction_available() -> bool:
    """Return whether fused state-prep scan can emit correction without K/V writeback."""

    return bool(_HAS_TRITON and torch is not None)


def fused_recurrent_scan_state_prep_output_prepare_available() -> bool:
    """Return whether fused state-prep, scan, and output-prep can run."""

    return bool(_HAS_TRITON and torch is not None)


def fused_recurrent_scan_output_prepare_available() -> bool:
    """Return whether fused prefill scan plus attention-output prep can run."""

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


def _as_bthn(x: Any, H: int, N: int, *, name: str):
    if torch is None:
        raise RuntimeError("fused_recurrent_scan requires torch")
    if x.dim() == 4:
        if int(x.shape[2]) != H or int(x.shape[3]) != N:
            raise ValueError(f"{name} must be shaped [batch,tokens,{H},{N}] or [batch,tokens,{H * N}]; got {tuple(x.shape)}")
        return x.contiguous(), False
    if x.dim() == 3:
        if int(x.shape[2]) != H * N:
            raise ValueError(f"{name} must be shaped [batch,tokens,{H},{N}] or [batch,tokens,{H * N}]; got {tuple(x.shape)}")
        return x.reshape(int(x.shape[0]), int(x.shape[1]), H, N).contiguous(), True
    raise ValueError(f"{name} must be shaped [batch,tokens,{H},{N}] or [batch,tokens,{H * N}]")


def torch_recurrent_scan(r: Any, w: Any, k: Any, v: Any, kk: Any, a: Any, state: Any):
    """Reference multi-token recurrent scan matching the native_jit formula.

    Inputs are post-projection tensors shaped ``[B, T, H, N]`` or
    ``[B, T, H*N]`` plus an initial state ``[B, H, N, N]``.  Returns the
    recurrent output for every token and the final state.  FLA
    ``chunk_rwkv7`` uses the same high-level RWKV-7 DPLR recurrence but a
    different interface/orientation convention, so benchmark rows use it as a
    speed target while strict correctness is checked against this reference.
    """

    if torch is None:
        raise RuntimeError("torch_recurrent_scan requires torch")
    if state.dim() != 4:
        raise ValueError("state must be shaped [batch, heads, head_dim, head_dim]")
    B, H, N, N2 = (int(vv) for vv in state.shape)
    if N != N2:
        raise ValueError("state must be square in the last two dimensions")
    r4, flat = _as_bthn(r, H, N, name="r")
    w4, _ = _as_bthn(w, H, N, name="w")
    k4, _ = _as_bthn(k, H, N, name="k")
    v4, _ = _as_bthn(v, H, N, name="v")
    kk4, _ = _as_bthn(kk, H, N, name="kk")
    a4, _ = _as_bthn(a, H, N, name="a")
    if int(r4.shape[0]) != B:
        raise ValueError("r/w/k/v/kk/a batch size must match state")

    cur_state = state
    outs = []
    for t in range(int(r4.shape[1])):
        out, cur_state = torch_recurrent_update(
            r4[:, t],
            w4[:, t],
            k4[:, t],
            v4[:, t],
            kk4[:, t],
            a4[:, t],
            cur_state,
        )
        out4, _ = _as_bhn(out, H, N, name="scan_out")
        outs.append(out4)
    stacked = torch.stack(outs, dim=1) if outs else r4.new_empty((B, 0, H, N))
    if flat:
        return stacked.reshape(B, int(r4.shape[1]), H * N), cur_state
    return stacked, cur_state


def torch_recurrent_scan_clampw(r: Any, w_raw: Any, k: Any, v: Any, kk: Any, a: Any, state: Any):
    """Reference scan that consumes raw W and computes RWKV-7 decay internally."""

    if torch is None:
        raise RuntimeError("torch_recurrent_scan_clampw requires torch")
    if state.dim() != 4:
        raise ValueError("state must be shaped [batch, heads, head_dim, head_dim]")
    B, H, N, N2 = (int(vv) for vv in state.shape)
    if N != N2:
        raise ValueError("state must be square in the last two dimensions")
    w4, flat = _as_bthn(w_raw, H, N, name="w_raw")
    w_decay = torch.exp(-0.606531 * torch.sigmoid(w4.float()))
    return torch_recurrent_scan(
        r,
        w_decay.reshape(B, int(w4.shape[1]), H * N) if flat else w_decay,
        k,
        v,
        kk,
        a,
        state,
    )


def torch_recurrent_output_prepare(
    r: Any,
    w: Any,
    k: Any,
    v: Any,
    kk: Any,
    a: Any,
    state: Any,
    g: Any,
    r_k: Any,
    group_norm_weight: Any,
    group_norm_bias: Any,
    *,
    eps: float,
):
    """Reference recurrent update followed by attention output prep."""

    if torch is None or F is None:
        raise RuntimeError("torch_recurrent_output_prepare requires torch")
    recurrent, new_state = torch_recurrent_update(r, w, k, v, kk, a, state)
    if state.dim() != 4:
        raise ValueError("state must be shaped [batch, heads, head_dim, head_dim]")
    B, H, N, _ = (int(vv) for vv in state.shape)
    rec3, flat = _as_bhn(recurrent, H, N, name="recurrent")
    r3, _ = _as_bhn(r, H, N, name="r")
    k3, _ = _as_bhn(k, H, N, name="k")
    v3, _ = _as_bhn(v, H, N, name="v")
    g3, _ = _as_bhn(g, H, N, name="g")
    if r_k.dim() != 2 or int(r_k.shape[0]) != H or int(r_k.shape[1]) != N:
        raise ValueError(f"r_k must be [{H}, {N}], got {tuple(r_k.shape)}")
    if group_norm_weight.dim() != 1 or int(group_norm_weight.shape[0]) != H * N:
        raise ValueError(f"group_norm_weight must be [{H * N}], got {tuple(group_norm_weight.shape)}")
    if group_norm_bias.dim() != 1 or int(group_norm_bias.shape[0]) != H * N:
        raise ValueError(f"group_norm_bias must be [{H * N}], got {tuple(group_norm_bias.shape)}")
    normed = F.group_norm(
        rec3.reshape(B, H * N),
        num_groups=H,
        weight=group_norm_weight,
        bias=group_norm_bias,
        eps=float(eps),
    ).reshape(B, H, N)
    correction = ((r3 * k3 * r_k.view(1, H, N)).sum(-1, keepdim=True) * v3)
    prepared = (normed + correction) * g3
    if flat:
        return prepared.reshape(B, H * N), new_state
    return prepared, new_state


def torch_recurrent_scan_output_prepare(
    r: Any,
    w: Any,
    k: Any,
    v: Any,
    kk: Any,
    a: Any,
    state: Any,
    g: Any,
    r_k: Any,
    group_norm_weight: Any,
    group_norm_bias: Any,
    *,
    eps: float,
):
    """Reference prefill recurrent scan followed by per-token output prep."""

    if torch is None or F is None:
        raise RuntimeError("torch_recurrent_scan_output_prepare requires torch")
    if state.dim() != 4:
        raise ValueError("state must be shaped [batch, heads, head_dim, head_dim]")
    B, H, N, _ = (int(vv) for vv in state.shape)
    recurrent, new_state = torch_recurrent_scan(r, w, k, v, kk, a, state)
    rec4, flat = _as_bthn(recurrent, H, N, name="recurrent")
    r4, _ = _as_bthn(r, H, N, name="r")
    k4, _ = _as_bthn(k, H, N, name="k")
    v4, _ = _as_bthn(v, H, N, name="v")
    g4, _ = _as_bthn(g, H, N, name="g")
    T = int(rec4.shape[1])
    if r_k.dim() != 2 or int(r_k.shape[0]) != H or int(r_k.shape[1]) != N:
        raise ValueError(f"r_k must be [{H}, {N}], got {tuple(r_k.shape)}")
    if group_norm_weight.dim() != 1 or int(group_norm_weight.shape[0]) != H * N:
        raise ValueError(f"group_norm_weight must be [{H * N}], got {tuple(group_norm_weight.shape)}")
    if group_norm_bias.dim() != 1 or int(group_norm_bias.shape[0]) != H * N:
        raise ValueError(f"group_norm_bias must be [{H * N}], got {tuple(group_norm_bias.shape)}")
    normed = F.group_norm(
        rec4.reshape(B * T, H * N),
        num_groups=H,
        weight=group_norm_weight,
        bias=group_norm_bias,
        eps=float(eps),
    ).reshape(B, T, H, N)
    correction = ((r4 * k4 * r_k.view(1, 1, H, N)).sum(-1, keepdim=True) * v4)
    prepared = (normed + correction) * g4
    if flat:
        return prepared.reshape(B, T, H * N), new_state
    return prepared, new_state


def fused_recurrent_output_prepare(
    r: Any,
    w: Any,
    k: Any,
    v: Any,
    kk: Any,
    a: Any,
    state: Any,
    g: Any,
    r_k: Any,
    group_norm_weight: Any,
    group_norm_bias: Any,
    *,
    eps: float,
    block_n: int = 64,
    force_fallback: bool = False,
):
    """Fuse recurrent update/readout with output prep before ``o_proj``.

    ``r,w,k,v,kk,a,g`` may be shaped ``[batch, heads, head_dim]`` or flattened
    as ``[batch, hidden]``. ``state`` must be ``[batch, heads, head_dim,
    head_dim]``. The returned prepared output follows the input rank.
    """

    if torch is None:
        raise RuntimeError("fused_recurrent_output_prepare requires torch")
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
    g3, _ = _as_bhn(g, H, N, name="g")
    if int(r3.shape[0]) != B:
        raise ValueError("r/w/k/v/kk/a/g batch size must match state")
    if r_k.dim() != 2 or int(r_k.shape[0]) != H or int(r_k.shape[1]) != N:
        raise ValueError(f"r_k must be [{H}, {N}], got {tuple(r_k.shape)}")
    if group_norm_weight.dim() != 1 or int(group_norm_weight.shape[0]) != H * N:
        raise ValueError(f"group_norm_weight must be [{H * N}], got {tuple(group_norm_weight.shape)}")
    if group_norm_bias.dim() != 1 or int(group_norm_bias.shape[0]) != H * N:
        raise ValueError(f"group_norm_bias must be [{H * N}], got {tuple(group_norm_bias.shape)}")

    use_triton = (
        not force_fallback
        and fused_recurrent_output_prepare_available()
        and r3.is_cuda
        and w3.is_cuda
        and k3.is_cuda
        and v3.is_cuda
        and kk3.is_cuda
        and a3.is_cuda
        and g3.is_cuda
        and state.is_cuda
        and r_k.is_cuda
        and group_norm_weight.is_cuda
        and group_norm_bias.is_cuda
        and state.dtype == torch.float32
        and r3.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and w3.dtype in (r3.dtype, torch.float32)
        and all(t.dtype == r3.dtype for t in (k3, v3, kk3, a3, g3, r_k, group_norm_weight, group_norm_bias))
    )
    if not use_triton:
        return torch_recurrent_output_prepare(
            r3.reshape(B, H * N) if flat else r3,
            w3,
            k3,
            v3,
            kk3,
            a3,
            state,
            g3,
            r_k,
            group_norm_weight,
            group_norm_bias,
            eps=eps,
        )

    r_c = r3.contiguous()
    w_c = w3.contiguous()
    k_c = k3.contiguous()
    v_c = v3.contiguous()
    kk_c = kk3.contiguous()
    a_c = a3.contiguous()
    g_c = g3.contiguous()
    state_c = state.contiguous()
    rk_c = r_k.contiguous()
    gnw_c = group_norm_weight.contiguous()
    gnb_c = group_norm_bias.contiguous()
    out = torch.empty((B, H, N), device=r3.device, dtype=r3.dtype)
    new_state = torch.empty_like(state_c)
    _recurrent_output_prepare_kernel[(B * H,)](
        r_c,
        w_c,
        k_c,
        v_c,
        kk_c,
        a_c,
        state_c,
        g_c,
        rk_c,
        gnw_c,
        gnb_c,
        out,
        new_state,
        H,
        N,
        float(eps),
        BLOCK_N=int(block_n),
        num_warps=8,
    )
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
        and w3.dtype in (r3.dtype, torch.float32)
        and all(t.dtype == r3.dtype for t in (k3, v3, kk3, a3))
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


def fused_recurrent_scan(
    r: Any,
    w: Any,
    k: Any,
    v: Any,
    kk: Any,
    a: Any,
    state: Any,
    *,
    block_n: int = 64,
    block_m: int | None = None,
    num_warps: int | None = None,
    force_fallback: bool = False,
):
    """Compute a multi-token RWKV-7 recurrent scan with an optional Triton kernel.

    This is a prefill prototype over already projected tensors.  Inputs may be
    shaped ``[batch, tokens, heads, head_dim]`` or flattened as
    ``[batch, tokens, hidden]``. ``state`` must be ``[batch, heads, head_dim,
    head_dim]`` and is not modified in place.  The output shape follows the
    vector input rank.
    """

    if torch is None:
        raise RuntimeError("fused_recurrent_scan requires torch")
    if state.dim() != 4:
        raise ValueError("state must be shaped [batch, heads, head_dim, head_dim]")
    B, H, N, N2 = (int(vv) for vv in state.shape)
    if N != N2:
        raise ValueError("state must be square in the last two dimensions")
    if int(block_n) < N:
        raise ValueError(f"block_n must be >= head_dim={N}; got {block_n}")
    if block_m is None:
        block_m = N
    block_m = int(block_m)
    if block_m <= 0:
        raise ValueError(f"block_m must be positive; got {block_m}")
    if num_warps is None:
        num_warps = 4 if block_m < N else 8
    num_warps = int(num_warps)
    if num_warps not in {1, 2, 4, 8}:
        raise ValueError(f"num_warps must be one of 1, 2, 4, or 8; got {num_warps}")
    r4, flat = _as_bthn(r, H, N, name="r")
    w4, _ = _as_bthn(w, H, N, name="w")
    k4, _ = _as_bthn(k, H, N, name="k")
    v4, _ = _as_bthn(v, H, N, name="v")
    kk4, _ = _as_bthn(kk, H, N, name="kk")
    a4, _ = _as_bthn(a, H, N, name="a")
    if int(r4.shape[0]) != B:
        raise ValueError("r/w/k/v/kk/a batch size must match state")

    use_triton = (
        not force_fallback
        and fused_recurrent_scan_available()
        and r4.is_cuda
        and w4.is_cuda
        and k4.is_cuda
        and v4.is_cuda
        and kk4.is_cuda
        and a4.is_cuda
        and state.is_cuda
        and state.dtype == torch.float32
        and r4.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and w4.dtype in (r4.dtype, torch.float32)
        and all(t.dtype == r4.dtype for t in (k4, v4, kk4, a4))
    )
    if not use_triton:
        return torch_recurrent_scan(r4.reshape(B, int(r4.shape[1]), H * N) if flat else r4, w4, k4, v4, kk4, a4, state)

    T = int(r4.shape[1])
    r_c = r4.contiguous()
    w_c = w4.contiguous()
    k_c = k4.contiguous()
    v_c = v4.contiguous()
    kk_c = kk4.contiguous()
    a_c = a4.contiguous()
    state_c = state.contiguous()
    out = torch.empty((B, T, H, N), device=r4.device, dtype=r4.dtype)
    final_state = torch.empty_like(state_c)
    if block_m < N:
        row_blocks = triton.cdiv(N, block_m)
        _recurrent_scan_rows_kernel[(B * H * row_blocks,)](
            r_c,
            w_c,
            k_c,
            v_c,
            kk_c,
            a_c,
            state_c,
            out,
            final_state,
            T,
            H,
            N,
            ROW_BLOCKS=int(row_blocks),
            BLOCK_M=int(block_m),
            BLOCK_N=int(block_n),
            num_warps=int(num_warps),
        )
    else:
        _recurrent_scan_kernel[(B * H,)](
            r_c,
            w_c,
            k_c,
            v_c,
            kk_c,
            a_c,
            state_c,
            out,
            final_state,
            T,
            H,
            N,
            BLOCK_N=int(block_n),
            num_warps=int(num_warps),
        )
    if flat:
        return out.reshape(B, T, H * N), final_state
    return out, final_state


def fused_recurrent_scan_clampw(
    r: Any,
    w_raw: Any,
    k: Any,
    v: Any,
    kk: Any,
    a: Any,
    state: Any,
    *,
    block_n: int = 64,
    block_m: int | None = None,
    num_warps: int | None = None,
    force_fallback: bool = False,
):
    """Compute a recurrent scan while keeping raw W decay inside the kernel.

    This mirrors :func:`fused_recurrent_scan`, but consumes pre-clamp/pre-decay
    ``w_raw`` and computes ``exp(-0.606531 * sigmoid(w_raw))`` inside the scan.
    It is an opt-in prefill experiment for train_temp-style ``clampw`` fusion:
    state-prep can skip writing a full W-decay tensor that the scan immediately
    rereads.
    """

    if torch is None:
        raise RuntimeError("fused_recurrent_scan_clampw requires torch")
    if state.dim() != 4:
        raise ValueError("state must be shaped [batch, heads, head_dim, head_dim]")
    B, H, N, N2 = (int(vv) for vv in state.shape)
    if N != N2:
        raise ValueError("state must be square in the last two dimensions")
    if int(block_n) < N:
        raise ValueError(f"block_n must be >= head_dim={N}; got {block_n}")
    if block_m is None:
        block_m = N
    block_m = int(block_m)
    if block_m <= 0:
        raise ValueError(f"block_m must be positive; got {block_m}")
    if num_warps is None:
        num_warps = 4 if block_m < N else 8
    num_warps = int(num_warps)
    if num_warps not in {1, 2, 4, 8}:
        raise ValueError(f"num_warps must be one of 1, 2, 4, or 8; got {num_warps}")
    r4, flat = _as_bthn(r, H, N, name="r")
    w4, _ = _as_bthn(w_raw, H, N, name="w_raw")
    k4, _ = _as_bthn(k, H, N, name="k")
    v4, _ = _as_bthn(v, H, N, name="v")
    kk4, _ = _as_bthn(kk, H, N, name="kk")
    a4, _ = _as_bthn(a, H, N, name="a")
    if int(r4.shape[0]) != B:
        raise ValueError("r/w/k/v/kk/a batch size must match state")

    use_triton = (
        not force_fallback
        and fused_recurrent_scan_clampw_available()
        and r4.is_cuda
        and w4.is_cuda
        and k4.is_cuda
        and v4.is_cuda
        and kk4.is_cuda
        and a4.is_cuda
        and state.is_cuda
        and state.dtype == torch.float32
        and r4.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and w4.dtype == r4.dtype
        and all(t.dtype == r4.dtype for t in (k4, v4, kk4, a4))
    )
    if not use_triton:
        return torch_recurrent_scan_clampw(r4.reshape(B, int(r4.shape[1]), H * N) if flat else r4, w4, k4, v4, kk4, a4, state)

    T = int(r4.shape[1])
    r_c = r4.contiguous()
    w_c = w4.contiguous()
    k_c = k4.contiguous()
    v_c = v4.contiguous()
    kk_c = kk4.contiguous()
    a_c = a4.contiguous()
    state_c = state.contiguous()
    out = torch.empty((B, T, H, N), device=r4.device, dtype=r4.dtype)
    final_state = torch.empty_like(state_c)
    if block_m < N:
        row_blocks = triton.cdiv(N, block_m)
        _recurrent_scan_rows_clampw_kernel[(B * H * row_blocks,)](
            r_c,
            w_c,
            k_c,
            v_c,
            kk_c,
            a_c,
            state_c,
            out,
            final_state,
            T,
            H,
            N,
            ROW_BLOCKS=int(row_blocks),
            BLOCK_M=int(block_m),
            BLOCK_N=int(block_n),
            num_warps=int(num_warps),
        )
    else:
        _recurrent_scan_clampw_kernel[(B * H,)](
            r_c,
            w_c,
            k_c,
            v_c,
            kk_c,
            a_c,
            state_c,
            out,
            final_state,
            T,
            H,
            N,
            BLOCK_N=int(block_n),
            num_warps=int(num_warps),
        )
    if flat:
        return out.reshape(B, T, H * N), final_state
    return out, final_state


def fused_recurrent_scan_state_prep(
    r: Any,
    w_raw: Any,
    k_raw: Any,
    v_raw: Any,
    a: Any,
    state: Any,
    k_k: Any,
    k_a: Any,
    *,
    v_first: Any | None = None,
    v_gate: Any | None = None,
    block_n: int = 64,
    block_m: int | None = None,
    num_warps: int = 8,
    num_stages: int = 3,
    force_fallback: bool = False,
):
    """Fuse native-prefill state prep with the recurrent scan.

    This opt-in bsz=1 prompt-prefill prototype consumes raw W/K/V and already
    sigmoid'd A/V-gate tensors, computes W decay, adjusted K, interpolated V,
    and normalized KK inside the scan, and returns ``(out, final_state, k, v)``
    so the existing attention output-prep code can keep using adjusted K/V.

    The default Triton path is full-head.  Passing ``block_m < head_dim``
    enables an experimental split-row variant that trades duplicated per-head
    vector prep for more parallelism on low-batch prompt-prefill shapes.
    """

    if torch is None or F is None:
        raise RuntimeError("fused_recurrent_scan_state_prep requires torch")
    if state.dim() != 4:
        raise ValueError("state must be shaped [batch, heads, head_dim, head_dim]")
    B, H, N, N2 = (int(vv) for vv in state.shape)
    if N != N2:
        raise ValueError("state must be square in the last two dimensions")
    if int(block_n) < N:
        raise ValueError(f"block_n must be >= head_dim={N}; got {block_n}")
    if block_m is None:
        block_m = N
    block_m = int(block_m)
    if block_m <= 0 or block_m > N:
        raise ValueError(f"block_m must be in [1, head_dim={N}]; got {block_m}")
    if int(num_warps) not in {1, 2, 4, 8}:
        raise ValueError(f"num_warps must be one of 1, 2, 4, or 8; got {num_warps}")
    num_stages = int(num_stages)
    if num_stages < 1 or num_stages > 8:
        raise ValueError(f"num_stages must be in [1, 8]; got {num_stages}")
    r4, flat = _as_bthn(r, H, N, name="r")
    w4, _ = _as_bthn(w_raw, H, N, name="w_raw")
    k4, _ = _as_bthn(k_raw, H, N, name="k_raw")
    v4, _ = _as_bthn(v_raw, H, N, name="v_raw")
    a4, _ = _as_bthn(a, H, N, name="a")
    if int(r4.shape[0]) != B:
        raise ValueError("r/w/k/v/a batch size must match state")
    hidden = H * N
    if int(k_k.numel()) != hidden or int(k_a.numel()) != hidden:
        raise ValueError(f"k_k and k_a must have {hidden} elements")
    has_v_gate = v_first is not None and v_gate is not None
    if has_v_gate:
        vf4, _ = _as_bthn(v_first, H, N, name="v_first")
        vg4, _ = _as_bthn(v_gate, H, N, name="v_gate")
    else:
        vf4 = v4
        vg4 = v4

    use_triton = (
        not force_fallback
        and fused_recurrent_scan_state_prep_available()
        and r4.is_cuda
        and w4.is_cuda
        and k4.is_cuda
        and v4.is_cuda
        and a4.is_cuda
        and state.is_cuda
        and k_k.is_cuda
        and k_a.is_cuda
        and (not has_v_gate or (vf4.is_cuda and vg4.is_cuda))
        and state.dtype == torch.float32
        and r4.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and all(t.dtype == r4.dtype for t in (w4, k4, v4, a4))
        and k_k.dtype == r4.dtype
        and k_a.dtype == r4.dtype
        and (not has_v_gate or (vf4.dtype == r4.dtype and vg4.dtype == r4.dtype))
    )
    if not use_triton:
        kk = F.normalize((k4.reshape(B, int(k4.shape[1]), hidden) * k_k.reshape(1, 1, hidden)).view(B, int(k4.shape[1]), H, N), dim=-1, p=2.0)
        k_adj = (k4.reshape(B, int(k4.shape[1]), hidden) * (1 + (a4.reshape(B, int(a4.shape[1]), hidden) - 1) * k_a.reshape(1, 1, hidden))).view_as(k4)
        if has_v_gate:
            v_adj = v4 + (vf4 - v4) * vg4
        else:
            v_adj = v4
        w_decay = torch.exp(-0.606531 * torch.sigmoid(w4.float()))
        out, final_state = torch_recurrent_scan(r4.reshape(B, int(r4.shape[1]), H * N) if flat else r4, w_decay, k_adj, v_adj, kk, a4, state)
        if flat:
            return out, final_state, k_adj.reshape(B, int(k4.shape[1]), H * N), v_adj.reshape(B, int(v4.shape[1]), H * N)
        return out, final_state, k_adj, v_adj

    T = int(r4.shape[1])
    r_c = r4.contiguous()
    w_c = w4.contiguous()
    k_c = k4.contiguous()
    v_c = v4.contiguous()
    a_c = a4.contiguous()
    state_c = state.contiguous()
    kk_c = k_k.reshape(hidden).contiguous()
    ka_c = k_a.reshape(hidden).contiguous()
    vf_c = vf4.contiguous()
    vg_c = vg4.contiguous()
    out = torch.empty((B, T, H, N), device=r4.device, dtype=r4.dtype)
    final_state = torch.empty_like(state_c)
    k_out = torch.empty_like(k_c)
    v_out = torch.empty_like(v_c)
    if block_m < N:
        row_blocks = triton.cdiv(N, block_m)
        _recurrent_scan_state_prep_rows_kernel[(B * H * row_blocks,)](
            r_c,
            w_c,
            k_c,
            v_c,
            a_c,
            state_c,
            kk_c,
            ka_c,
            vf_c,
            vg_c,
            out,
            final_state,
            k_out,
            v_out,
            T,
            H,
            N,
            ROW_BLOCKS=int(row_blocks),
            HAS_V_GATE=bool(has_v_gate),
            BLOCK_M=int(block_m),
            BLOCK_N=int(block_n),
            num_warps=int(num_warps),
            num_stages=num_stages,
        )
    else:
        _recurrent_scan_state_prep_kernel[(B * H,)](
            r_c,
            w_c,
            k_c,
            v_c,
            a_c,
            state_c,
            kk_c,
            ka_c,
            vf_c,
            vg_c,
            out,
            final_state,
            k_out,
            v_out,
            T,
            H,
            N,
            HAS_V_GATE=bool(has_v_gate),
            BLOCK_N=int(block_n),
            num_warps=int(num_warps),
            num_stages=num_stages,
        )
    if flat:
        return out.reshape(B, T, H * N), final_state, k_out.reshape(B, T, H * N), v_out.reshape(B, T, H * N)
    return out, final_state, k_out, v_out


def fused_recurrent_scan_state_prep_correction(
    r: Any,
    w_raw: Any,
    k_raw: Any,
    v_raw: Any,
    a: Any,
    state: Any,
    k_k: Any,
    k_a: Any,
    r_k: Any,
    *,
    v_first: Any | None = None,
    v_gate: Any | None = None,
    block_n: int = 64,
    block_m: int | None = None,
    num_warps: int = 8,
    num_stages: int = 3,
    force_fallback: bool = False,
):
    """Fuse state prep and scan while emitting RWKV correction instead of K/V.

    This opt-in prefill experiment targets the current dominant 4090 path.  The
    existing ``fused_recurrent_scan_state_prep`` writes adjusted K and V for the
    later output-prep correction.  This variant computes
    ``sum(r * k_adj * r_k) * v_adj`` inside the scan loop and returns
    ``(recurrent, final_state, correction)`` so the full adjusted K/V tensors do
    not need to be materialized in global memory.

    The Triton fast path is intentionally full-head only; split-row remains a
    separate knob and falls back through the existing K/V-returning path.
    """

    if torch is None or F is None:
        raise RuntimeError("fused_recurrent_scan_state_prep_correction requires torch")
    if state.dim() != 4:
        raise ValueError("state must be shaped [batch, heads, head_dim, head_dim]")
    B, H, N, N2 = (int(vv) for vv in state.shape)
    if N != N2:
        raise ValueError("state must be square in the last two dimensions")
    if int(block_n) < N:
        raise ValueError(f"block_n must be >= head_dim={N}; got {block_n}")
    if block_m is None:
        block_m = N
    block_m = int(block_m)
    if block_m <= 0 or block_m > N:
        raise ValueError(f"block_m must be in [1, head_dim={N}]; got {block_m}")
    if int(num_warps) not in {1, 2, 4, 8}:
        raise ValueError(f"num_warps must be one of 1, 2, 4, or 8; got {num_warps}")
    num_stages = int(num_stages)
    if num_stages < 1 or num_stages > 8:
        raise ValueError(f"num_stages must be in [1, 8]; got {num_stages}")

    r4, flat = _as_bthn(r, H, N, name="r")
    w4, _ = _as_bthn(w_raw, H, N, name="w_raw")
    k4, _ = _as_bthn(k_raw, H, N, name="k_raw")
    v4, _ = _as_bthn(v_raw, H, N, name="v_raw")
    a4, _ = _as_bthn(a, H, N, name="a")
    if int(r4.shape[0]) != B:
        raise ValueError("r/w/k/v/a batch size must match state")
    hidden = H * N
    if int(k_k.numel()) != hidden or int(k_a.numel()) != hidden or int(r_k.numel()) != hidden:
        raise ValueError(f"k_k, k_a, and r_k must have {hidden} elements")
    has_v_gate = v_first is not None and v_gate is not None
    if has_v_gate:
        vf4, _ = _as_bthn(v_first, H, N, name="v_first")
        vg4, _ = _as_bthn(v_gate, H, N, name="v_gate")
    else:
        vf4 = v4
        vg4 = v4

    use_triton = (
        not force_fallback
        and int(block_m) == N
        and fused_recurrent_scan_state_prep_correction_available()
        and r4.is_cuda
        and w4.is_cuda
        and k4.is_cuda
        and v4.is_cuda
        and a4.is_cuda
        and state.is_cuda
        and k_k.is_cuda
        and k_a.is_cuda
        and r_k.is_cuda
        and (not has_v_gate or (vf4.is_cuda and vg4.is_cuda))
        and state.dtype == torch.float32
        and r4.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and all(t.dtype == r4.dtype for t in (w4, k4, v4, a4))
        and k_k.dtype == r4.dtype
        and k_a.dtype == r4.dtype
        and r_k.dtype == r4.dtype
        and (not has_v_gate or (vf4.dtype == r4.dtype and vg4.dtype == r4.dtype))
    )
    if not use_triton:
        out, final_state, k_adj, v_adj = fused_recurrent_scan_state_prep(
            r4,
            w4,
            k4,
            v4,
            a4,
            state,
            k_k,
            k_a,
            v_first=vf4 if has_v_gate else None,
            v_gate=vg4 if has_v_gate else None,
            block_n=block_n,
            block_m=block_m,
            num_warps=num_warps,
            num_stages=num_stages,
            force_fallback=force_fallback,
        )
        out4, _ = _as_bthn(out, H, N, name="out")
        k_adj4, _ = _as_bthn(k_adj, H, N, name="k_adj")
        v_adj4, _ = _as_bthn(v_adj, H, N, name="v_adj")
        correction = ((r4 * k_adj4 * r_k.reshape(1, 1, H, N)).sum(-1, keepdim=True) * v_adj4)
        if flat:
            return out4.reshape(B, int(r4.shape[1]), H * N), final_state, correction.reshape(B, int(r4.shape[1]), H * N)
        return out4, final_state, correction

    T = int(r4.shape[1])
    r_c = r4.contiguous()
    w_c = w4.contiguous()
    k_c = k4.contiguous()
    v_c = v4.contiguous()
    a_c = a4.contiguous()
    state_c = state.contiguous()
    kk_c = k_k.reshape(hidden).contiguous()
    ka_c = k_a.reshape(hidden).contiguous()
    rk_c = r_k.reshape(hidden).contiguous()
    vf_c = vf4.contiguous()
    vg_c = vg4.contiguous()
    out = torch.empty((B, T, H, N), device=r4.device, dtype=r4.dtype)
    final_state = torch.empty_like(state_c)
    correction = torch.empty_like(out)
    _recurrent_scan_state_prep_correction_kernel[(B * H,)](
        r_c,
        w_c,
        k_c,
        v_c,
        a_c,
        state_c,
        kk_c,
        ka_c,
        rk_c,
        vf_c,
        vg_c,
        out,
        final_state,
        correction,
        T,
        H,
        N,
        HAS_V_GATE=bool(has_v_gate),
        BLOCK_N=int(block_n),
        num_warps=int(num_warps),
        num_stages=num_stages,
    )
    if flat:
        return out.reshape(B, T, H * N), final_state, correction.reshape(B, T, H * N)
    return out, final_state, correction


def fused_recurrent_scan_state_prep_output_prepare(
    r: Any,
    w_raw: Any,
    k_raw: Any,
    v_raw: Any,
    a: Any,
    state: Any,
    k_k: Any,
    k_a: Any,
    g: Any,
    r_k: Any,
    group_norm_weight: Any,
    group_norm_bias: Any,
    *,
    eps: float,
    v_first: Any | None = None,
    v_gate: Any | None = None,
    block_n: int = 64,
    num_warps: int = 8,
    force_fallback: bool = False,
):
    """Fuse state prep, recurrent scan, and attention output prep.

    This is a larger opt-in prefill experiment for the 4090/Albatross gap.  It
    consumes raw W/K/V plus sigmoid'd A/V-gate, computes adjusted K/V and KK
    inside the recurrent scan, then immediately performs per-head group norm,
    RWKV correction, and G gating.  The final O projection intentionally stays
    on cuBLAS.
    """

    if torch is None or F is None:
        raise RuntimeError("fused_recurrent_scan_state_prep_output_prepare requires torch")
    if state.dim() != 4:
        raise ValueError("state must be shaped [batch, heads, head_dim, head_dim]")
    B, H, N, N2 = (int(vv) for vv in state.shape)
    if N != N2:
        raise ValueError("state must be square in the last two dimensions")
    if int(block_n) < N:
        raise ValueError(f"block_n must be >= head_dim={N}; got {block_n}")
    if int(num_warps) not in {1, 2, 4, 8}:
        raise ValueError(f"num_warps must be one of 1, 2, 4, or 8; got {num_warps}")
    r4, flat = _as_bthn(r, H, N, name="r")
    w4, _ = _as_bthn(w_raw, H, N, name="w_raw")
    k4, _ = _as_bthn(k_raw, H, N, name="k_raw")
    v4, _ = _as_bthn(v_raw, H, N, name="v_raw")
    a4, _ = _as_bthn(a, H, N, name="a")
    g4, _ = _as_bthn(g, H, N, name="g")
    if int(r4.shape[0]) != B:
        raise ValueError("r/w/k/v/a/g batch size must match state")
    hidden = H * N
    if int(k_k.numel()) != hidden or int(k_a.numel()) != hidden:
        raise ValueError(f"k_k and k_a must have {hidden} elements")
    if r_k.dim() != 2 or int(r_k.shape[0]) != H or int(r_k.shape[1]) != N:
        raise ValueError(f"r_k must be [{H}, {N}], got {tuple(r_k.shape)}")
    if group_norm_weight.dim() != 1 or int(group_norm_weight.shape[0]) != hidden:
        raise ValueError(f"group_norm_weight must be [{hidden}], got {tuple(group_norm_weight.shape)}")
    if group_norm_bias.dim() != 1 or int(group_norm_bias.shape[0]) != hidden:
        raise ValueError(f"group_norm_bias must be [{hidden}], got {tuple(group_norm_bias.shape)}")
    has_v_gate = v_first is not None and v_gate is not None
    if has_v_gate:
        vf4, _ = _as_bthn(v_first, H, N, name="v_first")
        vg4, _ = _as_bthn(v_gate, H, N, name="v_gate")
    else:
        vf4 = v4
        vg4 = v4

    use_triton = (
        not force_fallback
        and fused_recurrent_scan_state_prep_output_prepare_available()
        and r4.is_cuda
        and w4.is_cuda
        and k4.is_cuda
        and v4.is_cuda
        and a4.is_cuda
        and g4.is_cuda
        and state.is_cuda
        and k_k.is_cuda
        and k_a.is_cuda
        and r_k.is_cuda
        and group_norm_weight.is_cuda
        and group_norm_bias.is_cuda
        and (not has_v_gate or (vf4.is_cuda and vg4.is_cuda))
        and state.dtype == torch.float32
        and r4.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and all(t.dtype == r4.dtype for t in (w4, k4, v4, a4, g4, k_k, k_a, r_k, group_norm_weight, group_norm_bias))
        and (not has_v_gate or (vf4.dtype == r4.dtype and vg4.dtype == r4.dtype))
    )
    if not use_triton:
        kk = F.normalize((k4.reshape(B, int(k4.shape[1]), hidden) * k_k.reshape(1, 1, hidden)).view(B, int(k4.shape[1]), H, N), dim=-1, p=2.0)
        k_adj = (k4.reshape(B, int(k4.shape[1]), hidden) * (1 + (a4.reshape(B, int(a4.shape[1]), hidden) - 1) * k_a.reshape(1, 1, hidden))).view_as(k4)
        if has_v_gate:
            v_adj = v4 + (vf4 - v4) * vg4
        else:
            v_adj = v4
        w_decay = torch.exp(-0.606531 * torch.sigmoid(w4.float()))
        return torch_recurrent_scan_output_prepare(
            r4.reshape(B, int(r4.shape[1]), H * N) if flat else r4,
            w_decay,
            k_adj,
            v_adj,
            kk,
            a4,
            state,
            g4,
            r_k,
            group_norm_weight,
            group_norm_bias,
            eps=eps,
        )

    T = int(r4.shape[1])
    r_c = r4.contiguous()
    w_c = w4.contiguous()
    k_c = k4.contiguous()
    v_c = v4.contiguous()
    a_c = a4.contiguous()
    g_c = g4.contiguous()
    state_c = state.contiguous()
    kk_c = k_k.reshape(hidden).contiguous()
    ka_c = k_a.reshape(hidden).contiguous()
    vf_c = vf4.contiguous()
    vg_c = vg4.contiguous()
    rk_c = r_k.contiguous()
    gnw_c = group_norm_weight.contiguous()
    gnb_c = group_norm_bias.contiguous()
    out = torch.empty((B, T, H, N), device=r4.device, dtype=r4.dtype)
    final_state = torch.empty_like(state_c)
    _recurrent_scan_state_prep_output_prepare_kernel[(B * H,)](
        r_c,
        w_c,
        k_c,
        v_c,
        a_c,
        state_c,
        kk_c,
        ka_c,
        vf_c,
        vg_c,
        g_c,
        rk_c,
        gnw_c,
        gnb_c,
        out,
        final_state,
        T,
        H,
        N,
        HAS_V_GATE=bool(has_v_gate),
        eps=float(eps),
        BLOCK_N=int(block_n),
        num_warps=int(num_warps),
    )
    if flat:
        return out.reshape(B, T, H * N), final_state
    return out, final_state


def fused_recurrent_scan_output_prepare(
    r: Any,
    w: Any,
    k: Any,
    v: Any,
    kk: Any,
    a: Any,
    state: Any,
    g: Any,
    r_k: Any,
    group_norm_weight: Any,
    group_norm_bias: Any,
    *,
    eps: float,
    block_n: int = 64,
    force_fallback: bool = False,
):
    """Fuse prefill recurrent scan with RWKV-7 attention output prep.

    This is an opt-in full-head prefill prototype.  It intentionally leaves the
    final ``o_proj`` on cuBLAS, but avoids materializing the intermediate
    recurrent output before group-norm/correction/gate.  Unlike
    :func:`fused_recurrent_scan`, this kernel must own all rows of a head in one
    Triton program so it can compute group-norm statistics for the head.
    """

    if torch is None:
        raise RuntimeError("fused_recurrent_scan_output_prepare requires torch")
    if state.dim() != 4:
        raise ValueError("state must be shaped [batch, heads, head_dim, head_dim]")
    B, H, N, N2 = (int(vv) for vv in state.shape)
    if N != N2:
        raise ValueError("state must be square in the last two dimensions")
    if int(block_n) < N:
        raise ValueError(f"block_n must be >= head_dim={N}; got {block_n}")
    r4, flat = _as_bthn(r, H, N, name="r")
    w4, _ = _as_bthn(w, H, N, name="w")
    k4, _ = _as_bthn(k, H, N, name="k")
    v4, _ = _as_bthn(v, H, N, name="v")
    kk4, _ = _as_bthn(kk, H, N, name="kk")
    a4, _ = _as_bthn(a, H, N, name="a")
    g4, _ = _as_bthn(g, H, N, name="g")
    if int(r4.shape[0]) != B:
        raise ValueError("r/w/k/v/kk/a/g batch size must match state")
    if r_k.dim() != 2 or int(r_k.shape[0]) != H or int(r_k.shape[1]) != N:
        raise ValueError(f"r_k must be [{H}, {N}], got {tuple(r_k.shape)}")
    if group_norm_weight.dim() != 1 or int(group_norm_weight.shape[0]) != H * N:
        raise ValueError(f"group_norm_weight must be [{H * N}], got {tuple(group_norm_weight.shape)}")
    if group_norm_bias.dim() != 1 or int(group_norm_bias.shape[0]) != H * N:
        raise ValueError(f"group_norm_bias must be [{H * N}], got {tuple(group_norm_bias.shape)}")

    use_triton = (
        not force_fallback
        and fused_recurrent_scan_output_prepare_available()
        and r4.is_cuda
        and w4.is_cuda
        and k4.is_cuda
        and v4.is_cuda
        and kk4.is_cuda
        and a4.is_cuda
        and g4.is_cuda
        and state.is_cuda
        and r_k.is_cuda
        and group_norm_weight.is_cuda
        and group_norm_bias.is_cuda
        and state.dtype == torch.float32
        and r4.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and w4.dtype in (r4.dtype, torch.float32)
        and all(t.dtype == r4.dtype for t in (k4, v4, kk4, a4, g4, r_k, group_norm_weight, group_norm_bias))
    )
    if not use_triton:
        return torch_recurrent_scan_output_prepare(
            r4.reshape(B, int(r4.shape[1]), H * N) if flat else r4,
            w4,
            k4,
            v4,
            kk4,
            a4,
            state,
            g4,
            r_k,
            group_norm_weight,
            group_norm_bias,
            eps=eps,
        )

    T = int(r4.shape[1])
    r_c = r4.contiguous()
    w_c = w4.contiguous()
    k_c = k4.contiguous()
    v_c = v4.contiguous()
    kk_c = kk4.contiguous()
    a_c = a4.contiguous()
    g_c = g4.contiguous()
    state_c = state.contiguous()
    rk_c = r_k.contiguous()
    gnw_c = group_norm_weight.contiguous()
    gnb_c = group_norm_bias.contiguous()
    out = torch.empty((B, T, H, N), device=r4.device, dtype=r4.dtype)
    final_state = torch.empty_like(state_c)
    _recurrent_scan_output_prepare_kernel[(B * H,)](
        r_c,
        w_c,
        k_c,
        v_c,
        kk_c,
        a_c,
        state_c,
        g_c,
        rk_c,
        gnw_c,
        gnb_c,
        out,
        final_state,
        T,
        H,
        N,
        float(eps),
        BLOCK_N=int(block_n),
        num_warps=8,
    )
    if flat:
        return out.reshape(B, T, H * N), final_state
    return out, final_state
