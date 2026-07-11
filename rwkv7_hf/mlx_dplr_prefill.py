# coding=utf-8
"""Correctness-first MLX DPLR/WY chunked-prefill mathematics.

This module ports the compact summary contract from
``dplr_prefill_triton.py`` to MLX.  It deliberately uses high-level MLX
operations and Python chunk/token loops: the purpose is to establish and test
the Apple tensor layout for the three mathematical stages before replacing
each stage with fused Metal kernels.

For each token the RWKV-7 state update is an affine map::

    S' = S @ (diag(w) + (-kk) (kk*a)^T) + v k^T

A chunk is represented without dense transition/additive ``[N,N]`` matrices::

    transition = diag(diag) + transition_left @ transition_right.T
    additive   = additive_left @ additive_right.T

The exported helpers cover compact chunk summary, chunk-prefix combine, and
chunk apply/output.  They are parity oracles and integration seams, not a
production-speed backend yet.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from .mlx_bridge import mlx_available, require_mlx


def mlx_dplr_metal_available() -> bool:
    """Return whether the MLX runtime exposes custom Metal kernels."""

    if not mlx_available():
        return False
    try:
        mx = require_mlx()
        return bool(hasattr(mx, "fast") and hasattr(mx.fast, "metal_kernel"))
    except Exception:
        return False


@lru_cache(maxsize=1)
def _compact_wy_summary_metal_kernel() -> Any:
    mx = require_mlx()
    if not mlx_dplr_metal_available():
        raise RuntimeError("MLX custom Metal kernels are unavailable")
    source = r'''
        uint summary_id = thread_position_in_grid.x;
        uint B = uint(dims[0]);
        uint tokens = uint(dims[1]);
        uint heads = uint(dims[2]);
        uint chunks = uint(dims[4]);
        uint total = B * chunks * heads;
        if (summary_id >= total) {
            return;
        }

        uint head = summary_id % heads;
        uint chunk = (summary_id / heads) % chunks;
        uint batch = summary_id / (heads * chunks);
        uint diag_base = summary_id * N;
        uint factor_base = summary_id * N * C;

        for (uint n = 0; n < N; ++n) {
            transition_diag[diag_base + n] = 1.0f;
            for (uint rank = 0; rank < C; ++rank) {
                uint offset = factor_base + n * C + rank;
                transition_left[offset] = 0.0f;
                transition_right[offset] = 0.0f;
                additive_left[offset] = 0.0f;
                additive_right[offset] = 0.0f;
            }
        }

        float transition_coeff[C];
        float additive_coeff[C];
        for (uint local = 0; local < C; ++local) {
            uint token = chunk * C + local;

            for (uint rank = 0; rank < local; ++rank) {
                float tc = 0.0f;
                float ac = 0.0f;
                for (uint n = 0; n < N; ++n) {
                    uint input_offset = ((batch * tokens + token) * heads + head) * N + n;
                    float p = -float(kk[input_offset]);
                    uint factor_offset = factor_base + n * C + rank;
                    tc += transition_right[factor_offset] * p;
                    ac += additive_right[factor_offset] * p;
                }
                transition_coeff[rank] = tc;
                additive_coeff[rank] = ac;
            }

            for (uint n = 0; n < N; ++n) {
                uint input_offset = ((batch * tokens + token) * heads + head) * N + n;
                float wv = float(w[input_offset]);
                float kv = float(k[input_offset]);
                float vv = float(v[input_offset]);
                float kkv = float(kk[input_offset]);
                float p = -kkv;
                float q = kkv * float(a[input_offset]);
                float new_left = transition_diag[diag_base + n] * p;

                for (uint rank = 0; rank < local; ++rank) {
                    uint factor_offset = factor_base + n * C + rank;
                    new_left += transition_left[factor_offset] * transition_coeff[rank];
                    transition_right[factor_offset] *= wv;
                    additive_right[factor_offset] = additive_right[factor_offset] * wv
                                                    + q * additive_coeff[rank];
                }

                transition_diag[diag_base + n] *= wv;
                uint new_offset = factor_base + n * C + local;
                transition_left[new_offset] = new_left;
                transition_right[new_offset] = q;
                additive_left[new_offset] = vv;
                additive_right[new_offset] = kv;
            }
        }
    '''
    return mx.fast.metal_kernel(
        name="rwkv7_dplr_compact_wy_summary",
        input_names=["w", "k", "v", "kk", "a", "dims"],
        output_names=[
            "transition_diag",
            "transition_left",
            "transition_right",
            "additive_left",
            "additive_right",
        ],
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=1)
def _compact_wy_summary_tiled_metal_kernel() -> Any:
    """Return the head-dimension-parallel compact-summary kernel.

    One threadgroup owns one ``(batch, chunk, head)`` summary and one thread
    owns one head-dimension row.  Token order remains sequential, as required
    by the recurrence, while the rank dot-products and factor-row updates run
    in parallel across the threadgroup.
    """

    mx = require_mlx()
    if not mlx_dplr_metal_available():
        raise RuntimeError("MLX custom Metal kernels are unavailable")
    source = r'''
        uint lane = thread_position_in_threadgroup.x;
        uint summary_id = threadgroup_position_in_grid.x;
        uint B = uint(dims[0]);
        uint tokens = uint(dims[1]);
        uint heads = uint(dims[2]);
        uint chunks = uint(dims[4]);
        uint total = B * chunks * heads;
        if (summary_id >= total || lane >= N) {
            return;
        }

        uint head = summary_id % heads;
        uint chunk = (summary_id / heads) % chunks;
        uint batch = summary_id / (heads * chunks);
        uint diag_base = summary_id * N;
        uint factor_base = summary_id * N * C;
        uint row_base = factor_base + lane * C;

        transition_diag[diag_base + lane] = 1.0f;
        for (uint rank = 0; rank < C; ++rank) {
            uint offset = row_base + rank;
            transition_left[offset] = 0.0f;
            transition_right[offset] = 0.0f;
            additive_left[offset] = 0.0f;
            additive_right[offset] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_device);

        threadgroup float transition_coeff[C];
        threadgroup float additive_coeff[C];
        for (uint local = 0; local < C; ++local) {
            uint token = chunk * C + local;

            if (lane < local) {
                float tc = 0.0f;
                float ac = 0.0f;
                for (uint n = 0; n < N; ++n) {
                    uint input_offset = ((batch * tokens + token) * heads + head) * N + n;
                    float p = -float(kk[input_offset]);
                    uint factor_offset = factor_base + n * C + lane;
                    tc += transition_right[factor_offset] * p;
                    ac += additive_right[factor_offset] * p;
                }
                transition_coeff[lane] = tc;
                additive_coeff[lane] = ac;
            }
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);

            uint input_offset = ((batch * tokens + token) * heads + head) * N + lane;
            float wv = float(w[input_offset]);
            float kv = float(k[input_offset]);
            float vv = float(v[input_offset]);
            float kkv = float(kk[input_offset]);
            float p = -kkv;
            float q = kkv * float(a[input_offset]);
            float new_left = transition_diag[diag_base + lane] * p;

            for (uint rank = 0; rank < local; ++rank) {
                uint factor_offset = row_base + rank;
                new_left += transition_left[factor_offset] * transition_coeff[rank];
                transition_right[factor_offset] *= wv;
                additive_right[factor_offset] = additive_right[factor_offset] * wv
                                                + q * additive_coeff[rank];
            }

            transition_diag[diag_base + lane] *= wv;
            uint new_offset = row_base + local;
            transition_left[new_offset] = new_left;
            transition_right[new_offset] = q;
            additive_left[new_offset] = vv;
            additive_right[new_offset] = kv;
            threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
        }
    '''
    return mx.fast.metal_kernel(
        name="rwkv7_dplr_compact_wy_summary_tiled",
        input_names=["w", "k", "v", "kk", "a", "dims"],
        output_names=[
            "transition_diag",
            "transition_left",
            "transition_right",
            "additive_left",
            "additive_right",
        ],
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=1)
def _compact_wy_chunk_apply_metal_kernel() -> Any:
    mx = require_mlx()
    if not mlx_dplr_metal_available():
        raise RuntimeError("MLX custom Metal kernels are unavailable")
    source = r'''
        uint row_id = thread_position_in_grid.x;
        uint B = uint(dims[0]);
        uint tokens = uint(dims[1]);
        uint heads = uint(dims[2]);
        uint chunks = uint(dims[4]);
        uint total_rows = B * chunks * heads * N;
        if (row_id >= total_rows) {
            return;
        }

        uint row = row_id % N;
        uint head = (row_id / N) % heads;
        uint chunk = (row_id / (N * heads)) % chunks;
        uint batch = row_id / (N * heads * chunks);
        uint start_base = ((((batch * chunks + chunk) * heads + head) * N + row) * N);
        float state_row[N];
        for (uint col = 0; col < N; ++col) {
            state_row[col] = float(start_states[start_base + col]);
        }

        for (uint local = 0; local < C; ++local) {
            uint token = chunk * C + local;
            uint input_base = ((batch * tokens + token) * heads + head) * N;
            float dot_kk = 0.0f;
            for (uint col = 0; col < N; ++col) {
                dot_kk += state_row[col] * float(kk[input_base + col]);
            }

            float output = 0.0f;
            float value_row = float(v[input_base + row]);
            for (uint col = 0; col < N; ++col) {
                float kkv = float(kk[input_base + col]);
                float next = state_row[col] * float(w[input_base + col])
                           - dot_kk * (kkv * float(a[input_base + col]))
                           + value_row * float(k[input_base + col]);
                state_row[col] = next;
                output += next * float(r[input_base + col]);
            }
            uint output_offset = ((batch * tokens + token) * heads + head) * N + row;
            outputs[output_offset] = output;
        }

        uint end_base = ((((batch * chunks + chunk) * heads + head) * N + row) * N);
        for (uint col = 0; col < N; ++col) {
            chunk_ends[end_base + col] = state_row[col];
        }
    '''
    return mx.fast.metal_kernel(
        name="rwkv7_dplr_compact_wy_chunk_apply",
        input_names=["r", "w", "k", "v", "kk", "a", "start_states", "dims"],
        output_names=["outputs", "chunk_ends"],
        source=source,
        ensure_row_contiguous=True,
    )


@lru_cache(maxsize=1)
def _compact_wy_chunk_apply_output_metal_kernel() -> Any:
    """Return the serving kernel that omits unused per-chunk end states."""

    mx = require_mlx()
    if not mlx_dplr_metal_available():
        raise RuntimeError("MLX custom Metal kernels are unavailable")
    source = r'''
        uint row_id = thread_position_in_grid.x;
        uint B = uint(dims[0]);
        uint tokens = uint(dims[1]);
        uint heads = uint(dims[2]);
        uint chunks = uint(dims[4]);
        uint total_rows = B * chunks * heads * N;
        if (row_id >= total_rows) {
            return;
        }

        uint row = row_id % N;
        uint head = (row_id / N) % heads;
        uint chunk = (row_id / (N * heads)) % chunks;
        uint batch = row_id / (N * heads * chunks);
        uint start_base = ((((batch * chunks + chunk) * heads + head) * N + row) * N);
        float state_row[N];
        for (uint col = 0; col < N; ++col) {
            state_row[col] = float(start_states[start_base + col]);
        }

        for (uint local = 0; local < C; ++local) {
            uint token = chunk * C + local;
            uint input_base = ((batch * tokens + token) * heads + head) * N;
            float dot_kk = 0.0f;
            for (uint col = 0; col < N; ++col) {
                dot_kk += state_row[col] * float(kk[input_base + col]);
            }

            float output = 0.0f;
            float value_row = float(v[input_base + row]);
            for (uint col = 0; col < N; ++col) {
                float kkv = float(kk[input_base + col]);
                float next = state_row[col] * float(w[input_base + col])
                           - dot_kk * (kkv * float(a[input_base + col]))
                           + value_row * float(k[input_base + col]);
                state_row[col] = next;
                output += next * float(r[input_base + col]);
            }
            uint output_offset = ((batch * tokens + token) * heads + head) * N + row;
            outputs[output_offset] = output;
        }
    '''
    return mx.fast.metal_kernel(
        name="rwkv7_dplr_compact_wy_chunk_apply_output",
        input_names=["r", "w", "k", "v", "kk", "a", "start_states", "dims"],
        output_names=["outputs"],
        source=source,
        ensure_row_contiguous=True,
    )


def _validate_inputs(w: Any, k: Any, v: Any, kk: Any, a: Any, *, chunk_size: int) -> tuple[int, int, int, int, int]:
    try:
        chunk_size_i = int(chunk_size)
    except (TypeError, ValueError) as exc:
        raise TypeError("chunk_size must be an integer") from exc
    if chunk_size_i <= 0:
        raise ValueError("chunk_size must be positive")
    if len(w.shape) != 4:
        raise ValueError("summary inputs must be [batch,tokens,heads,head_dim]")
    shape = tuple(int(value) for value in w.shape)
    for name, value in (("k", k), ("v", v), ("kk", kk), ("a", a)):
        if tuple(int(dim) for dim in value.shape) != shape:
            raise ValueError(f"{name} shape must match w; got {tuple(value.shape)}, expected {shape}")
    batch, tokens, heads, head_dim = shape
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    if tokens % chunk_size_i != 0:
        raise ValueError(f"tokens={tokens} must be divisible by chunk_size={chunk_size_i}")
    return batch, tokens, heads, head_dim, chunk_size_i


def _append_column(mx: Any, factors: Any, column: Any) -> Any:
    return mx.concatenate((factors, column[..., None]), axis=-1)


def _factor_dot(mx: Any, factors: Any, vector: Any) -> Any:
    if int(factors.shape[-1]) == 0:
        return mx.zeros((*vector.shape[:-1], 0), dtype=vector.dtype)
    return mx.sum(factors * vector[..., :, None], axis=-2)


def _factor_weighted_sum(mx: Any, factors: Any, weights: Any) -> Any:
    if int(factors.shape[-1]) == 0:
        return mx.zeros(factors.shape[:-1], dtype=factors.dtype)
    return mx.sum(factors * weights[..., None, :], axis=-1)


def _outer_to_dense(mx: Any, left: Any, right: Any) -> Any:
    if int(left.shape[-1]) == 0:
        n = int(left.shape[-2])
        return mx.zeros((*left.shape[:-1], n), dtype=left.dtype)
    return left @ mx.swapaxes(right, -1, -2)


def mlx_compact_wy_chunk_summary(
    w: Any,
    k: Any,
    v: Any,
    kk: Any,
    a: Any,
    *,
    chunk_size: int = 64,
) -> dict[str, Any]:
    """Build compact per-chunk DPLR affine summaries on MLX.

    Inputs are ``[B,T,H,N]``. Factors are fp32 and shaped
    ``[B,chunks,H,N,chunk_size]``; the transition diagonal is
    ``[B,chunks,H,N]``.
    """

    mx = require_mlx()
    batch, tokens, heads, head_dim, chunk_size_i = _validate_inputs(
        w, k, v, kk, a, chunk_size=chunk_size
    )
    chunks = tokens // chunk_size_i
    diag_rows = []
    trans_left_rows = []
    trans_right_rows = []
    add_left_rows = []
    add_right_rows = []

    for chunk in range(chunks):
        diag = mx.ones((batch, heads, head_dim), dtype=mx.float32)
        empty_shape = (batch, heads, head_dim, 0)
        trans_left = mx.zeros(empty_shape, dtype=mx.float32)
        trans_right = mx.zeros(empty_shape, dtype=mx.float32)
        add_left = mx.zeros(empty_shape, dtype=mx.float32)
        add_right = mx.zeros(empty_shape, dtype=mx.float32)

        start = chunk * chunk_size_i
        for local_index in range(chunk_size_i):
            token_index = start + local_index
            w_i = w[:, token_index].astype(mx.float32)
            k_i = k[:, token_index].astype(mx.float32)
            v_i = v[:, token_index].astype(mx.float32)
            kk_i = kk[:, token_index].astype(mx.float32)
            a_i = a[:, token_index].astype(mx.float32)
            p_i = -kk_i
            q_i = kk_i * a_i

            new_left = diag * p_i + _factor_weighted_sum(
                mx,
                trans_left,
                _factor_dot(mx, trans_right, p_i),
            )
            trans_right = trans_right * w_i[..., None]
            diag = diag * w_i
            trans_left = _append_column(mx, trans_left, new_left)
            trans_right = _append_column(mx, trans_right, q_i)

            if int(add_right.shape[-1]) != 0:
                add_coeff = _factor_dot(mx, add_right, p_i)
                add_right = add_right * w_i[..., None] + q_i[..., None] * add_coeff[..., None, :]
            add_left = _append_column(mx, add_left, v_i)
            add_right = _append_column(mx, add_right, k_i)

        diag_rows.append(diag)
        trans_left_rows.append(trans_left)
        trans_right_rows.append(trans_right)
        add_left_rows.append(add_left)
        add_right_rows.append(add_right)

    return {
        "algorithm": "mlx_compact_wy_summary_reference",
        "chunk_size": chunk_size_i,
        "rank": chunk_size_i,
        "transition_diag": mx.stack(diag_rows, axis=1),
        "transition_left": mx.stack(trans_left_rows, axis=1),
        "transition_right": mx.stack(trans_right_rows, axis=1),
        "additive_left": mx.stack(add_left_rows, axis=1),
        "additive_right": mx.stack(add_right_rows, axis=1),
    }


def mlx_compact_wy_chunk_summary_metal(
    w: Any,
    k: Any,
    v: Any,
    kk: Any,
    a: Any,
    *,
    chunk_size: int = 64,
    implementation: str = "tiled",
) -> dict[str, Any]:
    """Build compact chunk factors in one custom Metal launch.

    The default tiled implementation assigns one threadgroup to each
    ``(batch, chunk, head)`` summary and one thread to each head-dimension row.
    The retained scalar implementation assigns one thread to a whole summary
    and is useful as an independent Metal oracle. Both preserve fp32 factors.
    Current target bounds are ``head_dim<=64`` and ``chunk_size<=64``.
    """

    mx = require_mlx()
    if not mlx_dplr_metal_available():
        raise RuntimeError("MLX custom Metal kernels are unavailable")
    batch, tokens, heads, head_dim, chunk_size_i = _validate_inputs(
        w, k, v, kk, a, chunk_size=chunk_size
    )
    if head_dim > 64 or chunk_size_i > 64:
        raise ValueError("Metal compact summary currently requires head_dim<=64 and chunk_size<=64")
    implementation_i = str(implementation).lower().strip()
    if implementation_i not in {"scalar", "tiled"}:
        raise ValueError("Metal compact summary implementation must be scalar or tiled")
    chunks = tokens // chunk_size_i
    total = batch * chunks * heads
    dims = mx.array([batch, tokens, heads, head_dim, chunks], dtype=mx.uint32)
    factor_shape = (batch, chunks, heads, head_dim, chunk_size_i)
    if implementation_i == "tiled":
        kernel = _compact_wy_summary_tiled_metal_kernel()
        grid = (total * head_dim, 1, 1)
        threadgroup = (head_dim, 1, 1)
    else:
        kernel = _compact_wy_summary_metal_kernel()
        grid = (total, 1, 1)
        threadgroup = (min(256, max(1, total)), 1, 1)
    transition_diag, transition_left, transition_right, additive_left, additive_right = (
        kernel(
            inputs=[w, k, v, kk, a, dims],
            template=[("N", head_dim), ("C", chunk_size_i)],
            grid=grid,
            threadgroup=threadgroup,
            output_shapes=[
                (batch, chunks, heads, head_dim),
                factor_shape,
                factor_shape,
                factor_shape,
                factor_shape,
            ],
            output_dtypes=[mx.float32, mx.float32, mx.float32, mx.float32, mx.float32],
        )
    )
    return {
        "algorithm": f"mlx_metal_compact_wy_summary_{implementation_i}",
        "implementation": implementation_i,
        "chunk_size": chunk_size_i,
        "rank": chunk_size_i,
        "transition_diag": transition_diag,
        "transition_left": transition_left,
        "transition_right": transition_right,
        "additive_left": additive_left,
        "additive_right": additive_right,
    }


def mlx_compact_wy_summary_to_dense(summary: dict[str, Any]) -> dict[str, Any]:
    """Materialize dense transition/additive matrices for parity tests."""

    mx = require_mlx()
    diag = summary["transition_diag"].astype(mx.float32)
    trans_left = summary["transition_left"].astype(mx.float32)
    trans_right = summary["transition_right"].astype(mx.float32)
    add_left = summary["additive_left"].astype(mx.float32)
    add_right = summary["additive_right"].astype(mx.float32)
    if len(diag.shape) != 4:
        raise ValueError("transition_diag must be [B,chunks,H,N]")
    if len(trans_left.shape) != 5 or trans_left.shape != trans_right.shape:
        raise ValueError("transition factors must have matching [B,chunks,H,N,R] shapes")
    if len(add_left.shape) != 5 or add_left.shape != add_right.shape:
        raise ValueError("additive factors must have matching [B,chunks,H,N,R] shapes")
    head_dim = int(diag.shape[-1])
    eye = mx.eye(head_dim, dtype=mx.float32).reshape(1, 1, 1, head_dim, head_dim)
    transition = eye * diag[..., None, :] + _outer_to_dense(mx, trans_left, trans_right)
    additive = _outer_to_dense(mx, add_left, add_right)
    return {
        "algorithm": "mlx_compact_wy_dense_oracle",
        "chunk_size": int(summary.get("chunk_size", 0)),
        "rank": int(summary.get("rank", int(trans_left.shape[-1]))),
        "transition": transition,
        "additive": additive,
    }


def _apply_transition(mx: Any, state: Any, diag: Any, left: Any, right: Any) -> Any:
    out = state.astype(mx.float32) * diag.astype(mx.float32)[..., None, :]
    if int(left.shape[-1]) != 0:
        out = out + (state.astype(mx.float32) @ left.astype(mx.float32)) @ mx.swapaxes(
            right.astype(mx.float32), -1, -2
        )
    return out


def mlx_compact_wy_prefix_combine(state: Any, summary: dict[str, Any]) -> tuple[Any, Any]:
    """Return dense chunk-start states and final state from compact factors."""

    mx = require_mlx()
    if len(state.shape) != 4 or int(state.shape[-1]) != int(state.shape[-2]):
        raise ValueError("state must be [B,H,N,N]")
    diag = summary["transition_diag"]
    trans_left = summary["transition_left"]
    trans_right = summary["transition_right"]
    add_left = summary["additive_left"]
    add_right = summary["additive_right"]
    if len(diag.shape) != 4 or len(trans_left.shape) != 5 or len(add_left.shape) != 5:
        raise ValueError("invalid compact summary rank")
    expected = (int(state.shape[0]), int(state.shape[1]), int(state.shape[2]))
    if (int(diag.shape[0]), int(diag.shape[2]), int(diag.shape[3])) != expected:
        raise ValueError("compact summary shapes must match state")

    cur = state.astype(mx.float32)
    starts = []
    for chunk in range(int(diag.shape[1])):
        starts.append(cur)
        cur = _apply_transition(mx, cur, diag[:, chunk], trans_left[:, chunk], trans_right[:, chunk])
        cur = cur + _outer_to_dense(mx, add_left[:, chunk].astype(mx.float32), add_right[:, chunk].astype(mx.float32))
    return mx.stack(starts, axis=1), cur


def mlx_dplr_recurrent_scan_reference(
    r: Any,
    w: Any,
    k: Any,
    v: Any,
    kk: Any,
    a: Any,
    state: Any,
) -> tuple[Any, Any]:
    """Sequential fp32 DPLR recurrence oracle for ``[B,T,H,N]`` inputs."""

    mx = require_mlx()
    batch, tokens, heads, head_dim, _ = _validate_inputs(w, k, v, kk, a, chunk_size=int(w.shape[1]))
    if tuple(int(dim) for dim in r.shape) != (batch, tokens, heads, head_dim):
        raise ValueError("r shape must match w")
    if tuple(int(dim) for dim in state.shape) != (batch, heads, head_dim, head_dim):
        raise ValueError("state must be [B,H,N,N]")
    cur = state.astype(mx.float32)
    outputs = []
    for token_index in range(tokens):
        r_i = r[:, token_index].astype(mx.float32)
        w_i = w[:, token_index].astype(mx.float32)
        k_i = k[:, token_index].astype(mx.float32)
        v_i = v[:, token_index].astype(mx.float32)
        kk_i = kk[:, token_index].astype(mx.float32)
        a_i = a[:, token_index].astype(mx.float32)
        transition_rank1 = (-kk_i)[..., :, None] @ (kk_i * a_i)[..., None, :]
        additive = v_i[..., :, None] @ k_i[..., None, :]
        cur = cur * w_i[..., None, :] + cur @ transition_rank1 + additive
        outputs.append((cur @ r_i[..., :, None]).reshape(batch, heads, head_dim))
    return mx.stack(outputs, axis=1), cur


def mlx_compact_wy_chunk_apply(
    r: Any,
    w: Any,
    k: Any,
    v: Any,
    kk: Any,
    a: Any,
    start_states: Any,
    *,
    chunk_size: int = 64,
) -> tuple[Any, Any]:
    """Apply each chunk recurrence from its prefix-combined start state."""

    mx = require_mlx()
    batch, tokens, heads, head_dim, chunk_size_i = _validate_inputs(
        w, k, v, kk, a, chunk_size=chunk_size
    )
    if tuple(int(dim) for dim in r.shape) != (batch, tokens, heads, head_dim):
        raise ValueError("r shape must match w")
    chunks = tokens // chunk_size_i
    if tuple(int(dim) for dim in start_states.shape) != (batch, chunks, heads, head_dim, head_dim):
        raise ValueError("start_states must be [B,chunks,H,N,N]")

    outputs = []
    chunk_ends = []
    for chunk in range(chunks):
        cur = start_states[:, chunk].astype(mx.float32)
        for local_index in range(chunk_size_i):
            token_index = chunk * chunk_size_i + local_index
            r_i = r[:, token_index].astype(mx.float32)
            w_i = w[:, token_index].astype(mx.float32)
            k_i = k[:, token_index].astype(mx.float32)
            v_i = v[:, token_index].astype(mx.float32)
            kk_i = kk[:, token_index].astype(mx.float32)
            a_i = a[:, token_index].astype(mx.float32)
            transition_rank1 = (-kk_i)[..., :, None] @ (kk_i * a_i)[..., None, :]
            additive = v_i[..., :, None] @ k_i[..., None, :]
            cur = cur * w_i[..., None, :] + cur @ transition_rank1 + additive
            outputs.append((cur @ r_i[..., :, None]).reshape(batch, heads, head_dim))
        chunk_ends.append(cur)
    return mx.stack(outputs, axis=1), mx.stack(chunk_ends, axis=1)


def mlx_compact_wy_chunk_apply_metal(
    r: Any,
    w: Any,
    k: Any,
    v: Any,
    kk: Any,
    a: Any,
    start_states: Any,
    *,
    chunk_size: int = 64,
) -> tuple[Any, Any]:
    """Run independent chunk recurrence/output in one custom Metal launch."""

    mx = require_mlx()
    if not mlx_dplr_metal_available():
        raise RuntimeError("MLX custom Metal kernels are unavailable")
    batch, tokens, heads, head_dim, chunk_size_i = _validate_inputs(
        w, k, v, kk, a, chunk_size=chunk_size
    )
    if tuple(int(dim) for dim in r.shape) != (batch, tokens, heads, head_dim):
        raise ValueError("r shape must match w")
    chunks = tokens // chunk_size_i
    if tuple(int(dim) for dim in start_states.shape) != (batch, chunks, heads, head_dim, head_dim):
        raise ValueError("start_states must be [B,chunks,H,N,N]")
    if head_dim > 64 or chunk_size_i > 64:
        raise ValueError("Metal chunk apply currently requires head_dim<=64 and chunk_size<=64")
    dims = mx.array([batch, tokens, heads, head_dim, chunks], dtype=mx.uint32)
    total_rows = batch * chunks * heads * head_dim
    outputs, chunk_ends = _compact_wy_chunk_apply_metal_kernel()(
        inputs=[r, w, k, v, kk, a, start_states, dims],
        template=[("N", head_dim), ("C", chunk_size_i)],
        grid=(total_rows, 1, 1),
        threadgroup=(min(256, max(1, total_rows)), 1, 1),
        output_shapes=[
            (batch, tokens, heads, head_dim),
            (batch, chunks, heads, head_dim, head_dim),
        ],
        output_dtypes=[mx.float32, mx.float32],
    )
    return outputs, chunk_ends


def mlx_compact_wy_chunk_apply_output_metal(
    r: Any,
    w: Any,
    k: Any,
    v: Any,
    kk: Any,
    a: Any,
    start_states: Any,
    *,
    chunk_size: int = 64,
) -> Any:
    """Run serving-shaped chunk apply without allocating chunk-end telemetry."""

    mx = require_mlx()
    if not mlx_dplr_metal_available():
        raise RuntimeError("MLX custom Metal kernels are unavailable")
    batch, tokens, heads, head_dim, chunk_size_i = _validate_inputs(
        w, k, v, kk, a, chunk_size=chunk_size
    )
    if tuple(int(dim) for dim in r.shape) != (batch, tokens, heads, head_dim):
        raise ValueError("r shape must match w")
    chunks = tokens // chunk_size_i
    if tuple(int(dim) for dim in start_states.shape) != (batch, chunks, heads, head_dim, head_dim):
        raise ValueError("start_states must be [B,chunks,H,N,N]")
    if head_dim > 64 or chunk_size_i > 64:
        raise ValueError("Metal chunk apply currently requires head_dim<=64 and chunk_size<=64")
    dims = mx.array([batch, tokens, heads, head_dim, chunks], dtype=mx.uint32)
    total_rows = batch * chunks * heads * head_dim
    (outputs,) = _compact_wy_chunk_apply_output_metal_kernel()(
        inputs=[r, w, k, v, kk, a, start_states, dims],
        template=[("N", head_dim), ("C", chunk_size_i)],
        grid=(total_rows, 1, 1),
        threadgroup=(min(256, max(1, total_rows)), 1, 1),
        output_shapes=[(batch, tokens, heads, head_dim)],
        output_dtypes=[mx.float32],
    )
    return outputs


def mlx_compact_wy_three_stage(
    r: Any,
    w: Any,
    k: Any,
    v: Any,
    kk: Any,
    a: Any,
    state: Any,
    *,
    chunk_size: int = 64,
) -> tuple[Any, Any, dict[str, Any]]:
    """Run summary -> prefix combine -> chunk apply/output on MLX."""

    summary = mlx_compact_wy_chunk_summary(w, k, v, kk, a, chunk_size=chunk_size)
    start_states, final_state = mlx_compact_wy_prefix_combine(state, summary)
    outputs, chunk_ends = mlx_compact_wy_chunk_apply(
        r, w, k, v, kk, a, start_states, chunk_size=chunk_size
    )
    return outputs, final_state, {
        "summary": summary,
        "start_states": start_states,
        "chunk_ends": chunk_ends,
    }


def mlx_compact_wy_three_stage_metal(
    r: Any,
    w: Any,
    k: Any,
    v: Any,
    kk: Any,
    a: Any,
    state: Any,
    *,
    chunk_size: int = 64,
    summary_implementation: str = "tiled",
    return_telemetry: bool = True,
) -> tuple[Any, Any, dict[str, Any]]:
    """Run Metal summary -> MLX prefix -> Metal chunk apply/output."""

    summary = mlx_compact_wy_chunk_summary_metal(
        w,
        k,
        v,
        kk,
        a,
        chunk_size=chunk_size,
        implementation=summary_implementation,
    )
    start_states, final_state = mlx_compact_wy_prefix_combine(state, summary)
    if return_telemetry:
        outputs, chunk_ends = mlx_compact_wy_chunk_apply_metal(
            r, w, k, v, kk, a, start_states, chunk_size=chunk_size
        )
        telemetry = {
            "summary": summary,
            "start_states": start_states,
            "chunk_ends": chunk_ends,
            "summary_backend": "metal",
            "summary_implementation": str(summary_implementation),
            "prefix_backend": "mlx",
            "apply_backend": "metal",
        }
    else:
        outputs = mlx_compact_wy_chunk_apply_output_metal(
            r, w, k, v, kk, a, start_states, chunk_size=chunk_size
        )
        telemetry = {
            "summary_backend": "metal",
            "summary_implementation": str(summary_implementation),
            "prefix_backend": "mlx",
            "apply_backend": "metal_output_only",
        }
    return outputs, final_state, telemetry


__all__ = [
    "mlx_compact_wy_chunk_apply",
    "mlx_compact_wy_chunk_apply_metal",
    "mlx_compact_wy_chunk_apply_output_metal",
    "mlx_compact_wy_chunk_summary",
    "mlx_compact_wy_chunk_summary_metal",
    "mlx_compact_wy_prefix_combine",
    "mlx_compact_wy_summary_to_dense",
    "mlx_compact_wy_three_stage",
    "mlx_compact_wy_three_stage_metal",
    "mlx_dplr_recurrent_scan_reference",
    "mlx_dplr_metal_available",
]
