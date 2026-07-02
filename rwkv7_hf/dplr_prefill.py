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

``dplr_chunk_scan`` keeps the original correctness-first sequential path as the
default.  It also exposes an experimental ``algorithm="affine"`` path that
materializes the per-token affine transitions and composes chunk prefix/suffix
transforms in torch.  That affine path is intentionally O(T*N^3): it is a
correctness prototype and the marked replacement point for a future WY/Triton
implementation.

The newer experimental ``algorithm="lowrank"`` / ``algorithm="wy"`` path keeps
chunk affine products as

    A_{0:i} = diag(d_i) + U_i V_i^T

and keeps the additive chunk term as ``X_i Y_i^T``.  It is still a pure torch
correctness prototype, but it avoids explicitly constructing each dense
``A_t`` and avoids dense ``N x N`` by ``N x N`` transition products.
"""
from __future__ import annotations

import os
from typing import Any

try:  # pragma: no cover - exercised in lightweight environments without torch
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


__all__ = ["dplr_chunk_scan", "lowrank_chunk_summary"]


_ALGORITHM_ENV = "RWKV7_DPLR_PREFILL_ALGORITHM"
_SUPPORTED_ALGORITHMS = ("sequential", "affine", "lowrank", "wy", "triton_wy", "cuda_wy", "triton_dense3")


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


def _resolve_algorithm(algorithm: Any) -> str:
    """Normalize the scan algorithm selector.

    Passing ``None`` (the public default) reads
    ``RWKV7_DPLR_PREFILL_ALGORITHM`` and falls back to ``"sequential"`` when it
    is unset, so existing callers keep the same behavior unless they opt in.
    """

    if algorithm is None:
        algorithm = os.environ.get(_ALGORITHM_ENV, "sequential")
    if not isinstance(algorithm, str):
        raise TypeError(
            "algorithm must be 'sequential', 'affine', 'lowrank', 'wy', "
            "'triton_wy', 'cuda_wy', 'triton_dense3', or None"
        )
    normalized = algorithm.strip().lower()
    if normalized not in _SUPPORTED_ALGORITHMS:
        supported = "', '".join(_SUPPORTED_ALGORITHMS)
        raise ValueError(f"algorithm must be one of '{supported}', got {algorithm!r}")
    return normalized


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


def _dplr_affine_transition(w_t: Any, k_t: Any, v_t: Any, kk_t: Any, a_t: Any):
    """Build one explicit affine transition ``S_t = S_{t-1} A_t + B_t``.

    Shapes are ``[B,H,N]`` for inputs and ``[B,H,N,N]`` for returned matrices.
    This intentionally materializes the dense ``A_t`` matrix even though it is
    diagonal-plus-rank-1; the future optimized path should replace this with a
    compact WY-style factorization and/or a Triton kernel.
    """

    B, H, N = (int(w_t.shape[0]), int(w_t.shape[1]), int(w_t.shape[2]))
    diag_w = torch.diag_embed(w_t)
    rank1 = (-kk_t).view(B, H, N, 1) @ (kk_t * a_t).view(B, H, 1, N)
    affine_a = diag_w + rank1
    affine_b = v_t.view(B, H, N, 1) @ k_t.view(B, H, 1, N)
    return affine_a, affine_b


def _factor_dot(factors: Any, vec: Any):
    """Return ``factors^T vec`` for factors shaped ``[B,H,N,R]``."""

    if int(factors.shape[-1]) == 0:
        return vec.new_zeros((*vec.shape[:-1], 0))
    return torch.einsum("bhnr,bhn->bhr", factors, vec)


def _factor_weighted_sum(factors: Any, weights: Any):
    """Return ``factors weights`` for factors shaped ``[B,H,N,R]``."""

    if int(factors.shape[-1]) == 0:
        return factors.new_zeros(factors.shape[:-1])
    return torch.einsum("bhnr,bhr->bhn", factors, weights)


def _append_factor_column(factors: Any, col: Any):
    return torch.cat((factors, col.unsqueeze(-1)), dim=-1)


def _lowrank_apply_transition_to_vector(diag: Any, left: Any, right: Any, vec: Any):
    """Apply ``diag(diag) + left right^T`` to ``vec`` without forming it."""

    out = diag * vec
    if int(left.shape[-1]) != 0:
        out = out + _factor_weighted_sum(left, _factor_dot(right, vec))
    return out


def _lowrank_apply_outer_to_vector(left: Any, right: Any, vec: Any):
    """Apply ``left right^T`` to ``vec`` without forming the dense matrix."""

    return _factor_weighted_sum(left, _factor_dot(right, vec))


def _lowrank_apply_transition_to_state(state: Any, diag: Any, left: Any, right: Any):
    """Apply ``state @ (diag(diag) + left right^T)`` without dense N^3 work."""

    out = state * diag.unsqueeze(-2)
    if int(left.shape[-1]) != 0:
        out = out + (state @ left) @ right.transpose(-1, -2)
    return out


def _lowrank_outer_to_dense(left: Any, right: Any):
    if int(left.shape[-1]) == 0:
        return left.new_zeros((*left.shape[:-1], int(left.shape[-2])))
    return left @ right.transpose(-1, -2)


def _check_chunk_tensor_compat(name: str, x: Any, *, shape: Any, device: Any) -> None:
    if tuple(x.shape) != tuple(shape):
        raise ValueError(f"{name} chunk shape must match w; got {tuple(x.shape)}")
    if x.device != device:
        raise ValueError(f"{name} chunk must be on the same device as w")
    if not x.is_floating_point():
        raise TypeError(f"{name} chunk must be a floating point tensor")


def lowrank_chunk_summary(w: Any, k: Any, v: Any, kk: Any, a: Any, *, include_prefix: bool = True):
    """Return diagonal-plus-low-rank metadata for one BTHN chunk.

    Inputs must be normalized chunk tensors shaped ``[B, L, H, N]``.  For token
    ``i`` in the chunk define

    ``A_i = diag(w_i) + p_i q_i^T``, with ``p_i = -kk_i`` and
    ``q_i = kk_i * a_i``, and ``B_i = v_i k_i^T``.

    The returned summary stores the full chunk transition and additive term as

    ``A_0 ... A_{L-1} = diag(d) + U V^T``
    ``sum_j B_j A_{j+1} ... A_{L-1} = X Y^T``

    plus optional inclusive prefix summaries.  The update rules never build a
    dense ``A_i`` and never multiply two dense ``N x N`` transition matrices;
    rank grows by one per token, so the pure torch prototype uses O(L^2*N)
    metadata work plus O(L*N^2) state application at chunk boundaries.
    """

    _require_torch()
    if not hasattr(w, "dim"):
        raise TypeError("w chunk must be a torch.Tensor")
    if w.dim() != 4:
        raise ValueError("chunk tensors must be shaped [batch, chunk_tokens, heads, head_dim]")
    if not w.is_floating_point():
        raise TypeError("w chunk must be a floating point tensor")

    B, L, H, N = (int(vv) for vv in w.shape)
    shape = tuple(w.shape)
    device = w.device
    for name, x in (("k", k), ("v", v), ("kk", kk), ("a", a)):
        if not hasattr(x, "dim"):
            raise TypeError(f"{name} chunk must be a torch.Tensor")
        _check_chunk_tensor_compat(name, x, shape=shape, device=device)

    diag = w.new_ones((B, H, N))
    trans_left = w.new_empty((B, H, N, 0))
    trans_right = w.new_empty((B, H, N, 0))
    add_left = w.new_empty((B, H, N, 0))
    add_right = w.new_empty((B, H, N, 0))
    prefixes = []

    for i in range(L):
        w_i = w[:, i]
        key_i = k[:, i]
        val_i = v[:, i]
        p_i = -kk[:, i]
        q_i = kk[:, i] * a[:, i]

        # If P = diag(d) + U V^T is the previous transition product, then
        # P (diag(w_i) + p_i q_i^T) =
        #   diag(d*w_i) + U (diag(w_i)V)^T + (P p_i) q_i^T.
        new_left_col = diag * p_i + _factor_weighted_sum(trans_left, _factor_dot(trans_right, p_i))
        trans_right = trans_right * w_i.unsqueeze(-1)
        diag = diag * w_i
        trans_left = _append_factor_column(trans_left, new_left_col)
        trans_right = _append_factor_column(trans_right, q_i)

        # If Q = X Y^T is the previous additive term, then
        # Q (diag(w_i) + p_i q_i^T) =
        #   X (diag(w_i)Y + q_i(Y^T p_i)^T)^T.
        if int(add_right.shape[-1]) != 0:
            add_coeff = _factor_dot(add_right, p_i)
            add_right = add_right * w_i.unsqueeze(-1) + q_i.unsqueeze(-1) * add_coeff.unsqueeze(-2)
        add_left = _append_factor_column(add_left, val_i)
        add_right = _append_factor_column(add_right, key_i)

        if include_prefix:
            prefixes.append(
                {
                    "length": i + 1,
                    "transition_diag": diag,
                    "transition_left": trans_left,
                    "transition_right": trans_right,
                    "additive_left": add_left,
                    "additive_right": add_right,
                }
            )

    return {
        "algorithm": "lowrank-wy",
        "length": L,
        "rank": int(trans_left.shape[-1]),
        "transition_diag": diag,
        "transition_left": trans_left,
        "transition_right": trans_right,
        "additive_left": add_left,
        "additive_right": add_right,
        "prefix": tuple(prefixes),
    }


def _identity_affine(B: int, H: int, N: int, *, device: Any, dtype: Any):
    eye = torch.eye(N, device=device, dtype=dtype).view(1, 1, N, N)
    return eye.expand(B, H, N, N).clone()


def _compose_prefix_affines(a_terms: Any, b_terms: Any, identity: Any, zero: Any):
    """Return inclusive chunk prefixes for ``S_i = S_start P_i + Q_i``."""

    prefix_a = []
    prefix_b = []
    cur_a = identity
    cur_b = zero
    for a_t, b_t in zip(a_terms, b_terms):
        cur_a = cur_a @ a_t
        cur_b = cur_b @ a_t + b_t
        prefix_a.append(cur_a)
        prefix_b.append(cur_b)
    return prefix_a, prefix_b


def _compose_suffix_affines(a_terms: Any, b_terms: Any, identity: Any, zero: Any):
    """Return transforms from each token state to the chunk end.

    ``suffix_a[i]``/``suffix_b[i]`` satisfy
    ``S_chunk_end = S_after_token_i suffix_a[i] + suffix_b[i]``.  The current
    prototype builds these explicitly to make the affine chunk contract clear;
    later WY/Triton work can use the same boundary but avoid dense O(N^3)
    suffix composition.
    """

    L = len(a_terms)
    suffix_a = [identity for _ in range(L)]
    suffix_b = [zero for _ in range(L)]
    cur_a = identity
    cur_b = zero
    for i in range(L - 1, -1, -1):
        suffix_a[i] = cur_a
        suffix_b[i] = cur_b
        a_t = a_terms[i]
        b_t = b_terms[i]
        cur_b = b_t @ cur_a + cur_b
        cur_a = a_t @ cur_a
    return suffix_a, suffix_b


def _dplr_chunk_scan_sequential(r32: Any, w32: Any, k32: Any, v32: Any, kk32: Any, a32: Any, state32: Any, chunk_size: int):
    """Original correctness-first chunk loop: sequential scan inside chunks."""

    T = int(r32.shape[1])
    cur_state = state32
    outs = []
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)

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
    return outs, cur_state


def _dplr_chunk_scan_affine(r32: Any, w32: Any, k32: Any, v32: Any, kk32: Any, a32: Any, state32: Any, chunk_size: int):
    """Experimental dense affine chunk prototype.

    For every chunk this path explicitly builds per-token
    ``A_t = diag(w_t) + (-kk_t)(kk_t*a_t)^T`` and ``B_t = v_t k_t^T``, composes
    inclusive prefix transforms, and computes every token state from the chunk
    start state.  Complexity is O(T*N^3) because transforms are dense matrices;
    the marked construction/composition loops are the intended replacement
    points for a future WY representation and Triton/CUDA implementation.
    """

    B, T, H, N = (int(r32.shape[0]), int(r32.shape[1]), int(r32.shape[2]), int(r32.shape[3]))
    cur_state = state32
    outs = []
    identity = _identity_affine(B, H, N, device=r32.device, dtype=r32.dtype)
    zero = r32.new_zeros((B, H, N, N))

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        chunk_start_state = cur_state

        # Future WY/Triton hook:
        #   Replace these dense per-token A_t/B_t materializations with a
        #   diagonal-plus-low-rank chunk composer.  The public contract should
        #   still expose prefix states for outputs and a chunk-end transform.
        a_terms = []
        b_terms = []
        for t in range(start, end):
            a_t, b_t = _dplr_affine_transition(w32[:, t], k32[:, t], v32[:, t], kk32[:, t], a32[:, t])
            a_terms.append(a_t)
            b_terms.append(b_t)

        prefix_a, prefix_b = _compose_prefix_affines(a_terms, b_terms, identity, zero)

        # Suffix transforms are not needed for the scalar output below, but are
        # intentionally constructed to pin down the chunk affine prototype API:
        # each token's post-update state can be mapped to the chunk-end state.
        # A production WY/Triton path should generate equivalent suffix/prefix
        # metadata without dense O(N^3) matrices.
        suffix_a, suffix_b = _compose_suffix_affines(a_terms, b_terms, identity, zero)

        for local_i, t in enumerate(range(start, end)):
            state_t = chunk_start_state @ prefix_a[local_i] + prefix_b[local_i]
            out_t = state_t @ r32[:, t].view(B, H, N, 1)
            outs.append(out_t.view(B, H, N))

        # Use the inclusive full-prefix transform for the next chunk.  The
        # suffix transform gives the same mathematical chunk-end state from any
        # token state; it is built above as a correctness-oriented prototype
        # artifact for later parallelization.
        _ = suffix_a, suffix_b
        cur_state = chunk_start_state @ prefix_a[-1] + prefix_b[-1]

    return outs, cur_state


def _dplr_chunk_scan_lowrank(r32: Any, w32: Any, k32: Any, v32: Any, kk32: Any, a32: Any, state32: Any, chunk_size: int):
    """Experimental diagonal-plus-low-rank/WY chunk prototype.

    For a chunk start state ``S`` and inclusive prefix metadata
    ``P_i = diag(d_i) + U_i V_i^T`` and ``Q_i = X_i Y_i^T``, token outputs are

        ``out_i = S @ (P_i r_i) + Q_i @ r_i``.

    The chunk-end state is

        ``S_end = S @ diag(d) + (S @ U) V^T + X Y^T``.

    This is still a correctness prototype (rank grows with chunk length), but
    unlike the dense affine path it never constructs per-token dense
    ``A_i`` matrices and never composes dense ``N x N`` transitions.
    """

    B, T, H, N = (int(r32.shape[0]), int(r32.shape[1]), int(r32.shape[2]), int(r32.shape[3]))
    cur_state = state32
    outs = []

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        chunk_start_state = cur_state
        summary = lowrank_chunk_summary(
            w32[:, start:end],
            k32[:, start:end],
            v32[:, start:end],
            kk32[:, start:end],
            a32[:, start:end],
            include_prefix=True,
        )

        for local_i, t in enumerate(range(start, end)):
            prefix = summary["prefix"][local_i]
            r_t = r32[:, t]
            pr_t = _lowrank_apply_transition_to_vector(
                prefix["transition_diag"],
                prefix["transition_left"],
                prefix["transition_right"],
                r_t,
            )
            out_t = chunk_start_state @ pr_t.view(B, H, N, 1)
            out_t = out_t.view(B, H, N) + _lowrank_apply_outer_to_vector(
                prefix["additive_left"],
                prefix["additive_right"],
                r_t,
            )
            outs.append(out_t)

        cur_state = _lowrank_apply_transition_to_state(
            chunk_start_state,
            summary["transition_diag"],
            summary["transition_left"],
            summary["transition_right"],
        )
        cur_state = cur_state + _lowrank_outer_to_dense(summary["additive_left"], summary["additive_right"])

    return outs, cur_state


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
    algorithm: Any = None,
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
    algorithm:
        ``"sequential"`` preserves the original chunked reference behavior.
        ``"affine"`` enables an experimental dense affine chunk prototype that
        explicitly composes per-token ``A_t``/``B_t`` transforms.  ``"lowrank"``
        and ``"wy"`` enable the experimental diagonal-plus-low-rank chunk
        summary path from ``lowrank_chunk_summary``. ``"triton_wy"`` and
        ``"cuda_wy"`` dispatch to the opt-in fused recurrent compiled prototype
        in ``dplr_prefill_triton``.  ``"triton_dense3"`` dispatches to the
        explicit dense three-stage Triton scaffold (summary -> prefix ->
        chunk-apply) for synthetic benchmarking. ``None`` (the default) reads
        ``RWKV7_DPLR_PREFILL_ALGORITHM`` and falls back to ``"sequential"``
        when the environment variable is unset.

    Returns
    -------
    (out, final_state):
        ``out`` has the same shape style and dtype as ``r``. ``final_state`` has
        shape ``[B, H, N, N]`` and is cast back to the input state's dtype after
        fp32 accumulation for fp16/bf16 inputs.
    """

    _require_torch()
    chunk_size_i = _validate_chunk_size(chunk_size)
    algorithm_i = _resolve_algorithm(algorithm)
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

    if algorithm_i == "triton_dense3":
        try:
            from .dplr_prefill_triton import dplr_dense_three_stage_triton
        except Exception:  # pragma: no cover - direct remote-file execution fallback
            try:
                from dplr_prefill_triton import dplr_dense_three_stage_triton  # type: ignore[no-redef]
            except Exception as exc:
                raise RuntimeError("triton_dense3 requested but dplr_prefill_triton is unavailable") from exc
        return dplr_dense_three_stage_triton(
            r,
            w,
            k,
            v,
            kk,
            a,
            state,
            chunk_size=chunk_size_i,
            force_fallback=force_fallback,
        )

    if algorithm_i in {"triton_wy", "cuda_wy"}:
        try:
            from .dplr_prefill_triton import dplr_chunk_scan_triton
        except Exception:  # pragma: no cover - direct remote-file execution fallback
            try:
                from dplr_prefill_triton import dplr_chunk_scan_triton  # type: ignore[no-redef]
            except Exception as exc:
                raise RuntimeError("triton_wy requested but dplr_prefill_triton is unavailable") from exc
        return dplr_chunk_scan_triton(
            r,
            w,
            k,
            v,
            kk,
            a,
            state,
            chunk_size=chunk_size_i,
            force_fallback=force_fallback,
        )

    # Correctness/stability policy for the reference prototype: fp16/bf16 (and
    # fp32) scan in fp32, then cast public outputs back.  This mirrors the
    # native reference formula's explicit float accumulation for rank-1 terms.
    r32, w32, k32, v32, kk32, a32 = (x.to(dtype=torch.float32) for x in tensors)
    cur_state = state.to(dtype=torch.float32)

    if T == 0:
        empty = r4.new_empty((B, 0, H, N), dtype=out_dtype)
        return (empty.reshape(B, 0, H * N) if flat else empty), cur_state.to(dtype=state_dtype)

    if algorithm_i == "sequential":
        outs, cur_state = _dplr_chunk_scan_sequential(r32, w32, k32, v32, kk32, a32, cur_state, chunk_size_i)
    elif algorithm_i == "affine":
        outs, cur_state = _dplr_chunk_scan_affine(r32, w32, k32, v32, kk32, a32, cur_state, chunk_size_i)
    else:
        # ``wy`` is an alias for the same diagonal-plus-low-rank product
        # metadata used by ``lowrank``.  It is a true low-rank prototype, not a
        # fallback to the dense affine implementation.
        outs, cur_state = _dplr_chunk_scan_lowrank(r32, w32, k32, v32, kk32, a32, cur_state, chunk_size_i)

    stacked = torch.stack(outs, dim=1).to(dtype=out_dtype)
    if flat:
        stacked = stacked.reshape(B, T, H * N)
    return stacked, cur_state.to(dtype=state_dtype)
