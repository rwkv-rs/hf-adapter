# coding=utf-8
"""Optional fused FFN kernels for RWKV-7 decode and sequence prefill.

The prototype keeps the HF model path unchanged.  It combines FFN shift-mix,
key projection, and relu² activation in one Triton launch, then computes the
value projection in a second launch.  Benchmarks decide whether this should be
integrated behind ``rwkv7_forward_token`` later.

The sequence path uses tensor-core tiled matrix multiplication rather than the
one-token GEMV kernels.  A single pointwise kernel performs temporal shift-mix,
the key GEMM applies relu² in its epilogue, and the value GEMM projects the
activated rows back to the model width.  This removes the full
``[B,T,4H]`` pointwise round-trip while retaining fp32 accumulation.
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
    def _sequence_ffn_shift_kernel(
        x_ptr,
        prev_ptr,
        mix_ptr,
        shifted_ptr,
        M: tl.constexpr,
        T: tl.constexpr,
        K: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        total = M * K
        mask = offs < total
        row = offs // K
        col = offs % K
        token = row % T
        batch = row // T
        cur = tl.load(x_ptr + offs, mask=mask, other=0.0)
        prev0 = tl.load(
            prev_ptr + batch * K + col,
            mask=mask & (token == 0),
            other=0.0,
        )
        prevn = tl.load(
            x_ptr + (row - 1) * K + col,
            mask=mask & (token != 0),
            other=0.0,
        )
        prev = tl.where(token == 0, prev0, prevn)
        mix = tl.load(mix_ptr + col, mask=mask, other=0.0)
        tl.store(shifted_ptr + offs, cur + (prev - cur) * mix, mask=mask)

    @triton.jit
    def _sequence_ffn_key_relu_kernel(
        shifted_ptr,
        key_weight_ptr,
        mid_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        """Key tensor-core GEMM + relu² for flattened sequence rows."""

        pid = tl.program_id(0)
        grid_m = tl.cdiv(M, BLOCK_M)
        grid_n = tl.cdiv(N, BLOCK_N)
        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // group_size

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        mask_m = offs_m < M
        mask_n = offs_n < N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for start in range(0, K, BLOCK_K):
            kidx = start + offs_k
            mask_k = kidx < K
            shifted = tl.load(
                shifted_ptr + offs_m[:, None] * K + kidx[None, :],
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            weight = tl.load(
                key_weight_ptr + offs_n[None, :] * K + kidx[:, None],
                mask=mask_k[:, None] & mask_n[None, :],
                other=0.0,
            )
            acc += tl.dot(shifted, weight)

        activated = tl.maximum(acc, 0.0)
        activated *= activated
        tl.store(
            mid_ptr + offs_m[:, None] * N + offs_n[None, :],
            activated,
            mask=mask_m[:, None] & mask_n[None, :],
        )

    @triton.jit
    def _sequence_ffn_value_kernel(
        mid_ptr,
        value_weight_ptr,
        out_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        """Tensor-core value projection for flattened sequence rows."""

        pid = tl.program_id(0)
        grid_m = tl.cdiv(M, BLOCK_M)
        grid_n = tl.cdiv(N, BLOCK_N)
        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // group_size

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        mask_m = offs_m < M
        mask_n = offs_n < N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for start in range(0, K, BLOCK_K):
            kidx = start + offs_k
            mask_k = kidx < K
            lhs = tl.load(
                mid_ptr + offs_m[:, None] * K + kidx[None, :],
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            )
            weight = tl.load(
                value_weight_ptr + offs_n[None, :] * K + kidx[:, None],
                mask=mask_k[:, None] & mask_n[None, :],
                other=0.0,
            )
            acc += tl.dot(lhs, weight)

        tl.store(
            out_ptr + offs_m[:, None] * N + offs_n[None, :],
            acc,
            mask=mask_m[:, None] & mask_n[None, :],
        )

    @triton.jit
    def _ffn_key_relu_kernel(
        x_ptr,
        prev_ptr,
        mix_ptr,
        key_weight_ptr,
        mid_ptr,
        hidden: tl.constexpr,
        intermediate: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        block_id = tl.program_id(1)
        offs_m = block_id * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_K)
        mask_m = offs_m < intermediate
        acc = tl.zeros((BLOCK_M,), tl.float32)
        for start in range(0, hidden, BLOCK_K):
            kidx = start + offs_k
            mask_k = kidx < hidden
            x = tl.load(x_ptr + batch_id * hidden + kidx, mask=mask_k, other=0.0).to(tl.float32)
            prev = tl.load(prev_ptr + batch_id * hidden + kidx, mask=mask_k, other=0.0).to(tl.float32)
            mix = tl.load(mix_ptr + kidx, mask=mask_k, other=0.0).to(tl.float32)
            shifted = x + (prev - x) * mix
            w = tl.load(key_weight_ptr + offs_m[:, None] * hidden + kidx[None, :], mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
            acc += tl.sum(w * shifted[None, :], axis=1)
        relu = tl.maximum(acc, 0.0)
        tl.store(mid_ptr + batch_id * intermediate + offs_m, relu * relu, mask=mask_m)

    @triton.jit
    def _ffn_value_kernel(
        mid_ptr,
        value_weight_ptr,
        out_ptr,
        hidden: tl.constexpr,
        intermediate: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        block_id = tl.program_id(1)
        offs_m = block_id * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_K)
        mask_m = offs_m < hidden
        acc = tl.zeros((BLOCK_M,), tl.float32)
        for start in range(0, intermediate, BLOCK_K):
            kidx = start + offs_k
            mask_k = kidx < intermediate
            h = tl.load(mid_ptr + batch_id * intermediate + kidx, mask=mask_k, other=0.0).to(tl.float32)
            w = tl.load(value_weight_ptr + offs_m[:, None] * intermediate + kidx[None, :], mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)
            acc += tl.sum(w * h[None, :], axis=1)
        tl.store(out_ptr + batch_id * hidden + offs_m, acc, mask=mask_m)


def fused_ffn_available() -> bool:
    """Return whether the optional Triton FFN prototype can run."""

    return bool(_HAS_TRITON and torch is not None)


def fused_sequence_ffn_available() -> bool:
    """Return whether the tensor-core sequence FFN path can run."""

    return bool(_HAS_TRITON and torch is not None)


def fused_sequence_ffn(
    hidden_states: Any,
    prev_states: Any,
    mix_x: Any,
    key_weight: Any,
    value_weight: Any,
    *,
    block_m: int = 128,
    block_n: int = 128,
    key_block_k: int = 32,
    value_block_k: int = 64,
    group_m: int = 8,
    num_stages: int = 3,
    num_warps: int = 4,
    workspace: tuple[Any, Any] | None = None,
    force_fallback: bool = False,
):
    """Run exact sequence FFN with shift-mix fused into the key GEMM.

    ``hidden_states`` is ``[B,T,H]`` and ``prev_states`` is ``[B,H]``.  The
    return value is ``(output, next_state)`` with shapes ``[B,T,H]`` and
    ``[B,H]``. Unsupported devices or dtypes retain the ordinary PyTorch
    expression, making this helper safe to call behind an exact-card policy.
    """

    if torch is None or F is None:
        raise RuntimeError("fused_sequence_ffn requires torch")
    if int(num_warps) not in {1, 2, 4, 8}:
        raise ValueError("num_warps must be one of 1, 2, 4, or 8")
    if int(num_stages) not in {1, 2, 3, 4, 5}:
        raise ValueError("num_stages must be between 1 and 5")
    if hidden_states.dim() != 3:
        raise ValueError(f"hidden_states must be [B,T,H], got {tuple(hidden_states.shape)}")
    batch, tokens, hidden = (int(v) for v in hidden_states.shape)
    if prev_states.dim() == 3 and int(prev_states.shape[1]) == 1:
        prev_states = prev_states[:, 0, :]
    if prev_states.dim() != 2 or tuple(prev_states.shape) != (batch, hidden):
        raise ValueError(f"prev_states must be [{batch},{hidden}], got {tuple(prev_states.shape)}")
    mix = mix_x.reshape(-1)
    if int(mix.numel()) != hidden:
        raise ValueError(f"mix_x must have {hidden} elements, got {int(mix.numel())}")
    if key_weight.dim() != 2 or int(key_weight.shape[1]) != hidden:
        raise ValueError(f"key_weight must be [intermediate,{hidden}], got {tuple(key_weight.shape)}")
    intermediate = int(key_weight.shape[0])
    if value_weight.dim() != 2 or tuple(value_weight.shape) != (hidden, intermediate):
        raise ValueError(f"value_weight must be [{hidden},{intermediate}], got {tuple(value_weight.shape)}")

    rows = batch * tokens
    use_triton = (
        not force_fallback
        and fused_sequence_ffn_available()
        and hidden_states.is_cuda
        and prev_states.is_cuda
        and mix.is_cuda
        and key_weight.is_cuda
        and value_weight.is_cuda
        and hidden_states.dtype in (torch.float16, torch.bfloat16)
        and prev_states.dtype == hidden_states.dtype
        and mix.dtype == hidden_states.dtype
        and key_weight.dtype == hidden_states.dtype
        and value_weight.dtype == hidden_states.dtype
        and rows >= 128
        and hidden % int(key_block_k) == 0
        and intermediate % int(value_block_k) == 0
    )
    if not use_triton:
        prev_seq = torch.cat([prev_states[:, None, :], hidden_states[:, :-1, :]], dim=1)
        shifted = hidden_states + (prev_seq - hidden_states) * mix.view(1, 1, hidden)
        mid = torch.relu(F.linear(shifted, key_weight)) ** 2
        return F.linear(mid, value_weight), hidden_states[:, -1, :].contiguous()

    x = hidden_states.contiguous()
    prev = prev_states.contiguous()
    mix = mix.contiguous()
    key = key_weight.contiguous()
    value = value_weight.contiguous()
    if workspace is None:
        shifted = torch.empty((rows, hidden), device=x.device, dtype=x.dtype)
        mid = torch.empty((rows, intermediate), device=x.device, dtype=x.dtype)
    else:
        shifted, mid = workspace
        if tuple(shifted.shape) != (rows, hidden) or shifted.device != x.device or shifted.dtype != x.dtype:
            raise ValueError("sequence FFN shifted workspace has the wrong shape/device/dtype")
        if tuple(mid.shape) != (rows, intermediate) or mid.device != x.device or mid.dtype != x.dtype:
            raise ValueError("sequence FFN intermediate workspace has the wrong shape/device/dtype")
    _sequence_ffn_shift_kernel[(triton.cdiv(rows * hidden, 256),)](
        x,
        prev,
        mix,
        shifted,
        rows,
        tokens,
        hidden,
        BLOCK=256,
        num_warps=4,
    )
    grid_key = (triton.cdiv(rows, int(block_m)) * triton.cdiv(intermediate, int(block_n)),)
    _sequence_ffn_key_relu_kernel[grid_key](
        shifted,
        key,
        mid,
        rows,
        intermediate,
        hidden,
        BLOCK_M=int(block_m),
        BLOCK_N=int(block_n),
        BLOCK_K=int(key_block_k),
        GROUP_M=int(group_m),
        num_stages=int(num_stages),
        num_warps=int(num_warps),
    )
    # The value projection no longer needs the shifted activation, so reuse
    # that allocation for the final output just like the optimized compiled
    # schedule. Stream ordering keeps the overwrite after the key GEMM.
    out = shifted
    grid_value = (triton.cdiv(rows, int(block_m)) * triton.cdiv(hidden, int(block_n)),)
    _sequence_ffn_value_kernel[grid_value](
        mid,
        value,
        out,
        rows,
        hidden,
        intermediate,
        BLOCK_M=int(block_m),
        BLOCK_N=int(block_n),
        BLOCK_K=int(value_block_k),
        GROUP_M=int(group_m),
        num_stages=int(num_stages),
        num_warps=int(num_warps),
    )
    return out.view(batch, tokens, hidden), hidden_states[:, -1, :].contiguous()


def _flatten(x: Any, hidden: int | None = None, *, name: str):
    if torch is None:
        raise RuntimeError("fused_ffn requires torch")
    if x.dim() == 3:
        if int(x.shape[1]) != 1:
            raise ValueError(f"{name} must be [batch, 1, hidden] or [batch, hidden], got {tuple(x.shape)}")
        if hidden is not None and int(x.shape[2]) != hidden:
            raise ValueError(f"{name} hidden mismatch: got {int(x.shape[2])}, expected {hidden}")
        return x.reshape(int(x.shape[0]), int(x.shape[2])), True
    if x.dim() == 2:
        if hidden is not None and int(x.shape[1]) != hidden:
            raise ValueError(f"{name} hidden mismatch: got {int(x.shape[1])}, expected {hidden}")
        return x, False
    raise ValueError(f"{name} must be [batch, 1, hidden] or [batch, hidden]")


def fused_ffn(
    hidden_states: Any,
    prev_states: Any,
    mix_x: Any,
    key_weight: Any,
    value_weight: Any,
    *,
    block_m: int = 64,
    block_k: int = 64,
    force_fallback: bool = False,
):
    """Compute RWKV-7 FFN one-token output and next FFN state.

    Args mirror the FFN decode expression:

    ``k = hidden + (prev - hidden) * mix_x``
    ``out = value(relu(key(k)) ** 2)``

    Returns ``(out, next_state)``.  ``next_state`` is the original hidden input,
    preserving the same rank/layout as ``hidden_states``.
    """

    if torch is None or F is None:
        raise RuntimeError("fused_ffn requires torch")
    x2, had_seq = _flatten(hidden_states, name="hidden_states")
    hidden = int(x2.shape[1])
    prev2, prev_had_seq = _flatten(prev_states, hidden, name="prev_states")
    if prev_had_seq != had_seq or tuple(prev2.shape) != tuple(x2.shape):
        raise ValueError("hidden_states and prev_states must have identical flattened shape/layout")
    if mix_x.dim() not in (1, 2, 3):
        raise ValueError("mix_x must be broadcastable to hidden")
    mix = mix_x.reshape(-1)
    if int(mix.shape[0]) != hidden:
        raise ValueError(f"mix_x must have {hidden} elements, got {int(mix.shape[0])}")
    if key_weight.dim() != 2 or int(key_weight.shape[1]) != hidden:
        raise ValueError(f"key_weight must be [intermediate, {hidden}], got {tuple(key_weight.shape)}")
    intermediate = int(key_weight.shape[0])
    if value_weight.dim() != 2 or int(value_weight.shape[0]) != hidden or int(value_weight.shape[1]) != intermediate:
        raise ValueError(f"value_weight must be [{hidden}, {intermediate}], got {tuple(value_weight.shape)}")

    use_triton = (
        not force_fallback
        and fused_ffn_available()
        and x2.is_cuda
        and prev2.is_cuda
        and mix.is_cuda
        and key_weight.is_cuda
        and value_weight.is_cuda
        and x2.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and prev2.dtype == x2.dtype
        and mix.dtype == x2.dtype
        and key_weight.dtype == x2.dtype
        and value_weight.dtype == x2.dtype
    )
    if not use_triton:
        shifted = x2 + (prev2 - x2) * mix.view(1, -1)
        mid = torch.relu(F.linear(shifted, key_weight)) ** 2
        out = F.linear(mid, value_weight)
    else:
        batch = int(x2.shape[0])
        x_c = x2.contiguous()
        prev_c = prev2.contiguous()
        mix_c = mix.contiguous()
        key_c = key_weight.contiguous()
        value_c = value_weight.contiguous()
        mid = torch.empty((batch, intermediate), device=x2.device, dtype=x2.dtype)
        out = torch.empty((batch, hidden), device=x2.device, dtype=x2.dtype)
        _ffn_key_relu_kernel[(batch, triton.cdiv(intermediate, int(block_m)))](
            x_c,
            prev_c,
            mix_c,
            key_c,
            mid,
            hidden,
            intermediate,
            BLOCK_M=int(block_m),
            BLOCK_K=int(block_k),
            num_warps=4,
        )
        _ffn_value_kernel[(batch, triton.cdiv(hidden, int(block_m)))](
            mid,
            value_c,
            out,
            hidden,
            intermediate,
            BLOCK_M=int(block_m),
            BLOCK_K=int(block_k),
            num_warps=4,
        )
    if had_seq:
        return out.unsqueeze(1), hidden_states[:, -1:].contiguous() if hidden_states.dim() == 3 else hidden_states.unsqueeze(1)
    return out, hidden_states.contiguous()
