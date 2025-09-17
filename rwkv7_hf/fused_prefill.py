# coding=utf-8
"""Optional Triton kernels for RWKV-7 native prefill.

The first prefill bottleneck after the fused recurrent scan is the
LoRA/state-prep bucket.  This module keeps the public HF path unchanged and
offers an opt-in state-prep kernel that fuses the elementwise/normalization
tail after W/A/G/V-gate projections:

* W raw -> recurrent decay ``exp(-0.606531 * sigmoid(w))`` in fp32
* K raw + ``k_a`` -> adjusted recurrent K
* K raw + ``k_k`` -> normalized ``kk`` per head
* optional V interpolation with ``v_first`` and V-gate

It deliberately does not replace cuBLAS-backed projections.  The goal is to
remove several small pointwise/normalize launches from prompt prefill while
leaving dense matmuls on the vendor libraries.
"""
from __future__ import annotations

from typing import Any

try:  # pragma: no cover - optional dependency in lightweight CI
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
    def _prefill_state_prep_kernel(
        w_raw_ptr,
        k_raw_ptr,
        v_raw_ptr,
        a_ptr,
        k_k_ptr,
        k_a_ptr,
        v_first_ptr,
        v_gate_ptr,
        w_out_ptr,
        k_out_ptr,
        v_out_ptr,
        kk_out_ptr,
        hidden: tl.constexpr,
        head_dim: tl.constexpr,
        has_v_gate: tl.constexpr,
        block_n: tl.constexpr,
    ):
        row = tl.program_id(0)
        head = tl.program_id(1)
        offs_n = tl.arange(0, block_n)
        mask = offs_n < head_dim
        offs = row * hidden + head * head_dim + offs_n
        param_offs = head * head_dim + offs_n

        w_raw = tl.load(w_raw_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        k_raw = tl.load(k_raw_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        v_raw = tl.load(v_raw_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        a_val = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        kk_scale = tl.load(k_k_ptr + param_offs, mask=mask, other=0.0).to(tl.float32)
        ka_scale = tl.load(k_a_ptr + param_offs, mask=mask, other=0.0).to(tl.float32)

        kk_raw = k_raw * kk_scale
        norm2 = tl.sum(tl.where(mask, kk_raw * kk_raw, 0.0), axis=0)
        inv_norm = tl.rsqrt(tl.maximum(norm2, 1.0e-20))
        kk = kk_raw * inv_norm
        k_adj = k_raw * (1.0 + (a_val - 1.0) * ka_scale)
        w_decay = tl.exp(-0.606531 * tl.sigmoid(w_raw))

        v_out = v_raw
        if has_v_gate:
            v_first = tl.load(v_first_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            v_gate = tl.load(v_gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            v_out = v_raw + (v_first - v_raw) * v_gate

        tl.store(w_out_ptr + offs, w_decay, mask=mask)
        tl.store(k_out_ptr + offs, k_adj, mask=mask)
        tl.store(v_out_ptr + offs, v_out, mask=mask)
        tl.store(kk_out_ptr + offs, kk, mask=mask)

    @triton.jit
    def _prefill_kv_kk_prep_kernel(
        k_raw_ptr,
        v_raw_ptr,
        a_ptr,
        k_k_ptr,
        k_a_ptr,
        v_first_ptr,
        v_gate_ptr,
        k_out_ptr,
        v_out_ptr,
        kk_out_ptr,
        hidden: tl.constexpr,
        head_dim: tl.constexpr,
        has_v_gate: tl.constexpr,
        block_n: tl.constexpr,
    ):
        row = tl.program_id(0)
        head = tl.program_id(1)
        offs_n = tl.arange(0, block_n)
        mask = offs_n < head_dim
        offs = row * hidden + head * head_dim + offs_n
        param_offs = head * head_dim + offs_n

        k_raw = tl.load(k_raw_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        v_raw = tl.load(v_raw_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        a_val = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        kk_scale = tl.load(k_k_ptr + param_offs, mask=mask, other=0.0).to(tl.float32)
        ka_scale = tl.load(k_a_ptr + param_offs, mask=mask, other=0.0).to(tl.float32)

        kk_raw = k_raw * kk_scale
        norm2 = tl.sum(tl.where(mask, kk_raw * kk_raw, 0.0), axis=0)
        inv_norm = tl.rsqrt(tl.maximum(norm2, 1.0e-20))
        kk = kk_raw * inv_norm
        k_adj = k_raw * (1.0 + (a_val - 1.0) * ka_scale)

        v_out = v_raw
        if has_v_gate:
            v_first = tl.load(v_first_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            v_gate = tl.load(v_gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            v_out = v_raw + (v_first - v_raw) * v_gate

        tl.store(k_out_ptr + offs, k_adj, mask=mask)
        tl.store(v_out_ptr + offs, v_out, mask=mask)
        tl.store(kk_out_ptr + offs, kk, mask=mask)


def fused_prefill_state_prep_available() -> bool:
    """Return whether the optional Triton native-prefill state-prep can run."""

    return bool(_HAS_TRITON and torch is not None)


def fused_prefill_kv_kk_prep_available() -> bool:
    """Return whether no-W native-prefill K/V/KK state-prep can run."""

    return bool(_HAS_TRITON and torch is not None)


def _flatten_seq_hidden(x: Any, *, hidden: int | None = None, name: str):
    if torch is None:
        raise RuntimeError("fused_prefill_state_prep requires torch")
    if x.dim() == 3:
        if hidden is not None and int(x.shape[2]) != int(hidden):
            raise ValueError(f"{name} hidden mismatch: got {int(x.shape[2])}, expected {hidden}")
        return x.reshape(int(x.shape[0]) * int(x.shape[1]), int(x.shape[2])), tuple(x.shape[:2])
    if x.dim() == 2:
        if hidden is not None and int(x.shape[1]) != int(hidden):
            raise ValueError(f"{name} hidden mismatch: got {int(x.shape[1])}, expected {hidden}")
        return x, None
    raise ValueError(f"{name} must be [batch, tokens, hidden] or [rows, hidden], got {tuple(x.shape)}")


def _restore_seq_hidden(x: Any, prefix: tuple[int, int] | None):
    if prefix is None:
        return x
    return x.reshape(int(prefix[0]), int(prefix[1]), int(x.shape[-1]))


def fused_prefill_state_prep(
    w_raw: Any,
    k_raw: Any,
    v_raw: Any,
    a: Any,
    k_k: Any,
    k_a: Any,
    *,
    v_first: Any | None = None,
    v_gate: Any | None = None,
    num_heads: int,
    head_dim: int,
    w_out_dtype: str = "fp32",
    force_fallback: bool = False,
):
    """Fuse native-prefill recurrent state preparation.

    Parameters use the same tensors as ``native_jit.prefill`` after the dense
    projections and LoRA modules.  ``a`` must already include its outer sigmoid.
    ``v_gate`` must already include its sigmoid; if omitted, V is passed through
    unchanged (layer 0 behavior).

    Returns ``(w_decay, k_adjusted, v_interpolated, kk_normalized)`` with the
    original `[B,T,H*N]` or `[rows,H*N]` layout restored.
    """

    if torch is None or F is None:
        raise RuntimeError("fused_prefill_state_prep requires torch")
    if w_out_dtype not in {"fp32", "input"}:
        raise ValueError(f"w_out_dtype must be 'fp32' or 'input'; got {w_out_dtype!r}")
    hidden = int(num_heads) * int(head_dim)
    w2, prefix = _flatten_seq_hidden(w_raw, hidden=hidden, name="w_raw")
    k2, k_prefix = _flatten_seq_hidden(k_raw, hidden=hidden, name="k_raw")
    v2, v_prefix = _flatten_seq_hidden(v_raw, hidden=hidden, name="v_raw")
    a2, a_prefix = _flatten_seq_hidden(a, hidden=hidden, name="a")
    if k_prefix != prefix or v_prefix != prefix or a_prefix != prefix:
        raise ValueError("w_raw, k_raw, v_raw and a must have identical layouts")
    if int(k_k.numel()) != hidden or int(k_a.numel()) != hidden:
        raise ValueError(f"k_k and k_a must have {hidden} elements")

    has_v_gate = v_first is not None and v_gate is not None
    if has_v_gate:
        vf2, vf_prefix = _flatten_seq_hidden(v_first, hidden=hidden, name="v_first")
        vg2, vg_prefix = _flatten_seq_hidden(v_gate, hidden=hidden, name="v_gate")
        if vf_prefix != prefix or vg_prefix != prefix:
            raise ValueError("v_first/v_gate layout must match v_raw")
    else:
        vf2 = v2
        vg2 = v2

    use_triton = (
        not force_fallback
        and fused_prefill_state_prep_available()
        and w2.is_cuda
        and k2.is_cuda
        and v2.is_cuda
        and a2.is_cuda
        and k_k.is_cuda
        and k_a.is_cuda
        and (not has_v_gate or (vf2.is_cuda and vg2.is_cuda))
        and w2.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and k2.dtype == w2.dtype
        and v2.dtype == w2.dtype
        and a2.dtype == w2.dtype
        and k_k.dtype == w2.dtype
        and k_a.dtype == w2.dtype
        and (not has_v_gate or (vf2.dtype == w2.dtype and vg2.dtype == w2.dtype))
    )
    if not use_triton:
        shaped = (int(w2.shape[0]), int(num_heads), int(head_dim))
        kk = F.normalize((k2 * k_k.reshape(1, hidden)).reshape(shaped), dim=-1, p=2.0).reshape_as(k2)
        k_out = k2 * (1 + (a2 - 1) * k_a.reshape(1, hidden))
        if has_v_gate:
            v_out = v2 + (vf2 - v2) * vg2
        else:
            v_out = v2
        w_out = torch.exp(-0.606531 * torch.sigmoid(w2.float()))
        if w_out_dtype == "input":
            w_out = w_out.to(w2.dtype)
        return (
            _restore_seq_hidden(w_out, prefix),
            _restore_seq_hidden(k_out, prefix),
            _restore_seq_hidden(v_out, prefix),
            _restore_seq_hidden(kk, prefix),
        )

    rows = int(w2.shape[0])
    w_c = w2.contiguous()
    k_c = k2.contiguous()
    v_c = v2.contiguous()
    a_c = a2.contiguous()
    kk_c = k_k.reshape(hidden).contiguous()
    ka_c = k_a.reshape(hidden).contiguous()
    vf_c = vf2.contiguous()
    vg_c = vg2.contiguous()
    w_out = torch.empty((rows, hidden), device=w2.device, dtype=torch.float32 if w_out_dtype == "fp32" else w2.dtype)
    k_out = torch.empty_like(k_c)
    v_out = torch.empty_like(v_c)
    kk_out = torch.empty_like(k_c)
    block_n = triton.next_power_of_2(int(head_dim))
    _prefill_state_prep_kernel[(rows, int(num_heads))](
        w_c,
        k_c,
        v_c,
        a_c,
        kk_c,
        ka_c,
        vf_c,
        vg_c,
        w_out,
        k_out,
        v_out,
        kk_out,
        hidden,
        int(head_dim),
        has_v_gate=bool(has_v_gate),
        block_n=block_n,
        num_warps=1,
    )
    return (
        _restore_seq_hidden(w_out, prefix),
        _restore_seq_hidden(k_out, prefix),
        _restore_seq_hidden(v_out, prefix),
        _restore_seq_hidden(kk_out, prefix),
    )


def fused_prefill_kv_kk_prep(
    k_raw: Any,
    v_raw: Any,
    a: Any,
    k_k: Any,
    k_a: Any,
    *,
    v_first: Any | None = None,
    v_gate: Any | None = None,
    num_heads: int,
    head_dim: int,
    force_fallback: bool = False,
):
    """Fuse native-prefill K/V/KK preparation without materializing W decay.

    This is paired with the raw-W ``clampw`` recurrent scan path.  It keeps the
    exact K adjustment, per-head KK normalization, and optional V interpolation
    from :func:`fused_prefill_state_prep`, but returns only
    ``(k_adjusted, v_interpolated, kk_normalized)`` so the scan can compute
    decay from raw ``w`` internally.
    """

    if torch is None or F is None:
        raise RuntimeError("fused_prefill_kv_kk_prep requires torch")
    hidden = int(num_heads) * int(head_dim)
    k2, prefix = _flatten_seq_hidden(k_raw, hidden=hidden, name="k_raw")
    v2, v_prefix = _flatten_seq_hidden(v_raw, hidden=hidden, name="v_raw")
    a2, a_prefix = _flatten_seq_hidden(a, hidden=hidden, name="a")
    if v_prefix != prefix or a_prefix != prefix:
        raise ValueError("k_raw, v_raw and a must have identical layouts")
    if int(k_k.numel()) != hidden or int(k_a.numel()) != hidden:
        raise ValueError(f"k_k and k_a must have {hidden} elements")

    has_v_gate = v_first is not None and v_gate is not None
    if has_v_gate:
        vf2, vf_prefix = _flatten_seq_hidden(v_first, hidden=hidden, name="v_first")
        vg2, vg_prefix = _flatten_seq_hidden(v_gate, hidden=hidden, name="v_gate")
        if vf_prefix != prefix or vg_prefix != prefix:
            raise ValueError("v_first/v_gate layout must match v_raw")
    else:
        vf2 = v2
        vg2 = v2

    use_triton = (
        not force_fallback
        and fused_prefill_kv_kk_prep_available()
        and k2.is_cuda
        and v2.is_cuda
        and a2.is_cuda
        and k_k.is_cuda
        and k_a.is_cuda
        and (not has_v_gate or (vf2.is_cuda and vg2.is_cuda))
        and k2.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and v2.dtype == k2.dtype
        and a2.dtype == k2.dtype
        and k_k.dtype == k2.dtype
        and k_a.dtype == k2.dtype
        and (not has_v_gate or (vf2.dtype == k2.dtype and vg2.dtype == k2.dtype))
    )
    if not use_triton:
        shaped = (int(k2.shape[0]), int(num_heads), int(head_dim))
        kk = F.normalize((k2 * k_k.reshape(1, hidden)).reshape(shaped), dim=-1, p=2.0).reshape_as(k2)
        k_out = k2 * (1 + (a2 - 1) * k_a.reshape(1, hidden))
        if has_v_gate:
            v_out = v2 + (vf2 - v2) * vg2
        else:
            v_out = v2
        return (
            _restore_seq_hidden(k_out, prefix),
            _restore_seq_hidden(v_out, prefix),
            _restore_seq_hidden(kk, prefix),
        )

    rows = int(k2.shape[0])
    k_c = k2.contiguous()
    v_c = v2.contiguous()
    a_c = a2.contiguous()
    kk_c = k_k.reshape(hidden).contiguous()
    ka_c = k_a.reshape(hidden).contiguous()
    vf_c = vf2.contiguous()
    vg_c = vg2.contiguous()
    k_out = torch.empty_like(k_c)
    v_out = torch.empty_like(v_c)
    kk_out = torch.empty_like(k_c)
    block_n = triton.next_power_of_2(int(head_dim))
    _prefill_kv_kk_prep_kernel[(rows, int(num_heads))](
        k_c,
        v_c,
        a_c,
        kk_c,
        ka_c,
        vf_c,
        vg_c,
        k_out,
        v_out,
        kk_out,
        hidden,
        int(head_dim),
        has_v_gate=bool(has_v_gate),
        block_n=block_n,
        num_warps=1,
    )
    return (
        _restore_seq_hidden(k_out, prefix),
        _restore_seq_hidden(v_out, prefix),
        _restore_seq_hidden(kk_out, prefix),
    )
