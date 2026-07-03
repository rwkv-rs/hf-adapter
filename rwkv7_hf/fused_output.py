# coding=utf-8
"""Optional fused attention output preparation prototypes for RWKV-7 decode.

The prototype targets the `attn_norm_out_proj` bucket without replacing the
cuBLAS output projection.  It fuses the pointwise/norm portion before `o_proj`:

    GroupNorm(recurrent_out) + recurrent correction, then multiply by g.

The default profitable path keeps the final dense `o_proj` as a regular
linear projection.  A second telemetry-only prototype also folds `o_proj` into
the Triton kernel so benchmarks can test whether deeper output fusion is worth
integrating.
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
    def _attn_output_prepare_kernel(
        recurrent_ptr,
        r_ptr,
        k_ptr,
        v_ptr,
        g_ptr,
        rk_ptr,
        gn_weight_ptr,
        gn_bias_ptr,
        out_ptr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        head_v_dim: tl.constexpr,
        value_dim: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        head_id = tl.program_id(1)
        offs_n = tl.arange(0, BLOCK_N)
        offs_v = tl.arange(0, BLOCK_V)
        mask_n = offs_n < head_dim
        mask_v = offs_v < head_v_dim
        base_v = batch_id * value_dim + head_id * head_v_dim
        base_n = batch_id * num_heads * head_dim + head_id * head_dim

        rec = tl.load(recurrent_ptr + base_v + offs_v, mask=mask_v, other=0.0).to(tl.float32)
        mean = tl.sum(rec, axis=0) / head_v_dim
        centered = tl.where(mask_v, rec - mean, 0.0)
        var = tl.sum(centered * centered, axis=0) / head_v_dim
        inv_std = tl.rsqrt(var + eps)
        normed = centered * inv_std

        rr = tl.load(r_ptr + base_n + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        kk = tl.load(k_ptr + base_n + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        rk = tl.load(rk_ptr + head_id * head_dim + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        corr_scale = tl.sum(rr * kk * rk, axis=0)
        vv = tl.load(v_ptr + base_v + offs_v, mask=mask_v, other=0.0).to(tl.float32)
        gate = tl.load(g_ptr + base_v + offs_v, mask=mask_v, other=0.0).to(tl.float32)
        weight = tl.load(gn_weight_ptr + head_id * head_v_dim + offs_v, mask=mask_v, other=1.0).to(tl.float32)
        bias = tl.load(gn_bias_ptr + head_id * head_v_dim + offs_v, mask=mask_v, other=0.0).to(tl.float32)
        prepared = (normed * weight + bias + corr_scale * vv) * gate
        tl.store(out_ptr + base_v + offs_v, prepared, mask=mask_v)

    @triton.jit
    def _attn_output_prepare_from_correction_kernel(
        recurrent_ptr,
        correction_ptr,
        g_ptr,
        gn_weight_ptr,
        gn_bias_ptr,
        out_ptr,
        num_heads: tl.constexpr,
        head_v_dim: tl.constexpr,
        value_dim: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        head_id = tl.program_id(1)
        offs_v = tl.arange(0, BLOCK_V)
        mask_v = offs_v < head_v_dim
        base_v = batch_id * value_dim + head_id * head_v_dim

        rec = tl.load(recurrent_ptr + base_v + offs_v, mask=mask_v, other=0.0).to(tl.float32)
        mean = tl.sum(rec, axis=0) / head_v_dim
        centered = tl.where(mask_v, rec - mean, 0.0)
        var = tl.sum(centered * centered, axis=0) / head_v_dim
        normed = centered * tl.rsqrt(var + eps)

        corr = tl.load(correction_ptr + base_v + offs_v, mask=mask_v, other=0.0).to(tl.float32)
        gate = tl.load(g_ptr + base_v + offs_v, mask=mask_v, other=0.0).to(tl.float32)
        weight = tl.load(gn_weight_ptr + head_id * head_v_dim + offs_v, mask=mask_v, other=1.0).to(tl.float32)
        bias = tl.load(gn_bias_ptr + head_id * head_v_dim + offs_v, mask=mask_v, other=0.0).to(tl.float32)
        prepared = (normed * weight + bias + corr) * gate
        tl.store(out_ptr + base_v + offs_v, prepared, mask=mask_v)

    @triton.jit
    def _attn_output_prepare_from_sk_raw_v_kernel(
        recurrent_ptr,
        sk_ptr,
        v_raw_ptr,
        v_first_ptr,
        v_gate_ptr,
        g_ptr,
        gn_weight_ptr,
        gn_bias_ptr,
        out_ptr,
        num_heads: tl.constexpr,
        head_v_dim: tl.constexpr,
        value_dim: tl.constexpr,
        has_v_gate: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        head_id = tl.program_id(1)
        offs_v = tl.arange(0, BLOCK_V)
        mask_v = offs_v < head_v_dim
        base_v = batch_id * value_dim + head_id * head_v_dim

        rec = tl.load(recurrent_ptr + base_v + offs_v, mask=mask_v, other=0.0).to(tl.float32)
        mean = tl.sum(rec, axis=0) / head_v_dim
        centered = tl.where(mask_v, rec - mean, 0.0)
        var = tl.sum(centered * centered, axis=0) / head_v_dim
        normed = centered * tl.rsqrt(var + eps)

        vv = tl.load(v_raw_ptr + base_v + offs_v, mask=mask_v, other=0.0).to(tl.float32)
        if has_v_gate:
            vf = tl.load(v_first_ptr + base_v + offs_v, mask=mask_v, other=0.0).to(tl.float32)
            vg = tl.load(v_gate_ptr + base_v + offs_v, mask=mask_v, other=0.0).to(tl.float32)
            vv = vv + (vf - vv) * vg
        sk = tl.load(sk_ptr + batch_id * num_heads + head_id).to(tl.float32)
        gate = tl.load(g_ptr + base_v + offs_v, mask=mask_v, other=0.0).to(tl.float32)
        weight = tl.load(gn_weight_ptr + head_id * head_v_dim + offs_v, mask=mask_v, other=1.0).to(tl.float32)
        bias = tl.load(gn_bias_ptr + head_id * head_v_dim + offs_v, mask=mask_v, other=0.0).to(tl.float32)
        prepared = (normed * weight + bias + sk * vv) * gate
        tl.store(out_ptr + base_v + offs_v, prepared, mask=mask_v)

    @triton.jit
    def _attn_output_prepare_raw_kv_kernel(
        recurrent_ptr,
        r_ptr,
        k_raw_ptr,
        v_raw_ptr,
        a_ptr,
        v_first_ptr,
        v_gate_ptr,
        g_ptr,
        k_a_ptr,
        rk_ptr,
        gn_weight_ptr,
        gn_bias_ptr,
        out_ptr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        head_v_dim: tl.constexpr,
        value_dim: tl.constexpr,
        has_v_gate: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        head_id = tl.program_id(1)
        offs_n = tl.arange(0, BLOCK_N)
        offs_v = tl.arange(0, BLOCK_V)
        mask_n = offs_n < head_dim
        mask_v = offs_v < head_v_dim
        base_v = batch_id * value_dim + head_id * head_v_dim
        base_n = batch_id * num_heads * head_dim + head_id * head_dim
        param_base = head_id * head_dim

        rec = tl.load(recurrent_ptr + base_v + offs_v, mask=mask_v, other=0.0).to(tl.float32)
        mean = tl.sum(rec, axis=0) / head_v_dim
        centered = tl.where(mask_v, rec - mean, 0.0)
        var = tl.sum(centered * centered, axis=0) / head_v_dim
        normed = centered * tl.rsqrt(var + eps)

        rr = tl.load(r_ptr + base_n + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        kk_raw = tl.load(k_raw_ptr + base_n + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        aa = tl.load(a_ptr + base_n + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        ka = tl.load(k_a_ptr + param_base + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        rk = tl.load(rk_ptr + param_base + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        k_adj = kk_raw * (1.0 + (aa - 1.0) * ka)
        corr_scale = tl.sum(rr * k_adj * rk, axis=0)

        vv = tl.load(v_raw_ptr + base_v + offs_v, mask=mask_v, other=0.0).to(tl.float32)
        if has_v_gate:
            vf = tl.load(v_first_ptr + base_v + offs_v, mask=mask_v, other=0.0).to(tl.float32)
            vg = tl.load(v_gate_ptr + base_v + offs_v, mask=mask_v, other=0.0).to(tl.float32)
            vv = vv + (vf - vv) * vg
        gate = tl.load(g_ptr + base_v + offs_v, mask=mask_v, other=0.0).to(tl.float32)
        weight = tl.load(gn_weight_ptr + head_id * head_v_dim + offs_v, mask=mask_v, other=1.0).to(tl.float32)
        bias = tl.load(gn_bias_ptr + head_id * head_v_dim + offs_v, mask=mask_v, other=0.0).to(tl.float32)
        prepared = (normed * weight + bias + corr_scale * vv) * gate
        tl.store(out_ptr + base_v + offs_v, prepared, mask=mask_v)

    @triton.jit
    def _attn_output_project_kernel(
        recurrent_ptr,
        r_ptr,
        k_ptr,
        v_ptr,
        g_ptr,
        rk_ptr,
        gn_weight_ptr,
        gn_bias_ptr,
        o_weight_ptr,
        o_bias_ptr,
        out_ptr,
        num_heads: tl.constexpr,
        head_dim: tl.constexpr,
        head_v_dim: tl.constexpr,
        value_dim: tl.constexpr,
        eps: tl.constexpr,
        HAS_O_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        block_m = tl.program_id(1)
        offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
        mask_m = offs_m < value_dim
        offs_n = tl.arange(0, BLOCK_N)
        offs_v = tl.arange(0, BLOCK_V)
        mask_n = offs_n < head_dim
        mask_v = offs_v < head_v_dim
        acc = tl.zeros((BLOCK_M,), tl.float32)

        for head_id in range(0, num_heads):
            base_v = batch_id * value_dim + head_id * head_v_dim
            base_n = batch_id * num_heads * head_dim + head_id * head_dim
            rec = tl.load(recurrent_ptr + base_v + offs_v, mask=mask_v, other=0.0).to(tl.float32)
            mean = tl.sum(rec, axis=0) / head_v_dim
            centered = tl.where(mask_v, rec - mean, 0.0)
            var = tl.sum(centered * centered, axis=0) / head_v_dim
            normed = centered * tl.rsqrt(var + eps)

            rr = tl.load(r_ptr + base_n + offs_n, mask=mask_n, other=0.0).to(tl.float32)
            kk = tl.load(k_ptr + base_n + offs_n, mask=mask_n, other=0.0).to(tl.float32)
            rk = tl.load(rk_ptr + head_id * head_dim + offs_n, mask=mask_n, other=0.0).to(tl.float32)
            corr_scale = tl.sum(rr * kk * rk, axis=0)

            vv = tl.load(v_ptr + base_v + offs_v, mask=mask_v, other=0.0).to(tl.float32)
            gate = tl.load(g_ptr + base_v + offs_v, mask=mask_v, other=0.0).to(tl.float32)
            weight = tl.load(gn_weight_ptr + head_id * head_v_dim + offs_v, mask=mask_v, other=1.0).to(tl.float32)
            bias = tl.load(gn_bias_ptr + head_id * head_v_dim + offs_v, mask=mask_v, other=0.0).to(tl.float32)
            prepared = (normed * weight + bias + corr_scale * vv) * gate

            kidx = head_id * head_v_dim + offs_v
            o_offsets = offs_m[:, None] * value_dim + kidx[None, :]
            o_mask = mask_m[:, None] & mask_v[None, :]
            ow = tl.load(o_weight_ptr + o_offsets, mask=o_mask, other=0.0).to(tl.float32)
            acc += tl.sum(ow * prepared[None, :], axis=1)

        if HAS_O_BIAS:
            ob = tl.load(o_bias_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
            acc += ob
        tl.store(out_ptr + batch_id * value_dim + offs_m, acc, mask=mask_m)


def fused_attn_output_prepare_available() -> bool:
    """Return whether the optional fused output-prepare kernel can run."""

    return bool(_HAS_TRITON and torch is not None)


def fused_attn_output_prepare_from_correction_available() -> bool:
    """Return whether fused output-prep can consume precomputed correction."""

    return bool(_HAS_TRITON and torch is not None)


def fused_attn_output_prepare_from_sk_raw_v_available() -> bool:
    """Return whether fused output-prep can consume correction scale plus raw V."""

    return bool(_HAS_TRITON and torch is not None)


def fused_attn_output_prepare_raw_kv_available() -> bool:
    """Return whether fused output-prep can recompute correction from raw K/V."""

    return bool(_HAS_TRITON and torch is not None)


def fused_attn_output_project_available() -> bool:
    """Return whether the optional fused output-prepare-plus-o_proj kernel can run."""

    return bool(_HAS_TRITON and torch is not None)


def _flatten_2d(x: Any, expected: int | None = None, *, name: str):
    if torch is None:
        raise RuntimeError("fused_attn_output_prepare requires torch")
    if x.dim() == 3:
        if int(x.shape[1]) != 1:
            raise ValueError(f"{name} must be [batch, 1, hidden] or [batch, hidden], got {tuple(x.shape)}")
        if expected is not None and int(x.shape[2]) != expected:
            raise ValueError(f"{name} hidden mismatch: got {int(x.shape[2])}, expected {expected}")
        return x.reshape(int(x.shape[0]), int(x.shape[2])), True
    if x.dim() == 2:
        if expected is not None and int(x.shape[1]) != expected:
            raise ValueError(f"{name} hidden mismatch: got {int(x.shape[1])}, expected {expected}")
        return x, False
    raise ValueError(f"{name} must be [batch, 1, hidden] or [batch, hidden]")


def _flatten_head(x: Any, num_heads: int, dim: int, *, name: str):
    if x.dim() == 4:
        if int(x.shape[1]) != 1 or int(x.shape[2]) != num_heads or int(x.shape[3]) != dim:
            raise ValueError(f"{name} must be [batch, 1, {num_heads}, {dim}], got {tuple(x.shape)}")
        return x.reshape(int(x.shape[0]), num_heads, dim)
    if x.dim() == 3:
        if int(x.shape[1]) != num_heads or int(x.shape[2]) != dim:
            raise ValueError(f"{name} must be [batch, {num_heads}, {dim}], got {tuple(x.shape)}")
        return x
    raise ValueError(f"{name} must be [batch, 1, heads, dim] or [batch, heads, dim]")


def fused_attn_output_prepare(
    recurrent_out: Any,
    r: Any,
    k: Any,
    v: Any,
    g: Any,
    r_k: Any,
    group_norm_weight: Any,
    group_norm_bias: Any,
    *,
    num_heads: int,
    head_dim: int,
    head_v_dim: int,
    eps: float,
    force_fallback: bool = False,
):
    """Prepare attention output before `o_proj`.

    Returns a tensor matching the rank of `recurrent_out`: `[batch, hidden]` or
    `[batch, 1, hidden]`.  R/K/V may be supplied in head format.
    """

    if torch is None or F is None:
        raise RuntimeError("fused_attn_output_prepare requires torch")
    value_dim = int(num_heads) * int(head_v_dim)
    rec2, had_seq = _flatten_2d(recurrent_out, value_dim, name="recurrent_out")
    g2, g_had_seq = _flatten_2d(g, value_dim, name="g")
    if g_had_seq != had_seq or tuple(g2.shape) != tuple(rec2.shape):
        raise ValueError("recurrent_out and g must have identical flattened shape/layout")
    r3 = _flatten_head(r, int(num_heads), int(head_dim), name="r")
    k3 = _flatten_head(k, int(num_heads), int(head_dim), name="k")
    v3 = _flatten_head(v, int(num_heads), int(head_v_dim), name="v")
    batch = int(rec2.shape[0])
    if int(r3.shape[0]) != batch or int(k3.shape[0]) != batch or int(v3.shape[0]) != batch:
        raise ValueError("r/k/v batch size must match recurrent_out")
    if r_k.dim() != 2 or int(r_k.shape[0]) != int(num_heads) or int(r_k.shape[1]) != int(head_dim):
        raise ValueError(f"r_k must be [{num_heads}, {head_dim}], got {tuple(r_k.shape)}")
    if group_norm_weight.dim() != 1 or int(group_norm_weight.shape[0]) != value_dim:
        raise ValueError(f"group_norm_weight must be [{value_dim}], got {tuple(group_norm_weight.shape)}")
    if group_norm_bias.dim() != 1 or int(group_norm_bias.shape[0]) != value_dim:
        raise ValueError(f"group_norm_bias must be [{value_dim}], got {tuple(group_norm_bias.shape)}")

    tensors = [rec2, r3, k3, v3, g2, r_k, group_norm_weight, group_norm_bias]
    use_triton = (
        not force_fallback
        and fused_attn_output_prepare_available()
        and all(t.is_cuda for t in tensors)
        and rec2.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and all(t.dtype == rec2.dtype for t in tensors)
        and int(head_dim) <= 128
        and int(head_v_dim) <= 128
    )
    if not use_triton:
        normed = F.group_norm(rec2, num_groups=int(num_heads), weight=group_norm_weight, bias=group_norm_bias, eps=float(eps))
        correction = ((r3 * k3 * r_k.view(1, int(num_heads), int(head_dim))).sum(-1, keepdim=True) * v3).reshape(batch, value_dim)
        out = (normed + correction) * g2
    else:
        rec_c, r_c, k_c, v_c, g_c = [t.contiguous() for t in (rec2, r3, k3, v3, g2)]
        rk_c = r_k.contiguous()
        w_c = group_norm_weight.contiguous()
        b_c = group_norm_bias.contiguous()
        out = torch.empty_like(rec2)
        block_n = triton.next_power_of_2(int(head_dim))
        block_v = triton.next_power_of_2(int(head_v_dim))
        _attn_output_prepare_kernel[(batch, int(num_heads))](
            rec_c,
            r_c,
            k_c,
            v_c,
            g_c,
            rk_c,
            w_c,
            b_c,
            out,
            num_heads=int(num_heads),
            head_dim=int(head_dim),
            head_v_dim=int(head_v_dim),
            value_dim=value_dim,
            eps=float(eps),
            BLOCK_N=block_n,
            BLOCK_V=block_v,
            num_warps=1,
        )
    if had_seq:
        return out.unsqueeze(1)
    return out


def fused_attn_output_prepare_from_correction(
    recurrent_out: Any,
    correction: Any,
    g: Any,
    group_norm_weight: Any,
    group_norm_bias: Any,
    *,
    num_heads: int,
    head_v_dim: int,
    eps: float,
    force_fallback: bool = False,
):
    """Prepare attention output using a correction tensor emitted by scan.

    ``correction`` must already contain the RWKV correction term
    ``sum(r * k * r_k) * v``.  This lets upstream fused scans avoid writing the
    full adjusted K/V tensors when output prep only needs their product.
    """

    if torch is None or F is None:
        raise RuntimeError("fused_attn_output_prepare_from_correction requires torch")
    value_dim = int(num_heads) * int(head_v_dim)
    rec2, had_seq = _flatten_2d(recurrent_out, value_dim, name="recurrent_out")
    corr2, corr_had_seq = _flatten_2d(correction, value_dim, name="correction")
    g2, g_had_seq = _flatten_2d(g, value_dim, name="g")
    if corr_had_seq != had_seq or g_had_seq != had_seq:
        raise ValueError("recurrent_out, correction, and g must use the same flattened/sequence layout")
    if tuple(corr2.shape) != tuple(rec2.shape) or tuple(g2.shape) != tuple(rec2.shape):
        raise ValueError("recurrent_out, correction, and g must have identical shape")
    if group_norm_weight.dim() != 1 or int(group_norm_weight.shape[0]) != value_dim:
        raise ValueError(f"group_norm_weight must be [{value_dim}], got {tuple(group_norm_weight.shape)}")
    if group_norm_bias.dim() != 1 or int(group_norm_bias.shape[0]) != value_dim:
        raise ValueError(f"group_norm_bias must be [{value_dim}], got {tuple(group_norm_bias.shape)}")

    tensors = [rec2, corr2, g2, group_norm_weight, group_norm_bias]
    use_triton = (
        not force_fallback
        and fused_attn_output_prepare_from_correction_available()
        and all(t.is_cuda for t in tensors)
        and rec2.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and all(t.dtype == rec2.dtype for t in tensors)
        and int(head_v_dim) <= 128
    )
    if not use_triton:
        normed = F.group_norm(rec2, num_groups=int(num_heads), weight=group_norm_weight, bias=group_norm_bias, eps=float(eps))
        out = (normed + corr2) * g2
    else:
        rec_c, corr_c, g_c = [t.contiguous() for t in (rec2, corr2, g2)]
        w_c = group_norm_weight.contiguous()
        b_c = group_norm_bias.contiguous()
        out = torch.empty_like(rec2)
        block_v = triton.next_power_of_2(int(head_v_dim))
        _attn_output_prepare_from_correction_kernel[(int(rec2.shape[0]), int(num_heads))](
            rec_c,
            corr_c,
            g_c,
            w_c,
            b_c,
            out,
            num_heads=int(num_heads),
            head_v_dim=int(head_v_dim),
            value_dim=value_dim,
            eps=float(eps),
            BLOCK_V=block_v,
            num_warps=1,
        )
    if had_seq:
        return out.unsqueeze(1)
    return out


def _flatten_sk(x: Any, num_heads: int, *, name: str):
    if x.dim() == 3:
        if int(x.shape[2]) != int(num_heads):
            raise ValueError(f"{name} must be [batch, tokens, {num_heads}] or [batch, {num_heads}], got {tuple(x.shape)}")
        return x.reshape(int(x.shape[0]) * int(x.shape[1]), int(num_heads))
    if x.dim() == 2:
        if int(x.shape[1]) != int(num_heads):
            raise ValueError(f"{name} must be [batch, {num_heads}], got {tuple(x.shape)}")
        return x
    raise ValueError(f"{name} must be [batch, tokens, heads] or [batch, heads]")


def fused_attn_output_prepare_from_sk_raw_v(
    recurrent_out: Any,
    sk: Any,
    v_raw: Any,
    g: Any,
    group_norm_weight: Any,
    group_norm_bias: Any,
    *,
    v_first: Any | None = None,
    v_gate: Any | None = None,
    num_heads: int,
    head_v_dim: int,
    eps: float,
    force_fallback: bool = False,
):
    """Prepare attention output from recurrent output, per-head sk, and raw V.

    ``sk`` is the per-token/head correction scale ``sum(r * k_adj * r_k)``
    emitted by a no-K/V-writeback state-prep scan.  This avoids recomputing
    adjusted K in output prep and avoids writing a full correction tensor.
    """

    if torch is None or F is None:
        raise RuntimeError("fused_attn_output_prepare_from_sk_raw_v requires torch")
    value_dim = int(num_heads) * int(head_v_dim)
    rec2, had_seq = _flatten_2d(recurrent_out, value_dim, name="recurrent_out")
    g2, g_had_seq = _flatten_2d(g, value_dim, name="g")
    if g_had_seq != had_seq or tuple(g2.shape) != tuple(rec2.shape):
        raise ValueError("recurrent_out and g must have identical flattened shape/layout")
    sk2 = _flatten_sk(sk, int(num_heads), name="sk")
    v3 = _flatten_head(v_raw, int(num_heads), int(head_v_dim), name="v_raw")
    batch = int(rec2.shape[0])
    if int(sk2.shape[0]) != batch or int(v3.shape[0]) != batch:
        raise ValueError("sk/v_raw batch size must match recurrent_out")
    has_v_gate = v_first is not None and v_gate is not None
    if has_v_gate:
        vf3 = _flatten_head(v_first, int(num_heads), int(head_v_dim), name="v_first")
        vg3 = _flatten_head(v_gate, int(num_heads), int(head_v_dim), name="v_gate")
        if int(vf3.shape[0]) != batch or int(vg3.shape[0]) != batch:
            raise ValueError("v_first/v_gate batch size must match recurrent_out")
    else:
        vf3 = v3
        vg3 = v3
    if group_norm_weight.dim() != 1 or int(group_norm_weight.shape[0]) != value_dim:
        raise ValueError(f"group_norm_weight must be [{value_dim}], got {tuple(group_norm_weight.shape)}")
    if group_norm_bias.dim() != 1 or int(group_norm_bias.shape[0]) != value_dim:
        raise ValueError(f"group_norm_bias must be [{value_dim}], got {tuple(group_norm_bias.shape)}")

    tensors = [rec2, sk2, v3, g2, group_norm_weight, group_norm_bias]
    if has_v_gate:
        tensors.extend([vf3, vg3])
    use_triton = (
        not force_fallback
        and fused_attn_output_prepare_from_sk_raw_v_available()
        and all(t.is_cuda for t in tensors)
        and rec2.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and all(t.dtype == rec2.dtype for t in tensors)
        and int(head_v_dim) <= 128
    )
    if not use_triton:
        normed = F.group_norm(rec2, num_groups=int(num_heads), weight=group_norm_weight, bias=group_norm_bias, eps=float(eps))
        v_adj = v3 + (vf3 - v3) * vg3 if has_v_gate else v3
        correction = (sk2.to(v_adj.dtype).unsqueeze(-1) * v_adj).reshape(batch, value_dim)
        out = (normed + correction) * g2
    else:
        rec_c, sk_c, v_c, g_c = [t.contiguous() for t in (rec2, sk2, v3, g2)]
        vf_c = vf3.contiguous()
        vg_c = vg3.contiguous()
        w_c = group_norm_weight.contiguous()
        b_c = group_norm_bias.contiguous()
        out = torch.empty_like(rec2)
        block_v = triton.next_power_of_2(int(head_v_dim))
        _attn_output_prepare_from_sk_raw_v_kernel[(batch, int(num_heads))](
            rec_c,
            sk_c,
            v_c,
            vf_c,
            vg_c,
            g_c,
            w_c,
            b_c,
            out,
            num_heads=int(num_heads),
            head_v_dim=int(head_v_dim),
            value_dim=value_dim,
            has_v_gate=bool(has_v_gate),
            eps=float(eps),
            BLOCK_V=block_v,
            num_warps=1,
        )
    if had_seq:
        return out.unsqueeze(1)
    return out


def fused_attn_output_prepare_raw_kv(
    recurrent_out: Any,
    r: Any,
    k_raw: Any,
    v_raw: Any,
    a: Any,
    g: Any,
    k_a: Any,
    r_k: Any,
    group_norm_weight: Any,
    group_norm_bias: Any,
    *,
    v_first: Any | None = None,
    v_gate: Any | None = None,
    num_heads: int,
    head_dim: int,
    head_v_dim: int,
    eps: float,
    force_fallback: bool = False,
):
    """Prepare attention output while recomputing correction from raw K/V/A.

    This pairs with a no-K/V-writeback state-prep scan.  It recomputes the
    adjusted K and interpolated V needed by the RWKV correction outside the
    dominant scan kernel, leaving group norm and G gating fused as usual.
    """

    if torch is None or F is None:
        raise RuntimeError("fused_attn_output_prepare_raw_kv requires torch")
    value_dim = int(num_heads) * int(head_v_dim)
    rec2, had_seq = _flatten_2d(recurrent_out, value_dim, name="recurrent_out")
    g2, g_had_seq = _flatten_2d(g, value_dim, name="g")
    if g_had_seq != had_seq or tuple(g2.shape) != tuple(rec2.shape):
        raise ValueError("recurrent_out and g must have identical flattened shape/layout")
    r3 = _flatten_head(r, int(num_heads), int(head_dim), name="r")
    k3 = _flatten_head(k_raw, int(num_heads), int(head_dim), name="k_raw")
    v3 = _flatten_head(v_raw, int(num_heads), int(head_v_dim), name="v_raw")
    a3 = _flatten_head(a, int(num_heads), int(head_dim), name="a")
    batch = int(rec2.shape[0])
    if any(int(t.shape[0]) != batch for t in (r3, k3, v3, a3)):
        raise ValueError("r/k/v/a batch size must match recurrent_out")
    has_v_gate = v_first is not None and v_gate is not None
    if has_v_gate:
        vf3 = _flatten_head(v_first, int(num_heads), int(head_v_dim), name="v_first")
        vg3 = _flatten_head(v_gate, int(num_heads), int(head_v_dim), name="v_gate")
        if int(vf3.shape[0]) != batch or int(vg3.shape[0]) != batch:
            raise ValueError("v_first/v_gate batch size must match recurrent_out")
    else:
        vf3 = v3
        vg3 = v3
    if k_a.dim() != 2 or int(k_a.shape[0]) != int(num_heads) or int(k_a.shape[1]) != int(head_dim):
        raise ValueError(f"k_a must be [{num_heads}, {head_dim}], got {tuple(k_a.shape)}")
    if r_k.dim() != 2 or int(r_k.shape[0]) != int(num_heads) or int(r_k.shape[1]) != int(head_dim):
        raise ValueError(f"r_k must be [{num_heads}, {head_dim}], got {tuple(r_k.shape)}")
    if group_norm_weight.dim() != 1 or int(group_norm_weight.shape[0]) != value_dim:
        raise ValueError(f"group_norm_weight must be [{value_dim}], got {tuple(group_norm_weight.shape)}")
    if group_norm_bias.dim() != 1 or int(group_norm_bias.shape[0]) != value_dim:
        raise ValueError(f"group_norm_bias must be [{value_dim}], got {tuple(group_norm_bias.shape)}")

    tensors = [rec2, r3, k3, v3, a3, g2, k_a, r_k, group_norm_weight, group_norm_bias]
    if has_v_gate:
        tensors.extend([vf3, vg3])
    use_triton = (
        not force_fallback
        and fused_attn_output_prepare_raw_kv_available()
        and all(t.is_cuda for t in tensors)
        and rec2.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and all(t.dtype == rec2.dtype for t in tensors)
        and int(head_dim) <= 128
        and int(head_v_dim) <= 128
    )
    if not use_triton:
        normed = F.group_norm(rec2, num_groups=int(num_heads), weight=group_norm_weight, bias=group_norm_bias, eps=float(eps))
        k_adj = k3 * (1 + (a3 - 1) * k_a.view(1, int(num_heads), int(head_dim)))
        if has_v_gate:
            v_adj = v3 + (vf3 - v3) * vg3
        else:
            v_adj = v3
        correction = ((r3 * k_adj * r_k.view(1, int(num_heads), int(head_dim))).sum(-1, keepdim=True) * v_adj).reshape(batch, value_dim)
        out = (normed + correction) * g2
    else:
        rec_c, r_c, k_c, v_c, a_c, g_c = [t.contiguous() for t in (rec2, r3, k3, v3, a3, g2)]
        vf_c = vf3.contiguous()
        vg_c = vg3.contiguous()
        ka_c = k_a.contiguous()
        rk_c = r_k.contiguous()
        w_c = group_norm_weight.contiguous()
        b_c = group_norm_bias.contiguous()
        out = torch.empty_like(rec2)
        block_n = triton.next_power_of_2(int(head_dim))
        block_v = triton.next_power_of_2(int(head_v_dim))
        _attn_output_prepare_raw_kv_kernel[(batch, int(num_heads))](
            rec_c,
            r_c,
            k_c,
            v_c,
            a_c,
            vf_c,
            vg_c,
            g_c,
            ka_c,
            rk_c,
            w_c,
            b_c,
            out,
            num_heads=int(num_heads),
            head_dim=int(head_dim),
            head_v_dim=int(head_v_dim),
            value_dim=value_dim,
            has_v_gate=bool(has_v_gate),
            eps=float(eps),
            BLOCK_N=block_n,
            BLOCK_V=block_v,
            num_warps=1,
        )
    if had_seq:
        return out.unsqueeze(1)
    return out


def fused_attn_output_project(
    recurrent_out: Any,
    r: Any,
    k: Any,
    v: Any,
    g: Any,
    r_k: Any,
    group_norm_weight: Any,
    group_norm_bias: Any,
    o_proj_weight: Any,
    o_proj_bias: Any | None = None,
    *,
    num_heads: int,
    head_dim: int,
    head_v_dim: int,
    eps: float,
    block_m: int = 32,
    force_fallback: bool = False,
):
    """Compute attention output prep and the final ``o_proj`` in one prototype.

    This is benchmark-only telemetry. It intentionally recomputes the prepared
    output for each output-projection row block, so it is not expected to be the
    final production kernel unless benchmarks prove the launch reduction wins.
    """

    if torch is None or F is None:
        raise RuntimeError("fused_attn_output_project requires torch")
    value_dim = int(num_heads) * int(head_v_dim)
    rec2, had_seq = _flatten_2d(recurrent_out, value_dim, name="recurrent_out")
    g2, g_had_seq = _flatten_2d(g, value_dim, name="g")
    if g_had_seq != had_seq or tuple(g2.shape) != tuple(rec2.shape):
        raise ValueError("recurrent_out and g must have identical flattened shape/layout")
    r3 = _flatten_head(r, int(num_heads), int(head_dim), name="r")
    k3 = _flatten_head(k, int(num_heads), int(head_dim), name="k")
    v3 = _flatten_head(v, int(num_heads), int(head_v_dim), name="v")
    batch = int(rec2.shape[0])
    if int(r3.shape[0]) != batch or int(k3.shape[0]) != batch or int(v3.shape[0]) != batch:
        raise ValueError("r/k/v batch size must match recurrent_out")
    if r_k.dim() != 2 or int(r_k.shape[0]) != int(num_heads) or int(r_k.shape[1]) != int(head_dim):
        raise ValueError(f"r_k must be [{num_heads}, {head_dim}], got {tuple(r_k.shape)}")
    if group_norm_weight.dim() != 1 or int(group_norm_weight.shape[0]) != value_dim:
        raise ValueError(f"group_norm_weight must be [{value_dim}], got {tuple(group_norm_weight.shape)}")
    if group_norm_bias.dim() != 1 or int(group_norm_bias.shape[0]) != value_dim:
        raise ValueError(f"group_norm_bias must be [{value_dim}], got {tuple(group_norm_bias.shape)}")
    if o_proj_weight.dim() != 2 or int(o_proj_weight.shape[0]) != value_dim or int(o_proj_weight.shape[1]) != value_dim:
        raise ValueError(f"o_proj_weight must be [{value_dim}, {value_dim}], got {tuple(o_proj_weight.shape)}")
    if o_proj_bias is not None and (o_proj_bias.dim() != 1 or int(o_proj_bias.shape[0]) != value_dim):
        raise ValueError(f"o_proj_bias must be [{value_dim}], got {tuple(o_proj_bias.shape)}")

    tensors = [rec2, r3, k3, v3, g2, r_k, group_norm_weight, group_norm_bias, o_proj_weight]
    use_triton = (
        not force_fallback
        and fused_attn_output_project_available()
        and all(t.is_cuda for t in tensors)
        and (o_proj_bias is None or o_proj_bias.is_cuda)
        and rec2.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and all(t.dtype == rec2.dtype for t in tensors)
        and (o_proj_bias is None or o_proj_bias.dtype == rec2.dtype)
        and int(head_dim) <= 128
        and int(head_v_dim) <= 128
    )
    if not use_triton:
        prep = fused_attn_output_prepare(
            rec2,
            r3,
            k3,
            v3,
            g2,
            r_k,
            group_norm_weight,
            group_norm_bias,
            num_heads=int(num_heads),
            head_dim=int(head_dim),
            head_v_dim=int(head_v_dim),
            eps=float(eps),
            force_fallback=True,
        )
        out = F.linear(prep, o_proj_weight, o_proj_bias)
    else:
        rec_c, r_c, k_c, v_c, g_c = [t.contiguous() for t in (rec2, r3, k3, v3, g2)]
        rk_c = r_k.contiguous()
        gnw_c = group_norm_weight.contiguous()
        gnb_c = group_norm_bias.contiguous()
        ow_c = o_proj_weight.contiguous()
        ob_c = o_proj_bias.contiguous() if o_proj_bias is not None else gnw_c
        out = torch.empty_like(rec2)
        block_n = triton.next_power_of_2(int(head_dim))
        block_v = triton.next_power_of_2(int(head_v_dim))
        _attn_output_project_kernel[(batch, triton.cdiv(value_dim, int(block_m)))](
            rec_c,
            r_c,
            k_c,
            v_c,
            g_c,
            rk_c,
            gnw_c,
            gnb_c,
            ow_c,
            ob_c,
            out,
            num_heads=int(num_heads),
            head_dim=int(head_dim),
            head_v_dim=int(head_v_dim),
            value_dim=value_dim,
            eps=float(eps),
            HAS_O_BIAS=o_proj_bias is not None,
            BLOCK_M=int(block_m),
            BLOCK_N=block_n,
            BLOCK_V=block_v,
            num_warps=4,
        )
    if had_seq:
        return out.unsqueeze(1)
    return out
