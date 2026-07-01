# coding=utf-8
"""Native quantization prototypes for RWKV-7 serving.

This module starts the RWKV-native W8 path with a simple row-wise int8 weight
pack plus fused dequant GEMV.  It is intentionally optional and safe to import
without Triton/CUDA: CPU-only or unsupported hosts fall back to a torch
reference that reconstructs the dequantized weight.

The first target is decode-hot linear layers where generic bitsandbytes kernels
are currently much slower than fp16 on V100.  This prototype is telemetry-first;
it is not wired into the HF model path until correctness and speed are both
validated by benchmark rows.
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
    def _int8_rowwise_gemv_kernel(
        x_ptr,
        q_weight_ptr,
        scale_ptr,
        bias_ptr,
        out_ptr,
        in_features: tl.constexpr,
        out_features: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        batch_id = tl.program_id(0)
        block_id = tl.program_id(1)
        offs_m = block_id * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = tl.arange(0, BLOCK_K)
        mask_m = offs_m < out_features

        acc = tl.zeros((BLOCK_M,), tl.float32)
        for start in range(0, in_features, BLOCK_K):
            kidx = start + offs_k
            mask_k = kidx < in_features
            x = tl.load(x_ptr + batch_id * in_features + kidx, mask=mask_k, other=0.0).to(tl.float32)
            q_offsets = offs_m[:, None] * in_features + kidx[None, :]
            q = tl.load(q_weight_ptr + q_offsets, mask=mask_m[:, None] & mask_k[None, :], other=0).to(tl.float32)
            acc += tl.sum(q * x[None, :], axis=1)

        scale = tl.load(scale_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
        acc = acc * scale
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)
            acc += bias
        tl.store(out_ptr + batch_id * out_features + offs_m, acc, mask=mask_m)


def native_int8_gemv_available() -> bool:
    """Return whether the optional Triton int8 dequant-GEMV prototype can run."""

    return bool(_HAS_TRITON and torch is not None)


def quantize_int8_rowwise(weight: Any, *, eps: float = 1e-8):
    """Pack a dense weight matrix into signed row-wise int8 plus fp32 scales.

    Args:
        weight: ``[out_features, in_features]`` tensor.

    Returns:
        ``(q_weight, scales)`` where ``q_weight`` is int8 and ``scales`` is
        fp32 with one scale per output row. The dequantized weight is
        approximately ``q_weight.float() * scales[:, None]``.
    """

    if torch is None:
        raise RuntimeError("quantize_int8_rowwise requires torch")
    if weight.dim() != 2:
        raise ValueError(f"weight must be [out_features, in_features], got {tuple(weight.shape)}")
    w = weight.detach().float()
    scales = (w.abs().amax(dim=1).clamp_min(float(eps)) / 127.0).to(torch.float32)
    q = torch.round(w / scales[:, None]).clamp(-127, 127).to(torch.int8)
    return q.contiguous(), scales.contiguous()


def _flatten_input(x: Any, in_features: int, *, name: str):
    if torch is None:
        raise RuntimeError("int8_rowwise_gemv requires torch")
    if x.dim() == 3:
        if int(x.shape[1]) != 1 or int(x.shape[2]) != in_features:
            raise ValueError(f"{name} must be [batch, 1, {in_features}] or [batch, {in_features}], got {tuple(x.shape)}")
        return x.reshape(int(x.shape[0]), in_features), True
    if x.dim() == 2:
        if int(x.shape[1]) != in_features:
            raise ValueError(f"{name} must be [batch, 1, {in_features}] or [batch, {in_features}], got {tuple(x.shape)}")
        return x, False
    raise ValueError(f"{name} must be [batch, 1, in_features] or [batch, in_features]")


def dequantize_int8_rowwise(q_weight: Any, scales: Any):
    if torch is None:
        raise RuntimeError("dequantize_int8_rowwise requires torch")
    return q_weight.float() * scales.float().reshape(-1, 1)


def int8_rowwise_gemv(
    x: Any,
    q_weight: Any,
    scales: Any,
    bias: Any | None = None,
    *,
    block_m: int = 16,
    block_k: int = 64,
    force_fallback: bool = False,
):
    """Run row-wise int8 dequant GEMV/GEMM for decode-sized batches.

    Inputs may be shaped ``[batch, in_features]`` or ``[batch, 1, in_features]``.
    Outputs preserve the input rank. This prototype assumes int8 weights and
    fp32/fp16/bf16 activations; unsupported hosts fall back to torch.
    """

    if torch is None or F is None:
        raise RuntimeError("int8_rowwise_gemv requires torch")
    if q_weight.dim() != 2:
        raise ValueError("q_weight must be [out_features, in_features]")
    out_features, in_features = int(q_weight.shape[0]), int(q_weight.shape[1])
    if scales.dim() != 1 or int(scales.shape[0]) != out_features:
        raise ValueError(f"scales must be [{out_features}], got {tuple(scales.shape)}")
    x2, had_seq = _flatten_input(x, in_features, name="x")
    if q_weight.dtype != torch.int8:
        raise ValueError(f"q_weight must be torch.int8, got {q_weight.dtype}")
    if bias is not None and (bias.dim() != 1 or int(bias.shape[0]) != out_features):
        raise ValueError(f"bias must be [{out_features}], got {tuple(bias.shape)}")

    use_triton = (
        not force_fallback
        and native_int8_gemv_available()
        and x2.is_cuda
        and q_weight.is_cuda
        and scales.is_cuda
        and (bias is None or bias.is_cuda)
        and x2.dtype in (torch.float16, torch.bfloat16, torch.float32)
    )
    if not use_triton:
        weight = dequantize_int8_rowwise(q_weight, scales).to(dtype=x2.dtype, device=x2.device)
        out = F.linear(x2, weight, bias.to(device=x2.device, dtype=x2.dtype) if bias is not None else None)
    else:
        x_c = x2.contiguous()
        q_c = q_weight.contiguous()
        s_c = scales.contiguous()
        b_c = bias.contiguous() if bias is not None else scales  # unused when HAS_BIAS=False
        out = torch.empty((int(x2.shape[0]), out_features), device=x2.device, dtype=x2.dtype)
        grid = (int(x2.shape[0]), triton.cdiv(out_features, int(block_m)))
        _int8_rowwise_gemv_kernel[grid](
            x_c,
            q_c,
            s_c,
            b_c,
            out,
            in_features,
            out_features,
            HAS_BIAS=bias is not None,
            BLOCK_M=int(block_m),
            BLOCK_K=int(block_k),
            num_warps=4,
        )
    if had_seq:
        return out.unsqueeze(1)
    return out


def int8_weight_footprint_bytes(q_weight: Any, scales: Any, bias: Any | None = None) -> int:
    """Approximate packed weight footprint for telemetry."""

    total = int(q_weight.numel()) + int(scales.numel()) * 4
    if bias is not None:
        total += int(bias.numel()) * int(bias.element_size())
    return total
