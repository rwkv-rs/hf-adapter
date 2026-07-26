# coding=utf-8
"""Graph-capturable dynamic-A8 / per-channel-W8 Linear for CUDA/HIP serving.

Small output-head matrices use W8A16 directly so activation quantization cannot
amplify the tiny hidden-state differences between the readable and optimized
model routes.  FP16 multi-row heads use one tiled ``tl.dot`` kernel; scalar and
non-FP16 inputs keep the portable GEMV kernels. Larger matrices use a fused
Triton per-row quantizer, ``torch._int_mm``, and a fused dequantization epilogue.
CUDA's cuBLASLt path requires the row count to be a multiple of 32 on current
PyTorch builds; ROCm retains the historical minimum of 17 rows.
The speed policy quantizes ``lm_head`` only. The memory policy quantizes all
size-gated linears covered by the native kernel (input width <= 4096), while
leaving wider FFN-down projections dense instead of dequantizing them on every
call.
"""
from __future__ import annotations

import os

try:  # pragma: no cover - optional in lightweight environments
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

from .native_quant_policy import normalize_native_mm_policy, should_quantize_linear

try:
    from .kernel_policy import current_kernel_policy
except Exception:  # pragma: no cover - remote-code fallback
    current_kernel_policy = None  # type: ignore[assignment]

try:
    from .sm70_quant import is_sm70, quantize_w8_row, w8_linear as sm70_w8_linear
except Exception:  # pragma: no cover
    def is_sm70(_device=None):
        return False

    quantize_w8_row = None  # type: ignore[assignment]
    sm70_w8_linear = None  # type: ignore[assignment]

try:  # pragma: no cover - optional Triton GPU dependency
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]

_HAS_TRITON = triton is not None and tl is not None


if _HAS_TRITON:

    @triton.jit
    def _w8a16_gemv_kernel(
        x_ptr,
        q_weight_ptr,
        weight_scale_ptr,
        bias_ptr,
        output_ptr,
        K: tl.constexpr,
        N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        tile = tl.program_id(0)
        offsets_n = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offsets_n < N
        accum = tl.zeros((BLOCK_N,), dtype=tl.float32)
        offsets_k = tl.arange(0, BLOCK_K)
        for start in range(0, K, BLOCK_K):
            k = start + offsets_k
            mask_k = k < K
            activation = tl.load(x_ptr + k, mask=mask_k, other=0.0).to(tl.float32)
            weight_offsets = k[:, None] * N + offsets_n[None, :]
            weight = tl.load(
                q_weight_ptr + weight_offsets,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0,
            ).to(tl.float32)
            accum += tl.sum(activation[:, None] * weight, axis=0)
        output = accum * tl.load(weight_scale_ptr + offsets_n, mask=mask_n, other=0.0).to(tl.float32)
        if HAS_BIAS:
            output += tl.load(bias_ptr + offsets_n, mask=mask_n, other=0.0).to(tl.float32)
        tl.store(output_ptr + offsets_n, output, mask=mask_n)

    @triton.jit
    def _w8a16_batched_gemv_kernel(
        x_ptr,
        q_weight_ptr,
        weight_scale_ptr,
        bias_ptr,
        output_ptr,
        x_row_stride,
        K: tl.constexpr,
        N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        """One graph-safe W8A16 launch for the small cached-decode batch."""

        row = tl.program_id(0)
        tile = tl.program_id(1)
        offsets_n = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offsets_n < N
        accum = tl.zeros((BLOCK_N,), dtype=tl.float32)
        offsets_k = tl.arange(0, BLOCK_K)
        for start in range(0, K, BLOCK_K):
            k = start + offsets_k
            mask_k = k < K
            activation = tl.load(
                x_ptr + row * x_row_stride + k,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            weight_offsets = k[:, None] * N + offsets_n[None, :]
            weight = tl.load(
                q_weight_ptr + weight_offsets,
                mask=mask_k[:, None] & mask_n[None, :],
                other=0,
            ).to(tl.float32)
            accum += tl.sum(activation[:, None] * weight, axis=0)
        output = accum * tl.load(
            weight_scale_ptr + offsets_n,
            mask=mask_n,
            other=0.0,
        ).to(tl.float32)
        if HAS_BIAS:
            output += tl.load(
                bias_ptr + offsets_n,
                mask=mask_n,
                other=0.0,
            ).to(tl.float32)
        tl.store(output_ptr + row * N + offsets_n, output, mask=mask_n)

    @triton.jit
    def _w8a16_tiled_matmul_kernel(
        x_ptr,
        q_weight_ptr,
        weight_scale_ptr,
        bias_ptr,
        output_ptr,
        x_row_stride,
        M: tl.constexpr,
        K: tl.constexpr,
        N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        """Tensor-core W8A16 matmul for a small FP16 output head."""

        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offsets_k = tl.arange(0, BLOCK_K)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for start in range(0, K, BLOCK_K):
            k = start + offsets_k
            activation = tl.load(
                x_ptr + offsets_m[:, None] * x_row_stride + k[None, :],
                mask=(offsets_m[:, None] < M) & (k[None, :] < K),
                other=0.0,
            )
            weight = tl.load(
                q_weight_ptr + k[:, None] * N + offsets_n[None, :],
                mask=(k[:, None] < K) & (offsets_n[None, :] < N),
                other=0,
            ).to(tl.float16)
            accumulator = tl.dot(activation, weight, accumulator)
        output = accumulator * tl.load(
            weight_scale_ptr + offsets_n,
            mask=offsets_n < N,
            other=0.0,
        )[None, :].to(tl.float32)
        if HAS_BIAS:
            output += tl.load(
                bias_ptr + offsets_n,
                mask=offsets_n < N,
                other=0.0,
            )[None, :].to(tl.float32)
        tl.store(
            output_ptr + offsets_m[:, None] * N + offsets_n[None, :],
            output,
            mask=(offsets_m[:, None] < M) & (offsets_n[None, :] < N),
        )

    @triton.jit
    def _dynamic_a8_quant_kernel(
        x_ptr,
        q_ptr,
        scale_ptr,
        x_row_stride,
        K: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_K)
        mask = offsets < K
        values = tl.load(x_ptr + row * x_row_stride + offsets, mask=mask, other=0.0).to(tl.float32)
        maximum = tl.max(tl.abs(values), axis=0)
        scale = tl.maximum(maximum / 127.0, 1.0e-5)
        # ``libdevice.rint`` is available while compiling the CUDA backend but
        # is not exported by Triton's HIP libdevice facade on ROCm 7.2.  Keep
        # the quantizer backend-neutral by expressing round-to-nearest with
        # core Triton operations instead of calling a vendor libdevice symbol.
        # Values exactly halfway between integers are vanishingly rare for
        # scaled activations; rounding them away from zero is also symmetric
        # and stays within the existing int8 quality tolerance.
        scaled = values / scale
        quantized = tl.where(
            scaled >= 0.0,
            tl.floor(scaled + 0.5),
            tl.ceil(scaled - 0.5),
        )
        quantized = tl.maximum(tl.minimum(quantized, 127.0), -127.0).to(tl.int8)
        tl.store(q_ptr + row * K + offsets, quantized, mask=mask)
        tl.store(scale_ptr + row, scale)

    @triton.jit
    def _a8w8_dequant_kernel(
        accum_ptr,
        activation_scale_ptr,
        weight_scale_ptr,
        bias_ptr,
        output_ptr,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        row = tl.program_id(0)
        tile = tl.program_id(1)
        offsets = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < N
        accum = tl.load(accum_ptr + row * N + offsets, mask=mask, other=0).to(tl.float32)
        activation_scale = tl.load(activation_scale_ptr + row)
        weight_scale = tl.load(weight_scale_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        output = accum * activation_scale * weight_scale
        if HAS_BIAS:
            output += tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        tl.store(output_ptr + row * N + offsets, output, mask=mask)

    @triton.jit
    def _a8w8_relu2_requant_kernel(
        accum_ptr,
        activation_scale_ptr,
        weight_scale_ptr,
        bias_ptr,
        q_ptr,
        output_scale_ptr,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        """Dequantize W8 FFN-up, apply fp16 ReLU2, and requantize per row."""

        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_N)
        mask = offsets < N
        accum = tl.load(accum_ptr + row * N + offsets, mask=mask, other=0).to(tl.float32)
        activation_scale = tl.load(activation_scale_ptr + row).to(tl.float32)
        weight_scale = tl.load(
            weight_scale_ptr + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        preactivation = accum * activation_scale * weight_scale
        if HAS_BIAS:
            preactivation += tl.load(
                bias_ptr + offsets, mask=mask, other=0.0
            ).to(tl.float32)
        # Match the normal fp16 Linear -> ReLU -> square boundary before the
        # second dynamic quantizer observes the activation.
        preactivation = preactivation.to(tl.float16).to(tl.float32)
        activated = tl.maximum(preactivation, 0.0)
        activated = (activated * activated).to(tl.float16).to(tl.float32)
        maximum = tl.max(tl.where(mask, tl.abs(activated), 0.0), axis=0)
        scale = tl.maximum(maximum / 127.0, 1.0e-5)
        scaled = activated / scale
        quantized = tl.where(
            scaled >= 0.0,
            tl.floor(scaled + 0.5),
            tl.ceil(scaled - 0.5),
        )
        quantized = tl.maximum(tl.minimum(quantized, 127.0), -127.0).to(tl.int8)
        tl.store(q_ptr + row * N + offsets, quantized, mask=mask)
        tl.store(output_scale_ptr + row, scale)

    @triton.jit
    def _a8w8_dequant_add_kernel(
        accum_ptr,
        activation_scale_ptr,
        weight_scale_ptr,
        bias_ptr,
        residual_ptr,
        output_ptr,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        """Dequantize W8 FFN-down and fuse the residual epilogue."""

        row = tl.program_id(0)
        tile = tl.program_id(1)
        offsets = tile * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offsets < N
        accum = tl.load(accum_ptr + row * N + offsets, mask=mask, other=0).to(tl.float32)
        activation_scale = tl.load(activation_scale_ptr + row).to(tl.float32)
        weight_scale = tl.load(
            weight_scale_ptr + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        output = accum * activation_scale * weight_scale
        if HAS_BIAS:
            output += tl.load(
                bias_ptr + offsets, mask=mask, other=0.0
            ).to(tl.float32)
        output += tl.load(
            residual_ptr + row * N + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        tl.store(output_ptr + row * N + offsets, output, mask=mask)


def a8w8_available(device=None) -> bool:
    if not (_HAS_TRITON and torch is not None and torch.cuda.is_available()):
        return False
    resolved = torch.device("cuda" if device is None else device)
    return resolved.type == "cuda" and hasattr(torch, "_int_mm")


def a8w8_gemv_max_rows(device=None) -> int:
    """Return the exact-card small-batch W8A16 route limit.

    Multi-row direct GEMV is promoted only on the measured RTX 3090 policy.
    Other cards retain the previously safe one-row route unless an exact-card
    policy or environment override selects zero (disable) or a larger limit.
    """

    raw = os.environ.get("RWKV7_A8W8_GEMV_MAX_ROWS")
    if raw is not None:
        try:
            return min(max(0, int(raw)), 32)
        except ValueError:
            return 1
    if current_kernel_policy is None or torch is None:
        return 1
    try:
        policy = current_kernel_policy(device=device, torch_module=torch)
        return min(max(0, int(getattr(policy, "a8w8_gemv_max_rows", 1))), 32)
    except Exception:
        return 1


def a8w8_fused_ffn_enabled(device=None) -> bool:
    """Select the paired FFN route only on an explicitly promoted card."""

    raw = os.environ.get("RWKV7_A8W8_FUSED_FFN")
    if raw is not None:
        return raw.strip().lower() not in {"", "0", "false", "no", "off"}
    if current_kernel_policy is None or torch is None:
        return False
    try:
        policy = current_kernel_policy(device=device, torch_module=torch)
        return bool(getattr(policy, "a8w8_fused_ffn", False))
    except Exception:
        return False


def quantize_weight_per_channel_int8(weight):
    """Return contiguous ``[in,out]`` int8 weights and fp32 output scales."""

    if torch is None:
        raise RuntimeError("A8W8 quantization requires torch")
    fp32 = weight.detach().float()
    scales = (fp32.abs().amax(dim=1) / 127.0).clamp_min(1.0e-5)
    quantized = (fp32 / scales[:, None]).round().clamp(-127, 127).to(torch.int8)
    return quantized.t().contiguous(), scales.contiguous()


def _int_mm_padded_rows(rows: int, *, hip: bool | None = None) -> int:
    """Return a legal activation-row count for ``torch._int_mm``.

    PyTorch rejects 16 or fewer rows on both GPU backends. CUDA 13's
    cuBLASLt implementation additionally rejects non-multiples of 32, even
    though older releases accepted the historical 17-row padding. Keep the
    ROCm behavior unchanged and round CUDA rows to the next complete tile.
    The extra rows remain zero/unwritten and are discarded after GEMM.
    """

    rows = max(int(rows), 17)
    if hip is None:
        version = getattr(torch, "version", None) if torch is not None else None
        hip = bool(getattr(version, "hip", None))
    return rows if hip else ((rows + 31) // 32) * 32


def a8w8_linear(
    x,
    q_weight_t,
    weight_scale,
    bias=None,
    *,
    out=None,
    tiled_w8a16: bool = False,
):
    """Apply graph-capturable dynamic-A8 / W8 matrix multiplication."""

    if torch is None or F is None:
        raise RuntimeError("A8W8 linear requires torch")
    scalar = x.dim() == 1
    leading = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    rows, inputs = int(x2.shape[0]), int(x2.shape[1])
    outputs = int(q_weight_t.shape[1])
    supported = bool(
        a8w8_available(x.device)
        and x.is_cuda
        and x.dtype in {torch.float16, torch.bfloat16}
        and q_weight_t.is_cuda
        and q_weight_t.dtype == torch.int8
        and q_weight_t.is_contiguous()
        and int(q_weight_t.shape[0]) == inputs
        and inputs % 8 == 0
        and outputs % 8 == 0
        and inputs <= 4096
    )
    if not supported:
        dense = (q_weight_t.t().to(x.dtype) * weight_scale.to(x.dtype)[:, None]).contiguous()
        result = F.linear(x, dense, bias)
        if out is not None:
            out.copy_(result)
            return out
        return result

    # Do not pad small rows into dynamic-A8 GEMM. The output head uses a tiled
    # tensor-core W8A16 kernel, while scalar and body-module fallbacks retain
    # the portable GEMV kernels. Every route honors last-token row strides.
    gemv_max_rows = a8w8_gemv_max_rows(x.device)
    if rows <= gemv_max_rows:
        output2 = (
            torch.empty((rows, outputs), device=x.device, dtype=x.dtype)
            if out is None
            else out.reshape(rows, outputs)
        )
        block_k = int(os.environ.get("RWKV7_A8W8_GEMV_BLOCK_K", "256"))
        block_n = int(os.environ.get("RWKV7_A8W8_GEMV_BLOCK_N", "64"))
        num_warps = int(os.environ.get("RWKV7_A8W8_GEMV_WARPS", "1"))
        if rows == 1:
            _w8a16_gemv_kernel[(triton.cdiv(outputs, block_n),)](
                x2,
                q_weight_t,
                weight_scale,
                bias if bias is not None else weight_scale,
                output2,
                K=inputs,
                N=outputs,
                BLOCK_K=block_k,
                BLOCK_N=block_n,
                HAS_BIAS=bias is not None,
                num_warps=num_warps,
            )
        elif tiled_w8a16 and x.dtype == torch.float16:
            block_m = 16
            _w8a16_tiled_matmul_kernel[
                (triton.cdiv(rows, block_m), triton.cdiv(outputs, block_n))
            ](
                x2,
                q_weight_t,
                weight_scale,
                bias if bias is not None else weight_scale,
                output2,
                x2.stride(0),
                M=rows,
                K=inputs,
                N=outputs,
                BLOCK_M=block_m,
                BLOCK_K=32,
                BLOCK_N=block_n,
                HAS_BIAS=bias is not None,
                num_warps=4,
            )
        else:
            _w8a16_batched_gemv_kernel[(rows, triton.cdiv(outputs, block_n))](
                x2,
                q_weight_t,
                weight_scale,
                bias if bias is not None else weight_scale,
                output2,
                x2.stride(0),
                K=inputs,
                N=outputs,
                BLOCK_K=block_k,
                BLOCK_N=block_n,
                HAS_BIAS=bias is not None,
                num_warps=num_warps,
            )
        if scalar:
            return output2.reshape(outputs)
        return output2.reshape(*leading, outputs)

    padded_rows = _int_mm_padded_rows(rows)
    q_activation = torch.empty((padded_rows, inputs), device=x.device, dtype=torch.int8)
    activation_scale = torch.empty((rows,), device=x.device, dtype=torch.float32)
    block_k = triton.next_power_of_2(inputs)
    _dynamic_a8_quant_kernel[(rows,)](
        x2,
        q_activation,
        activation_scale,
        x2.stride(0),
        K=inputs,
        BLOCK_K=block_k,
        num_warps=8,
    )
    accum = torch._int_mm(q_activation, q_weight_t)
    output = (
        torch.empty((rows, outputs), device=x.device, dtype=x.dtype)
        if out is None
        else out.reshape(rows, outputs)
    )
    block_n = 256
    _a8w8_dequant_kernel[(rows, triton.cdiv(outputs, block_n))](
        accum,
        activation_scale,
        weight_scale,
        bias if bias is not None else weight_scale,
        output,
        N=outputs,
        BLOCK_N=block_n,
        HAS_BIAS=bias is not None,
        num_warps=4,
    )
    if scalar:
        return output.reshape(outputs)
    return output.reshape(*leading, outputs)


def a8w8_ffn(x, up, down, residual):
    """Fuse a paired A8W8 ``up -> ReLU2 -> down + residual`` boundary.

    The intermediate stays quantized: the up GEMM's int32 accumulator is
    dequantized, rounded through the normal fp16 activation boundary, squared,
    and requantized in one Triton launch.  This removes the materialized fp16
    FFN expansion plus two pointwise launches while retaining the same packed
    weights and per-row/per-channel A8W8 scheme.

    ``None`` means the exact fast-path contract is not met and lets callers use
    their existing two-Linear fallback.
    """

    if torch is None or not _HAS_TRITON:
        return None
    if not isinstance(up, A8W8Linear) or not isinstance(down, A8W8Linear):
        return None
    if (
        up.sm70_rowwise
        or down.sm70_rowwise
        or not x.is_cuda
        or not a8w8_fused_ffn_enabled(x.device)
        or x.dtype != torch.float16
        or not hasattr(torch, "_int_mm")
        or int(up.out_features) != int(down.in_features)
        or int(up.in_features) != int(down.out_features)
        or int(up.in_features) > 4096
        or int(up.in_features) % 8
        or int(up.out_features) % 8
    ):
        return None
    leading = x.shape[:-1]
    x2 = x.reshape(-1, int(up.in_features))
    residual2 = residual.reshape(-1, int(down.out_features))
    rows = int(x2.shape[0])
    if rows < 1 or int(residual2.shape[0]) != rows or not residual2.is_contiguous():
        return None
    padded_rows = _int_mm_padded_rows(rows)

    q_input = torch.empty(
        (padded_rows, int(up.in_features)), device=x.device, dtype=torch.int8
    )
    input_scale = torch.empty((rows,), device=x.device, dtype=torch.float32)
    input_block = triton.next_power_of_2(int(up.in_features))
    _dynamic_a8_quant_kernel[(rows,)](
        x2,
        q_input,
        input_scale,
        x2.stride(0),
        K=int(up.in_features),
        BLOCK_K=input_block,
        num_warps=8,
    )
    up_accum = torch._int_mm(q_input, up.q_weight_t)

    q_hidden = torch.empty(
        (padded_rows, int(up.out_features)), device=x.device, dtype=torch.int8
    )
    hidden_scale = torch.empty((rows,), device=x.device, dtype=torch.float32)
    hidden_block = triton.next_power_of_2(int(up.out_features))
    _a8w8_relu2_requant_kernel[(rows,)](
        up_accum,
        input_scale,
        up.weight_scale,
        up.bias if up.bias is not None else up.weight_scale,
        q_hidden,
        hidden_scale,
        N=int(up.out_features),
        BLOCK_N=hidden_block,
        HAS_BIAS=up.bias is not None,
        num_warps=8,
    )
    down_accum = torch._int_mm(q_hidden, down.q_weight_t)
    output2 = torch.empty(
        (rows, int(down.out_features)), device=x.device, dtype=x.dtype
    )
    block_n = 256
    _a8w8_dequant_add_kernel[(rows, triton.cdiv(int(down.out_features), block_n))](
        down_accum,
        hidden_scale,
        down.weight_scale,
        down.bias if down.bias is not None else down.weight_scale,
        residual2,
        output2,
        N=int(down.out_features),
        BLOCK_N=block_n,
        HAS_BIAS=down.bias is not None,
        num_warps=4,
    )
    return output2.reshape(*leading, int(down.out_features))


class A8W8Linear(torch.nn.Module):
    """Inference-only ``nn.Linear`` replacement using dynamic A8 and W8."""

    def __init__(self, linear, *, tiled_w8a16: bool = False):
        super().__init__()
        self.tiled_w8a16 = bool(tiled_w8a16)
        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        self.fused_ffn = a8w8_fused_ffn_enabled(linear.weight.device)
        self.sm70_rowwise = bool(is_sm70(linear.weight.device) and quantize_w8_row is not None)
        if self.sm70_rowwise:
            q_weight_row, weight_scale = quantize_w8_row(linear.weight)
            self.register_buffer("q_weight_row", q_weight_row)
        else:
            q_weight_t, weight_scale = quantize_weight_per_channel_int8(linear.weight)
            self.register_buffer("q_weight_t", q_weight_t)
        self.register_buffer("weight_scale", weight_scale)
        if linear.bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", linear.bias.detach().clone())

    def forward(self, x):
        if self.sm70_rowwise and sm70_w8_linear is not None:
            result = sm70_w8_linear(x, self.q_weight_row, self.weight_scale)
            return result if self.bias is None else result + self.bias
        return a8w8_linear(
            x,
            self.q_weight_t,
            self.weight_scale,
            self.bias,
            tiled_w8a16=self.tiled_w8a16,
        )

    def rwkv7_forward_into(self, x, out):
        """Write graph-replay head output directly into a stable buffer."""

        if self.sm70_rowwise and sm70_w8_linear is not None:
            if self.bias is None:
                return sm70_w8_linear(x, self.q_weight_row, self.weight_scale, out=out)
            result = sm70_w8_linear(x, self.q_weight_row, self.weight_scale) + self.bias
            out.copy_(result)
            return out
        return a8w8_linear(
            x,
            self.q_weight_t,
            self.weight_scale,
            self.bias,
            out=out,
            tiled_w8a16=self.tiled_w8a16,
        )

    def rwkv7_forward_ffn(self, x, down, residual):
        """Attempt the paired A8W8 FFN route; return ``None`` to fall back."""

        if not self.fused_ffn:
            return None
        return a8w8_ffn(x, self, down, residual)

    def extra_repr(self) -> str:
        small_rows = "tiled_w8a16" if self.tiled_w8a16 else "batched_w8a16"
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"dynamic_a8w8, small_rows={small_rows}"
        )


def quantize_model_a8w8(
    model,
    *,
    min_params: int = 8_000_000,
    policy: str = "speed",
) -> int:
    """Replace selected linears; ``speed`` intentionally selects lm_head only."""

    if torch is None:
        raise RuntimeError("A8W8 quantization requires torch")
    policy = normalize_native_mm_policy(policy)
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not should_quantize_linear(
            name,
            int(module.weight.numel()),
            min_params=int(min_params),
            policy=policy,
        ):
            continue
        # The fused CUDA path is intentionally bounded to width 4096. Keeping
        # unsupported FFN-down matrices dense is both faster and avoids a
        # transient full dequantized copy in the fallback path.
        if int(module.in_features) > 4096:
            continue
        targets.append(name)
    for full_name in targets:
        parent_name, _, attribute = full_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        is_output_head = full_name == "lm_head" or full_name.endswith(".lm_head")
        setattr(
            parent,
            attribute,
            A8W8Linear(
                getattr(parent, attribute),
                tiled_w8a16=is_output_head,
            ),
        )
    fused_ffn_modules = 0
    for name, module in model.named_modules():
        if not (
            isinstance(module, A8W8Linear)
            and module.fused_ffn
            and name.endswith(".ffn.key")
        ):
            continue
        value_name = name[: -len("key")] + "value"
        try:
            down = model.get_submodule(value_name)
        except AttributeError:
            continue
        if isinstance(down, A8W8Linear) and down.fused_ffn:
            fused_ffn_modules += 1
    setattr(model, "_rwkv7_native_mm_quantization", "a8w8")
    setattr(model, "_rwkv7_native_mm_replaced_modules", len(targets))
    setattr(
        model,
        "_rwkv7_native_mm_block_replaced_modules",
        sum(name.startswith("model.layers.") for name in targets),
    )
    setattr(
        model,
        "_rwkv7_native_mm_fused_relu2_ffn_modules",
        fused_ffn_modules,
    )
    for attr in (
        "_rwkv7_native_jit_pack_cache",
        "_rwkv7_native_graph_pack_cache",
        "_rwkv7_native_graph_runner_cache",
        "_rwkv7_native_prefill_graph_runner_cache",
        "_rwkv7_native_prefill_graph_hot_runner",
    ):
        if hasattr(model, attr):
            delattr(model, attr)
    return len(targets)
