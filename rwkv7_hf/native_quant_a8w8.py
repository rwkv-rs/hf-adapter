# coding=utf-8
"""Graph-capturable dynamic-A8 / per-channel-W8 Linear for CUDA serving.

PyTorch's CUDA int8 GEMM requires more than sixteen activation rows. Cached
decode normally has only one to eight rows, so this module pads the activation
matrix to seventeen rows, uses a fused Triton per-row quantizer, dispatches
``torch._int_mm``, and fuses output scaling/casting in a second Triton kernel.
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
    from .sm70_quant import (
        is_sm70,
        quantize_w8_row,
        w8_linear as sm70_w8_linear,
        w8_prefill_dequant_weight as sm70_w8_prefill_dequant_weight,
    )
except Exception:  # pragma: no cover
    is_sm70 = lambda _device=None: False  # type: ignore[assignment]
    quantize_w8_row = None  # type: ignore[assignment]
    sm70_w8_linear = None  # type: ignore[assignment]
    sm70_w8_prefill_dequant_weight = None  # type: ignore[assignment]

try:  # pragma: no cover - optional CUDA dependency
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except Exception:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    libdevice = None  # type: ignore[assignment]

_HAS_TRITON = triton is not None and tl is not None and libdevice is not None


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
        quantized = libdevice.rint(values / scale)
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


def a8w8_available(device=None) -> bool:
    if not (_HAS_TRITON and torch is not None and torch.cuda.is_available()):
        return False
    resolved = torch.device("cuda" if device is None else device)
    return resolved.type == "cuda" and hasattr(torch, "_int_mm")


def a8w8_gemv_max_rows(device=None) -> int:
    """Return the exact-card small-batch W8A16 route limit.

    Multi-row direct GEMV is promoted only on the measured RTX 3090 policy.
    Other cards retain the previously safe one-row route unless an explicit
    environment override is supplied for a local sweep.
    """

    raw = os.environ.get("RWKV7_A8W8_GEMV_MAX_ROWS")
    if raw is not None:
        try:
            return min(max(1, int(raw)), 32)
        except ValueError:
            return 1
    if current_kernel_policy is None or torch is None:
        return 1
    try:
        policy = current_kernel_policy(device=device, torch_module=torch)
        return min(max(1, int(getattr(policy, "a8w8_gemv_max_rows", 1))), 32)
    except Exception:
        return 1


def quantize_weight_per_channel_int8(weight):
    """Return contiguous ``[in,out]`` int8 weights and fp32 output scales."""

    if torch is None:
        raise RuntimeError("A8W8 quantization requires torch")
    fp32 = weight.detach().float()
    scales = (fp32.abs().amax(dim=1) / 127.0).clamp_min(1.0e-5)
    quantized = (fp32 / scales[:, None]).round().clamp(-127, 127).to(torch.int8)
    return quantized.t().contiguous(), scales.contiguous()


def a8w8_linear(x, q_weight_t, weight_scale, bias=None, *, out=None):
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

    # Small cached-decode batches do not fill tensor-core tiles. Reading W8
    # directly and dequantizing in registers avoids padding every projection to
    # seventeen rows. The batched kernel still launches once per projection and
    # honors non-contiguous last-token row strides.
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

    padded_rows = max(rows, 17)
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


class A8W8Linear(torch.nn.Module):
    """Inference-only ``nn.Linear`` replacement using dynamic A8 and W8."""

    def __init__(self, linear):
        super().__init__()
        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
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
            rows = int(x.numel() // x.shape[-1])
            if rows <= 8:
                result = sm70_w8_linear(x, self.q_weight_row, self.weight_scale)
            else:
                dense = (
                    sm70_w8_prefill_dequant_weight(
                        self.q_weight_row,
                        self.weight_scale,
                        rows,
                    )
                    if sm70_w8_prefill_dequant_weight is not None
                    else None
                )
                if dense is None:
                    dense = self.q_weight_row.to(x.dtype) * self.weight_scale[:, None]
                result = F.linear(x, dense)
            return result if self.bias is None else result + self.bias
        return a8w8_linear(x, self.q_weight_t, self.weight_scale, self.bias)

    def rwkv7_forward_into(self, x, out):
        """Write graph-replay head output directly into a stable buffer."""

        if self.sm70_rowwise and sm70_w8_linear is not None:
            rows = int(x.numel() // x.shape[-1])
            if self.bias is None and rows <= 8:
                return sm70_w8_linear(x, self.q_weight_row, self.weight_scale, out=out)
            result = self.forward(x)
            out.copy_(result)
            return out
        return a8w8_linear(x, self.q_weight_t, self.weight_scale, self.bias, out=out)

    def rwkv7_prefill_dequant_shape(self, rows):
        """Expose the bounded sm70 dense-prefill workspace contract."""

        try:
            threshold = max(
                0,
                int(os.environ.get("RWKV7_SM70_W8_PREFILL_DENSE_MIN_ROWS", "16")),
            )
        except ValueError:
            threshold = 16
        if (
            not self.sm70_rowwise
            or sm70_w8_prefill_dequant_weight is None
            or threshold <= 0
            or int(rows) < threshold
            or self.bias is not None
            or not self.q_weight_row.is_cuda
            or self.weight_scale.dtype != torch.float16
        ):
            return None
        return (self.out_features, self.in_features)

    def rwkv7_prefill_dequant_weight(self, rows, out=None):
        """Fill the reusable fp16 operand used by native tensor-core prefill."""

        if self.rwkv7_prefill_dequant_shape(rows) is None:
            return None
        return sm70_w8_prefill_dequant_weight(
            self.q_weight_row,
            self.weight_scale,
            rows,
            out=out,
        )

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, dynamic_a8w8"


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
        setattr(parent, attribute, A8W8Linear(getattr(parent, attribute)))
    setattr(model, "_rwkv7_native_mm_quantization", "a8w8")
    setattr(model, "_rwkv7_native_mm_replaced_modules", len(targets))
    setattr(
        model,
        "_rwkv7_native_mm_quantized_head",
        any(name == "lm_head" or name.endswith(".lm_head") for name in targets),
    )
    setattr(
        model,
        "_rwkv7_native_mm_block_replaced_modules",
        sum(name.startswith("model.layers.") for name in targets),
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
