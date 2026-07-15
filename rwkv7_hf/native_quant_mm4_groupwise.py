# coding=utf-8
"""Default-off K-grouped affine W4 quality prototype.

This module is a correctness oracle, not a speed backend. It keeps independent
scale and bias values for each input-feature group and output column, then
materializes the dequantized weight for torch matmul. A fused kernel should only
be added after exact-model quality evidence justifies the format.
"""
from __future__ import annotations

try:  # pragma: no cover
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from .native_quant_policy import normalize_native_mm_policy, should_quantize_linear

try:  # pragma: no cover
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]

from .native_quant_mm4 import (
    _mm4_decode_blocks,
    _mm4_dot_blocks,
    _mm4_residual_ptr,
)


GROUPWISE_MM4_GROUP_SIZES = (32, 64, 128)
_HAS_TRITON = triton is not None and tl is not None


def _validate_group_size(group_size: int) -> int:
    value = int(group_size)
    if value not in GROUPWISE_MM4_GROUP_SIZES:
        raise ValueError(
            f"group_size must be one of {GROUPWISE_MM4_GROUP_SIZES}; got {value}"
        )
    return value


def quantize_groupwise_mm4(weight, *, group_size: int = 64, eps: float = 1e-8):
    """Quantize ``weight [K, N]`` with affine W4 groups along K."""

    if torch is None:
        raise RuntimeError("quantize_groupwise_mm4 requires torch")
    if weight.dim() != 2:
        raise ValueError(f"weight must be rank 2, got shape {tuple(weight.shape)}")
    group_size = _validate_group_size(group_size)
    k_orig, n_orig = (int(weight.shape[0]), int(weight.shape[1]))
    k_padded = ((k_orig + group_size - 1) // group_size) * group_size
    n_padded = ((n_orig + 1) // 2) * 2
    work = torch.zeros(
        (k_padded, n_padded), dtype=torch.float32, device=weight.device
    )
    work[:k_orig, :n_orig] = weight.float()
    grouped = work.reshape(k_padded // group_size, group_size, n_padded)
    biases = grouped.amin(dim=1)
    maxima = grouped.amax(dim=1)
    scales = ((maxima - biases) / 15.0).clamp(min=float(eps))
    q = torch.round((grouped - biases[:, None, :]) / scales[:, None, :])
    q = q.clamp(0, 15).to(torch.uint8).reshape(k_padded, n_padded)
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()
    out_dtype = weight.dtype
    return (
        packed,
        scales.to(out_dtype),
        biases.to(out_dtype),
        k_orig,
        n_orig,
        k_padded,
        n_padded,
        group_size,
    )


def dequantize_groupwise_mm4(
    packed,
    scales,
    biases,
    k_orig: int,
    n_orig: int,
    group_size: int,
    *,
    out_dtype=None,
):
    """Dequantize a packed groupwise W4 matrix to ``[K, N]``."""

    if torch is None:
        raise RuntimeError("dequantize_groupwise_mm4 requires torch")
    group_size = _validate_group_size(group_size)
    dtype = out_dtype if out_dtype is not None else scales.dtype
    k_padded, n_half = packed.shape
    n_padded = int(n_half) * 2
    q = torch.empty((k_padded, n_padded), dtype=dtype, device=packed.device)
    q[:, 0::2] = (packed & 0x0F).to(dtype)
    q[:, 1::2] = ((packed >> 4) & 0x0F).to(dtype)
    grouped = q.reshape(k_padded // group_size, group_size, n_padded)
    dense = grouped * scales[:, None, :].to(dtype) + biases[:, None, :].to(dtype)
    return dense.reshape(k_padded, n_padded)[: int(k_orig), : int(n_orig)]


def groupwise_mm4_matmul(x, packed, scales, biases, k_orig: int, n_orig: int, group_size: int):
    dense = dequantize_groupwise_mm4(
        packed,
        scales,
        biases,
        k_orig,
        n_orig,
        group_size,
        out_dtype=x.dtype,
    )
    return x @ dense


if _HAS_TRITON:

    @triton.jit
    def _groupwise_mm4_batched_gemv_kernel(
        x_ptr,
        packed_ptr,
        scales_ptr,
        biases_ptr,
        residual_ptr,
        y_ptr,
        B,
        K_ORIG,
        K_PADDED,
        M,
        MH,
        GROUP_SIZE: tl.constexpr,
        BLOCK_PAIRS: tl.constexpr,
        RELU2: tl.constexpr,
        ADD_RESIDUAL: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_m = tl.program_id(1)
        offs_p = pid_m * BLOCK_PAIRS + tl.arange(0, BLOCK_PAIRS)
        mask_p = offs_p < MH
        m0 = offs_p * 2
        m1 = m0 + 1
        mask0 = m0 < M
        mask1 = m1 < M
        offs_k = tl.arange(0, GROUP_SIZE)
        acc0 = tl.zeros((BLOCK_PAIRS,), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_PAIRS,), dtype=tl.float32)
        for k0 in range(0, K_PADDED, GROUP_SIZE):
            k = k0 + offs_k
            mask_k = k < K_ORIG
            x = tl.load(
                x_ptr + pid_b * K_ORIG + k,
                mask=mask_k,
                other=0.0,
            ).to(tl.float32)
            byte = tl.load(
                packed_ptr
                + k[:, None].to(tl.int64) * MH
                + offs_p[None, :].to(tl.int64),
                mask=(k < K_PADDED)[:, None] & mask_p[None, :],
                other=0,
            ).to(tl.int32)
            lo = (byte & 0xF).to(tl.float32)
            hi = ((byte >> 4) & 0xF).to(tl.float32)
            group_offset = (k0 // GROUP_SIZE) * M
            scale0 = tl.load(
                scales_ptr + group_offset + m0, mask=mask0, other=0.0
            ).to(tl.float32)
            scale1 = tl.load(
                scales_ptr + group_offset + m1, mask=mask1, other=0.0
            ).to(tl.float32)
            bias0 = tl.load(
                biases_ptr + group_offset + m0, mask=mask0, other=0.0
            ).to(tl.float32)
            bias1 = tl.load(
                biases_ptr + group_offset + m1, mask=mask1, other=0.0
            ).to(tl.float32)
            sum_x = tl.sum(x, axis=0)
            acc0 += tl.sum(x[:, None] * lo, axis=0) * scale0 + sum_x * bias0
            acc1 += tl.sum(x[:, None] * hi, axis=0) * scale1 + sum_x * bias1
        base = pid_b * M
        if RELU2:
            acc0 = tl.maximum(acc0, 0.0)
            acc1 = tl.maximum(acc1, 0.0)
            acc0 = acc0 * acc0
            acc1 = acc1 * acc1
        if ADD_RESIDUAL:
            acc0 += tl.load(
                residual_ptr + base + m0, mask=mask0, other=0.0
            ).to(tl.float32)
            acc1 += tl.load(
                residual_ptr + base + m1, mask=mask1, other=0.0
            ).to(tl.float32)
        tl.store(y_ptr + base + m0, acc0, mask=mask0)
        tl.store(y_ptr + base + m1, acc1, mask=mask1)

    @triton.jit
    def _groupwise_mm4_batched_dot_kernel(
        x_ptr,
        packed_ptr,
        scales_ptr,
        biases_ptr,
        residual_ptr,
        y_ptr,
        B,
        K_ORIG,
        K_PADDED,
        M,
        MH,
        GROUP_SIZE: tl.constexpr,
        BLOCK_B: tl.constexpr,
        BLOCK_PAIRS: tl.constexpr,
        RELU2: tl.constexpr,
        ADD_RESIDUAL: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_m = tl.program_id(1)
        offs_b = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
        offs_p = pid_m * BLOCK_PAIRS + tl.arange(0, BLOCK_PAIRS)
        mask_b = offs_b < B
        mask_p = offs_p < MH
        m0 = offs_p * 2
        m1 = m0 + 1
        mask0 = m0 < M
        mask1 = m1 < M
        offs_k = tl.arange(0, GROUP_SIZE)
        acc0 = tl.zeros((BLOCK_B, BLOCK_PAIRS), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_B, BLOCK_PAIRS), dtype=tl.float32)
        for k0 in range(0, K_PADDED, GROUP_SIZE):
            k = k0 + offs_k
            mask_k = k < K_ORIG
            x = tl.load(
                x_ptr + offs_b[:, None] * K_ORIG + k[None, :],
                mask=mask_b[:, None] & mask_k[None, :],
                other=0.0,
            ).to(tl.float16)
            byte = tl.load(
                packed_ptr
                + k[:, None].to(tl.int64) * MH
                + offs_p[None, :].to(tl.int64),
                mask=(k < K_PADDED)[:, None] & mask_p[None, :],
                other=0,
            ).to(tl.int32)
            lo = (byte & 0xF).to(tl.float16)
            hi = ((byte >> 4) & 0xF).to(tl.float16)
            group_offset = (k0 // GROUP_SIZE) * M
            scale0 = tl.load(
                scales_ptr + group_offset + m0, mask=mask0, other=0.0
            ).to(tl.float32)
            scale1 = tl.load(
                scales_ptr + group_offset + m1, mask=mask1, other=0.0
            ).to(tl.float32)
            bias0 = tl.load(
                biases_ptr + group_offset + m0, mask=mask0, other=0.0
            ).to(tl.float32)
            bias1 = tl.load(
                biases_ptr + group_offset + m1, mask=mask1, other=0.0
            ).to(tl.float32)
            sum_x = tl.sum(x.to(tl.float32), axis=1)
            acc0 += tl.dot(x, lo) * scale0[None, :] + sum_x[:, None] * bias0[None, :]
            acc1 += tl.dot(x, hi) * scale1[None, :] + sum_x[:, None] * bias1[None, :]
        base = offs_b[:, None] * M
        if RELU2:
            acc0 = tl.maximum(acc0, 0.0)
            acc1 = tl.maximum(acc1, 0.0)
            acc0 = acc0 * acc0
            acc1 = acc1 * acc1
        if ADD_RESIDUAL:
            acc0 += tl.load(
                residual_ptr + base + m0[None, :],
                mask=mask_b[:, None] & mask0[None, :],
                other=0.0,
            ).to(tl.float32)
            acc1 += tl.load(
                residual_ptr + base + m1[None, :],
                mask=mask_b[:, None] & mask1[None, :],
                other=0.0,
            ).to(tl.float32)
        tl.store(
            y_ptr + base + m0[None, :],
            acc0,
            mask=mask_b[:, None] & mask0[None, :],
        )
        tl.store(
            y_ptr + base + m1[None, :],
            acc1,
            mask=mask_b[:, None] & mask1[None, :],
        )


def groupwise_mm4_gemv_available(device=None) -> bool:
    if not (_HAS_TRITON and torch is not None and torch.cuda.is_available()):
        return False
    if device is None:
        return True
    return torch.device(device).type == "cuda"


def groupwise_mm4_matmul_triton(
    x,
    packed,
    scales,
    biases,
    k_orig: int,
    n_orig: int,
    group_size: int,
    *,
    max_gemv_rows: int = 16,
    relu2: bool = False,
    residual=None,
):
    """Fused decode W4A16 path for the groupwise quality format."""

    group_size = _validate_group_size(group_size)
    if not (x.is_cuda and groupwise_mm4_gemv_available(x.device)):
        out = groupwise_mm4_matmul(
            x, packed, scales, biases, k_orig, n_orig, group_size
        )
        out = torch.relu(out) ** 2 if relu2 else out
        return out + residual if residual is not None else out
    original_dim = x.dim()
    if original_dim == 1:
        x2 = x.reshape(1, -1)
        residual2 = None if residual is None else residual.reshape(1, -1)
    elif original_dim == 2:
        x2 = x
        residual2 = residual
    else:
        raise ValueError(f"x must be rank 1 or 2, got shape {tuple(x.shape)}")
    if int(x2.shape[1]) != int(k_orig):
        raise ValueError(f"input width {x2.shape[1]} does not match {k_orig}")
    if int(x2.shape[0]) > int(max_gemv_rows):
        out = groupwise_mm4_matmul(
            x2, packed, scales, biases, k_orig, n_orig, group_size
        )
        out = torch.relu(out) ** 2 if relu2 else out
        out = out + residual2 if residual2 is not None else out
        return out[0] if original_dim == 1 else out
    x2 = x2.contiguous()
    batch = int(x2.shape[0])
    k_padded, mh = (int(packed.shape[0]), int(packed.shape[1]))
    m_padded = mh * 2
    y = torch.empty((batch, m_padded), device=x.device, dtype=x.dtype)
    residual_ptr = _mm4_residual_ptr(residual2, y, n_orig, m_padded)
    blackwell = torch.cuda.get_device_capability(x.device)[0] >= 12
    use_dot = blackwell and batch >= 2 and x.dtype == torch.float16
    if use_dot:
        block_b, block_pairs, _ = _mm4_dot_blocks(
            x2, None, None, None, n_orig
        )
        grid = (triton.cdiv(batch, block_b), triton.cdiv(mh, block_pairs))
        _groupwise_mm4_batched_dot_kernel[grid](
            x2,
            packed,
            scales.reshape(-1),
            biases.reshape(-1),
            residual_ptr,
            y,
            batch,
            int(k_orig),
            k_padded,
            m_padded,
            mh,
            GROUP_SIZE=group_size,
            BLOCK_B=block_b,
            BLOCK_PAIRS=block_pairs,
            RELU2=bool(relu2),
            ADD_RESIDUAL=residual is not None,
            num_warps=8,
        )
    else:
        block_pairs, _ = _mm4_decode_blocks(x2, None, None)
        grid = (batch, triton.cdiv(mh, block_pairs))
        _groupwise_mm4_batched_gemv_kernel[grid](
            x2,
            packed,
            scales.reshape(-1),
            biases.reshape(-1),
            residual_ptr,
            y,
            batch,
            int(k_orig),
            k_padded,
            m_padded,
            mh,
            GROUP_SIZE=group_size,
            BLOCK_PAIRS=block_pairs,
            RELU2=bool(relu2),
            ADD_RESIDUAL=residual is not None,
            num_warps=4,
        )
    out = y[:, : int(n_orig)]
    return out[0] if original_dim == 1 else out


def groupwise_mm4_storage_bytes(packed, scales, biases, bias=None) -> int:
    tensors = (packed, scales, biases, bias)
    return sum(int(t.numel()) * int(t.element_size()) for t in tensors if t is not None)


class MM4GroupwiseLinear(torch.nn.Module):
    """Quality-first groupwise W4 replacement for ``nn.Linear``."""

    def __init__(self, linear, *, group_size: int = 64, fused: bool = True):
        super().__init__()
        self.in_features = int(linear.weight.shape[1])
        self.out_features = int(linear.weight.shape[0])
        values = quantize_groupwise_mm4(
            linear.weight.data.t().contiguous(), group_size=group_size
        )
        packed, scales, biases, k_orig, n_orig, _, _, resolved_group_size = values
        self.register_buffer("packed", packed)
        self.register_buffer("scales", scales)
        self.register_buffer("biases", biases)
        self.k_orig = int(k_orig)
        self.n_orig = int(n_orig)
        self.group_size = int(resolved_group_size)
        self.fused = bool(fused)
        if linear.bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", linear.bias.data.clone())

    def forward(self, x):
        leading = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features)
        if self.fused and x2.is_cuda and groupwise_mm4_gemv_available(x2.device):
            y = groupwise_mm4_matmul_triton(
                x2,
                self.packed,
                self.scales,
                self.biases,
                self.k_orig,
                self.n_orig,
                self.group_size,
            )
        else:
            y = groupwise_mm4_matmul(
                x2,
                self.packed,
                self.scales,
                self.biases,
                self.k_orig,
                self.n_orig,
                self.group_size,
            )
        y = y.reshape(*leading, self.out_features)
        return y if self.bias is None else y + self.bias

    def rwkv7_forward_relu2(self, x):
        if self.bias is not None or not (
            self.fused and x.is_cuda and groupwise_mm4_gemv_available(x.device)
        ):
            return torch.relu(self.forward(x)) ** 2
        leading = x.shape[:-1]
        y = groupwise_mm4_matmul_triton(
            x.reshape(-1, self.in_features),
            self.packed,
            self.scales,
            self.biases,
            self.k_orig,
            self.n_orig,
            self.group_size,
            relu2=True,
        )
        return y.reshape(*leading, self.out_features)

    def rwkv7_forward_add(self, x, residual):
        expected = (*x.shape[:-1], self.out_features)
        if tuple(residual.shape) != expected:
            raise ValueError(
                f"residual shape {tuple(residual.shape)} does not match {expected}"
            )
        if self.bias is not None or not (
            self.fused and x.is_cuda and groupwise_mm4_gemv_available(x.device)
        ):
            return residual + self.forward(x)
        leading = x.shape[:-1]
        y = groupwise_mm4_matmul_triton(
            x.reshape(-1, self.in_features),
            self.packed,
            self.scales,
            self.biases,
            self.k_orig,
            self.n_orig,
            self.group_size,
            residual=residual.reshape(-1, self.out_features),
        )
        return y.reshape(*leading, self.out_features)

    def extra_repr(self):
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"groupwise_mm4(group_size={self.group_size}, fused={self.fused})"
        )


def quantize_model_mm4_groupwise(
    model,
    *,
    min_params: int = 8_000_000,
    policy: str = "memory",
    group_size: int = 64,
    fused: bool = True,
) -> int:
    """Replace eligible linears with the default-off groupwise W4 oracle."""

    if torch is None:
        raise RuntimeError("quantize_model_mm4_groupwise requires torch")
    group_size = _validate_group_size(group_size)
    policy = normalize_native_mm_policy(policy)
    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and should_quantize_linear(
            name, int(module.weight.numel()), min_params=min_params, policy=policy
        )
    ]
    for full_name in targets:
        parent_name, _, attr = full_name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(
            parent,
            attr,
            MM4GroupwiseLinear(
                getattr(parent, attr), group_size=group_size, fused=fused
            ),
        )
    setattr(model, "_rwkv7_native_mm_quantization", "mm4_groupwise")
    setattr(model, "_rwkv7_native_mm_replaced_modules", len(targets))
    for cache_attr in (
        "_rwkv7_native_jit_pack_cache",
        "_rwkv7_native_graph_pack_cache",
        "_rwkv7_native_graph_runner_cache",
        "_rwkv7_native_prefill_graph_runner_cache",
        "_rwkv7_native_prefill_graph_hot_runner",
    ):
        if hasattr(model, cache_attr):
            delattr(model, cache_attr)
    return len(targets)
