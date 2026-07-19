# coding=utf-8
"""Opt-in official-order FP16 FFN norm/mix boundary for SM120 decode.

The CUDA reduction and half2 output order are derived from the Apache-2.0
RWKV-Gradio-3 ``rwkv7_v3a_ops`` implementation pinned by the official/native
alignment harness at commit ``cc57df475465c6cacd42ecd4f2f05a588ee5473b``.
Only the FFN residual-add, layer-normalization, mix, and shift-state boundary is
included here; dense projections and recurrence remain repository-native.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import threading
from typing import Any

try:  # pragma: no cover - optional in lightweight environments
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


_CPP_SOURCE = r"""
#include <torch/extension.h>

std::vector<torch::Tensor> rwkv7_blackwell_ffn_add_norm_mix_cuda(
    torch::Tensor residual, torch::Tensor attention, torch::Tensor previous,
    torch::Tensor weight, torch::Tensor bias, torch::Tensor mix, double eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ffn_add_norm_mix", &rwkv7_blackwell_ffn_add_norm_mix_cuda,
        "Official-order FP16 FFN residual add, layer norm, and time mix");
}
"""


_CUDA_SOURCE = r"""
#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF_OPERATORS__

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

namespace {

constexpr int THREADS = 256;

__device__ __forceinline__ float warp_sum(float value) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

template <int BlockThreads>
__device__ __forceinline__ float block_sum(float value) {
  __shared__ float partial[BlockThreads / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = warp_sum(value);
  if (lane == 0) partial[warp] = value;
  __syncthreads();
  value = threadIdx.x < BlockThreads / 32 ? partial[lane] : 0.0f;
  if (warp == 0) value = warp_sum(value);
  if (threadIdx.x == 0) partial[0] = value;
  __syncthreads();
  return partial[0];
}

template <int BlockThreads>
__global__ __launch_bounds__(BlockThreads, 1) void ffn_add_norm_mix_kernel(
    const half* __restrict__ residual,
    const half* __restrict__ attention,
    half* __restrict__ previous,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    const half* __restrict__ mix,
    half* __restrict__ residual_out,
    half* __restrict__ mixed,
    int rows,
    int hidden,
    float eps) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int64_t base = static_cast<int64_t>(row) * hidden;
  float sum = 0.0f;
  for (int c = threadIdx.x; c < hidden; c += BlockThreads) {
    sum += __half2float(residual[base + c]) + __half2float(attention[base + c]);
  }
  const float mean = block_sum<BlockThreads>(sum) / static_cast<float>(hidden);
  float sum_var = 0.0f;
  for (int c = threadIdx.x; c < hidden; c += BlockThreads) {
    const float value = __half2float(residual[base + c])
                        + __half2float(attention[base + c]);
    const float delta = value - mean;
    sum_var += delta * delta;
  }
  const float rstd = rsqrtf(
      block_sum<BlockThreads>(sum_var) / static_cast<float>(hidden) + eps);
  const int pairs = hidden >> 1;
  const int64_t pair_base = base >> 1;
  for (int p = threadIdx.x; p < pairs; p += BlockThreads) {
    const float2 x = __half22float2(reinterpret_cast<const half2*>(residual)[pair_base + p]);
    const float2 a = __half22float2(reinterpret_cast<const half2*>(attention)[pair_base + p]);
    const float2 w = __half22float2(reinterpret_cast<const half2*>(weight)[p]);
    const float2 b = __half22float2(reinterpret_cast<const half2*>(bias)[p]);
    const float2 prev = __half22float2(reinterpret_cast<const half2*>(previous)[pair_base + p]);
    const float2 m = __half22float2(reinterpret_cast<const half2*>(mix)[p]);
    const float x0 = x.x + a.x;
    const float x1 = x.y + a.y;
    const half2 normalized = __floats2half2_rn(
        (x0 - mean) * rstd * w.x + b.x,
        (x1 - mean) * rstd * w.y + b.y);
    const float2 n = __half22float2(normalized);
    reinterpret_cast<half2*>(residual_out)[pair_base + p] =
        __floats2half2_rn(x0, x1);
    reinterpret_cast<half2*>(mixed)[pair_base + p] = __floats2half2_rn(
        n.x + (prev.x - n.x) * m.x,
        n.y + (prev.y - n.y) * m.y);
    reinterpret_cast<half2*>(previous)[pair_base + p] = normalized;
  }
}

void check_half_cuda(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.scalar_type() == at::kHalf, name, " must be fp16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

std::vector<torch::Tensor> rwkv7_blackwell_ffn_add_norm_mix_cuda(
    torch::Tensor residual, torch::Tensor attention, torch::Tensor previous,
    torch::Tensor weight, torch::Tensor bias, torch::Tensor mix, double eps) {
  check_half_cuda(residual, "residual");
  check_half_cuda(attention, "attention");
  check_half_cuda(previous, "previous");
  check_half_cuda(weight, "weight");
  check_half_cuda(bias, "bias");
  check_half_cuda(mix, "mix");
  TORCH_CHECK(residual.dim() == 1 || residual.dim() == 2,
              "residual must be [hidden] or [batch,hidden]");
  TORCH_CHECK(attention.sizes() == residual.sizes(), "attention shape mismatch");
  TORCH_CHECK(previous.sizes() == residual.sizes(), "previous shape mismatch");
  const int hidden = static_cast<int>(residual.size(-1));
  const int rows = static_cast<int>(residual.numel() / hidden);
  TORCH_CHECK(hidden > 0 && (hidden % 2) == 0, "hidden must be positive and even");
  for (const auto& tensor : {weight, bias, mix}) {
    TORCH_CHECK(tensor.numel() == hidden, "parameter shape mismatch");
  }
  c10::cuda::CUDAGuard guard(residual.device());
  auto residual_out = torch::empty_like(residual);
  auto mixed = torch::empty_like(residual);
  auto stream = at::cuda::getCurrentCUDAStream(residual.get_device());
  ffn_add_norm_mix_kernel<THREADS><<<rows, THREADS, 0, stream>>>(
      reinterpret_cast<const half*>(residual.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(attention.data_ptr<at::Half>()),
      reinterpret_cast<half*>(previous.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(bias.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(mix.data_ptr<at::Half>()),
      reinterpret_cast<half*>(residual_out.data_ptr<at::Half>()),
      reinterpret_cast<half*>(mixed.data_ptr<at::Half>()),
      rows, hidden, static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {residual_out, mixed};
}
"""


_EXTENSION: Any | None = None
_EXTENSION_ERROR: str | None = None
_EXTENSION_LOCK = threading.Lock()


def blackwell_norm_mix_should_use(residual: Any, attention: Any, previous: Any) -> bool:
    if torch is None or not residual.is_cuda:
        return False
    major, _minor = torch.cuda.get_device_capability(residual.device)
    return bool(
        major >= 12
        and residual.dtype == torch.float16
        and residual.dim() in (1, 2)
        and int(residual.shape[-1]) > 0
        and int(residual.shape[-1]) % 2 == 0
        and attention.shape == residual.shape
        and previous.shape == residual.shape
    )


def _load_extension() -> Any | None:
    global _EXTENSION, _EXTENSION_ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    if _EXTENSION_ERROR is not None or torch is None or not torch.cuda.is_available():
        return None
    with _EXTENSION_LOCK:
        if _EXTENSION is not None:
            return _EXTENSION
        if _EXTENSION_ERROR is not None:
            return None
        try:
            python_bin = str(Path(sys.executable).absolute().parent)
            if python_bin not in os.environ.get("PATH", "").split(os.pathsep):
                os.environ["PATH"] = python_bin + os.pathsep + os.environ.get("PATH", "")
            capability = torch.cuda.get_device_capability()
            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{capability[0]}.{capability[1]}")
            from torch.utils.cpp_extension import load_inline

            _EXTENSION = load_inline(
                name="rwkv7_blackwell_norm_mix_v1",
                cpp_sources=_CPP_SOURCE,
                cuda_sources=_CUDA_SOURCE,
                functions=None,
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3", "--extra-device-vectorization"],
                with_cuda=True,
                verbose=os.environ.get("RWKV7_BLACKWELL_NORM_MIX_BUILD_VERBOSE", "0").lower()
                in {"1", "true", "yes", "on"},
            )
        except Exception as exc:  # pragma: no cover - host toolchain dependent
            _EXTENSION_ERROR = f"{type(exc).__name__}: {exc}"
            return None
    return _EXTENSION


def blackwell_norm_mix_available(*, build: bool = False) -> bool:
    if torch is None or not torch.cuda.is_available():
        return False
    return _load_extension() is not None if build else True


def blackwell_norm_mix_build_error() -> str | None:
    return _EXTENSION_ERROR


def blackwell_ffn_add_norm_mix(
    residual: Any,
    attention: Any,
    previous: Any,
    weight: Any,
    bias: Any,
    mix: Any,
    *,
    eps: float = 1.0e-5,
) -> tuple[Any, Any]:
    if not blackwell_norm_mix_should_use(residual, attention, previous):
        raise ValueError("SM120 norm/mix received an unsupported device, dtype, or shape")
    tensors = (residual, attention, previous, weight, bias, mix)
    if not all(item.is_contiguous() and item.dtype == torch.float16 for item in tensors):
        raise ValueError("SM120 norm/mix requires contiguous CUDA fp16 tensors")
    extension = _load_extension()
    if extension is None:
        raise RuntimeError(
            "SM120 norm/mix extension is unavailable: "
            f"{blackwell_norm_mix_build_error()}"
        )
    residual_out, mixed = extension.ffn_add_norm_mix(
        residual, attention, previous, weight, bias, mix, float(eps)
    )
    return residual_out, mixed


__all__ = [
    "blackwell_ffn_add_norm_mix",
    "blackwell_norm_mix_available",
    "blackwell_norm_mix_build_error",
    "blackwell_norm_mix_should_use",
]
