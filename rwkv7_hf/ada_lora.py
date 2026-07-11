# coding=utf-8
"""Optional RTX 4090 fused W/A/G/V low-rank decode kernels.

The layer>0 RWKV-7 time-mix path contains four independent rank-in projections
and four rank-out projections.  For one to four decode rows, launching each as
a separate GEMV plus separate tanh/sigmoid/interpolation kernels is expensive.
This module groups rank-in into one launch and rank-out, activations, biases,
and V interpolation into a second launch.

The CUDA implementation is derived from Albatross' Apache-2.0
``linear_wagv_rank_{in,out}_f16_kernel``.  It uses the HF ``nn.Linear`` weight
layouts directly, adds no packed copy, is inference-only, and falls back to
ordinary PyTorch for every unsupported device, dtype, shape, or build failure.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import threading
from typing import Any

try:  # pragma: no cover - optional in lightweight environments
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> rwkv7_ada_wagv_rank_in_cuda(
    torch::Tensor xw, torch::Tensor xa, torch::Tensor xg, torch::Tensor xv,
    torch::Tensor w1, torch::Tensor a1, torch::Tensor g1, torch::Tensor v1,
    bool compute_v);
std::vector<torch::Tensor> rwkv7_ada_wagv_rank_out_cuda(
    torch::Tensor wh, torch::Tensor ah, torch::Tensor gh, torch::Tensor vh,
    torch::Tensor w2, torch::Tensor a2, torch::Tensor g2, torch::Tensor v2,
    torch::Tensor w0, torch::Tensor a0, torch::Tensor v0,
    torch::Tensor v, torch::Tensor v_first, bool sigmoid_a, bool compute_v);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rank_in", &rwkv7_ada_wagv_rank_in_cuda,
        "RWKV-7 Ada fused W/A/G/V rank-in");
  m.def("rank_out", &rwkv7_ada_wagv_rank_out_cuda,
        "RWKV-7 Ada fused W/A/G/V rank-out and V interpolation");
}
"""


_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <vector>

namespace {

template <typename T>
__device__ __forceinline__ float load_float(const T* pointer);
template <>
__device__ __forceinline__ float load_float<half>(const half* pointer) {
  return __half2float(*pointer);
}
template <>
__device__ __forceinline__ float load_float<nv_bfloat16>(const nv_bfloat16* pointer) {
  return __bfloat162float(*pointer);
}

template <typename T>
__device__ __forceinline__ float2 load_float2(const T* pointer);
template <>
__device__ __forceinline__ float2 load_float2<half>(const half* pointer) {
  return __half22float2(*reinterpret_cast<const half2*>(pointer));
}
template <>
__device__ __forceinline__ float2 load_float2<nv_bfloat16>(const nv_bfloat16* pointer) {
  const nv_bfloat162 value = *reinterpret_cast<const nv_bfloat162*>(pointer);
  return make_float2(__bfloat162float(value.x), __bfloat162float(value.y));
}

template <typename T>
__device__ __forceinline__ T store_float(float value);
template <>
__device__ __forceinline__ half store_float<half>(float value) {
  return __float2half_rn(value);
}
template <>
__device__ __forceinline__ nv_bfloat16 store_float<nv_bfloat16>(float value) {
  return __float2bfloat16_rn(value);
}

__device__ __forceinline__ float warp_sum(float value) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

template <int Threads>
__device__ __forceinline__ float block_sum(float value) {
  __shared__ float partial[Threads / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = warp_sum(value);
  if (lane == 0) partial[warp] = value;
  __syncthreads();
  value = threadIdx.x < Threads / 32 ? partial[lane] : 0.0f;
  if (warp == 0) value = warp_sum(value);
  if (threadIdx.x == 0) partial[0] = value;
  __syncthreads();
  return partial[0];
}

template <typename scalar_t, int Threads>
__global__ __launch_bounds__(Threads, 2) void wagv_rank_in_kernel(
    int rows,
    int hidden,
    int rw,
    int ra,
    int rg,
    int rv,
    int max_rank,
    const scalar_t* __restrict__ xw,
    const scalar_t* __restrict__ xa,
    const scalar_t* __restrict__ xg,
    const scalar_t* __restrict__ xv,
    const scalar_t* __restrict__ w1,
    const scalar_t* __restrict__ a1,
    const scalar_t* __restrict__ g1,
    const scalar_t* __restrict__ v1,
    scalar_t* __restrict__ wh,
    scalar_t* __restrict__ ah,
    scalar_t* __restrict__ gh,
    scalar_t* __restrict__ vh) {
  const int rank_index = blockIdx.x;
  const int row = blockIdx.y;
  const int group = blockIdx.z;
  int rank = rw;
  const scalar_t* input = xw;
  const scalar_t* weight = w1;
  scalar_t* output = wh;
  if (group == 1) {
    rank = ra; input = xa; weight = a1; output = ah;
  } else if (group == 2) {
    rank = rg; input = xg; weight = g1; output = gh;
  } else if (group == 3) {
    rank = rv; input = xv; weight = v1; output = vh;
  }
  if (row >= rows || rank_index >= rank || rank_index >= max_rank) return;

  const scalar_t* input_row = input + static_cast<int64_t>(row) * hidden;
  const scalar_t* weight_row = weight + static_cast<int64_t>(rank_index) * hidden;
  float accumulator = 0.0f;
  for (int pair = threadIdx.x; pair < hidden / 2; pair += Threads) {
    const float2 activation = load_float2(input_row + pair * 2);
    const float2 coefficient = load_float2(weight_row + pair * 2);
    accumulator = fmaf(activation.x, coefficient.x, accumulator);
    accumulator = fmaf(activation.y, coefficient.y, accumulator);
  }
  accumulator = block_sum<Threads>(accumulator);
  if (threadIdx.x == 0) {
    output[static_cast<int64_t>(row) * rank + rank_index] = store_float<scalar_t>(accumulator);
  }
}

template <typename scalar_t, int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 2) void wagv_rank_out_kernel(
    int rows,
    int hidden,
    int rw,
    int ra,
    int rg,
    int rv,
    const scalar_t* __restrict__ wh,
    const scalar_t* __restrict__ ah,
    const scalar_t* __restrict__ gh,
    const scalar_t* __restrict__ vh,
    const scalar_t* __restrict__ w2,
    const scalar_t* __restrict__ a2,
    const scalar_t* __restrict__ g2,
    const scalar_t* __restrict__ v2,
    const scalar_t* __restrict__ w0,
    const scalar_t* __restrict__ a0,
    const scalar_t* __restrict__ v0,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ v_first,
    scalar_t* __restrict__ w,
    scalar_t* __restrict__ a,
    scalar_t* __restrict__ g,
    scalar_t* __restrict__ v_out,
    bool sigmoid_a) {
  const int hidden_start = blockIdx.x * OutTile;
  const int row = blockIdx.y;
  const int group = blockIdx.z;
  int rank = rw;
  const scalar_t* input = wh;
  const scalar_t* weight = w2;
  scalar_t* output = w;
  if (group == 1) {
    rank = ra; input = ah; weight = a2; output = a;
  } else if (group == 2) {
    rank = rg; input = gh; weight = g2; output = g;
  } else if (group == 3) {
    rank = rv; input = vh; weight = v2; output = v_out;
  }
  if (row >= rows) return;

  float accumulators[OutTile];
  #pragma unroll
  for (int out = 0; out < OutTile; ++out) accumulators[out] = 0.0f;
  const scalar_t* input_row = input + static_cast<int64_t>(row) * rank;
  for (int k = threadIdx.x; k < rank; k += Threads) {
    float activation = load_float(input_row + k);
    if (group == 0) {
      activation = tanhf(activation);
    } else if (group == 2) {
      activation = 1.0f / (1.0f + expf(-activation));
    }
    #pragma unroll
    for (int out = 0; out < OutTile; ++out) {
      const int hidden_index = hidden_start + out;
      if (hidden_index < hidden) {
        accumulators[out] = fmaf(
            activation,
            load_float(weight + static_cast<int64_t>(hidden_index) * rank + k),
            accumulators[out]);
      }
    }
  }

  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  #pragma unroll
  for (int out = 0; out < OutTile; ++out) {
    accumulators[out] = warp_sum(accumulators[out]);
    if (lane == 0) partial[warp][out] = accumulators[out];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    #pragma unroll
    for (int out = 0; out < OutTile; ++out) {
      const int hidden_index = hidden_start + out;
      if (hidden_index < hidden) {
        float sum = 0.0f;
        #pragma unroll
        for (int warp_index = 0; warp_index < Threads / 32; ++warp_index) {
          sum += partial[warp_index][out];
        }
        const int64_t index = static_cast<int64_t>(row) * hidden + hidden_index;
        if (group == 0) {
          output[index] = store_float<scalar_t>(sum + load_float(w0 + hidden_index));
        } else if (group == 1) {
          float value = sum + load_float(a0 + hidden_index);
          if (sigmoid_a) value = 1.0f / (1.0f + expf(-value));
          output[index] = store_float<scalar_t>(value);
        } else if (group == 3) {
          const float current = load_float(v + index);
          const float first = load_float(v_first + index);
          const float gate = 1.0f / (1.0f + expf(-(load_float(v0 + hidden_index) + sum)));
          output[index] = store_float<scalar_t>(current + (first - current) * gate);
        } else {
          output[index] = store_float<scalar_t>(sum);
        }
      }
    }
  }
}

void check_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.scalar_type() == at::kHalf || tensor.scalar_type() == at::kBFloat16,
              name, " must be fp16 or bf16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

std::vector<torch::Tensor> rwkv7_ada_wagv_rank_in_cuda(
    torch::Tensor xw, torch::Tensor xa, torch::Tensor xg, torch::Tensor xv,
    torch::Tensor w1, torch::Tensor a1, torch::Tensor g1, torch::Tensor v1,
    bool compute_v) {
  check_tensor(xw, "xw"); check_tensor(xa, "xa");
  check_tensor(xg, "xg"); check_tensor(xv, "xv");
  check_tensor(w1, "w1"); check_tensor(a1, "a1");
  check_tensor(g1, "g1"); check_tensor(v1, "v1");
  TORCH_CHECK(xw.dim() == 2 && xa.sizes() == xw.sizes() && xg.sizes() == xw.sizes()
              && xv.sizes() == xw.sizes(), "rank-in inputs must share [rows, hidden]");
  const int rows = static_cast<int>(xw.size(0));
  const int hidden = static_cast<int>(xw.size(1));
  TORCH_CHECK(rows >= 1 && rows <= 4 && hidden >= 1024 && hidden % 2 == 0,
              "rank-in supports rows 1..4 and even hidden >= 1024");
  TORCH_CHECK(w1.dim() == 2 && a1.dim() == 2 && g1.dim() == 2 && v1.dim() == 2,
              "rank-in weights must be rank-2");
  TORCH_CHECK(w1.size(1) == hidden && a1.size(1) == hidden && g1.size(1) == hidden
              && v1.size(1) == hidden, "rank-in hidden mismatch");
  const int rw = static_cast<int>(w1.size(0));
  const int ra = static_cast<int>(a1.size(0));
  const int rg = static_cast<int>(g1.size(0));
  const int rv = static_cast<int>(v1.size(0));
  const int max_rank = std::max(std::max(rw, ra), std::max(rg, rv));
  TORCH_CHECK(max_rank > 0 && max_rank <= 512, "rank-in rank must be 1..512");

  c10::cuda::CUDAGuard guard(xw.device());
  auto wh = torch::empty({rows, rw}, xw.options());
  auto ah = torch::empty({rows, ra}, xw.options());
  auto gh = torch::empty({rows, rg}, xw.options());
  auto vh = torch::empty({rows, rv}, xw.options());
  auto stream = at::cuda::getCurrentCUDAStream(xw.get_device());
  if (xw.scalar_type() == at::kHalf) {
    wagv_rank_in_kernel<half, 256><<<dim3(max_rank, rows, compute_v ? 4 : 3), 256, 0, stream>>>(
        rows, hidden, rw, ra, rg, rv, max_rank,
        reinterpret_cast<const half*>(xw.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(xa.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(xg.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(xv.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(w1.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(a1.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(g1.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v1.data_ptr<at::Half>()),
        reinterpret_cast<half*>(wh.data_ptr<at::Half>()),
        reinterpret_cast<half*>(ah.data_ptr<at::Half>()),
        reinterpret_cast<half*>(gh.data_ptr<at::Half>()),
        reinterpret_cast<half*>(vh.data_ptr<at::Half>()));
  } else {
    wagv_rank_in_kernel<nv_bfloat16, 256><<<dim3(max_rank, rows, compute_v ? 4 : 3), 256, 0, stream>>>(
        rows, hidden, rw, ra, rg, rv, max_rank,
        reinterpret_cast<const nv_bfloat16*>(xw.data_ptr<at::BFloat16>()),
        reinterpret_cast<const nv_bfloat16*>(xa.data_ptr<at::BFloat16>()),
        reinterpret_cast<const nv_bfloat16*>(xg.data_ptr<at::BFloat16>()),
        reinterpret_cast<const nv_bfloat16*>(xv.data_ptr<at::BFloat16>()),
        reinterpret_cast<const nv_bfloat16*>(w1.data_ptr<at::BFloat16>()),
        reinterpret_cast<const nv_bfloat16*>(a1.data_ptr<at::BFloat16>()),
        reinterpret_cast<const nv_bfloat16*>(g1.data_ptr<at::BFloat16>()),
        reinterpret_cast<const nv_bfloat16*>(v1.data_ptr<at::BFloat16>()),
        reinterpret_cast<nv_bfloat16*>(wh.data_ptr<at::BFloat16>()),
        reinterpret_cast<nv_bfloat16*>(ah.data_ptr<at::BFloat16>()),
        reinterpret_cast<nv_bfloat16*>(gh.data_ptr<at::BFloat16>()),
        reinterpret_cast<nv_bfloat16*>(vh.data_ptr<at::BFloat16>()));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {wh, ah, gh, vh};
}

std::vector<torch::Tensor> rwkv7_ada_wagv_rank_out_cuda(
    torch::Tensor wh, torch::Tensor ah, torch::Tensor gh, torch::Tensor vh,
    torch::Tensor w2, torch::Tensor a2, torch::Tensor g2, torch::Tensor v2,
    torch::Tensor w0, torch::Tensor a0, torch::Tensor v0,
    torch::Tensor v, torch::Tensor v_first, bool sigmoid_a, bool compute_v) {
  check_tensor(wh, "wh"); check_tensor(ah, "ah");
  check_tensor(gh, "gh"); check_tensor(vh, "vh");
  check_tensor(w2, "w2"); check_tensor(a2, "a2");
  check_tensor(g2, "g2"); check_tensor(v2, "v2");
  check_tensor(w0, "w0"); check_tensor(a0, "a0"); check_tensor(v0, "v0");
  check_tensor(v, "v"); check_tensor(v_first, "v_first");
  TORCH_CHECK(wh.dim() == 2 && ah.dim() == 2 && gh.dim() == 2 && vh.dim() == 2,
              "rank-out inputs must be rank-2");
  const int rows = static_cast<int>(wh.size(0));
  const int hidden = static_cast<int>(w2.size(0));
  TORCH_CHECK(rows >= 1 && rows <= 4 && hidden >= 1024 && hidden % 4 == 0,
              "rank-out supports rows 1..4 and hidden divisible by four");
  TORCH_CHECK(ah.size(0) == rows && gh.size(0) == rows && vh.size(0) == rows,
              "rank-out row mismatch");
  TORCH_CHECK(w2.dim() == 2 && a2.dim() == 2 && g2.dim() == 2 && v2.dim() == 2,
              "rank-out weights must be rank-2");
  TORCH_CHECK(w2.size(1) == wh.size(1) && a2.size(0) == hidden && a2.size(1) == ah.size(1)
              && g2.size(0) == hidden && g2.size(1) == gh.size(1)
              && v2.size(0) == hidden && v2.size(1) == vh.size(1), "rank-out weight mismatch");
  TORCH_CHECK(w0.numel() == hidden && a0.numel() == hidden && v0.numel() == hidden,
              "rank-out bias mismatch");
  TORCH_CHECK(v.dim() == 2 && v.size(0) == rows && v.size(1) == hidden
              && v_first.sizes() == v.sizes(),
              "V tensors must have [rows, hidden] shape");

  c10::cuda::CUDAGuard guard(wh.device());
  auto w = torch::empty({rows, hidden}, wh.options());
  auto a = torch::empty_like(w);
  auto g = torch::empty_like(w);
  // Keep the custom-op outputs non-aliasing even when the V branch is skipped.
  // CUDA graph pools may retain several batch-size captures concurrently and
  // cannot infer pybind-only input/output aliasing from a function schema.
  auto v_out = torch::empty_like(w);
  auto stream = at::cuda::getCurrentCUDAStream(wh.get_device());
  if (wh.scalar_type() == at::kHalf) {
    wagv_rank_out_kernel<half, 128, 4><<<dim3(hidden / 4, rows, compute_v ? 4 : 3), 128, 0, stream>>>(
        rows, hidden, static_cast<int>(wh.size(1)), static_cast<int>(ah.size(1)),
        static_cast<int>(gh.size(1)), static_cast<int>(vh.size(1)),
        reinterpret_cast<const half*>(wh.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(ah.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(gh.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(vh.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(w2.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(a2.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(g2.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v2.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(w0.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(a0.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v0.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v_first.data_ptr<at::Half>()),
        reinterpret_cast<half*>(w.data_ptr<at::Half>()),
        reinterpret_cast<half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<half*>(g.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v_out.data_ptr<at::Half>()),
        sigmoid_a);
  } else {
    wagv_rank_out_kernel<nv_bfloat16, 128, 4><<<dim3(hidden / 4, rows, compute_v ? 4 : 3), 128, 0, stream>>>(
        rows, hidden, static_cast<int>(wh.size(1)), static_cast<int>(ah.size(1)),
        static_cast<int>(gh.size(1)), static_cast<int>(vh.size(1)),
        reinterpret_cast<const nv_bfloat16*>(wh.data_ptr<at::BFloat16>()),
        reinterpret_cast<const nv_bfloat16*>(ah.data_ptr<at::BFloat16>()),
        reinterpret_cast<const nv_bfloat16*>(gh.data_ptr<at::BFloat16>()),
        reinterpret_cast<const nv_bfloat16*>(vh.data_ptr<at::BFloat16>()),
        reinterpret_cast<const nv_bfloat16*>(w2.data_ptr<at::BFloat16>()),
        reinterpret_cast<const nv_bfloat16*>(a2.data_ptr<at::BFloat16>()),
        reinterpret_cast<const nv_bfloat16*>(g2.data_ptr<at::BFloat16>()),
        reinterpret_cast<const nv_bfloat16*>(v2.data_ptr<at::BFloat16>()),
        reinterpret_cast<const nv_bfloat16*>(w0.data_ptr<at::BFloat16>()),
        reinterpret_cast<const nv_bfloat16*>(a0.data_ptr<at::BFloat16>()),
        reinterpret_cast<const nv_bfloat16*>(v0.data_ptr<at::BFloat16>()),
        reinterpret_cast<const nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
        reinterpret_cast<const nv_bfloat16*>(v_first.data_ptr<at::BFloat16>()),
        reinterpret_cast<nv_bfloat16*>(w.data_ptr<at::BFloat16>()),
        reinterpret_cast<nv_bfloat16*>(a.data_ptr<at::BFloat16>()),
        reinterpret_cast<nv_bfloat16*>(g.data_ptr<at::BFloat16>()),
        reinterpret_cast<nv_bfloat16*>(v_out.data_ptr<at::BFloat16>()),
        sigmoid_a);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {w, a, g, v_out};
}
"""


_EXTENSION: Any | None = None
_EXTENSION_ERROR: str | None = None
_EXTENSION_LOCK = threading.Lock()


def _is_sm89(device: Any = None) -> bool:
    if torch is None or not torch.cuda.is_available():
        return False
    try:
        resolved = torch.device("cuda" if device is None else device)
        index = torch.cuda.current_device() if resolved.index is None else int(resolved.index)
        return tuple(int(v) for v in torch.cuda.get_device_capability(index)) == (8, 9)
    except Exception:
        return False


def _load_extension() -> Any | None:
    global _EXTENSION, _EXTENSION_ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    if _EXTENSION_ERROR is not None or torch is None or not _is_sm89():
        return None
    with _EXTENSION_LOCK:
        if _EXTENSION is not None:
            return _EXTENSION
        if _EXTENSION_ERROR is not None:
            return None
        try:
            python_bin = str(Path(sys.executable).resolve().parent)
            if python_bin not in os.environ.get("PATH", "").split(os.pathsep):
                os.environ["PATH"] = python_bin + os.pathsep + os.environ.get("PATH", "")
            nvcc = Path(python_bin) / "nvcc"
            if nvcc.exists() and "CUDA_HOME" not in os.environ:
                os.environ["CUDA_HOME"] = str(nvcc.parent.parent)
            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
            from torch.utils.cpp_extension import load_inline

            _EXTENSION = load_inline(
                name="rwkv7_ada_lora_v6",
                cpp_sources=_CPP_SOURCE,
                cuda_sources=_CUDA_SOURCE,
                functions=None,
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3", "--use_fast_math", "--extra-device-vectorization"],
                with_cuda=True,
                verbose=os.environ.get("RWKV7_ADA_LORA_BUILD_VERBOSE", "0").lower()
                in {"1", "true", "yes", "on"},
            )
        except Exception as exc:  # pragma: no cover - host toolchain dependent
            _EXTENSION_ERROR = f"{type(exc).__name__}: {exc}"
            return None
    return _EXTENSION


def ada_wagv_lora_available(device: Any = None, *, build: bool = False) -> bool:
    if not _is_sm89(device):
        return False
    return _load_extension() is not None if build else True


def ada_wagv_lora_build_error() -> str | None:
    return _EXTENSION_ERROR


def ada_wagv_lora_should_use(rows: int, hidden: int, max_rank: int) -> bool:
    return 1 <= int(rows) <= 4 and int(hidden) >= 1024 and int(hidden) % 4 == 0 and 1 <= int(max_rank) <= 512


def _fallback(
    xw, xa, xg, xv, w1, a1, g1, v1, w2, a2, g2, v2, w0, a0, v0, v, v_first
):
    w = F.linear(torch.tanh(F.linear(xw, w1)), w2, w0)
    a = F.linear(F.linear(xa, a1), a2, a0)
    g = F.linear(torch.sigmoid(F.linear(xg, g1)), g2)
    gate = torch.sigmoid(F.linear(F.linear(xv, v1), v2, v0))
    return w, a, g, v + (v_first - v) * gate


def ada_wagv_lora(
    xw: Any,
    xa: Any,
    xg: Any,
    xv: Any,
    w1: Any,
    a1: Any,
    g1: Any,
    v1: Any,
    w2: Any,
    a2: Any,
    g2: Any,
    v2: Any,
    w0: Any,
    a0: Any,
    v0: Any,
    v: Any,
    v_first: Any,
    *,
    sigmoid_a: bool = False,
    compute_v: bool = True,
    force_fallback: bool = False,
) -> tuple[Any, Any, Any, Any]:
    """Return grouped W/A/G/V outputs for layer>0 decode.

    ``sigmoid_a=True`` folds the A-gate sigmoid into the rank-out kernel and
    avoids a separate pointwise launch in the captured decode graph.
    """

    if torch is None or F is None:
        raise RuntimeError("ada_wagv_lora requires torch")
    scalar = xw.dim() == 1
    tensors = [xw, xa, xg, xv, v, v_first]
    flat = [item.reshape(1, -1) if scalar else item for item in tensors]
    xw2, xa2, xg2, xv2, v_current, v_first2 = flat
    rows, hidden = int(xw2.shape[0]), int(xw2.shape[1])
    max_rank = max(int(item.shape[0]) for item in (w1, a1, g1, v1))
    all_tensors = flat + [w1, a1, g1, v1, w2, a2, g2, v2, w0, a0, v0]
    valid = bool(
        not force_fallback
        and not torch.is_grad_enabled()
        and ada_wagv_lora_should_use(rows, hidden, max_rank)
        and xw2.dtype in {torch.float16, torch.bfloat16}
        and all(item.is_cuda and item.dtype == xw2.dtype and item.is_contiguous() for item in all_tensors)
        and all(tuple(item.shape) == (rows, hidden) for item in flat)
        and all(int(item.shape[1]) == hidden for item in (w1, a1, g1, v1))
        and tuple(w2.shape) == (hidden, int(w1.shape[0]))
        and tuple(a2.shape) == (hidden, int(a1.shape[0]))
        and tuple(g2.shape) == (hidden, int(g1.shape[0]))
        and tuple(v2.shape) == (hidden, int(v1.shape[0]))
        and all(int(item.numel()) == hidden for item in (w0, a0, v0))
        and _is_sm89(xw2.device)
    )
    extension = _load_extension() if valid else None
    if extension is None:
        if compute_v:
            outputs = _fallback(
                xw2, xa2, xg2, xv2, w1, a1, g1, v1, w2, a2, g2, v2,
                w0, a0, v0, v_current, v_first2,
            )
        else:
            outputs = (
                F.linear(torch.tanh(F.linear(xw2, w1)), w2, w0),
                F.linear(F.linear(xa2, a1), a2, a0),
                F.linear(torch.sigmoid(F.linear(xg2, g1)), g2),
                v_current,
            )
        if sigmoid_a:
            outputs = (outputs[0], torch.sigmoid(outputs[1]), outputs[2], outputs[3])
    else:
        hidden_states = extension.rank_in(
            xw2, xa2, xg2, xv2, w1, a1, g1, v1, bool(compute_v)
        )
        outputs = extension.rank_out(
            *hidden_states, w2, a2, g2, v2, w0, a0, v0, v_current, v_first2,
            bool(sigmoid_a), bool(compute_v),
        )
    if scalar:
        return tuple(item.reshape(hidden) for item in outputs)  # type: ignore[return-value]
    return tuple(outputs)  # type: ignore[return-value]


__all__ = [
    "ada_wagv_lora",
    "ada_wagv_lora_available",
    "ada_wagv_lora_build_error",
    "ada_wagv_lora_should_use",
]
