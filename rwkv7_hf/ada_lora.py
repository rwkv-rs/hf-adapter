# coding=utf-8
"""Optional sm_86/sm_89/sm_120 fused W/A/G/V low-rank decode kernels.

The layer>0 RWKV-7 time-mix path contains four independent rank-in projections
and four rank-out projections.  For one to eight decode rows, separate
tanh/sigmoid/interpolation launches around every projection are expensive.
This module provides a grouped graph formulation through B8.  The custom
rank-in/rank-out CUDA extension remains separately gated to its exact-card
B1-B4 range; B8 keeps cuBLAS projections and only folds the surrounding
pointwise work. SM86 remains opt-in until its full-model evidence passes.

The CUDA implementation is derived from Albatross' Apache-2.0
``linear_wagv_rank_{in,out}_f16_kernel``.  It uses the HF ``nn.Linear`` weight
layouts directly, adds no packed copy, is inference-only, and falls back to
ordinary PyTorch for every unsupported device, dtype, shape, or build failure.
"""
from __future__ import annotations

import os
import threading
from typing import Any

try:
    from .extension_build import cuda_extension_build_environment
except ImportError:  # pragma: no cover - direct remote-file execution
    from extension_build import cuda_extension_build_environment

try:  # pragma: no cover - optional in lightweight environments
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
    def _sm120_wagv_bmm_down_epilogue_kernel(
        w_hidden_ptr,
        g_hidden_ptr,
        numel,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < numel
        w_hidden = tl.load(w_hidden_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        g_hidden = tl.load(g_hidden_ptr + offsets, mask=mask, other=0.0).to(
            tl.float32
        )
        # Match the existing fused-LoRA tanh formulation. Stores provide the
        # FP16 barrier consumed by the second tensor-core BMM.
        w_activated = (2.0 * tl.sigmoid(2.0 * w_hidden) - 1.0).to(tl.float16)
        g_activated = tl.sigmoid(g_hidden).to(tl.float16)
        tl.store(w_hidden_ptr + offsets, w_activated, mask=mask)
        tl.store(g_hidden_ptr + offsets, g_activated, mask=mask)

    @triton.jit
    def _sm120_wagv_bmm_up_epilogue_kernel(
        w_ptr,
        a_ptr,
        v_gate_ptr,
        w_bias_ptr,
        a_bias_ptr,
        v_bias_ptr,
        v_ptr,
        v_first_ptr,
        numel,
        hidden: tl.constexpr,
        COMPUTE_V: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < numel
        bias_offsets = offsets % hidden
        w_raw = tl.load(w_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        a_raw = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        w_bias = tl.load(
            w_bias_ptr + bias_offsets, mask=mask, other=0.0
        ).to(tl.float32)
        a_bias = tl.load(
            a_bias_ptr + bias_offsets, mask=mask, other=0.0
        ).to(tl.float32)
        # The pure rank-out BMM materializes FP16. Match eager's bias result
        # before feeding it to the following sigmoid.
        w_out = (w_raw + w_bias).to(tl.float16)
        a_biased = (a_raw + a_bias).to(tl.float16).to(tl.float32)
        a_gate = tl.sigmoid(a_biased).to(tl.float16)
        tl.store(w_ptr + offsets, w_out, mask=mask)
        tl.store(a_ptr + offsets, a_gate, mask=mask)

        if COMPUTE_V:
            gate_raw = tl.load(
                v_gate_ptr + offsets, mask=mask, other=0.0
            ).to(tl.float32)
            current = tl.load(v_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            first = tl.load(v_first_ptr + offsets, mask=mask, other=0.0).to(
                tl.float32
            )
            v_bias = tl.load(
                v_bias_ptr + bias_offsets, mask=mask, other=0.0
            ).to(tl.float32)
            # Preserve the eager FP16 expression's materialization points:
            # sigmoid, subtraction, multiplication, then addition each round
            # through FP16 before the next operation.
            gate_biased = (gate_raw + v_bias).to(tl.float16).to(tl.float32)
            gate = tl.sigmoid(gate_biased).to(tl.float16).to(tl.float32)
            delta = (first - current).to(tl.float16).to(tl.float32)
            scaled = (delta * gate).to(tl.float16).to(tl.float32)
            mixed = (current + scaled).to(tl.float16)
            # The V-group BMM output is private, so reuse it for the final V.
            tl.store(v_gate_ptr + offsets, mixed, mask=mask)


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
    torch::Tensor v, torch::Tensor v_first, bool sigmoid_a, bool compute_v,
    bool add_bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rank_in", &rwkv7_ada_wagv_rank_in_cuda,
        "RWKV-7 small-row fused W/A/G/V rank-in");
  m.def("rank_out", &rwkv7_ada_wagv_rank_out_cuda,
        "RWKV-7 small-row fused W/A/G/V rank-out and V interpolation");
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
    bool sigmoid_a,
    bool add_bias) {
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
          if (add_bias) sum += load_float(w0 + hidden_index);
          output[index] = store_float<scalar_t>(sum);
        } else if (group == 1) {
          float value = sum;
          if (add_bias) value += load_float(a0 + hidden_index);
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
    torch::Tensor v, torch::Tensor v_first, bool sigmoid_a, bool compute_v,
    bool add_bias) {
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
        sigmoid_a,
        add_bias);
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
        sigmoid_a,
        add_bias);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {w, a, g, v_out};
}
"""


_EXTENSION: Any | None = None
_EXTENSION_ERROR: str | None = None
_EXTENSIONS: dict[tuple[int, int], Any] = {}
_EXTENSION_ERRORS: dict[tuple[int, int], str] = {}
_EXTENSION_LOCK = threading.Lock()


def _is_small_row_cuda_device(device: Any = None) -> bool:
    return _small_row_capability(device) in {(8, 6), (8, 9), (12, 0)}


def _small_row_capability(device: Any = None) -> tuple[int, int] | None:
    if torch is None or not torch.cuda.is_available():
        return None
    try:
        resolved = torch.device("cuda" if device is None else device)
        if resolved.type != "cuda":
            return None
        index = torch.cuda.current_device() if resolved.index is None else int(resolved.index)
        return tuple(int(v) for v in torch.cuda.get_device_capability(index))
    except Exception:
        return None


def _load_extension(device: Any = None) -> Any | None:
    global _EXTENSION, _EXTENSION_ERROR
    capability = _small_row_capability(device)
    if capability not in {(8, 6), (8, 9), (12, 0)}:
        return None
    if capability in _EXTENSIONS:
        return _EXTENSIONS[capability]
    if capability in _EXTENSION_ERRORS:
        return None
    with _EXTENSION_LOCK:
        if capability in _EXTENSIONS:
            return _EXTENSIONS[capability]
        if capability in _EXTENSION_ERRORS:
            return None
        try:
            with cuda_extension_build_environment(
                arch_list=f"{capability[0]}.{capability[1]}"
            ) as runtime_lib:
                from torch.utils.cpp_extension import load_inline

                extra_ldflags = (
                    [f"-Wl,-rpath,{runtime_lib}"]
                    if runtime_lib is not None
                    else []
                )
                extension = load_inline(
                    name=f"rwkv7_ada_lora_v8_sm{capability[0]}{capability[1]}",
                    cpp_sources=_CPP_SOURCE,
                    cuda_sources=_CUDA_SOURCE,
                    functions=None,
                    extra_cflags=["-O3"],
                    extra_cuda_cflags=["-O3", "--use_fast_math", "--extra-device-vectorization"],
                    extra_ldflags=extra_ldflags,
                    with_cuda=True,
                    verbose=os.environ.get("RWKV7_ADA_LORA_BUILD_VERBOSE", "0").lower()
                    in {"1", "true", "yes", "on"},
                )
            _EXTENSION = extension
            _EXTENSIONS[capability] = extension
        except Exception as exc:  # pragma: no cover - host toolchain dependent
            message = f"{type(exc).__name__}: {exc}"
            _EXTENSION_ERROR = message
            _EXTENSION_ERRORS[capability] = message
            return None
    return _EXTENSIONS.get(capability)


def ada_wagv_lora_available(device: Any = None, *, build: bool = False) -> bool:
    if not _is_small_row_cuda_device(device):
        return False
    return _load_extension(device) is not None if build else True


def ada_wagv_bmm_available(device: Any = None) -> bool:
    """Return whether the grouped-BMM formulation may be selected.

    Unlike :func:`ada_wagv_lora_available`, this route uses ordinary PyTorch
    ``bmm``/``baddbmm`` operators and does not require the custom extension to
    build.  Keep the capability allowlist explicit so an environment request
    cannot be reported as selected while the implementation silently falls
    back on an unsupported device. SM86 is admitted only for explicit
    exact-card validation; policy selection remains disabled until that
    end-to-end evidence passes.
    """

    return _small_row_capability(device) in {(8, 6), (8, 9), (12, 0)}


def ada_wagv_lora_build_error(device: Any = None) -> str | None:
    capability = _small_row_capability(device)
    return _EXTENSION_ERRORS.get(capability) if capability is not None else _EXTENSION_ERROR


def ada_wagv_lora_should_use(rows: int, hidden: int, max_rank: int) -> bool:
    return 1 <= int(rows) <= 8 and int(hidden) >= 1024 and int(hidden) % 4 == 0 and 1 <= int(max_rank) <= 512


def ada_wagv_bmm_should_use(rows: int, hidden: int, max_rank: int) -> bool:
    """Return whether the measured sm89 tensor-core B8 route fits."""

    return (
        int(rows) == 8
        and int(hidden) in (1024, 2048, 2560)
        and 1 <= int(max_rank) <= 512
    )


def sm120_wagv_bmm_g_available(device: Any = None) -> bool:
    """Whether the Ampere/Ada/Blackwell all-W/A/G/V BMM probe may run.

    The route is one indivisible experiment: both padded BMMs and both Triton
    pointwise epilogues must be available.  Returning false when Triton is
    absent prevents dispatch telemetry from claiming a partial implementation.
    """

    return bool(
        _HAS_TRITON and _small_row_capability(device) in {(8, 6), (8, 9), (12, 0)}
    )


def sm120_wagv_bmm_g_should_use(rows: int, hidden: int, max_rank: int) -> bool:
    """Exact SM86/SM89/SM120 shapes admitted to the all-group microprobe."""

    return bool(
        int(rows) == 8
        and int(hidden) in {1024, 2048}
        and 1 <= int(max_rank) <= 512
    )


def _sm120_wagv_bmm_down_epilogue(w_hidden: Any, g_hidden: Any) -> None:
    """Apply W tanh and G sigmoid in one exact-SM89/SM120 Triton launch."""

    if not _HAS_TRITON:
        raise RuntimeError("SM89/SM120 W/A/G/V BMM requires Triton epilogues")
    if (
        torch is None
        or not w_hidden.is_cuda
        or w_hidden.dtype != torch.float16
        or g_hidden.dtype != w_hidden.dtype
        or tuple(g_hidden.shape) != tuple(w_hidden.shape)
        or not w_hidden.is_contiguous()
        or not g_hidden.is_contiguous()
    ):
        raise RuntimeError(
            "SM89/SM120 W/A/G/V down epilogue contract was not satisfied"
        )
    numel = int(w_hidden.numel())
    block = 256
    _sm120_wagv_bmm_down_epilogue_kernel[(triton.cdiv(numel, block),)](
        w_hidden,
        g_hidden,
        numel,
        BLOCK=block,
        num_warps=4,
    )


def _sm120_wagv_bmm_up_epilogue(
    w: Any,
    a: Any,
    v_gate: Any,
    w_bias: Any,
    a_bias: Any,
    v_bias: Any,
    v: Any,
    v_first: Any,
    *,
    compute_v: bool,
) -> None:
    """Apply A/V gates and FP16-barrier V interpolation in one launch."""

    if not _HAS_TRITON:
        raise RuntimeError("SM89/SM120 W/A/G/V BMM requires Triton epilogues")
    values = (
        (w, a, w_bias, a_bias)
        if not compute_v
        else (w, a, v_gate, w_bias, a_bias, v_bias, v, v_first)
    )
    if (
        torch is None
        or not a.is_cuda
        or a.dtype != torch.float16
        or any(
            item.dtype != a.dtype or not item.is_contiguous() for item in values
        )
        or tuple(w.shape) != tuple(a.shape)
        or (compute_v and tuple(v_gate.shape) != tuple(a.shape))
        or any(int(item.numel()) != int(a.shape[-1]) for item in (w_bias, a_bias, v_bias))
        or (compute_v and (tuple(v.shape) != tuple(a.shape) or tuple(v_first.shape) != tuple(a.shape)))
    ):
        raise RuntimeError(
            "SM89/SM120 W/A/G/V up epilogue contract was not satisfied"
        )
    numel = int(a.numel())
    block = 256
    # The non-V specialization erases every access to the dummy pointers.
    safe_v_gate = v_gate if compute_v else a
    safe_v = v if compute_v else a
    safe_v_first = v_first if compute_v else a
    _sm120_wagv_bmm_up_epilogue_kernel[(triton.cdiv(numel, block),)](
        w,
        a,
        safe_v_gate,
        w_bias,
        a_bias,
        v_bias,
        safe_v,
        safe_v_first,
        numel,
        hidden=int(a.shape[-1]),
        COMPUTE_V=bool(compute_v),
        BLOCK=block,
        num_warps=4,
    )


def _tensor_pack_identity(value: Any) -> tuple[Any, ...]:
    try:
        version = int(value._version)
    except Exception:
        version = -1
    return (
        int(value.data_ptr()),
        version,
        tuple(value.shape),
        tuple(value.stride()),
        value.dtype,
        value.device,
    )


def _ada_wagv_bmm_pack(
    w1: Any,
    a1: Any,
    v1: Any,
    w2: Any,
    a2: Any,
    v2: Any,
    w0: Any,
    a0: Any,
    v0: Any,
    *,
    include_v: bool = True,
    include_g: bool = False,
    g1: Any | None = None,
    g2: Any | None = None,
) -> tuple[Any, Any, Any]:
    """Pad W/A[/V] ranks once for two grouped tensor-core BMMs.

    The cache is attached to ``w1`` so it follows the source parameter's
    lifetime and is ignored automatically after a device move or in-place
    weight update. It is not part of the state dict or converted checkpoint.
    """

    if include_g:
        if g1 is None or g2 is None:
            raise ValueError("include_g requires G rank-in and rank-out weights")
        # The all-six norm/mix backing store is R/K/V/W/A/G.  Starting at V
        # gives a zero-copy V/W/A/G BMM view while preserving R/K/V's view.
        if include_v:
            down_weights = (v1, w1, a1, g1)
            up_weights = (v2, w2, a2, g2)
            bias_sources = (v0, w0, a0)
        else:
            down_weights = (w1, a1, g1)
            up_weights = (w2, a2, g2)
            bias_sources = (w0, a0)
    else:
        down_weights = (w1, a1, v1) if include_v else (w1, a1)
        up_weights = (w2, a2, v2) if include_v else (w2, a2)
        bias_sources = (w0, a0, v0) if include_v else (w0, a0)
    weights = (*down_weights, *up_weights, *bias_sources)
    identity = tuple(_tensor_pack_identity(value) for value in weights)
    cache_name = (
        "_rwkv7_sm120_wagv_bmm_g_pack"
        if include_g and include_v
        else "_rwkv7_sm120_wag_bmm_g_pack"
        if include_g
        else "_rwkv7_ada_wagv_bmm_pack"
        if include_v
        else "_rwkv7_ada_wa_bmm_pack"
    )
    cached = getattr(w1, cache_name, None)
    if isinstance(cached, tuple) and len(cached) == 4 and cached[0] == identity:
        return cached[1], cached[2], cached[3]

    if include_g:
        zero_bias = torch.zeros_like(w0)
        biases = (*bias_sources, zero_bias)
    else:
        biases = bias_sources
    hidden = int(w1.shape[1])
    ranks = tuple(int(value.shape[0]) for value in down_weights)
    max_rank = max(ranks)
    groups = len(ranks)
    down = w1.new_zeros((groups, max_rank, hidden))
    up = w1.new_zeros((groups, hidden, max_rank))
    for group, (rank, down_weight, up_weight) in enumerate(
        zip(ranks, down_weights, up_weights, strict=True)
    ):
        down[group, :rank].copy_(down_weight)
        up[group, :, :rank].copy_(up_weight)
    bias = torch.stack(biases).unsqueeze(1).contiguous()
    packed = (identity, down, up.transpose(1, 2), bias)
    try:
        setattr(w1, cache_name, packed)
    except Exception:
        # Parameters and ordinary tensors support attributes. Keep a correct
        # one-shot pack for unusual tensor subclasses rather than making the
        # optional route mandatory.
        pass
    return down, packed[2], bias


def _stack_wav_inputs(
    xw: Any,
    xa: Any,
    xv: Any,
    *,
    include_v: bool = True,
) -> Any:
    """Return contiguous W/A[/V] input, reusing fused storage when possible."""

    rows, hidden = int(xw.shape[0]), int(xw.shape[1])
    row_values = rows * hidden
    try:
        storage = xw.untyped_storage()
        shared_storage = xa.untyped_storage().data_ptr() == storage.data_ptr()
        if include_v:
            shared_storage = bool(
                shared_storage
                and xv.untyped_storage().data_ptr() == storage.data_ptr()
            )
    except Exception:
        shared_storage = False
    if (
        shared_storage
        and xw.is_contiguous()
        and xa.is_contiguous()
        and int(xa.storage_offset()) == int(xw.storage_offset()) + row_values
        and (
            not include_v
            or (
                xv.is_contiguous()
                and int(xv.storage_offset())
                == int(xw.storage_offset()) + 2 * row_values
            )
        )
    ):
        return xw.as_strided(
            (3 if include_v else 2, rows, hidden),
            (row_values, hidden, 1),
            storage_offset=int(xw.storage_offset()),
        )
    return torch.stack((xw, xa, xv) if include_v else (xw, xa))


def _stack_sm120_wagv_inputs(
    xw: Any,
    xa: Any,
    xg: Any,
    xv: Any,
    *,
    include_v: bool,
) -> Any:
    """Reuse the R/K/V/W/A/G norm-mix backing store without a cat/copy."""

    rows, hidden = int(xw.shape[0]), int(xw.shape[1])
    row_values = rows * hidden
    first = xv if include_v else xw
    expected = (xv, xw, xa, xg) if include_v else (xw, xa, xg)
    try:
        storage = first.untyped_storage()
        shared = all(
            value.untyped_storage().data_ptr() == storage.data_ptr()
            and value.is_contiguous()
            and int(value.storage_offset())
            == int(first.storage_offset()) + index * row_values
            for index, value in enumerate(expected)
        )
    except Exception:
        shared = False
    if shared:
        return first.as_strided(
            (len(expected), rows, hidden),
            (row_values, hidden, 1),
            storage_offset=int(first.storage_offset()),
        )
    # This allocation is correct but intentionally observable in the profiler.
    # The selected native_graph route requests the fused norm/mix layout, so a
    # production capture should always take the shared branch.
    return torch.stack(expected)


def ada_wagv_bmm(
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
    require_bmm: bool = False,
    include_g: bool = False,
    require_zero_copy: bool = False,
) -> tuple[Any, Any, Any, Any]:
    """Grouped B8 W/A/V or opt-in W/A/G/V using two padded BMMs.

    W/A/V have nearby ranks in released checkpoints, so the portable route
    shares a compact padded pack and leaves larger G on its original GEMMs.
    ``include_g`` is the exact-SM89/SM120 B8 experiment whose 1024/2048 shapes
    were positive in a raw-graph microprobe; it pads G too and requires the
    combined R/K/V/W/A/G norm-mix backing store when ``require_zero_copy`` is
    true.
    """

    if torch is None or F is None:
        raise RuntimeError("ada_wagv_bmm requires torch")
    rows = 1 if xw.dim() == 1 else int(xw.shape[0])
    hidden = int(xw.numel()) if xw.dim() == 1 else int(xw.shape[1])
    max_rank = max(int(item.shape[0]) for item in (w1, a1, g1, v1))
    values = (
        xw,
        xa,
        xg,
        xv,
        w1,
        a1,
        g1,
        v1,
        w2,
        a2,
        g2,
        v2,
        w0,
        a0,
        v0,
        v,
        v_first,
    )
    valid = bool(
        not force_fallback
        and not torch.is_grad_enabled()
        and ada_wagv_bmm_should_use(rows, hidden, max_rank)
        and (
            not include_g
            or (
                sigmoid_a
                and sm120_wagv_bmm_g_should_use(rows, hidden, max_rank)
                and sm120_wagv_bmm_g_available(xw.device)
            )
        )
        and xw.dtype == torch.float16
        and ada_wagv_bmm_available(xw.device)
        and all(
            item.is_cuda and item.dtype == xw.dtype and item.is_contiguous()
            for item in values
        )
        and all(
            tuple(item.shape) == (rows, hidden)
            for item in (xw, xa, xg, xv, v, v_first)
        )
        and all(int(item.shape[1]) == hidden for item in (w1, a1, g1, v1))
        and tuple(w2.shape) == (hidden, int(w1.shape[0]))
        and tuple(a2.shape) == (hidden, int(a1.shape[0]))
        and tuple(g2.shape) == (hidden, int(g1.shape[0]))
        and tuple(v2.shape) == (hidden, int(v1.shape[0]))
        and all(int(item.numel()) == hidden for item in (w0, a0, v0))
    )
    if not valid:
        if require_bmm:
            raise RuntimeError(
                "ada_wagv_bmm was selected for native_graph but its exact "
                "device/dtype/layout contract was not satisfied"
            )
        return ada_wagv_lora(
            xw,
            xa,
            xg,
            xv,
            w1,
            a1,
            g1,
            v1,
            w2,
            a2,
            g2,
            v2,
            w0,
            a0,
            v0,
            v,
            v_first,
            sigmoid_a=sigmoid_a,
            compute_v=compute_v,
            force_fallback=True,
        )

    down, up_transposed, bias = _ada_wagv_bmm_pack(
        w1,
        a1,
        v1,
        w2,
        a2,
        v2,
        w0,
        a0,
        v0,
        include_v=compute_v,
        include_g=include_g,
        g1=g1,
        g2=g2,
    )
    if include_g:
        mixed = _stack_sm120_wagv_inputs(
            xw, xa, xg, xv, include_v=compute_v
        )
        if require_zero_copy:
            first = xv if compute_v else xw
            if mixed.untyped_storage().data_ptr() != first.untyped_storage().data_ptr():
                raise RuntimeError(
                    "SM89/SM120 W/A/G/V BMM was selected but norm/mix did not "
                    "provide its required zero-copy R/K/V/W/A/G layout"
                )
    else:
        mixed = _stack_wav_inputs(xw, xa, xv, include_v=compute_v)
    hidden_states = torch.bmm(mixed, down.transpose(1, 2))
    if include_g:
        w_index = 1 if compute_v else 0
        a_index = 2 if compute_v else 1
        g_index = 3 if compute_v else 2
        v_index = 0 if compute_v else None
        _sm120_wagv_bmm_down_epilogue(
            hidden_states[w_index], hidden_states[g_index]
        )
    else:
        w_index, a_index, g_index = 0, 1, None
        v_index = 2 if compute_v else None
        # The grouped BMM result is private to this route. Apply W's activation
        # in place so A/V do not need a second stacked copy before rank-out.
        hidden_states[w_index].tanh_()
    if include_g:
        # Bias is fused into the single pointwise up epilogue. Keeping
        # baddbmm here costs a measurable per-layer launch/copy on SM89/SM120.
        outputs = torch.bmm(hidden_states, up_transposed)
        _sm120_wagv_bmm_up_epilogue(
            outputs[w_index],
            outputs[a_index],
            outputs[v_index] if compute_v else outputs[a_index],
            w0,
            a0,
            v0,
            v,
            v_first,
            compute_v=compute_v,
        )
    else:
        outputs = torch.baddbmm(bias, hidden_states, up_transposed)
        if sigmoid_a:
            outputs[a_index].sigmoid_()
    a = outputs[a_index]
    if include_g:
        g = outputs[g_index]
    else:
        g_hidden = F.linear(xg, g1)
        g_hidden.sigmoid_()
        g = F.linear(g_hidden, g2)
    if compute_v:
        if include_g:
            v_out = outputs[v_index]
        else:
            outputs[v_index].sigmoid_()
            v_out = v + (v_first - v) * outputs[v_index]
    else:
        v_out = v
    return outputs[w_index], a, g, v_out


def _ada_wagv_lora_extension_should_use(rows: int, hidden: int, max_rank: int) -> bool:
    """Return whether the measured small-row CUDA extension may be used.

    B8 can still use the grouped graph formulation, which removes pointwise
    launches around the ordinary cuBLAS LoRA calls.  The custom Albatross-
    derived GEMV kernels remain limited to their separately validated B1-B4
    range instead of being widened implicitly with the graph policy.
    """

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
    require_extension: bool = False,
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
        and _ada_wagv_lora_extension_should_use(rows, hidden, max_rank)
        and xw2.dtype in {torch.float16, torch.bfloat16}
        and all(item.is_cuda and item.dtype == xw2.dtype and item.is_contiguous() for item in all_tensors)
        and all(tuple(item.shape) == (rows, hidden) for item in flat)
        and all(int(item.shape[1]) == hidden for item in (w1, a1, g1, v1))
        and tuple(w2.shape) == (hidden, int(w1.shape[0]))
        and tuple(a2.shape) == (hidden, int(a1.shape[0]))
        and tuple(g2.shape) == (hidden, int(g1.shape[0]))
        and tuple(v2.shape) == (hidden, int(v1.shape[0]))
        and all(int(item.numel()) == hidden for item in (w0, a0, v0))
        and _is_small_row_cuda_device(xw2.device)
    )
    extension = _load_extension(xw2.device) if valid else None
    if require_extension and extension is None:
        detail = ada_wagv_lora_build_error(xw2.device)
        raise RuntimeError(
            "Ada W/A/G/V extension was required for native_graph decode, "
            "but its exact device/dtype/layout/build contract was not satisfied; "
            f"fallback is forbidden; build_error={detail!r}"
        )
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
            bool(sigmoid_a), bool(compute_v), True,
        )
    if scalar:
        return tuple(item.reshape(hidden) for item in outputs)  # type: ignore[return-value]
    return tuple(outputs)  # type: ignore[return-value]


def ada_wag_lora(
    xw: Any,
    xa: Any,
    xg: Any,
    w1: Any,
    a1: Any,
    g1: Any,
    w2: Any,
    a2: Any,
    g2: Any,
    w0: Any,
    a0: Any,
    *,
    force_fallback: bool = False,
) -> tuple[Any, Any, Any]:
    """Return W/A/G outputs while leaving the V gate on its normal path.

    The small-row CUDA extension is used for rows 1..4. Larger batches retain
    the grouped PyTorch formulation so callers can select one graph route for
    both latency and throughput validation without extending the small-row
    kernel beyond its measured range.
    """

    if torch is None or F is None:
        raise RuntimeError("ada_wag_lora requires torch")
    scalar = xw.dim() == 1
    xw2, xa2, xg2 = (
        item.reshape(1, -1) if scalar else item for item in (xw, xa, xg)
    )
    rows, hidden = int(xw2.shape[0]), int(xw2.shape[1])
    max_rank = max(int(item.shape[0]) for item in (w1, a1, g1))
    tensors = [xw2, xa2, xg2, w1, a1, g1, w2, a2, g2, w0, a0]
    valid = bool(
        not force_fallback
        and not torch.is_grad_enabled()
        and _ada_wagv_lora_extension_should_use(rows, hidden, max_rank)
        and xw2.dtype in {torch.float16, torch.bfloat16}
        and all(
            item.is_cuda and item.dtype == xw2.dtype and item.is_contiguous()
            for item in tensors
        )
        and tuple(xa2.shape) == tuple(xw2.shape)
        and tuple(xg2.shape) == tuple(xw2.shape)
        and all(int(item.shape[1]) == hidden for item in (w1, a1, g1))
        and tuple(w2.shape) == (hidden, int(w1.shape[0]))
        and tuple(a2.shape) == (hidden, int(a1.shape[0]))
        and tuple(g2.shape) == (hidden, int(g1.shape[0]))
        and int(w0.numel()) == hidden
        and int(a0.numel()) == hidden
        and _is_small_row_cuda_device(xw2.device)
    )
    extension = _load_extension(xw2.device) if valid else None
    if extension is None:
        outputs = (
            F.linear(torch.tanh(F.linear(xw2, w1)), w2, w0),
            F.linear(F.linear(xa2, a1), a2, a0),
            F.linear(torch.sigmoid(F.linear(xg2, g1)), g2),
        )
    else:
        hidden_states = extension.rank_in(
            xw2, xa2, xg2, xg2, w1, a1, g1, g1, False
        )
        w, a, g, _unused_v = extension.rank_out(
            *hidden_states,
            w2,
            a2,
            g2,
            g2,
            w0,
            a0,
            a0,
            xg2,
            xg2,
            False,
            False,
            False,
        )
        # Match the official two-stage WAG boundary exactly: rank-out rounds to
        # the model dtype first, then the W/A biases are added pointwise.
        outputs = (w + w0, a + a0, g)
    if scalar:
        return tuple(item.reshape(hidden) for item in outputs)  # type: ignore[return-value]
    return tuple(outputs)  # type: ignore[return-value]


__all__ = [
    "ada_wag_lora",
    "ada_wagv_bmm",
    "ada_wagv_bmm_available",
    "ada_wagv_bmm_should_use",
    "sm120_wagv_bmm_g_available",
    "sm120_wagv_bmm_g_should_use",
    "ada_wagv_lora",
    "ada_wagv_lora_available",
    "ada_wagv_lora_build_error",
    "ada_wagv_lora_should_use",
]
