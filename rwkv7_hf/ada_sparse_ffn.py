# coding=utf-8
"""Optional sm_89/sm_120 fp16 sparse FFN contraction for small-row decode.

RWKV-7 applies ``ReLU(key(x)) ** 2`` before the FFN value projection.  At
decode batch sizes the activation is naturally sparse, so reading only the
positive rows of a packed ``[ffn, hidden]`` value matrix can be faster than a
dense GEMM on measured GPU generations. The CUDA kernel is derived from
Albatross' Apache-2.0 ``cmix_sparse_spmv_relu_rows_kernel`` and adds the
residual while initializing the output, avoiding a separate residual-add
launch.

The extension is deliberately narrow: fp16, exact sm_89 or sm_120, at most 19
rows, and the normal RWKV ``ffn == 4 * hidden`` shape. Unsupported shapes,
training, and build failures retain the ordinary PyTorch implementation. Value
weights are transposed once and cached; callers can prewarm the cache before
CUDA graph capture with :func:`ada_sparse_ffn_pack_weight`.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import threading
from typing import Any
import weakref

try:  # pragma: no cover - optional in lightweight environments
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]

try:  # pragma: no cover - direct remote-file execution fallback
    from .kernel_policy import current_kernel_policy, env_flag
except Exception:  # pragma: no cover
    try:
        from kernel_policy import current_kernel_policy, env_flag
    except Exception:
        current_kernel_policy = None  # type: ignore[assignment]
        env_flag = None  # type: ignore[assignment]


def _kernel_policy():
    if current_kernel_policy is None:
        return None
    try:
        return current_kernel_policy(torch_module=torch)
    except Exception:
        return None


def _policy_flag(env_name: str, policy_name: str) -> bool:
    policy = _kernel_policy()
    default = bool(getattr(policy, policy_name, False)) if policy is not None else False
    if env_flag is not None:
        return bool(env_flag(env_name, default))
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


_CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor rwkv7_ada_sparse_ffn_cuda(
    torch::Tensor preact, torch::Tensor packed_value, torch::Tensor residual);
torch::Tensor rwkv7_ada_sparse_ffn_out_cuda(
    torch::Tensor preact, torch::Tensor packed_value, torch::Tensor residual,
    torch::Tensor output);
torch::Tensor rwkv7_ada_sparse_ffn_fp32_cuda(
    torch::Tensor preact, torch::Tensor packed_value, torch::Tensor residual,
    torch::Tensor scratch);
torch::Tensor rwkv7_ada_sparse_ffn_fp32_out_cuda(
    torch::Tensor preact, torch::Tensor packed_value, torch::Tensor residual,
    torch::Tensor scratch, torch::Tensor output);
torch::Tensor rwkv7_ada_sparse_ffn_official_cuda(
    torch::Tensor preact, torch::Tensor packed_value, torch::Tensor residual);
torch::Tensor rwkv7_ada_sparse_ffn_official_out_cuda(
    torch::Tensor preact, torch::Tensor packed_value, torch::Tensor residual,
    torch::Tensor output);
torch::Tensor rwkv7_ada_sparse_ffn_deterministic4_cuda(
    torch::Tensor preact, torch::Tensor packed_value, torch::Tensor residual,
    torch::Tensor scratch);
torch::Tensor rwkv7_ada_sparse_ffn_deterministic4_out_cuda(
    torch::Tensor preact, torch::Tensor packed_value, torch::Tensor residual,
    torch::Tensor scratch, torch::Tensor output);
torch::Tensor rwkv7_ada_linear_cuda(torch::Tensor x, torch::Tensor weight);
torch::Tensor rwkv7_blackwell_ffn_up_cuda(torch::Tensor x, torch::Tensor weight);
torch::Tensor rwkv7_blackwell_sparse_ffn_out_cuda(
    torch::Tensor preact, torch::Tensor packed_value, torch::Tensor residual,
    torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sparse_down_add", &rwkv7_ada_sparse_ffn_cuda,
        "RWKV-7 sm_89 sparse ReLU2 FFN down projection + residual");
  m.def("sparse_down_add_out", &rwkv7_ada_sparse_ffn_out_cuda,
        "RWKV-7 sm_89 sparse ReLU2 FFN down projection + residual (out)");
  m.def("sparse_down_add_fp32", &rwkv7_ada_sparse_ffn_fp32_cuda,
        "RWKV-7 sparse ReLU2 FFN down projection with FP32 tile accumulation");
  m.def("sparse_down_add_fp32_out", &rwkv7_ada_sparse_ffn_fp32_out_cuda,
        "RWKV-7 sparse ReLU2 FFN down projection with FP32 tile accumulation (out)");
  m.def("sparse_down_add_official", &rwkv7_ada_sparse_ffn_official_cuda,
        "RWKV-7 sparse ReLU2 FFN with official accumulate-round-add boundary");
  m.def("sparse_down_add_official_out", &rwkv7_ada_sparse_ffn_official_out_cuda,
        "RWKV-7 sparse ReLU2 FFN with official accumulate-round-add boundary (out)");
  m.def("sparse_down_add_deterministic4", &rwkv7_ada_sparse_ffn_deterministic4_cuda,
        "RWKV-7 sparse ReLU2 FFN with deterministic four-way reduction");
  m.def("sparse_down_add_deterministic4_out", &rwkv7_ada_sparse_ffn_deterministic4_out_cuda,
        "RWKV-7 sparse ReLU2 FFN with deterministic four-way reduction (out)");
  m.def("ffn_up", &rwkv7_ada_linear_cuda,
        "RWKV-7 sm_89 small-row FFN expansion projection");
  m.def("linear", &rwkv7_ada_linear_cuda,
        "RWKV-7 sm_89 small-row fp16 linear");
  m.def("blackwell_ffn_up", &rwkv7_blackwell_ffn_up_cuda,
        "RWKV-7 SM120 row-one FFN expansion projection");
  m.def("blackwell_sparse_down_add_out", &rwkv7_blackwell_sparse_ffn_out_cuda,
        "RWKV-7 SM120 row-one sparse FFN down projection + residual (out)");
}
"""


# The sparse compaction and half2 accumulation below are derived from
# Albatross/faster3a_2605/cuda/rwkv7_fast_ops_fp16.cu (Apache-2.0), with the
# output-zero kernel replaced by a residual-copy kernel for the HF block API.
_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

namespace {

constexpr int THREADS = 128;
constexpr int FFN_TILE = 128;
using blackwell_dtype = at::Half;

__device__ inline float load_h1(const half* ptr) {
  return __half2float(*ptr);
}

__device__ inline float warp_sum(float value) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ inline float blackwell_load_h1(const blackwell_dtype* ptr) {
  return __half2float(*reinterpret_cast<const __half*>(ptr));
}

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 1) void blackwell_ffn_up_row1_exact4_kernel(
    int hidden,
    int ffn,
    const blackwell_dtype* __restrict__ x,
    const blackwell_dtype* __restrict__ weight,
    blackwell_dtype* __restrict__ output) {
  const int output_start = blockIdx.x * OutTile;
  float accumulators[OutTile];
  #pragma unroll
  for (int out = 0; out < OutTile; ++out) {
    accumulators[out] = 0.0f;
  }
  for (int k = threadIdx.x << 2; k < hidden; k += Threads << 2) {
    const float2 x0 = __half22float2(*reinterpret_cast<const __half2*>(x + k));
    const float2 x1 = __half22float2(*reinterpret_cast<const __half2*>(x + k + 2));
    #pragma unroll
    for (int out = 0; out < OutTile; ++out) {
      const blackwell_dtype* weight_row =
          weight + static_cast<int64_t>(output_start + out) * hidden + k;
      const float2 w0 = __half22float2(*reinterpret_cast<const __half2*>(weight_row));
      const float2 w1 = __half22float2(*reinterpret_cast<const __half2*>(weight_row + 2));
      accumulators[out] = fmaf(x0.x, w0.x, accumulators[out]);
      accumulators[out] = fmaf(x0.y, w0.y, accumulators[out]);
      accumulators[out] = fmaf(x1.x, w1.x, accumulators[out]);
      accumulators[out] = fmaf(x1.y, w1.y, accumulators[out]);
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  #pragma unroll
  for (int out = 0; out < OutTile; ++out) {
    const float value = warp_sum(accumulators[out]);
    if (lane == 0) partial[warp][out] = value;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    #pragma unroll
    for (int out = 0; out < OutTile; ++out) {
      float sum = 0.0f;
      #pragma unroll
      for (int w = 0; w < Threads / 32; ++w) sum += partial[w][out];
      output[output_start + out] = __float2half_rn(sum);
    }
  }
}

__global__ __launch_bounds__(THREADS, 4) void blackwell_sparse_relu2_down_row1_kernel(
    int hidden,
    const blackwell_dtype* __restrict__ preact,
    const blackwell_dtype* __restrict__ packed_value,
    blackwell_dtype* __restrict__ output) {
  __shared__ __align__(256) __half values[FFN_TILE];
  __shared__ __align__(256) int nonzero_ids[FFN_TILE];
  __shared__ int nonzero_count;
  __shared__ int warp_counts[FFN_TILE / 32];
  __shared__ int warp_prefix[FFN_TILE / 32];

  const int f_block = blockIdx.x;
  const int hidden_block = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int start_f = f_block * FFN_TILE;

  if (tid < FFN_TILE) {
    const float value = fmaxf(blackwell_load_h1(preact + start_f + tid), 0.0f);
    values[tid] = __float2half_rn(value * value);
  }
  __syncthreads();

  bool nonzero = false;
  int local_position = 0;
  if (tid < FFN_TILE) {
    nonzero = bool(__half_as_ushort(values[tid]) << 1);
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    local_position = __popc(mask & ((1u << lane) - 1u));
    if (lane == 0) warp_counts[warp] = __popc(mask);
  }
  __syncthreads();

  if (tid == 0) {
    int prefix = 0;
    #pragma unroll
    for (int w = 0; w < FFN_TILE / 32; ++w) {
      warp_prefix[w] = prefix;
      prefix += warp_counts[w];
    }
    nonzero_count = prefix;
  }
  __syncthreads();

  if (tid < FFN_TILE && nonzero) {
    nonzero_ids[warp_prefix[warp] + local_position] = tid;
  }
  __syncthreads();

  __half2 accumulator;
  *reinterpret_cast<int*>(&accumulator) = 0;
  for (int i = 0; i < nonzero_count; ++i) {
    const int actual_f = start_f + nonzero_ids[i];
    const __half2 matrix = *reinterpret_cast<const __half2*>(
        packed_value + static_cast<int64_t>(actual_f) * hidden
        + hidden_block * (2 * THREADS) + tid * 2);
    accumulator = __hfma2(__half2half2(values[nonzero_ids[i]]), matrix, accumulator);
  }
  atomicAdd(
      reinterpret_cast<__half2*>(
          output + hidden_block * (2 * THREADS) + tid * 2),
      accumulator);
}

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 1) void ffn_up_row1_exact4_kernel(
    int hidden,
    int ffn,
    const half* __restrict__ x,
    const half* __restrict__ weight,
    half* __restrict__ output) {
  const int output_start = blockIdx.x * OutTile;
  float accumulators[OutTile];
  #pragma unroll
  for (int out = 0; out < OutTile; ++out) {
    accumulators[out] = 0.0f;
  }
  for (int k = threadIdx.x << 2; k < hidden; k += Threads << 2) {
    const float2 x0 = __half22float2(*reinterpret_cast<const half2*>(x + k));
    const float2 x1 = __half22float2(*reinterpret_cast<const half2*>(x + k + 2));
    #pragma unroll
    for (int out = 0; out < OutTile; ++out) {
      const half* weight_row = weight + static_cast<int64_t>(output_start + out) * hidden + k;
      const float2 w0 = __half22float2(*reinterpret_cast<const half2*>(weight_row));
      const float2 w1 = __half22float2(*reinterpret_cast<const half2*>(weight_row + 2));
      accumulators[out] = fmaf(x0.x, w0.x, accumulators[out]);
      accumulators[out] = fmaf(x0.y, w0.y, accumulators[out]);
      accumulators[out] = fmaf(x1.x, w1.x, accumulators[out]);
      accumulators[out] = fmaf(x1.y, w1.y, accumulators[out]);
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  #pragma unroll
  for (int out = 0; out < OutTile; ++out) {
    const float value = warp_sum(accumulators[out]);
    if (lane == 0) partial[warp][out] = value;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    #pragma unroll
    for (int out = 0; out < OutTile; ++out) {
      float sum = 0.0f;
      #pragma unroll
      for (int w = 0; w < Threads / 32; ++w) sum += partial[w][out];
      output[output_start + out] = __float2half_rn(sum);
    }
  }
}

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 1) void ffn_up_row2_exact4_kernel(
    int hidden,
    int ffn,
    const half* __restrict__ x,
    const half* __restrict__ weight,
    half* __restrict__ output) {
  const int output_start = blockIdx.x * OutTile;
  float accumulators0[OutTile];
  float accumulators1[OutTile];
  #pragma unroll
  for (int out = 0; out < OutTile; ++out) {
    accumulators0[out] = 0.0f;
    accumulators1[out] = 0.0f;
  }
  for (int k = threadIdx.x << 2; k < hidden; k += Threads << 2) {
    const float2 x00 = __half22float2(*reinterpret_cast<const half2*>(x + k));
    const float2 x01 = __half22float2(*reinterpret_cast<const half2*>(x + k + 2));
    const float2 x10 = __half22float2(*reinterpret_cast<const half2*>(x + hidden + k));
    const float2 x11 = __half22float2(*reinterpret_cast<const half2*>(x + hidden + k + 2));
    #pragma unroll
    for (int out = 0; out < OutTile; ++out) {
      const half* weight_row = weight + static_cast<int64_t>(output_start + out) * hidden + k;
      const float2 w0 = __half22float2(*reinterpret_cast<const half2*>(weight_row));
      const float2 w1 = __half22float2(*reinterpret_cast<const half2*>(weight_row + 2));
      accumulators0[out] = fmaf(x00.x, w0.x, accumulators0[out]);
      accumulators0[out] = fmaf(x00.y, w0.y, accumulators0[out]);
      accumulators0[out] = fmaf(x01.x, w1.x, accumulators0[out]);
      accumulators0[out] = fmaf(x01.y, w1.y, accumulators0[out]);
      accumulators1[out] = fmaf(x10.x, w0.x, accumulators1[out]);
      accumulators1[out] = fmaf(x10.y, w0.y, accumulators1[out]);
      accumulators1[out] = fmaf(x11.x, w1.x, accumulators1[out]);
      accumulators1[out] = fmaf(x11.y, w1.y, accumulators1[out]);
    }
  }
  __shared__ float partial[Threads / 32][2][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  #pragma unroll
  for (int out = 0; out < OutTile; ++out) {
    const float value0 = warp_sum(accumulators0[out]);
    const float value1 = warp_sum(accumulators1[out]);
    if (lane == 0) {
      partial[warp][0][out] = value0;
      partial[warp][1][out] = value1;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    #pragma unroll
    for (int out = 0; out < OutTile; ++out) {
      float sum0 = 0.0f;
      float sum1 = 0.0f;
      #pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum0 += partial[w][0][out];
        sum1 += partial[w][1][out];
      }
      output[output_start + out] = __float2half_rn(sum0);
      output[ffn + output_start + out] = __float2half_rn(sum1);
    }
  }
}

template <int Threads, int RowTile, int OutTile>
__global__ __launch_bounds__(Threads, 1) void ffn_up_rows_kernel(
    int rows,
    int hidden,
    int ffn,
    const half* __restrict__ x,
    const half* __restrict__ weight,
    half* __restrict__ output) {
  const int output_start = blockIdx.x * OutTile;
  const int row_start = blockIdx.y * RowTile;
  float accumulators[RowTile][OutTile];
  #pragma unroll
  for (int row = 0; row < RowTile; ++row) {
    #pragma unroll
    for (int out = 0; out < OutTile; ++out) {
      accumulators[row][out] = 0.0f;
    }
  }

  const int hidden_pairs = hidden >> 1;
  for (int pair = threadIdx.x; pair < hidden_pairs; pair += Threads) {
    const int k = pair << 1;
    float2 weights[OutTile];
    #pragma unroll
    for (int out = 0; out < OutTile; ++out) {
      const int output_index = output_start + out;
      weights[out] = __half22float2(*reinterpret_cast<const half2*>(
          weight + static_cast<int64_t>(output_index) * hidden + k));
    }
    #pragma unroll
    for (int row = 0; row < RowTile; ++row) {
      const int row_index = row_start + row;
      if (row_index < rows) {
        const float2 activation = __half22float2(*reinterpret_cast<const half2*>(
            x + static_cast<int64_t>(row_index) * hidden + k));
        #pragma unroll
        for (int out = 0; out < OutTile; ++out) {
          accumulators[row][out] = fmaf(
              activation.x, weights[out].x, accumulators[row][out]);
          accumulators[row][out] = fmaf(
              activation.y, weights[out].y, accumulators[row][out]);
        }
      }
    }
  }

  __shared__ float partial[Threads / 32][RowTile][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  #pragma unroll
  for (int row = 0; row < RowTile; ++row) {
    #pragma unroll
    for (int out = 0; out < OutTile; ++out) {
      const float value = warp_sum(accumulators[row][out]);
      if (lane == 0) {
        partial[warp][row][out] = value;
      }
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    #pragma unroll
    for (int row = 0; row < RowTile; ++row) {
      const int row_index = row_start + row;
      if (row_index < rows) {
        #pragma unroll
        for (int out = 0; out < OutTile; ++out) {
          float sum = 0.0f;
          #pragma unroll
          for (int w = 0; w < Threads / 32; ++w) {
            sum += partial[w][row][out];
          }
          output[static_cast<int64_t>(row_index) * ffn + output_start + out] =
              __float2half_rn(sum);
        }
      }
    }
  }
}

__global__ void copy_residual_vec4_kernel(
    const half* __restrict__ residual,
    half* __restrict__ output,
    int64_t n_vec4) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n_vec4) {
    reinterpret_cast<int4*>(output)[i] = reinterpret_cast<const int4*>(residual)[i];
  }
}

__global__ void zero_output_vec4_kernel(
    half* __restrict__ output,
    int64_t n_vec4) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < n_vec4) {
    reinterpret_cast<int4*>(output)[index] = make_int4(0, 0, 0, 0);
  }
}

__global__ void add_residual_half2_kernel(
    const half* __restrict__ residual,
    half* __restrict__ output,
    int64_t pairs) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < pairs) {
    half2* out2 = reinterpret_cast<half2*>(output);
    const half2* residual2 = reinterpret_cast<const half2*>(residual);
    out2[index] = __hadd2(out2[index], residual2[index]);
  }
}

__global__ __launch_bounds__(THREADS, 4) void sparse_relu2_down_rows_kernel(
    int hidden,
    int ffn,
    const half* __restrict__ preact,
    const half* __restrict__ packed_value,
    half* __restrict__ output) {
  __shared__ __align__(256) half values[FFN_TILE];
  __shared__ __align__(256) int nonzero_ids[FFN_TILE];
  __shared__ int nonzero_count;
  __shared__ int warp_counts[FFN_TILE / 32];
  __shared__ int warp_prefix[FFN_TILE / 32];

  const int f_block = blockIdx.x;
  const int hidden_block = blockIdx.y;
  const int row = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int start_f = f_block * FFN_TILE;
  const half* pre_row = preact + static_cast<int64_t>(row) * ffn;

  const float positive = fmaxf(load_h1(pre_row + start_f + tid), 0.0f);
  values[tid] = __float2half_rn(positive * positive);
  __syncthreads();

  const bool nonzero = (__half_as_ushort(values[tid]) << 1) != 0;
  const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
  const int local_position = __popc(mask & ((1u << lane) - 1u));
  if (lane == 0) {
    warp_counts[warp] = __popc(mask);
  }
  __syncthreads();

  if (tid == 0) {
    int prefix = 0;
    #pragma unroll
    for (int w = 0; w < FFN_TILE / 32; ++w) {
      warp_prefix[w] = prefix;
      prefix += warp_counts[w];
    }
    nonzero_count = prefix;
  }
  __syncthreads();

  if (nonzero) {
    nonzero_ids[warp_prefix[warp] + local_position] = tid;
  }
  __syncthreads();

  half2 accumulator = __float2half2_rn(0.0f);
  #pragma unroll 1
  for (int i = 0; i < nonzero_count; ++i) {
    const int local_f = nonzero_ids[i];
    const int actual_f = start_f + local_f;
    const half2 matrix = *reinterpret_cast<const half2*>(
        packed_value + static_cast<int64_t>(actual_f) * hidden
        + hidden_block * (2 * THREADS) + tid * 2);
    accumulator = __hfma2(__half2half2(values[local_f]), matrix, accumulator);
  }
  atomicAdd(
      reinterpret_cast<half2*>(output + static_cast<int64_t>(row) * hidden
                               + hidden_block * (2 * THREADS) + tid * 2),
      accumulator);
}

__global__ __launch_bounds__(256, 2) void sparse_relu2_down_rows_t512_kernel(
    int hidden,
    int ffn,
    const half* __restrict__ preact,
    const half* __restrict__ packed_value,
    half* __restrict__ output) {
  constexpr int TILE = 512;
  constexpr int TILE_THREADS = 256;
  __shared__ __align__(256) half values[TILE];
  __shared__ __align__(256) int nonzero_ids[TILE];
  __shared__ int nonzero_count;
  __shared__ int warp_counts[TILE / 32];
  __shared__ int warp_prefix[TILE / 32];

  const int f_block = blockIdx.x;
  const int hidden_block = blockIdx.y;
  const int row = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int start_f = f_block * TILE;
  const half* pre_row = preact + static_cast<int64_t>(row) * ffn;

  #pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int local_f = tid + u * TILE_THREADS;
    const float positive = fmaxf(load_h1(pre_row + start_f + local_f), 0.0f);
    values[local_f] = __float2half_rn(positive * positive);
  }
  __syncthreads();

  #pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int local_f = tid + u * TILE_THREADS;
    const bool nonzero = (__half_as_ushort(values[local_f]) << 1) != 0;
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    if (lane == 0) {
      warp_counts[warp + u * (TILE_THREADS / 32)] = __popc(mask);
    }
  }
  __syncthreads();

  if (tid == 0) {
    int prefix = 0;
    #pragma unroll
    for (int w = 0; w < TILE / 32; ++w) {
      warp_prefix[w] = prefix;
      prefix += warp_counts[w];
    }
    nonzero_count = prefix;
  }
  __syncthreads();

  #pragma unroll
  for (int u = 0; u < 2; ++u) {
    const int local_f = tid + u * TILE_THREADS;
    const bool nonzero = (__half_as_ushort(values[local_f]) << 1) != 0;
    const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
    const int local_position = __popc(mask & ((1u << lane) - 1u));
    const int group = warp + u * (TILE_THREADS / 32);
    if (nonzero) {
      nonzero_ids[warp_prefix[group] + local_position] = local_f;
    }
  }
  __syncthreads();

  half2 accumulator = __float2half2_rn(0.0f);
  for (int i = 0; i < nonzero_count; ++i) {
    const int local_f = nonzero_ids[i];
    const int actual_f = start_f + local_f;
    const half2 matrix = *reinterpret_cast<const half2*>(
        packed_value + static_cast<int64_t>(actual_f) * hidden
        + hidden_block * (2 * TILE_THREADS) + tid * 2);
    accumulator = __hfma2(__half2half2(values[local_f]), matrix, accumulator);
  }
  atomicAdd(
      reinterpret_cast<half2*>(output + static_cast<int64_t>(row) * hidden
                               + hidden_block * (2 * TILE_THREADS) + tid * 2),
      accumulator);
}

__global__ __launch_bounds__(256, 2) void sparse_relu2_down_deterministic4_kernel(
    int hidden,
    int ffn,
    int rows,
    const half* __restrict__ preact,
    const half* __restrict__ packed_value,
    half* __restrict__ scratch) {
  constexpr int TILE = 512;
  constexpr int TILE_THREADS = 256;
  constexpr int SPLITS = 4;
  __shared__ __align__(256) half values[TILE];
  __shared__ __align__(256) int nonzero_ids[TILE];
  __shared__ int nonzero_count;
  __shared__ int warp_counts[TILE / 32];
  __shared__ int warp_prefix[TILE / 32];

  const int split = blockIdx.x;
  const int hidden_block = blockIdx.y;
  const int row = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int tiles_per_split = ffn / (SPLITS * TILE);
  const half* pre_row = preact + static_cast<int64_t>(row) * ffn;
  half2 accumulator = __float2half2_rn(0.0f);

  #pragma unroll 1
  for (int tile = 0; tile < tiles_per_split; ++tile) {
    const int start_f = (split * tiles_per_split + tile) * TILE;
    #pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int local_f = tid + u * TILE_THREADS;
      const float positive = fmaxf(load_h1(pre_row + start_f + local_f), 0.0f);
      values[local_f] = __float2half_rn(positive * positive);
    }
    __syncthreads();

    #pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int local_f = tid + u * TILE_THREADS;
      const bool nonzero = (__half_as_ushort(values[local_f]) << 1) != 0;
      const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
      if (lane == 0) {
        warp_counts[warp + u * (TILE_THREADS / 32)] = __popc(mask);
      }
    }
    __syncthreads();

    if (tid == 0) {
      int prefix = 0;
      #pragma unroll
      for (int w = 0; w < TILE / 32; ++w) {
        warp_prefix[w] = prefix;
        prefix += warp_counts[w];
      }
      nonzero_count = prefix;
    }
    __syncthreads();

    #pragma unroll
    for (int u = 0; u < 2; ++u) {
      const int local_f = tid + u * TILE_THREADS;
      const bool nonzero = (__half_as_ushort(values[local_f]) << 1) != 0;
      const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
      const int local_position = __popc(mask & ((1u << lane) - 1u));
      const int group = warp + u * (TILE_THREADS / 32);
      if (nonzero) {
        nonzero_ids[warp_prefix[group] + local_position] = local_f;
      }
    }
    __syncthreads();

    half2 tile_accumulator = __float2half2_rn(0.0f);
    for (int i = 0; i < nonzero_count; ++i) {
      const int actual_f = start_f + nonzero_ids[i];
      const half2 matrix = *reinterpret_cast<const half2*>(
          packed_value + static_cast<int64_t>(actual_f) * hidden
          + hidden_block * (2 * TILE_THREADS) + tid * 2);
      tile_accumulator = __hfma2(
          __half2half2(values[nonzero_ids[i]]), matrix, tile_accumulator);
    }
    accumulator = __hadd2(accumulator, tile_accumulator);
    __syncthreads();
  }

  half2* destination = reinterpret_cast<half2*>(
      scratch + (static_cast<int64_t>(split) * rows + row) * hidden
      + hidden_block * (2 * TILE_THREADS));
  destination[tid] = accumulator;
}

__global__ __launch_bounds__(256, 2) void finalize_sparse_deterministic4_kernel(
    int hidden,
    int rows,
    const half* __restrict__ scratch,
    const half* __restrict__ residual,
    half* __restrict__ output) {
  constexpr int TILE_THREADS = 256;
  constexpr int SPLITS = 4;
  const int hidden_block = blockIdx.x;
  const int row = blockIdx.y;
  const int tid = threadIdx.x;
  const int pair_offset = hidden_block * TILE_THREADS + tid;
  half2 accumulator = __float2half2_rn(0.0f);
  #pragma unroll
  for (int split = 0; split < SPLITS; ++split) {
    const half2 value = reinterpret_cast<const half2*>(
        scratch + (static_cast<int64_t>(split) * rows + row) * hidden)[pair_offset];
    accumulator = __hadd2(accumulator, value);
  }
  const half2 residual_value = reinterpret_cast<const half2*>(
      residual + static_cast<int64_t>(row) * hidden)[pair_offset];
  reinterpret_cast<half2*>(output + static_cast<int64_t>(row) * hidden)[pair_offset] =
      __hadd2(residual_value, accumulator);
}

__global__ __launch_bounds__(THREADS, 4) void sparse_relu2_down_fp32_kernel(
    int hidden,
    int ffn,
    const half* __restrict__ preact,
    const half* __restrict__ packed_value,
    float* __restrict__ scratch) {
  __shared__ __align__(256) half values[FFN_TILE];
  __shared__ __align__(256) int nonzero_ids[FFN_TILE];
  __shared__ int nonzero_count;
  __shared__ int warp_counts[FFN_TILE / 32];
  __shared__ int warp_prefix[FFN_TILE / 32];

  const int f_block = blockIdx.x;
  const int hidden_block = blockIdx.y;
  const int row = blockIdx.z;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int start_f = f_block * FFN_TILE;
  const half* pre_row = preact + static_cast<int64_t>(row) * ffn;

  const float positive = fmaxf(load_h1(pre_row + start_f + tid), 0.0f);
  values[tid] = __float2half_rn(positive * positive);
  __syncthreads();

  const bool nonzero = (__half_as_ushort(values[tid]) << 1) != 0;
  const unsigned mask = __ballot_sync(0xffffffffu, nonzero);
  const int local_position = __popc(mask & ((1u << lane) - 1u));
  if (lane == 0) {
    warp_counts[warp] = __popc(mask);
  }
  __syncthreads();

  if (tid == 0) {
    int prefix = 0;
    #pragma unroll
    for (int w = 0; w < FFN_TILE / 32; ++w) {
      warp_prefix[w] = prefix;
      prefix += warp_counts[w];
    }
    nonzero_count = prefix;
  }
  __syncthreads();

  if (nonzero) {
    nonzero_ids[warp_prefix[warp] + local_position] = tid;
  }
  __syncthreads();

  half2 accumulator = __float2half2_rn(0.0f);
  #pragma unroll 1
  for (int i = 0; i < nonzero_count; ++i) {
    const int local_f = nonzero_ids[i];
    const int actual_f = start_f + local_f;
    const half2 matrix = *reinterpret_cast<const half2*>(
        packed_value + static_cast<int64_t>(actual_f) * hidden
        + hidden_block * (2 * THREADS) + tid * 2);
    accumulator = __hfma2(__half2half2(values[local_f]), matrix, accumulator);
  }
  const float2 value = __half22float2(accumulator);
  float* destination = scratch + static_cast<int64_t>(row) * hidden
      + hidden_block * (2 * THREADS) + tid * 2;
  atomicAdd(destination, value.x);
  atomicAdd(destination + 1, value.y);
}

__global__ void finalize_sparse_fp32_add_residual_kernel(
    int64_t elements,
    float* __restrict__ scratch,
    const half* __restrict__ residual,
    half* __restrict__ output) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const float value = scratch[index] + __half2float(residual[index]);
  output[index] = __float2half_rn(value);
  scratch[index] = 0.0f;
}

}  // namespace

torch::Tensor rwkv7_blackwell_ffn_up_cuda(
    torch::Tensor x,
    torch::Tensor weight) {
  TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(x.scalar_type() == at::kHalf && weight.scalar_type() == at::kHalf,
              "fp16 tensors required");
  TORCH_CHECK(x.dim() == 2 && weight.dim() == 2, "x and weight must be rank-2");
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(), "contiguous tensors required");
  const int64_t rows = x.size(0);
  const int64_t hidden = x.size(1);
  const int64_t ffn = weight.size(0);
  TORCH_CHECK(rows == 1, "SM120 FFN expansion requires exactly one row");
  TORCH_CHECK(weight.size(1) == hidden && (hidden % 4) == 0 && (ffn % 2) == 0,
              "unsupported SM120 FFN expansion shape");

  c10::cuda::CUDAGuard device_guard(x.device());
  auto output = torch::empty({rows, ffn}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  blackwell_ffn_up_row1_exact4_kernel<128, 2><<<
      dim3(static_cast<unsigned>(ffn / 2), 1, 1), 128, 0, stream>>>(
      static_cast<int>(hidden), static_cast<int>(ffn),
      x.data_ptr<at::Half>(),
      weight.data_ptr<at::Half>(),
      output.data_ptr<at::Half>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor rwkv7_blackwell_sparse_ffn_out_cuda(
    torch::Tensor preact,
    torch::Tensor packed_value,
    torch::Tensor residual,
    torch::Tensor output) {
  TORCH_CHECK(preact.is_cuda() && packed_value.is_cuda() && residual.is_cuda() && output.is_cuda(),
              "CUDA tensors required");
  TORCH_CHECK(preact.scalar_type() == at::kHalf && packed_value.scalar_type() == at::kHalf
              && residual.scalar_type() == at::kHalf && output.scalar_type() == at::kHalf,
              "fp16 tensors required");
  TORCH_CHECK(preact.dim() == 2 && packed_value.dim() == 2 && residual.dim() == 2 && output.dim() == 2,
              "preact, packed_value, residual, and output must be rank-2");
  TORCH_CHECK(preact.is_contiguous() && packed_value.is_contiguous()
              && residual.is_contiguous() && output.is_contiguous(),
              "contiguous tensors required");
  const int64_t rows = preact.size(0);
  const int64_t ffn = preact.size(1);
  const int64_t hidden = residual.size(1);
  TORCH_CHECK(rows == 1 && residual.size(0) == 1 && output.sizes() == residual.sizes(),
              "SM120 sparse FFN requires one matching row");
  TORCH_CHECK(packed_value.size(0) == ffn && packed_value.size(1) == hidden,
              "packed value weight must have shape [ffn, hidden]");
  TORCH_CHECK(ffn == 4 * hidden && (ffn % FFN_TILE) == 0 && (hidden % 256) == 0,
              "expected RWKV ffn == 4 * hidden with supported alignment");

  c10::cuda::CUDAGuard device_guard(preact.device());
  auto stream = at::cuda::getCurrentCUDAStream(preact.get_device());
  const int64_t vec4_count = output.numel() / 8;
  zero_output_vec4_kernel<<<static_cast<int>((vec4_count + 127) / 128), 128, 0, stream>>>(
      reinterpret_cast<half*>(output.data_ptr<at::Half>()), vec4_count);
  blackwell_sparse_relu2_down_row1_kernel<<<
      dim3(static_cast<unsigned>(ffn / FFN_TILE),
           static_cast<unsigned>(hidden / (2 * THREADS)), 1),
      THREADS, 0, stream>>>(
      static_cast<int>(hidden),
      preact.data_ptr<at::Half>(),
      packed_value.data_ptr<at::Half>(),
      output.data_ptr<at::Half>());
  const int64_t pairs = output.numel() / 2;
  add_residual_half2_kernel<<<static_cast<int>((pairs + 255) / 256), 256, 0, stream>>>(
      reinterpret_cast<const half*>(residual.data_ptr<at::Half>()),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor rwkv7_ada_linear_cuda(torch::Tensor x, torch::Tensor weight) {
  TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(x.scalar_type() == at::kHalf && weight.scalar_type() == at::kHalf,
              "fp16 tensors required");
  TORCH_CHECK(x.dim() == 2 && weight.dim() == 2, "x and weight must be rank-2");
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(), "contiguous tensors required");
  const int64_t rows = x.size(0);
  const int64_t hidden = x.size(1);
  const int64_t ffn = weight.size(0);
  TORCH_CHECK(rows == 1 || rows == 2 || rows == 4,
              "sm_89 linear supports one, two, or four rows");
  TORCH_CHECK(weight.size(1) == hidden, "linear shape mismatch");
  TORCH_CHECK((hidden % 4) == 0 && (ffn % 2) == 0,
              "linear input must be divisible by four and output by two");

  c10::cuda::CUDAGuard device_guard(x.device());
  auto output = torch::empty({rows, ffn}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  if (rows == 1) {
    ffn_up_row1_exact4_kernel<128, 2><<<
        dim3(static_cast<unsigned>(ffn / 2), 1, 1), 128, 0, stream>>>(
        static_cast<int>(hidden), static_cast<int>(ffn),
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()));
  } else if (rows == 2) {
    ffn_up_row2_exact4_kernel<64, 2><<<
        dim3(static_cast<unsigned>(ffn / 2), 1, 1), 64, 0, stream>>>(
        static_cast<int>(hidden), static_cast<int>(ffn),
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()));
  } else if (ffn == hidden) {
    ffn_up_rows_kernel<128, 2, 2><<<
        dim3(static_cast<unsigned>(ffn / 2), 2, 1), 128, 0, stream>>>(
        static_cast<int>(rows), static_cast<int>(hidden), static_cast<int>(ffn),
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()));
  } else if (ffn == 4 * hidden) {
    ffn_up_rows_kernel<64, 2, 4><<<
        dim3(static_cast<unsigned>(ffn / 4), 2, 1), 64, 0, stream>>>(
        static_cast<int>(rows), static_cast<int>(hidden), static_cast<int>(ffn),
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()));
  } else {
    TORCH_CHECK(false, "four-row sm_89 linear supports square or 4x expansion shapes");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor rwkv7_ada_sparse_ffn_out_cuda(
    torch::Tensor preact,
    torch::Tensor packed_value,
    torch::Tensor residual,
    torch::Tensor output) {
  TORCH_CHECK(preact.is_cuda() && packed_value.is_cuda() && residual.is_cuda() && output.is_cuda(),
              "CUDA tensors required");
  TORCH_CHECK(preact.scalar_type() == at::kHalf &&
              packed_value.scalar_type() == at::kHalf &&
              residual.scalar_type() == at::kHalf &&
              output.scalar_type() == at::kHalf, "fp16 tensors required");
  TORCH_CHECK(preact.dim() == 2 && packed_value.dim() == 2 && residual.dim() == 2 && output.dim() == 2,
              "preact, packed_value, residual, and output must be rank-2");
  TORCH_CHECK(preact.is_contiguous() && packed_value.is_contiguous() && residual.is_contiguous() && output.is_contiguous(),
              "contiguous tensors required");
  const int64_t rows = preact.size(0);
  const int64_t ffn = preact.size(1);
  const int64_t hidden = residual.size(1);
  TORCH_CHECK(rows >= 1 && rows <= 19, "sparse FFN supports 1..19 rows");
  TORCH_CHECK(residual.size(0) == rows, "residual row mismatch");
  TORCH_CHECK(output.sizes() == residual.sizes(), "output shape must match residual");
  TORCH_CHECK(packed_value.size(0) == ffn && packed_value.size(1) == hidden,
              "packed value weight must have shape [ffn, hidden]");
  TORCH_CHECK(ffn == 4 * hidden, "expected RWKV ffn == 4 * hidden");
  TORCH_CHECK((ffn % FFN_TILE) == 0 && (hidden % (2 * THREADS)) == 0,
              "ffn must be divisible by 128 and hidden by 256");

  c10::cuda::CUDAGuard device_guard(preact.device());
  auto stream = at::cuda::getCurrentCUDAStream(preact.get_device());
  const int64_t vec4_count = output.numel() / 8;
  if (output.data_ptr() != residual.data_ptr()) {
    copy_residual_vec4_kernel<<<static_cast<int>((vec4_count + 127) / 128), 128, 0, stream>>>(
        reinterpret_cast<const half*>(residual.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        vec4_count);
  }
  sparse_relu2_down_rows_kernel<<<
      dim3(static_cast<unsigned>(ffn / FFN_TILE),
           static_cast<unsigned>(hidden / (2 * THREADS)),
           static_cast<unsigned>(rows)),
      THREADS, 0, stream>>>(
      static_cast<int>(hidden),
      static_cast<int>(ffn),
      reinterpret_cast<const half*>(preact.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(packed_value.data_ptr<at::Half>()),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor rwkv7_ada_sparse_ffn_cuda(
    torch::Tensor preact,
    torch::Tensor packed_value,
    torch::Tensor residual) {
  auto output = torch::empty_like(residual);
  return rwkv7_ada_sparse_ffn_out_cuda(preact, packed_value, residual, output);
}

torch::Tensor rwkv7_ada_sparse_ffn_fp32_out_cuda(
    torch::Tensor preact,
    torch::Tensor packed_value,
    torch::Tensor residual,
    torch::Tensor scratch,
    torch::Tensor output) {
  TORCH_CHECK(preact.is_cuda() && packed_value.is_cuda() && residual.is_cuda() && scratch.is_cuda() && output.is_cuda(),
              "CUDA tensors required");
  TORCH_CHECK(preact.scalar_type() == at::kHalf &&
              packed_value.scalar_type() == at::kHalf &&
              residual.scalar_type() == at::kHalf &&
              output.scalar_type() == at::kHalf, "fp16 tensors required");
  TORCH_CHECK(scratch.scalar_type() == at::kFloat, "scratch must be fp32");
  TORCH_CHECK(preact.dim() == 2 && packed_value.dim() == 2 && residual.dim() == 2 && scratch.dim() == 2 && output.dim() == 2,
              "preact, packed_value, residual, scratch, and output must be rank-2");
  TORCH_CHECK(preact.is_contiguous() && packed_value.is_contiguous() && residual.is_contiguous() && scratch.is_contiguous() && output.is_contiguous(),
              "contiguous tensors required");
  const int64_t rows = preact.size(0);
  const int64_t ffn = preact.size(1);
  const int64_t hidden = residual.size(1);
  TORCH_CHECK(rows >= 1 && rows <= 19, "sparse FFN supports 1..19 rows");
  TORCH_CHECK(residual.size(0) == rows, "residual row mismatch");
  TORCH_CHECK(output.sizes() == residual.sizes(), "output shape must match residual");
  TORCH_CHECK(scratch.sizes() == residual.sizes(), "scratch shape must match residual");
  TORCH_CHECK(packed_value.size(0) == ffn && packed_value.size(1) == hidden,
              "packed value weight must have shape [ffn, hidden]");
  TORCH_CHECK(ffn == 4 * hidden, "expected RWKV ffn == 4 * hidden");
  TORCH_CHECK((ffn % FFN_TILE) == 0 && (hidden % (2 * THREADS)) == 0,
              "ffn must be divisible by 128 and hidden by 256");

  c10::cuda::CUDAGuard device_guard(preact.device());
  auto stream = at::cuda::getCurrentCUDAStream(preact.get_device());
  sparse_relu2_down_fp32_kernel<<<
      dim3(static_cast<unsigned>(ffn / FFN_TILE),
           static_cast<unsigned>(hidden / (2 * THREADS)),
           static_cast<unsigned>(rows)),
      THREADS, 0, stream>>>(
      static_cast<int>(hidden),
      static_cast<int>(ffn),
      reinterpret_cast<const half*>(preact.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(packed_value.data_ptr<at::Half>()),
      scratch.data_ptr<float>());
  const int64_t elements = output.numel();
  finalize_sparse_fp32_add_residual_kernel<<<
      static_cast<int>((elements + 255) / 256), 256, 0, stream>>>(
      elements,
      scratch.data_ptr<float>(),
      reinterpret_cast<const half*>(residual.data_ptr<at::Half>()),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor rwkv7_ada_sparse_ffn_fp32_cuda(
    torch::Tensor preact,
    torch::Tensor packed_value,
    torch::Tensor residual,
    torch::Tensor scratch) {
  auto output = torch::empty_like(residual);
  return rwkv7_ada_sparse_ffn_fp32_out_cuda(
      preact, packed_value, residual, scratch, output);
}

torch::Tensor rwkv7_ada_sparse_ffn_official_out_cuda(
    torch::Tensor preact,
    torch::Tensor packed_value,
    torch::Tensor residual,
    torch::Tensor output) {
  TORCH_CHECK(preact.is_cuda() && packed_value.is_cuda() && residual.is_cuda() && output.is_cuda(),
              "CUDA tensors required");
  TORCH_CHECK(preact.scalar_type() == at::kHalf && packed_value.scalar_type() == at::kHalf
              && residual.scalar_type() == at::kHalf && output.scalar_type() == at::kHalf,
              "fp16 tensors required");
  TORCH_CHECK(preact.dim() == 2 && packed_value.dim() == 2 && residual.dim() == 2 && output.dim() == 2,
              "preact, packed_value, residual, and output must be rank-2");
  TORCH_CHECK(preact.is_contiguous() && packed_value.is_contiguous() && residual.is_contiguous() && output.is_contiguous(),
              "contiguous tensors required");
  const int64_t rows = preact.size(0);
  const int64_t ffn = preact.size(1);
  const int64_t hidden = residual.size(1);
  TORCH_CHECK(rows >= 1 && rows <= 19, "sparse FFN supports 1..19 rows");
  TORCH_CHECK(residual.size(0) == rows && output.sizes() == residual.sizes(),
              "residual/output shape mismatch");
  TORCH_CHECK(packed_value.size(0) == ffn && packed_value.size(1) == hidden,
              "packed value weight must have shape [ffn, hidden]");
  TORCH_CHECK(ffn == 4 * hidden && (ffn % FFN_TILE) == 0 && (hidden % 256) == 0,
              "expected RWKV ffn == 4 * hidden with supported alignment");

  c10::cuda::CUDAGuard device_guard(preact.device());
  auto stream = at::cuda::getCurrentCUDAStream(preact.get_device());
  const int64_t vec4_count = output.numel() / 8;
  zero_output_vec4_kernel<<<static_cast<int>((vec4_count + 127) / 128), 128, 0, stream>>>(
      reinterpret_cast<half*>(output.data_ptr<at::Half>()), vec4_count);
  if (rows >= 8 && (ffn % 512) == 0 && (hidden % 512) == 0) {
    sparse_relu2_down_rows_t512_kernel<<<
        dim3(static_cast<unsigned>(ffn / 512), static_cast<unsigned>(hidden / 512), static_cast<unsigned>(rows)),
        256, 0, stream>>>(
        static_cast<int>(hidden), static_cast<int>(ffn),
        reinterpret_cast<const half*>(preact.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(packed_value.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()));
  } else {
    sparse_relu2_down_rows_kernel<<<
        dim3(static_cast<unsigned>(ffn / FFN_TILE),
             static_cast<unsigned>(hidden / (2 * THREADS)),
             static_cast<unsigned>(rows)),
        THREADS, 0, stream>>>(
        static_cast<int>(hidden), static_cast<int>(ffn),
        reinterpret_cast<const half*>(preact.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(packed_value.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()));
  }
  const int64_t pairs = output.numel() / 2;
  add_residual_half2_kernel<<<static_cast<int>((pairs + 255) / 256), 256, 0, stream>>>(
      reinterpret_cast<const half*>(residual.data_ptr<at::Half>()),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor rwkv7_ada_sparse_ffn_official_cuda(
    torch::Tensor preact,
    torch::Tensor packed_value,
    torch::Tensor residual) {
  auto output = torch::empty_like(residual);
  return rwkv7_ada_sparse_ffn_official_out_cuda(
      preact, packed_value, residual, output);
}

torch::Tensor rwkv7_ada_sparse_ffn_deterministic4_out_cuda(
    torch::Tensor preact,
    torch::Tensor packed_value,
    torch::Tensor residual,
    torch::Tensor scratch,
    torch::Tensor output) {
  TORCH_CHECK(preact.is_cuda() && packed_value.is_cuda() && residual.is_cuda()
              && scratch.is_cuda() && output.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(preact.scalar_type() == at::kHalf && packed_value.scalar_type() == at::kHalf
              && residual.scalar_type() == at::kHalf && scratch.scalar_type() == at::kHalf
              && output.scalar_type() == at::kHalf, "fp16 tensors required");
  TORCH_CHECK(preact.dim() == 2 && packed_value.dim() == 2 && residual.dim() == 2
              && scratch.dim() == 3 && output.dim() == 2, "invalid tensor ranks");
  TORCH_CHECK(preact.is_contiguous() && packed_value.is_contiguous()
              && residual.is_contiguous() && scratch.is_contiguous()
              && output.is_contiguous(), "contiguous tensors required");
  const int64_t rows = preact.size(0);
  const int64_t ffn = preact.size(1);
  const int64_t hidden = residual.size(1);
  TORCH_CHECK(rows >= 8 && rows <= 19, "deterministic four-way FFN supports 8..19 rows");
  TORCH_CHECK(residual.size(0) == rows && output.sizes() == residual.sizes(),
              "residual/output shape mismatch");
  TORCH_CHECK(packed_value.size(0) == ffn && packed_value.size(1) == hidden,
              "packed value weight must have shape [ffn, hidden]");
  TORCH_CHECK(scratch.size(0) == 4 && scratch.size(1) == rows && scratch.size(2) == hidden,
              "scratch must have shape [4, rows, hidden]");
  TORCH_CHECK(ffn == 4 * hidden && (ffn % 2048) == 0 && (hidden % 512) == 0,
              "deterministic four-way FFN requires aligned RWKV 4x expansion");

  c10::cuda::CUDAGuard device_guard(preact.device());
  auto stream = at::cuda::getCurrentCUDAStream(preact.get_device());
  sparse_relu2_down_deterministic4_kernel<<<
      dim3(4, static_cast<unsigned>(hidden / 512), static_cast<unsigned>(rows)),
      256, 0, stream>>>(
      static_cast<int>(hidden), static_cast<int>(ffn), static_cast<int>(rows),
      reinterpret_cast<const half*>(preact.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(packed_value.data_ptr<at::Half>()),
      reinterpret_cast<half*>(scratch.data_ptr<at::Half>()));
  finalize_sparse_deterministic4_kernel<<<
      dim3(static_cast<unsigned>(hidden / 512), static_cast<unsigned>(rows), 1),
      256, 0, stream>>>(
      static_cast<int>(hidden), static_cast<int>(rows),
      reinterpret_cast<const half*>(scratch.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(residual.data_ptr<at::Half>()),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor rwkv7_ada_sparse_ffn_deterministic4_cuda(
    torch::Tensor preact,
    torch::Tensor packed_value,
    torch::Tensor residual,
    torch::Tensor scratch) {
  auto output = torch::empty_like(residual);
  return rwkv7_ada_sparse_ffn_deterministic4_out_cuda(
      preact, packed_value, residual, scratch, output);
}
"""


_EXTENSION: Any | None = None
_EXTENSION_ERROR: str | None = None
_EXTENSION_LOCK = threading.Lock()
_PACK_LOCK = threading.Lock()
_PACKED_WEIGHTS: dict[tuple[Any, ...], tuple[weakref.ReferenceType[Any], Any]] = {}
_FP32_SCRATCH: dict[tuple[Any, ...], tuple[weakref.ReferenceType[Any], Any]] = {}
_DETERMINISTIC_SCRATCH: dict[tuple[Any, ...], tuple[weakref.ReferenceType[Any], Any]] = {}


def _is_sparse_ffn_device(device: Any = None) -> bool:
    if torch is None or not torch.cuda.is_available():
        return False
    try:
        resolved = torch.device("cuda" if device is None else device)
        if resolved.type != "cuda":
            return False
        index = torch.cuda.current_device() if resolved.index is None else int(resolved.index)
        return tuple(int(v) for v in torch.cuda.get_device_capability(index)) in {
            (7, 0),
            (8, 9),
            (12, 0),
        }
    except Exception:
        return False


def _is_blackwell_device(device: Any = None) -> bool:
    if torch is None or not torch.cuda.is_available():
        return False
    try:
        resolved = torch.device("cuda" if device is None else device)
        if resolved.type != "cuda":
            return False
        index = torch.cuda.current_device() if resolved.index is None else int(resolved.index)
        return tuple(int(v) for v in torch.cuda.get_device_capability(index)) == (12, 0)
    except Exception:
        return False


def blackwell_cmix_should_use(rows: int, outputs: int, inputs: int) -> bool:
    """Return whether the opt-in SM120 row-one CMIX kernel supports a shape."""

    return (
        int(rows) == 1
        and int(inputs) == 4 * int(outputs)
        and int(outputs) % 256 == 0
    )


def _blackwell_cmix_enabled(device: Any = None) -> bool:
    return (
        _policy_flag("RWKV7_NATIVE_GRAPH_BLACKWELL_CMIX", "blackwell_cmix")
        and _is_blackwell_device(device)
    )


def _load_extension() -> Any | None:
    global _EXTENSION, _EXTENSION_ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    if _EXTENSION_ERROR is not None or torch is None or not _is_sparse_ffn_device():
        return None
    with _EXTENSION_LOCK:
        if _EXTENSION is not None:
            return _EXTENSION
        if _EXTENSION_ERROR is not None:
            return None
        try:
            # Keep the virtualenv bin directory. Resolving the Python symlink
            # can incorrectly replace it with /usr/bin and hide venv tools
            # such as Ninja from torch.utils.cpp_extension.
            python_bin = str(Path(sys.executable).absolute().parent)
            path_items = os.environ.get("PATH", "").split(os.pathsep)
            if python_bin not in path_items:
                os.environ["PATH"] = python_bin + os.pathsep + os.environ.get("PATH", "")
            nvcc = Path(python_bin) / "nvcc"
            if nvcc.exists() and "CUDA_HOME" not in os.environ:
                os.environ["CUDA_HOME"] = str(nvcc.parent.parent)
            capability = torch.cuda.get_device_capability()
            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{capability[0]}.{capability[1]}")
            runtime_lib = (
                Path(sys.prefix)
                / "lib"
                / f"python{sys.version_info.major}.{sys.version_info.minor}"
                / "site-packages"
                / "nvidia"
                / "cuda_runtime"
                / "lib"
            )
            extra_ldflags: list[str] = []
            if runtime_lib.is_dir():
                for variable in ("LIBRARY_PATH", "LD_LIBRARY_PATH"):
                    items = os.environ.get(variable, "").split(os.pathsep)
                    if str(runtime_lib) not in items:
                        os.environ[variable] = str(runtime_lib) + os.pathsep + os.environ.get(variable, "")
                extra_ldflags.append(f"-Wl,-rpath,{runtime_lib}")
            from torch.utils.cpp_extension import load_inline

            _EXTENSION = load_inline(
                name="rwkv7_sparse_ffn_v20",
                cpp_sources=_CPP_SOURCE,
                cuda_sources=_CUDA_SOURCE,
                functions=None,
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3", "--use_fast_math", "--extra-device-vectorization"],
                extra_ldflags=extra_ldflags,
                with_cuda=True,
                verbose=os.environ.get("RWKV7_ADA_SPARSE_FFN_BUILD_VERBOSE", "0").lower()
                in {"1", "true", "yes", "on"},
            )
        except Exception as exc:  # pragma: no cover - depends on host toolchain
            _EXTENSION_ERROR = f"{type(exc).__name__}: {exc}"
            return None
    return _EXTENSION


def ada_sparse_ffn_should_use(rows: int, outputs: int, inputs: int) -> bool:
    """Return whether a shape is in the measured sm_89 sparse decode set."""

    rows, outputs, inputs = int(rows), int(outputs), int(inputs)
    return 1 <= rows <= 19 and inputs == 4 * outputs and outputs % 256 == 0


def ada_ffn_up_should_use(rows: int, outputs: int, inputs: int) -> bool:
    rows, outputs, inputs = int(rows), int(outputs), int(inputs)
    return 1 <= rows <= 2 and outputs == 4 * inputs and inputs % 256 == 0


def ada_linear_should_use(rows: int, outputs: int, inputs: int) -> bool:
    rows, outputs, inputs = int(rows), int(outputs), int(inputs)
    common = outputs > 0 and outputs % 2 == 0 and inputs >= 1024 and inputs % 4 == 0
    return common and (1 <= rows <= 2 or (rows == 4 and outputs in {inputs, 4 * inputs}))


def ada_sparse_ffn_available(device: Any = None, *, build: bool = False) -> bool:
    if not _is_sparse_ffn_device(device):
        return False
    return _load_extension() is not None if build else True


def ada_sparse_ffn_build_error() -> str | None:
    return _EXTENSION_ERROR


def _weight_cache_key(weight: Any, cache_tag: Any = None) -> tuple[Any, ...]:
    device = weight.device
    index = device.index
    if device.type == "cuda" and index is None and torch is not None:
        index = torch.cuda.current_device()
    if index is None:
        index = -1
    try:
        version = int(weight._version)
    except RuntimeError:
        # Tensors constructed under inference_mode intentionally have no
        # version counter.  Their storage is immutable for this route.
        version = -1
    return (
        str(device.type),
        int(index),
        int(weight.data_ptr()),
        version,
        tuple(int(v) for v in weight.shape),
        weight.dtype,
        cache_tag,
    )


def ada_sparse_ffn_pack_weight(weight: Any, *, cache_tag: Any = None) -> Any:
    """Return a cached contiguous ``[ffn, hidden]`` inference layout."""

    if _policy_flag(
        "RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_SHARE_PACK",
        "ada_sparse_ffn_share_pack",
    ):
        cache_tag = None
    key = _weight_cache_key(weight, cache_tag)
    cached = _PACKED_WEIGHTS.get(key)
    if cached is not None and cached[0]() is weight:
        return cached[1]
    with _PACK_LOCK:
        cached = _PACKED_WEIGHTS.get(key)
        if cached is not None and cached[0]() is weight:
            return cached[1]
        if torch is not None and weight.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "sparse FFN weights must be packed before CUDA graph capture; "
                "call prewarm_ada_sparse_ffn first"
            )
        packed = weight.transpose(0, 1).contiguous()
        _PACKED_WEIGHTS[key] = (weakref.ref(weight), packed)
        stale = [item for item, value in _PACKED_WEIGHTS.items() if value[0]() is None]
        for item in stale:
            _PACKED_WEIGHTS.pop(item, None)
        return packed


def ada_sparse_ffn_prepare_fp32_scratch(weight: Any, rows: int) -> Any:
    """Preallocate graph-stable FP32 accumulation storage for one batch shape."""

    rows = int(rows)
    key = _weight_cache_key(weight, ("fp32_scratch", rows))
    cached = _FP32_SCRATCH.get(key)
    if cached is not None and cached[0]() is weight:
        return cached[1]
    with _PACK_LOCK:
        cached = _FP32_SCRATCH.get(key)
        if cached is not None and cached[0]() is weight:
            return cached[1]
        if torch is not None and weight.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "sparse FFN FP32 scratch must be allocated before CUDA graph capture; "
                "call prewarm_ada_sparse_ffn first"
            )
        scratch = torch.zeros(
            rows,
            int(weight.shape[0]),
            dtype=torch.float32,
            device=weight.device,
        )
        _FP32_SCRATCH[key] = (weakref.ref(weight), scratch)
        return scratch


def ada_sparse_ffn_prepare_deterministic_scratch(weight: Any, rows: int) -> Any:
    """Preallocate graph-stable four-way FP16 partial sums for B8+ decode."""

    rows = int(rows)
    key = _weight_cache_key(weight, ("deterministic4_scratch", rows))
    cached = _DETERMINISTIC_SCRATCH.get(key)
    if cached is not None and cached[0]() is weight:
        return cached[1]
    with _PACK_LOCK:
        cached = _DETERMINISTIC_SCRATCH.get(key)
        if cached is not None and cached[0]() is weight:
            return cached[1]
        if torch is not None and weight.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "sparse FFN deterministic scratch must be allocated before CUDA graph capture; "
                "call prewarm_ada_sparse_ffn first"
            )
        scratch = torch.empty(
            4,
            rows,
            int(weight.shape[0]),
            dtype=torch.float16,
            device=weight.device,
        )
        _DETERMINISTIC_SCRATCH[key] = (weakref.ref(weight), scratch)
        return scratch


def clear_ada_sparse_ffn_weight_cache() -> None:
    with _PACK_LOCK:
        _PACKED_WEIGHTS.clear()
        _FP32_SCRATCH.clear()
        _DETERMINISTIC_SCRATCH.clear()


def ada_sparse_ffn_down_add(
    preact: Any,
    weight: Any,
    residual: Any,
    *,
    out: Any | None = None,
    force_fallback: bool = False,
) -> Any:
    """Apply sparse ``ReLU²`` contraction and residual add on sm_89 decode."""

    if torch is None or F is None:
        raise RuntimeError("ada_sparse_ffn_down_add requires torch")
    scalar = preact.dim() == 1
    preact2 = preact.reshape(1, -1) if scalar else preact
    residual2 = residual.reshape(1, -1) if scalar else residual
    rows, inputs = int(preact2.shape[0]), int(preact2.shape[1])
    outputs = int(weight.shape[0])
    valid = bool(
        not force_fallback
        and not torch.is_grad_enabled()
        and ada_sparse_ffn_should_use(rows, outputs, inputs)
        and preact2.is_cuda
        and weight.is_cuda
        and residual2.is_cuda
        and preact2.dtype == torch.float16
        and weight.dtype == torch.float16
        and residual2.dtype == torch.float16
        and preact2.is_contiguous()
        and (weight.is_contiguous() or weight.transpose(0, 1).is_contiguous())
        and residual2.is_contiguous()
        and tuple(weight.shape) == (outputs, inputs)
        and tuple(residual2.shape) == (rows, outputs)
        and _is_sparse_ffn_device(preact2.device)
    )
    extension = _load_extension() if valid else None
    if extension is None:
        result = residual + F.linear(torch.relu(preact) ** 2, weight)
        if out is not None:
            if tuple(out.shape) != tuple(result.shape):
                raise ValueError(
                    f"out shape must match result shape {tuple(result.shape)}; got {tuple(out.shape)}"
                )
            out.copy_(result)
            return out
        return result
    # Batch-shape graph runners keep distinct packed storage by default. The
    # opt-in shared-pack route is limited to immutable inference weights and
    # must be revalidated when graph capture or the sparse kernel changes.
    packed = ada_sparse_ffn_pack_weight(weight, cache_tag=rows)
    blackwell_cmix = bool(
        _blackwell_cmix_enabled(preact2.device)
        and blackwell_cmix_should_use(rows, outputs, inputs)
    )
    if blackwell_cmix:
        out2 = torch.empty_like(residual2) if out is None else (
            out.reshape(1, -1) if scalar else out
        )
        if tuple(out2.shape) != tuple(residual2.shape):
            raise ValueError(
                f"out shape must match residual shape {tuple(residual2.shape)}; got {tuple(out2.shape)}"
            )
        output = extension.blackwell_sparse_down_add_out(
            preact2, packed, residual2, out2
        )
        return output.reshape(outputs) if scalar else output
    fp32_accum = _policy_flag(
        "RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_FP32_ACCUM",
        "ada_sparse_ffn_fp32_accum",
    )
    official_boundary = _policy_flag(
        "RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_OFFICIAL_BOUNDARY",
        "ada_sparse_ffn_official_boundary",
    )
    policy = _kernel_policy()
    deterministic_splits = int(
        os.environ.get(
            "RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_DETERMINISTIC_SPLITS",
            str(getattr(policy, "ada_sparse_ffn_deterministic_splits", 0)),
        )
    )
    if deterministic_splits not in {0, 4}:
        raise ValueError("RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_DETERMINISTIC_SPLITS must be 0 or 4")
    deterministic = deterministic_splits == 4 and rows >= 8
    scratch = ada_sparse_ffn_prepare_fp32_scratch(weight, rows) if fp32_accum else None
    deterministic_scratch = (
        ada_sparse_ffn_prepare_deterministic_scratch(weight, rows)
        if deterministic and not fp32_accum
        else None
    )
    if out is None:
        if fp32_accum:
            output = extension.sparse_down_add_fp32(
                preact2, packed, residual2, scratch
            )
        elif deterministic:
            output = extension.sparse_down_add_deterministic4(
                preact2, packed, residual2, deterministic_scratch
            )
        elif official_boundary:
            output = extension.sparse_down_add_official(
                preact2, packed, residual2
            )
        else:
            output = extension.sparse_down_add(preact2, packed, residual2)
    else:
        out2 = out.reshape(1, -1) if scalar else out
        if tuple(out2.shape) != tuple(residual2.shape):
            raise ValueError(
                f"out shape must match residual shape {tuple(residual2.shape)}; got {tuple(out2.shape)}"
            )
        if fp32_accum:
            output = extension.sparse_down_add_fp32_out(
                preact2, packed, residual2, scratch, out2
            )
        elif deterministic:
            output = extension.sparse_down_add_deterministic4_out(
                preact2, packed, residual2, deterministic_scratch, out2
            )
        elif official_boundary:
            output = extension.sparse_down_add_official_out(
                preact2, packed, residual2, out2
            )
        else:
            output = extension.sparse_down_add_out(
                preact2, packed, residual2, out2
            )
    return output.reshape(outputs) if scalar else output


def ada_ffn_up(x: Any, weight: Any, *, force_fallback: bool = False) -> Any:
    """Apply the measured no-copy small-row FFN expansion on sm_89."""

    if torch is None or F is None:
        raise RuntimeError("ada_ffn_up requires torch")
    scalar = x.dim() == 1
    x2 = x.reshape(1, -1) if scalar else x
    rows, inputs = int(x2.shape[0]), int(x2.shape[1])
    outputs = int(weight.shape[0])
    valid = bool(
        not force_fallback
        and not torch.is_grad_enabled()
        and ada_ffn_up_should_use(rows, outputs, inputs)
        and x2.is_cuda
        and weight.is_cuda
        and x2.dtype == torch.float16
        and weight.dtype == torch.float16
        and x2.is_contiguous()
        and weight.is_contiguous()
        and tuple(weight.shape) == (outputs, inputs)
        and _is_sparse_ffn_device(x2.device)
    )
    extension = _load_extension() if valid else None
    if extension is None:
        return F.linear(x, weight)
    if (
        _blackwell_cmix_enabled(x2.device)
        and rows == 1
        and outputs == 4 * inputs
        and inputs % 256 == 0
    ):
        output = extension.blackwell_ffn_up(x2, weight)
    else:
        output = extension.ffn_up(x2, weight)
    return output.reshape(outputs) if scalar else output


def ada_linear(x: Any, weight: Any, *, force_fallback: bool = False) -> Any:
    """Apply the no-copy exact-row sm_89 linear probe with a torch fallback."""

    if torch is None or F is None:
        raise RuntimeError("ada_linear requires torch")
    scalar = x.dim() == 1
    x2 = x.reshape(1, -1) if scalar else x
    rows, inputs = int(x2.shape[0]), int(x2.shape[1])
    outputs = int(weight.shape[0])
    valid = bool(
        not force_fallback
        and not torch.is_grad_enabled()
        and ada_linear_should_use(rows, outputs, inputs)
        and x2.is_cuda
        and weight.is_cuda
        and x2.dtype == torch.float16
        and weight.dtype == torch.float16
        and x2.is_contiguous()
        and weight.is_contiguous()
        and tuple(weight.shape) == (outputs, inputs)
        and _is_sparse_ffn_device(x2.device)
    )
    extension = _load_extension() if valid else None
    if extension is None:
        return F.linear(x, weight)
    output = extension.linear(x2, weight)
    return output.reshape(outputs) if scalar else output


__all__ = [
    "ada_ffn_up",
    "ada_ffn_up_should_use",
    "ada_linear",
    "ada_linear_should_use",
    "ada_sparse_ffn_available",
    "ada_sparse_ffn_build_error",
    "ada_sparse_ffn_down_add",
    "ada_sparse_ffn_pack_weight",
    "ada_sparse_ffn_prepare_deterministic_scratch",
    "ada_sparse_ffn_prepare_fp32_scratch",
    "ada_sparse_ffn_should_use",
    "blackwell_cmix_should_use",
    "clear_ada_sparse_ffn_weight_cache",
]
