# coding=utf-8
"""Optional CUDA state-scan prototypes for RWKV-7 native prefill.

This module is intentionally experimental.  It JIT-builds a tiny CUDA extension
for the current 4090 target shape (head_dim=64) so we can compare a real CUDA
state-layout baseline against the Triton full-head scan before committing to a
larger persistent-kernel rewrite.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

try:  # pragma: no cover - optional on CPU-only hosts
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


_CPP_SRC = r"""
#include <torch/extension.h>

std::vector<torch::Tensor> rwkv7_state_scan_prep_forward_cuda(
    torch::Tensor r,
    torch::Tensor w_raw,
    torch::Tensor k_raw,
    torch::Tensor v_raw,
    torch::Tensor a,
    torch::Tensor state,
    torch::Tensor k_k,
    torch::Tensor k_a,
    torch::Tensor v_first,
    torch::Tensor v_gate,
    bool has_v_gate,
    int lanes_per_row,
    int precompute_mode,
    int rows_per_block,
    int schedule_mode,
    bool w_precomputed);

std::vector<torch::Tensor> rwkv7_state_scan_prep_sk_forward_cuda(
    torch::Tensor r,
    torch::Tensor w_raw,
    torch::Tensor k_raw,
    torch::Tensor v_raw,
    torch::Tensor a,
    torch::Tensor state,
    torch::Tensor k_k,
    torch::Tensor k_a,
    torch::Tensor r_k,
    torch::Tensor v_first,
    torch::Tensor v_gate,
    bool has_v_gate,
    int rows_per_block,
    int schedule_mode);

std::vector<torch::Tensor> rwkv7_state_scan_rowblock_phase_forward_cuda(
    torch::Tensor r,
    torch::Tensor w_raw,
    torch::Tensor k_raw,
    torch::Tensor v_raw,
    torch::Tensor a,
    torch::Tensor state,
    torch::Tensor k_k,
    torch::Tensor k_a,
    torch::Tensor v_first,
    torch::Tensor v_gate,
    bool has_v_gate,
    int phase);

std::vector<torch::Tensor> state_scan_prep_forward(
    torch::Tensor r,
    torch::Tensor w_raw,
    torch::Tensor k_raw,
    torch::Tensor v_raw,
    torch::Tensor a,
    torch::Tensor state,
    torch::Tensor k_k,
    torch::Tensor k_a,
    torch::Tensor v_first,
    torch::Tensor v_gate,
    bool has_v_gate,
    int lanes_per_row,
    int precompute_mode,
    int rows_per_block,
    int schedule_mode,
    bool w_precomputed) {
  return rwkv7_state_scan_prep_forward_cuda(
      r, w_raw, k_raw, v_raw, a, state, k_k, k_a, v_first, v_gate, has_v_gate, lanes_per_row, precompute_mode, rows_per_block, schedule_mode, w_precomputed);
}

std::vector<torch::Tensor> state_scan_prep_sk_forward(
    torch::Tensor r,
    torch::Tensor w_raw,
    torch::Tensor k_raw,
    torch::Tensor v_raw,
    torch::Tensor a,
    torch::Tensor state,
    torch::Tensor k_k,
    torch::Tensor k_a,
    torch::Tensor r_k,
    torch::Tensor v_first,
    torch::Tensor v_gate,
    bool has_v_gate,
    int rows_per_block,
    int schedule_mode) {
  return rwkv7_state_scan_prep_sk_forward_cuda(
      r, w_raw, k_raw, v_raw, a, state, k_k, k_a, r_k, v_first, v_gate, has_v_gate, rows_per_block, schedule_mode);
}

std::vector<torch::Tensor> state_scan_rowblock_phase_forward(
    torch::Tensor r,
    torch::Tensor w_raw,
    torch::Tensor k_raw,
    torch::Tensor v_raw,
    torch::Tensor a,
    torch::Tensor state,
    torch::Tensor k_k,
    torch::Tensor k_a,
    torch::Tensor v_first,
    torch::Tensor v_gate,
    bool has_v_gate,
    int phase) {
  return rwkv7_state_scan_rowblock_phase_forward_cuda(
      r, w_raw, k_raw, v_raw, a, state, k_k, k_a, v_first, v_gate, has_v_gate, phase);
}
"""


_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Half.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#endif
#ifndef CHECK_CONTIGUOUS
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#endif
#ifndef CHECK_INPUT
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#endif

template <typename scalar_t>
__global__ void rwkv7_state_scan_prep_n64_kernel(
    const scalar_t* __restrict__ r,
    const scalar_t* __restrict__ w_raw,
    const scalar_t* __restrict__ k_raw,
    const scalar_t* __restrict__ v_raw,
    const scalar_t* __restrict__ a,
    const float* __restrict__ state,
    const scalar_t* __restrict__ k_k,
    const scalar_t* __restrict__ k_a,
    const scalar_t* __restrict__ v_first,
    const scalar_t* __restrict__ v_gate,
    scalar_t* __restrict__ out,
    float* __restrict__ final_state,
    scalar_t* __restrict__ k_out,
    scalar_t* __restrict__ v_out,
    int B,
    int T,
    int H,
    bool has_v_gate) {
  constexpr int N = 64;
  const int bh = blockIdx.x;
  const int head = bh % H;
  const int batch = bh / H;
  const int tid = threadIdx.x;

  __shared__ float st[N * N];
  __shared__ float r_s[N];
  __shared__ float w_s[N];
  __shared__ float k_s[N];
  __shared__ float v_s[N];
  __shared__ float a_s[N];
  __shared__ float kk_s[N];
  __shared__ float norm_inv_s;

  const int state_base = (batch * H + head) * N * N;
  for (int idx = tid; idx < N * N; idx += blockDim.x) {
    st[idx] = state[state_base + idx];
  }
  __syncthreads();

  const int param_base = head * N;
  for (int t = 0; t < T; ++t) {
    const int vec_base = ((batch * T + t) * H + head) * N;
    if (tid < N) {
      const float rv = static_cast<float>(r[vec_base + tid]);
      const float wr = static_cast<float>(w_raw[vec_base + tid]);
      const float kr = static_cast<float>(k_raw[vec_base + tid]);
      const float vr = static_cast<float>(v_raw[vec_base + tid]);
      const float av = static_cast<float>(a[vec_base + tid]);
      const float kk_raw = kr * static_cast<float>(k_k[param_base + tid]);
      r_s[tid] = rv;
      w_s[tid] = __expf(-0.606531f / (1.0f + __expf(-wr)));
      a_s[tid] = av;
      kk_s[tid] = kk_raw;
      k_s[tid] = kr * (1.0f + (av - 1.0f) * static_cast<float>(k_a[param_base + tid]));
      if (has_v_gate) {
        const float vf = static_cast<float>(v_first[vec_base + tid]);
        const float vg = static_cast<float>(v_gate[vec_base + tid]);
        v_s[tid] = vr + (vf - vr) * vg;
      } else {
        v_s[tid] = vr;
      }
    }
    __syncthreads();

    if (tid == 0) {
      float norm2 = 0.0f;
      #pragma unroll
      for (int j = 0; j < N; ++j) {
        norm2 += kk_s[j] * kk_s[j];
      }
      norm_inv_s = rsqrtf(fmaxf(norm2, 1.0e-20f));
    }
    __syncthreads();

    if (tid < N) {
      kk_s[tid] *= norm_inv_s;
    }
    __syncthreads();

    if (tid < N) {
      const int row = tid;
      float state_dot_kk = 0.0f;
      #pragma unroll
      for (int j = 0; j < N; ++j) {
        state_dot_kk += st[row * N + j] * kk_s[j];
      }

      float recurrent = 0.0f;
      const float v_row = v_s[row];
      #pragma unroll
      for (int j = 0; j < N; ++j) {
        const float new_st = st[row * N + j] * w_s[j]
            + v_row * k_s[j]
            - state_dot_kk * kk_s[j] * a_s[j];
        st[row * N + j] = new_st;
        recurrent += new_st * r_s[j];
      }
      out[vec_base + row] = static_cast<scalar_t>(recurrent);
      k_out[vec_base + row] = static_cast<scalar_t>(k_s[row]);
      v_out[vec_base + row] = static_cast<scalar_t>(v_s[row]);
    }
    __syncthreads();
  }

  for (int idx = tid; idx < N * N; idx += blockDim.x) {
    final_state[state_base + idx] = st[idx];
  }
}

template <typename scalar_t, int LANES_PER_ROW>
__global__ void rwkv7_state_scan_prep_n64_rowgroup_kernel(
    const scalar_t* __restrict__ r,
    const scalar_t* __restrict__ w_raw,
    const scalar_t* __restrict__ k_raw,
    const scalar_t* __restrict__ v_raw,
    const scalar_t* __restrict__ a,
    const float* __restrict__ state,
    const scalar_t* __restrict__ k_k,
    const scalar_t* __restrict__ k_a,
    const scalar_t* __restrict__ v_first,
    const scalar_t* __restrict__ v_gate,
    scalar_t* __restrict__ out,
    float* __restrict__ final_state,
    scalar_t* __restrict__ k_out,
    scalar_t* __restrict__ v_out,
    int B,
    int T,
    int H,
    bool has_v_gate) {
  constexpr int N = 64;
  constexpr int BLOCK_THREADS = N * LANES_PER_ROW;
  const int bh = blockIdx.x;
  const int head = bh % H;
  const int batch = bh / H;
  const int tid = threadIdx.x;
  const int row = tid / LANES_PER_ROW;
  const int lane = tid - row * LANES_PER_ROW;

  __shared__ float st[N * N];
  __shared__ float r_s[N];
  __shared__ float w_s[N];
  __shared__ float k_s[N];
  __shared__ float v_s[N];
  __shared__ float a_s[N];
  __shared__ float kk_s[N];
  __shared__ float norm_inv_s;
  __shared__ float row_dot_s[N];
  __shared__ float partial_dot_s[N * LANES_PER_ROW];
  __shared__ float partial_recur_s[N * LANES_PER_ROW];

  const int state_base = (batch * H + head) * N * N;
  for (int idx = tid; idx < N * N; idx += BLOCK_THREADS) {
    st[idx] = state[state_base + idx];
  }
  __syncthreads();

  const int param_base = head * N;
  for (int t = 0; t < T; ++t) {
    const int vec_base = ((batch * T + t) * H + head) * N;
    if (tid < N) {
      const float rv = static_cast<float>(r[vec_base + tid]);
      const float wr = static_cast<float>(w_raw[vec_base + tid]);
      const float kr = static_cast<float>(k_raw[vec_base + tid]);
      const float vr = static_cast<float>(v_raw[vec_base + tid]);
      const float av = static_cast<float>(a[vec_base + tid]);
      const float kk_raw = kr * static_cast<float>(k_k[param_base + tid]);
      r_s[tid] = rv;
      w_s[tid] = __expf(-0.606531f / (1.0f + __expf(-wr)));
      a_s[tid] = av;
      kk_s[tid] = kk_raw;
      k_s[tid] = kr * (1.0f + (av - 1.0f) * static_cast<float>(k_a[param_base + tid]));
      if (has_v_gate) {
        const float vf = static_cast<float>(v_first[vec_base + tid]);
        const float vg = static_cast<float>(v_gate[vec_base + tid]);
        v_s[tid] = vr + (vf - vr) * vg;
      } else {
        v_s[tid] = vr;
      }
    }
    __syncthreads();

    if (tid < N) {
      partial_dot_s[tid] = kk_s[tid] * kk_s[tid];
    }
    __syncthreads();

    if (tid == 0) {
      float norm2 = 0.0f;
      #pragma unroll
      for (int j = 0; j < N; ++j) {
        norm2 += partial_dot_s[j];
      }
      norm_inv_s = rsqrtf(fmaxf(norm2, 1.0e-20f));
    }
    __syncthreads();

    if (tid < N) {
      kk_s[tid] *= norm_inv_s;
    }
    __syncthreads();

    if (row < N) {
      float state_dot_kk = 0.0f;
      #pragma unroll
      for (int j = lane; j < N; j += LANES_PER_ROW) {
        state_dot_kk += st[row * N + j] * kk_s[j];
      }
      partial_dot_s[row * LANES_PER_ROW + lane] = state_dot_kk;
    }
    __syncthreads();

    if (row < N && lane == 0) {
      float state_dot_kk = 0.0f;
      #pragma unroll
      for (int l = 0; l < LANES_PER_ROW; ++l) {
        state_dot_kk += partial_dot_s[row * LANES_PER_ROW + l];
      }
      row_dot_s[row] = state_dot_kk;
    }
    __syncthreads();

    if (row < N) {
      const float state_dot_kk = row_dot_s[row];
      const float v_row = v_s[row];
      float recurrent = 0.0f;
      #pragma unroll
      for (int j = lane; j < N; j += LANES_PER_ROW) {
        const float new_st = st[row * N + j] * w_s[j]
            + v_row * k_s[j]
            - state_dot_kk * kk_s[j] * a_s[j];
        st[row * N + j] = new_st;
        recurrent += new_st * r_s[j];
      }
      partial_recur_s[row * LANES_PER_ROW + lane] = recurrent;
    }
    __syncthreads();

    if (row < N && lane == 0) {
      float recurrent = 0.0f;
      #pragma unroll
      for (int l = 0; l < LANES_PER_ROW; ++l) {
        recurrent += partial_recur_s[row * LANES_PER_ROW + l];
      }
      out[vec_base + row] = static_cast<scalar_t>(recurrent);
      k_out[vec_base + row] = static_cast<scalar_t>(k_s[row]);
      v_out[vec_base + row] = static_cast<scalar_t>(v_s[row]);
    }
    __syncthreads();
  }

  for (int idx = tid; idx < N * N; idx += BLOCK_THREADS) {
    final_state[state_base + idx] = st[idx];
  }
}

__device__ __forceinline__ float rwkv7_block_reduce_sum_64(float value, float* scratch) {
  const unsigned mask = 0xffffffffu;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(mask, value, offset);
  }
  if (lane == 0) {
    scratch[warp] = value;
  }
  __syncthreads();
  value = threadIdx.x < 2 ? scratch[lane] : 0.0f;
  if (warp == 0) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(mask, value, offset);
    }
  }
  if (threadIdx.x == 0) {
    scratch[0] = value;
  }
  __syncthreads();
  return scratch[0];
}

template <typename scalar_t>
__global__ void rwkv7_state_scan_prep_n64_rowblock_kernel(
    const scalar_t* __restrict__ r,
    const scalar_t* __restrict__ w_raw,
    const scalar_t* __restrict__ k_raw,
    const scalar_t* __restrict__ v_raw,
    const scalar_t* __restrict__ a,
    const float* __restrict__ state,
    const scalar_t* __restrict__ k_k,
    const scalar_t* __restrict__ k_a,
    const scalar_t* __restrict__ v_first,
    const scalar_t* __restrict__ v_gate,
    scalar_t* __restrict__ out,
    float* __restrict__ final_state,
    scalar_t* __restrict__ k_out,
    scalar_t* __restrict__ v_out,
    int B,
    int T,
    int H,
    bool has_v_gate) {
  constexpr int N = 64;
  const int block = blockIdx.x;
  const int row = block % N;
  const int bh = block / N;
  const int head = bh % H;
  const int batch = bh / H;
  const int col = threadIdx.x;

  __shared__ float partial[N];
  __shared__ float v_row_s;

  const int state_base = (batch * H + head) * N * N;
  const int param_base = head * N;
  float st = state[state_base + row * N + col];

  for (int t = 0; t < T; ++t) {
    const int vec_base = ((batch * T + t) * H + head) * N;
    const float rv = static_cast<float>(r[vec_base + col]);
    const float wr = static_cast<float>(w_raw[vec_base + col]);
    const float kr = static_cast<float>(k_raw[vec_base + col]);
    const float av = static_cast<float>(a[vec_base + col]);
    const float kk_raw = kr * static_cast<float>(k_k[param_base + col]);
    const float wv = __expf(-0.606531f / (1.0f + __expf(-wr)));
    const float kv = kr * (1.0f + (av - 1.0f) * static_cast<float>(k_a[param_base + col]));

    if (threadIdx.x == 0) {
      const float vr = static_cast<float>(v_raw[vec_base + row]);
      if (has_v_gate) {
        const float vf = static_cast<float>(v_first[vec_base + row]);
        const float vg = static_cast<float>(v_gate[vec_base + row]);
        v_row_s = vr + (vf - vr) * vg;
      } else {
        v_row_s = vr;
      }
      v_out[vec_base + row] = static_cast<scalar_t>(v_row_s);
    }
    if (row == 0) {
      k_out[vec_base + col] = static_cast<scalar_t>(kv);
    }

    const float norm2 = rwkv7_block_reduce_sum_64(kk_raw * kk_raw, partial);
    const float norm_inv = rsqrtf(fmaxf(norm2, 1.0e-20f));
    const float kk = kk_raw * norm_inv;
    const float state_dot_kk = rwkv7_block_reduce_sum_64(st * kk, partial);

    const float new_st = st * wv + v_row_s * kv - state_dot_kk * kk * av;
    st = new_st;
    const float recurrent = rwkv7_block_reduce_sum_64(new_st * rv, partial);
    if (threadIdx.x == 0) {
      out[vec_base + row] = static_cast<scalar_t>(recurrent);
    }
    __syncthreads();
  }

  final_state[state_base + row * N + col] = st;
}

template <typename scalar_t, int PHASE>
__global__ void rwkv7_state_scan_prep_n64_rowblock_phase_kernel(
    const scalar_t* __restrict__ r,
    const scalar_t* __restrict__ w_raw,
    const scalar_t* __restrict__ k_raw,
    const scalar_t* __restrict__ v_raw,
    const scalar_t* __restrict__ a,
    const float* __restrict__ state,
    const scalar_t* __restrict__ k_k,
    const scalar_t* __restrict__ k_a,
    const scalar_t* __restrict__ v_first,
    const scalar_t* __restrict__ v_gate,
    scalar_t* __restrict__ out,
    float* __restrict__ final_state,
    scalar_t* __restrict__ k_out,
    scalar_t* __restrict__ v_out,
    int B,
    int T,
    int H,
    bool has_v_gate) {
  constexpr int N = 64;
  const int block = blockIdx.x;
  const int row = block % N;
  const int bh = block / N;
  const int head = bh % H;
  const int batch = bh / H;
  const int col = threadIdx.x;

  __shared__ float partial[N];
  __shared__ float v_row_s;

  const int state_base = (batch * H + head) * N * N;
  const int param_base = head * N;
  float st = state[state_base + row * N + col];

  for (int t = 0; t < T; ++t) {
    const int vec_base = ((batch * T + t) * H + head) * N;
    const float rv = static_cast<float>(r[vec_base + col]);
    const float wr = static_cast<float>(w_raw[vec_base + col]);
    const float kr = static_cast<float>(k_raw[vec_base + col]);
    const float av = static_cast<float>(a[vec_base + col]);
    const float kk_raw = kr * static_cast<float>(k_k[param_base + col]);
    const float wv = __expf(-0.606531f / (1.0f + __expf(-wr)));
    const float kv = kr * (1.0f + (av - 1.0f) * static_cast<float>(k_a[param_base + col]));

    if (threadIdx.x == 0) {
      const float vr = static_cast<float>(v_raw[vec_base + row]);
      if (has_v_gate) {
        const float vf = static_cast<float>(v_first[vec_base + row]);
        const float vg = static_cast<float>(v_gate[vec_base + row]);
        v_row_s = vr + (vf - vr) * vg;
      } else {
        v_row_s = vr;
      }
      v_out[vec_base + row] = static_cast<scalar_t>(v_row_s);
    }
    if (row == 0) {
      k_out[vec_base + col] = static_cast<scalar_t>(kv);
    }

    const float norm2 = rwkv7_block_reduce_sum_64(kk_raw * kk_raw, partial);
    const float norm_inv = rsqrtf(fmaxf(norm2, 1.0e-20f));
    const float kk = kk_raw * norm_inv;

    if constexpr (PHASE == 0) {
      const float prep_guard = rwkv7_block_reduce_sum_64(rv + wv + kv + av + kk, partial);
      if (threadIdx.x == 0) {
        out[vec_base + row] = static_cast<scalar_t>(prep_guard + v_row_s);
      }
    } else {
      const float state_dot_kk = rwkv7_block_reduce_sum_64(st * kk, partial);
      if constexpr (PHASE == 1) {
        if (threadIdx.x == 0) {
          out[vec_base + row] = static_cast<scalar_t>(state_dot_kk + 1.0e-7f * (rv + wv + kv + av));
        }
      } else {
        const float new_st = st * wv + v_row_s * kv - state_dot_kk * kk * av;
        st = new_st;
        if constexpr (PHASE == 2) {
          if (threadIdx.x == 0) {
            out[vec_base + row] = static_cast<scalar_t>(new_st + 1.0e-7f * state_dot_kk);
          }
        } else {
          const float recurrent = rwkv7_block_reduce_sum_64(new_st * rv, partial);
          if (threadIdx.x == 0) {
            out[vec_base + row] = static_cast<scalar_t>(recurrent);
          }
        }
      }
    }
    __syncthreads();
  }

  final_state[state_base + row * N + col] = st;
}

template <int ROWS_PER_BLOCK>
__device__ __forceinline__ float rwkv7_row_reduce_sum_64(float value, float* scratch, int row_lane, int col) {
  const unsigned mask = 0xffffffffu;
  const int lane = col & 31;
  const int local_warp = col >> 5;
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(mask, value, offset);
  }
  if (lane == 0) {
    scratch[row_lane * 2 + local_warp] = value;
  }
  __syncthreads();
  value = (local_warp == 0 && lane < 2) ? scratch[row_lane * 2 + lane] : 0.0f;
  if (local_warp == 0) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(mask, value, offset);
    }
  }
  if (local_warp == 0 && lane == 0) {
    scratch[row_lane * 2] = value;
  }
  __syncthreads();
  return scratch[row_lane * 2];
}

template <typename scalar_t, int ROWS_PER_BLOCK>
__global__ void rwkv7_state_scan_prep_n64_rowblock_coop_kernel(
    const scalar_t* __restrict__ r,
    const scalar_t* __restrict__ w_raw,
    const scalar_t* __restrict__ k_raw,
    const scalar_t* __restrict__ v_raw,
    const scalar_t* __restrict__ a,
    const float* __restrict__ state,
    const scalar_t* __restrict__ k_k,
    const scalar_t* __restrict__ k_a,
    const scalar_t* __restrict__ v_first,
    const scalar_t* __restrict__ v_gate,
    scalar_t* __restrict__ out,
    float* __restrict__ final_state,
    scalar_t* __restrict__ k_out,
    scalar_t* __restrict__ v_out,
    int B,
    int T,
    int H,
    bool has_v_gate) {
  constexpr int N = 64;
  constexpr int ROW_BLOCKS = N / ROWS_PER_BLOCK;
  const int block = blockIdx.x;
  const int row_block = block % ROW_BLOCKS;
  const int bh = block / ROW_BLOCKS;
  const int head = bh % H;
  const int batch = bh / H;
  const int row_lane = threadIdx.x >> 6;
  const int col = threadIdx.x & 63;
  const int row = row_block * ROWS_PER_BLOCK + row_lane;

  __shared__ float r_s[N];
  __shared__ float w_s[N];
  __shared__ float k_s[N];
  __shared__ float a_s[N];
  __shared__ float kk_s[N];
  __shared__ float v_row_s[ROWS_PER_BLOCK];
  __shared__ float reduce_s[ROWS_PER_BLOCK * 2];

  const int state_base = (batch * H + head) * N * N;
  const int param_base = head * N;
  float st = state[state_base + row * N + col];

  for (int t = 0; t < T; ++t) {
    const int vec_base = ((batch * T + t) * H + head) * N;
    float rv = 0.0f;
    float wr = 0.0f;
    float kr = 0.0f;
    float av = 0.0f;
    float kk_raw = 0.0f;
    if (row_lane == 0) {
      rv = static_cast<float>(r[vec_base + col]);
      wr = static_cast<float>(w_raw[vec_base + col]);
      kr = static_cast<float>(k_raw[vec_base + col]);
      av = static_cast<float>(a[vec_base + col]);
      kk_raw = kr * static_cast<float>(k_k[param_base + col]);
    }

    const float norm2 = rwkv7_row_reduce_sum_64<ROWS_PER_BLOCK>(
        row_lane == 0 ? kk_raw * kk_raw : 0.0f, reduce_s, row_lane, col);
    const float norm_inv = rsqrtf(fmaxf(norm2, 1.0e-20f));

    if (row_lane == 0) {
      const float kv = kr * (1.0f + (av - 1.0f) * static_cast<float>(k_a[param_base + col]));
      r_s[col] = rv;
      w_s[col] = __expf(-0.606531f / (1.0f + __expf(-wr)));
      k_s[col] = kv;
      a_s[col] = av;
      kk_s[col] = kk_raw * norm_inv;
      k_out[vec_base + col] = static_cast<scalar_t>(kv);
      if (col < ROWS_PER_BLOCK) {
        const int v_row = row_block * ROWS_PER_BLOCK + col;
        const float vr = static_cast<float>(v_raw[vec_base + v_row]);
        float vv = vr;
        if (has_v_gate) {
          const float vf = static_cast<float>(v_first[vec_base + v_row]);
          const float vg = static_cast<float>(v_gate[vec_base + v_row]);
          vv = vr + (vf - vr) * vg;
        }
        v_row_s[col] = vv;
        v_out[vec_base + v_row] = static_cast<scalar_t>(vv);
      }
    }
    __syncthreads();

    const float state_dot_kk = rwkv7_row_reduce_sum_64<ROWS_PER_BLOCK>(
        st * kk_s[col], reduce_s, row_lane, col);
    const float new_st = st * w_s[col] + v_row_s[row_lane] * k_s[col] - state_dot_kk * kk_s[col] * a_s[col];
    st = new_st;
    const float recurrent = rwkv7_row_reduce_sum_64<ROWS_PER_BLOCK>(
        new_st * r_s[col], reduce_s, row_lane, col);
    if (col == 0) {
      out[vec_base + row] = static_cast<scalar_t>(recurrent);
    }
  }

  final_state[state_base + row * N + col] = st;
}

__device__ __forceinline__ float rwkv7_warp_reduce_sum_broadcast(float value) {
  const unsigned mask = 0xffffffffu;
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(mask, value, offset);
  }
  return __shfl_sync(mask, value, 0);
}

__device__ __forceinline__ float rwkv7_half_warp_reduce_sum_broadcast(float value) {
  const int warp_lane = threadIdx.x & 31;
  const unsigned mask = (warp_lane < 16) ? 0x0000ffffu : 0xffff0000u;
  #pragma unroll
  for (int offset = 8; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(mask, value, offset, 16);
  }
  return __shfl_sync(mask, value, 0, 16);
}

template <typename scalar_t>
__global__ void rwkv7_state_scan_prep_n64_head_reg16_kernel(
    const scalar_t* __restrict__ r,
    const scalar_t* __restrict__ w_raw,
    const scalar_t* __restrict__ k_raw,
    const scalar_t* __restrict__ v_raw,
    const scalar_t* __restrict__ a,
    const float* __restrict__ state,
    const scalar_t* __restrict__ k_k,
    const scalar_t* __restrict__ k_a,
    const scalar_t* __restrict__ v_first,
    const scalar_t* __restrict__ v_gate,
    scalar_t* __restrict__ out,
    float* __restrict__ final_state,
    scalar_t* __restrict__ k_out,
    scalar_t* __restrict__ v_out,
    int B,
    int T,
    int H,
    bool has_v_gate) {
  constexpr int N = 64;
  constexpr int LANES_PER_ROW = 16;
  const int bh = blockIdx.x;
  const int head = bh % H;
  const int batch = bh / H;
  const int tid = threadIdx.x;
  const int row = tid >> 4;
  const int lane = tid & 15;
  const int col0 = lane;
  const int col1 = lane + 16;
  const int col2 = lane + 32;
  const int col3 = lane + 48;

  __shared__ float r_s[N];
  __shared__ float w_s[N];
  __shared__ float k_s[N];
  __shared__ float a_s[N];
  __shared__ float kk_s[N];
  __shared__ float v_s[N];
  __shared__ float norm_s;

  const int state_base = (batch * H + head) * N * N;
  const int param_base = head * N;
  float st0 = state[state_base + row * N + col0];
  float st1 = state[state_base + row * N + col1];
  float st2 = state[state_base + row * N + col2];
  float st3 = state[state_base + row * N + col3];

  for (int t = 0; t < T; ++t) {
    const int vec_base = ((batch * T + t) * H + head) * N;
    if (tid < N) {
      const float rv = static_cast<float>(r[vec_base + tid]);
      const float wr = static_cast<float>(w_raw[vec_base + tid]);
      const float kr = static_cast<float>(k_raw[vec_base + tid]);
      const float vr = static_cast<float>(v_raw[vec_base + tid]);
      const float av = static_cast<float>(a[vec_base + tid]);
      const float kk_raw = kr * static_cast<float>(k_k[param_base + tid]);
      r_s[tid] = rv;
      w_s[tid] = __expf(-0.606531f / (1.0f + __expf(-wr)));
      k_s[tid] = kr * (1.0f + (av - 1.0f) * static_cast<float>(k_a[param_base + tid]));
      a_s[tid] = av;
      kk_s[tid] = kk_raw;
      if (has_v_gate) {
        const float vf = static_cast<float>(v_first[vec_base + tid]);
        const float vg = static_cast<float>(v_gate[vec_base + tid]);
        v_s[tid] = vr + (vf - vr) * vg;
      } else {
        v_s[tid] = vr;
      }
    }
    __syncthreads();

    if (tid == 0) {
      float norm2 = 0.0f;
      #pragma unroll
      for (int j = 0; j < N; ++j) {
        norm2 += kk_s[j] * kk_s[j];
      }
      norm_s = rsqrtf(fmaxf(norm2, 1.0e-20f));
    }
    __syncthreads();

    if (tid < N) {
      kk_s[tid] *= norm_s;
      k_out[vec_base + tid] = static_cast<scalar_t>(k_s[tid]);
      v_out[vec_base + tid] = static_cast<scalar_t>(v_s[tid]);
    }
    __syncthreads();

    const float kk0 = kk_s[col0];
    const float kk1 = kk_s[col1];
    const float kk2 = kk_s[col2];
    const float kk3 = kk_s[col3];
    const float state_dot_partial = st0 * kk0 + st1 * kk1 + st2 * kk2 + st3 * kk3;
    const float state_dot_kk = rwkv7_half_warp_reduce_sum_broadcast(state_dot_partial);
    const float v_row = v_s[row];

    const float new_st0 = st0 * w_s[col0] + v_row * k_s[col0] - state_dot_kk * kk0 * a_s[col0];
    const float new_st1 = st1 * w_s[col1] + v_row * k_s[col1] - state_dot_kk * kk1 * a_s[col1];
    const float new_st2 = st2 * w_s[col2] + v_row * k_s[col2] - state_dot_kk * kk2 * a_s[col2];
    const float new_st3 = st3 * w_s[col3] + v_row * k_s[col3] - state_dot_kk * kk3 * a_s[col3];
    st0 = new_st0;
    st1 = new_st1;
    st2 = new_st2;
    st3 = new_st3;

    const float recurrent_partial =
        new_st0 * r_s[col0] + new_st1 * r_s[col1] + new_st2 * r_s[col2] + new_st3 * r_s[col3];
    const float recurrent = rwkv7_half_warp_reduce_sum_broadcast(recurrent_partial);
    if (lane == 0) {
      out[vec_base + row] = static_cast<scalar_t>(recurrent);
    }
    __syncthreads();
  }

  final_state[state_base + row * N + col0] = st0;
  final_state[state_base + row * N + col1] = st1;
  final_state[state_base + row * N + col2] = st2;
  final_state[state_base + row * N + col3] = st3;
}

template <typename scalar_t, int ROWS_PER_BLOCK, bool W_PRECOMPUTED=false>
__global__ void rwkv7_state_scan_prep_n64_rowblock_warp_specialized_kernel(
    const scalar_t* __restrict__ r,
    const scalar_t* __restrict__ w_raw,
    const scalar_t* __restrict__ k_raw,
    const scalar_t* __restrict__ v_raw,
    const scalar_t* __restrict__ a,
    const float* __restrict__ state,
    const scalar_t* __restrict__ k_k,
    const scalar_t* __restrict__ k_a,
    const scalar_t* __restrict__ v_first,
    const scalar_t* __restrict__ v_gate,
    scalar_t* __restrict__ out,
    float* __restrict__ final_state,
    scalar_t* __restrict__ k_out,
    scalar_t* __restrict__ v_out,
    int B,
    int T,
    int H,
    bool has_v_gate) {
  constexpr int N = 64;
  constexpr int ROW_BLOCKS = N / ROWS_PER_BLOCK;
  const int block = blockIdx.x;
  const int row_block = block % ROW_BLOCKS;
  const int bh = block / ROW_BLOCKS;
  const int head = bh % H;
  const int batch = bh / H;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const bool is_producer = warp == 0;
  const int row_lane = warp - 1;
  const int row = row_block * ROWS_PER_BLOCK + row_lane;
  const int col0 = lane;
  const int col1 = lane + 32;

  __shared__ float r_s[N];
  __shared__ float w_s[N];
  __shared__ float k_s[N];
  __shared__ float a_s[N];
  __shared__ float kk_s[N];
  __shared__ float v_row_s[ROWS_PER_BLOCK];

  const int state_base = (batch * H + head) * N * N;
  const int param_base = head * N;
  float st0 = 0.0f;
  float st1 = 0.0f;
  if (!is_producer) {
    st0 = state[state_base + row * N + col0];
    st1 = state[state_base + row * N + col1];
  }

  for (int t = 0; t < T; ++t) {
    const int vec_base = ((batch * T + t) * H + head) * N;
    if (is_producer) {
      const float kr0 = static_cast<float>(k_raw[vec_base + col0]);
      const float kr1 = static_cast<float>(k_raw[vec_base + col1]);
      const float av0 = static_cast<float>(a[vec_base + col0]);
      const float av1 = static_cast<float>(a[vec_base + col1]);
      const float kk_raw0 = kr0 * static_cast<float>(k_k[param_base + col0]);
      const float kk_raw1 = kr1 * static_cast<float>(k_k[param_base + col1]);
      const float norm2 = rwkv7_warp_reduce_sum_broadcast(kk_raw0 * kk_raw0 + kk_raw1 * kk_raw1);
      const float norm_inv = rsqrtf(fmaxf(norm2, 1.0e-20f));
      const float kv0 = kr0 * (1.0f + (av0 - 1.0f) * static_cast<float>(k_a[param_base + col0]));
      const float kv1 = kr1 * (1.0f + (av1 - 1.0f) * static_cast<float>(k_a[param_base + col1]));

      r_s[col0] = static_cast<float>(r[vec_base + col0]);
      r_s[col1] = static_cast<float>(r[vec_base + col1]);
      if constexpr (W_PRECOMPUTED) {
        w_s[col0] = static_cast<float>(w_raw[vec_base + col0]);
        w_s[col1] = static_cast<float>(w_raw[vec_base + col1]);
      } else {
        w_s[col0] = __expf(-0.606531f / (1.0f + __expf(-static_cast<float>(w_raw[vec_base + col0]))));
        w_s[col1] = __expf(-0.606531f / (1.0f + __expf(-static_cast<float>(w_raw[vec_base + col1]))));
      }
      k_s[col0] = kv0;
      k_s[col1] = kv1;
      a_s[col0] = av0;
      a_s[col1] = av1;
      kk_s[col0] = kk_raw0 * norm_inv;
      kk_s[col1] = kk_raw1 * norm_inv;

      if (row_block == 0) {
        k_out[vec_base + col0] = static_cast<scalar_t>(kv0);
        k_out[vec_base + col1] = static_cast<scalar_t>(kv1);
      }
      if (lane < ROWS_PER_BLOCK) {
        const int v_row = row_block * ROWS_PER_BLOCK + lane;
        const float vr = static_cast<float>(v_raw[vec_base + v_row]);
        float vv = vr;
        if (has_v_gate) {
          const float vf = static_cast<float>(v_first[vec_base + v_row]);
          const float vg = static_cast<float>(v_gate[vec_base + v_row]);
          vv = vr + (vf - vr) * vg;
        }
        v_row_s[lane] = vv;
        v_out[vec_base + v_row] = static_cast<scalar_t>(vv);
      }
    }
    __syncthreads();

    if (!is_producer) {
      const float state_dot_partial = st0 * kk_s[col0] + st1 * kk_s[col1];
      const float state_dot_kk = rwkv7_warp_reduce_sum_broadcast(state_dot_partial);
      const float v_row = v_row_s[row_lane];
      const float new_st0 = st0 * w_s[col0] + v_row * k_s[col0] - state_dot_kk * kk_s[col0] * a_s[col0];
      const float new_st1 = st1 * w_s[col1] + v_row * k_s[col1] - state_dot_kk * kk_s[col1] * a_s[col1];
      st0 = new_st0;
      st1 = new_st1;
      const float recurrent = rwkv7_warp_reduce_sum_broadcast(new_st0 * r_s[col0] + new_st1 * r_s[col1]);
      if (lane == 0) {
        out[vec_base + row] = static_cast<scalar_t>(recurrent);
      }
    }
    __syncthreads();
  }

  if (!is_producer) {
    final_state[state_base + row * N + col0] = st0;
    final_state[state_base + row * N + col1] = st1;
  }
}

template <typename scalar_t, int ROWS_PER_BLOCK, bool W_PRECOMPUTED=false>
__global__ void rwkv7_state_scan_prep_n64_rowblock_warp_pipelined_kernel(
    const scalar_t* __restrict__ r,
    const scalar_t* __restrict__ w_raw,
    const scalar_t* __restrict__ k_raw,
    const scalar_t* __restrict__ v_raw,
    const scalar_t* __restrict__ a,
    const float* __restrict__ state,
    const scalar_t* __restrict__ k_k,
    const scalar_t* __restrict__ k_a,
    const scalar_t* __restrict__ v_first,
    const scalar_t* __restrict__ v_gate,
    scalar_t* __restrict__ out,
    float* __restrict__ final_state,
    scalar_t* __restrict__ k_out,
    scalar_t* __restrict__ v_out,
    int B,
    int T,
    int H,
    bool has_v_gate) {
  constexpr int N = 64;
  constexpr int ROW_BLOCKS = N / ROWS_PER_BLOCK;
  const int block = blockIdx.x;
  const int row_block = block % ROW_BLOCKS;
  const int bh = block / ROW_BLOCKS;
  const int head = bh % H;
  const int batch = bh / H;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const bool is_producer = warp == 0;
  const int row_lane = warp - 1;
  const int row = row_block * ROWS_PER_BLOCK + row_lane;
  const int col0 = lane;
  const int col1 = lane + 32;

  __shared__ float r_s[2][N];
  __shared__ float w_s[2][N];
  __shared__ float k_s[2][N];
  __shared__ float a_s[2][N];
  __shared__ float kk_s[2][N];
  __shared__ float v_row_s[2][ROWS_PER_BLOCK];

  const int state_base = (batch * H + head) * N * N;
  const int param_base = head * N;
  float st0 = 0.0f;
  float st1 = 0.0f;
  if (!is_producer) {
    st0 = state[state_base + row * N + col0];
    st1 = state[state_base + row * N + col1];
  }

  if (T <= 0) {
    if (!is_producer) {
      final_state[state_base + row * N + col0] = st0;
      final_state[state_base + row * N + col1] = st1;
    }
    return;
  }

  if (is_producer) {
    const int vec_base = (batch * T * H + head) * N;
    const float kr0 = static_cast<float>(k_raw[vec_base + col0]);
    const float kr1 = static_cast<float>(k_raw[vec_base + col1]);
    const float av0 = static_cast<float>(a[vec_base + col0]);
    const float av1 = static_cast<float>(a[vec_base + col1]);
    const float kk_raw0 = kr0 * static_cast<float>(k_k[param_base + col0]);
    const float kk_raw1 = kr1 * static_cast<float>(k_k[param_base + col1]);
    const float norm2 = rwkv7_warp_reduce_sum_broadcast(kk_raw0 * kk_raw0 + kk_raw1 * kk_raw1);
    const float norm_inv = rsqrtf(fmaxf(norm2, 1.0e-20f));
    const float kv0 = kr0 * (1.0f + (av0 - 1.0f) * static_cast<float>(k_a[param_base + col0]));
    const float kv1 = kr1 * (1.0f + (av1 - 1.0f) * static_cast<float>(k_a[param_base + col1]));

    r_s[0][col0] = static_cast<float>(r[vec_base + col0]);
    r_s[0][col1] = static_cast<float>(r[vec_base + col1]);
    if constexpr (W_PRECOMPUTED) {
      w_s[0][col0] = static_cast<float>(w_raw[vec_base + col0]);
      w_s[0][col1] = static_cast<float>(w_raw[vec_base + col1]);
    } else {
      w_s[0][col0] = __expf(-0.606531f / (1.0f + __expf(-static_cast<float>(w_raw[vec_base + col0]))));
      w_s[0][col1] = __expf(-0.606531f / (1.0f + __expf(-static_cast<float>(w_raw[vec_base + col1]))));
    }
    k_s[0][col0] = kv0;
    k_s[0][col1] = kv1;
    a_s[0][col0] = av0;
    a_s[0][col1] = av1;
    kk_s[0][col0] = kk_raw0 * norm_inv;
    kk_s[0][col1] = kk_raw1 * norm_inv;

    if (row_block == 0) {
      k_out[vec_base + col0] = static_cast<scalar_t>(kv0);
      k_out[vec_base + col1] = static_cast<scalar_t>(kv1);
    }
    if (lane < ROWS_PER_BLOCK) {
      const int v_row = row_block * ROWS_PER_BLOCK + lane;
      const float vr = static_cast<float>(v_raw[vec_base + v_row]);
      float vv = vr;
      if (has_v_gate) {
        const float vf = static_cast<float>(v_first[vec_base + v_row]);
        const float vg = static_cast<float>(v_gate[vec_base + v_row]);
        vv = vr + (vf - vr) * vg;
      }
      v_row_s[0][lane] = vv;
      v_out[vec_base + v_row] = static_cast<scalar_t>(vv);
    }
  }
  __syncthreads();

  for (int t = 0; t < T; ++t) {
    const int cur = t & 1;
    const int next = cur ^ 1;
    const int vec_base = ((batch * T + t) * H + head) * N;
    if (is_producer && t + 1 < T) {
      const int next_vec_base = ((batch * T + t + 1) * H + head) * N;
      const float kr0 = static_cast<float>(k_raw[next_vec_base + col0]);
      const float kr1 = static_cast<float>(k_raw[next_vec_base + col1]);
      const float av0 = static_cast<float>(a[next_vec_base + col0]);
      const float av1 = static_cast<float>(a[next_vec_base + col1]);
      const float kk_raw0 = kr0 * static_cast<float>(k_k[param_base + col0]);
      const float kk_raw1 = kr1 * static_cast<float>(k_k[param_base + col1]);
      const float norm2 = rwkv7_warp_reduce_sum_broadcast(kk_raw0 * kk_raw0 + kk_raw1 * kk_raw1);
      const float norm_inv = rsqrtf(fmaxf(norm2, 1.0e-20f));
      const float kv0 = kr0 * (1.0f + (av0 - 1.0f) * static_cast<float>(k_a[param_base + col0]));
      const float kv1 = kr1 * (1.0f + (av1 - 1.0f) * static_cast<float>(k_a[param_base + col1]));

      r_s[next][col0] = static_cast<float>(r[next_vec_base + col0]);
      r_s[next][col1] = static_cast<float>(r[next_vec_base + col1]);
      if constexpr (W_PRECOMPUTED) {
        w_s[next][col0] = static_cast<float>(w_raw[next_vec_base + col0]);
        w_s[next][col1] = static_cast<float>(w_raw[next_vec_base + col1]);
      } else {
        w_s[next][col0] = __expf(-0.606531f / (1.0f + __expf(-static_cast<float>(w_raw[next_vec_base + col0]))));
        w_s[next][col1] = __expf(-0.606531f / (1.0f + __expf(-static_cast<float>(w_raw[next_vec_base + col1]))));
      }
      k_s[next][col0] = kv0;
      k_s[next][col1] = kv1;
      a_s[next][col0] = av0;
      a_s[next][col1] = av1;
      kk_s[next][col0] = kk_raw0 * norm_inv;
      kk_s[next][col1] = kk_raw1 * norm_inv;

      if (row_block == 0) {
        k_out[next_vec_base + col0] = static_cast<scalar_t>(kv0);
        k_out[next_vec_base + col1] = static_cast<scalar_t>(kv1);
      }
      if (lane < ROWS_PER_BLOCK) {
        const int v_row = row_block * ROWS_PER_BLOCK + lane;
        const float vr = static_cast<float>(v_raw[next_vec_base + v_row]);
        float vv = vr;
        if (has_v_gate) {
          const float vf = static_cast<float>(v_first[next_vec_base + v_row]);
          const float vg = static_cast<float>(v_gate[next_vec_base + v_row]);
          vv = vr + (vf - vr) * vg;
        }
        v_row_s[next][lane] = vv;
        v_out[next_vec_base + v_row] = static_cast<scalar_t>(vv);
      }
    }

    if (!is_producer) {
      const float state_dot_partial = st0 * kk_s[cur][col0] + st1 * kk_s[cur][col1];
      const float state_dot_kk = rwkv7_warp_reduce_sum_broadcast(state_dot_partial);
      const float v_row = v_row_s[cur][row_lane];
      const float new_st0 = st0 * w_s[cur][col0] + v_row * k_s[cur][col0] - state_dot_kk * kk_s[cur][col0] * a_s[cur][col0];
      const float new_st1 = st1 * w_s[cur][col1] + v_row * k_s[cur][col1] - state_dot_kk * kk_s[cur][col1] * a_s[cur][col1];
      st0 = new_st0;
      st1 = new_st1;
      const float recurrent = rwkv7_warp_reduce_sum_broadcast(new_st0 * r_s[cur][col0] + new_st1 * r_s[cur][col1]);
      if (lane == 0) {
        out[vec_base + row] = static_cast<scalar_t>(recurrent);
      }
    }

    if (t + 1 < T) {
      __syncthreads();
    }
  }

  if (!is_producer) {
    final_state[state_base + row * N + col0] = st0;
    final_state[state_base + row * N + col1] = st1;
  }
}

template <typename scalar_t, int ROWS_PER_BLOCK>
__global__ void rwkv7_state_scan_prep_n64_rowblock_warp_specialized_sk_kernel(
    const scalar_t* __restrict__ r,
    const scalar_t* __restrict__ w_raw,
    const scalar_t* __restrict__ k_raw,
    const scalar_t* __restrict__ v_raw,
    const scalar_t* __restrict__ a,
    const float* __restrict__ state,
    const scalar_t* __restrict__ k_k,
    const scalar_t* __restrict__ k_a,
    const scalar_t* __restrict__ r_k,
    const scalar_t* __restrict__ v_first,
    const scalar_t* __restrict__ v_gate,
    scalar_t* __restrict__ out,
    float* __restrict__ final_state,
    scalar_t* __restrict__ sk_out,
    int B,
    int T,
    int H,
    bool has_v_gate) {
  constexpr int N = 64;
  constexpr int ROW_BLOCKS = N / ROWS_PER_BLOCK;
  const int block = blockIdx.x;
  const int row_block = block % ROW_BLOCKS;
  const int bh = block / ROW_BLOCKS;
  const int head = bh % H;
  const int batch = bh / H;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const bool is_producer = warp == 0;
  const int row_lane = warp - 1;
  const int row = row_block * ROWS_PER_BLOCK + row_lane;
  const int col0 = lane;
  const int col1 = lane + 32;

  __shared__ float r_s[N];
  __shared__ float w_s[N];
  __shared__ float k_s[N];
  __shared__ float a_s[N];
  __shared__ float kk_s[N];
  __shared__ float v_row_s[ROWS_PER_BLOCK];

  const int state_base = (batch * H + head) * N * N;
  const int param_base = head * N;
  float st0 = 0.0f;
  float st1 = 0.0f;
  if (!is_producer) {
    st0 = state[state_base + row * N + col0];
    st1 = state[state_base + row * N + col1];
  }

  for (int t = 0; t < T; ++t) {
    const int vec_base = ((batch * T + t) * H + head) * N;
    if (is_producer) {
      const float rv0 = static_cast<float>(r[vec_base + col0]);
      const float rv1 = static_cast<float>(r[vec_base + col1]);
      const float kr0 = static_cast<float>(k_raw[vec_base + col0]);
      const float kr1 = static_cast<float>(k_raw[vec_base + col1]);
      const float av0 = static_cast<float>(a[vec_base + col0]);
      const float av1 = static_cast<float>(a[vec_base + col1]);
      const float kk_raw0 = kr0 * static_cast<float>(k_k[param_base + col0]);
      const float kk_raw1 = kr1 * static_cast<float>(k_k[param_base + col1]);
      const float norm2 = rwkv7_warp_reduce_sum_broadcast(kk_raw0 * kk_raw0 + kk_raw1 * kk_raw1);
      const float norm_inv = rsqrtf(fmaxf(norm2, 1.0e-20f));
      const float kv0 = kr0 * (1.0f + (av0 - 1.0f) * static_cast<float>(k_a[param_base + col0]));
      const float kv1 = kr1 * (1.0f + (av1 - 1.0f) * static_cast<float>(k_a[param_base + col1]));

      r_s[col0] = rv0;
      r_s[col1] = rv1;
      w_s[col0] = __expf(-0.606531f / (1.0f + __expf(-static_cast<float>(w_raw[vec_base + col0]))));
      w_s[col1] = __expf(-0.606531f / (1.0f + __expf(-static_cast<float>(w_raw[vec_base + col1]))));
      k_s[col0] = kv0;
      k_s[col1] = kv1;
      a_s[col0] = av0;
      a_s[col1] = av1;
      kk_s[col0] = kk_raw0 * norm_inv;
      kk_s[col1] = kk_raw1 * norm_inv;

      if (row_block == 0) {
        const float sk_part =
            rv0 * kv0 * static_cast<float>(r_k[param_base + col0])
            + rv1 * kv1 * static_cast<float>(r_k[param_base + col1]);
        const float sk_val = rwkv7_warp_reduce_sum_broadcast(sk_part);
        if (lane == 0) {
          sk_out[(batch * T + t) * H + head] = static_cast<scalar_t>(sk_val);
        }
      }
      if (lane < ROWS_PER_BLOCK) {
        const int v_row = row_block * ROWS_PER_BLOCK + lane;
        const float vr = static_cast<float>(v_raw[vec_base + v_row]);
        float vv = vr;
        if (has_v_gate) {
          const float vf = static_cast<float>(v_first[vec_base + v_row]);
          const float vg = static_cast<float>(v_gate[vec_base + v_row]);
          vv = vr + (vf - vr) * vg;
        }
        v_row_s[lane] = vv;
      }
    }
    __syncthreads();

    if (!is_producer) {
      const float state_dot_partial = st0 * kk_s[col0] + st1 * kk_s[col1];
      const float state_dot_kk = rwkv7_warp_reduce_sum_broadcast(state_dot_partial);
      const float v_row = v_row_s[row_lane];
      const float new_st0 = st0 * w_s[col0] + v_row * k_s[col0] - state_dot_kk * kk_s[col0] * a_s[col0];
      const float new_st1 = st1 * w_s[col1] + v_row * k_s[col1] - state_dot_kk * kk_s[col1] * a_s[col1];
      st0 = new_st0;
      st1 = new_st1;
      const float recurrent = rwkv7_warp_reduce_sum_broadcast(new_st0 * r_s[col0] + new_st1 * r_s[col1]);
      if (lane == 0) {
        out[vec_base + row] = static_cast<scalar_t>(recurrent);
      }
    }
    __syncthreads();
  }

  if (!is_producer) {
    final_state[state_base + row * N + col0] = st0;
    final_state[state_base + row * N + col1] = st1;
  }
}

template <typename scalar_t, int ROWS_PER_BLOCK>
__global__ void rwkv7_state_scan_prep_n64_rowblock_warp2_kernel(
    const scalar_t* __restrict__ r,
    const scalar_t* __restrict__ w_raw,
    const scalar_t* __restrict__ k_raw,
    const scalar_t* __restrict__ v_raw,
    const scalar_t* __restrict__ a,
    const float* __restrict__ state,
    const scalar_t* __restrict__ k_k,
    const scalar_t* __restrict__ k_a,
    const scalar_t* __restrict__ v_first,
    const scalar_t* __restrict__ v_gate,
    scalar_t* __restrict__ out,
    float* __restrict__ final_state,
    scalar_t* __restrict__ k_out,
    scalar_t* __restrict__ v_out,
    int B,
    int T,
    int H,
    bool has_v_gate) {
  constexpr int N = 64;
  constexpr int ROW_BLOCKS = N / ROWS_PER_BLOCK;
  const int block = blockIdx.x;
  const int row_block = block % ROW_BLOCKS;
  const int bh = block / ROW_BLOCKS;
  const int head = bh % H;
  const int batch = bh / H;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const bool is_producer = warp == 0;
  const int worker_warp = warp - 1;
  const int row_lane = worker_warp >> 1;
  const int local_warp = worker_warp & 1;
  const int row = row_block * ROWS_PER_BLOCK + row_lane;
  const int col = local_warp * 32 + lane;

  __shared__ float r_s[N];
  __shared__ float w_s[N];
  __shared__ float k_s[N];
  __shared__ float a_s[N];
  __shared__ float kk_s[N];
  __shared__ float v_row_s[ROWS_PER_BLOCK];
  __shared__ float reduce_s[ROWS_PER_BLOCK * 2];

  const int state_base = (batch * H + head) * N * N;
  const int param_base = head * N;
  float st = 0.0f;
  if (!is_producer) {
    st = state[state_base + row * N + col];
  }

  for (int t = 0; t < T; ++t) {
    const int vec_base = ((batch * T + t) * H + head) * N;
    if (is_producer) {
      const int col0 = lane;
      const int col1 = lane + 32;
      const float kr0 = static_cast<float>(k_raw[vec_base + col0]);
      const float kr1 = static_cast<float>(k_raw[vec_base + col1]);
      const float av0 = static_cast<float>(a[vec_base + col0]);
      const float av1 = static_cast<float>(a[vec_base + col1]);
      const float kk_raw0 = kr0 * static_cast<float>(k_k[param_base + col0]);
      const float kk_raw1 = kr1 * static_cast<float>(k_k[param_base + col1]);
      const float norm2 = rwkv7_warp_reduce_sum_broadcast(kk_raw0 * kk_raw0 + kk_raw1 * kk_raw1);
      const float norm_inv = rsqrtf(fmaxf(norm2, 1.0e-20f));
      const float kv0 = kr0 * (1.0f + (av0 - 1.0f) * static_cast<float>(k_a[param_base + col0]));
      const float kv1 = kr1 * (1.0f + (av1 - 1.0f) * static_cast<float>(k_a[param_base + col1]));

      r_s[col0] = static_cast<float>(r[vec_base + col0]);
      r_s[col1] = static_cast<float>(r[vec_base + col1]);
      w_s[col0] = __expf(-0.606531f / (1.0f + __expf(-static_cast<float>(w_raw[vec_base + col0]))));
      w_s[col1] = __expf(-0.606531f / (1.0f + __expf(-static_cast<float>(w_raw[vec_base + col1]))));
      k_s[col0] = kv0;
      k_s[col1] = kv1;
      a_s[col0] = av0;
      a_s[col1] = av1;
      kk_s[col0] = kk_raw0 * norm_inv;
      kk_s[col1] = kk_raw1 * norm_inv;

      if (row_block == 0) {
        k_out[vec_base + col0] = static_cast<scalar_t>(kv0);
        k_out[vec_base + col1] = static_cast<scalar_t>(kv1);
      }
      if (lane < ROWS_PER_BLOCK) {
        const int v_row = row_block * ROWS_PER_BLOCK + lane;
        const float vr = static_cast<float>(v_raw[vec_base + v_row]);
        float vv = vr;
        if (has_v_gate) {
          const float vf = static_cast<float>(v_first[vec_base + v_row]);
          const float vg = static_cast<float>(v_gate[vec_base + v_row]);
          vv = vr + (vf - vr) * vg;
        }
        v_row_s[lane] = vv;
        v_out[vec_base + v_row] = static_cast<scalar_t>(vv);
      }
    }
    __syncthreads();

    float state_dot_kk = 0.0f;
    if (!is_producer) {
      float value = st * kk_s[col];
      #pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
      }
      if (lane == 0) {
        reduce_s[row_lane * 2 + local_warp] = value;
      }
    }
    __syncthreads();

    if (!is_producer) {
      state_dot_kk = reduce_s[row_lane * 2] + reduce_s[row_lane * 2 + 1];
      const float v_row = v_row_s[row_lane];
      const float new_st = st * w_s[col] + v_row * k_s[col] - state_dot_kk * kk_s[col] * a_s[col];
      st = new_st;
      float value = new_st * r_s[col];
      #pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
      }
      if (lane == 0) {
        reduce_s[row_lane * 2 + local_warp] = value;
      }
    }
    __syncthreads();

    if (!is_producer && local_warp == 0 && lane == 0) {
      const float recurrent = reduce_s[row_lane * 2] + reduce_s[row_lane * 2 + 1];
      out[vec_base + row] = static_cast<scalar_t>(recurrent);
    }
    __syncthreads();
  }

  if (!is_producer) {
    final_state[state_base + row * N + col] = st;
  }
}

template <typename scalar_t>
__global__ void rwkv7_state_scan_prep_n64_vector_precompute_kernel(
    const scalar_t* __restrict__ w_raw,
    const scalar_t* __restrict__ k_raw,
    const scalar_t* __restrict__ v_raw,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ k_k,
    const scalar_t* __restrict__ k_a,
    const scalar_t* __restrict__ v_first,
    const scalar_t* __restrict__ v_gate,
    float* __restrict__ w_prep,
    float* __restrict__ kk_prep,
    float* __restrict__ k_prep,
    float* __restrict__ v_prep,
    scalar_t* __restrict__ k_out,
    scalar_t* __restrict__ v_out,
    int B,
    int T,
    int H,
    bool has_v_gate) {
  constexpr int N = 64;
  const int bth = blockIdx.x;
  const int head = bth % H;
  const int tmp = bth / H;
  const int t = tmp % T;
  const int batch = tmp / T;
  const int col = threadIdx.x;

  __shared__ float partial[N];

  const int vec_base = ((batch * T + t) * H + head) * N;
  const int param_base = head * N;
  const float wr = static_cast<float>(w_raw[vec_base + col]);
  const float kr = static_cast<float>(k_raw[vec_base + col]);
  const float vr = static_cast<float>(v_raw[vec_base + col]);
  const float av = static_cast<float>(a[vec_base + col]);
  const float kk_raw = kr * static_cast<float>(k_k[param_base + col]);

  const float norm2 = rwkv7_block_reduce_sum_64(kk_raw * kk_raw, partial);
  const float norm_inv = rsqrtf(fmaxf(norm2, 1.0e-20f));
  w_prep[vec_base + col] = __expf(-0.606531f / (1.0f + __expf(-wr)));
  kk_prep[vec_base + col] = kk_raw * norm_inv;
  const float kv = kr * (1.0f + (av - 1.0f) * static_cast<float>(k_a[param_base + col]));
  k_prep[vec_base + col] = kv;
  k_out[vec_base + col] = static_cast<scalar_t>(kv);
  float vv = vr;
  if (has_v_gate) {
    const float vf = static_cast<float>(v_first[vec_base + col]);
    const float vg = static_cast<float>(v_gate[vec_base + col]);
    vv = vr + (vf - vr) * vg;
  }
  v_prep[vec_base + col] = vv;
  v_out[vec_base + col] = static_cast<scalar_t>(vv);
}

template <typename scalar_t>
__global__ void rwkv7_state_scan_prep_n64_vector_precompute_wk_kernel(
    const scalar_t* __restrict__ w_raw,
    const scalar_t* __restrict__ k_raw,
    const scalar_t* __restrict__ v_raw,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ k_k,
    const scalar_t* __restrict__ k_a,
    const scalar_t* __restrict__ v_first,
    const scalar_t* __restrict__ v_gate,
    float* __restrict__ w_prep,
    float* __restrict__ kk_prep,
    scalar_t* __restrict__ k_out,
    scalar_t* __restrict__ v_out,
    int B,
    int T,
    int H,
    bool has_v_gate) {
  constexpr int N = 64;
  const int bth = blockIdx.x;
  const int head = bth % H;
  const int tmp = bth / H;
  const int t = tmp % T;
  const int batch = tmp / T;
  const int col = threadIdx.x;

  __shared__ float partial[N];

  const int vec_base = ((batch * T + t) * H + head) * N;
  const int param_base = head * N;
  const float wr = static_cast<float>(w_raw[vec_base + col]);
  const float kr = static_cast<float>(k_raw[vec_base + col]);
  const float vr = static_cast<float>(v_raw[vec_base + col]);
  const float av = static_cast<float>(a[vec_base + col]);
  const float kk_raw = kr * static_cast<float>(k_k[param_base + col]);

  const float norm2 = rwkv7_block_reduce_sum_64(kk_raw * kk_raw, partial);
  const float norm_inv = rsqrtf(fmaxf(norm2, 1.0e-20f));
  w_prep[vec_base + col] = __expf(-0.606531f / (1.0f + __expf(-wr)));
  kk_prep[vec_base + col] = kk_raw * norm_inv;
  k_out[vec_base + col] = static_cast<scalar_t>(
      kr * (1.0f + (av - 1.0f) * static_cast<float>(k_a[param_base + col])));
  if (has_v_gate) {
    const float vf = static_cast<float>(v_first[vec_base + col]);
    const float vg = static_cast<float>(v_gate[vec_base + col]);
    v_out[vec_base + col] = static_cast<scalar_t>(vr + (vf - vr) * vg);
  } else {
    v_out[vec_base + col] = static_cast<scalar_t>(vr);
  }
}

template <typename scalar_t>
__global__ void rwkv7_state_scan_prep_n64_vector_precompute_wk_half_kernel(
    const scalar_t* __restrict__ w_raw,
    const scalar_t* __restrict__ k_raw,
    const scalar_t* __restrict__ v_raw,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ k_k,
    const scalar_t* __restrict__ k_a,
    const scalar_t* __restrict__ v_first,
    const scalar_t* __restrict__ v_gate,
    scalar_t* __restrict__ w_prep,
    scalar_t* __restrict__ kk_prep,
    scalar_t* __restrict__ k_out,
    scalar_t* __restrict__ v_out,
    int B,
    int T,
    int H,
    bool has_v_gate) {
  constexpr int N = 64;
  const int bth = blockIdx.x;
  const int head = bth % H;
  const int tmp = bth / H;
  const int t = tmp % T;
  const int batch = tmp / T;
  const int col = threadIdx.x;

  __shared__ float partial[N];

  const int vec_base = ((batch * T + t) * H + head) * N;
  const int param_base = head * N;
  const float wr = static_cast<float>(w_raw[vec_base + col]);
  const float kr = static_cast<float>(k_raw[vec_base + col]);
  const float vr = static_cast<float>(v_raw[vec_base + col]);
  const float av = static_cast<float>(a[vec_base + col]);
  const float kk_raw = kr * static_cast<float>(k_k[param_base + col]);

  const float norm2 = rwkv7_block_reduce_sum_64(kk_raw * kk_raw, partial);
  const float norm_inv = rsqrtf(fmaxf(norm2, 1.0e-20f));
  w_prep[vec_base + col] = static_cast<scalar_t>(__expf(-0.606531f / (1.0f + __expf(-wr))));
  kk_prep[vec_base + col] = static_cast<scalar_t>(kk_raw * norm_inv);
  k_out[vec_base + col] = static_cast<scalar_t>(
      kr * (1.0f + (av - 1.0f) * static_cast<float>(k_a[param_base + col])));
  if (has_v_gate) {
    const float vf = static_cast<float>(v_first[vec_base + col]);
    const float vg = static_cast<float>(v_gate[vec_base + col]);
    v_out[vec_base + col] = static_cast<scalar_t>(vr + (vf - vr) * vg);
  } else {
    v_out[vec_base + col] = static_cast<scalar_t>(vr);
  }
}

template <typename scalar_t>
__global__ void rwkv7_state_scan_prep_n64_rowblock_precomputed_kernel(
    const scalar_t* __restrict__ r,
    const float* __restrict__ w_prep,
    const float* __restrict__ k_prep,
    const float* __restrict__ v_prep,
    const scalar_t* __restrict__ a,
    const float* __restrict__ kk_prep,
    const float* __restrict__ state,
    scalar_t* __restrict__ out,
    float* __restrict__ final_state,
    int B,
    int T,
    int H) {
  constexpr int N = 64;
  const int block = blockIdx.x;
  const int row = block % N;
  const int bh = block / N;
  const int head = bh % H;
  const int batch = bh / H;
  const int col = threadIdx.x;

  __shared__ float partial[N];

  const int state_base = (batch * H + head) * N * N;
  float st = state[state_base + row * N + col];

  for (int t = 0; t < T; ++t) {
    const int vec_base = ((batch * T + t) * H + head) * N;
    const float rv = static_cast<float>(r[vec_base + col]);
    const float wv = w_prep[vec_base + col];
    const float kv = k_prep[vec_base + col];
    const float av = static_cast<float>(a[vec_base + col]);
    const float kk = kk_prep[vec_base + col];
    const float v_row = v_prep[vec_base + row];

    const float state_dot_kk = rwkv7_block_reduce_sum_64(st * kk, partial);
    const float new_st = st * wv + v_row * kv - state_dot_kk * kk * av;
    st = new_st;
    const float recurrent = rwkv7_block_reduce_sum_64(new_st * rv, partial);
    if (threadIdx.x == 0) {
      out[vec_base + row] = static_cast<scalar_t>(recurrent);
    }
  }

  final_state[state_base + row * N + col] = st;
}

template <typename scalar_t>
__global__ void rwkv7_state_scan_prep_n64_rowblock_precomputed_wk_kernel(
    const scalar_t* __restrict__ r,
    const float* __restrict__ w_prep,
    const scalar_t* __restrict__ k_prep,
    const scalar_t* __restrict__ v_prep,
    const scalar_t* __restrict__ a,
    const float* __restrict__ kk_prep,
    const float* __restrict__ state,
    scalar_t* __restrict__ out,
    float* __restrict__ final_state,
    int B,
    int T,
    int H) {
  constexpr int N = 64;
  const int block = blockIdx.x;
  const int row = block % N;
  const int bh = block / N;
  const int head = bh % H;
  const int batch = bh / H;
  const int col = threadIdx.x;

  __shared__ float partial[N];

  const int state_base = (batch * H + head) * N * N;
  float st = state[state_base + row * N + col];

  for (int t = 0; t < T; ++t) {
    const int vec_base = ((batch * T + t) * H + head) * N;
    const float rv = static_cast<float>(r[vec_base + col]);
    const float wv = w_prep[vec_base + col];
    const float kv = static_cast<float>(k_prep[vec_base + col]);
    const float av = static_cast<float>(a[vec_base + col]);
    const float kk = kk_prep[vec_base + col];
    const float v_row = static_cast<float>(v_prep[vec_base + row]);

    const float state_dot_kk = rwkv7_block_reduce_sum_64(st * kk, partial);
    const float new_st = st * wv + v_row * kv - state_dot_kk * kk * av;
    st = new_st;
    const float recurrent = rwkv7_block_reduce_sum_64(new_st * rv, partial);
    if (threadIdx.x == 0) {
      out[vec_base + row] = static_cast<scalar_t>(recurrent);
    }
  }

  final_state[state_base + row * N + col] = st;
}

template <typename scalar_t>
__global__ void rwkv7_state_scan_prep_n64_rowblock_precomputed_wk_half_kernel(
    const scalar_t* __restrict__ r,
    const scalar_t* __restrict__ w_prep,
    const scalar_t* __restrict__ k_prep,
    const scalar_t* __restrict__ v_prep,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ kk_prep,
    const float* __restrict__ state,
    scalar_t* __restrict__ out,
    float* __restrict__ final_state,
    int B,
    int T,
    int H) {
  constexpr int N = 64;
  const int block = blockIdx.x;
  const int row = block % N;
  const int bh = block / N;
  const int head = bh % H;
  const int batch = bh / H;
  const int col = threadIdx.x;

  __shared__ float partial[N];

  const int state_base = (batch * H + head) * N * N;
  float st = state[state_base + row * N + col];

  for (int t = 0; t < T; ++t) {
    const int vec_base = ((batch * T + t) * H + head) * N;
    const float rv = static_cast<float>(r[vec_base + col]);
    const float wv = static_cast<float>(w_prep[vec_base + col]);
    const float kv = static_cast<float>(k_prep[vec_base + col]);
    const float av = static_cast<float>(a[vec_base + col]);
    const float kk = static_cast<float>(kk_prep[vec_base + col]);
    const float v_row = static_cast<float>(v_prep[vec_base + row]);

    const float state_dot_kk = rwkv7_block_reduce_sum_64(st * kk, partial);
    const float new_st = st * wv + v_row * kv - state_dot_kk * kk * av;
    st = new_st;
    const float recurrent = rwkv7_block_reduce_sum_64(new_st * rv, partial);
    if (threadIdx.x == 0) {
      out[vec_base + row] = static_cast<scalar_t>(recurrent);
    }
  }

  final_state[state_base + row * N + col] = st;
}

std::vector<torch::Tensor> rwkv7_state_scan_prep_forward_cuda(
    torch::Tensor r,
    torch::Tensor w_raw,
    torch::Tensor k_raw,
    torch::Tensor v_raw,
    torch::Tensor a,
    torch::Tensor state,
    torch::Tensor k_k,
    torch::Tensor k_a,
    torch::Tensor v_first,
    torch::Tensor v_gate,
    bool has_v_gate,
    int lanes_per_row,
    int precompute_mode,
    int rows_per_block,
    int schedule_mode,
    bool w_precomputed) {
  CHECK_INPUT(r);
  CHECK_INPUT(w_raw);
  CHECK_INPUT(k_raw);
  CHECK_INPUT(v_raw);
  CHECK_INPUT(a);
  CHECK_INPUT(state);
  CHECK_INPUT(k_k);
  CHECK_INPUT(k_a);
  if (has_v_gate) {
    CHECK_INPUT(v_first);
    CHECK_INPUT(v_gate);
  }
  TORCH_CHECK(r.dim() == 4, "r must be [B,T,H,N]");
  TORCH_CHECK(state.dim() == 4, "state must be [B,H,N,N]");
  const int B = static_cast<int>(r.size(0));
  const int T = static_cast<int>(r.size(1));
  const int H = static_cast<int>(r.size(2));
  const int N = static_cast<int>(r.size(3));
  TORCH_CHECK(N == 64, "CUDA prototype only supports head_dim=64");
  TORCH_CHECK(state.size(0) == B && state.size(1) == H && state.size(2) == 64 && state.size(3) == 64,
              "state shape mismatch");
  TORCH_CHECK(state.scalar_type() == torch::kFloat32, "state must be fp32");
  TORCH_CHECK(r.scalar_type() == torch::kFloat16, "CUDA prototype currently supports fp16 only");
  TORCH_CHECK(w_raw.scalar_type() == r.scalar_type() && k_raw.scalar_type() == r.scalar_type() &&
              v_raw.scalar_type() == r.scalar_type() && a.scalar_type() == r.scalar_type() &&
              k_k.scalar_type() == r.scalar_type() && k_a.scalar_type() == r.scalar_type(),
              "all vector inputs must match r dtype");
  TORCH_CHECK(lanes_per_row == 1 || lanes_per_row == 2 || lanes_per_row == 4 ||
              lanes_per_row == 8 || lanes_per_row == 16 || lanes_per_row == 64,
              "CUDA prototype lanes_per_row must be one of 1, 2, 4, 8, 16, or 64");
  TORCH_CHECK(precompute_mode == 0 || precompute_mode == 1 || precompute_mode == 2 || precompute_mode == 3,
              "CUDA vector precompute mode must be 0 (none), 1 (full), 2 (wk), or 3 (wk_half)");
  TORCH_CHECK(precompute_mode == 0 || lanes_per_row == 64,
              "CUDA vector precompute prototypes currently require lanes_per_row=64");
  TORCH_CHECK(rows_per_block == 1 || rows_per_block == 2 || rows_per_block == 4 || rows_per_block == 8 || rows_per_block == 16,
              "CUDA row-block cooperative rows_per_block must be one of 1, 2, 4, 8, or 16");
  TORCH_CHECK(rows_per_block == 1 || (lanes_per_row == 64 && precompute_mode == 0),
              "CUDA rows_per_block>1 currently requires lanes_per_row=64 and precompute_mode=none");
  TORCH_CHECK(schedule_mode == 0 || schedule_mode == 1 || schedule_mode == 2 || schedule_mode == 3 || schedule_mode == 4,
              "CUDA state-scan schedule_mode must be 0 (default), 1 (warp_specialized), 2 (warp2), 3 (head_reg16), or 4 (warp_pipelined)");
  TORCH_CHECK(schedule_mode == 0 || (lanes_per_row == 64 && precompute_mode == 0),
              "CUDA specialized schedules currently require lanes_per_row=64 and precompute_mode=none");
  TORCH_CHECK(!(schedule_mode == 2 && rows_per_block == 16),
              "CUDA warp2 rows_per_block=16 would exceed the max CUDA block size");
  TORCH_CHECK(!(schedule_mode == 3 && rows_per_block != 1),
              "CUDA head_reg16 schedule uses one CTA per head and requires rows_per_block=1");
  TORCH_CHECK(!w_precomputed || (lanes_per_row == 64 && precompute_mode == 0 && (schedule_mode == 1 || schedule_mode == 4)),
              "CUDA w_precomputed currently requires lanes_per_row=64, precompute_mode=none, and schedule=warp_specialized or warp_pipelined");

  auto out = torch::empty_like(r);
  auto final_state = torch::empty_like(state);
  auto k_out = torch::empty_like(k_raw);
  auto v_out = torch::empty_like(v_raw);

  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 grid(B * H);
  if (precompute_mode == 1) {
    auto w_prep = torch::empty({B, T, H, 64}, state.options());
    auto kk_prep = torch::empty({B, T, H, 64}, state.options());
    auto k_prep = torch::empty({B, T, H, 64}, state.options());
    auto v_prep = torch::empty({B, T, H, 64}, state.options());
    const dim3 pre_grid(B * T * H);
    const dim3 pre_block(64);
    rwkv7_state_scan_prep_n64_vector_precompute_kernel<at::Half><<<pre_grid, pre_block, 0, stream>>>(
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        w_prep.data_ptr<float>(),
        kk_prep.data_ptr<float>(),
        k_prep.data_ptr<float>(),
        v_prep.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);

    const dim3 row_grid(B * H * 64);
    const dim3 block(64);
    rwkv7_state_scan_prep_n64_rowblock_precomputed_kernel<at::Half><<<row_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_prep.data_ptr<float>(),
        k_prep.data_ptr<float>(),
        v_prep.data_ptr<float>(),
        a.data_ptr<at::Half>(),
        kk_prep.data_ptr<float>(),
        state.data_ptr<float>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        B, T, H);
  } else if (precompute_mode == 2) {
    auto w_prep = torch::empty({B, T, H, 64}, state.options());
    auto kk_prep = torch::empty({B, T, H, 64}, state.options());
    const dim3 pre_grid(B * T * H);
    const dim3 pre_block(64);
    rwkv7_state_scan_prep_n64_vector_precompute_wk_kernel<at::Half><<<pre_grid, pre_block, 0, stream>>>(
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        w_prep.data_ptr<float>(),
        kk_prep.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);

    const dim3 row_grid(B * H * 64);
    const dim3 block(64);
    rwkv7_state_scan_prep_n64_rowblock_precomputed_wk_kernel<at::Half><<<row_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_prep.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        kk_prep.data_ptr<float>(),
        state.data_ptr<float>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        B, T, H);
  } else if (precompute_mode == 3) {
    auto w_prep = torch::empty_like(w_raw);
    auto kk_prep = torch::empty_like(k_raw);
    const dim3 pre_grid(B * T * H);
    const dim3 pre_block(64);
    rwkv7_state_scan_prep_n64_vector_precompute_wk_half_kernel<at::Half><<<pre_grid, pre_block, 0, stream>>>(
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        w_prep.data_ptr<at::Half>(),
        kk_prep.data_ptr<at::Half>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);

    const dim3 row_grid(B * H * 64);
    const dim3 block(64);
    rwkv7_state_scan_prep_n64_rowblock_precomputed_wk_half_kernel<at::Half><<<row_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_prep.data_ptr<at::Half>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        kk_prep.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        B, T, H);
  } else if (lanes_per_row == 64 && schedule_mode == 3) {
    const dim3 head_grid(B * H);
    const dim3 block(1024);
    rwkv7_state_scan_prep_n64_head_reg16_kernel<at::Half><<<head_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  } else if (lanes_per_row == 64 && schedule_mode == 4 && w_precomputed && rows_per_block == 16) {
    const dim3 row_grid(B * H * 4);
    const dim3 block(32 * (16 + 1));
    rwkv7_state_scan_prep_n64_rowblock_warp_pipelined_kernel<at::Half, 16, true><<<row_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  } else if (lanes_per_row == 64 && schedule_mode == 4 && w_precomputed && rows_per_block == 8) {
    const dim3 row_grid(B * H * 8);
    const dim3 block(32 * (8 + 1));
    rwkv7_state_scan_prep_n64_rowblock_warp_pipelined_kernel<at::Half, 8, true><<<row_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  } else if (lanes_per_row == 64 && schedule_mode == 4 && w_precomputed && rows_per_block == 4) {
    const dim3 row_grid(B * H * 16);
    const dim3 block(32 * (4 + 1));
    rwkv7_state_scan_prep_n64_rowblock_warp_pipelined_kernel<at::Half, 4, true><<<row_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  } else if (lanes_per_row == 64 && schedule_mode == 4 && w_precomputed && rows_per_block == 2) {
    const dim3 row_grid(B * H * 32);
    const dim3 block(32 * (2 + 1));
    rwkv7_state_scan_prep_n64_rowblock_warp_pipelined_kernel<at::Half, 2, true><<<row_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  } else if (lanes_per_row == 64 && schedule_mode == 4 && w_precomputed) {
    const dim3 row_grid(B * H * 64);
    const dim3 block(32 * (1 + 1));
    rwkv7_state_scan_prep_n64_rowblock_warp_pipelined_kernel<at::Half, 1, true><<<row_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  } else if (lanes_per_row == 64 && schedule_mode == 4 && rows_per_block == 16) {
    const dim3 row_grid(B * H * 4);
    const dim3 block(32 * (16 + 1));
    rwkv7_state_scan_prep_n64_rowblock_warp_pipelined_kernel<at::Half, 16><<<row_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  } else if (lanes_per_row == 64 && schedule_mode == 4 && rows_per_block == 8) {
    const dim3 row_grid(B * H * 8);
    const dim3 block(32 * (8 + 1));
    rwkv7_state_scan_prep_n64_rowblock_warp_pipelined_kernel<at::Half, 8><<<row_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  } else if (lanes_per_row == 64 && schedule_mode == 4 && rows_per_block == 4) {
    const dim3 row_grid(B * H * 16);
    const dim3 block(32 * (4 + 1));
    rwkv7_state_scan_prep_n64_rowblock_warp_pipelined_kernel<at::Half, 4><<<row_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  } else if (lanes_per_row == 64 && schedule_mode == 4 && rows_per_block == 2) {
    const dim3 row_grid(B * H * 32);
    const dim3 block(32 * (2 + 1));
    rwkv7_state_scan_prep_n64_rowblock_warp_pipelined_kernel<at::Half, 2><<<row_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  } else if (lanes_per_row == 64 && schedule_mode == 4) {
    const dim3 row_grid(B * H * 64);
    const dim3 block(32 * (1 + 1));
    rwkv7_state_scan_prep_n64_rowblock_warp_pipelined_kernel<at::Half, 1><<<row_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  } else if (lanes_per_row == 64 && schedule_mode == 1 && w_precomputed && rows_per_block == 16) {
    const dim3 row_grid(B * H * 4);
    const dim3 block(32 * (16 + 1));
    rwkv7_state_scan_prep_n64_rowblock_warp_specialized_kernel<at::Half, 16, true><<<row_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  } else if (lanes_per_row == 64 && schedule_mode == 1 && w_precomputed && rows_per_block == 8) {
    const dim3 row_grid(B * H * 8);
    const dim3 block(32 * (8 + 1));
    rwkv7_state_scan_prep_n64_rowblock_warp_specialized_kernel<at::Half, 8, true><<<row_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  } else if (lanes_per_row == 64 && schedule_mode == 1 && w_precomputed && rows_per_block == 4) {
    const dim3 row_grid(B * H * 16);
    const dim3 block(32 * (4 + 1));
    rwkv7_state_scan_prep_n64_rowblock_warp_specialized_kernel<at::Half, 4, true><<<row_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  } else if (lanes_per_row == 64 && schedule_mode == 1 && w_precomputed && rows_per_block == 2) {
    const dim3 row_grid(B * H * 32);
    const dim3 block(32 * (2 + 1));
    rwkv7_state_scan_prep_n64_rowblock_warp_specialized_kernel<at::Half, 2, true><<<row_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  } else if (lanes_per_row == 64 && schedule_mode == 1 && w_precomputed) {
    const dim3 row_grid(B * H * 64);
    const dim3 block(32 * (1 + 1));
    rwkv7_state_scan_prep_n64_rowblock_warp_specialized_kernel<at::Half, 1, true><<<row_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  } else if (lanes_per_row == 64 && schedule_mode == 2 && rows_per_block == 8) {
    const dim3 row_grid(B * H * 8);
    const dim3 block(32 * (1 + 2 * 8));
    rwkv7_state_scan_prep_n64_rowblock_warp2_kernel<at::Half, 8><<<row_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  } else if (lanes_per_row == 64 && schedule_mode == 2 && rows_per_block == 4) {
    const dim3 row_grid(B * H * 16);
    const dim3 block(32 * (1 + 2 * 4));
    rwkv7_state_scan_prep_n64_rowblock_warp2_kernel<at::Half, 4><<<row_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  } else if (lanes_per_row == 64 && schedule_mode == 2 && rows_per_block == 2) {
    const dim3 row_grid(B * H * 32);
    const dim3 block(32 * (1 + 2 * 2));
    rwkv7_state_scan_prep_n64_rowblock_warp2_kernel<at::Half, 2><<<row_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  } else if (lanes_per_row == 64 && schedule_mode == 2) {
    const dim3 row_grid(B * H * 64);
    const dim3 block(32 * (1 + 2));
    rwkv7_state_scan_prep_n64_rowblock_warp2_kernel<at::Half, 1><<<row_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  } else if (lanes_per_row == 64 && schedule_mode == 1 && rows_per_block == 16) {
    const dim3 row_grid(B * H * 4);
    const dim3 block(32 * (16 + 1));
    rwkv7_state_scan_prep_n64_rowblock_warp_specialized_kernel<at::Half, 16><<<row_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  } else if (lanes_per_row == 64 && schedule_mode == 1 && rows_per_block == 8) {
    const dim3 row_grid(B * H * 8);
    const dim3 block(32 * (8 + 1));
    rwkv7_state_scan_prep_n64_rowblock_warp_specialized_kernel<at::Half, 8><<<row_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  } else if (lanes_per_row == 64 && schedule_mode == 1 && rows_per_block == 4) {
    const dim3 row_grid(B * H * 16);
    const dim3 block(32 * (4 + 1));
    rwkv7_state_scan_prep_n64_rowblock_warp_specialized_kernel<at::Half, 4><<<row_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  } else if (lanes_per_row == 64 && schedule_mode == 1 && rows_per_block == 2) {
    const dim3 row_grid(B * H * 32);
    const dim3 block(32 * (2 + 1));
    rwkv7_state_scan_prep_n64_rowblock_warp_specialized_kernel<at::Half, 2><<<row_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  } else if (lanes_per_row == 64 && schedule_mode == 1) {
    const dim3 row_grid(B * H * 64);
    const dim3 block(32 * (1 + 1));
    rwkv7_state_scan_prep_n64_rowblock_warp_specialized_kernel<at::Half, 1><<<row_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  } else if (lanes_per_row == 64 && rows_per_block == 16) {
    const dim3 row_grid(B * H * 4);
    const dim3 block(64 * 16);
    rwkv7_state_scan_prep_n64_rowblock_coop_kernel<at::Half, 16><<<row_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  } else if (lanes_per_row == 64 && rows_per_block == 8) {
    const dim3 row_grid(B * H * 8);
    const dim3 block(64 * 8);
    rwkv7_state_scan_prep_n64_rowblock_coop_kernel<at::Half, 8><<<row_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  } else if (lanes_per_row == 64 && rows_per_block == 4) {
    const dim3 row_grid(B * H * 16);
    const dim3 block(64 * 4);
    rwkv7_state_scan_prep_n64_rowblock_coop_kernel<at::Half, 4><<<row_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  } else if (lanes_per_row == 64 && rows_per_block == 2) {
    const dim3 row_grid(B * H * 32);
    const dim3 block(64 * 2);
    rwkv7_state_scan_prep_n64_rowblock_coop_kernel<at::Half, 2><<<row_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  } else if (lanes_per_row == 64) {
    const dim3 row_grid(B * H * 64);
    const dim3 block(64);
    rwkv7_state_scan_prep_n64_rowblock_kernel<at::Half><<<row_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  } else if (lanes_per_row == 2) {
    const dim3 block(64 * 2);
    rwkv7_state_scan_prep_n64_rowgroup_kernel<at::Half, 2><<<grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  } else if (lanes_per_row == 4) {
    const dim3 block(64 * 4);
    rwkv7_state_scan_prep_n64_rowgroup_kernel<at::Half, 4><<<grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  } else if (lanes_per_row == 8) {
    const dim3 block(64 * 8);
    rwkv7_state_scan_prep_n64_rowgroup_kernel<at::Half, 8><<<grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  } else if (lanes_per_row == 16) {
    const dim3 block(64 * 16);
    rwkv7_state_scan_prep_n64_rowgroup_kernel<at::Half, 16><<<grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  } else {
    const dim3 block(64);
    rwkv7_state_scan_prep_n64_kernel<at::Half><<<grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, final_state, k_out, v_out};
}

std::vector<torch::Tensor> rwkv7_state_scan_prep_sk_forward_cuda(
    torch::Tensor r,
    torch::Tensor w_raw,
    torch::Tensor k_raw,
    torch::Tensor v_raw,
    torch::Tensor a,
    torch::Tensor state,
    torch::Tensor k_k,
    torch::Tensor k_a,
    torch::Tensor r_k,
    torch::Tensor v_first,
    torch::Tensor v_gate,
    bool has_v_gate,
    int rows_per_block,
    int schedule_mode) {
  CHECK_INPUT(r);
  CHECK_INPUT(w_raw);
  CHECK_INPUT(k_raw);
  CHECK_INPUT(v_raw);
  CHECK_INPUT(a);
  CHECK_INPUT(state);
  CHECK_INPUT(k_k);
  CHECK_INPUT(k_a);
  CHECK_INPUT(r_k);
  if (has_v_gate) {
    CHECK_INPUT(v_first);
    CHECK_INPUT(v_gate);
  }
  TORCH_CHECK(r.dim() == 4, "r must be [B,T,H,N]");
  TORCH_CHECK(state.dim() == 4, "state must be [B,H,N,N]");
  const int B = static_cast<int>(r.size(0));
  const int T = static_cast<int>(r.size(1));
  const int H = static_cast<int>(r.size(2));
  const int N = static_cast<int>(r.size(3));
  TORCH_CHECK(N == 64, "CUDA SK prototype only supports head_dim=64");
  TORCH_CHECK(state.size(0) == B && state.size(1) == H && state.size(2) == 64 && state.size(3) == 64,
              "state shape mismatch");
  TORCH_CHECK(state.scalar_type() == torch::kFloat32, "state must be fp32");
  TORCH_CHECK(r.scalar_type() == torch::kFloat16, "CUDA SK prototype currently supports fp16 only");
  TORCH_CHECK(w_raw.scalar_type() == r.scalar_type() && k_raw.scalar_type() == r.scalar_type() &&
              v_raw.scalar_type() == r.scalar_type() && a.scalar_type() == r.scalar_type() &&
              k_k.scalar_type() == r.scalar_type() && k_a.scalar_type() == r.scalar_type() &&
              r_k.scalar_type() == r.scalar_type(),
              "all vector inputs must match r dtype");
  TORCH_CHECK(rows_per_block == 1 || rows_per_block == 2 || rows_per_block == 4 || rows_per_block == 8 || rows_per_block == 16,
              "CUDA SK rows_per_block must be one of 1, 2, 4, 8, or 16");
  TORCH_CHECK(schedule_mode == 1,
              "CUDA SK prototype currently supports only warp_specialized schedule");

  auto out = torch::empty_like(r);
  auto final_state = torch::empty_like(state);
  auto sk = torch::empty({B, T, H}, r.options());

  auto stream = at::cuda::getCurrentCUDAStream();
  if (rows_per_block == 16) {
    const dim3 row_grid(B * H * 4);
    const dim3 block(32 * (16 + 1));
    rwkv7_state_scan_prep_n64_rowblock_warp_specialized_sk_kernel<at::Half, 16><<<row_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        r_k.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        sk.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  } else if (rows_per_block == 8) {
    const dim3 row_grid(B * H * 8);
    const dim3 block(32 * (8 + 1));
    rwkv7_state_scan_prep_n64_rowblock_warp_specialized_sk_kernel<at::Half, 8><<<row_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        r_k.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        sk.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  } else if (rows_per_block == 4) {
    const dim3 row_grid(B * H * 16);
    const dim3 block(32 * (4 + 1));
    rwkv7_state_scan_prep_n64_rowblock_warp_specialized_sk_kernel<at::Half, 4><<<row_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        r_k.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        sk.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  } else if (rows_per_block == 2) {
    const dim3 row_grid(B * H * 32);
    const dim3 block(32 * (2 + 1));
    rwkv7_state_scan_prep_n64_rowblock_warp_specialized_sk_kernel<at::Half, 2><<<row_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        r_k.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        sk.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  } else {
    const dim3 row_grid(B * H * 64);
    const dim3 block(32 * (1 + 1));
    rwkv7_state_scan_prep_n64_rowblock_warp_specialized_sk_kernel<at::Half, 1><<<row_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        r_k.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        sk.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, final_state, sk};
}

std::vector<torch::Tensor> rwkv7_state_scan_rowblock_phase_forward_cuda(
    torch::Tensor r,
    torch::Tensor w_raw,
    torch::Tensor k_raw,
    torch::Tensor v_raw,
    torch::Tensor a,
    torch::Tensor state,
    torch::Tensor k_k,
    torch::Tensor k_a,
    torch::Tensor v_first,
    torch::Tensor v_gate,
    bool has_v_gate,
    int phase) {
  CHECK_INPUT(r);
  CHECK_INPUT(w_raw);
  CHECK_INPUT(k_raw);
  CHECK_INPUT(v_raw);
  CHECK_INPUT(a);
  CHECK_INPUT(state);
  CHECK_INPUT(k_k);
  CHECK_INPUT(k_a);
  if (has_v_gate) {
    CHECK_INPUT(v_first);
    CHECK_INPUT(v_gate);
  }
  TORCH_CHECK(r.dim() == 4, "r must be [B,T,H,N]");
  TORCH_CHECK(state.dim() == 4, "state must be [B,H,N,N]");
  const int B = static_cast<int>(r.size(0));
  const int T = static_cast<int>(r.size(1));
  const int H = static_cast<int>(r.size(2));
  const int N = static_cast<int>(r.size(3));
  TORCH_CHECK(N == 64, "CUDA rowblock phase prototype only supports head_dim=64");
  TORCH_CHECK(state.size(0) == B && state.size(1) == H && state.size(2) == 64 && state.size(3) == 64,
              "state shape mismatch");
  TORCH_CHECK(state.scalar_type() == torch::kFloat32, "state must be fp32");
  TORCH_CHECK(r.scalar_type() == torch::kFloat16, "CUDA rowblock phase prototype currently supports fp16 only");
  TORCH_CHECK(w_raw.scalar_type() == r.scalar_type() && k_raw.scalar_type() == r.scalar_type() &&
              v_raw.scalar_type() == r.scalar_type() && a.scalar_type() == r.scalar_type() &&
              k_k.scalar_type() == r.scalar_type() && k_a.scalar_type() == r.scalar_type(),
              "all vector inputs must match r dtype");
  TORCH_CHECK(phase == 0 || phase == 1 || phase == 2 || phase == 3,
              "CUDA rowblock phase must be 0 (prep_norm), 1 (state_dot), 2 (update), or 3 (full)");

  auto out = torch::empty_like(r);
  auto final_state = torch::empty_like(state);
  auto k_out = torch::empty_like(k_raw);
  auto v_out = torch::empty_like(v_raw);
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 row_grid(B * H * 64);
  const dim3 block(64);
  if (phase == 0) {
    rwkv7_state_scan_prep_n64_rowblock_phase_kernel<at::Half, 0><<<row_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  } else if (phase == 1) {
    rwkv7_state_scan_prep_n64_rowblock_phase_kernel<at::Half, 1><<<row_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  } else if (phase == 2) {
    rwkv7_state_scan_prep_n64_rowblock_phase_kernel<at::Half, 2><<<row_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  } else {
    rwkv7_state_scan_prep_n64_rowblock_phase_kernel<at::Half, 3><<<row_grid, block, 0, stream>>>(
        r.data_ptr<at::Half>(),
        w_raw.data_ptr<at::Half>(),
        k_raw.data_ptr<at::Half>(),
        v_raw.data_ptr<at::Half>(),
        a.data_ptr<at::Half>(),
        state.data_ptr<float>(),
        k_k.data_ptr<at::Half>(),
        k_a.data_ptr<at::Half>(),
        has_v_gate ? v_first.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        has_v_gate ? v_gate.data_ptr<at::Half>() : v_raw.data_ptr<at::Half>(),
        out.data_ptr<at::Half>(),
        final_state.data_ptr<float>(),
        k_out.data_ptr<at::Half>(),
        v_out.data_ptr<at::Half>(),
        B, T, H, has_v_gate);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, final_state, k_out, v_out};
}
"""


@lru_cache(maxsize=1)
def _load_extension():
    if torch is None:
        raise RuntimeError("cuda_state_scan requires torch")
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name="rwkv7_cuda_state_scan_v17",
        cpp_sources=_CPP_SRC,
        cuda_sources=_CUDA_SRC,
        functions=["state_scan_prep_forward", "state_scan_prep_sk_forward", "state_scan_rowblock_phase_forward"],
        with_cuda=True,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )


def cuda_state_scan_prep_available() -> bool:
    return bool(torch is not None and torch.cuda.is_available())


def cuda_state_scan_prep_sk_available() -> bool:
    return bool(torch is not None and torch.cuda.is_available())


def cuda_state_scan_prep(
    r: Any,
    w_raw: Any,
    k_raw: Any,
    v_raw: Any,
    a: Any,
    state: Any,
    k_k: Any,
    k_a: Any,
    *,
    v_first: Any | None = None,
    v_gate: Any | None = None,
    lanes_per_row: int | None = None,
    precompute_vector: bool | None = None,
    precompute_mode: str | None = None,
    rows_per_block: int | None = None,
    schedule: str | None = None,
    w_precomputed: bool | None = None,
):
    """Run the experimental CUDA N=64 state-prep scan.

    All sequence tensors must be contiguous `[B,T,H,64]` fp16 CUDA tensors and
    `state` must be `[B,H,64,64]` fp32.  Returns `(out, final_state, k, v)`.
    """

    if torch is None:
        raise RuntimeError("cuda_state_scan_prep requires torch")
    if int(r.shape[-1]) != 64:
        raise ValueError("cuda_state_scan_prep only supports head_dim=64")
    if r.dtype is not torch.float16:
        raise ValueError("cuda_state_scan_prep currently supports fp16 only")
    has_v_gate = v_first is not None and v_gate is not None
    if not has_v_gate:
        v_first = v_raw
        v_gate = v_raw
    if lanes_per_row is None:
        import os

        try:
            lanes_per_row = int(os.environ.get("RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_LANES", "1"))
        except ValueError:
            lanes_per_row = 1
    if precompute_vector is None:
        import os

        if precompute_mode is not None:
            precompute_vector = str(precompute_mode).strip().lower().replace("-", "_") not in {
                "0",
                "false",
                "no",
                "off",
                "none",
            }
        else:
            precompute_vector = os.environ.get("RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_PRECOMPUTE", "0").lower() not in {
                "0",
                "false",
                "no",
                "off",
            }
    mode_name = "none"
    if bool(precompute_vector):
        import os

        raw_mode = precompute_mode
        if raw_mode is None:
            raw_mode = os.environ.get("RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_PRECOMPUTE_MODE", "full")
        mode_name = str(raw_mode).strip().lower().replace("-", "_")
        if mode_name in {"1", "true", "yes", "on"}:
            mode_name = "full"
        elif mode_name in {"2", "wk", "wkk", "w_kk", "reduced", "reduced_temp", "wk_fp16kv", "fp16kv"}:
            mode_name = "wk"
        elif mode_name in {"3", "wk_half", "wk16", "half", "fp16", "fp16_temp", "half_temp"}:
            mode_name = "wk_half"
        elif mode_name in {"0", "false", "no", "off", "none"}:
            mode_name = "none"
        elif mode_name != "full":
            raise ValueError(
                "cuda_state_scan_prep precompute_mode must be one of full, wk/reduced_temp, wk_half/fp16_temp, or none"
            )
    if int(lanes_per_row) not in {1, 2, 4, 8, 16, 64}:
        raise ValueError("cuda_state_scan_prep lanes_per_row must be one of 1, 2, 4, 8, 16, or 64")
    precompute_mode_id = 0 if mode_name == "none" else (1 if mode_name == "full" else (3 if mode_name == "wk_half" else 2))
    if precompute_mode_id and int(lanes_per_row) != 64:
        raise ValueError("cuda_state_scan_prep precompute_vector currently requires lanes_per_row=64")
    if rows_per_block is None:
        import os

        try:
            rows_per_block = int(os.environ.get("RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_ROWS_PER_BLOCK", "1"))
        except ValueError:
            rows_per_block = 1
    if int(rows_per_block) not in {1, 2, 4, 8, 16}:
        raise ValueError("cuda_state_scan_prep rows_per_block must be one of 1, 2, 4, 8, or 16")
    if int(rows_per_block) != 1 and (int(lanes_per_row) != 64 or precompute_mode_id != 0):
        raise ValueError("cuda_state_scan_prep rows_per_block>1 requires lanes_per_row=64 and precompute_mode=none")
    if schedule is None:
        import os

        schedule = os.environ.get("RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_SCHEDULE", "default")
    schedule_name = str(schedule).strip().lower().replace("-", "_")
    if schedule_name in {"", "0", "default", "normal", "rowblock", "none"}:
        schedule_id = 0
    elif schedule_name in {"1", "warp", "warp_specialized", "warp_specialised", "producer_worker", "producer"}:
        schedule_id = 1
    elif schedule_name in {
        "2",
        "warp2",
        "warp_2",
        "warp_specialized2",
        "warp_specialized_2",
        "producer_worker2",
        "producer_worker_wide",
        "wide",
    }:
        schedule_id = 2
    elif schedule_name in {"3", "head_reg16", "head_reg", "head", "reg16", "head_level", "headlevel"}:
        schedule_id = 3
    elif schedule_name in {"4", "warp_pipelined", "warp_pipe", "pipelined", "pipeline", "pipe"}:
        schedule_id = 4
    else:
        raise ValueError("cuda_state_scan_prep schedule must be default, warp_specialized, warp2, head_reg16, or warp_pipelined")
    if schedule_id and (int(lanes_per_row) != 64 or precompute_mode_id != 0):
        raise ValueError("cuda_state_scan_prep specialized schedules require lanes_per_row=64 and precompute_mode=none")
    if schedule_id == 2 and int(rows_per_block) == 16:
        raise ValueError("cuda_state_scan_prep warp2 rows_per_block=16 would exceed the max CUDA block size")
    if schedule_id == 3 and int(rows_per_block) != 1:
        raise ValueError("cuda_state_scan_prep head_reg16 uses one CTA per head and requires rows_per_block=1")
    if w_precomputed is None:
        import os

        w_precomputed = os.environ.get("RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_W_PRECOMPUTED", "0").lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
    if bool(w_precomputed) and (int(lanes_per_row) != 64 or precompute_mode_id != 0 or schedule_id not in {1, 4}):
        raise ValueError(
            "cuda_state_scan_prep w_precomputed requires lanes_per_row=64, precompute_mode=none, and schedule=warp_specialized or warp_pipelined"
        )
    ext = _load_extension()
    out, final_state, k_out, v_out = ext.state_scan_prep_forward(
        r.contiguous(),
        w_raw.contiguous(),
        k_raw.contiguous(),
        v_raw.contiguous(),
        a.contiguous(),
        state.contiguous(),
        k_k.reshape(-1).contiguous(),
        k_a.reshape(-1).contiguous(),
        v_first.contiguous(),
        v_gate.contiguous(),
        bool(has_v_gate),
        int(lanes_per_row),
        int(precompute_mode_id),
        int(rows_per_block),
        int(schedule_id),
        bool(w_precomputed),
    )
    return out, final_state, k_out, v_out


def cuda_state_scan_prep_sk(
    r: Any,
    w_raw: Any,
    k_raw: Any,
    v_raw: Any,
    a: Any,
    state: Any,
    k_k: Any,
    k_a: Any,
    r_k: Any,
    *,
    v_first: Any | None = None,
    v_gate: Any | None = None,
    rows_per_block: int | None = None,
    schedule: str | None = None,
):
    """Run the experimental CUDA state scan and emit only per-head ``sk``.

    This variant keeps adjusted K/V local to the scan and returns
    ``(out, final_state, sk)`` where ``sk = sum(r * k_adj * r_k)``.  It is meant
    to pair with ``fused_attn_output_prepare_from_sk_raw_v`` so the route avoids
    writing full K/V tensors out of the scan.
    """

    if torch is None:
        raise RuntimeError("cuda_state_scan_prep_sk requires torch")
    if int(r.shape[-1]) != 64:
        raise ValueError("cuda_state_scan_prep_sk only supports head_dim=64")
    if r.dtype is not torch.float16:
        raise ValueError("cuda_state_scan_prep_sk currently supports fp16 only")
    has_v_gate = v_first is not None and v_gate is not None
    if not has_v_gate:
        v_first = v_raw
        v_gate = v_raw
    if rows_per_block is None:
        import os

        try:
            rows_per_block = int(os.environ.get("RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_ROWS_PER_BLOCK", "1"))
        except ValueError:
            rows_per_block = 1
    if int(rows_per_block) not in {1, 2, 4, 8, 16}:
        raise ValueError("cuda_state_scan_prep_sk rows_per_block must be one of 1, 2, 4, 8, or 16")
    if schedule is None:
        import os

        schedule = os.environ.get("RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_SCHEDULE", "warp_specialized")
    schedule_name = str(schedule).strip().lower().replace("-", "_")
    if schedule_name in {"1", "warp", "warp_specialized", "warp_specialised", "producer_worker", "producer"}:
        schedule_id = 1
    else:
        raise ValueError("cuda_state_scan_prep_sk currently supports only warp_specialized schedule")
    ext = _load_extension()
    out, final_state, sk = ext.state_scan_prep_sk_forward(
        r.contiguous(),
        w_raw.contiguous(),
        k_raw.contiguous(),
        v_raw.contiguous(),
        a.contiguous(),
        state.contiguous(),
        k_k.reshape(-1).contiguous(),
        k_a.reshape(-1).contiguous(),
        r_k.reshape(-1).contiguous(),
        v_first.contiguous(),
        v_gate.contiguous(),
        bool(has_v_gate),
        int(rows_per_block),
        int(schedule_id),
    )
    return out, final_state, sk


def cuda_state_scan_rowblock_phase(
    r: Any,
    w_raw: Any,
    k_raw: Any,
    v_raw: Any,
    a: Any,
    state: Any,
    k_k: Any,
    k_a: Any,
    *,
    v_first: Any | None = None,
    v_gate: Any | None = None,
    phase: int | str = 3,
):
    """Run a row-block cumulative phase micro-kernel for profiling.

    Phases are cumulative: ``0`` = vector prep + K-normalization,
    ``1`` additionally computes state-dot-KK, ``2`` additionally updates the
    row state, and ``3`` additionally computes recurrent output.  The wrapper
    is synthetic/profiling-only and returns the same tensor tuple shape as
    :func:`cuda_state_scan_prep`.
    """

    if torch is None:
        raise RuntimeError("cuda_state_scan_rowblock_phase requires torch")
    if int(r.shape[-1]) != 64:
        raise ValueError("cuda_state_scan_rowblock_phase only supports head_dim=64")
    if r.dtype is not torch.float16:
        raise ValueError("cuda_state_scan_rowblock_phase currently supports fp16 only")
    has_v_gate = v_first is not None and v_gate is not None
    if not has_v_gate:
        v_first = v_raw
        v_gate = v_raw
    if isinstance(phase, str):
        phase_name = phase.strip().lower().replace("-", "_")
        phase_map = {
            "prep": 0,
            "prep_norm": 0,
            "vector_prep": 0,
            "state_dot": 1,
            "dot": 1,
            "update": 2,
            "state_update": 2,
            "full": 3,
            "recurrent": 3,
            "recurrent_output": 3,
        }
        if phase_name not in phase_map:
            raise ValueError("phase must be prep_norm, state_dot, update, full, or an integer 0..3")
        phase_id = phase_map[phase_name]
    else:
        phase_id = int(phase)
    if phase_id not in {0, 1, 2, 3}:
        raise ValueError("phase must be 0, 1, 2, or 3")
    ext = _load_extension()
    out, final_state, k_out, v_out = ext.state_scan_rowblock_phase_forward(
        r.contiguous(),
        w_raw.contiguous(),
        k_raw.contiguous(),
        v_raw.contiguous(),
        a.contiguous(),
        state.contiguous(),
        k_k.reshape(-1).contiguous(),
        k_a.reshape(-1).contiguous(),
        v_first.contiguous(),
        v_gate.contiguous(),
        bool(has_v_gate),
        int(phase_id),
    )
    return out, final_state, k_out, v_out
