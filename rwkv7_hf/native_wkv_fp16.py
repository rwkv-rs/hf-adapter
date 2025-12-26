# coding=utf-8
"""Opt-in exact-shape FP16 recurrent-state kernel for native decode.

The production Native/HF cache keeps FP32 recurrent state by default. This
module owns the lower-precision compatibility lane used only when callers
explicitly request FP16 state and the exact CUDA shape is supported.
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
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


_CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor rwkv7_native_fp16_recurrent_output_raw_cuda(
    torch::Tensor r, torch::Tensor w_raw, torch::Tensor k_raw,
    torch::Tensor v, torch::Tensor a, torch::Tensor state, torch::Tensor g,
    torch::Tensor k_k, torch::Tensor k_a, torch::Tensor r_k,
    torch::Tensor norm_weight, torch::Tensor norm_bias,
    torch::Tensor elapsed, bool advance_elapsed, double eps);

torch::Tensor rwkv7_native_fp16_sequence_cuda(
    torch::Tensor r, torch::Tensor w, torch::Tensor k, torch::Tensor v,
    torch::Tensor neg_kk, torch::Tensor kka, torch::Tensor state,
    torch::Tensor elapsed, torch::Tensor w0, bool add_w0);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("recurrent_output_raw", &rwkv7_native_fp16_recurrent_output_raw_cuda,
        "RWKV-7 native FP16-state recurrent update and output preparation");
  m.def("sequence", &rwkv7_native_fp16_sequence_cuda,
        "RWKV-7 native FP16-state sequence recurrence");
}
"""


# The half-state recurrence and deterministic rotator follow the Apache-2.0
# RWKV-Gradio-3 v3a kernel. The raw K/A preparation and output boundary are
# fused here for the Native/HF graph-state layout.
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

constexpr int N = 64;
constexpr int HALF2_N = N / 2;
constexpr int LDG_ELEMS = sizeof(int4) / sizeof(half);
constexpr float TWO_NEG_41 = 4.547473508864641e-13f;
constexpr float NEXP_HALF_LOG2_E = -0.8750387749145276f;
constexpr float NLOG2_E = -1.4426950408889634f;
constexpr uint32_t ROT1 = 2654435769u;

__device__ __forceinline__ float warp_sum(float value) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ float block_sum_64(float value, float* partial) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = warp_sum(value);
  if (lane == 0) partial[warp] = value;
  __syncthreads();
  const float total = partial[0] + partial[1];
  __syncthreads();
  return total;
}

__device__ __forceinline__ float rotator1(int x) {
  const uint32_t bits = ROT1 * static_cast<uint32_t>(x);
  return TWO_NEG_41 * static_cast<float>(static_cast<int32_t>(bits));
}

__device__ __forceinline__ half w_delta(half w_raw, int phase) {
  const float w = __half2float(w_raw);
  const float d = exp2f(NEXP_HALF_LOG2_E / (1.0f + exp2f(NLOG2_E * w)))
                  - 1.0f + rotator1(phase);
  return __float2half_rn(d);
}

template <bool AddW0>
__device__ __forceinline__ half w_delta_maybe_w0(
    half w_raw, const half* __restrict__ w0_ptr, int channel, int phase) {
  float value = __half2float(w_raw);
  if constexpr (AddW0) value += __half2float(w0_ptr[channel]);
  const float d = exp2f(NEXP_HALF_LOG2_E / (1.0f + exp2f(NLOG2_E * value)))
                  - 1.0f + rotator1(phase);
  return __float2half_rn(d);
}

template <int Bytes>
__device__ __forceinline__ void sequence_cp_async(
    void* shared_ptr, const void* global_ptr, bool predicate) {
  static_assert(Bytes == 4, "the sequence kernel copies one half2 per lane");
#if __CUDA_ARCH__ >= 800
  const int source_bytes = predicate ? Bytes : 0;
  const unsigned int shared_address = __cvta_generic_to_shared(shared_ptr);
  asm volatile(
      "cp.async.ca.shared.global [%0], [%1], %2, %3;"
      :: "r"(shared_address), "l"(global_ptr), "n"(Bytes), "r"(source_bytes));
#else
  *reinterpret_cast<half2*>(shared_ptr) = predicate
      ? *reinterpret_cast<const half2*>(global_ptr)
      : __float2half2_rn(0.0f);
#endif
}

__device__ __forceinline__ void sequence_cp_commit() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__ void sequence_cp_wait() {
#if __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_all;\n" ::);
#endif
}

__device__ __forceinline__ void sequence_prefetch_token(
    int thread_id,
    int lane,
    int token,
    half2* r,
    half2* w,
    half2* k,
    half2* neg_kk,
    half2* kka,
    half2* dummy,
    const half* r_ptr,
    const half* w_ptr,
    const half* k_ptr,
    const half* neg_kk_ptr,
    const half* kka_ptr) {
  sequence_cp_async<4>(
      (thread_id < 32 ? w : neg_kk) + lane,
      reinterpret_cast<const half2*>(thread_id < 32 ? w_ptr + token : neg_kk_ptr + token) + lane,
      true);
  sequence_cp_commit();
  sequence_cp_async<4>(
      (thread_id < 32 ? r : k) + lane,
      reinterpret_cast<const half2*>(thread_id < 32 ? r_ptr + token : k_ptr + token) + lane,
      true);
  sequence_cp_async<4>(
      (thread_id < 32 ? kka : dummy) + lane,
      reinterpret_cast<const half2*>(kka_ptr + token) + lane,
      thread_id < 32);
  sequence_cp_commit();
}

__global__ __launch_bounds__(N, 2) void native_fp16_recurrent_output_raw_kernel(
    int H,
    const half* __restrict__ r_ptr,
    const half* __restrict__ w_raw_ptr,
    const half* __restrict__ k_raw_ptr,
    const half* __restrict__ v_ptr,
    const half* __restrict__ a_ptr,
    half* __restrict__ state_ptr,
    const half* __restrict__ g_ptr,
    const half* __restrict__ k_k_ptr,
    const half* __restrict__ k_a_ptr,
    const half* __restrict__ r_k_ptr,
    const half* __restrict__ norm_weight_ptr,
    const half* __restrict__ norm_bias_ptr,
    int* __restrict__ elapsed_ptr,
    half* __restrict__ output_ptr,
    bool advance_elapsed,
    float eps) {
  const int bh = blockIdx.x;
  const int b = bh / H;
  const int h = bh % H;
  const int i = threadIdx.x;
  const int lane = i & 31;
  const int64_t vec_base = static_cast<int64_t>(bh) * N;
  const int64_t param_base = static_cast<int64_t>(h) * N;
  half* state_base = state_ptr + static_cast<int64_t>(bh) * N * N;

  __shared__ __align__(256) half2 state_smem[N][HALF2_N];
  __shared__ __align__(128) half r[N], w[N], k[N], neg_kk[N], bvec[N];
  __shared__ __align__(128) half v[N], g[N], recurrent[N];
  __shared__ float partial[2];

  #pragma unroll
  for (int j0 = 0; j0 < N / LDG_ELEMS; ++j0) {
    const int4 state_vec = reinterpret_cast<int4*>(state_base)[j0 * N + i];
    #pragma unroll
    for (int j1 = 0; j1 < LDG_ELEMS / 2; ++j1) {
      const int row = j0 * LDG_ELEMS + i * LDG_ELEMS / N;
      const int col = i * LDG_ELEMS % N / 2 + j1;
      state_smem[row][(row & 31) ^ col] =
          reinterpret_cast<const half2*>(&state_vec)[j1];
    }
  }
  __syncthreads();

  half2 state[HALF2_N];
  #pragma unroll
  for (int j = 0; j < HALF2_N; ++j) {
    state[j] = state_smem[i][lane ^ j];
  }

  const half r_value = r_ptr[vec_base + i];
  const half k_raw_value = k_raw_ptr[vec_base + i];
  const half v_value = v_ptr[vec_base + i];
  const half a_value = a_ptr[vec_base + i];
  const float kk_unscaled = __half2float(k_raw_value)
                            * __half2float(k_k_ptr[param_base + i]);
  const float kk_norm_sq = block_sum_64(kk_unscaled * kk_unscaled, partial);
  const float kk_value = kk_unscaled / fmaxf(sqrtf(kk_norm_sq), 1.0e-12f);
  const float a_float = __half2float(a_value);
  const float ka_float = __half2float(k_a_ptr[param_base + i]);
  const half k_value = __float2half_rn(
      __half2float(k_raw_value) * fmaf(a_float, ka_float, 1.0f - ka_float));
  const half kk_half = __float2half_rn(kk_value);

  r[i] = r_value;
  w[i] = w_delta(w_raw_ptr[vec_base + i], elapsed_ptr[b] + h * N + i);
  k[i] = k_value;
  neg_kk[i] = __hneg(kk_half);
  bvec[i] = __float2half_rn(__half2float(kk_half) * a_float);
  v[i] = v_value;
  g[i] = g_ptr[vec_base + i];
  __syncthreads();

  half2 sa2 = __float2half2_rn(0.0f);
  #pragma unroll
  for (int j = 0; j < HALF2_N; ++j) {
    const half2 neg_kk2 = *reinterpret_cast<const half2*>(neg_kk + j * 2);
    sa2 = __hfma2(neg_kk2, state[j], sa2);
  }
  const half sa = __hadd(sa2.x, sa2.y);
  const half2 sa_broadcast = __halves2half2(sa, sa);
  const half2 vv2 = __halves2half2(v_value, v_value);
  half2 y2 = __float2half2_rn(0.0f);
  #pragma unroll
  for (int j = 0; j < HALF2_N; ++j) {
    const half2 w2 = *reinterpret_cast<const half2*>(w + j * 2);
    const half2 k2 = *reinterpret_cast<const half2*>(k + j * 2);
    const half2 b2 = *reinterpret_cast<const half2*>(bvec + j * 2);
    const half2 r2 = *reinterpret_cast<const half2*>(r + j * 2);
    half2 s = state[j];
    s = __hfma2(s, w2, __hfma2(k2, vv2, __hfma2(sa_broadcast, b2, s)));
    state[j] = s;
    y2 = __hfma2(s, r2, y2);
  }
  recurrent[i] = __hadd(y2.x, y2.y);

  #pragma unroll
  for (int j = 0; j < HALF2_N; ++j) {
    state_smem[i][lane ^ j] = state[j];
  }
  __syncthreads();

  const float recurrent_value = __half2float(recurrent[i]);
  const float mean = block_sum_64(recurrent_value, partial) * (1.0f / 64.0f);
  const float centered = recurrent_value - mean;
  const float variance = block_sum_64(centered * centered, partial) * (1.0f / 64.0f);
  const float correction = block_sum_64(
      __half2float(r_value) * __half2float(k_value)
      * __half2float(r_k_ptr[param_base + i]),
      partial);
  const float prepared = (
      centered * rsqrtf(variance + eps)
      * __half2float(norm_weight_ptr[param_base + i])
      + __half2float(norm_bias_ptr[param_base + i])
      + correction * __half2float(v_value))
      * __half2float(g[i]);
  output_ptr[vec_base + i] = __float2half_rn(prepared);

  __syncthreads();
  #pragma unroll
  for (int j0 = 0; j0 < N / LDG_ELEMS; ++j0) {
    int4 state_vec;
    #pragma unroll
    for (int j1 = 0; j1 < LDG_ELEMS / 2; ++j1) {
      const int row = j0 * LDG_ELEMS + i * LDG_ELEMS / N;
      const int col = i * LDG_ELEMS % N / 2 + j1;
      reinterpret_cast<half2*>(&state_vec)[j1] =
          state_smem[row][(row & 31) ^ col];
    }
    reinterpret_cast<int4*>(state_base)[j0 * N + i] = state_vec;
  }
  if (advance_elapsed && h == 0 && i == 0) elapsed_ptr[b] += 1;
}

template <bool AddW0>
__global__ __launch_bounds__(N, 2) void native_fp16_sequence_kernel(
    int T,
    int C,
    int H,
    half* __restrict__ state_ptr,
    const half* __restrict__ r_ptr,
    const half* __restrict__ w_ptr,
    const half* __restrict__ w0_ptr,
    const half* __restrict__ k_ptr,
    const half* __restrict__ v_ptr,
    const half* __restrict__ neg_kk_ptr,
    const half* __restrict__ kka_ptr,
    half* __restrict__ output_ptr,
    const int* __restrict__ elapsed_ptr) {
  const int bh = blockIdx.x;
  const int batch = bh / H;
  const int head = bh - batch * H;
  const int i = threadIdx.x;
  const int lane = i & 31;

  __shared__ __align__(256) half2 state_smem[N][HALF2_N];
  half* state_base = state_ptr
      + static_cast<int64_t>(batch) * C * N
      + head * N * N;

  #pragma unroll
  for (int j0 = 0; j0 < N / LDG_ELEMS; ++j0) {
    const int4 state_vec = reinterpret_cast<int4*>(state_base)[j0 * N + i];
    #pragma unroll
    for (int j1 = 0; j1 < LDG_ELEMS / 2; ++j1) {
      const int row = j0 * LDG_ELEMS + i * LDG_ELEMS / N;
      const int col = i * LDG_ELEMS % N / 2 + j1;
      state_smem[row][(row & 31) ^ col] =
          reinterpret_cast<const half2*>(&state_vec)[j1];
    }
  }
  __syncthreads();

  half2 state[HALF2_N];
  #pragma unroll
  for (int j = 0; j < HALF2_N; ++j) state[j] = state_smem[i][lane ^ j];

  __shared__ __align__(128) half2 r[2][HALF2_N];
  __shared__ __align__(128) half2 w[2][HALF2_N];
  __shared__ __align__(128) half2 k[2][HALF2_N];
  __shared__ __align__(128) half2 neg_kk[2][HALF2_N];
  __shared__ __align__(128) half2 kka[2][HALF2_N];
  __shared__ __align__(128) half2 dummy[HALF2_N];

  int token = (batch * T) * C + head * N;
  sequence_prefetch_token(
      i, lane, token, r[0], w[0], k[0], neg_kk[0], kka[0], dummy,
      r_ptr, w_ptr, k_ptr, neg_kk_ptr, kka_ptr);

  for (int tt = 0; tt < T; ++tt) {
    const int current = tt & 1;
    sequence_cp_wait();
    __syncthreads();

    half2 state_a = __float2half2_rn(0.0f);
    #pragma unroll
    for (int j = 0; j < HALF2_N; ++j) {
      state_a = __hfma2(neg_kk[current][j], state[j], state_a);
    }
    const half state_a_sum = __hadd(state_a.x, state_a.y);
    const half2 state_a_broadcast = __halves2half2(state_a_sum, state_a_sum);
    reinterpret_cast<half*>(w[current])[i] = w_delta_maybe_w0<AddW0>(
        reinterpret_cast<half*>(w[current])[i],
        w0_ptr,
        head * N + i,
        elapsed_ptr[batch] + head * N + i + tt);
    __syncthreads();

    if (tt + 1 < T) {
      sequence_prefetch_token(
          i, lane, token + C,
          r[current ^ 1], w[current ^ 1], k[current ^ 1],
          neg_kk[current ^ 1], kka[current ^ 1], dummy,
          r_ptr, w_ptr, k_ptr, neg_kk_ptr, kka_ptr);
    }

    const half value = v_ptr[token + i];
    const half2 value2 = __halves2half2(value, value);
    half2 output2 = __float2half2_rn(0.0f);
    #pragma unroll
    for (int j = 0; j < HALF2_N; ++j) {
      half2 updated = state[j];
      updated = __hfma2(
          updated,
          w[current][j],
          __hfma2(
              k[current][j],
              value2,
              __hfma2(state_a_broadcast, kka[current][j], updated)));
      state[j] = updated;
      output2 = __hfma2(updated, r[current][j], output2);
    }
    output_ptr[token + i] = __hadd(output2.x, output2.y);
    token += C;
  }

  #pragma unroll
  for (int j = 0; j < HALF2_N; ++j) state_smem[i][lane ^ j] = state[j];
  __syncthreads();
  #pragma unroll
  for (int j0 = 0; j0 < N / LDG_ELEMS; ++j0) {
    int4 state_vec;
    #pragma unroll
    for (int j1 = 0; j1 < LDG_ELEMS / 2; ++j1) {
      const int row = j0 * LDG_ELEMS + i * LDG_ELEMS / N;
      const int col = i * LDG_ELEMS % N / 2 + j1;
      reinterpret_cast<half2*>(&state_vec)[j1] =
          state_smem[row][(row & 31) ^ col];
    }
    reinterpret_cast<int4*>(state_base)[j0 * N + i] = state_vec;
  }
}

void check_half_cuda(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.scalar_type() == at::kHalf, name, " must be fp16");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

}  // namespace

torch::Tensor rwkv7_native_fp16_recurrent_output_raw_cuda(
    torch::Tensor r, torch::Tensor w_raw, torch::Tensor k_raw,
    torch::Tensor v, torch::Tensor a, torch::Tensor state, torch::Tensor g,
    torch::Tensor k_k, torch::Tensor k_a, torch::Tensor r_k,
    torch::Tensor norm_weight, torch::Tensor norm_bias,
    torch::Tensor elapsed, bool advance_elapsed, double eps) {
  check_half_cuda(r, "r");
  check_half_cuda(w_raw, "w_raw");
  check_half_cuda(k_raw, "k_raw");
  check_half_cuda(v, "v");
  check_half_cuda(a, "a");
  check_half_cuda(state, "state");
  check_half_cuda(g, "g");
  check_half_cuda(k_k, "k_k");
  check_half_cuda(k_a, "k_a");
  check_half_cuda(r_k, "r_k");
  check_half_cuda(norm_weight, "norm_weight");
  check_half_cuda(norm_bias, "norm_bias");
  TORCH_CHECK(state.dim() == 4 && state.size(2) == N && state.size(3) == N,
              "state must be [B,H,64,64]");
  const int B = static_cast<int>(state.size(0));
  const int H = static_cast<int>(state.size(1));
  TORCH_CHECK(r.numel() == static_cast<int64_t>(B) * H * N,
              "r shape does not match state");
  for (const auto& tensor : {w_raw, k_raw, v, a, g}) {
    TORCH_CHECK(tensor.sizes() == r.sizes(), "token tensors must share shape");
  }
  for (const auto& tensor : {k_k, k_a, r_k, norm_weight, norm_bias}) {
    TORCH_CHECK(tensor.numel() == static_cast<int64_t>(H) * N,
                "per-channel tensor shape mismatch");
  }
  TORCH_CHECK(elapsed.is_cuda() && elapsed.scalar_type() == at::kInt
              && elapsed.is_contiguous() && elapsed.numel() == B,
              "elapsed must be contiguous CUDA int32 [B]");

  c10::cuda::CUDAGuard guard(r.device());
  auto output = torch::empty_like(r);
  auto stream = at::cuda::getCurrentCUDAStream(r.get_device());
  native_fp16_recurrent_output_raw_kernel<<<B * H, N, 0, stream>>>(
      H,
      reinterpret_cast<const half*>(r.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(w_raw.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(k_raw.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(a.data_ptr<at::Half>()),
      reinterpret_cast<half*>(state.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(g.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(k_k.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(k_a.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(r_k.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(norm_weight.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(norm_bias.data_ptr<at::Half>()),
      elapsed.data_ptr<int>(),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      advance_elapsed,
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor rwkv7_native_fp16_sequence_cuda(
    torch::Tensor r, torch::Tensor w, torch::Tensor k, torch::Tensor v,
    torch::Tensor neg_kk, torch::Tensor kka, torch::Tensor state,
    torch::Tensor elapsed, torch::Tensor w0, bool add_w0) {
  check_half_cuda(r, "r");
  check_half_cuda(w, "w");
  check_half_cuda(k, "k");
  check_half_cuda(v, "v");
  check_half_cuda(neg_kk, "neg_kk");
  check_half_cuda(kka, "kka");
  check_half_cuda(state, "state");
  TORCH_CHECK(r.dim() == 4 && r.size(3) == N, "r must be [B,T,H,64]");
  TORCH_CHECK(state.dim() == 4 && state.size(2) == N && state.size(3) == N,
              "state must be [B,H,64,64]");
  const int B = static_cast<int>(r.size(0));
  const int T = static_cast<int>(r.size(1));
  const int H = static_cast<int>(r.size(2));
  const int C = H * N;
  TORCH_CHECK(T > 0, "sequence requires at least one token");
  TORCH_CHECK(state.size(0) == B && state.size(1) == H,
              "state batch/head shape does not match sequence");
  TORCH_CHECK(w.sizes() == r.sizes(), "w shape must match r");
  TORCH_CHECK(k.sizes() == r.sizes(), "k shape must match r");
  TORCH_CHECK(v.sizes() == r.sizes(), "v shape must match r");
  TORCH_CHECK(neg_kk.sizes() == r.sizes(), "neg_kk shape must match r");
  TORCH_CHECK(kka.sizes() == r.sizes(), "kka shape must match r");
  TORCH_CHECK(elapsed.is_cuda() && elapsed.scalar_type() == at::kInt
              && elapsed.is_contiguous() && elapsed.numel() == B,
              "elapsed must be contiguous CUDA int32 [B]");
  if (add_w0) {
    check_half_cuda(w0, "w0");
    TORCH_CHECK(w0.numel() == C, "w0 must have H*64 elements");
  }

  c10::cuda::CUDAGuard guard(r.device());
  auto output = torch::empty_like(r);
  auto stream = at::cuda::getCurrentCUDAStream(r.get_device());
  if (add_w0) {
    native_fp16_sequence_kernel<true><<<B * H, N, 0, stream>>>(
        T, C, H,
        reinterpret_cast<half*>(state.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(r.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(w.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(w0.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(neg_kk.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(kka.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        elapsed.data_ptr<int>());
  } else {
    native_fp16_sequence_kernel<false><<<B * H, N, 0, stream>>>(
        T, C, H,
        reinterpret_cast<half*>(state.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(r.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(w.data_ptr<at::Half>()),
        nullptr,
        reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(neg_kk.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(kka.data_ptr<at::Half>()),
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        elapsed.data_ptr<int>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
"""


_EXTENSION: Any | None = None
_EXTENSION_ERROR: str | None = None
_EXTENSION_LOCK = threading.Lock()


def native_fp16_recurrent_should_use(
    *,
    state_dtype: Any,
    input_dtype: Any,
    head_dim: int,
) -> bool:
    return bool(
        torch is not None
        and state_dtype == torch.float16
        and input_dtype == torch.float16
        and int(head_dim) == 64
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
            capability = torch.cuda.get_device_capability()
            with cuda_extension_build_environment(
                arch_list=f"{capability[0]}.{capability[1]}"
            ) as runtime_lib:
                from torch.utils.cpp_extension import load_inline

                extra_ldflags = (
                    [f"-Wl,-rpath,{runtime_lib}"]
                    if runtime_lib is not None
                    else []
                )
                _EXTENSION = load_inline(
                    name="rwkv7_native_wkv_fp16_v7",
                    cpp_sources=_CPP_SOURCE,
                    cuda_sources=_CUDA_SOURCE,
                    functions=None,
                    extra_cflags=["-O3"],
                    extra_cuda_cflags=[
                        "-O3",
                        # This path is promoted only behind model-level FP16-state
                        # logits, recurrent-state and greedy-token gates. Fast
                        # exp2 is the measured long-prefill speed win for this route.
                        "--use_fast_math",
                        "--extra-device-vectorization",
                    ],
                    extra_ldflags=extra_ldflags,
                    with_cuda=True,
                    verbose=os.environ.get(
                        "RWKV7_NATIVE_WKV_FP16_BUILD_VERBOSE", "0"
                    ).lower()
                    in {"1", "true", "yes", "on"},
                )
        except Exception as exc:  # pragma: no cover - host toolchain dependent
            _EXTENSION_ERROR = f"{type(exc).__name__}: {exc}"
            return None
    return _EXTENSION


def native_fp16_recurrent_available(*, build: bool = False) -> bool:
    if torch is None or not torch.cuda.is_available():
        return False
    return _load_extension() is not None if build else True


def native_fp16_recurrent_build_error() -> str | None:
    return _EXTENSION_ERROR


def native_fp16_recurrent_output_prepare_raw(
    r: Any,
    w_raw: Any,
    k_raw: Any,
    v: Any,
    a: Any,
    state: Any,
    g: Any,
    k_k: Any,
    k_a: Any,
    r_k: Any,
    norm_weight: Any,
    norm_bias: Any,
    elapsed: Any,
    *,
    advance_elapsed: bool,
    eps: float,
) -> Any:
    if torch is None:
        raise RuntimeError("native FP16 recurrence requires torch")
    if state.dim() != 4:
        raise ValueError("state must be [B,H,64,64]")
    if not native_fp16_recurrent_should_use(
        state_dtype=state.dtype,
        input_dtype=r.dtype,
        head_dim=int(state.shape[-1]),
    ):
        raise ValueError("native FP16 recurrence received an unsupported dtype or shape")
    extension = _load_extension()
    if extension is None:
        raise RuntimeError(
            "native FP16 recurrence extension is unavailable: "
            f"{native_fp16_recurrent_build_error()}"
        )
    tensors = (
        r,
        w_raw,
        k_raw,
        v,
        a,
        state,
        g,
        k_k,
        k_a,
        r_k,
        norm_weight,
        norm_bias,
    )
    if not all(
        item.is_cuda and item.dtype == torch.float16 and item.is_contiguous()
        for item in tensors
    ):
        raise ValueError("native FP16 recurrence requires contiguous CUDA fp16 tensors")
    if not (
        elapsed.is_cuda
        and elapsed.dtype == torch.int32
        and elapsed.is_contiguous()
    ):
        raise ValueError("elapsed must be a contiguous CUDA int32 tensor")
    return extension.recurrent_output_raw(
        r,
        w_raw,
        k_raw,
        v,
        a,
        state,
        g,
        k_k,
        k_a,
        r_k,
        norm_weight,
        norm_bias,
        elapsed,
        bool(advance_elapsed),
        float(eps),
    )


def native_fp16_sequence(
    r: Any,
    w: Any,
    k: Any,
    v: Any,
    neg_kk: Any,
    kka: Any,
    state: Any,
    elapsed: Any,
    *,
    w0: Any | None = None,
) -> Any:
    """Run the official-order FP16 recurrent scan over ``[B,T,H,64]``.

    ``w0`` is supplied separately for the official short-sequence path. Long
    prompts pass an already rounded ``w + w0`` tensor and leave it unset.
    State is updated in place and the returned sequence uses the same layout as
    the input projections.
    """

    if torch is None:
        raise RuntimeError("native FP16 sequence recurrence requires torch")
    if state.dim() != 4 or r.dim() != 4:
        raise ValueError("state and sequence tensors must be rank four")
    if not native_fp16_recurrent_should_use(
        state_dtype=state.dtype,
        input_dtype=r.dtype,
        head_dim=int(state.shape[-1]),
    ):
        raise ValueError("native FP16 sequence received an unsupported dtype or shape")
    extension = _load_extension()
    if extension is None:
        raise RuntimeError(
            "native FP16 sequence extension is unavailable: "
            f"{native_fp16_recurrent_build_error()}"
        )
    tensors = (r, w, k, v, neg_kk, kka, state)
    if not all(
        item.is_cuda and item.dtype == torch.float16 and item.is_contiguous()
        for item in tensors
    ):
        raise ValueError("native FP16 sequence requires contiguous CUDA fp16 tensors")
    if not (
        elapsed.is_cuda
        and elapsed.dtype == torch.int32
        and elapsed.is_contiguous()
        and int(elapsed.numel()) == int(r.shape[0])
    ):
        raise ValueError("elapsed must be a contiguous CUDA int32 [B] tensor")
    add_w0 = w0 is not None
    if w0 is None:
        w0 = torch.empty(0, device=r.device, dtype=torch.float16)
    elif not (w0.is_cuda and w0.dtype == torch.float16 and w0.is_contiguous()):
        raise ValueError("w0 must be contiguous CUDA fp16")
    return extension.sequence(
        r,
        w,
        k,
        v,
        neg_kk,
        kka,
        state,
        elapsed,
        w0,
        bool(add_w0),
    )


__all__ = [
    "native_fp16_recurrent_available",
    "native_fp16_recurrent_build_error",
    "native_fp16_recurrent_output_prepare_raw",
    "native_fp16_recurrent_should_use",
    "native_fp16_sequence",
]
