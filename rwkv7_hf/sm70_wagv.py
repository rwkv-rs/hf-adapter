# coding=utf-8
"""V100 grouped RWKV-7 W/A/G/V low-rank decode kernels.

The CUDA kernels are derived from Albatross faster3a (Apache-2.0).  This
self-contained lazy extension keeps HF checkpoints in their native layout and
fuses four low-rank projections without duplicate persistent weights.
"""
from __future__ import annotations
import os, sys, threading
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None


_CPP_SOURCE = r"""#include <torch/extension.h>
#include <vector>
std::vector<torch::Tensor> linear_wagv_rank_in_f16_cuda(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor);
std::vector<torch::Tensor> linear_wagv_rank_out_f16_cuda(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor);
torch::Tensor rwkv7_sm70_orig_linear_cuda(torch::Tensor,torch::Tensor);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("rank_in",&linear_wagv_rank_in_f16_cuda);m.def("rank_out",&linear_wagv_rank_out_f16_cuda);m.def("orig_linear",&rwkv7_sm70_orig_linear_cuda);}
"""
_CUDA_SOURCE = r"""#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <climits>
#include <vector>
using dtype = at::Half;
namespace {
inline int64_t ceil_div(int64_t n, int64_t d) { return (n+d-1)/d; }
__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
 for (int offset=16; offset>0; offset>>=1) x += __shfl_down_sync(0xffffffffu,x,offset);
 return x;
}
template <int Threads> __device__ __forceinline__ float block_sum_t(float x) {
 __shared__ float partial[Threads/32]; const int lane=threadIdx.x&31; const int warp=threadIdx.x>>5; x=warp_sum(x);
 if(lane==0) partial[warp]=x; __syncthreads(); x=(threadIdx.x<(Threads/32))?partial[lane]:0.0f;
 if(warp==0) x=warp_sum(x); if(threadIdx.x==0) partial[0]=x; __syncthreads(); return partial[0];
}
template <int Threads>
__global__ __launch_bounds__(Threads, 2) void linear_wagv_rank_in_f16_kernel(
    int M,
    int K,
    int Rw,
    int Ra,
    int Rg,
    int Rv,
    int Rmax,
    const dtype* __restrict__ xw,
    const dtype* __restrict__ xa,
    const dtype* __restrict__ xg,
    const dtype* __restrict__ xv,
    const dtype* __restrict__ w1_t,
    const dtype* __restrict__ a1_t,
    const dtype* __restrict__ g1_t,
    const dtype* __restrict__ v1_t,
    dtype* __restrict__ w1,
    dtype* __restrict__ a1,
    dtype* __restrict__ g1,
    dtype* __restrict__ v1) {
  const int r = blockIdx.x;
  const int m = blockIdx.y;
  const int group = blockIdx.z;
  int R = Rw;
  const dtype* x = xw;
  const dtype* wt = w1_t;
  dtype* y = w1;
  if (group == 1) {
    R = Ra;
    x = xa;
    wt = a1_t;
    y = a1;
  } else if (group == 2) {
    R = Rg;
    x = xg;
    wt = g1_t;
    y = g1;
  } else if (group == 3) {
    R = Rv;
    x = xv;
    wt = v1_t;
    y = v1;
  }
  if (m >= M || r >= R || r >= Rmax) {
    return;
  }
  float acc = 0.0f;
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  const dtype* w_row = wt + static_cast<int64_t>(r) * K;
  const int K2 = K >> 1;
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const int k = k2 << 1;
    const float2 xv2 = __half22float2(*reinterpret_cast<const __half2*>(x_row + k));
    const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(w_row + k));
    acc = fmaf(xv2.x, wv.x, acc);
    acc = fmaf(xv2.y, wv.y, acc);
  }
  if ((K & 1) && threadIdx.x == 0) {
    acc = fmaf(__half2float(*reinterpret_cast<const __half*>(x_row + K - 1)),
               __half2float(*reinterpret_cast<const __half*>(w_row + K - 1)),
               acc);
  }
  acc = block_sum_t<Threads>(acc);
  if (threadIdx.x == 0) {
    *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * R + r) = __float2half_rn(acc);
  }
}
template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 2) void linear_wagv_rank_out_f16_kernel(
    int M,
    int C,
    int Kw,
    int Ka,
    int Kg,
    int Kv,
    const dtype* __restrict__ w1,
    const dtype* __restrict__ a1,
    const dtype* __restrict__ g1,
    const dtype* __restrict__ v1,
    const dtype* __restrict__ w2_t,
    const dtype* __restrict__ a2_t,
    const dtype* __restrict__ g2_t,
    const dtype* __restrict__ v2_t,
    const dtype* __restrict__ v,
    const dtype* __restrict__ v_first,
    const dtype* __restrict__ v0,
    dtype* __restrict__ w,
    dtype* __restrict__ a,
    dtype* __restrict__ g,
    dtype* __restrict__ v_out) {
  const int n0 = blockIdx.x * OutTile;
  const int m = blockIdx.y;
  const int group = blockIdx.z;
  int K = Kw;
  const dtype* x = w1;
  const dtype* wt = w2_t;
  dtype* y = w;
  if (group == 1) {
    K = Ka;
    x = a1;
    wt = a2_t;
    y = a;
  } else if (group == 2) {
    K = Kg;
    x = g1;
    wt = g2_t;
    y = g;
  } else if (group == 3) {
    K = Kv;
    x = v1;
    wt = v2_t;
    y = v_out;
  }
  if (m >= M) {
    return;
  }
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  for (int k = threadIdx.x; k < K; k += Threads) {
    float xv = __half2float(*reinterpret_cast<const __half*>(x_row + k));
    if (group == 0) {
      xv = tanhf(xv);
    } else if (group == 2) {
      xv = 1.0f / (1.0f + expf(-xv));
    }
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < C) {
        acc[j] = fmaf(xv, __half2float(*reinterpret_cast<const __half*>(wt + static_cast<int64_t>(n) * K + k)), acc[j]);
      }
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = acc[j];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int u = 0; u < Threads / 32; ++u) {
        sum += partial[u][j];
      }
      const int n = n0 + j;
      if (n < C) {
        if (group == 3) {
          const int64_t idx = static_cast<int64_t>(m) * C + n;
          const float vv = __half2float(*reinterpret_cast<const __half*>(v + idx));
          const float vf = __half2float(*reinterpret_cast<const __half*>(v_first + idx));
          const float gate = 1.0f / (1.0f + expf(-(__half2float(*reinterpret_cast<const __half*>(v0 + n)) + sum)));
          *reinterpret_cast<__half*>(y + idx) = __float2half_rn(vv + (vf - vv) * gate);
        } else {
          *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * C + n) = __float2half_rn(sum);
        }
      }
    }
  }
}
template <int Threads, int RowTile, int OutTile>
__global__ __launch_bounds__(Threads, 1) void linear_orig_rows_f16_kernel(
    int M,
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_orig,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int m0 = blockIdx.y * RowTile;
  float acc[RowTile][OutTile];
#pragma unroll
  for (int r = 0; r < RowTile; ++r) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      acc[r][j] = 0.0f;
    }
  }
  const int K2 = K >> 1;
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const int k = k2 << 1;
    float2 wv[OutTile];
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      wv[j] = (n < N)
          ? __half22float2(*reinterpret_cast<const __half2*>(weight_orig + static_cast<int64_t>(n) * K + k))
          : make_float2(0.0f, 0.0f);
    }
#pragma unroll
    for (int r = 0; r < RowTile; ++r) {
      const int m = m0 + r;
      if (m < M) {
        const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(x + static_cast<int64_t>(m) * K + k));
#pragma unroll
        for (int j = 0; j < OutTile; ++j) {
          acc[r][j] = fmaf(xv.x, wv[j].x, acc[r][j]);
          acc[r][j] = fmaf(xv.y, wv[j].y, acc[r][j]);
        }
      }
    }
  }
  if ((K & 1) && threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        const float wv = __half2float(*reinterpret_cast<const __half*>(weight_orig + static_cast<int64_t>(n) * K + K - 1));
#pragma unroll
        for (int r = 0; r < RowTile; ++r) {
          const int m = m0 + r;
          if (m < M) {
            const float xv = __half2float(*reinterpret_cast<const __half*>(x + static_cast<int64_t>(m) * K + K - 1));
            acc[r][j] = fmaf(xv, wv, acc[r][j]);
          }
        }
      }
    }
  }
  __shared__ float partial[Threads / 32][RowTile][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int r = 0; r < RowTile; ++r) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const float v = warp_sum(acc[r][j]);
      if (lane == 0) {
        partial[warp][r][j] = v;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int r = 0; r < RowTile; ++r) {
      const int m = m0 + r;
      if (m < M) {
#pragma unroll
        for (int j = 0; j < OutTile; ++j) {
          const int n = n0 + j;
          if (n < N) {
            float sum = 0.0f;
#pragma unroll
            for (int w = 0; w < Threads / 32; ++w) {
              sum += partial[w][r][j];
            }
            *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * N + n) = __float2half_rn(sum);
          }
        }
      }
    }
  }
}
template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 1) void linear_orig_row2_exact4_f16_kernel(
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_orig,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  float acc0[OutTile];
  float acc1[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc0[j] = 0.0f;
    acc1[j] = 0.0f;
  }
  for (int k = threadIdx.x << 2; k < K; k += Threads << 2) {
    const float2 x00 = __half22float2(*reinterpret_cast<const __half2*>(x + k));
    const float2 x01 = __half22float2(*reinterpret_cast<const __half2*>(x + k + 2));
    const float2 x10 = __half22float2(*reinterpret_cast<const __half2*>(x + K + k));
    const float2 x11 = __half22float2(*reinterpret_cast<const __half2*>(x + K + k + 2));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const dtype* wj = weight_orig + static_cast<int64_t>(n0 + j) * K + k;
      const float2 w0 = __half22float2(*reinterpret_cast<const __half2*>(wj));
      const float2 w1 = __half22float2(*reinterpret_cast<const __half2*>(wj + 2));
      acc0[j] = fmaf(x00.x, w0.x, acc0[j]);
      acc0[j] = fmaf(x00.y, w0.y, acc0[j]);
      acc0[j] = fmaf(x01.x, w1.x, acc0[j]);
      acc0[j] = fmaf(x01.y, w1.y, acc0[j]);
      acc1[j] = fmaf(x10.x, w0.x, acc1[j]);
      acc1[j] = fmaf(x10.y, w0.y, acc1[j]);
      acc1[j] = fmaf(x11.x, w1.x, acc1[j]);
      acc1[j] = fmaf(x11.y, w1.y, acc1[j]);
    }
  }
  __shared__ float partial[Threads / 32][2][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    const float v0 = warp_sum(acc0[j]);
    const float v1 = warp_sum(acc1[j]);
    if (lane == 0) {
      partial[warp][0][j] = v0;
      partial[warp][1][j] = v1;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum0 = 0.0f;
      float sum1 = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum0 += partial[w][0][j];
        sum1 += partial[w][1][j];
      }
      const int n = n0 + j;
      y[n] = __float2half_rn(sum0);
      y[N + n] = __float2half_rn(sum1);
    }
  }
}
}
std::vector<at::Tensor> linear_wagv_rank_in_f16_cuda(
    at::Tensor xw,
    at::Tensor xa,
    at::Tensor xg,
    at::Tensor xv,
    at::Tensor w1_t,
    at::Tensor a1_t,
    at::Tensor g1_t,
    at::Tensor v1_t) {
  const int64_t k64 = xw.size(-1);
  const int64_t rw64 = w1_t.size(0);
  const int64_t ra64 = a1_t.size(0);
  const int64_t rg64 = g1_t.size(0);
  const int64_t rv64 = v1_t.size(0);
  const int64_t m64 = xw.numel() / k64;
  TORCH_CHECK(k64 <= INT_MAX && rw64 <= INT_MAX && ra64 <= INT_MAX && rg64 <= INT_MAX && rv64 <= INT_MAX && m64 <= INT_MAX,
              "linear_wagv_rank_in_f16 shape too large");
  const int K = static_cast<int>(k64);
  const int Rw = static_cast<int>(rw64);
  const int Ra = static_cast<int>(ra64);
  const int Rg = static_cast<int>(rg64);
  const int Rv = static_cast<int>(rv64);
  const int Rmax = std::max(std::max(Rw, Ra), std::max(Rg, Rv));
  const int M = static_cast<int>(m64);
  TORCH_CHECK(K >= 1024 && Rmax <= 512 && M <= 8, "linear_wagv_rank_in_f16 supports only K>=1024,R<=512,M<=8");
  std::vector<int64_t> w_sizes(xw.sizes().begin(), xw.sizes().end());
  std::vector<int64_t> a_sizes = w_sizes;
  std::vector<int64_t> g_sizes = w_sizes;
  std::vector<int64_t> v_sizes = w_sizes;
  w_sizes.back() = rw64;
  a_sizes.back() = ra64;
  g_sizes.back() = rg64;
  v_sizes.back() = rv64;
  auto w1 = at::empty(w_sizes, xw.options());
  auto a1 = at::empty(a_sizes, xw.options());
  auto g1 = at::empty(g_sizes, xw.options());
  auto v1 = at::empty(v_sizes, xw.options());
  if (M == 0 || K == 0 || Rmax == 0) {
    return {w1, a1, g1, v1};
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  linear_wagv_rank_in_f16_kernel<256><<<dim3(Rmax, M, 4), 256, 0, stream>>>(
      M, K, Rw, Ra, Rg, Rv, Rmax,
      xw.data_ptr<dtype>(), xa.data_ptr<dtype>(), xg.data_ptr<dtype>(), xv.data_ptr<dtype>(),
      w1_t.data_ptr<dtype>(), a1_t.data_ptr<dtype>(), g1_t.data_ptr<dtype>(), v1_t.data_ptr<dtype>(),
      w1.data_ptr<dtype>(), a1.data_ptr<dtype>(), g1.data_ptr<dtype>(), v1.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {w1, a1, g1, v1};
}
std::vector<at::Tensor> linear_wagv_rank_out_f16_cuda(
    at::Tensor w1,
    at::Tensor a1,
    at::Tensor g1,
    at::Tensor v1,
    at::Tensor w2_t,
    at::Tensor a2_t,
    at::Tensor g2_t,
    at::Tensor v2_t,
    at::Tensor v,
    at::Tensor v_first,
    at::Tensor v0) {
  const int64_t kw64 = w1.size(-1);
  const int64_t ka64 = a1.size(-1);
  const int64_t kg64 = g1.size(-1);
  const int64_t kv64 = v1.size(-1);
  const int64_t c64 = w2_t.size(0);
  const int64_t m64 = w1.numel() / kw64;
  TORCH_CHECK(kw64 <= INT_MAX && ka64 <= INT_MAX && kg64 <= INT_MAX && kv64 <= INT_MAX && c64 <= INT_MAX && m64 <= INT_MAX,
              "linear_wagv_rank_out_f16 shape too large");
  const int Kw = static_cast<int>(kw64);
  const int Ka = static_cast<int>(ka64);
  const int Kg = static_cast<int>(kg64);
  const int Kv = static_cast<int>(kv64);
  const int C = static_cast<int>(c64);
  const int M = static_cast<int>(m64);
  TORCH_CHECK(Kw <= 512 && Ka <= 512 && Kg <= 512 && Kv <= 512 && C >= 1024 && M <= 4,
              "linear_wagv_rank_out_f16 supports only small-rank M<=4");
  std::vector<int64_t> out_sizes(w1.sizes().begin(), w1.sizes().end());
  out_sizes.back() = c64;
  auto w = at::empty(out_sizes, w1.options());
  auto a = at::empty(out_sizes, w1.options());
  auto g = at::empty(out_sizes, w1.options());
  auto v_out = at::empty(out_sizes, w1.options());
  if (M == 0 || C == 0 || Kw == 0 || Ka == 0 || Kg == 0 || Kv == 0) {
    return {w, a, g, v_out};
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  if (M == 1) {
    linear_wagv_rank_out_f16_kernel<128, 4><<<dim3(ceil_div(C, 4), M, 4), 128, 0, stream>>>(
        M, C, Kw, Ka, Kg, Kv,
        w1.data_ptr<dtype>(), a1.data_ptr<dtype>(), g1.data_ptr<dtype>(), v1.data_ptr<dtype>(),
        w2_t.data_ptr<dtype>(), a2_t.data_ptr<dtype>(), g2_t.data_ptr<dtype>(), v2_t.data_ptr<dtype>(),
        v.data_ptr<dtype>(), v_first.data_ptr<dtype>(), v0.data_ptr<dtype>(),
        w.data_ptr<dtype>(), a.data_ptr<dtype>(), g.data_ptr<dtype>(), v_out.data_ptr<dtype>());
  } else {
    linear_wagv_rank_out_f16_kernel<128, 4><<<dim3(ceil_div(C, 4), M, 4), 128, 0, stream>>>(
        M, C, Kw, Ka, Kg, Kv,
        w1.data_ptr<dtype>(), a1.data_ptr<dtype>(), g1.data_ptr<dtype>(), v1.data_ptr<dtype>(),
        w2_t.data_ptr<dtype>(), a2_t.data_ptr<dtype>(), g2_t.data_ptr<dtype>(), v2_t.data_ptr<dtype>(),
        v.data_ptr<dtype>(), v_first.data_ptr<dtype>(), v0.data_ptr<dtype>(),
        w.data_ptr<dtype>(), a.data_ptr<dtype>(), g.data_ptr<dtype>(), v_out.data_ptr<dtype>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {w, a, g, v_out};
}
at::Tensor rwkv7_sm70_orig_linear_cuda(at::Tensor x, at::Tensor weight) {
 const int K=static_cast<int>(x.size(-1)); const int N=static_cast<int>(weight.size(0)); const int rows=static_cast<int>(x.numel()/x.size(-1));
 std::vector<int64_t> sizes(x.sizes().begin(),x.sizes().end()); sizes.back()=N; auto y=at::empty(sizes,x.options()); auto stream=at::cuda::getCurrentCUDAStream();
 if(rows==2) linear_orig_row2_exact4_f16_kernel<64,2><<<N/2,64,0,stream>>>(K,N,x.data_ptr<dtype>(),weight.data_ptr<dtype>(),y.data_ptr<dtype>());
 else if(rows==4) linear_orig_rows_f16_kernel<128,4,2><<<dim3(ceil_div(N,2),1,1),128,0,stream>>>(rows,K,N,x.data_ptr<dtype>(),weight.data_ptr<dtype>(),y.data_ptr<dtype>());
 else TORCH_CHECK(false,"sm70 orig linear supports rows 2 or 4"); C10_CUDA_KERNEL_LAUNCH_CHECK(); return y;
}
"""
_EXTENSION = None
_EXTENSION_ERROR = None
_LOCK = threading.Lock()


def _is_sm70(device=None):
    if torch is None or not torch.cuda.is_available():
        return False
    try:
        d = torch.device("cuda" if device is None else device)
        i = torch.cuda.current_device() if d.index is None else int(d.index)
        return tuple(torch.cuda.get_device_capability(i)) == (7, 0)
    except Exception:
        return False


def _load_extension():
    global _EXTENSION, _EXTENSION_ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    if _EXTENSION_ERROR is not None or not _is_sm70():
        return None
    with _LOCK:
        if _EXTENSION is not None:
            return _EXTENSION
        try:
            pb = str(Path(sys.executable).resolve().parent)
            os.environ["PATH"] = pb + os.pathsep + os.environ.get("PATH", "")
            nv = Path(pb) / "nvcc"
            if nv.exists():
                os.environ.setdefault("CUDA_HOME", str(nv.parent.parent))
            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
            rt = (
                Path(sys.prefix)
                / "lib"
                / f"python{sys.version_info.major}.{sys.version_info.minor}"
                / "site-packages"
                / "nvidia"
                / "cuda_runtime"
                / "lib"
            )
            ld = [f"-L{rt}", f"-Wl,-rpath,{rt}"] if rt.is_dir() else []
            from torch.utils.cpp_extension import load_inline

            _EXTENSION = load_inline(
                name="rwkv7_sm70_wagv_v3",
                cpp_sources=_CPP_SOURCE,
                cuda_sources=_CUDA_SOURCE,
                functions=None,
                extra_cflags=["-O3"],
                extra_cuda_cflags=[
                    "-O3",
                    "--use_fast_math",
                    "--extra-device-vectorization",
                ],
                extra_ldflags=ld,
                with_cuda=True,
                verbose=False,
            )
        except Exception as e:
            _EXTENSION_ERROR = f"{type(e).__name__}: {e}"
    return _EXTENSION


def sm70_wagv_lora_available(device=None, *, build=False):
    if not _is_sm70(device):
        return False
    return _load_extension() is not None if build else True


def sm70_wagv_lora_build_error():
    return _EXTENSION_ERROR


def sm70_orig_linear(x, weight):
    ext = _load_extension()
    rows = 1 if x.dim() == 1 else int(x.shape[0])
    if ext is None or rows not in {2, 4} or int(x.shape[-1]) < 2048:
        return F.linear(x, weight)
    return ext.orig_linear(x, weight)


def sm70_orig_rkv(xr, xk, xv, wr, wk, wv):
    return tuple(sm70_orig_linear(x, w) for x, w in zip((xr, xk, xv), (wr, wk, wv)))


def sm70_wagv_lora(
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
    *,
    force_fallback=False,
):
    scalar = xw.dim() == 1
    xs = [q.reshape(1, -1) if scalar else q for q in (xw, xa, xg, xv)]
    rows, hidden = xs[0].shape
    valid = (
        not force_fallback
        and not torch.is_grad_enabled()
        and rows <= 4
        and hidden >= 1024
        and _is_sm70(xs[0].device)
        and all(
            q.is_cuda and q.dtype == torch.float16 and q.is_contiguous()
            for q in xs + ([w1, a1, g1, v1, w2, a2, g2, v2, v, v_first])
        )
    )
    ext = _load_extension() if valid else None
    if ext is None:
        ww = F.linear(torch.tanh(F.linear(xs[0], w1)), w2, w0)
        aa = F.linear(F.linear(xs[1], a1), a2, a0)
        gg = F.linear(torch.sigmoid(F.linear(xs[2], g1)), g2)
        gate = torch.sigmoid(F.linear(F.linear(xs[3], v1), v2, v0))
        vv = (
            v.reshape(rows, hidden)
            + (v_first.reshape(rows, hidden) - v.reshape(rows, hidden)) * gate
        )
        out = (ww, aa, gg, vv)
    else:
        mids = ext.rank_in(*xs, w1, a1, g1, v1)
        out = list(
            ext.rank_out(
                *mids,
                w2,
                a2,
                g2,
                v2,
                v.reshape(rows, hidden),
                v_first.reshape(rows, hidden),
                v0,
            )
        )
        out[0] = out[0] + w0
        out[1] = out[1] + a0
        out = tuple(out)
    return tuple(q.reshape(hidden) for q in out) if scalar else out


__all__ = [
    "sm70_orig_linear",
    "sm70_orig_rkv",
    "sm70_wagv_lora",
    "sm70_wagv_lora_available",
    "sm70_wagv_lora_build_error",
]
