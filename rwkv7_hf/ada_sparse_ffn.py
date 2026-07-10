# coding=utf-8
"""Optional Ada fp16 sparse FFN contraction for small-row decode.

RWKV-7 applies ``ReLU(key(x)) ** 2`` before the FFN value projection.  At
decode batch sizes the activation is naturally sparse, so reading only the
positive rows of a packed ``[ffn, hidden]`` value matrix is faster than a
dense GEMM on RTX 4090.  The CUDA kernel is derived from Albatross' Apache-2.0
``cmix_sparse_spmv_relu_rows_kernel`` and adds the residual while initializing
the output, avoiding a separate residual-add launch.

The extension is deliberately narrow: fp16, exact sm_89, at most 19 rows, and
the normal RWKV ``ffn == 4 * hidden`` shape.  Unsupported shapes, training,
and build failures retain the ordinary PyTorch implementation.  Value weights
are transposed once and cached; callers can prewarm the cache before CUDA graph
capture with :func:`ada_sparse_ffn_pack_weight`.
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


_CPP_SOURCE = r"""
#include <torch/extension.h>

torch::Tensor rwkv7_ada_sparse_ffn_cuda(
    torch::Tensor preact, torch::Tensor packed_value, torch::Tensor residual);
torch::Tensor rwkv7_ada_linear_cuda(torch::Tensor x, torch::Tensor weight);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sparse_down_add", &rwkv7_ada_sparse_ffn_cuda,
        "RWKV-7 Ada sparse ReLU2 FFN down projection + residual");
  m.def("ffn_up", &rwkv7_ada_linear_cuda,
        "RWKV-7 Ada small-row FFN expansion projection");
  m.def("linear", &rwkv7_ada_linear_cuda,
        "RWKV-7 Ada small-row fp16 linear");
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

}  // namespace

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
              "Ada linear supports one, two, or four rows");
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
    TORCH_CHECK(false, "four-row Ada linear supports square or 4x expansion shapes");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor rwkv7_ada_sparse_ffn_cuda(
    torch::Tensor preact,
    torch::Tensor packed_value,
    torch::Tensor residual) {
  TORCH_CHECK(preact.is_cuda() && packed_value.is_cuda() && residual.is_cuda(),
              "CUDA tensors required");
  TORCH_CHECK(preact.scalar_type() == at::kHalf &&
              packed_value.scalar_type() == at::kHalf &&
              residual.scalar_type() == at::kHalf, "fp16 tensors required");
  TORCH_CHECK(preact.dim() == 2 && packed_value.dim() == 2 && residual.dim() == 2,
              "preact, packed_value, and residual must be rank-2");
  TORCH_CHECK(preact.is_contiguous() && packed_value.is_contiguous() && residual.is_contiguous(),
              "contiguous tensors required");
  const int64_t rows = preact.size(0);
  const int64_t ffn = preact.size(1);
  const int64_t hidden = residual.size(1);
  TORCH_CHECK(rows >= 1 && rows <= 19, "sparse FFN supports 1..19 rows");
  TORCH_CHECK(residual.size(0) == rows, "residual row mismatch");
  TORCH_CHECK(packed_value.size(0) == ffn && packed_value.size(1) == hidden,
              "packed value weight must have shape [ffn, hidden]");
  TORCH_CHECK(ffn == 4 * hidden, "expected RWKV ffn == 4 * hidden");
  TORCH_CHECK((ffn % FFN_TILE) == 0 && (hidden % (2 * THREADS)) == 0,
              "ffn must be divisible by 128 and hidden by 256");

  c10::cuda::CUDAGuard device_guard(preact.device());
  auto output = torch::empty_like(residual);
  auto stream = at::cuda::getCurrentCUDAStream(preact.get_device());
  const int64_t vec4_count = output.numel() / 8;
  copy_residual_vec4_kernel<<<static_cast<int>((vec4_count + 127) / 128), 128, 0, stream>>>(
      reinterpret_cast<const half*>(residual.data_ptr<at::Half>()),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      vec4_count);
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
"""


_EXTENSION: Any | None = None
_EXTENSION_ERROR: str | None = None
_EXTENSION_LOCK = threading.Lock()
_PACK_LOCK = threading.Lock()
_PACKED_WEIGHTS: dict[tuple[Any, ...], tuple[weakref.ReferenceType[Any], Any]] = {}


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
            path_items = os.environ.get("PATH", "").split(os.pathsep)
            if python_bin not in path_items:
                os.environ["PATH"] = python_bin + os.pathsep + os.environ.get("PATH", "")
            nvcc = Path(python_bin) / "nvcc"
            if nvcc.exists() and "CUDA_HOME" not in os.environ:
                os.environ["CUDA_HOME"] = str(nvcc.parent.parent)
            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
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
                name="rwkv7_ada_sparse_ffn_v5",
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
    """Return whether a shape is in the measured RTX 4090 sparse decode set."""

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
    if not _is_sm89(device):
        return False
    return _load_extension() is not None if build else True


def ada_sparse_ffn_build_error() -> str | None:
    return _EXTENSION_ERROR


def _weight_cache_key(weight: Any) -> tuple[Any, ...]:
    index = weight.device.index
    if index is None and torch is not None:
        index = torch.cuda.current_device()
    try:
        version = int(weight._version)
    except RuntimeError:
        # Tensors constructed under inference_mode intentionally have no
        # version counter.  Their storage is immutable for this route.
        version = -1
    return (
        int(index),
        int(weight.data_ptr()),
        version,
        tuple(int(v) for v in weight.shape),
        weight.dtype,
    )


def ada_sparse_ffn_pack_weight(weight: Any) -> Any:
    """Return a cached contiguous ``[ffn, hidden]`` inference layout."""

    key = _weight_cache_key(weight)
    cached = _PACKED_WEIGHTS.get(key)
    if cached is not None and cached[0]() is weight:
        return cached[1]
    with _PACK_LOCK:
        cached = _PACKED_WEIGHTS.get(key)
        if cached is not None and cached[0]() is weight:
            return cached[1]
        packed = weight.transpose(0, 1).contiguous()
        _PACKED_WEIGHTS[key] = (weakref.ref(weight), packed)
        stale = [item for item, value in _PACKED_WEIGHTS.items() if value[0]() is None]
        for item in stale:
            _PACKED_WEIGHTS.pop(item, None)
        return packed


def clear_ada_sparse_ffn_weight_cache() -> None:
    with _PACK_LOCK:
        _PACKED_WEIGHTS.clear()


def ada_sparse_ffn_down_add(
    preact: Any,
    weight: Any,
    residual: Any,
    *,
    force_fallback: bool = False,
) -> Any:
    """Apply sparse ``ReLU²`` contraction and residual add on Ada decode."""

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
        and weight.is_contiguous()
        and residual2.is_contiguous()
        and tuple(weight.shape) == (outputs, inputs)
        and tuple(residual2.shape) == (rows, outputs)
        and _is_sm89(preact2.device)
    )
    extension = _load_extension() if valid else None
    if extension is None:
        return residual + F.linear(torch.relu(preact) ** 2, weight)
    packed = ada_sparse_ffn_pack_weight(weight)
    output = extension.sparse_down_add(preact2, packed, residual2)
    return output.reshape(outputs) if scalar else output


def ada_ffn_up(x: Any, weight: Any, *, force_fallback: bool = False) -> Any:
    """Apply the measured no-copy small-row FFN expansion on RTX 4090."""

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
        and _is_sm89(x2.device)
    )
    extension = _load_extension() if valid else None
    if extension is None:
        return F.linear(x, weight)
    output = extension.ffn_up(x2, weight)
    return output.reshape(outputs) if scalar else output


def ada_linear(x: Any, weight: Any, *, force_fallback: bool = False) -> Any:
    """Apply the no-copy exact-row Ada linear probe with a torch fallback."""

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
        and _is_sm89(x2.device)
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
    "ada_sparse_ffn_should_use",
    "clear_ada_sparse_ffn_weight_cache",
]
