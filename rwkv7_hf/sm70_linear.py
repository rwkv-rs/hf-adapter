# coding=utf-8
"""Optional fp16 small-row linear kernel for NVIDIA sm_70.

cuBLAS is retained for normal matrix-matrix shapes.  sm_70 decode has a
different bottleneck at one or two rows: launch/setup overhead and GEMV memory
traffic dominate.  This extension assigns one warp to one output element and
accumulates fp16 input/weight products in fp32 without materializing any
duplicate weights.

The extension is compiled lazily, only on exact sm_70 devices and only when a
caller requests the route.  Every unsupported or failed-build case falls back
to ``torch.nn.functional.linear``.
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

torch::Tensor rwkv7_sm70_linear_cuda(torch::Tensor x, torch::Tensor weight, int64_t threads);
torch::Tensor rwkv7_sm70_linear_relu2_cuda(torch::Tensor x, torch::Tensor weight, int64_t threads);
torch::Tensor rwkv7_sm70_linear_add_cuda(
    torch::Tensor x, torch::Tensor weight, torch::Tensor residual, int64_t threads);
torch::Tensor rwkv7_sm70_rkv_cuda(
    torch::Tensor xr,
    torch::Tensor xk,
    torch::Tensor xv,
    torch::Tensor wr,
    torch::Tensor wk,
    torch::Tensor wv,
    int64_t threads);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear", &rwkv7_sm70_linear_cuda, "RWKV-7 sm70 small-row fp16 linear");
  m.def("linear_relu2", &rwkv7_sm70_linear_relu2_cuda, "RWKV-7 sm70 fp16 linear + relu2");
  m.def("linear_add", &rwkv7_sm70_linear_add_cuda, "RWKV-7 sm70 fp16 linear + residual");
  m.def("rkv", &rwkv7_sm70_rkv_cuda, "RWKV-7 sm70 grouped R/K/V fp16 linear");
}
"""


_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

namespace {

__inline__ __device__ float warp_sum(float value) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__global__ void linear_half_warp_kernel(
    const half* __restrict__ x,
    const half* __restrict__ weight,
    const half* __restrict__ residual,
    half* __restrict__ output,
    int batch,
    int outputs,
    int inputs,
    int mode) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  const int output_index = blockIdx.x * warps + warp;
  const int batch_index = blockIdx.y;
  if (output_index >= outputs || batch_index >= batch) {
    return;
  }

  const half* x_row = x + static_cast<int64_t>(batch_index) * inputs;
  const half* w_row = weight + static_cast<int64_t>(output_index) * inputs;
  const half2* x2 = reinterpret_cast<const half2*>(x_row);
  const half2* w2 = reinterpret_cast<const half2*>(w_row);
  const int pairs = inputs >> 1;
  float sum = 0.0f;
  for (int pair = lane; pair < pairs; pair += 32) {
    const float2 xv = __half22float2(x2[pair]);
    const float2 wv = __half22float2(w2[pair]);
    sum = fmaf(xv.x, wv.x, sum);
    sum = fmaf(xv.y, wv.y, sum);
  }
  if ((inputs & 1) != 0 && lane == 0) {
    sum = fmaf(__half2float(x_row[inputs - 1]), __half2float(w_row[inputs - 1]), sum);
  }
  sum = warp_sum(sum);
  if (lane == 0) {
    const int64_t index = static_cast<int64_t>(batch_index) * outputs + output_index;
    half value = __float2half_rn(sum);
    if (mode == 1) {
      const float rounded = fmaxf(__half2float(value), 0.0f);
      value = __float2half_rn(rounded * rounded);
    } else if (mode == 2) {
      value = __hadd(value, residual[index]);
    }
    output[index] = value;
  }
}

__global__ void rkv_half_warp_kernel(
    const half* __restrict__ xr,
    const half* __restrict__ xk,
    const half* __restrict__ xv,
    const half* __restrict__ wr,
    const half* __restrict__ wk,
    const half* __restrict__ wv,
    half* __restrict__ output,
    int batch,
    int hidden) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  const int output_index = blockIdx.x * warps + warp;
  const int matrix = blockIdx.y / batch;
  const int batch_index = blockIdx.y - matrix * batch;
  if (output_index >= hidden || matrix >= 3) {
    return;
  }

  const half* x_base = matrix == 0 ? xr : (matrix == 1 ? xk : xv);
  const half* w_base = matrix == 0 ? wr : (matrix == 1 ? wk : wv);
  const half* x_row = x_base + static_cast<int64_t>(batch_index) * hidden;
  const half* w_row = w_base + static_cast<int64_t>(output_index) * hidden;
  const half2* x2 = reinterpret_cast<const half2*>(x_row);
  const half2* w2 = reinterpret_cast<const half2*>(w_row);
  const int pairs = hidden >> 1;
  float sum = 0.0f;
  for (int pair = lane; pair < pairs; pair += 32) {
    const float2 xv2 = __half22float2(x2[pair]);
    const float2 wv2 = __half22float2(w2[pair]);
    sum = fmaf(xv2.x, wv2.x, sum);
    sum = fmaf(xv2.y, wv2.y, sum);
  }
  if ((hidden & 1) != 0 && lane == 0) {
    sum = fmaf(__half2float(x_row[hidden - 1]), __half2float(w_row[hidden - 1]), sum);
  }
  sum = warp_sum(sum);
  if (lane == 0) {
    const int64_t row = static_cast<int64_t>(matrix) * batch + batch_index;
    output[row * hidden + output_index] = __float2half_rn(sum);
  }
}

}  // namespace

torch::Tensor rwkv7_sm70_linear_cuda(torch::Tensor x, torch::Tensor weight, int64_t threads) {
  TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(x.scalar_type() == at::kHalf && weight.scalar_type() == at::kHalf, "fp16 tensors required");
  TORCH_CHECK(x.dim() == 2 && weight.dim() == 2, "x and weight must be rank-2");
  TORCH_CHECK(x.size(1) == weight.size(1), "linear input dimension mismatch");
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(), "contiguous tensors required");
  TORCH_CHECK(threads == 64 || threads == 128 || threads == 256, "threads must be 64, 128, or 256");
  TORCH_CHECK(x.size(0) <= 8, "sm70 small-row linear supports at most 8 rows");

  c10::cuda::CUDAGuard device_guard(x.device());
  auto output = torch::empty({x.size(0), weight.size(0)}, x.options());
  const int batch = static_cast<int>(x.size(0));
  const int outputs = static_cast<int>(weight.size(0));
  const int inputs = static_cast<int>(weight.size(1));
  const int warps = static_cast<int>(threads / 32);
  const dim3 grid((outputs + warps - 1) / warps, batch);
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  linear_half_warp_kernel<<<grid, static_cast<int>(threads), 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
      nullptr,
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      batch,
      outputs,
      inputs,
      0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor rwkv7_sm70_linear_relu2_cuda(torch::Tensor x, torch::Tensor weight, int64_t threads) {
  TORCH_CHECK(x.is_cuda() && weight.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(x.scalar_type() == at::kHalf && weight.scalar_type() == at::kHalf, "fp16 tensors required");
  TORCH_CHECK(x.dim() == 2 && weight.dim() == 2 && x.size(1) == weight.size(1), "shape mismatch");
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(), "contiguous tensors required");
  TORCH_CHECK(threads == 64 || threads == 128 || threads == 256, "threads must be 64, 128, or 256");
  TORCH_CHECK(x.size(0) == 1, "sm70 fused FFN up currently supports one row");
  c10::cuda::CUDAGuard device_guard(x.device());
  auto output = torch::empty({x.size(0), weight.size(0)}, x.options());
  const int batch = static_cast<int>(x.size(0));
  const int outputs = static_cast<int>(weight.size(0));
  const int inputs = static_cast<int>(weight.size(1));
  const int warps = static_cast<int>(threads / 32);
  const dim3 grid((outputs + warps - 1) / warps, batch);
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  linear_half_warp_kernel<<<grid, static_cast<int>(threads), 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
      nullptr,
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      batch,
      outputs,
      inputs,
      1);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor rwkv7_sm70_linear_add_cuda(
    torch::Tensor x, torch::Tensor weight, torch::Tensor residual, int64_t threads) {
  TORCH_CHECK(x.is_cuda() && weight.is_cuda() && residual.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(x.scalar_type() == at::kHalf && weight.scalar_type() == at::kHalf &&
              residual.scalar_type() == at::kHalf, "fp16 tensors required");
  TORCH_CHECK(x.dim() == 2 && weight.dim() == 2 && x.size(1) == weight.size(1), "shape mismatch");
  TORCH_CHECK(residual.dim() == 2 && residual.size(0) == x.size(0) &&
              residual.size(1) == weight.size(0), "residual shape mismatch");
  TORCH_CHECK(x.is_contiguous() && weight.is_contiguous() && residual.is_contiguous(),
              "contiguous tensors required");
  TORCH_CHECK(threads == 64 || threads == 128 || threads == 256, "threads must be 64, 128, or 256");
  TORCH_CHECK(x.size(0) <= 2, "sm70 fused FFN down currently supports at most two rows");
  c10::cuda::CUDAGuard device_guard(x.device());
  auto output = torch::empty({x.size(0), weight.size(0)}, x.options());
  const int batch = static_cast<int>(x.size(0));
  const int outputs = static_cast<int>(weight.size(0));
  const int inputs = static_cast<int>(weight.size(1));
  const int warps = static_cast<int>(threads / 32);
  const dim3 grid((outputs + warps - 1) / warps, batch);
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  linear_half_warp_kernel<<<grid, static_cast<int>(threads), 0, stream>>>(
      reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(residual.data_ptr<at::Half>()),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      batch,
      outputs,
      inputs,
      2);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor rwkv7_sm70_rkv_cuda(
    torch::Tensor xr,
    torch::Tensor xk,
    torch::Tensor xv,
    torch::Tensor wr,
    torch::Tensor wk,
    torch::Tensor wv,
    int64_t threads) {
  TORCH_CHECK(xr.is_cuda() && xk.is_cuda() && xv.is_cuda(), "CUDA activations required");
  TORCH_CHECK(wr.is_cuda() && wk.is_cuda() && wv.is_cuda(), "CUDA weights required");
  TORCH_CHECK(xr.scalar_type() == at::kHalf && xk.scalar_type() == at::kHalf && xv.scalar_type() == at::kHalf,
              "fp16 activations required");
  TORCH_CHECK(wr.scalar_type() == at::kHalf && wk.scalar_type() == at::kHalf && wv.scalar_type() == at::kHalf,
              "fp16 weights required");
  TORCH_CHECK(xr.dim() == 2 && xk.sizes() == xr.sizes() && xv.sizes() == xr.sizes(),
              "R/K/V activation shapes must match");
  TORCH_CHECK(wr.dim() == 2 && wr.size(0) == wr.size(1), "R weight must be square");
  TORCH_CHECK(wk.sizes() == wr.sizes() && wv.sizes() == wr.sizes(), "R/K/V weight shapes must match");
  TORCH_CHECK(xr.size(1) == wr.size(1), "R/K/V hidden size mismatch");
  TORCH_CHECK(xr.is_contiguous() && xk.is_contiguous() && xv.is_contiguous(), "contiguous activations required");
  TORCH_CHECK(wr.is_contiguous() && wk.is_contiguous() && wv.is_contiguous(), "contiguous weights required");
  TORCH_CHECK(threads == 64 || threads == 128 || threads == 256, "threads must be 64, 128, or 256");
  TORCH_CHECK(xr.size(0) <= 2, "sm70 grouped R/K/V supports at most 2 rows");

  c10::cuda::CUDAGuard device_guard(xr.device());
  const int batch = static_cast<int>(xr.size(0));
  const int hidden = static_cast<int>(xr.size(1));
  auto output = torch::empty({3, batch, hidden}, xr.options());
  const int warps = static_cast<int>(threads / 32);
  const dim3 grid((hidden + warps - 1) / warps, 3 * batch);
  auto stream = at::cuda::getCurrentCUDAStream(xr.get_device());
  rkv_half_warp_kernel<<<grid, static_cast<int>(threads), 0, stream>>>(
      reinterpret_cast<const half*>(xr.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(xk.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(xv.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(wr.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(wk.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(wv.data_ptr<at::Half>()),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      batch,
      hidden);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
"""


_EXTENSION: Any | None = None
_EXTENSION_ERROR: str | None = None
_EXTENSION_LOCK = threading.Lock()


def _is_sm70(device: Any = None) -> bool:
    if torch is None or not torch.cuda.is_available():
        return False
    try:
        resolved = torch.device("cuda" if device is None else device)
        index = torch.cuda.current_device() if resolved.index is None else int(resolved.index)
        return tuple(int(v) for v in torch.cuda.get_device_capability(index)) == (7, 0)
    except Exception:
        return False


def _load_extension() -> Any | None:
    global _EXTENSION, _EXTENSION_ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    if _EXTENSION_ERROR is not None or torch is None or not _is_sm70():
        return None
    with _EXTENSION_LOCK:
        if _EXTENSION is not None:
            return _EXTENSION
        if _EXTENSION_ERROR is not None:
            return None
        try:
            # A venv-installed ``ninja`` often lives beside the interpreter
            # even when non-interactive SSH does not prepend that directory.
            # Conda CUDA toolkits place ``nvcc`` there as well, so establish
            # CUDA_HOME before importing cpp_extension (it caches the value).
            python_bin = str(Path(sys.executable).resolve().parent)
            path_items = os.environ.get("PATH", "").split(os.pathsep)
            if python_bin not in path_items:
                os.environ["PATH"] = python_bin + os.pathsep + os.environ.get("PATH", "")
            nvcc = Path(python_bin) / "nvcc"
            if nvcc.exists() and "CUDA_HOME" not in os.environ:
                os.environ["CUDA_HOME"] = str(nvcc.parent.parent)
            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
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
                name="rwkv7_sm70_linear_v3",
                cpp_sources=_CPP_SOURCE,
                cuda_sources=_CUDA_SOURCE,
                functions=None,
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3", "--use_fast_math"],
                extra_ldflags=extra_ldflags,
                with_cuda=True,
                verbose=os.environ.get("RWKV7_SM70_LINEAR_BUILD_VERBOSE", "0") in {"1", "true", "yes", "on"},
            )
        except Exception as exc:  # pragma: no cover - depends on host toolchain
            _EXTENSION_ERROR = f"{type(exc).__name__}: {exc}"
            return None
    return _EXTENSION


def sm70_linear_available(device: Any = None, *, build: bool = False) -> bool:
    """Return route availability; ``build=True`` also verifies the toolchain."""

    if not _is_sm70(device):
        return False
    return _load_extension() is not None if build else True


def sm70_linear_build_error() -> str | None:
    return _EXTENSION_ERROR


def sm70_linear_should_use(rows: int, outputs: int, inputs: int, *, role: str) -> bool:
    """sm_70 route selected from measured bsz=1/2 shape sweeps.

    The rule intentionally rejects shapes where cuBLAS won.  ``hidden`` is
    used for R/K/V/O square projections; ``ffn_up`` and ``ffn_down`` are the
    two 4x FFN matrices; ``head`` is the full-vocabulary output projection.
    """

    rows, outputs, inputs = int(rows), int(outputs), int(inputs)
    if rows <= 0 or outputs <= 0 or inputs <= 0:
        return False
    if role == "head":
        return rows == 1
    if role == "hidden":
        if outputs != inputs:
            return False
        return rows == 1 or (rows == 2 and outputs == 2048)
    if role == "ffn_up":
        return rows == 1 and outputs == 4 * inputs
    if role == "ffn_down":
        if inputs != 4 * outputs:
            return False
        return (rows == 1 and (outputs <= 768 or outputs >= 2048)) or (
            rows == 2 and outputs <= 2048
        )
    return False


def sm70_linear_threads(rows: int, outputs: int, inputs: int, *, role: str) -> int:
    """Best balanced thread count from the same sm_70 shape sweep."""

    rows, outputs, inputs = int(rows), int(outputs), int(inputs)
    if role == "head":
        return 128
    if rows == 2:
        return 128 if outputs >= 2048 else 256
    hidden = min(outputs, inputs)
    if hidden >= 4096:
        return 64
    if hidden >= 2048:
        return 128
    return 256


def sm70_rkv_should_use(rows: int, hidden: int) -> bool:
    """Measured grouped-RKV route without stacked duplicate weights."""

    rows, hidden = int(rows), int(hidden)
    return rows == 1 or (rows == 2 and hidden <= 2048)


def sm70_rkv_threads(rows: int, hidden: int) -> int:
    rows, hidden = int(rows), int(hidden)
    if rows == 1:
        return 256 if hidden <= 768 else 64
    return 128 if hidden == 1024 else 64


def sm70_rkv(
    xr: Any,
    xk: Any,
    xv: Any,
    wr: Any,
    wk: Any,
    wv: Any,
    *,
    threads: int = 128,
    force_fallback: bool = False,
) -> tuple[Any, Any, Any]:
    """Project R/K/V in one launch while retaining the original weights."""

    if torch is None or F is None:
        raise RuntimeError("sm70_rkv requires torch")
    scalar = xr.dim() == 1
    activations = tuple(value.reshape(1, -1) if scalar else value for value in (xr, xk, xv))
    rows, hidden = int(activations[0].shape[0]), int(activations[0].shape[1])
    weights = (wr, wk, wv)
    valid = bool(
        not force_fallback
        and sm70_rkv_should_use(rows, hidden)
        and all(value.is_cuda and value.dtype == torch.float16 and value.is_contiguous() for value in activations)
        and all(
            value.is_cuda
            and value.dtype == torch.float16
            and value.is_contiguous()
            and tuple(value.shape) == (hidden, hidden)
            for value in weights
        )
        and _is_sm70(activations[0].device)
    )
    extension = _load_extension() if valid else None
    if extension is None:
        return F.linear(xr, wr), F.linear(xk, wk), F.linear(xv, wv)
    output = extension.rkv(*activations, *weights, int(threads))
    if scalar:
        return output[0, 0], output[1, 0], output[2, 0]
    return output[0], output[1], output[2]


def sm70_ffn_up_relu2_should_use(rows: int, outputs: int, inputs: int) -> bool:
    return int(rows) == 1 and int(outputs) == 4 * int(inputs)


def sm70_ffn_down_add_should_use(rows: int, outputs: int, inputs: int) -> bool:
    rows, outputs, inputs = int(rows), int(outputs), int(inputs)
    return inputs == 4 * outputs and (rows == 1 or (rows == 2 and outputs <= 2048))


def sm70_ffn_up_relu2(
    x: Any,
    weight: Any,
    *,
    threads: int = 128,
    force_fallback: bool = False,
) -> Any:
    """Fuse the FFN expansion projection with its ReLU-squared boundary."""

    if torch is None or F is None:
        raise RuntimeError("sm70_ffn_up_relu2 requires torch")
    scalar = x.dim() == 1
    x2 = x.reshape(1, -1) if scalar else x
    rows, inputs, outputs = int(x2.shape[0]), int(x2.shape[1]), int(weight.shape[0])
    valid = bool(
        not force_fallback
        and sm70_ffn_up_relu2_should_use(rows, outputs, inputs)
        and x2.is_cuda
        and weight.is_cuda
        and x2.dtype == torch.float16
        and weight.dtype == torch.float16
        and x2.is_contiguous()
        and weight.is_contiguous()
        and _is_sm70(x2.device)
    )
    extension = _load_extension() if valid else None
    if extension is None:
        return torch.relu(F.linear(x, weight)) ** 2
    output = extension.linear_relu2(x2, weight, int(threads))
    return output.reshape(outputs) if scalar else output


def sm70_ffn_down_add(
    x: Any,
    weight: Any,
    residual: Any,
    *,
    threads: int = 128,
    force_fallback: bool = False,
) -> Any:
    """Fuse the FFN contraction projection with its residual add."""

    if torch is None or F is None:
        raise RuntimeError("sm70_ffn_down_add requires torch")
    scalar = x.dim() == 1
    x2 = x.reshape(1, -1) if scalar else x
    residual2 = residual.reshape(1, -1) if scalar else residual
    rows, inputs, outputs = int(x2.shape[0]), int(x2.shape[1]), int(weight.shape[0])
    valid = bool(
        not force_fallback
        and sm70_ffn_down_add_should_use(rows, outputs, inputs)
        and x2.is_cuda
        and weight.is_cuda
        and residual2.is_cuda
        and x2.dtype == torch.float16
        and weight.dtype == torch.float16
        and residual2.dtype == torch.float16
        and x2.is_contiguous()
        and weight.is_contiguous()
        and residual2.is_contiguous()
        and tuple(residual2.shape) == (rows, outputs)
        and _is_sm70(x2.device)
    )
    extension = _load_extension() if valid else None
    if extension is None:
        return residual + F.linear(x, weight)
    output = extension.linear_add(x2, weight, residual2, int(threads))
    return output.reshape(outputs) if scalar else output


def sm70_linear(
    x: Any,
    weight: Any,
    *,
    threads: int = 128,
    force_fallback: bool = False,
) -> Any:
    """Apply bias-free ``F.linear`` through the sm_70 small-row route."""

    if torch is None or F is None:
        raise RuntimeError("sm70_linear requires torch")
    original_shape = tuple(x.shape)
    if x.dim() == 1:
        x2 = x.reshape(1, -1)
    elif x.dim() == 2:
        x2 = x
    else:
        raise ValueError(f"x must be [hidden] or [rows, hidden], got {original_shape}")
    valid = bool(
        not force_fallback
        and int(x2.shape[0]) <= 8
        and x2.is_cuda
        and weight.is_cuda
        and x2.dtype == torch.float16
        and weight.dtype == torch.float16
        and x2.is_contiguous()
        and weight.is_contiguous()
        and int(x2.shape[1]) == int(weight.shape[1])
        and _is_sm70(x2.device)
    )
    extension = _load_extension() if valid else None
    if extension is None:
        return F.linear(x, weight)
    output = extension.linear(x2, weight, int(threads))
    return output.reshape(int(weight.shape[0])) if len(original_shape) == 1 else output


__all__ = [
    "sm70_linear",
    "sm70_linear_available",
    "sm70_linear_build_error",
    "sm70_linear_should_use",
    "sm70_linear_threads",
    "sm70_ffn_down_add",
    "sm70_ffn_down_add_should_use",
    "sm70_ffn_up_relu2",
    "sm70_ffn_up_relu2_should_use",
    "sm70_rkv",
    "sm70_rkv_should_use",
    "sm70_rkv_threads",
]
