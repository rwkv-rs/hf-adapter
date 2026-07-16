#!/usr/bin/env python3
"""Measure explicit CUDA block-N (BN) and thread-N (TN) W8/W4 decode tiles.

This is deliberately a handwritten-CUDA probe.  Triton exposes ``BLOCK_N``
and launch warps, but its physical per-thread MMA tile is compiler controlled;
calling a warp sweep ``TN`` would be misleading.  The kernels below assign
exactly ``TN`` output columns to each thread and exactly ``BN`` columns to each
block, so the two dimensions can be measured independently before a winner is
considered for production dispatch.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rwkv7_hf.bn_tn_tuning import bn_tn_candidates, select_best_bn_tn
from rwkv7_hf.native_quant_mm4 import mm4_matmul_triton, quantize_mm4
from rwkv7_hf.native_quant_mm8 import mm8_matmul_triton, quantize_mm8


CPP_SOURCE = r"""
#include <torch/extension.h>
torch::Tensor rwkv7_bn_tn_mm8_cuda(torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t);
torch::Tensor rwkv7_bn_tn_mm4_cuda(torch::Tensor, torch::Tensor, torch::Tensor,
    torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mm8", &rwkv7_bn_tn_mm8_cuda);
  m.def("mm4", &rwkv7_bn_tn_mm4_cuda);
}
"""


CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace {

template<int BN, int TN>
__global__ void mm8_bn_tn_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ q,
    const half* __restrict__ mx,
    const half* __restrict__ rx,
    const half* __restrict__ my,
    const half* __restrict__ ry,
    half* __restrict__ y,
    int B, int K, int N) {
  constexpr int THREADS = BN / TN;
  const int b = blockIdx.y;
  const int tile = blockIdx.x * BN;
  const int lane = threadIdx.x;
  float acc[TN] = {0.0f};
  float sum_x = 0.0f;
  float sum_x_my = 0.0f;
  const half* x_row = x + static_cast<int64_t>(b) * K;

  #pragma unroll 4
  for (int k = 0; k < K; ++k) {
    const float xv = __half2float(x_row[k]);
    const float xry = xv * __half2float(ry[k]);
    sum_x += xv;
    sum_x_my = fmaf(xv, __half2float(my[k]), sum_x_my);
    #pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int n = tile + lane + j * THREADS;
      if (n < N) {
        const float w = static_cast<float>(q[static_cast<int64_t>(k) * N + n]) + 0.5f;
        acc[j] = fmaf(xry, w, acc[j]);
      }
    }
  }
  #pragma unroll
  for (int j = 0; j < TN; ++j) {
    const int n = tile + lane + j * THREADS;
    if (n < N) {
      const float out = acc[j] * __half2float(rx[n]) + sum_x_my + sum_x * __half2float(mx[n]);
      y[static_cast<int64_t>(b) * N + n] = __float2half_rn(out);
    }
  }
}

template<int BN, int TN>
__global__ void mm4_bn_tn_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ q,
    const half* __restrict__ mx,
    const half* __restrict__ rx,
    const half* __restrict__ my,
    const half* __restrict__ ry,
    half* __restrict__ y,
    int B, int K, int N, int NP) {
  constexpr int THREADS = BN / TN;
  const int b = blockIdx.y;
  const int tile = blockIdx.x * BN;
  const int lane = threadIdx.x;
  float acc[TN] = {0.0f};
  float sum_x = 0.0f;
  float sum_x_my = 0.0f;
  const half* x_row = x + static_cast<int64_t>(b) * K;

  #pragma unroll 4
  for (int k = 0; k < K; ++k) {
    const float xv = __half2float(x_row[k]);
    const float xry = xv * __half2float(ry[k]);
    sum_x += xv;
    sum_x_my = fmaf(xv, __half2float(my[k]), sum_x_my);
    #pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int n = tile + lane + j * THREADS;
      if (n < N) {
        const uint8_t packed = q[static_cast<int64_t>(k) * NP + (n >> 1)];
        const float w = static_cast<float>((n & 1) ? (packed >> 4) : (packed & 15)) + 0.5f;
        acc[j] = fmaf(xry, w, acc[j]);
      }
    }
  }
  #pragma unroll
  for (int j = 0; j < TN; ++j) {
    const int n = tile + lane + j * THREADS;
    if (n < N) {
      const float out = acc[j] * __half2float(rx[n]) + sum_x_my + sum_x * __half2float(mx[n]);
      y[static_cast<int64_t>(b) * N + n] = __float2half_rn(out);
    }
  }
}

void check_common(torch::Tensor x, torch::Tensor q, torch::Tensor mx,
                  torch::Tensor rx, torch::Tensor my, torch::Tensor ry,
                  int64_t bn, int64_t tn) {
  TORCH_CHECK(x.is_cuda() && q.is_cuda() && mx.is_cuda() && rx.is_cuda() && my.is_cuda() && ry.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(x.scalar_type() == at::kHalf && mx.scalar_type() == at::kHalf && rx.scalar_type() == at::kHalf && my.scalar_type() == at::kHalf && ry.scalar_type() == at::kHalf, "fp16 activation and scales required");
  TORCH_CHECK(q.scalar_type() == at::kByte, "uint8 quantized weights required");
  TORCH_CHECK(x.dim() == 2 && q.dim() == 2, "rank-2 x and weight required");
  TORCH_CHECK(x.is_contiguous() && q.is_contiguous() && mx.is_contiguous() && rx.is_contiguous() && my.is_contiguous() && ry.is_contiguous(), "contiguous tensors required");
  TORCH_CHECK(x.size(1) == q.size(0), "input/weight K mismatch");
  TORCH_CHECK(bn > 0 && tn > 0 && bn % tn == 0, "invalid BN/TN");
}

}  // namespace

torch::Tensor rwkv7_bn_tn_mm8_cuda(
    torch::Tensor x, torch::Tensor q, torch::Tensor mx, torch::Tensor rx,
    torch::Tensor my, torch::Tensor ry, int64_t bn, int64_t tn) {
  check_common(x, q, mx, rx, my, ry, bn, tn);
  c10::cuda::CUDAGuard guard(x.device());
  const int B = static_cast<int>(x.size(0));
  const int K = static_cast<int>(x.size(1));
  const int N = static_cast<int>(q.size(1));
  TORCH_CHECK(mx.numel() == N && rx.numel() == N && my.numel() == K && ry.numel() == K, "MM8 scale shape mismatch");
  auto y = torch::empty({B, N}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  const dim3 grid((N + bn - 1) / bn, B);
  const half* xp = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
  const uint8_t* qp = q.data_ptr<uint8_t>();
  const half* mxp = reinterpret_cast<const half*>(mx.data_ptr<at::Half>());
  const half* rxp = reinterpret_cast<const half*>(rx.data_ptr<at::Half>());
  const half* myp = reinterpret_cast<const half*>(my.data_ptr<at::Half>());
  const half* ryp = reinterpret_cast<const half*>(ry.data_ptr<at::Half>());
  half* yp = reinterpret_cast<half*>(y.data_ptr<at::Half>());
  bool launched = false;
  #define LAUNCH8(BN_, TN_) if (bn == BN_ && tn == TN_) { \
    mm8_bn_tn_kernel<BN_, TN_><<<grid, BN_ / TN_, 0, stream>>>(xp, qp, mxp, rxp, myp, ryp, yp, B, K, N); launched = true; }
  LAUNCH8(64, 1) LAUNCH8(64, 2)
  LAUNCH8(128, 1) LAUNCH8(128, 2) LAUNCH8(128, 4)
  LAUNCH8(256, 1) LAUNCH8(256, 2) LAUNCH8(256, 4) LAUNCH8(256, 8)
  #undef LAUNCH8
  TORCH_CHECK(launched, "unsupported BN/TN pair");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

torch::Tensor rwkv7_bn_tn_mm4_cuda(
    torch::Tensor x, torch::Tensor q, torch::Tensor mx, torch::Tensor rx,
    torch::Tensor my, torch::Tensor ry, int64_t n_orig, int64_t bn, int64_t tn) {
  check_common(x, q, mx, rx, my, ry, bn, tn);
  c10::cuda::CUDAGuard guard(x.device());
  const int B = static_cast<int>(x.size(0));
  const int K = static_cast<int>(x.size(1));
  const int N = static_cast<int>(n_orig);
  const int NP = static_cast<int>(q.size(1));
  TORCH_CHECK(N > 0 && NP * 2 >= N, "MM4 packed-N mismatch");
  TORCH_CHECK(mx.numel() >= N && rx.numel() >= N && my.numel() == K && ry.numel() == K, "MM4 scale shape mismatch");
  auto y = torch::empty({B, N}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream(x.get_device());
  const dim3 grid((N + bn - 1) / bn, B);
  const half* xp = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
  const uint8_t* qp = q.data_ptr<uint8_t>();
  const half* mxp = reinterpret_cast<const half*>(mx.data_ptr<at::Half>());
  const half* rxp = reinterpret_cast<const half*>(rx.data_ptr<at::Half>());
  const half* myp = reinterpret_cast<const half*>(my.data_ptr<at::Half>());
  const half* ryp = reinterpret_cast<const half*>(ry.data_ptr<at::Half>());
  half* yp = reinterpret_cast<half*>(y.data_ptr<at::Half>());
  bool launched = false;
  #define LAUNCH4(BN_, TN_) if (bn == BN_ && tn == TN_) { \
    mm4_bn_tn_kernel<BN_, TN_><<<grid, BN_ / TN_, 0, stream>>>(xp, qp, mxp, rxp, myp, ryp, yp, B, K, N, NP); launched = true; }
  LAUNCH4(64, 1) LAUNCH4(64, 2)
  LAUNCH4(128, 1) LAUNCH4(128, 2) LAUNCH4(128, 4)
  LAUNCH4(256, 1) LAUNCH4(256, 2) LAUNCH4(256, 4) LAUNCH4(256, 8)
  #undef LAUNCH4
  TORCH_CHECK(launched, "unsupported BN/TN pair");
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
"""


def load_extension():
    from torch.utils.cpp_extension import load_inline

    capability = torch.cuda.get_device_capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{capability[0]}.{capability[1]}")
    digest = hashlib.sha256((CPP_SOURCE + CUDA_SOURCE).encode()).hexdigest()[:10]
    return load_inline(
        name=f"rwkv7_bn_tn_probe_{digest}",
        cpp_sources=CPP_SOURCE,
        cuda_sources=CUDA_SOURCE,
        functions=None,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "--extra-device-vectorization"],
        with_cuda=True,
        verbose=False,
    )


def timed_ms(fn: Callable[[], torch.Tensor], warmup: int, runs: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(runs):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / runs


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.float().reshape(1, -1), b.float().reshape(1, -1)).item())


@lru_cache(maxsize=1)
def driver_version() -> str | None:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            timeout=5,
        ).splitlines()[0].strip()
    except (IndexError, OSError, subprocess.SubprocessError):
        return None


def parse_shape(raw: str) -> tuple[int, int]:
    k, n = (int(v) for v in raw.lower().split("x", 1))
    if k <= 0 or n <= 0:
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return k, n


def run_case(args, ext, mode: str, batch: int, shape: tuple[int, int]) -> list[dict[str, object]]:
    k, n = shape
    x = torch.randn((batch, k), device="cuda", dtype=torch.float16) * 0.1
    dense = torch.randn((k, n), device="cuda", dtype=torch.float16) * 0.02
    fp16_ms = timed_ms(lambda: x @ dense, args.warmup, args.runs)
    if mode == "mm8":
        q, mx, rx, my, ry = quantize_mm8(dense)
        current_fn = lambda: mm8_matmul_triton(x, q, mx, rx, my, ry)
        candidate_fn = lambda c: ext.mm8(x, q, mx, rx, my, ry, c.block_n, c.thread_n)
    else:
        q, mx, rx, my, ry, n_orig, _ = quantize_mm4(dense)
        current_fn = lambda: mm4_matmul_triton(x, q, mx, rx, my, ry, n_orig)
        candidate_fn = lambda c: ext.mm4(x, q, mx, rx, my, ry, n_orig, c.block_n, c.thread_n)
    current = current_fn()
    current_ms = timed_ms(current_fn, args.warmup, args.runs)
    rows: list[dict[str, object]] = []
    for config in bn_tn_candidates(args.block_n, args.thread_n):
        got = candidate_fn(config)
        row: dict[str, object] = {
            "axis": "quant_bn_tn_sweep",
            "status": "pass",
            "mode": mode,
            "device": torch.cuda.get_device_name(),
            "compute_capability": list(torch.cuda.get_device_capability()),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "driver_version": driver_version(),
            "batch_size": batch,
            "k": k,
            "n": n,
            **config.as_dict(),
            "fp16_ms": round(fp16_ms, 6),
            "current_ms": round(current_ms, 6),
            "candidate_ms": round(timed_ms(lambda: candidate_fn(config), args.warmup, args.runs), 6),
            "max_abs_vs_current": round(float((got.float() - current.float()).abs().max().item()), 6),
            "cosine_vs_current": round(cosine(got, current), 9),
        }
        row["speedup_vs_current"] = round(float(row["current_ms"]) / float(row["candidate_ms"]), 6)
        row["speedup_vs_fp16"] = round(float(row["fp16_ms"]) / float(row["candidate_ms"]), 6)
        if float(row["cosine_vs_current"]) < args.min_cosine:
            row["status"] = "fail"
        rows.append(row)
    del dense, x
    torch.cuda.empty_cache()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", choices=("mm8", "mm4"), default=("mm8", "mm4"))
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=(1, 8))
    parser.add_argument("--shapes", nargs="+", type=parse_shape, default=((2048, 2048), (2048, 8192), (8192, 2048)))
    parser.add_argument("--block-n", nargs="+", type=int, default=(64, 128, 256))
    parser.add_argument("--thread-n", nargs="+", type=int, default=(1, 2, 4, 8))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(1234)
    ext = load_extension()
    all_rows: list[dict[str, object]] = []
    for mode in args.modes:
        for batch in args.batch_sizes:
            for shape in args.shapes:
                rows = run_case(args, ext, mode, batch, shape)
                all_rows.extend(rows)
                best = select_best_bn_tn(rows, min_cosine=args.min_cosine)
                print(json.dumps({"case": [mode, batch, *shape], "best": best}, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in all_rows), encoding="utf-8")
    return 0 if all(row["status"] == "pass" for row in all_rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
