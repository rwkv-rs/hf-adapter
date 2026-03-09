# coding=utf-8
"""Optional exact-card MUSA elementwise fusions for RWKV-7.

The initial MTT S70 lane is opt-in and inference-only. S70 is a legacy
first-generation validation device with no Tensor Core and very slow fp16
compute; this fusion only reduces pointwise launches and must not define policy
for later MUSA hardware. Build or runtime failures return ``None`` so the
canonical eager expressions remain the source of truth. The extension is
separate from ``musa_wkv`` so an experiment cannot disable the validated
recurrent kernel.
"""
from __future__ import annotations

import os
from typing import Any

import torch

from .kernel_policy import is_mtt_s70_name
from .musa_build import load_musa_inline

_TRUE_VALUES = {"1", "true", "yes", "on"}
_MODULE: Any | None = None
_MODULE_ERROR: Exception | None = None
_CALLS = 0
_S70_DEVICE_CACHE: dict[int, bool] = {}

_CPP_SOURCE = r"""
#include <torch/extension.h>
void attn_shift_mix_6(torch::Tensor x, torch::Tensor previous,
                      torch::Tensor mix_r, torch::Tensor mix_w,
                      torch::Tensor mix_k, torch::Tensor mix_v,
                      torch::Tensor mix_a, torch::Tensor mix_g,
                      torch::Tensor out_r, torch::Tensor out_w,
                      torch::Tensor out_k, torch::Tensor out_v,
                      torch::Tensor out_a, torch::Tensor out_g);
"""

_MUSA_SOURCE = r"""
#include <torch/extension.h>
#include <musa_fp16.h>
#include "torch_musa/csrc/core/MUSAException.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/MUSAStream.h"

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

__global__ void attn_shift_mix_6_kernel(
    const __half *__restrict__ x,
    const __half *__restrict__ previous,
    const __half *__restrict__ mix_r,
    const __half *__restrict__ mix_w,
    const __half *__restrict__ mix_k,
    const __half *__restrict__ mix_v,
    const __half *__restrict__ mix_a,
    const __half *__restrict__ mix_g,
    __half *__restrict__ out_r,
    __half *__restrict__ out_w,
    __half *__restrict__ out_k,
    __half *__restrict__ out_v,
    __half *__restrict__ out_a,
    __half *__restrict__ out_g,
    const int hidden,
    const int total) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total) return;
    const int column = index % hidden;
    const __half x_value = x[index];
    volatile __half delta = __hsub(previous[index], x_value);
    volatile __half product_r = __hmul(delta, mix_r[column]);
    volatile __half product_w = __hmul(delta, mix_w[column]);
    volatile __half product_k = __hmul(delta, mix_k[column]);
    volatile __half product_v = __hmul(delta, mix_v[column]);
    volatile __half product_a = __hmul(delta, mix_a[column]);
    volatile __half product_g = __hmul(delta, mix_g[column]);
    out_r[index] = __hadd(x_value, product_r);
    out_w[index] = __hadd(x_value, product_w);
    out_k[index] = __hadd(x_value, product_k);
    out_v[index] = __hadd(x_value, product_v);
    out_a[index] = __hadd(x_value, product_a);
    out_g[index] = __hadd(x_value, product_g);
}

void attn_shift_mix_6(torch::Tensor x, torch::Tensor previous,
                      torch::Tensor mix_r, torch::Tensor mix_w,
                      torch::Tensor mix_k, torch::Tensor mix_v,
                      torch::Tensor mix_a, torch::Tensor mix_g,
                      torch::Tensor out_r, torch::Tensor out_w,
                      torch::Tensor out_k, torch::Tensor out_v,
                      torch::Tensor out_a, torch::Tensor out_g) {
    CHECK_CONTIGUOUS(x); CHECK_CONTIGUOUS(previous);
    CHECK_CONTIGUOUS(mix_r); CHECK_CONTIGUOUS(mix_w); CHECK_CONTIGUOUS(mix_k);
    CHECK_CONTIGUOUS(mix_v); CHECK_CONTIGUOUS(mix_a); CHECK_CONTIGUOUS(mix_g);
    CHECK_CONTIGUOUS(out_r); CHECK_CONTIGUOUS(out_w); CHECK_CONTIGUOUS(out_k);
    CHECK_CONTIGUOUS(out_v); CHECK_CONTIGUOUS(out_a); CHECK_CONTIGUOUS(out_g);
    TORCH_CHECK(x.dim() == 2 && previous.sizes() == x.sizes(),
                "MUSA attention shift mix expects matching [B,D] inputs");
    TORCH_CHECK(x.scalar_type() == at::kHalf && previous.scalar_type() == at::kHalf,
                "MUSA attention shift mix requires fp16 inputs");
    TORCH_CHECK(x.device() == previous.device(),
                "MUSA attention shift mix inputs must share one device");
    const int hidden = x.size(1);
    const auto check_vector = [&](const torch::Tensor &value, const char *name) {
        TORCH_CHECK(value.device() == x.device() && value.scalar_type() == at::kHalf &&
                    value.numel() == hidden, name, " must be fp16 [D] on the input device");
    };
    check_vector(mix_r, "mix_r"); check_vector(mix_w, "mix_w");
    check_vector(mix_k, "mix_k"); check_vector(mix_v, "mix_v");
    check_vector(mix_a, "mix_a"); check_vector(mix_g, "mix_g");
    for (const auto &value : {out_r, out_w, out_k, out_v, out_a, out_g}) {
        TORCH_CHECK(value.device() == x.device() && value.scalar_type() == at::kHalf &&
                    value.sizes() == x.sizes(),
                    "MUSA attention shift mix outputs must match the fp16 input");
    }
    const int total = x.numel();
    const auto device_index = static_cast<c10::DeviceIndex>(x.get_device());
    const c10::musa::MUSAGuard device_guard(device_index);
    const auto stream = c10::musa::getCurrentMUSAStream(device_index);
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    attn_shift_mix_6_kernel<<<blocks, threads, 0, stream>>>(
        (const __half*)x.data_ptr(), (const __half*)previous.data_ptr(),
        (const __half*)mix_r.data_ptr(), (const __half*)mix_w.data_ptr(),
        (const __half*)mix_k.data_ptr(), (const __half*)mix_v.data_ptr(),
        (const __half*)mix_a.data_ptr(), (const __half*)mix_g.data_ptr(),
        (__half*)out_r.data_ptr(), (__half*)out_w.data_ptr(),
        (__half*)out_k.data_ptr(), (__half*)out_v.data_ptr(),
        (__half*)out_a.data_ptr(), (__half*)out_g.data_ptr(), hidden, total);
    C10_MUSA_KERNEL_LAUNCH_CHECK();
}
"""


def _enabled() -> bool:
    return os.environ.get("RWKV7_MUSA_ATTN_SHIFT_MIX", "0").strip().lower() in _TRUE_VALUES


def _musa_available() -> bool:
    try:
        import torch_musa  # noqa: F401
    except (ImportError, ModuleNotFoundError):
        return False
    musa = getattr(torch, "musa", None)
    is_available = getattr(musa, "is_available", None)
    try:
        return bool(callable(is_available) and is_available())
    except Exception:
        return False


def _is_mtt_s70_device(device: torch.device) -> bool:
    index = 0 if device.index is None else int(device.index)
    cached = _S70_DEVICE_CACHE.get(index)
    if cached is not None:
        return cached
    musa = getattr(torch, "musa", None)
    get_device_name = getattr(musa, "get_device_name", None)
    if not callable(get_device_name):
        return False
    try:
        matched = is_mtt_s70_name(get_device_name(index))
    except Exception:
        return False
    _S70_DEVICE_CACHE[index] = matched
    return matched


def musa_attn_shift_mix_available() -> bool:
    return _enabled() and _musa_available() and _MODULE_ERROR is None


def _load_module():
    global _MODULE, _MODULE_ERROR
    if _MODULE_ERROR is not None:
        raise RuntimeError("RWKV-7 MUSA fusion previously failed to build") from _MODULE_ERROR
    if _MODULE is not None:
        return _MODULE
    if not musa_attn_shift_mix_available():
        raise RuntimeError("RWKV-7 MUSA fusion requested without an available MUSA runtime")
    try:
        _MODULE = load_musa_inline(
            "rwkv7_hf_attn_shift_mix_musa_v2_strict",
            [_CPP_SOURCE],
            [_MUSA_SOURCE],
            ["attn_shift_mix_6"],
            extra_musa_cflags=["-O3", "-ffp-contract=off"],
            verbose=False,
        )
    except Exception as exc:
        _MODULE_ERROR = exc
        raise
    return _MODULE


def musa_attn_shift_mix_can_run(x: torch.Tensor, previous: torch.Tensor, *mixes: torch.Tensor) -> bool:
    if torch.is_grad_enabled() or not musa_attn_shift_mix_available() or len(mixes) != 6:
        return False
    if (
        x.device.type != "musa"
        or not _is_mtt_s70_device(x.device)
        or x.dtype != torch.float16
        or x.dim() != 2
    ):
        return False
    if previous.device != x.device or previous.dtype != x.dtype or previous.shape != x.shape:
        return False
    hidden = int(x.shape[1])
    return all(
        mix.device == x.device
        and mix.dtype == x.dtype
        and int(mix.numel()) == hidden
        for mix in mixes
    )


def try_musa_attn_shift_mix(
    x: torch.Tensor,
    previous: torch.Tensor,
    *mixes: torch.Tensor,
) -> tuple[torch.Tensor, ...] | None:
    global _CALLS
    try:
        if not musa_attn_shift_mix_can_run(x, previous, *mixes):
            return None
        module = _load_module()
        x_c = x.contiguous()
        previous_c = previous.contiguous()
        mixes_c = tuple(mix.reshape(-1).contiguous() for mix in mixes)
        outputs = tuple(torch.empty_like(x_c) for _ in range(6))
        module.attn_shift_mix_6(x_c, previous_c, *mixes_c, *outputs)
        _CALLS += 1
        return outputs
    except Exception:
        return None


__all__ = [
    "musa_attn_shift_mix_available",
    "musa_attn_shift_mix_can_run",
    "try_musa_attn_shift_mix",
]
