# coding=utf-8
"""Optional MUSA WKV-7 recurrent kernel bridge.

The implementation is derived from KakaruHayate/RWKV-MUSA and ultimately from
BlinkDL/RWKV-LM's Apache-2.0 ``wkv7_cuda.cu``. The retained fp16-IO/fp32-compute
contract comes from the legacy first-generation MTT S70; it is not a capability
claim or scheduling default for later MUSA cards. The path is capability gated
and falls back to the canonical pure-PyTorch recurrence when unavailable.
"""
from __future__ import annotations

import os
from typing import Any

import torch

from .kernel_policy import classify_gpu
from .musa_build import load_musa_inline
from .musa_wkv_source import WKV7_MUSA_HEADER

_FALSE_VALUES = {"0", "false", "no", "off"}
_TRUE_VALUES = {"1", "true", "yes", "on"}
_MODULE: Any | None = None
_MODULE_ERROR: Exception | None = None
_VALIDATED_DEVICE_CACHE: dict[int, bool] = {}

_CPP_SOURCE = r"""
#include <torch/extension.h>
void wkv7_forward(torch::Tensor w, torch::Tensor q, torch::Tensor k, torch::Tensor v,
                  torch::Tensor a, torch::Tensor b, torch::Tensor y,
                  torch::Tensor s0, torch::Tensor sT);
"""

_MUSA_SOURCE_TEMPLATE = r'''
#include <torch/extension.h>
#include "torch_musa/csrc/core/MUSAException.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/MUSAStream.h"
#include "__WKV7_MUSA_HEADER__"

#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

void wkv7_forward(torch::Tensor w, torch::Tensor q, torch::Tensor k, torch::Tensor v,
                  torch::Tensor a, torch::Tensor b, torch::Tensor y,
                  torch::Tensor s0, torch::Tensor sT) {
    CHECK_CONTIGUOUS(w); CHECK_CONTIGUOUS(q); CHECK_CONTIGUOUS(k);
    CHECK_CONTIGUOUS(v); CHECK_CONTIGUOUS(a); CHECK_CONTIGUOUS(b);
    CHECK_CONTIGUOUS(y); CHECK_CONTIGUOUS(s0); CHECK_CONTIGUOUS(sT);
    TORCH_CHECK(w.scalar_type() == at::kHalf, "wkv7 MUSA path requires fp16 IO");
    TORCH_CHECK(s0.scalar_type() == at::kFloat && sT.scalar_type() == at::kFloat,
                "wkv7 MUSA state must be fp32");
    TORCH_CHECK(w.sizes() == q.sizes() && w.sizes() == k.sizes() &&
                w.sizes() == v.sizes() && w.sizes() == a.sizes() &&
                w.sizes() == b.sizes(), "wkv7 MUSA operands must share one shape");

    const int B = w.size(0), T = w.size(1), H = w.size(2), C = w.size(3);
    TORCH_CHECK(C == 64, "wkv7 MUSA currently supports head_size=64 only");
    TORCH_CHECK(s0.dim() == 4 && s0.size(0) == B && s0.size(1) == H &&
                s0.size(2) == C && s0.size(3) == C,
                "wkv7 MUSA state shape must be [B,H,64,64]");
    TORCH_CHECK(sT.sizes() == s0.sizes(), "wkv7 MUSA output state shape mismatch");

    const auto device_index = static_cast<c10::DeviceIndex>(w.get_device());
    const c10::musa::MUSAGuard device_guard(device_index);
    const auto stream = c10::musa::getCurrentMUSAStream(device_index);
    wkv7_forward_kernel<64, __half><<<dim3(H, B), dim3(64), 0, stream>>>(
        T, H,
        (const __half*)w.data_ptr(), (const __half*)q.data_ptr(),
        (const __half*)k.data_ptr(), (const __half*)v.data_ptr(),
        (const __half*)a.data_ptr(), (const __half*)b.data_ptr(),
        (__half*)y.data_ptr(), s0.data_ptr<float>(), sT.data_ptr<float>());
    C10_MUSA_KERNEL_LAUNCH_CHECK();
}
'''


def _mode() -> str:
    value = os.environ.get("RWKV7_MUSA_WKV", "auto").strip().lower()
    if value in _FALSE_VALUES:
        return "off"
    if value in _TRUE_VALUES:
        return "on"
    return "auto"


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


def _is_validated_device(device: torch.device) -> bool:
    index = 0 if device.index is None else int(device.index)
    cached = _VALIDATED_DEVICE_CACHE.get(index)
    if cached is not None:
        return cached
    musa = getattr(torch, "musa", None)
    get_device_name = getattr(musa, "get_device_name", None)
    if not callable(get_device_name):
        return False
    try:
        profile = classify_gpu(str(get_device_name(index)), None, is_musa=True)
        validated = profile.validation_scope == "exact_card_smoke"
    except Exception:
        return False
    _VALIDATED_DEVICE_CACHE[index] = validated
    return validated


def musa_wkv_available(device: torch.device | None = None) -> bool:
    """Return whether the optional MUSA WKV runtime can be considered.

    ``auto`` is fail-closed to exact-card evidence (currently legacy MTT S70).
    Later MUSA devices may use ``RWKV7_MUSA_WKV=1`` for explicit bring-up, but
    are not reported as validated until their own correctness and speed rows
    are retained. Pass a device when deciding whether to prepare fp16 operands.
    """

    mode = _mode()
    if mode == "off" or not _musa_available() or _MODULE_ERROR is not None:
        return False
    if mode == "on":
        return True
    return device is not None and _is_validated_device(device)


def musa_wkv_can_run(*tensors: torch.Tensor) -> bool:
    """Fail closed unless every runtime constraint from RWKV-MUSA is satisfied."""

    if torch.is_grad_enabled() or not tensors:
        return False
    first = tensors[0]
    if not musa_wkv_available(first.device):
        return False
    if first.device.type != "musa" or first.dtype != torch.float16 or first.dim() != 4:
        return False
    if int(first.shape[-1]) != 64:
        return False
    return all(
        tensor.device == first.device
        and tensor.dtype == torch.float16
        and tensor.shape == first.shape
        for tensor in tensors[1:]
    )


def _load_module(device: torch.device):
    global _MODULE, _MODULE_ERROR
    if _MODULE_ERROR is not None:
        raise RuntimeError("RWKV-7 MUSA extension previously failed to build") from _MODULE_ERROR
    if _MODULE is not None:
        return _MODULE
    if not musa_wkv_available(device):
        raise RuntimeError("RWKV-7 MUSA extension requested without an available MUSA runtime")
    source = _MUSA_SOURCE_TEMPLATE.replace(
        '#include "__WKV7_MUSA_HEADER__"',
        WKV7_MUSA_HEADER,
    )
    try:
        _MODULE = load_musa_inline(
            "rwkv7_hf_wkv_musa",
            [_CPP_SOURCE],
            [source],
            ["wkv7_forward"],
            extra_musa_cflags=["-O3"],
            verbose=False,
        )
    except Exception as exc:
        _MODULE_ERROR = exc
        raise
    return _MODULE


def try_musa_wkv(
    decay: torch.Tensor,
    receptance: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Try the optional kernel and return ``None`` for a safe eager fallback."""

    operands = (decay, receptance, key, value, a, b)
    if not musa_wkv_can_run(*operands):
        return None
    try:
        return musa_wkv(*operands, state)
    except Exception:  # Optional acceleration must never block the eager fallback.
        return None


def musa_wkv(
    decay: torch.Tensor,
    receptance: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run MUSA WKV-7 with ``[B,T,H,64]`` fp16 IO and fp32 state."""

    operands = (decay, receptance, key, value, a, b)
    if not musa_wkv_can_run(*operands):
        raise ValueError("MUSA WKV operands do not satisfy the validated runtime contract")
    expected_state = (decay.shape[0], decay.shape[2], decay.shape[3], decay.shape[3])
    if state.device != decay.device or state.dtype != torch.float32 or tuple(state.shape) != expected_state:
        raise ValueError("MUSA WKV state must be fp32 [B,H,64,64] on the input device")
    module = _load_module(decay.device)
    packed = [tensor.contiguous() for tensor in operands]
    initial_state = state.contiguous()
    output = torch.empty_like(decay)
    final_state = torch.empty_like(initial_state)
    module.wkv7_forward(*packed, output, initial_state, final_state)
    return output, final_state


__all__ = ["musa_wkv", "musa_wkv_available", "musa_wkv_can_run", "try_musa_wkv"]
