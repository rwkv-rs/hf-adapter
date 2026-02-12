# coding=utf-8
"""Optional MUSA WKV-7 recurrent kernel bridge.

The implementation is derived from KakaruHayate/RWKV-MUSA and ultimately from
BlinkDL/RWKV-LM's Apache-2.0 ``wkv7_cuda.cu``. The MUSA path is capability
gated and falls back to the canonical pure-PyTorch recurrence when unavailable.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from .musa_build import load_musa_inline

_FALSE_VALUES = {"0", "false", "no", "off"}
_MODULE: Any | None = None
_MODULE_ERROR: Exception | None = None

_CPP_SOURCE = r"""
#include <torch/extension.h>
void wkv7_forward(torch::Tensor w, torch::Tensor q, torch::Tensor k, torch::Tensor v,
                  torch::Tensor a, torch::Tensor b, torch::Tensor y,
                  torch::Tensor s0, torch::Tensor sT);
"""

_MUSA_SOURCE_TEMPLATE = r'''
#include <torch/extension.h>
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

    wkv7_forward_kernel<64, __half><<<dim3(H, B), dim3(64)>>>(
        T, H,
        (const __half*)w.data_ptr(), (const __half*)q.data_ptr(),
        (const __half*)k.data_ptr(), (const __half*)v.data_ptr(),
        (const __half*)a.data_ptr(), (const __half*)b.data_ptr(),
        (__half*)y.data_ptr(), s0.data_ptr<float>(), sT.data_ptr<float>());
}
'''


def _enabled() -> bool:
    return os.environ.get("RWKV7_MUSA_WKV", "1").strip().lower() not in _FALSE_VALUES


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


def musa_wkv_available() -> bool:
    """Return whether the validated fp16-IO/fp32-state MUSA route can be tried."""

    return _enabled() and _musa_available() and _MODULE_ERROR is None


def musa_wkv_can_run(*tensors: torch.Tensor) -> bool:
    """Fail closed unless every runtime constraint from RWKV-MUSA is satisfied."""

    if torch.is_grad_enabled() or not musa_wkv_available() or not tensors:
        return False
    first = tensors[0]
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


def _load_module():
    global _MODULE, _MODULE_ERROR
    if not musa_wkv_available():
        raise RuntimeError("RWKV-7 MUSA extension requested without an available MUSA runtime")
    if _MODULE is not None:
        return _MODULE
    if _MODULE_ERROR is not None:
        raise RuntimeError("RWKV-7 MUSA extension previously failed to build") from _MODULE_ERROR
    header = Path(__file__).resolve().parent / "csrc" / "musa" / "wkv7_musa.muh"
    source = _MUSA_SOURCE_TEMPLATE.replace("__WKV7_MUSA_HEADER__", header.as_posix())
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
    except RuntimeError:
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
    module = _load_module()
    packed = [tensor.contiguous() for tensor in operands]
    initial_state = state.contiguous()
    output = torch.empty_like(decay)
    final_state = torch.empty_like(initial_state)
    module.wkv7_forward(*packed, output, initial_state, final_state)
    return output, final_state


__all__ = ["musa_wkv", "musa_wkv_available", "musa_wkv_can_run", "try_musa_wkv"]
