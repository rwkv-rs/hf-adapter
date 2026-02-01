# coding=utf-8
"""BitsAndBytes W8 dispatch helpers for native JIT prefill and decode.

This module owns BnB module detection, direct inference operators and fused
activation-quantization eligibility.  The native JIT facade re-exports the
historical private names as direct aliases, so hot projection calls do not gain
an extra Python wrapper.
"""
from __future__ import annotations

import os

import torch


try:  # pragma: no cover - optional BnB W8 FFN activation fusion
    from .native_quant_bnb8 import (
        fused_bnb8_attn_sequence_mix_quant,
        fused_bnb8_ffn_sequence_mix_quant,
        fused_bnb8_relu_square_quant,
        fused_bnb8_relu_square_quant_available,
    )
except Exception:  # pragma: no cover - direct remote-file execution fallback
    try:
        from native_quant_bnb8 import (
            fused_bnb8_attn_sequence_mix_quant,
            fused_bnb8_ffn_sequence_mix_quant,
            fused_bnb8_relu_square_quant,
            fused_bnb8_relu_square_quant_available,
        )
    except Exception:
        fused_bnb8_attn_sequence_mix_quant = None  # type: ignore[assignment]
        fused_bnb8_ffn_sequence_mix_quant = None  # type: ignore[assignment]
        fused_bnb8_relu_square_quant = None  # type: ignore[assignment]
        fused_bnb8_relu_square_quant_available = None  # type: ignore[assignment]

try:  # pragma: no cover - optional in older converted model dirs
    from .kernel_policy import current_kernel_policy
except Exception:  # pragma: no cover - direct remote-file execution fallback
    try:
        from kernel_policy import current_kernel_policy
    except Exception:
        current_kernel_policy = None  # type: ignore[assignment]


def _kernel_policy():
    if current_kernel_policy is None:
        return None
    try:
        return current_kernel_policy(torch_module=torch)
    except Exception:
        return None


def _native_bnb8_policy_flag(env_name: str, policy_name: str) -> bool:
    try:
        default = bool(getattr(_kernel_policy(), policy_name, False))
    except Exception:
        default = False
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _native_bnb8_policy_block(env_name: str, policy_name: str, fallback: int) -> int:
    try:
        default = int(getattr(_kernel_policy(), policy_name, fallback))
    except Exception:
        default = int(fallback)
    raw = os.environ.get(env_name)
    value = default if raw is None else int(raw)
    if value not in {256, 512, 1024, 2048, 4096}:
        raise ValueError(f"{env_name} must be 256, 512, 1024, 2048, or 4096")
    return value


def _native_bnb8_direct_enabled() -> bool:
    """Use the inference-only BnB W8 operator path without autograd wrappers."""

    return _native_bnb8_policy_flag(
        "RWKV7_NATIVE_BNB8_DIRECT",
        "native_bnb8_direct",
    )


def _is_bnb8_linear(operand) -> bool:
    cls = type(operand)
    return bool(
        cls.__name__ == "Linear8bitLt"
        and cls.__module__.startswith("bitsandbytes.")
        and hasattr(operand, "state")
        and hasattr(operand, "weight")
    )


def _bnb8_direct_linear(x: torch.Tensor, operand) -> torch.Tensor | None:
    """Run BnB's threshold-zero inference operators directly, if eligible."""

    if (
        not _native_bnb8_direct_enabled()
        or not _is_bnb8_linear(operand)
        or torch.is_grad_enabled()
        or bool(getattr(operand, "training", False))
    ):
        return None
    state = operand.state
    if float(getattr(state, "threshold", 0.0) or 0.0) != 0.0:
        return None
    if getattr(state, "CB", None) is None:
        if getattr(operand.weight, "CB", None) is None:
            return None
        operand.init_8bit_state()
    cb = getattr(state, "CB", None)
    scb = getattr(state, "SCB", None)
    if cb is None or scb is None:
        return None

    input_shape = tuple(x.shape)
    rows = x.reshape(-1, input_shape[-1]) if x.dim() != 2 else x
    quantized, scales, _ = torch.ops.bitsandbytes.int8_vectorwise_quant.default(
        rows.to(torch.float16),
        0.0,
    )
    bias = getattr(operand, "bias", None)
    if bias is not None and bias.dtype != x.dtype:
        bias = bias.to(x.dtype)
    out = torch.ops.bitsandbytes.int8_scaled_mm.default(
        quantized,
        cb,
        scales,
        scb,
        bias=bias,
        dtype=x.dtype,
    )
    if x.dim() != 2:
        out = out.reshape(*input_shape[:-1], int(operand.out_features))
    return out


def _bnb8_direct_relu_square_linear(x: torch.Tensor, operand) -> torch.Tensor | None:
    """Fuse RWKV FFN ReLU² preparation into BnB W8 activation quantization."""

    enabled = _native_bnb8_policy_flag(
        "RWKV7_NATIVE_BNB8_RELU_QUANT",
        "native_bnb8_relu_quant",
    )
    if (
        not enabled
        or not _native_bnb8_direct_enabled()
        or not _is_bnb8_linear(operand)
        or torch.is_grad_enabled()
        or bool(getattr(operand, "training", False))
        or fused_bnb8_relu_square_quant is None
        or fused_bnb8_relu_square_quant_available is None
        or not fused_bnb8_relu_square_quant_available()
    ):
        return None
    state = operand.state
    if float(getattr(state, "threshold", 0.0) or 0.0) != 0.0:
        return None
    if getattr(state, "CB", None) is None:
        if getattr(operand.weight, "CB", None) is None:
            return None
        operand.init_8bit_state()
    cb = getattr(state, "CB", None)
    scb = getattr(state, "SCB", None)
    if cb is None or scb is None:
        return None
    quantized, scales = fused_bnb8_relu_square_quant(x)
    bias = getattr(operand, "bias", None)
    if bias is not None and bias.dtype != x.dtype:
        bias = bias.to(x.dtype)
    out = torch.ops.bitsandbytes.int8_scaled_mm.default(
        quantized,
        cb,
        scales,
        scb,
        bias=bias,
        dtype=x.dtype,
    )
    input_shape = tuple(x.shape)
    if x.dim() != 2:
        out = out.reshape(*input_shape[:-1], int(operand.out_features))
    return out


def _bnb8_prequant_linear(quantized, scales, operand, *, dtype, output_shape):
    """Apply a BnB W8 matrix to already row-quantized activations."""

    if not _is_bnb8_linear(operand):
        raise TypeError("prequantized BnB dispatch requires Linear8bitLt")
    state = operand.state
    if float(getattr(state, "threshold", 0.0) or 0.0) != 0.0:
        raise ValueError("prequantized BnB dispatch requires threshold=0")
    if getattr(state, "CB", None) is None:
        operand.init_8bit_state()
    bias = getattr(operand, "bias", None)
    if bias is not None and bias.dtype != dtype:
        bias = bias.to(dtype)
    out = torch.ops.bitsandbytes.int8_scaled_mm.default(
        quantized,
        state.CB,
        scales,
        state.SCB,
        bias=bias,
        dtype=dtype,
    )
    return out.reshape(*output_shape, int(operand.out_features))


def _bnb8_rkv_mix_quant_enabled(*operands) -> bool:
    if (
        not _native_bnb8_policy_flag(
            "RWKV7_NATIVE_BNB8_RKV_MIX_QUANT",
            "native_bnb8_rkv_mix_quant",
        )
        or not _native_bnb8_direct_enabled()
        or fused_bnb8_attn_sequence_mix_quant is None
        or fused_bnb8_relu_square_quant_available is None
        or not fused_bnb8_relu_square_quant_available()
        or torch.is_grad_enabled()
    ):
        return False
    for operand in operands:
        if not _is_bnb8_linear(operand) or bool(getattr(operand, "training", False)):
            return False
        if float(getattr(operand.state, "threshold", 0.0) or 0.0) != 0.0:
            return False
    return True


def _bnb8_ffn_mix_quant_enabled(operand) -> bool:
    return bool(
        _native_bnb8_policy_flag(
            "RWKV7_NATIVE_BNB8_FFN_MIX_QUANT",
            "native_bnb8_ffn_mix_quant",
        )
        and _native_bnb8_direct_enabled()
        and fused_bnb8_ffn_sequence_mix_quant is not None
        and fused_bnb8_relu_square_quant_available is not None
        and fused_bnb8_relu_square_quant_available()
        and not torch.is_grad_enabled()
        and _is_bnb8_linear(operand)
        and not bool(getattr(operand, "training", False))
        and float(getattr(operand.state, "threshold", 0.0) or 0.0) == 0.0
    )


__all__ = [
    "_bnb8_direct_linear",
    "_bnb8_direct_relu_square_linear",
    "_bnb8_ffn_mix_quant_enabled",
    "_bnb8_prequant_linear",
    "_bnb8_rkv_mix_quant_enabled",
    "_is_bnb8_linear",
    "_native_bnb8_direct_enabled",
    "_native_bnb8_policy_block",
    "_native_bnb8_policy_flag",
]
