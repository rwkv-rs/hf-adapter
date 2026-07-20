"""Flattened CUDA linear leaf for optional RWKV-7 training.

The readable Hugging Face model deliberately uses a fixed 128-row projection
shape so results do not change when an evaluation framework regroups samples.
That execution rule is valuable for the reference line but leaves large
training matrices split into many small GEMMs.  This optional leaf flattens
``[batch, time, channels]`` once and lets PyTorch dispatch a single cuBLAS
linear operation.  PyTorch continues to own autograd, parameters, adapters,
and optimizer state; the kernel package owns only this stateless operation.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


IMPLEMENTATION = "torch-cuda-rwkv7-flattened-linear-training-v1"
_MIN_FLATTENED_ROWS = 128


def _unsupported(reason: str) -> dict[str, Any]:
    return {
        "supported": False,
        "implementation": IMPLEMENTATION,
        "reason": reason,
    }


def probe_linear_training_v1(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    fully_active: bool | None = None,
    token_aligned: bool | None = None,
) -> dict[str, Any]:
    """Report support for one stateless CUDA training projection."""

    del fully_active, token_aligned

    if not isinstance(value, torch.Tensor) or not isinstance(weight, torch.Tensor):
        return _unsupported("value and weight must be tensors")
    if value.ndim != 3 or weight.ndim != 2:
        return _unsupported("training linear expects value [B,T,C] and weight [O,C]")
    if int(value.shape[-1]) != int(weight.shape[1]):
        return _unsupported("linear input and weight dimensions do not match")
    if not value.is_cuda or not weight.is_cuda:
        return _unsupported("the optimized training linear requires CUDA tensors")
    if value.device != weight.device:
        return _unsupported("value and weight must share one CUDA device")
    if value.dtype not in (torch.float16, torch.bfloat16):
        return _unsupported("the optimized training linear requires FP16 or BF16")
    if weight.dtype != value.dtype:
        return _unsupported("value and weight must have the same dtype")
    if bias is not None:
        if not isinstance(bias, torch.Tensor) or bias.ndim != 1:
            return _unsupported("linear bias must be a rank-one tensor")
        if int(bias.shape[0]) != int(weight.shape[0]):
            return _unsupported("linear bias and output dimensions do not match")
        if bias.device != value.device or bias.dtype != value.dtype:
            return _unsupported("linear bias must share the value device and dtype")
    if not any(
        tensor.requires_grad
        for tensor in (value, weight, bias)
        if isinstance(tensor, torch.Tensor)
    ):
        return _unsupported("the optimized training linear requires autograd")
    flattened_rows = int(value.shape[0]) * int(value.shape[1])
    if flattened_rows < _MIN_FLATTENED_ROWS:
        return _unsupported(
            "the optimized training linear requires at least "
            f"{_MIN_FLATTENED_ROWS} flattened rows; smaller projections retain "
            "the reference accumulation contract"
        )
    return {
        "supported": True,
        "implementation": IMPLEMENTATION,
        "reason": "flattened CUDA training projection is supported by PyTorch cuBLAS",
    }


def linear_training_v1(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    fully_active: bool | None = None,
    token_aligned: bool | None = None,
) -> torch.Tensor:
    """Apply one flattened projection while retaining native PyTorch autograd."""

    support = probe_linear_training_v1(
        value,
        weight,
        bias,
        fully_active=fully_active,
        token_aligned=token_aligned,
    )
    if not support["supported"]:
        raise RuntimeError(str(support["reason"]))
    batch, tokens, channels = value.shape
    projected = F.linear(value.reshape(batch * tokens, channels), weight, bias)
    return projected.reshape(batch, tokens, int(weight.shape[0]))


__all__ = ["IMPLEMENTATION", "linear_training_v1", "probe_linear_training_v1"]
