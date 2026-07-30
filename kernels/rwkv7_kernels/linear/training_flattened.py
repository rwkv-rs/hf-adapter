"""Shape-stable CUDA linear leaf for optional RWKV-7 training.

The readable Hugging Face model deliberately uses a fixed 128-row projection
shape so results do not change when an evaluation framework regroups samples.
That execution rule is valuable for the reference line but leaves large
training matrices split into many small GEMMs.  This optional leaf normally
flattens ``[batch, time, channels]`` once. Four-times-wide FFN projections use
bounded row groups instead: a single 512-row BF16 GEMM changes the optimizer
gradient outside the accepted full-model envelope on Ada, while a 320-row
ceiling retains both numerical parity and most of the launch reduction. The
ceiling is expressed as five ordinary 64-row GEMM tiles and applies to the
matrix row count rather than to a device name, batch size, or sequence length.
PyTorch continues to own autograd, parameters, adapters, and optimizer state;
the kernel package owns only this stateless operation.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


IMPLEMENTATION = "torch-cuda-rwkv7-flattened-linear-training-v1"
_MIN_FLATTENED_ROWS = 128
_GEMM_ROW_TILE = 64
_STABLE_FFN_ROWS_PER_GEMM = 5 * _GEMM_ROW_TILE


def _is_four_times_wide_projection(weight: torch.Tensor) -> bool:
    """Return whether *weight* is one of RWKV's 4x FFN projections."""

    output_features, input_features = map(int, weight.shape)
    narrow = min(output_features, input_features)
    wide = max(output_features, input_features)
    return narrow > 0 and wide == 4 * narrow


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
        "reason": (
            "shape-stable CUDA training projection is supported by PyTorch cuBLAS"
        ),
    }


def flattened_linear(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Apply one flattened or numerically bounded stateless CUDA projection."""

    batch, tokens, channels = value.shape
    if _is_four_times_wide_projection(weight):
        batch_outputs = []
        for batch_start in range(0, batch, _STABLE_FFN_ROWS_PER_GEMM):
            batch_group = value[
                batch_start : batch_start + _STABLE_FFN_ROWS_PER_GEMM
            ]
            group_batch = int(batch_group.shape[0])
            tokens_per_gemm = max(1, _STABLE_FFN_ROWS_PER_GEMM // group_batch)
            token_outputs = [
                F.linear(
                    batch_group[
                        :, token_start : token_start + tokens_per_gemm
                    ].contiguous(),
                    weight,
                    bias,
                )
                for token_start in range(0, tokens, tokens_per_gemm)
            ]
            batch_outputs.append(torch.cat(token_outputs, dim=1))
        return torch.cat(batch_outputs, dim=0)

    projected = F.linear(value.reshape(batch * tokens, channels), weight, bias)
    return projected.reshape(batch, tokens, int(weight.shape[0]))


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
    return flattened_linear(value, weight, bias)


__all__ = [
    "IMPLEMENTATION",
    "flattened_linear",
    "linear_training_v1",
    "probe_linear_training_v1",
]
