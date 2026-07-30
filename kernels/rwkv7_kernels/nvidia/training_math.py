"""Canonical PyTorch math used inside the native training runtime.

The whole-model training route owns its complete internal layer calculation.
It must not call ``RWKV7Linear.forward`` again: doing so would recursively
enter the optional leaf dispatcher and make a supported native request fail
when one of its small projections intentionally stays on reference math.

These helpers mirror the readable Hugging Face equations without importing or
replacing any model class. Parameters remain the original module parameters,
so ordinary PyTorch autograd, optimizers, and adapter ownership are preserved.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

from ..linear.training_flattened import flattened_linear


REFERENCE_LINEAR_ROWS = 128


def fixed_row_linear(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply the canonical batch- and length-invariant projection contract."""

    if value.ndim != 3:
        return F.linear(value, weight, bias)

    batch_size, sequence_length, input_size = value.shape
    batch_groups: list[torch.Tensor] = []
    for batch_start in range(0, batch_size, REFERENCE_LINEAR_ROWS):
        group = value[batch_start : batch_start + REFERENCE_LINEAR_ROWS]
        valid_batch = int(group.shape[0])
        padded_batch = 1 << (valid_batch - 1).bit_length()
        tokens_per_block = REFERENCE_LINEAR_ROWS // padded_batch

        sequence_blocks: list[torch.Tensor] = []
        for token_start in range(0, sequence_length, tokens_per_block):
            block = group[:, token_start : token_start + tokens_per_block]
            valid_tokens = int(block.shape[1])
            if valid_batch < padded_batch or valid_tokens < tokens_per_block:
                block = F.pad(
                    block,
                    (
                        0,
                        0,
                        0,
                        tokens_per_block - valid_tokens,
                        0,
                        padded_batch - valid_batch,
                    ),
                )
            projected = F.linear(
                block.contiguous().view(REFERENCE_LINEAR_ROWS, input_size),
                weight,
                bias,
            ).view(padded_batch, tokens_per_block, -1)
            sequence_blocks.append(projected[:valid_batch, :valid_tokens])
        batch_groups.append(torch.cat(sequence_blocks, dim=1))
    return torch.cat(batch_groups, dim=0)


def training_linear(
    value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select the canonical one-tile path or shape-stable training GEMMs.

    A single canonical tile is kept byte-for-byte identical to the readable HF
    model. Larger ``B * T`` requests use the optional linear leaf, which keeps
    ordinary projections flattened and bounds the row group for 4x FFN
    projections. This boundary is private to the explicitly selected training
    runtime; the clean reference model continues to use
    :func:`fixed_row_linear` semantics.
    """

    if value.ndim != 3 or int(value.shape[0]) * int(value.shape[1]) <= REFERENCE_LINEAR_ROWS:
        return fixed_row_linear(value, weight, bias)
    return flattened_linear(value, weight, bias)


def module_linear(
    module: Any,
    value: torch.Tensor,
    *,
    include_bias: bool = True,
) -> torch.Tensor:
    """Project with a structurally compatible linear module without dispatch."""

    bias = getattr(module, "bias", None) if include_bias else None
    return training_linear(value, module.weight, bias)


def low_rank_projection(
    module: Any,
    value: torch.Tensor,
    *,
    activation: Callable[[torch.Tensor], torch.Tensor] | None = None,
    include_bias: bool = True,
) -> torch.Tensor:
    """Evaluate one clean-model low-rank projection without nested dispatch."""

    value = module_linear(module.lora[0], value)
    if activation is not None:
        value = activation(value)
    return module_linear(module.lora[2], value, include_bias=include_bias)


def channel_mix(module: Any, hidden_states: torch.Tensor) -> torch.Tensor:
    """Evaluate fully active, zero-initial-state ChannelMix canonically."""

    shifted = torch.cat(
        (torch.zeros_like(hidden_states[:, :1]), hidden_states[:, :-1]),
        dim=1,
    )
    mixed = hidden_states + (shifted - hidden_states) * module.x_k.view(1, 1, -1)
    activated = torch.relu(module_linear(module.key, mixed)).square()
    return module_linear(module.value, activated)


__all__ = [
    "REFERENCE_LINEAR_ROWS",
    "channel_mix",
    "flattened_linear",
    "fixed_row_linear",
    "low_rank_projection",
    "module_linear",
    "training_linear",
]
