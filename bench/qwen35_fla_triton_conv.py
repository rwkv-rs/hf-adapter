#!/usr/bin/env python3
"""FLA Triton causal-convolution bridge for Qwen3.5 benchmark references."""
from __future__ import annotations

from typing import Any


BACKEND_NAME = "fla_triton"


def qwen35_fla_triton_causal_conv1d(
    *,
    x,
    weight,
    bias=None,
    activation=None,
    seq_idx=None,
    **_kwargs,
):
    """Run Qwen `[B,D,T]` prefill through FLA's `[B,T,D]` Triton kernel."""

    if seq_idx is not None:
        raise RuntimeError("Qwen3.5 FLA Triton conv does not support packed seq_idx")
    from fla.modules.convolution import causal_conv1d

    y, _final_state = causal_conv1d(
        x.transpose(1, 2).contiguous(),
        weight=weight,
        bias=bias,
        activation=activation,
        backend="triton",
    )
    return y.transpose(1, 2)


def qwen35_fla_triton_causal_conv1d_update(
    x,
    conv_state,
    weight,
    bias=None,
    activation=None,
):
    """Run one cached Qwen token and update its convolution state in place."""

    from fla.modules.convolution import causal_conv1d_update

    y, updated_state = causal_conv1d_update(
        x.transpose(1, 2).contiguous(),
        conv_state,
        weight=weight,
        bias=bias,
        activation=activation,
    )
    if updated_state.data_ptr() != conv_state.data_ptr():
        # Qwen stores layer caches as non-contiguous views. FLA's input guard
        # makes those views contiguous before launching Triton, so copy the
        # updated contiguous state back to the original HF cache view.
        conv_state.copy_(updated_state)
    return y.transpose(1, 2)


def bind_qwen35_fla_triton_conv(model: Any) -> int:
    """Bind the bridge to every live Qwen3.5 GatedDeltaNet layer."""

    bound = 0
    for module in model.modules():
        if type(module).__name__ != "Qwen3_5GatedDeltaNet":
            continue
        module.causal_conv1d_fn = qwen35_fla_triton_causal_conv1d
        module.causal_conv1d_update = qwen35_fla_triton_causal_conv1d_update
        module._qwen35_conv_backend = BACKEND_NAME
        bound += 1
    if bound <= 0:
        raise RuntimeError("no Qwen3.5 GatedDeltaNet layers accepted the FLA Triton conv binding")
    return bound
