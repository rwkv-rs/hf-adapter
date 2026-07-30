"""Private whole-model training diagnostic for vendored official RWKV-LM training ops.

This module is retained as migration evidence and a focused kernel-development
diagnostic.  It is deliberately not imported by the public model-forward
protocol and is not an HF training route.  Standard training always executes
the readable ``modeling_rwkv7.py`` layer loop, where recurrent, linear, and
Mix6 tensor leaves may be replaced independently.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from . import official_training_cuda as official_training
from .training_math import channel_mix, module_linear


IMPLEMENTATION = "native-nvidia-official-training-autograd-v2"


def causal_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute the public shifted causal loss without copying shifted logits.

    The former implementation materialized ``logits[:, :-1].contiguous()`` and
    then promoted that large tensor to FP32.  Keep logits in their native
    contiguous ``[B, T, V]`` layout instead and shift the tiny integer target
    tensor.  A sum divided by the valid-token count also handles an all-ignored
    batch without a device-to-host synchronization or a NaN mean.
    """

    shifted_targets = torch.cat(
        (labels[:, 1:], labels.new_full((int(labels.shape[0]), 1), -100)),
        dim=1,
    )
    loss_sum = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        shifted_targets.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    valid_tokens = (shifted_targets != -100).sum().clamp_min(1)
    return loss_sum / valid_tokens.to(dtype=loss_sum.dtype)


def _run_training_diagnostic(owner: Any, request: dict[str, Any]) -> dict[str, Any]:
    """Exercise the historical dense BF16 runtime outside the public HF route."""

    # Only Mix6 and ClampW are used by this accepted route. Experimental fused
    # gates and the optional L2Wrap loss remain separately loadable diagnostics
    # and must not inflate ordinary HF training cold start.
    official_training.load_training_runtime_cuda_extensions()
    input_ids = request.get("input_ids")
    inputs_embeds = request.get("inputs_embeds")
    if input_ids is not None:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        hidden_states = owner.model.embeddings(input_ids)
    else:
        hidden_states = inputs_embeds

    v_first = hidden_states.new_zeros(1)

    def run_layer(layer, hidden, first_value):
        residual = layer.pre_norm(hidden) if hasattr(layer, "pre_norm") else hidden
        attention_input = layer.attn_norm(residual)
        attention_output, first_value = official_training._train_temp_attention_forward(
            layer.attn,
            attention_input,
            first_value,
            native_lora_math=True,
        )
        hidden = residual + attention_output
        ffn_input = layer.ffn_norm(hidden)
        # Preserve the canonical fixed-row HF ChannelMix contract without
        # recursively entering the optional stateless-linear dispatcher.
        ffn_output = channel_mix(layer.ffn, ffn_input)
        return hidden + ffn_output, first_value

    checkpointing = bool(request.get("gradient_checkpointing"))
    for layer in owner.model.layers:
        if checkpointing:
            hidden_states, v_first = official_training._train_temp_checkpoint(
                lambda hidden, first, current=layer: run_layer(current, hidden, first),
                hidden_states,
                v_first,
            )
        else:
            hidden_states, v_first = run_layer(layer, hidden_states, v_first)

    hidden_states = owner.model.norm(hidden_states)
    full_logits = module_linear(owner.lm_head, hidden_states)
    labels = request.get("labels")
    loss = None
    if labels is not None:
        # Preserve standard HF causal CE. The historical fused loss adds
        # L2Wrap to the gradient and therefore remains a separate leaf rather
        # than silently changing the public model loss.
        loss = causal_cross_entropy(full_logits, labels)

    keep = request.get("logits_to_keep")
    logits = full_logits
    if isinstance(keep, torch.Tensor):
        logits = logits.index_select(1, keep.to(logits.device))
    elif keep is not None and int(keep) > 0:
        logits = logits[:, -min(int(keep), int(logits.shape[1])) :]

    return {
        "output_kind": "causal_lm",
        "logits": logits,
        "loss": loss,
        "past_key_values": None,
        "hidden_states": None,
        "implementation": IMPLEMENTATION,
        "phase": "training",
    }


__all__: list[str] = []
