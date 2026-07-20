"""Direct whole-model training runtime for vendored train_temp autograd ops.

Unlike the historical adapter, this runtime never replaces ``forward`` methods
or adds backend flags to model/config/cache objects. The clean HF model invokes
it once through the versioned model-forward protocol.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from . import train_temp_cuda as train_temp
from .training_math import channel_mix, module_linear


IMPLEMENTATION = "native-nvidia-train-temp-autograd-v2"


def run_training(owner: Any, request: dict[str, Any]) -> dict[str, Any]:
    """Execute dense BF16 forward/backward-capable RWKV-7 training math."""

    train_temp.load_train_temp_cuda_extension()
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
        residual = (
            layer.pre_norm(hidden) if hasattr(layer, "pre_norm") else hidden
        )
        attention_input = layer.attn_norm(residual)
        attention_output, first_value = train_temp._train_temp_attention_forward(
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
            hidden_states, v_first = train_temp._train_temp_checkpoint(
                lambda hidden, first, current=layer: run_layer(
                    current, hidden, first
                ),
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
        shifted_logits = full_logits[:, :-1].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        if shifted_logits.numel() == 0 or not bool(
            (shifted_labels != -100).any().detach().cpu()
        ):
            loss = full_logits.float().sum() * 0.0
        else:
            # Preserve the public HF loss exactly. The historical train_temp
            # fused loss adds L2Wrap to the gradient and is therefore exposed
            # only as a leaf operator, never substituted for standard CE.
            loss = F.cross_entropy(
                shifted_logits.view(-1, shifted_logits.shape[-1]).float(),
                shifted_labels.reshape(-1),
                ignore_index=-100,
            )

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


__all__ = ["IMPLEMENTATION", "run_training"]
