"""Structural packing for the clean RWKV-7 Hugging Face module tree.

The optional package deliberately uses duck typing here: it never imports the
adapter's modeling module and it never owns a second model/config/cache class.
Every packed tensor remains the original Parameter, so save/load and optimizer
ownership stay with the Hugging Face model.
"""
from __future__ import annotations

from typing import Any

import torch


_PACK_CACHE_ATTR = "_rwkv7_kernel_dense_pack_v2"


def _required_tensor(value: Any, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"RWKV7 kernel packing requires tensor {name}")
    return value


def extract_dense_packs(model: Any):
    """Return the tensor-only ABI consumed by :func:`block_step_batched`."""

    cached = getattr(model, _PACK_CACHE_ATTR, None)
    if cached is not None:
        return cached

    layers = model.layers
    if not layers:
        raise ValueError("RWKV7 optimized model requires at least one layer")
    first_attention = layers[0].attn
    heads = int(first_attention.num_heads)
    head_dim = int(first_attention.head_dim)
    hidden = int(first_attention.hidden_size)
    attention_hidden = int(first_attention.attention_hidden_size)
    if attention_hidden != heads * head_dim:
        raise ValueError("attention_hidden_size must equal num_heads * head_dim")
    norm_eps = float(model.config.norm_eps)
    group_eps = float(head_dim * norm_eps)
    reference = model.embeddings.weight
    empty = reference.new_empty((0,))

    packs = []
    for layer_idx, layer in enumerate(layers):
        attention = layer.attn
        value_lora = getattr(attention, "v_lora", None)
        if value_lora is None:
            value_lora_1 = torch.zeros(
                1, hidden, device=reference.device, dtype=reference.dtype
            )
            value_lora_2 = torch.zeros(
                attention_hidden, 1, device=reference.device, dtype=reference.dtype
            )
            value_lora_bias = torch.zeros(
                attention_hidden, device=reference.device, dtype=reference.dtype
            )
        else:
            value_lora_1 = value_lora.lora[0].weight
            value_lora_2 = value_lora.lora[2].weight
            value_lora_bias = value_lora.lora[2].bias

        if hasattr(layer, "pre_norm"):
            pre_weight = layer.pre_norm.weight
            pre_bias = layer.pre_norm.bias
            has_pre = 1
        else:
            pre_weight = torch.zeros(
                hidden, device=reference.device, dtype=reference.dtype
            )
            pre_bias = torch.zeros_like(pre_weight)
            has_pre = 0

        named = {
            "pre_norm.bias": pre_bias,
            "attn_norm.bias": layer.attn_norm.bias,
            "ffn_norm.bias": layer.ffn_norm.bias,
            "w_lora.bias": attention.w_lora.lora[2].bias,
            "a_lora.bias": attention.a_lora.lora[2].bias,
            "v_lora.bias": value_lora_bias,
            "g_norm.bias": attention.g_norm.bias,
        }
        for name, value in named.items():
            _required_tensor(value, f"layers.{layer_idx}.{name}")

        packs.append(
            (
                layer_idx,
                heads,
                head_dim,
                group_eps,
                norm_eps,
                has_pre,
                pre_weight,
                pre_bias,
                layer.attn_norm.weight,
                layer.attn_norm.bias,
                layer.ffn_norm.weight,
                layer.ffn_norm.bias,
                attention.x_r.reshape(-1),
                attention.x_w.reshape(-1),
                attention.x_k.reshape(-1),
                attention.x_v.reshape(-1),
                attention.x_a.reshape(-1),
                attention.x_g.reshape(-1),
                attention.k_k,
                attention.k_a,
                attention.r_k,
                attention.r_proj.weight,
                attention.k_proj.weight,
                attention.v_proj.weight,
                attention.o_proj.weight,
                attention.w_lora.lora[0].weight,
                attention.w_lora.lora[2].weight,
                attention.w_lora.lora[2].bias,
                attention.a_lora.lora[0].weight,
                attention.a_lora.lora[2].weight,
                attention.a_lora.lora[2].bias,
                value_lora_1,
                value_lora_2,
                value_lora_bias,
                attention.g_lora.lora[0].weight,
                attention.g_lora.lora[2].weight,
                attention.g_norm.weight,
                attention.g_norm.bias,
                layer.ffn.x_k,
                layer.ffn.key.weight,
                layer.ffn.value.weight,
                empty,
            )
        )

    result = (packs, heads, head_dim)
    setattr(model, _PACK_CACHE_ATTR, result)
    return result


__all__ = ["extract_dense_packs"]
