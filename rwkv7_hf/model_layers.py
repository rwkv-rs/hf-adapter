# coding=utf-8
"""Native RWKV-7 attention, FFN, and block module definitions."""
from __future__ import annotations

import torch
import torch.nn as nn

from .model_config import NativeRWKV7Config
from .native import attn_step_batched, ffn_step_batched


class _LoRA(nn.Module):
    """Matches converted keys: ``*_lora.lora.{0,2}.weight`` / ``lora.2.bias``."""

    def __init__(
        self,
        input_size: int,
        low_rank: int,
        bias: bool,
        *,
        output_size: int | None = None,
    ):
        super().__init__()
        output_size = input_size if output_size is None else int(output_size)
        self.lora = nn.Sequential(
            nn.Linear(input_size, low_rank, bias=False),
            nn.Identity(),
            nn.Linear(low_rank, output_size, bias=bias),
        )

    def forward(self, x):
        # Index 1 is Identity and exists only to preserve official checkpoint
        # keys.  Calling the two linears explicitly also avoids attaching a TP
        # gather hook to the parameter-free identity wildcard match.
        return self.lora[2](self.lora[0](x))


class NativeRWKV7Attention(nn.Module):
    """TMix module with attributes consumed by ``rwkv7_hf.native.attn_step``."""

    def __init__(self, config: NativeRWKV7Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.attention_hidden_size = getattr(
            config,
            "attention_hidden_size",
            config.num_heads * config.head_dim,
        )
        hidden = config.hidden_size
        attention_hidden = self.attention_hidden_size
        for p in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"):
            setattr(self, p, nn.Parameter(torch.zeros(1, 1, hidden)))
        self.k_k = nn.Parameter(torch.zeros(attention_hidden))
        self.k_a = nn.Parameter(torch.zeros(attention_hidden))
        self.r_k = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.r_proj = nn.Linear(hidden, attention_hidden, bias=False)
        self.k_proj = nn.Linear(hidden, attention_hidden, bias=False)
        self.v_proj = nn.Linear(hidden, attention_hidden, bias=False)
        self.o_proj = nn.Linear(attention_hidden, hidden, bias=False)
        self.w_lora = _LoRA(
            hidden, config.decay_low_rank_dim, bias=True, output_size=attention_hidden
        )
        self.a_lora = _LoRA(
            hidden, config.a_low_rank_dim, bias=True, output_size=attention_hidden
        )
        self.g_lora = _LoRA(
            hidden, config.gate_low_rank_dim, bias=False, output_size=attention_hidden
        )
        if layer_idx != 0:
            self.v_lora = _LoRA(
                hidden, config.v_low_rank_dim, bias=True, output_size=attention_hidden
            )
        self.g_norm = nn.GroupNorm(
            self.num_heads, attention_hidden, eps=self.head_dim * 1e-5
        )

    def forward(
        self,
        x: torch.Tensor,
        x_prev: torch.Tensor | None = None,
        v_first: torch.Tensor | None = None,
        state: torch.Tensor | None = None,
    ):
        """Run one native attention step through ``Module.__call__``.

        DeepSpeed ZeRO-3 gathers partitioned parameters from module pre-forward
        hooks.  The original native loop passed ``self`` into the functional
        helper directly, which bypassed this module call for raw TMix
        parameters such as ``x_r`` / ``r_k`` / ``g_norm.weight`` and left them
        sharded under ZeRO-3.  Keeping this thin forward wrapper makes the same
        math usable for normal eager execution and ZeRO-3 resume training.
        """
        train_temp_forward = getattr(self, "_rwkv7_train_temp_forward", None)
        if callable(train_temp_forward):
            return train_temp_forward(x, x_prev)
        if x_prev is None or v_first is None or state is None:
            raise ValueError("native token attention requires x_prev, v_first, and recurrent state")
        return attn_step_batched(self, self.layer_idx, x, x_prev, v_first, state)


class NativeRWKV7FFN(nn.Module):
    """CMix module with attributes consumed by ``rwkv7_hf.native.ffn_step``."""

    def __init__(self, config: NativeRWKV7Config):
        super().__init__()
        self.x_k = nn.Parameter(torch.zeros(config.hidden_size))
        self.key = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.value = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, x_prev: torch.Tensor | None = None):
        """Run one native FFN step through ``Module.__call__`` for ZeRO-3 hooks."""
        train_temp_forward = getattr(self, "_rwkv7_train_temp_forward", None)
        if callable(train_temp_forward):
            return train_temp_forward(x)
        if x_prev is None:
            raise ValueError("native token FFN requires x_prev recurrent state")
        return ffn_step_batched(self, x, x_prev)


class NativeRWKV7Layer(nn.Module):
    def __init__(self, config: NativeRWKV7Config, layer_idx: int):
        super().__init__()
        self.attn = NativeRWKV7Attention(config, layer_idx)
        self.ffn = NativeRWKV7FFN(config)
        self.attn_norm = nn.LayerNorm(config.hidden_size)
        self.ffn_norm = nn.LayerNorm(config.hidden_size)
        if layer_idx == 0:
            self.pre_norm = nn.LayerNorm(config.hidden_size)


__all__ = [
    "NativeRWKV7Attention",
    "NativeRWKV7FFN",
    "NativeRWKV7Layer",
    "_LoRA",
]
