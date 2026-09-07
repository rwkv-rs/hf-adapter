# coding=utf-8
"""Configuration for the readable, pure-PyTorch RWKV-7 HF reference model."""
from __future__ import annotations

from transformers import PretrainedConfig


class RWKV7Config(PretrainedConfig):
    """Describe an RWKV-7 checkpoint without selecting a hardware backend.

    The names intentionally match the official RWKV-7 checkpoint
    configuration where practical. Performance policy is not part of the model
    contract: this configuration always selects the reference PyTorch model.
    """

    model_type = "rwkv7"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 65536,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_heads: int | None = None,
        num_attention_heads: int | None = None,
        head_dim: int | None = None,
        attention_hidden_size: int | None = None,
        intermediate_size: int | None = None,
        hidden_ratio: float = 4.0,
        decay_low_rank_dim: int = 64,
        gate_low_rank_dim: int = 128,
        a_low_rank_dim: int = 64,
        v_low_rank_dim: int = 32,
        value_dim: list[int] | None = None,
        norm_eps: float = 1e-5,
        norm_bias: bool = True,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        **kwargs,
    ):
        # The official trie vocabulary reserves id 0; using it as both the
        # generation sentinel and padding id matches converted checkpoints.
        kwargs.setdefault("pad_token_id", 0)
        kwargs.setdefault("eos_token_id", 0)
        kwargs.setdefault("bos_token_id", 1)
        kwargs.setdefault("tie_word_embeddings", False)
        kwargs.setdefault("architectures", ["RWKV7ForCausalLM"])
        kwargs.setdefault(
            "auto_map",
            {
                "AutoConfig": "configuration_rwkv7.RWKV7Config",
                "AutoModel": "modeling_rwkv7.RWKV7Model",
                "AutoModelForCausalLM": "modeling_rwkv7.RWKV7ForCausalLM",
            },
        )
        super().__init__(**kwargs)

        if num_heads is not None and num_attention_heads is not None:
            if int(num_heads) != int(num_attention_heads):
                raise ValueError(
                    "num_heads and num_attention_heads must match when both are provided"
                )
        resolved_heads = num_heads if num_heads is not None else num_attention_heads
        attention_width = int(attention_hidden_size or hidden_size)
        if resolved_heads is None and head_dim is None:
            head_dim = 64 if attention_width % 64 == 0 else attention_width
        if resolved_heads is None:
            if attention_width % int(head_dim):
                raise ValueError("attention_hidden_size must be divisible by head_dim")
            resolved_heads = attention_width // int(head_dim)
        if head_dim is None:
            if attention_width % int(resolved_heads):
                raise ValueError("attention_hidden_size must be divisible by num_heads")
            head_dim = attention_width // int(resolved_heads)
        if attention_width != int(resolved_heads) * int(head_dim):
            raise ValueError("attention_hidden_size must equal num_heads * head_dim")

        self.vocab_size = int(vocab_size)
        self.hidden_size = int(hidden_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_heads = int(resolved_heads)
        self.num_attention_heads = self.num_heads
        self.head_dim = int(head_dim)
        self.attention_hidden_size = attention_width
        self.hidden_ratio = float(hidden_ratio)
        self.intermediate_size = int(
            intermediate_size
            if intermediate_size is not None
            else round(self.hidden_size * self.hidden_ratio)
        )
        self.decay_low_rank_dim = int(decay_low_rank_dim)
        self.gate_low_rank_dim = int(gate_low_rank_dim)
        self.a_low_rank_dim = int(a_low_rank_dim)
        self.v_low_rank_dim = int(v_low_rank_dim)
        self.value_dim = (
            [self.attention_hidden_size] * self.num_hidden_layers
            if value_dim is None
            else [int(value) for value in value_dim]
        )
        if len(self.value_dim) != self.num_hidden_layers:
            raise ValueError("value_dim must contain one entry per hidden layer")
        if any(value != self.attention_hidden_size for value in self.value_dim):
            raise ValueError(
                "the reference implementation currently requires value_dim == "
                "attention_hidden_size in every layer"
            )
        self.norm_eps = float(norm_eps)
        self.norm_bias = bool(norm_bias)
        self.initializer_range = float(initializer_range)
        self.use_cache = bool(use_cache)


try:
    RWKV7Config.register_for_auto_class()
except Exception:  # pragma: no cover - older Transformers
    pass


__all__ = ["RWKV7Config"]
