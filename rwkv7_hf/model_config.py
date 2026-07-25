# coding=utf-8
"""Configuration contract for the native RWKV-7 Hugging Face model."""
from __future__ import annotations

from transformers import PretrainedConfig


class NativeRWKV7Config(PretrainedConfig):
    """Standalone RWKV-7 config carrying converted checkpoint fields."""

    model_type = "rwkv7_native"

    def __init__(self, **kwargs):
        # RWKV checkpoints have an independent output head. PretrainedConfig
        # otherwise defaults this to True, which makes from_pretrained replace
        # lm_head with the embedding matrix before native MM packing.
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)
        self.vocab_size = kwargs.get("vocab_size", 65536)
        self.hidden_size = kwargs.get("hidden_size", 768)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 12)
        self.num_heads = kwargs.get("num_heads", None) or kwargs.get("num_attention_heads", None)
        requested_attention_width = int(
            kwargs.get("attention_hidden_size", self.hidden_size)
        )
        requested_head_dim = kwargs.get("head_dim", None)
        if self.num_heads is None and requested_head_dim is None:
            requested_head_dim = (
                64 if requested_attention_width % 64 == 0 else requested_attention_width
            )
        if requested_head_dim is None:
            if requested_attention_width % int(self.num_heads):
                raise ValueError("attention_hidden_size must be divisible by num_heads")
            requested_head_dim = requested_attention_width // int(self.num_heads)
        self.head_dim = int(requested_head_dim)
        if self.num_heads is None:
            if requested_attention_width % self.head_dim:
                raise ValueError("attention_hidden_size must be divisible by head_dim")
            self.num_heads = requested_attention_width // self.head_dim
        self.attention_hidden_size = int(
            kwargs.get("attention_hidden_size", self.num_heads * self.head_dim)
        )
        if self.attention_hidden_size != int(self.num_heads) * int(self.head_dim):
            raise ValueError("attention_hidden_size must equal num_heads * head_dim")
        self.num_attention_heads = self.num_heads
        self.intermediate_size = kwargs.get("intermediate_size", self.hidden_size * 4)
        self.decay_low_rank_dim = kwargs.get("decay_low_rank_dim", 64)
        self.gate_low_rank_dim = kwargs.get("gate_low_rank_dim", 128)
        self.a_low_rank_dim = kwargs.get("a_low_rank_dim", 64)
        self.v_low_rank_dim = kwargs.get("v_low_rank_dim", 32)
        self.layer_types = kwargs.get("layer_types", None)
        self.use_cache = kwargs.get("use_cache", True)
        self.use_native_mm8 = kwargs.get("use_native_mm8", False)
        self.native_mm8_min_params = kwargs.get("native_mm8_min_params", 8_000_000)
        self.native_mm8_policy = kwargs.get("native_mm8_policy", "memory")
        self.use_native_mm4 = kwargs.get("use_native_mm4", False)
        self.native_mm4_min_params = kwargs.get("native_mm4_min_params", 8_000_000)
        self.native_mm4_policy = kwargs.get("native_mm4_policy", "memory")
        self.native_mm4_group_size = kwargs.get("native_mm4_group_size", 0)
        self.native_mm4_group_policy = kwargs.get("native_mm4_group_policy", "all")
        if getattr(self, "auto_map", None) is None:
            self.auto_map = {
                "AutoConfig": "native_model.NativeRWKV7Config",
                "AutoModel": "native_model.NativeRWKV7Model",
                "AutoModelForCausalLM": "native_model.NativeRWKV7ForCausalLM",
            }


__all__ = ["NativeRWKV7Config"]
