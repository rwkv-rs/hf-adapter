# coding=utf-8
"""Configuration contract for the native RWKV-7 Hugging Face model."""
from __future__ import annotations

from transformers import PretrainedConfig


def _resolve_num_heads(kwargs: dict) -> int | None:
    """Resolve the RWKV and Transformers head-count spellings."""

    num_heads = kwargs.get("num_heads")
    num_attention_heads = kwargs.get("num_attention_heads")
    if (
        num_heads is not None
        and num_attention_heads is not None
        and int(num_heads) != int(num_attention_heads)
    ):
        raise ValueError(
            "num_heads and num_attention_heads must match when both are provided"
        )
    resolved = num_heads if num_heads is not None else num_attention_heads
    return None if resolved is None else int(resolved)


class NativeRWKV7Config(PretrainedConfig):
    """Standalone RWKV-7 config carrying converted checkpoint fields."""

    model_type = "rwkv7_native"

    # Hugging Face's native ``tp_plan`` shards the large dense weights while
    # preserving RWKV's recurrent math.  Attention projections are gathered
    # before WKV because the eager/native kernels currently consume complete
    # head tensors; the output projection then splits that replicated tensor.
    # FFN key/value use the conventional colwise -> rowwise pair.  This is real
    # weight tensor parallelism (including the vocabulary tables), not an
    # Accelerate layer/device_map split.
    base_model_tp_plan = {
        "embeddings": "embedding_rowwise",
        "layers.*.attn.r_proj": "colwise_gather_output",
        "layers.*.attn.k_proj": "colwise_gather_output",
        "layers.*.attn.v_proj": "colwise_gather_output",
        "layers.*.attn.o_proj": "rowwise_split_input",
        # Transformers normalizes every numeric ModuleList/Sequential index to
        # ``*`` when matching TP rules.  Both LoRA linears are therefore
        # sharded-and-gathered; the no-op index 1 is skipped by _LoRA.forward.
        "layers.*.attn.w_lora.lora.*": "colwise_gather_output",
        "layers.*.attn.a_lora.lora.*": "colwise_gather_output",
        "layers.*.attn.g_lora.lora.*": "colwise_gather_output",
        "layers.*.attn.v_lora.lora.*": "colwise_gather_output",
        "layers.*.ffn.key": "colwise",
        "layers.*.ffn.value": "rowwise",
    }

    def __init__(self, **kwargs):
        # RWKV checkpoints have an independent output head. PretrainedConfig
        # otherwise defaults this to True, which makes from_pretrained replace
        # lm_head with the embedding matrix before native MM packing.
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)
        self.vocab_size = kwargs.get("vocab_size", 65536)
        self.hidden_size = kwargs.get("hidden_size", 768)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 12)
        self.num_heads = _resolve_num_heads(kwargs)
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
        # FP8 E4M3 quantization (requires Hopper/Ada/Blackwell for _scaled_mm)
        self.use_native_fp8 = kwargs.get("use_native_fp8", False)
        self.native_fp8_min_params = kwargs.get("native_fp8_min_params", 8_000_000)
        self.native_fp8_policy = kwargs.get("native_fp8_policy", "memory")
        if getattr(self, "auto_map", None) is None:
            self.auto_map = {
                "AutoConfig": "native_model.NativeRWKV7Config",
                "AutoModel": "native_model.NativeRWKV7Model",
                "AutoModelForCausalLM": "native_model.NativeRWKV7ForCausalLM",
            }


__all__ = ["NativeRWKV7Config"]
