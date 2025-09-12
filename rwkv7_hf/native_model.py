# coding=utf-8
"""Experimental native RWKV-7 PyTorch model (no mandatory FLA runtime).

This is the H1 native-Transformers track: a correctness-first RWKV-7 model
implemented with plain ``torch.nn`` modules and the math in ``rwkv7_hf.native``.
It loads the same converted HF checkpoints as the production FLA-backed wrapper,
so it can serve as the long-term upstream / AMD / CPU fallback base.

Important: this module is intentionally experimental and sequential. It is not a
replacement for the optimized wrapper path yet. Current scope is batched forward,
incremental greedy generation, and regression tests against the wrapper.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .native import _init_state_batched, _step_token_batched

_FALSE_VALUES = {"0", "false", "False", "no", "off"}

try:
    from .native_jit import extract as _native_jit_extract
    from .native_jit import step_batched as _native_jit_step_batched
except Exception:  # pragma: no cover - optional native acceleration
    _native_jit_extract = None
    _native_jit_step_batched = None


def _native_model_jit_enabled() -> bool:
    return os.environ.get("RWKV7_NATIVE_MODEL_JIT", "1") not in _FALSE_VALUES


class NativeRWKV7Config(PretrainedConfig):
    """Standalone RWKV-7 config carrying converted checkpoint fields."""

    model_type = "rwkv7_native"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = kwargs.get("hidden_size", 768)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 12)
        self.head_dim = kwargs.get("head_dim", 64)
        self.num_heads = kwargs.get("num_heads", None) or self.hidden_size // self.head_dim
        self.intermediate_size = kwargs.get("intermediate_size", self.hidden_size * 4)
        self.decay_low_rank_dim = kwargs.get("decay_low_rank_dim", 64)
        self.gate_low_rank_dim = kwargs.get("gate_low_rank_dim", 128)
        self.a_low_rank_dim = kwargs.get("a_low_rank_dim", 64)
        self.v_low_rank_dim = kwargs.get("v_low_rank_dim", 32)
        self.layer_types = kwargs.get("layer_types", None)
        self.use_cache = kwargs.get("use_cache", True)


class _LoRA(nn.Module):
    """Matches converted keys: ``*_lora.lora.{0,2}.weight`` / ``lora.2.bias``."""

    def __init__(self, hidden: int, low_rank: int, bias: bool):
        super().__init__()
        self.lora = nn.Sequential(
            nn.Linear(hidden, low_rank, bias=False),
            nn.Identity(),
            nn.Linear(low_rank, hidden, bias=bias),
        )

    def forward(self, x):
        return self.lora(x)


class NativeRWKV7Attention(nn.Module):
    """TMix module with attributes consumed by ``rwkv7_hf.native.attn_step``."""

    def __init__(self, config: NativeRWKV7Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        hidden = config.hidden_size
        for p in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g"):
            setattr(self, p, nn.Parameter(torch.zeros(1, 1, hidden)))
        self.k_k = nn.Parameter(torch.zeros(hidden))
        self.k_a = nn.Parameter(torch.zeros(hidden))
        self.r_k = nn.Parameter(torch.zeros(self.num_heads, self.head_dim))
        self.r_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)
        self.w_lora = _LoRA(hidden, config.decay_low_rank_dim, bias=True)
        self.a_lora = _LoRA(hidden, config.a_low_rank_dim, bias=True)
        self.g_lora = _LoRA(hidden, config.gate_low_rank_dim, bias=False)
        if layer_idx != 0:
            self.v_lora = _LoRA(hidden, config.v_low_rank_dim, bias=True)
        self.g_norm = nn.GroupNorm(self.num_heads, hidden, eps=self.head_dim * 1e-5)


class NativeRWKV7FFN(nn.Module):
    """CMix module with attributes consumed by ``rwkv7_hf.native.ffn_step``."""

    def __init__(self, config: NativeRWKV7Config):
        super().__init__()
        self.x_k = nn.Parameter(torch.zeros(config.hidden_size))
        self.key = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.value = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)


class NativeRWKV7Layer(nn.Module):
    def __init__(self, config: NativeRWKV7Config, layer_idx: int):
        super().__init__()
        self.attn = NativeRWKV7Attention(config, layer_idx)
        self.ffn = NativeRWKV7FFN(config)
        self.attn_norm = nn.LayerNorm(config.hidden_size)
        self.ffn_norm = nn.LayerNorm(config.hidden_size)
        if layer_idx == 0:
            self.pre_norm = nn.LayerNorm(config.hidden_size)


class NativeRWKV7Model(PreTrainedModel):
    config_class = NativeRWKV7Config
    base_model_prefix = "model"

    def __init__(self, config: NativeRWKV7Config):
        super().__init__(config)
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([NativeRWKV7Layer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size)


class NativeRWKV7ForCausalLM(PreTrainedModel, GenerationMixin):
    """Experimental batched native PyTorch CausalLM for converted RWKV-7 weights."""

    config_class = NativeRWKV7Config
    base_model_prefix = "model"
    _no_split_modules = ["NativeRWKV7Layer"]
    # Transformers >=5 expects dict-like _tied_weights_keys; RWKV-7 ties nothing.
    _tied_weights_keys = {}

    @property
    def all_tied_weights_keys(self):
        return {}

    def __init__(self, config: NativeRWKV7Config):
        super().__init__(config)
        self.model = NativeRWKV7Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def rwkv7_native_model_last_decode_backend(self) -> str | None:
        """Return the backend used by the previous native-model decode call."""
        return getattr(self, "_rwkv7_native_model_last_decode_backend", None)

    def _native_jit_packs(self):
        if not _native_model_jit_enabled() or _native_jit_extract is None or _native_jit_step_batched is None:
            return None
        weight = self.model.embeddings.weight
        key = (weight.device.type, weight.device.index, weight.dtype)
        cache = getattr(self, "_rwkv7_native_model_jit_pack_cache", None)
        if cache is None or cache[0] != key:
            extracted = _native_jit_extract(self)
            packs = extracted[0] if isinstance(extracted, tuple) and len(extracted) == 4 else extracted
            self._rwkv7_native_model_jit_pack_cache = (key, packs)
            return packs
        return cache[1]

    def _run(self, token_ids: torch.Tensor, state, xpa, xpf, v_first, *, use_jit: bool = False):
        """Sequentially advance over token ids shaped ``[batch, seq]``.

        The experimental native model is still correctness-first and sequential
        over time, but the per-token math is vectorized over batch rows. This
        keeps the native/upstream fallback path aligned with wrapper bsz tests
        without claiming it replaces the optimized wrapper backend.
        """
        if token_ids.dim() != 2:
            raise ValueError("NativeRWKV7ForCausalLM._run expects token ids shaped [batch, seq]")
        base = self.model
        x = None
        packs = self._native_jit_packs() if use_jit else None
        backend = "native_jit" if packs is not None else "eager"
        for t in range(token_ids.shape[1]):
            x = F.embedding(token_ids[:, t], base.embeddings.weight)
            if packs is not None:
                x, state, xpa, xpf, v_first = _native_jit_step_batched(self, x, state, xpa, xpf, v_first, packs)
            else:
                x, state, xpa, xpf, v_first = _step_token_batched(self, x, state, xpa, xpf, v_first)
        if x is None:
            raise ValueError("NativeRWKV7ForCausalLM requires at least one token")
        if use_jit:
            self._rwkv7_native_model_last_decode_backend = backend
        x = base.norm(x)
        logits = F.linear(x, self.lm_head.weight).view(token_ids.shape[0], 1, -1)
        return logits, state, xpa, xpf, v_first

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        past_key_values=None,
        use_cache: bool | None = None,
        return_dict: bool | None = True,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("NativeRWKV7ForCausalLM currently requires input_ids")
        if input_ids.dim() != 2:
            raise ValueError("Experimental NativeRWKV7ForCausalLM expects input_ids shaped [batch, seq]")
        if int(input_ids.shape[0]) <= 0 or int(input_ids.shape[1]) <= 0:
            raise ValueError("NativeRWKV7ForCausalLM requires a non-empty batch and sequence")
        use_cache = bool(self.config.use_cache if use_cache is None else use_cache)
        base = self.model
        device, dtype = input_ids.device, base.embeddings.weight.dtype
        if past_key_values is None:
            state, xpa, xpf, v_first = _init_state_batched(self, input_ids.shape[0], device, dtype)
            toks = input_ids
            use_jit = False
        else:
            state, xpa, xpf, v_first = past_key_values
            toks = input_ids[:, -1:]
            use_jit = True
        logits, state, xpa, xpf, v_first = self._run(toks, state, xpa, xpf, v_first, use_jit=use_jit)
        new_cache = (state, xpa, xpf, v_first) if use_cache else None
        if not return_dict:
            return (logits, new_cache) if use_cache else (logits,)
        return CausalLMOutputWithPast(logits=logits, past_key_values=new_cache)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx: torch.LongTensor):
        """Minimal beam/select helper for experimental batched native caches."""
        if past_key_values is None:
            return None
        state, xpa, xpf, v_first = past_key_values
        index = beam_idx.to(v_first.device)
        return (
            [s.index_select(0, index.to(s.device)) for s in state],
            [x.index_select(0, index.to(x.device)) for x in xpa],
            [x.index_select(0, index.to(x.device)) for x in xpf],
            v_first.index_select(0, index),
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        # Ensure GenerationMixin gets a cache on the first step. Earlier H1 code
        # only enabled cache after a cache already existed, causing full-prefix
        # recomputation on every greedy token.
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        use_cache = kwargs.get("use_cache", True)
        if use_cache is None:
            use_cache = True
        return {"input_ids": input_ids, "past_key_values": past_key_values, "use_cache": use_cache}
