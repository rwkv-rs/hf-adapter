# coding=utf-8
"""Remote-code wrapper around FLA RWKV7 HF modules.

Requires flash-linear-attention (`fla`) on PYTHONPATH / installed in the env.
"""
from __future__ import annotations

import os
from typing import Any

import torch
from fla.models.rwkv7.modeling_rwkv7 import RWKV7Model as _RWKV7Model
from fla.models.rwkv7.modeling_rwkv7 import RWKV7ForCausalLM as _RWKV7ForCausalLM
from fla.models.utils import Cache as _FLACache

try:
    from .configuration_rwkv7 import RWKV7Config
except ImportError:  # pragma: no cover - direct remote-file execution fallback
    from configuration_rwkv7 import RWKV7Config


_FALSE_VALUES = {"0", "false", "False", "no", "off"}


def _fast_cache_enabled() -> bool:
    """Runtime switch used by benchmarks to compare cache implementations."""
    return os.environ.get("RWKV7_FAST_CACHE", "1") not in _FALSE_VALUES


def _move_first_dim(value: Any, indices: torch.LongTensor) -> Any:
    """Reorder nested tensor state along batch dimension for HF beam helpers."""
    if isinstance(value, torch.Tensor):
        return value.index_select(0, indices.to(value.device))
    if isinstance(value, tuple):
        return tuple(_move_first_dim(v, indices) for v in value)
    if isinstance(value, list):
        return [_move_first_dim(v, indices) for v in value]
    if isinstance(value, dict):
        return {k: _move_first_dim(v, indices) for k, v in value.items()}
    return value


class RWKV7StateCache(_FLACache):
    """Lightweight recurrent-state cache for RWKV-7 inference.

    FLA's default cache mirrors the evolving Transformers CacheLayer API and is
    intentionally generic. RWKV-7 decode only needs one dictionary per layer
    (`recurrent_state`, `conv_state`, `ffn_state`, and optional `attn_state`), so
    this cache keeps the legacy list-of-dicts layout while still subclassing the
    FLA `Cache` class. That makes it accepted by FLA layers without a conversion
    step and removes per-token CacheLayer bookkeeping from the hot path.
    """

    is_compileable = True

    def __init__(self, seen_tokens: int = 0, **_: Any) -> None:
        # Do not call _FLACache.__init__(): it allocates HF CacheLayer wrappers
        # that are unnecessary for RWKV recurrent decode and add CPU overhead.
        self.states: list[dict[str, Any]] = []
        self._seen_tokens = int(seen_tokens)

    def __getitem__(self, layer_idx: int) -> dict[str, Any]:
        if layer_idx < len(self.states):
            return self.states[layer_idx]
        raise KeyError(f"Cache only has {len(self.states)} layers, attempted to access layer {layer_idx}")

    def __iter__(self):
        yield from self.states

    def __len__(self) -> int:
        return len(self.states)

    def update(
        self,
        recurrent_state: Any | None = None,
        attn_state: Any | None = None,
        conv_state: Any | None = None,
        ffn_state: Any | None = None,
        layer_idx: int = 0,
        offset: int | None = 1,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if cache_kwargs is None:
            cache_kwargs = {}
        offset = 1 if offset is None else int(offset)
        input_size = attn_state[0].shape[1] if attn_state is not None else 0
        window_size = cache_kwargs.get("window_size")

        if len(self.states) <= layer_idx:
            while len(self.states) < layer_idx:
                self.states.append({"recurrent_state": None, "attn_state": None, "conv_state": None, "ffn_state": None})
            if layer_idx == 0:
                self._seen_tokens += offset
            if attn_state is not None and window_size is not None and input_size > window_size:
                attn_state = [state[:, -window_size:].contiguous() for state in attn_state]
            state = {
                "recurrent_state": recurrent_state,
                "attn_state": attn_state,
                "conv_state": conv_state,
                "ffn_state": ffn_state,
            }
            self.states.append(state)
            return state

        state = self.states[layer_idx]
        if layer_idx == len(self.states) - 1:
            self._seen_tokens += offset
        if recurrent_state is not None:
            state["recurrent_state"] = recurrent_state
        if attn_state is not None:
            if state.get("attn_state") is None:
                state["attn_state"] = [
                    new_state[:, -window_size:].contiguous()
                    if window_size is not None and new_state.shape[1] > window_size
                    else new_state
                    for new_state in attn_state
                ]
            elif window_size is not None and input_size == 0:
                pass
            elif window_size is not None and state["attn_state"][0].shape[1] >= window_size:
                updated_attn_state = []
                for old_state, new_state in zip(state["attn_state"], attn_state, strict=False):
                    tail = new_state[:, -window_size:]
                    if tail.shape[1] >= window_size:
                        updated_attn_state.append(tail.contiguous())
                    else:
                        old_state = old_state[:, -window_size:].contiguous() if old_state.shape[1] > window_size else old_state
                        old_state = old_state.roll(-input_size, 1)
                        old_state[:, -tail.shape[1]:] = tail
                        updated_attn_state.append(old_state)
                state["attn_state"] = updated_attn_state
            else:
                updated_attn_state = []
                for old_state, new_state in zip(state["attn_state"], attn_state, strict=False):
                    updated = torch.cat([old_state, new_state], 1)
                    if window_size is not None and updated.shape[1] > window_size:
                        updated = updated[:, -window_size:].contiguous()
                    updated_attn_state.append(updated)
                state["attn_state"] = updated_attn_state
        if conv_state is not None:
            state["conv_state"] = conv_state
        if ffn_state is not None:
            state["ffn_state"] = ffn_state
        return state

    def get_seq_length(self, layer_idx: int | None = 0, cache_position=None) -> int:
        if len(self.states) <= (layer_idx or 0):
            return 0
        return self._seen_tokens

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return -1

    def get_mask_sizes(self, cache_position: torch.Tensor | None, layer_idx: int = 0) -> tuple[int, int]:
        query_len = int(cache_position.shape[0]) if cache_position is not None else 0
        return int(self.get_seq_length(layer_idx)) + query_len, 0

    def reset(self) -> None:
        self.states.clear()
        self._seen_tokens = 0

    def to_legacy_cache(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.states)

    def reorder_cache(self, beam_idx: torch.LongTensor):
        self.states = [_move_first_dim(state, beam_idx) for state in self.states]
        return self

    @classmethod
    def from_legacy_cache(
        cls,
        past_key_values: Any | None = None,
        seen_tokens: int = 0,
        **kwargs: Any,
    ) -> "RWKV7StateCache":
        if isinstance(past_key_values, cls):
            return past_key_values
        cache = cls(seen_tokens=seen_tokens, **kwargs)
        if isinstance(past_key_values, _FLACache) and hasattr(past_key_values, "to_legacy_cache"):
            past_key_values = past_key_values.to_legacy_cache()
        if isinstance(past_key_values, (list, tuple)):
            empty = {"recurrent_state": None, "attn_state": None, "conv_state": None, "ffn_state": None}
            cache.states = [dict(state) if state is not None else dict(empty) for state in past_key_values]
        return cache


class RWKV7Model(_RWKV7Model):
    config_class = RWKV7Config



class RWKV7ForCausalLM(_RWKV7ForCausalLM):
    config_class = RWKV7Config
    # Transformers >=5 expects dict-like _tied_weights_keys in save_pretrained.
    _tied_weights_keys = {}

    def forward(self, *args, **kwargs):
        use_cache = kwargs.get("use_cache")
        effective_use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        if effective_use_cache and _fast_cache_enabled():
            past_key_values = kwargs.get("past_key_values")
            if not isinstance(past_key_values, RWKV7StateCache):
                kwargs["past_key_values"] = RWKV7StateCache.from_legacy_cache(past_key_values)
        return super().forward(*args, **kwargs)
