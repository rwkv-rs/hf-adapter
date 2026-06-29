# coding=utf-8
"""Recurrent cache used by the pure-PyTorch RWKV-7 reference model."""
from __future__ import annotations

from collections.abc import Iterable

import torch

try:
    from transformers.cache_utils import Cache as _HFCache
except Exception:  # pragma: no cover - old Transformers fallback
    class _HFCache:  # type: ignore[no-redef]
        pass


def _map_tensor_list(values, fn):
    return [None if value is None else fn(value) for value in values]


class RWKV7Cache(_HFCache):
    """Canonical RWKV-7 state cache.

    Every layer owns exactly three tensors: recurrent_state has shape
    [batch, heads, key_dim, value_dim], and attention_shift / ffn_shift both
    have shape [batch, hidden]. v_first is deliberately absent because it only
    connects layers during one forward call.
    """

    def __init__(
        self,
        recurrent_state: Iterable[torch.Tensor | None] | None = None,
        attention_shift: Iterable[torch.Tensor | None] | None = None,
        ffn_shift: Iterable[torch.Tensor | None] | None = None,
        *,
        num_layers: int | None = None,
        seen_tokens: int = 0,
    ):
        recurrent = list(recurrent_state or [])
        attention = list(attention_shift or [])
        ffn = list(ffn_shift or [])
        inferred = max(len(recurrent), len(attention), len(ffn), int(num_layers or 0))
        recurrent.extend([None] * (inferred - len(recurrent)))
        attention.extend([None] * (inferred - len(attention)))
        ffn.extend([None] * (inferred - len(ffn)))
        self.recurrent_state = recurrent
        self.attention_shift = attention
        self.ffn_shift = ffn
        self._seen_tokens = int(seen_tokens)

    def __len__(self) -> int:
        return len(self.recurrent_state)

    def __iter__(self):
        return iter(zip(self.recurrent_state, self.attention_shift, self.ffn_shift))

    def __getitem__(self, layer_idx: int):
        return {
            "recurrent_state": self.recurrent_state[layer_idx],
            "attention_shift": self.attention_shift[layer_idx],
            "ffn_shift": self.ffn_shift[layer_idx],
        }

    @property
    def seen_tokens(self) -> int:
        return self._seen_tokens

    @seen_tokens.setter
    def seen_tokens(self, value: int) -> None:
        self._seen_tokens = int(value)

    @property
    def states(self):
        return [self[index] for index in range(len(self))]

    def get_seq_length(self, layer_idx: int = 0, cache_position=None) -> int:
        del layer_idx, cache_position
        return self._seen_tokens

    def get_max_length(self, layer_idx: int = 0):
        del layer_idx
        return None

    def get_max_cache_shape(self, layer_idx: int = 0):
        del layer_idx
        return None

    def get_mask_sizes(self, cache_position, layer_idx: int = 0):
        del layer_idx
        query_length = (
            int(cache_position.numel())
            if isinstance(cache_position, torch.Tensor) and cache_position.numel()
            else 1
        )
        return self._seen_tokens + query_length, 0

    def is_initialized(self, layer_idx: int | None = None) -> bool:
        if layer_idx is not None:
            return (
                0 <= int(layer_idx) < len(self)
                and self.recurrent_state[int(layer_idx)] is not None
            )
        return bool(self.recurrent_state) and all(
            value is not None for value in self.recurrent_state
        )

    def _is_initialized(self, layer_idx: int | None = None) -> bool:
        return self.is_initialized(layer_idx)

    @property
    def max_batch_size(self) -> int | None:
        return self.get_batch_size()

    def get_batch_size(self) -> int | None:
        for collection in (
            self.recurrent_state,
            self.attention_shift,
            self.ffn_shift,
        ):
            for value in collection:
                if value is not None:
                    return int(value.shape[0])
        return None

    def set_layer(
        self,
        layer_idx: int,
        recurrent_state: torch.Tensor,
        attention_shift: torch.Tensor,
        ffn_shift: torch.Tensor,
    ) -> None:
        layer_idx = int(layer_idx)
        if layer_idx < 0:
            raise IndexError("layer_idx must be non-negative")
        missing = layer_idx + 1 - len(self)
        if missing > 0:
            self.recurrent_state.extend([None] * missing)
            self.attention_shift.extend([None] * missing)
            self.ffn_shift.extend([None] * missing)
        self.recurrent_state[layer_idx] = recurrent_state
        self.attention_shift[layer_idx] = attention_shift
        self.ffn_shift[layer_idx] = ffn_shift

    def clone(self) -> "RWKV7Cache":
        return RWKV7Cache(
            _map_tensor_list(self.recurrent_state, torch.Tensor.clone),
            _map_tensor_list(self.attention_shift, torch.Tensor.clone),
            _map_tensor_list(self.ffn_shift, torch.Tensor.clone),
            seen_tokens=self._seen_tokens,
        )

    def reset(self) -> None:
        self.recurrent_state = [None] * len(self)
        self.attention_shift = [None] * len(self)
        self.ffn_shift = [None] * len(self)
        self._seen_tokens = 0

    def detach(self, *, inplace: bool = True) -> "RWKV7Cache":
        target = self if inplace else self.clone()
        target.recurrent_state = _map_tensor_list(
            target.recurrent_state, torch.Tensor.detach
        )
        target.attention_shift = _map_tensor_list(
            target.attention_shift, torch.Tensor.detach
        )
        target.ffn_shift = _map_tensor_list(target.ffn_shift, torch.Tensor.detach)
        return target

    def to(self, *args, **kwargs) -> "RWKV7Cache":
        self.recurrent_state = _map_tensor_list(
            self.recurrent_state, lambda value: value.to(*args, **kwargs)
        )
        self.attention_shift = _map_tensor_list(
            self.attention_shift, lambda value: value.to(*args, **kwargs)
        )
        self.ffn_shift = _map_tensor_list(
            self.ffn_shift, lambda value: value.to(*args, **kwargs)
        )
        return self

    def _select(self, indices: torch.Tensor) -> None:
        indices = indices.to(dtype=torch.long)

        def select(value: torch.Tensor):
            return value.index_select(0, indices.to(value.device))

        self.recurrent_state = _map_tensor_list(self.recurrent_state, select)
        self.attention_shift = _map_tensor_list(self.attention_shift, select)
        self.ffn_shift = _map_tensor_list(self.ffn_shift, select)

    def select_batch(
        self, indices: torch.Tensor, *, inplace: bool = True
    ) -> "RWKV7Cache":
        target = self if inplace else self.clone()
        target._select(indices)
        return target

    def batch_select(self, indices: torch.Tensor, *, inplace: bool = True):
        return self.select_batch(indices, inplace=inplace)

    def batch_select_indices(self, indices: torch.Tensor):
        self._select(indices)
        return self

    def reorder_cache(self, beam_idx: torch.LongTensor):
        self._select(beam_idx)
        return self

    def batch_repeat_interleave(self, repeats: int):
        repeats = int(repeats)
        if repeats <= 0:
            raise ValueError("repeats must be positive")

        def repeat(value: torch.Tensor):
            return value.repeat_interleave(repeats, dim=0)

        self.recurrent_state = _map_tensor_list(self.recurrent_state, repeat)
        self.attention_shift = _map_tensor_list(self.attention_shift, repeat)
        self.ffn_shift = _map_tensor_list(self.ffn_shift, repeat)
        return self

    def to_legacy_cache(self):
        return (
            tuple(self.recurrent_state),
            tuple(self.attention_shift),
            tuple(self.ffn_shift),
        )

    @classmethod
    def from_legacy_cache(cls, legacy, seen_tokens: int = 0):
        if legacy is None:
            return cls(seen_tokens=seen_tokens)
        if isinstance(legacy, cls):
            return legacy
        if not isinstance(legacy, (tuple, list)) or len(legacy) < 3:
            raise TypeError(
                "RWKV7 past_key_values must be RWKV7Cache or a three-list legacy cache"
            )
        return cls(legacy[0], legacy[1], legacy[2], seen_tokens=seen_tokens)


# 0.9 compatibility names.
NativeRWKV7Cache = RWKV7Cache
RWKV7StateCache = RWKV7Cache


__all__ = ["RWKV7Cache", "RWKV7StateCache", "NativeRWKV7Cache"]
