# coding=utf-8
"""HF-compatible recurrent cache for the native RWKV-7 model."""
from __future__ import annotations

import weakref

import torch

try:  # pragma: no cover - Transformers version compatibility
    from transformers.cache_utils import Cache as _HFCache
except Exception:  # pragma: no cover
    class _HFCache:
        pass


class _NativeRWKV7LegacyCache(tuple):
    """Tuple-compatible legacy cache carrying RWKV recurrent sequence length."""

    def __new__(cls, state, xpa, xpf, v_first, seen_tokens: int = 0):
        obj = super().__new__(cls, (state, xpa, xpf, v_first))
        obj._seen_tokens = int(seen_tokens)
        return obj

    def get_seq_length(self, layer_idx: int | None = 0, cache_position=None) -> int:
        if layer_idx is not None:
            layer_idx = int(layer_idx)
            state = self[0]
            if layer_idx < 0:
                return 0
            if state is not None and layer_idx >= len(state):
                return 0
            if state is None and layer_idx != 0:
                return 0
        return self._seen_tokens

    @property
    def seen_tokens(self) -> int:
        return int(self._seen_tokens)

    @seen_tokens.setter
    def seen_tokens(self, value: int) -> None:
        self._seen_tokens = int(value)

    def to_legacy_cache(self):
        return self


class NativeRWKV7Cache(_HFCache):
    """HF Cache-contract wrapper for ``NativeRWKV7ForCausalLM`` recurrent state.

    Native decode threads ``(state, xpa, xpf, v_first)`` as its recurrent
    cache (state=list per layer, xpa/xpf=list per layer, v_first is cross-layer).
    That raw tuple does not satisfy the HF ``Cache`` contract that
    ``GenerationMixin``/``Trainer`` want (``get_seq_length`` etc.). This wrapper
    stores the tuple but subclasses the HF ``Cache`` base so it is accepted,
    and stays **iterable** so existing tuple-unpacking in ``forward`` and
    ``_reorder_cache`` keeps working unchanged.
    """

    is_compileable = True

    def __init__(self, state=None, xpa=None, xpf=None, v_first=None, seen_tokens: int = 0):
        # Skip _HFCache.__init__: it allocates CacheLayer wrappers that RWKV
        # recurrent decode does not need (mirrors RWKV7StateCache).
        self._state = state
        self._xpa = xpa
        self._xpf = xpf
        self._v_first = v_first
        self._seen_tokens = int(seen_tokens)
        self.layers = []
        self._rwkv7_cache_metrics = {
            "clones": 0,
            "detaches": 0,
            "device_moves": 0,
            "select_batch_calls": 0,
            "batch_select_calls": 0,
            "batch_select_indices_calls": 0,
            "batch_repeat_interleave_calls": 0,
            "reorder_calls": 0,
            "crops": 0,
            "resets": 0,
            "native_graph_bound_selects": 0,
        }
        self._rwkv7_cache_version = 0
        self._rwkv7_native_graph_bound_runner_id: int | None = None
        self._rwkv7_native_graph_bound_version: int | None = None
        self._rwkv7_native_graph_bound_runner_ref: weakref.ReferenceType | None = None

    def _invalidate_native_graph_binding(self) -> None:
        self._rwkv7_cache_version += 1
        self._rwkv7_native_graph_bound_runner_id = None
        self._rwkv7_native_graph_bound_version = None
        self._rwkv7_native_graph_bound_runner_ref = None

    def _bind_native_graph_runner(self, runner: object) -> None:
        self._rwkv7_native_graph_bound_runner_id = id(runner)
        self._rwkv7_native_graph_bound_version = int(self._rwkv7_cache_version)
        try:
            self._rwkv7_native_graph_bound_runner_ref = weakref.ref(runner)
        except TypeError:
            self._rwkv7_native_graph_bound_runner_ref = None

    def _native_graph_bound_to(self, runner: object) -> bool:
        return (
            self._rwkv7_native_graph_bound_runner_id == id(runner)
            and self._rwkv7_native_graph_bound_version == int(self._rwkv7_cache_version)
        )

    def _native_graph_bound_runner(self) -> object | None:
        if self._rwkv7_native_graph_bound_version != int(self._rwkv7_cache_version):
            return None
        ref = self._rwkv7_native_graph_bound_runner_ref
        runner = ref() if ref is not None else None
        return runner if runner is not None and self._rwkv7_native_graph_bound_runner_id == id(runner) else None

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(seen_tokens={self._seen_tokens}, "
            f"batch_size={self.get_batch_size()}, layers={len(self._state) if self._state is not None else 0})"
        )

    def __iter__(self):
        yield self._state
        yield self._xpa
        yield self._xpf
        yield self._v_first

    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx):
        return self.to_legacy_cache()[idx]

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized()

    @property
    def is_sliding(self) -> bool:
        return False

    @property
    def max_batch_size(self) -> int | None:
        return self.get_batch_size()

    @property
    def max_cache_len(self) -> int:
        return -1

    @property
    def seen_tokens(self) -> int:
        return int(self._seen_tokens)

    @seen_tokens.setter
    def seen_tokens(self, value: int) -> None:
        self._seen_tokens = int(value)

    @property
    def states(self) -> list[dict[str, torch.Tensor | None]]:
        """RWKV7StateCache-style per-layer view for serving helpers.

        The native backend stores state in tuple-compatible parallel lists, but
        existing dynamic-batch/offload utilities often inspect ``cache.states``
        from the production HF wrapper.  Return a fresh read-only view so those
        helpers can find tensors without mutating the native layout.
        """

        if self._state is None:
            return []
        layer_count = len(self._state)
        xpa = self._xpa if self._xpa is not None else [None] * layer_count
        xpf = self._xpf if self._xpf is not None else [None] * layer_count
        return [
            {
                "recurrent_state": self._state[idx],
                "attn_state": xpa[idx] if idx < len(xpa) else None,
                "conv_state": None,
                "ffn_state": xpf[idx] if idx < len(xpf) else None,
            }
            for idx in range(layer_count)
        ]

    def get_seq_length(self, layer_idx: int | None = 0, cache_position=None) -> int:
        if layer_idx is not None:
            layer_idx = int(layer_idx)
            if layer_idx < 0:
                return 0
            if self._state is not None and layer_idx >= len(self._state):
                return 0
            if self._state is None and layer_idx != 0:
                return 0
        return self._seen_tokens

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return -1

    def get_mask_sizes(self, cache_position: torch.Tensor | int | None, layer_idx: int = 0) -> tuple[int, int]:
        if cache_position is None:
            query_len = 0
        elif isinstance(cache_position, torch.Tensor):
            query_len = int(cache_position.numel())
        else:
            query_len = int(cache_position)
        return int(self.get_seq_length(layer_idx)) + query_len, 0

    def to_legacy_cache(self):
        return _NativeRWKV7LegacyCache(
            self._state,
            self._xpa,
            self._xpf,
            self._v_first,
            seen_tokens=self._seen_tokens,
        )

    def clone(self) -> "NativeRWKV7Cache":
        def clone_list(values):
            if values is None:
                return None
            return [v.clone() for v in values]

        out = type(self)(
            clone_list(self._state),
            clone_list(self._xpa),
            clone_list(self._xpf),
            self._v_first.clone() if self._v_first is not None else None,
            seen_tokens=self._seen_tokens,
        )
        out._rwkv7_cache_metrics = dict(self._rwkv7_cache_metrics)
        out._rwkv7_cache_metrics["clones"] += 1
        return out

    def reset(self) -> None:
        self._invalidate_native_graph_binding()
        self._state = None
        self._xpa = None
        self._xpf = None
        self._v_first = None
        self._seen_tokens = 0
        self._rwkv7_cache_metrics["resets"] += 1

    def detach(self, *, inplace: bool = True) -> "NativeRWKV7Cache":
        target = self if inplace else self.clone()
        target._invalidate_native_graph_binding()

        def detach_list(values):
            if values is None:
                return None
            return [v.detach() for v in values]

        target._state = detach_list(target._state)
        target._xpa = detach_list(target._xpa)
        target._xpf = detach_list(target._xpf)
        if target._v_first is not None:
            target._v_first = target._v_first.detach()
        target._rwkv7_cache_metrics["detaches"] += 1
        return target

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        *,
        non_blocking: bool = False,
        copy: bool = False,
        inplace: bool = True,
    ) -> "NativeRWKV7Cache":
        target = self if inplace else self.clone()
        target._invalidate_native_graph_binding()

        def move_tensor(value: torch.Tensor) -> torch.Tensor:
            kwargs = {"non_blocking": non_blocking, "copy": copy}
            if device is not None:
                kwargs["device"] = device
            if dtype is not None and value.is_floating_point():
                kwargs["dtype"] = dtype
            if len(kwargs) == 2:
                return value.clone() if copy else value
            return value.to(**kwargs)

        def move_list(values):
            if values is None:
                return None
            return [move_tensor(v) for v in values]

        target._state = move_list(target._state)
        target._xpa = move_list(target._xpa)
        target._xpf = move_list(target._xpf)
        if target._v_first is not None:
            target._v_first = move_tensor(target._v_first)
        target._rwkv7_cache_metrics["device_moves"] += 1
        return target

    def get_batch_size(self) -> int | None:
        for values in (self._state, self._xpa, self._xpf):
            if values:
                return int(values[0].shape[0])
        if self._v_first is not None:
            return int(self._v_first.shape[0])
        return None

    def select_batch(self, indices: torch.LongTensor, *, inplace: bool = True) -> "NativeRWKV7Cache":
        if not isinstance(indices, torch.Tensor):
            indices = torch.as_tensor(indices, dtype=torch.long)
        else:
            indices = indices.to(dtype=torch.long)
        target = self if inplace else type(self)(
            self._state,
            self._xpa,
            self._xpf,
            self._v_first,
            seen_tokens=self._seen_tokens,
        )
        target._rwkv7_cache_metrics = dict(self._rwkv7_cache_metrics)
        runner = target._native_graph_bound_runner() if inplace else None
        if runner is not None and target.get_batch_size() == int(indices.numel()):
            if hasattr(runner, "reorder_batch_inplace") and runner.reorder_batch_inplace(indices):
                target._rwkv7_cache_metrics["select_batch_calls"] += 1
                target._rwkv7_cache_metrics["native_graph_bound_selects"] += 1
                return target
        target._invalidate_native_graph_binding()

        def select_list(values):
            if values is None:
                return None
            return [v.index_select(0, indices.to(v.device)) for v in values]

        target._state = select_list(target._state)
        target._xpa = select_list(target._xpa)
        target._xpf = select_list(target._xpf)
        if target._v_first is not None:
            target._v_first = target._v_first.index_select(0, indices.to(target._v_first.device))
        target._rwkv7_cache_metrics["select_batch_calls"] += 1
        return target

    def batch_select(self, indices: torch.LongTensor, *, inplace: bool = True) -> "NativeRWKV7Cache":
        target = self.select_batch(indices, inplace=inplace)
        target._rwkv7_cache_metrics["batch_select_calls"] += 1
        return target

    def compact(self, indices: torch.LongTensor, *, inplace: bool = True) -> "NativeRWKV7Cache":
        return self.batch_select(indices, inplace=inplace)

    def batch_select_indices(self, indices: torch.Tensor):
        target = self.select_batch(indices, inplace=True)
        target._rwkv7_cache_metrics["batch_select_indices_calls"] += 1
        return target

    def batch_repeat_interleave(self, repeats: int):
        repeats = int(repeats)
        if repeats <= 0:
            raise ValueError("NativeRWKV7Cache.batch_repeat_interleave requires repeats > 0")

        self._invalidate_native_graph_binding()

        def repeat_list(values):
            if values is None:
                return None
            return [v.repeat_interleave(repeats, dim=0) for v in values]

        self._state = repeat_list(self._state)
        self._xpa = repeat_list(self._xpa)
        self._xpf = repeat_list(self._xpf)
        if self._v_first is not None:
            self._v_first = self._v_first.repeat_interleave(repeats, dim=0)
        self._rwkv7_cache_metrics["batch_repeat_interleave_calls"] += 1
        return self

    def crop(self, max_length: int):
        max_length = int(max_length)
        target_length = self._seen_tokens + max_length if max_length < 0 else max_length
        if target_length >= self._seen_tokens:
            return self
        if target_length <= 0:
            self._rwkv7_cache_metrics["crops"] += 1
            self.reset()
            return self
        raise NotImplementedError(
            "NativeRWKV7Cache cannot crop recurrent state to a shorter positive prefix; "
            "run a fresh prefill for that prefix instead."
        )

    def _is_initialized(self, layer_idx: int | None = None) -> bool:
        if self._state is None or self._xpa is None or self._xpf is None or self._v_first is None:
            return False
        if layer_idx is not None and (int(layer_idx) < 0 or int(layer_idx) >= len(self._state)):
            return False
        return True

    def has_previous_state(self, layer_idx: int | None = None) -> bool:
        return self._is_initialized(layer_idx) and self._seen_tokens > 0

    def update(self, *args, **kwargs):
        raise NotImplementedError(
            "NativeRWKV7Cache is not a Transformer KV cache; update it through "
            "NativeRWKV7ForCausalLM.forward(..., past_key_values=...)."
        )

    def update_recurrent_state(self, *args, **kwargs):
        raise NotImplementedError(
            "NativeRWKV7Cache stores RWKV-7 state as (state, xpa, xpf, v_first); "
            "update it through NativeRWKV7ForCausalLM.forward(..., past_key_values=...)."
        )

    def update_conv_state(self, *args, **kwargs):
        raise NotImplementedError("NativeRWKV7Cache does not have convolution state.")

    def update_indexer(self, *args, **kwargs):
        raise NotImplementedError("NativeRWKV7Cache does not have an indexer key cache.")

    def early_initialization(self, *args, **kwargs):
        raise NotImplementedError(
            "NativeRWKV7Cache cannot be early-initialized as a Transformer KV cache; "
            "native recurrent state is initialized by NativeRWKV7ForCausalLM.forward."
        )

    def offload(self, *args, **kwargs):
        raise NotImplementedError("Use NativeRWKV7Cache.to(device='cpu') to offload native recurrent state.")

    def prefetch(self, *args, **kwargs):
        raise NotImplementedError("Use NativeRWKV7Cache.to(device=...) to restore native recurrent state.")

    def reorder_cache(self, beam_idx: torch.LongTensor):
        target = self.select_batch(beam_idx, inplace=True)
        target._rwkv7_cache_metrics["reorder_calls"] += 1
        return target

    def rwkv7_cache_metrics(self) -> dict:
        metrics = dict(self._rwkv7_cache_metrics)
        metrics.update(
            {
                "seen_tokens": int(self._seen_tokens),
                "batch_size": self.get_batch_size(),
                "layers": len(self._state) if self._state is not None else 0,
            }
        )
        return metrics

    @classmethod
    def from_legacy_cache(cls, legacy, seen_tokens: int = 0):
        if legacy is None:
            return cls(seen_tokens=seen_tokens)
        if isinstance(legacy, NativeRWKV7Cache):
            return legacy
        seen = int(seen_tokens)
        if hasattr(legacy, "get_seq_length"):
            try:
                legacy_seen = int(legacy.get_seq_length())
                if legacy_seen == 0:
                    return cls(seen_tokens=seen_tokens)
                seen = legacy_seen
            except Exception:
                pass
        if hasattr(legacy, "to_legacy_cache"):
            legacy = legacy.to_legacy_cache()
        if legacy is None:
            return cls(seen_tokens=seen_tokens)
        if isinstance(legacy, (list, tuple)) and len(legacy) == 0:
            return cls(seen_tokens=seen_tokens)
        if not isinstance(legacy, (list, tuple)) or len(legacy) != 4:
            raise TypeError(
                "NativeRWKV7Cache.from_legacy_cache expects None, an empty cache, "
                "or a 4-tuple recurrent cache"
            )
        state, xpa, xpf, v_first = legacy
        return cls(state, xpa, xpf, v_first, seen_tokens=seen)


def _cache_seen(past_key_values) -> int:
    """Best-effort seen-token count from a native cache (wrapper or raw tuple)."""
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        try:
            return int(past_key_values.get_seq_length())
        except Exception:
            return 0
    return 0


def _native_cache_tuple_or_none(past_key_values):
    """Return the native recurrent tuple, or ``None`` for an empty HF cache.

    Some Transformers generation paths pre-create a default ``DynamicCache``.
    RWKV recurrent state cannot consume Transformer KV cache layers, but an
    empty cache is equivalent to no cache and should run a full prompt prefill.
    """

    if past_key_values is None:
        return None
    try:
        values = tuple(past_key_values)
    except Exception as exc:
        if _cache_seen(past_key_values) == 0:
            return None
        raise TypeError(f"Unsupported NativeRWKV7 cache type: {type(past_key_values)!r}") from exc
    if len(values) == 4 and all(value is not None for value in values):
        return values
    if _cache_seen(past_key_values) == 0:
        return None
    raise TypeError(
        "NativeRWKV7 expects a NativeRWKV7Cache or 4-tuple recurrent cache; "
        f"got {type(past_key_values)!r} with length {len(values)}"
    )


def _native_cache_batch_size(native_cache) -> int | None:
    if native_cache is None:
        return None
    state, xpa, xpf, v_first = native_cache
    for values in (state, xpa, xpf):
        if values:
            return int(values[0].shape[0])
    if v_first is not None:
        return int(v_first.shape[0])
    return None


def _validate_native_cache_batch_size(native_cache, batch_size: int) -> None:
    cache_batch_size = _native_cache_batch_size(native_cache)
    if cache_batch_size is not None and int(cache_batch_size) != int(batch_size):
        raise ValueError(
            "NativeRWKV7 cache batch size must match inputs "
            f"(cache batch={cache_batch_size}, input batch={batch_size})"
        )


def _copy_native_cache_tuple(native_cache):
    state, xpa, xpf, v_first = native_cache
    return list(state), list(xpa), list(xpf), v_first


def _maybe_legacy_native_cache(cache, return_legacy_cache: bool | None):
    if cache is not None and return_legacy_cache is True:
        return cache.to_legacy_cache()
    return cache


def _native_last_token_slice(value):
    if isinstance(value, torch.Tensor):
        if value.dim() == 0:
            return value.reshape(1)
        return value[:, -1:] if value.dim() > 1 else value[-1:]
    return value


__all__ = [
    "NativeRWKV7Cache",
    "_NativeRWKV7LegacyCache",
    "_cache_seen",
    "_copy_native_cache_tuple",
    "_maybe_legacy_native_cache",
    "_native_cache_batch_size",
    "_native_cache_tuple_or_none",
    "_native_last_token_slice",
    "_validate_native_cache_batch_size",
]
