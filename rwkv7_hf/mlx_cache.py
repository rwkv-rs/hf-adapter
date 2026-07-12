# coding=utf-8
"""Bounded, model-isolated prefix-state cache for MLX RWKV-7 serving."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .mlx_bridge import mlx_array_nbytes, require_mlx


def _stable_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tokenizer_fingerprint(tokenizer: Any | None) -> str | None:
    if tokenizer is None:
        return None
    init_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}
    added_tokens = getattr(tokenizer, "added_tokens_encoder", {}) or {}
    payload = {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "revision": init_kwargs.get("_commit_hash") or init_kwargs.get("revision"),
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", None),
        "chat_template": getattr(tokenizer, "chat_template", None),
        "added_tokens": sorted((str(key), int(value)) for key, value in added_tokens.items()),
    }
    return _stable_digest(payload)


def _model_layout_fingerprint(model: Any) -> str:
    arrays = getattr(model, "arrays", {}) or {}
    quantized = getattr(model, "quantized_linears", {}) or {}
    payload = {
        "dense": [
            (str(key), tuple(int(x) for x in getattr(value, "shape", ())), str(getattr(value, "dtype", None)))
            for key, value in sorted(arrays.items())
        ],
        "quantized": [
            (
                str(key),
                int(getattr(value, "bits", 0)),
                tuple(int(x) for x in getattr(value, "shape", ())),
                int(getattr(value, "storage_bytes", 0)),
            )
            for key, value in sorted(quantized.items())
        ],
    }
    return _stable_digest(payload)


def _model_execution_config(model: Any) -> dict[str, Any]:
    """Return state/logit-affecting runtime choices without touching arrays."""

    names = (
        "loaded_dtype",
        "requested_quantization",
        "quantized_linear_bits",
        "quantized_linear_backend",
        "quantized_linear_min_params",
        "quantized_linear_rkv_min_params",
        "wkv_backend",
        "prefill_backend",
        "prefill_eval_interval",
        "wkv_scan_prefill_mode",
        "wkv_scan_prefill_min_tokens",
        "dplr_chunk_size",
        "dplr_min_tokens",
        "dplr_summary_implementation",
        "dplr_layer_eval_interval",
        "dplr_layer_eval_min_tokens",
        "dplr_window_tokens",
        "fast_layer_norm",
        "fast_group_norm",
        "fused_attn_mix",
        "fused_ffn_key_relu2",
        "group_rkv_quant_projection",
        "group_rkv_quant_projection_mode",
    )
    return {name: getattr(model, name, None) for name in names}


def mlx_model_cache_fingerprint(
    model: Any,
    *,
    tokenizer: Any | None = None,
    namespace: str = "default",
) -> str:
    """Return a stable routing key that prevents cross-model/cache reuse.

    Prefix token ids remain part of each individual cache key.  This routing
    fingerprint scopes those token ids to model source/revision, tokenizer,
    dtype, quantization, numerical backend and tenant namespace.
    """

    config = getattr(model, "config", {}) or {}
    payload = {
        "namespace": str(namespace),
        "model_type": config.get("model_type"),
        "name_or_path": config.get("_name_or_path"),
        "revision": config.get("_commit_hash") or config.get("revision"),
        "source_model_dir": getattr(model, "source_model_dir", None),
        "source_weight_manifest": getattr(model, "source_weight_manifest", None),
        "layout_fingerprint": _model_layout_fingerprint(model),
        "hidden_size": int(getattr(model, "hidden_size", 0)),
        "num_hidden_layers": int(getattr(model, "num_hidden_layers", 0)),
        "num_heads": int(getattr(model, "num_heads", 0)),
        "head_dim": int(getattr(model, "head_dim", 0)),
        "vocab_size": int(getattr(model, "vocab_size", 0)),
        "execution_config": _model_execution_config(model),
        "tokenizer_fingerprint": _tokenizer_fingerprint(tokenizer),
    }
    return _stable_digest(payload)


def _state_arrays(state: Any) -> tuple[Any, ...]:
    return (
        state.v_first,
        *state.recurrent_state,
        *state.attn_x_prev,
        *state.ffn_x_prev,
    )


@dataclass
class MLXPrefixCacheHit:
    prefix_tokens: int
    logits: Any
    state: Any
    exact: bool


@dataclass
class _Entry:
    token_ids: tuple[int, ...]
    logits: Any
    state: Any
    bytes: int
    created_s: float
    last_access_s: float
    hit_count: int = 0


class MLXPrefixStateCache:
    """Thread-safe LRU/TTL cache of immutable batch-one RWKV prefix states.

    Returned arrays/states are cloned, so a decode request cannot mutate the
    cached source or another request. Keys are scoped by a model fingerprint
    that includes revision/shape/dtype/quant metadata.
    """

    def __init__(
        self,
        model: Any,
        *,
        max_entries: int = 128,
        max_bytes: int = 2 * 1024**3,
        ttl_s: float | None = 3600.0,
        namespace: str = "default",
        tokenizer: Any | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        if int(max_entries) <= 0:
            raise ValueError("max_entries must be positive")
        if int(max_bytes) <= 0:
            raise ValueError("max_bytes must be positive")
        if ttl_s is not None and float(ttl_s) <= 0:
            raise ValueError("ttl_s must be positive or None")
        self._model = model
        self._tokenizer_fingerprint = _tokenizer_fingerprint(tokenizer)
        self._execution_fingerprint = _stable_digest(_model_execution_config(model))
        self.model_fingerprint = mlx_model_cache_fingerprint(
            model,
            tokenizer=tokenizer,
            namespace=namespace,
        )
        self.namespace = str(namespace)
        self.max_entries = int(max_entries)
        self.max_bytes = int(max_bytes)
        self.ttl_s = float(ttl_s) if ttl_s is not None else None
        self._clock = clock
        self._entries: OrderedDict[tuple[str, tuple[int, ...]], _Entry] = OrderedDict()
        self._bytes = 0
        self._lock = threading.RLock()
        self.hits = 0
        self.exact_hits = 0
        self.prefix_hits = 0
        self.misses = 0
        self.puts = 0
        self.replacements = 0
        self.evictions = 0
        self.expirations = 0
        self.rejected_oversize = 0

    def assert_compatible(self, model: Any, tokenizer: Any | None = None) -> None:
        """Reject accidental use with another model or tokenizer instance.

        The in-memory cache owns concrete MLX arrays, so even two equivalent
        model objects must not share entries implicitly.  A caller that wants
        cross-process persistence needs an explicit serialization/revalidation
        layer rather than bypassing this ownership check.
        """

        if model is not self._model:
            raise ValueError("prefix cache belongs to a different MLX model instance")
        current_tokenizer = _tokenizer_fingerprint(tokenizer)
        if self._tokenizer_fingerprint is not None and current_tokenizer != self._tokenizer_fingerprint:
            raise ValueError("prefix cache tokenizer fingerprint mismatch")
        if _stable_digest(_model_execution_config(model)) != self._execution_fingerprint:
            raise ValueError("prefix cache model execution configuration changed")

    def _key(self, token_ids: Iterable[int]) -> tuple[str, tuple[int, ...]]:
        tokens = tuple(int(value) for value in token_ids)
        if not tokens:
            raise ValueError("prefix cache token_ids must be non-empty")
        return self.model_fingerprint, tokens

    def _clone(self, logits: Any, state: Any) -> tuple[Any, Any]:
        mx = require_mlx()
        cloned_logits = mx.array(logits)
        cloned_state = state.clone()
        mx.eval(cloned_logits, *_state_arrays(cloned_state))
        return cloned_logits, cloned_state

    def _entry_bytes(self, logits: Any, state: Any) -> int:
        return int(mlx_array_nbytes(logits) + sum(mlx_array_nbytes(value) for value in _state_arrays(state)))

    def _expired(self, entry: _Entry, now: float) -> bool:
        return self.ttl_s is not None and now - entry.last_access_s >= self.ttl_s

    def _remove(self, key: tuple[str, tuple[int, ...]], *, expired: bool = False) -> None:
        entry = self._entries.pop(key)
        self._bytes -= int(entry.bytes)
        if expired:
            self.expirations += 1
        else:
            self.evictions += 1

    def _purge_expired(self, now: float) -> None:
        for key, entry in list(self._entries.items()):
            if self._expired(entry, now):
                self._remove(key, expired=True)

    def _evict_to_budget(self) -> None:
        while len(self._entries) > self.max_entries or self._bytes > self.max_bytes:
            oldest = next(iter(self._entries))
            self._remove(oldest)

    def put(self, token_ids: Iterable[int], logits: Any, state: Any) -> bool:
        key = self._key(token_ids)
        if int(state.batch_size) != 1:
            raise ValueError("prefix cache currently stores batch-one states")
        if int(state.seen_tokens) != len(key[1]):
            raise ValueError(
                f"state.seen_tokens={int(state.seen_tokens)} does not match prefix length {len(key[1])}"
            )
        cloned_logits, cloned_state = self._clone(logits, state)
        size = self._entry_bytes(cloned_logits, cloned_state)
        now = float(self._clock())
        with self._lock:
            self._purge_expired(now)
            if size > self.max_bytes:
                self.rejected_oversize += 1
                return False
            if key in self._entries:
                old = self._entries.pop(key)
                self._bytes -= int(old.bytes)
                self.replacements += 1
            self._entries[key] = _Entry(
                token_ids=key[1],
                logits=cloned_logits,
                state=cloned_state,
                bytes=size,
                created_s=now,
                last_access_s=now,
            )
            self._bytes += size
            self.puts += 1
            self._evict_to_budget()
        return True

    def _record_hit(
        self,
        key: tuple[str, tuple[int, ...]],
        entry: _Entry,
        *,
        exact: bool,
        now: float,
    ) -> tuple[Any, Any, int]:
        entry.last_access_s = now
        entry.hit_count += 1
        self._entries.move_to_end(key)
        self.hits += 1
        if exact:
            self.exact_hits += 1
        else:
            self.prefix_hits += 1
        return entry.logits, entry.state, len(entry.token_ids)

    def get_exact(self, token_ids: Iterable[int]) -> MLXPrefixCacheHit | None:
        key = self._key(token_ids)
        now = float(self._clock())
        with self._lock:
            self._purge_expired(now)
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            logits_source, state_source, prefix_tokens = self._record_hit(
                key,
                entry,
                exact=True,
                now=now,
            )
        logits, state = self._clone(logits_source, state_source)
        return MLXPrefixCacheHit(prefix_tokens, logits, state, True)

    def find_longest(
        self,
        token_ids: Iterable[int],
        *,
        min_prefix_tokens: int = 1,
    ) -> MLXPrefixCacheHit | None:
        requested = self._key(token_ids)[1]
        minimum = int(min_prefix_tokens)
        if minimum <= 0:
            raise ValueError("min_prefix_tokens must be positive")
        now = float(self._clock())
        with self._lock:
            self._purge_expired(now)
            exact_key = (self.model_fingerprint, requested)
            exact = self._entries.get(exact_key)
            if exact is not None:
                logits_source, state_source, prefix_tokens = self._record_hit(
                    exact_key,
                    exact,
                    exact=True,
                    now=now,
                )
                exact_hit = True
            else:
                candidates = [
                    (key, entry)
                    for key, entry in self._entries.items()
                    if len(entry.token_ids) >= minimum
                    and len(entry.token_ids) < len(requested)
                    and requested[: len(entry.token_ids)] == entry.token_ids
                ]
                if not candidates:
                    self.misses += 1
                    return None
                key, entry = max(candidates, key=lambda item: len(item[1].token_ids))
                logits_source, state_source, prefix_tokens = self._record_hit(
                    key,
                    entry,
                    exact=False,
                    now=now,
                )
                exact_hit = False
        logits, state = self._clone(logits_source, state_source)
        return MLXPrefixCacheHit(prefix_tokens, logits, state, exact_hit)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._bytes = 0

    def telemetry(self) -> dict[str, Any]:
        with self._lock:
            now = float(self._clock())
            self._purge_expired(now)
            lookups = self.hits + self.misses
            return {
                "model_fingerprint": self.model_fingerprint,
                "tokenizer_fingerprint": self._tokenizer_fingerprint,
                "namespace": self.namespace,
                "key_schema": [
                    "namespace",
                    "model_source_revision_layout",
                    "tokenizer",
                    "dtype",
                    "quantization",
                    "backend",
                    "prefix_token_ids",
                ],
                "entries": len(self._entries),
                "bytes": int(self._bytes),
                "max_entries": self.max_entries,
                "max_bytes": self.max_bytes,
                "ttl_s": self.ttl_s,
                "hits": self.hits,
                "exact_hits": self.exact_hits,
                "prefix_hits": self.prefix_hits,
                "misses": self.misses,
                "hit_rate": round(self.hits / lookups, 6) if lookups else 0.0,
                "puts": self.puts,
                "replacements": self.replacements,
                "evictions": self.evictions,
                "expirations": self.expirations,
                "rejected_oversize": self.rejected_oversize,
                "prefix_lengths": [len(entry.token_ids) for entry in self._entries.values()],
            }


__all__ = ["MLXPrefixCacheHit", "MLXPrefixStateCache", "mlx_model_cache_fingerprint"]
