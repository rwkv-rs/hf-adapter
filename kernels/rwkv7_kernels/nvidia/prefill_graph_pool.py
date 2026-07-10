"""Package-owned LRU pool for fixed-shape prefill CUDA Graph runners."""
from __future__ import annotations

from collections import OrderedDict
import threading
import weakref
from typing import Any

from .prefill_graph_runtime import (
    NativePrefillGraphRunner,
    prefill_graph_cache_size,
    prefill_graph_runtime_signature,
)


_LOCK = threading.RLock()
_POOLS: weakref.WeakKeyDictionary[
    Any, OrderedDict[tuple[Any, ...], NativePrefillGraphRunner]
] = weakref.WeakKeyDictionary()


def _weight_signature(owner: Any) -> tuple[tuple[int, int], ...]:
    return tuple(
        (int(parameter.data_ptr()), int(getattr(parameter, "_version", 0)))
        for parameter in owner.parameters()
    )


def _key(
    owner: Any,
    batch_size: int,
    prompt_tokens: int,
    logits_to_keep: int | None,
) -> tuple[Any, ...]:
    weight = owner.model.embeddings.weight
    return (
        int(batch_size),
        int(prompt_tokens),
        None if logits_to_keep is None else int(logits_to_keep),
        str(weight.device),
        str(weight.dtype),
        _weight_signature(owner),
        prefill_graph_runtime_signature(),
    )


def get_native_prefill_graph_runner(
    owner: Any,
    packs,
    batch_size: int,
    prompt_tokens: int,
    logits_to_keep: int | None,
) -> NativePrefillGraphRunner:
    key = _key(owner, batch_size, prompt_tokens, logits_to_keep)
    with _LOCK:
        pool = _POOLS.setdefault(owner, OrderedDict())
        runner = pool.pop(key, None)
        if runner is not None:
            pool[key] = runner
            return runner
        runner = NativePrefillGraphRunner(
            owner, packs, batch_size, prompt_tokens, logits_to_keep
        )
        pool[key] = runner
        while len(pool) > prefill_graph_cache_size(
            owner.model.embeddings.weight.device
        ):
            pool.popitem(last=False)
        return runner


def clear_native_prefill_graph_runners(owner: Any | None = None) -> None:
    with _LOCK:
        if owner is None:
            _POOLS.clear()
        else:
            _POOLS.pop(owner, None)


__all__ = [
    "clear_native_prefill_graph_runners",
    "get_native_prefill_graph_runner",
]
