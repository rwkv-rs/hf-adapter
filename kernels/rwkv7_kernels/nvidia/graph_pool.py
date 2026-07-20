"""Package-owned LRU pool for fixed-batch CUDA graph decode runners."""
from __future__ import annotations

from collections import OrderedDict
import threading
import weakref
from typing import Any

from .native_graph_runtime import (
    NativeGraphRunner,
    native_graph_cache_size,
    native_graph_runtime_signature,
)


_LOCK = threading.RLock()
_POOLS: weakref.WeakKeyDictionary[Any, OrderedDict[tuple[Any, ...], NativeGraphRunner]] = (
    weakref.WeakKeyDictionary()
)


def _weight_signature(owner: Any) -> tuple[tuple[int, int], ...]:
    return tuple(
        (int(parameter.data_ptr()), int(getattr(parameter, "_version", 0)))
        for parameter in owner.parameters()
    )


def _runner_key(owner: Any, batch_size: int) -> tuple[Any, ...]:
    weight = owner.model.embeddings.weight
    return (
        int(batch_size),
        str(weight.device),
        str(weight.dtype),
        _weight_signature(owner),
        native_graph_runtime_signature(),
    )


def get_native_graph_runner(owner: Any, packs, batch_size: int) -> NativeGraphRunner:
    """Return or capture the exact runner for one owner/batch/policy signature."""

    key = _runner_key(owner, batch_size)
    with _LOCK:
        pool = _POOLS.setdefault(owner, OrderedDict())
        runner = pool.pop(key, None)
        if runner is not None:
            pool[key] = runner
            return runner
        runner = NativeGraphRunner(owner, packs, batch_size)
        pool[key] = runner
        while len(pool) > native_graph_cache_size():
            _old_key, old_runner = pool.popitem(last=False)
            old_runner.detach_bound_cache()
        return runner


def clear_native_graph_runners(owner: Any | None = None) -> None:
    """Detach canonical caches and release package-owned graph runners."""

    with _LOCK:
        if owner is None:
            pools = list(_POOLS.values())
            _POOLS.clear()
        else:
            pool = _POOLS.pop(owner, None)
            pools = [] if pool is None else [pool]
        for pool in pools:
            for runner in pool.values():
                runner.detach_bound_cache()
            pool.clear()


__all__ = ["clear_native_graph_runners", "get_native_graph_runner"]
