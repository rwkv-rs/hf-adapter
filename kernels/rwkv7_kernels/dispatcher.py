"""Implementation selection for the recurrent-v1 kernel protocol."""
from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from .protocol import validate_support_result
from .recurrent.graph import probe_recurrent_v1 as _probe_graph
from .recurrent.graph import recurrent_v1 as _run_graph
from .recurrent.triton import probe_recurrent_v1 as _probe_triton
from .recurrent.triton import recurrent_v1 as _run_triton
from .trace import record_recurrent as _record_trace
from .trace import write_trace as _write_trace


_KERNEL_IMPL_ENV = "RWKV7_KERNEL_IMPL"
_KERNEL_IMPLS = ("auto", "graph", "triton")
def _requested_implementation() -> str:
    # Auto is the production policy: the native Triton scan handles the
    # latency-critical one-token decode shape, while exact CUDA-graph replay
    # handles multi-token prefill. Explicit modes remain available for
    # isolated validation and honest operator benchmarks.
    name = os.environ.get(_KERNEL_IMPL_ENV, "auto").strip().lower()
    if name not in _KERNEL_IMPLS:
        choices = ", ".join(_KERNEL_IMPLS)
        raise ValueError(
            f"{_KERNEL_IMPL_ENV} must be one of {choices}; got {name!r}"
        )
    return name


def _fixed(name: str) -> tuple[Callable[..., Any], Callable[..., Any]]:
    if name == "graph":
        return _probe_graph, _run_graph
    if name == "triton":
        return _probe_triton, _run_triton
    raise AssertionError(f"unexpected fixed implementation {name!r}")


def _select(*args: Any, **kwargs: Any):
    requested = _requested_implementation()
    if requested != "auto":
        probe, run = _fixed(requested)
        return validate_support_result(probe(*args, **kwargs)), run

    tokens = int(args[0].shape[1])
    if tokens == 1:
        triton_support = validate_support_result(_probe_triton(*args, **kwargs))
        if triton_support["supported"]:
            return triton_support, _run_triton
    graph_support = validate_support_result(_probe_graph(*args, **kwargs))
    return graph_support, _run_graph


def probe_recurrent_v1(*args: Any, **kwargs: Any):
    """Return support and the actual implementation selected for this call."""

    support, _ = _select(*args, **kwargs)
    return support


def recurrent_v1(*args: Any, **kwargs: Any):
    """Execute the selected recurrent-v1 implementation."""

    support, run = _select(*args, **kwargs)
    if not support["supported"]:
        raise RuntimeError(support["reason"])
    _record_trace(support["implementation"])
    return run(*args, **kwargs)


__all__ = ["probe_recurrent_v1", "recurrent_v1"]
