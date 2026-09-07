"""Backend-aware timing and temporary environment helpers."""

from __future__ import annotations

from contextlib import contextmanager
import os
import statistics
import time
from typing import Any, Callable, Iterator, Mapping, Sequence


def _torch():
    import torch

    return torch


def synchronize(device: str | object) -> None:
    """Synchronize CUDA only when the requested device is CUDA."""

    if str(device).startswith("cuda"):
        torch = _torch()
        if torch.cuda.is_available():
            torch.cuda.synchronize(device=device)


def measure_ms(
    fn: Callable[[], Any],
    device: str | object,
    *,
    mode: str = "cuda-event",
) -> float:
    """Measure one call using CUDA events when available, otherwise wall time."""

    device_text = str(device)
    if mode == "cuda-event" and device_text.startswith("cuda"):
        torch = _torch()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        fn()
        end.record()
        end.synchronize()
        return float(begin.elapsed_time(end))
    if mode not in {"cuda-event", "wall"}:
        raise ValueError(f"unsupported timing mode: {mode!r}")
    synchronize(device)
    started = time.perf_counter()
    fn()
    synchronize(device)
    return (time.perf_counter() - started) * 1000.0


def median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot compute a median from an empty sequence")
    return float(statistics.median(values))


@contextmanager
def temporary_environ(
    values: Mapping[str, str | None] | None = None,
    /,
    **overrides: str | None,
) -> Iterator[None]:
    """Apply environment overrides for one case and restore them exactly."""

    requested = dict(values or {})
    requested.update(overrides)
    previous = {name: os.environ.get(name) for name in requested}
    try:
        for name, value in requested.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
