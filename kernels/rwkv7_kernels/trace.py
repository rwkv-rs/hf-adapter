"""Process-wide actual-route trace shared by all optional protocols."""

from __future__ import annotations

import atexit
from collections import Counter
import json
import os
from pathlib import Path


_TRACE_ENV = "RWKV7_KERNEL_TRACE_PATH"
_RECURRENT: Counter[str] = Counter()
_LINEAR: Counter[str] = Counter()
_MIX6: Counter[str] = Counter()
_MODEL: Counter[str] = Counter()
_PHASES: Counter[str] = Counter()
_REGISTERED = False


def write_trace() -> None:
    destination = os.environ.get(_TRACE_ENV)
    if not destination or not (_RECURRENT or _LINEAR or _MIX6 or _MODEL):
        return
    path = Path(destination).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "rwkv7-kernel-route-trace-v2",
        "requested_policy": os.environ.get("RWKV7_KERNEL_IMPL", "auto"),
        "requested_model_policy": os.environ.get("RWKV7_MODEL_KERNEL_IMPL", "auto"),
        "requested_training_policy": os.environ.get(
            "RWKV7_TRAINING_KERNEL_IMPL", "auto"
        ),
        "process_id": os.getpid(),
        "actual_recurrent_calls": dict(sorted(_RECURRENT.items())),
        "actual_linear_calls": dict(sorted(_LINEAR.items())),
        "actual_mix6_calls": dict(sorted(_MIX6.items())),
        "actual_model_calls": dict(sorted(_MODEL.items())),
        "actual_model_phases": dict(sorted(_PHASES.items())),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _register() -> None:
    global _REGISTERED
    if os.environ.get(_TRACE_ENV) and not _REGISTERED:
        atexit.register(write_trace)
        _REGISTERED = True


def record_recurrent(implementation: str) -> None:
    if not os.environ.get(_TRACE_ENV):
        return
    _RECURRENT[str(implementation)] += 1
    _register()


def record_linear(implementation: str) -> None:
    """Record one stateless training-linear call when tracing is enabled."""

    if not os.environ.get(_TRACE_ENV):
        return
    _LINEAR[str(implementation)] += 1
    _register()


def record_mix6(implementation: str) -> None:
    """Record one stateless six-way training-mix call."""

    if not os.environ.get(_TRACE_ENV):
        return
    _MIX6[str(implementation)] += 1
    _register()


def record_model(implementation: str, phase: str) -> None:
    if not os.environ.get(_TRACE_ENV):
        return
    _MODEL[str(implementation)] += 1
    _PHASES[str(phase)] += 1
    _register()


__all__ = [
    "record_linear",
    "record_mix6",
    "record_model",
    "record_recurrent",
    "write_trace",
]
