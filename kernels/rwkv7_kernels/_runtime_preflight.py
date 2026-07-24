"""Private process-local certificates for optional CUDA training runtimes."""

from __future__ import annotations

import threading

import torch


_LOCK = threading.Lock()
_RECURRENT_DEVICES: set[tuple[str, int | None]] = set()


def _device_key(device: torch.device) -> tuple[str, int | None]:
    device = torch.device(device)
    return device.type, device.index


def _certify_recurrent_runtime(device: torch.device) -> None:
    """Record that recurrent capability and extension loading passed on *device*."""

    with _LOCK:
        _RECURRENT_DEVICES.add(_device_key(device))


def recurrent_runtime_certified(device: torch.device) -> bool:
    """Return whether this process already certified *this exact device*."""

    with _LOCK:
        return _device_key(device) in _RECURRENT_DEVICES


def _reset_for_tests() -> None:
    with _LOCK:
        _RECURRENT_DEVICES.clear()
