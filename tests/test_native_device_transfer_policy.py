from __future__ import annotations

import pytest
import torch

import rwkv7_hf.native as native


_SOURCE = torch.device("cuda", 0)
_TARGET = torch.device("cuda", 1)


def test_device_map_auto_preserves_p2p_on_healthy_pairs(monkeypatch) -> None:
    monkeypatch.delenv("RWKV7_DEVICE_MAP_TRANSFER", raising=False)
    calls = []

    def safe(source, target):
        calls.append((source, target))
        return True

    monkeypatch.setattr(native, "_native_cuda_p2p_transfer_safe", safe)
    assert native._native_device_map_transfer_route(_SOURCE, _TARGET) == "p2p"
    assert calls == [(_SOURCE, _TARGET)]


def test_device_map_auto_falls_back_only_after_failed_probe(monkeypatch) -> None:
    monkeypatch.setenv("RWKV7_DEVICE_MAP_TRANSFER", "auto")
    monkeypatch.setattr(
        native,
        "_native_cuda_p2p_transfer_safe",
        lambda _source, _target: False,
    )
    assert native._native_device_map_transfer_route(_SOURCE, _TARGET) == "cpu"


def test_device_map_explicit_routes_bypass_probe(monkeypatch) -> None:
    def unexpected(*_args):
        raise AssertionError("explicit route must not run the auto probe")

    monkeypatch.setattr(native, "_native_cuda_p2p_transfer_safe", unexpected)
    for value, expected in (("p2p", "p2p"), ("direct", "p2p"), ("cpu", "cpu"), ("host", "cpu")):
        monkeypatch.setenv("RWKV7_DEVICE_MAP_TRANSFER", value)
        assert native._native_device_map_transfer_route(_SOURCE, _TARGET) == expected


def test_device_map_invalid_route_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv("RWKV7_DEVICE_MAP_TRANSFER", "unsafe")
    with pytest.raises(ValueError, match="must be auto, p2p, or cpu"):
        native._native_device_map_transfer_route(_SOURCE, _TARGET)
