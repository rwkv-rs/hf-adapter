from __future__ import annotations

import torch

from bench.bench_batch_sweep import clear_native_graph_caches, timed


def test_timed_prefill_uses_inference_mode() -> None:
    grad_modes: list[bool] = []

    def record_grad_mode() -> None:
        grad_modes.append(torch.is_grad_enabled())

    with torch.enable_grad():
        elapsed = timed(record_grad_mode, "cpu", runs=3)

    assert elapsed >= 0.0
    assert grad_modes == [False, False, False]


def test_batch_sweep_clears_decode_and_prefill_graph_pools() -> None:
    class Model:
        def __init__(self) -> None:
            self.calls = []

        def rwkv7_clear_native_graph_cache(self) -> int:
            self.calls.append("decode")
            return 4

        def rwkv7_clear_native_prefill_graph_cache(self) -> int:
            self.calls.append("prefill")
            return 3

    model = Model()

    cleared = clear_native_graph_caches(model)

    assert model.calls == ["decode", "prefill"]
    assert cleared == {
        "rwkv7_clear_native_graph_cache": 4,
        "rwkv7_clear_native_prefill_graph_cache": 3,
    }
