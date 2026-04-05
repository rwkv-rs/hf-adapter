from __future__ import annotations

import torch

from bench.bench_batch_sweep import native_graph_state_route, timed


def test_timed_prefill_uses_inference_mode() -> None:
    grad_modes: list[bool] = []

    def record_grad_mode() -> None:
        grad_modes.append(torch.is_grad_enabled())

    with torch.enable_grad():
        elapsed = timed(record_grad_mode, "cpu", runs=3)

    assert elapsed >= 0.0
    assert grad_modes == [False, False, False]


def test_native_graph_state_route_reports_bound_runner() -> None:
    class Runner:
        state_dtype = torch.float16
        triton_fp16_state = True
        fp16_recurrent = False

    class Cache:
        def _native_graph_bound_runner(self):
            return Runner()

    assert native_graph_state_route(Cache()) == {
        "native_graph_state_dtype": "torch.float16",
        "native_graph_triton_fp16_state": True,
        "native_graph_native_fp16_recurrent": False,
    }
    assert native_graph_state_route(object()) == {}
