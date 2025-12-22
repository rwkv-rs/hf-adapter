from __future__ import annotations

import torch

from bench.bench_batch_sweep import timed


def test_timed_prefill_uses_inference_mode() -> None:
    grad_modes: list[bool] = []

    def record_grad_mode() -> None:
        grad_modes.append(torch.is_grad_enabled())

    with torch.enable_grad():
        elapsed = timed(record_grad_mode, "cpu", runs=3)

    assert elapsed >= 0.0
    assert grad_modes == [False, False, False]
