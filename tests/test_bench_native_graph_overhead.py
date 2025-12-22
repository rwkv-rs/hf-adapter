from __future__ import annotations

import pytest
import torch

from bench.bench_native_graph_overhead import (
    resolve_native_graph_runner,
    runner_token_buffer,
)


def test_resolve_native_model_graph_runner() -> None:
    sentinel = object()

    class Model:
        def _native_graph_runner(self, batch_size: int):
            assert batch_size == 4
            return sentinel

    runner, interface = resolve_native_graph_runner(Model(), 4)
    assert runner is sentinel
    assert interface == "native_model"


def test_resolve_legacy_wrapper_graph_runner() -> None:
    sentinel = object()
    packs = object()

    class Model:
        def _rwkv7_native_jit_packs(self):
            return packs

        def _rwkv7_native_graph_runner(self, actual_packs, batch_size: int):
            assert actual_packs is packs
            assert batch_size == 2
            return sentinel

    runner, interface = resolve_native_graph_runner(Model(), 2)
    assert runner is sentinel
    assert interface == "legacy_wrapper"


@pytest.mark.parametrize("name", ["token_ids", "tok_id"])
def test_runner_token_buffer_supports_both_interfaces(name: str) -> None:
    value = torch.zeros(2, dtype=torch.long)
    runner = type("Runner", (), {name: value})()
    assert runner_token_buffer(runner) is value


def test_runner_token_buffer_rejects_unknown_interface() -> None:
    with pytest.raises(ValueError, match="token-id buffer"):
        runner_token_buffer(object())
