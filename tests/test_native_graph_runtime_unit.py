#!/usr/bin/env python3
# coding=utf-8
"""CPU contracts for the FLA-free native CUDA-graph integration."""
from __future__ import annotations

import os

import torch

from rwkv7_hf.native_model import NativeRWKV7Cache, NativeRWKV7Config, NativeRWKV7ForCausalLM


def build_tiny_model() -> NativeRWKV7ForCausalLM:
    config = NativeRWKV7Config(
        vocab_size=17,
        hidden_size=8,
        num_hidden_layers=2,
        head_dim=4,
        intermediate_size=16,
        decay_low_rank_dim=3,
        gate_low_rank_dim=3,
        a_low_rank_dim=3,
        v_low_rank_dim=3,
        use_cache=True,
    )
    return NativeRWKV7ForCausalLM(config).eval()


def build_cache(batch_size: int = 2) -> NativeRWKV7Cache:
    state = [torch.zeros(batch_size, 2, 4, 4) for _ in range(2)]
    xpa = [torch.zeros(batch_size, 8) for _ in range(2)]
    xpf = [torch.zeros(batch_size, 8) for _ in range(2)]
    v_first = torch.zeros(batch_size, 8)
    return NativeRWKV7Cache(state, xpa, xpf, v_first, seen_tokens=3)


def test_native_cache_graph_binding_is_invalidated_by_mutation() -> None:
    cache = build_cache()
    runner = object()
    cache._bind_native_graph_runner(runner)
    assert cache._native_graph_bound_to(runner)

    cache.select_batch(torch.tensor([1, 0]), inplace=True)
    assert not cache._native_graph_bound_to(runner)

    cache._bind_native_graph_runner(runner)
    cache.detach(inplace=True)
    assert not cache._native_graph_bound_to(runner)

    cache._bind_native_graph_runner(runner)
    cache.reset()
    assert not cache._native_graph_bound_to(runner)


def test_native_graph_never_routes_on_cpu_or_training() -> None:
    model = build_tiny_model()
    cache = build_cache(batch_size=1)
    token = torch.tensor([[1]], dtype=torch.long)
    old = os.environ.get("RWKV7_NATIVE_MODEL_BACKEND")
    os.environ["RWKV7_NATIVE_MODEL_BACKEND"] = "native_graph"
    try:
        assert model._native_graph_can_run(token, cache, attention_mask=None, output_hidden_states=False) is False
        assert model._native_prefill_can_run(
            torch.tensor([[1, 2]], dtype=torch.long),
            attention_mask=None,
            output_hidden_states=False,
            use_cache=True,
            logits_to_keep=1,
        ) is False
        model.train()
        assert model._native_graph_can_run(token, cache, attention_mask=None, output_hidden_states=False) is False
    finally:
        if old is None:
            os.environ.pop("RWKV7_NATIVE_MODEL_BACKEND", None)
        else:
            os.environ["RWKV7_NATIVE_MODEL_BACKEND"] = old


def test_native_graph_cache_management_surface() -> None:
    model = build_tiny_model()
    model._rwkv7_native_graph_runner_cache = {("cpu", 1): object()}
    assert model.rwkv7_clear_native_graph_cache() == 1
    assert model.rwkv7_native_graph_cache_batch_sizes() == []
    stats = model.rwkv7_native_graph_cache_stats()
    assert stats["size"] == 0
    assert stats["limit"] >= 1


if __name__ == "__main__":
    test_native_cache_graph_binding_is_invalidated_by_mutation()
    test_native_graph_never_routes_on_cpu_or_training()
    test_native_graph_cache_management_surface()
