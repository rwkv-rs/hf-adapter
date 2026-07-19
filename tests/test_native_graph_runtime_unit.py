#!/usr/bin/env python3
# coding=utf-8
"""CPU contracts for the FLA-free native CUDA-graph integration."""
from __future__ import annotations

import os

import torch
import rwkv7_hf.native_model as native_model_module

from rwkv7_hf.native_graph_runtime import (
    NativeGraphRunner,
    native_graph_precompute_embedding_enabled,
    native_graph_state_dtype,
)
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


def test_native_prefill_graph_dispatch_preserves_continuation_cache(monkeypatch) -> None:
    model = build_tiny_model()
    source = build_cache(batch_size=1)
    source.seen_tokens = 3
    calls = []

    class FakePrefillRunner:
        def replay(self, input_ids, *, cache, initial_seen):
            calls.append((tuple(input_ids.shape), int(initial_seen), cache is source))
            cache.seen_tokens = int(initial_seen) + int(input_ids.shape[1])
            return torch.zeros(1, 1, model.config.vocab_size), cache

    monkeypatch.setattr(native_model_module, "_native_prefill_graph_enabled", lambda *args: True)
    monkeypatch.setattr(model, "_native_prefill_graph_runner", lambda *args: FakePrefillRunner())

    logits, cache = model._native_prefill(
        torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        logits_to_keep=1,
        seen_tokens=7,
        cache=source,
    )

    assert tuple(logits.shape) == (1, 1, model.config.vocab_size)
    assert cache is source
    assert cache.seen_tokens == 7
    assert calls == [((1, 4), 3, True)]
    assert model._rwkv7_native_model_last_prefill_backend == "native_prefill_graph"


def test_native_prefill_graph_cache_management_surface() -> None:
    model = build_tiny_model()

    class FakeRunner:
        batch_size = 1
        prompt_tokens = 128
        detached = False

        def detach_bound_cache(self):
            self.detached = True

    runner = FakeRunner()
    model._rwkv7_native_prefill_graph_runner_cache = {("shape",): runner}
    assert model.rwkv7_native_prefill_graph_cache_shapes() == [(1, 128)]
    assert model.rwkv7_clear_native_prefill_graph_cache() == 1
    assert runner.detached is True
    assert model.rwkv7_native_prefill_graph_cache_shapes() == []
    stats = model.rwkv7_native_prefill_graph_cache_stats()
    assert stats["size"] == 0
    assert stats["limit"] >= 1
    assert stats["shapes"] == []


def test_native_graph_state_dtype_is_explicit_and_fail_closed(monkeypatch) -> None:
    monkeypatch.delenv("RWKV7_NATIVE_GRAPH_STATE_DTYPE", raising=False)
    assert native_graph_state_dtype(torch.float16) == torch.float32
    monkeypatch.setenv("RWKV7_NATIVE_GRAPH_STATE_DTYPE", "fp16")
    assert native_graph_state_dtype(torch.float16) == torch.float16
    assert native_graph_state_dtype(torch.bfloat16) == torch.float32
    monkeypatch.setenv("RWKV7_NATIVE_GRAPH_STATE_DTYPE", "broken")
    try:
        native_graph_state_dtype(torch.float16)
    except ValueError as exc:
        assert "RWKV7_NATIVE_GRAPH_STATE_DTYPE" in str(exc)
    else:
        raise AssertionError("invalid state dtype must fail closed")


def test_allocated_zero_length_cache_is_initialized_without_history() -> None:
    cache = NativeRWKV7Cache(
        state=[torch.zeros(1, 2, 4, 4)],
        xpa=[torch.zeros(1, 8)],
        xpf=[torch.zeros(1, 8)],
        v_first=torch.zeros(1, 8),
        seen_tokens=0,
    )

    assert cache.is_initialized is True
    assert cache.has_previous_state() is False
    cache.seen_tokens = 1
    assert cache.has_previous_state() is True


def test_embedding_ln0_precompute_is_independent_and_default_off(monkeypatch) -> None:
    monkeypatch.delenv("RWKV7_NATIVE_GRAPH_PRECOMPUTE_EMB_LN0", raising=False)
    assert native_graph_precompute_embedding_enabled() is False
    monkeypatch.setenv("RWKV7_NATIVE_GRAPH_PRECOMPUTE_EMB_LN0", "1")
    assert native_graph_precompute_embedding_enabled() is True


def test_native_fast_token_cpu_contract_matches_forward() -> None:
    model = build_tiny_model()
    prompt = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    with torch.inference_mode():
        prefill = model(prompt, use_cache=True, logits_to_keep=1)
        token = prefill.logits[:, -1].argmax(dim=-1)
        reference = model(
            token[:, None],
            past_key_values=prefill.past_key_values.clone(),
            use_cache=True,
            logits_to_keep=1,
        )
        fast = model.rwkv7_forward_token(
            token,
            past_key_values=prefill.past_key_values.clone(),
        )
        tuple_logits, tuple_cache = model.rwkv7_forward_token(
            token[:, None],
            past_key_values=prefill.past_key_values.clone(),
            return_dict=False,
        )
    torch.testing.assert_close(fast.logits, reference.logits)
    torch.testing.assert_close(tuple_logits, reference.logits)
    assert fast.past_key_values.get_seq_length() == 4
    assert tuple_cache.get_seq_length() == 4
    assert model.rwkv7_last_fast_token_backend() in {"eager", "native_jit"}


def test_native_fast_token_rejects_invalid_usage() -> None:
    model = build_tiny_model()
    with torch.inference_mode():
        try:
            model.rwkv7_forward_token(torch.ones(1, 2, dtype=torch.long))
        except ValueError as exc:
            assert "[batch] or [batch, 1]" in str(exc)
        else:
            raise AssertionError("multi-token input must be rejected")
        try:
            model.rwkv7_forward_one(torch.ones(2, dtype=torch.long))
        except ValueError as exc:
            assert "batch size 1" in str(exc)
        else:
            raise AssertionError("rwkv7_forward_one must reject batch > 1")
    model.train()
    try:
        model.rwkv7_forward_token(torch.ones(1, dtype=torch.long))
    except RuntimeError as exc:
        assert "inference-only" in str(exc)
    else:
        raise AssertionError("training fast-token call must be rejected")


def test_native_graph_replay_can_borrow_logits_buffer() -> None:
    class FakeGraph:
        def replay(self) -> None:
            return None

    runner = object.__new__(NativeGraphRunner)
    runner.batch_size = 2
    runner.token_ids = torch.zeros(2, dtype=torch.long)
    runner.logits = torch.randn(2, 17)
    runner.graph = FakeGraph()
    runner.copy_from_cache = lambda cache: None
    runner.bind_cache = lambda cache: None
    cache = object()

    borrowed = runner.replay(torch.tensor([[1], [2]]), cache, copy_logits=False)
    owned = runner.replay(torch.tensor([[1], [2]]), cache)
    assert borrowed.data_ptr() == runner.logits.data_ptr()
    assert owned.data_ptr() != runner.logits.data_ptr()
    torch.testing.assert_close(borrowed, owned)


if __name__ == "__main__":
    test_native_cache_graph_binding_is_invalidated_by_mutation()
    test_native_graph_never_routes_on_cpu_or_training()
    test_native_graph_cache_management_surface()
    test_native_fast_token_cpu_contract_matches_forward()
    test_native_fast_token_rejects_invalid_usage()
    test_native_graph_replay_can_borrow_logits_buffer()
