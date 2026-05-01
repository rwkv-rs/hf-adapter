#!/usr/bin/env python3
# coding=utf-8
"""CPU contracts for the FLA-free native CUDA-graph integration."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from rwkv7_hf import native_graph_runtime as native_graph_runtime_module
from rwkv7_hf import native_model as native_model_module
from rwkv7_hf.native_graph_runtime import (
    NativeGraphRunner,
    native_graph_precompute_embedding_enabled,
    native_graph_state_dtype,
    native_graph_triton_fp16_state_enabled,
)
from rwkv7_hf.native_model import (
    NativeRWKV7Cache,
    NativeRWKV7Config,
    NativeRWKV7ForCausalLM,
)
from rwkv7_hf.sm120_compiled_ffn import CompiledFFNPreparation


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


class _FakeCudaStream:
    def wait_stream(self, _other) -> None:
        return None


class _FakeCudaContext:
    def __init__(self, on_enter=None) -> None:
        self.on_enter = on_enter

    def __enter__(self):
        if self.on_enter is not None:
            self.on_enter()
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> bool:
        return False


def _patch_raw_cuda_capture(monkeypatch, events: list[str]) -> None:
    monkeypatch.setattr(torch.cuda, "Stream", lambda **_kwargs: _FakeCudaStream())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: _FakeCudaStream())
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: _FakeCudaContext())

    def make_graph():
        events.append("construct")
        return object()

    monkeypatch.setattr(torch.cuda, "CUDAGraph", make_graph)
    monkeypatch.setattr(
        torch.cuda,
        "graph",
        lambda _graph: _FakeCudaContext(lambda: events.append("capture")),
    )


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
        assert (
            model._native_graph_can_run(
                token, cache, attention_mask=None, output_hidden_states=False
            )
            is False
        )
        assert (
            model._native_prefill_can_run(
                torch.tensor([[1, 2]], dtype=torch.long),
                attention_mask=None,
                output_hidden_states=False,
                use_cache=True,
                logits_to_keep=1,
            )
            is False
        )
        model.train()
        assert (
            model._native_graph_can_run(
                token, cache, attention_mask=None, output_hidden_states=False
            )
            is False
        )
    finally:
        if old is None:
            os.environ.pop("RWKV7_NATIVE_MODEL_BACKEND", None)
        else:
            os.environ["RWKV7_NATIVE_MODEL_BACKEND"] = old


def test_native_graph_rejects_adapter_layers(monkeypatch) -> None:
    model = build_tiny_model()
    cache = build_cache(batch_size=1)
    token = torch.tensor([[1]], dtype=torch.long)
    consulted = False

    def has_adapter_layers() -> bool:
        nonlocal consulted
        consulted = True
        return True

    monkeypatch.setattr(native_model_module, "_native_graph_available", lambda: True)
    monkeypatch.setattr(model, "_native_model_has_adapter_layers", has_adapter_layers)
    monkeypatch.setenv("RWKV7_NATIVE_MODEL_BACKEND", "native_graph")
    with torch.inference_mode():
        assert (
            model._native_graph_can_run(
                token,
                cache,
                attention_mask=None,
                output_hidden_states=False,
            )
            is False
        )
    assert consulted


def test_native_graph_rejects_non_native_quant_but_consults_native_safety(
    monkeypatch,
) -> None:
    model = build_tiny_model()
    cache = build_cache(batch_size=1)
    token = torch.tensor([[1]], dtype=torch.long)
    consulted = False

    def native_quant_graph_safe() -> bool:
        nonlocal consulted
        consulted = True
        return False

    monkeypatch.setattr(native_model_module, "_native_graph_available", lambda: True)
    monkeypatch.setattr(model, "_native_model_has_adapter_layers", lambda: False)
    monkeypatch.setattr(model, "_native_model_quantized", lambda: True)
    monkeypatch.setattr(
        model,
        "_native_model_native_quant_graph_safe",
        native_quant_graph_safe,
    )
    monkeypatch.setenv("RWKV7_NATIVE_MODEL_BACKEND", "native_graph")
    with torch.inference_mode():
        assert (
            model._native_graph_can_run(
                token,
                cache,
                attention_mask=None,
                output_hidden_states=False,
            )
            is False
        )
    assert consulted


def test_native_graph_cache_management_surface() -> None:
    model = build_tiny_model()
    model._rwkv7_native_graph_runner_cache = {("cpu", 1): object()}
    assert model.rwkv7_clear_native_graph_cache() == 1
    assert model.rwkv7_native_graph_cache_batch_sizes() == []
    stats = model.rwkv7_native_graph_cache_stats()
    assert stats["size"] == 0
    assert stats["limit"] >= 1


def test_native_graph_state_dtype_is_explicit_and_fail_closed(monkeypatch) -> None:
    # Keep the unit contract independent of the physical GPU running pytest.
    # Exact-card defaults (for example RTX 5090 FP16 graph state) are covered by
    # test_kernel_policy.py; this test exercises the generic fail-closed policy.
    monkeypatch.setattr(
        native_graph_runtime_module,
        "current_kernel_policy",
        lambda **_: SimpleNamespace(native_graph_state_dtype="fp32"),
    )
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


def test_native_graph_triton_fp16_state_is_exact_shape_and_overridable(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        native_graph_runtime_module,
        "current_kernel_policy",
        lambda **_: SimpleNamespace(
            native_graph_state_dtype="fp32",
            native_graph_triton_fp16_state=True,
            native_graph_triton_fp16_state_model_shapes=((4096, 32, 8),),
        ),
    )
    monkeypatch.delenv("RWKV7_NATIVE_GRAPH_STATE_DTYPE", raising=False)
    monkeypatch.delenv("RWKV7_NATIVE_GRAPH_TRITON_FP16_STATE", raising=False)

    assert native_graph_triton_fp16_state_enabled(4096, 32, 8)
    assert not native_graph_triton_fp16_state_enabled(4096, 32, 1)
    assert not native_graph_triton_fp16_state_enabled()
    assert (
        native_graph_state_dtype(
            torch.float16,
            hidden_size=4096,
            num_layers=32,
            batch_size=8,
        )
        == torch.float16
    )
    assert (
        native_graph_state_dtype(
            torch.float16,
            hidden_size=4096,
            num_layers=32,
            batch_size=1,
        )
        == torch.float32
    )

    monkeypatch.setenv("RWKV7_NATIVE_GRAPH_TRITON_FP16_STATE", "0")
    assert not native_graph_triton_fp16_state_enabled(4096, 32, 8)
    assert (
        native_graph_state_dtype(
            torch.float16,
            hidden_size=4096,
            num_layers=32,
            batch_size=8,
        )
        == torch.float32
    )

    monkeypatch.setenv("RWKV7_NATIVE_GRAPH_TRITON_FP16_STATE", "1")
    assert native_graph_triton_fp16_state_enabled(2560, 32, 8)
    assert (
        native_graph_state_dtype(
            torch.float16,
            hidden_size=2560,
            num_layers=32,
            batch_size=8,
        )
        == torch.float16
    )

    # The explicit dtype override has the final say even when the Triton route
    # itself is selected for an exploratory A/B run.
    monkeypatch.setenv("RWKV7_NATIVE_GRAPH_STATE_DTYPE", "fp32")
    assert (
        native_graph_state_dtype(
            torch.float16,
            hidden_size=4096,
            num_layers=32,
            batch_size=8,
        )
        == torch.float32
    )


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
    # Pin the generic policy so this assertion does not inherit a validated
    # exact-card default from the GPU on the test host.
    monkeypatch.setattr(
        native_graph_runtime_module,
        "current_kernel_policy",
        lambda **_: SimpleNamespace(native_graph_precompute_embedding=False),
    )
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


def test_fast_api_warmup_preserves_native_model_monkeypatch_surface(
    monkeypatch,
) -> None:
    model = build_tiny_model()
    warmed_batches: list[int] = []
    monkeypatch.setattr(native_model_module, "_native_graph_available", lambda: True)
    monkeypatch.setattr(
        native_model_module,
        "_native_model_backend_requested",
        lambda: "native_graph",
    )
    monkeypatch.setattr(
        model,
        "_native_graph_runner",
        lambda batch_size: warmed_batches.append(batch_size),
    )

    assert model.rwkv7_warmup_fast_token((1, 8)) == {
        1: "native_graph",
        8: "native_graph",
    }
    assert warmed_batches == [1, 8]


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


def test_native_graph_copy_stats_report_captured_bmm_route() -> None:
    runner = object.__new__(NativeGraphRunner)
    runner.copy_from_cache_calls = 3
    runner.copy_from_cache_fast_skips = 2
    runner.bind_cache_calls = 3
    runner.bind_cache_fast_skips = 2
    runner.num_layers = 2
    runner.ada_wagv_bmm_requested = True
    runner.sm120_wagv_bmm_g_requested = True
    runner.sm120_compiled_ffn_requested = True
    runner.sm120_compiled_ffn_preparation = CompiledFFNPreparation(
        hidden_size=1024,
        batch_size=8,
        layer_indices=(0, 1),
        min_cosine=0.99999,
        max_abs_diff=0.015625,
        argmax_all_equal=True,
        all_finite=True,
    )
    runner._decode_route_layers = {
        "ada_wagv_bmm_selected": {0, 1},
        "ada_wagv_bmm_effective": {0, 1},
        "sm120_wagv_bmm_g_selected": {0, 1},
        "sm120_wagv_bmm_g_effective": {0, 1},
        "sm120_compiled_ffn_selected": {0, 1},
        "sm120_compiled_ffn_effective": {0, 1},
    }

    stats = runner.copy_stats()
    assert stats["ada_wagv_bmm_requested"] is True
    assert stats["ada_wagv_bmm_selected"] is True
    assert stats["ada_wagv_bmm_effective"] is True
    assert stats["ada_wagv_bmm_effective_layers"] == [0, 1]
    assert stats["ada_wagv_bmm_full_model_effective"] is True
    assert stats["sm120_wagv_bmm_g_requested"] is True
    assert stats["sm120_wagv_bmm_g_selected"] is True
    assert stats["sm120_wagv_bmm_g_effective"] is True
    assert stats["sm120_wagv_bmm_g_effective_layers"] == [0, 1]
    assert stats["sm120_wagv_bmm_g_full_model_effective"] is True
    assert stats["sm120_compiled_ffn_requested"] is True
    assert stats["sm120_compiled_ffn_selected_layers"] == [0, 1]
    assert stats["sm120_compiled_ffn_effective_layers"] == [0, 1]
    assert stats["sm120_compiled_ffn_effective_layer_count"] == 2
    assert stats["sm120_compiled_ffn_full_model_effective"] is True
    assert stats["sm120_compiled_ffn_compile_effective"] is True
    assert stats["sm120_compiled_ffn_compile_reused"] is True
    assert stats["sm120_compiled_ffn_unique_graphs"] == 1
    assert stats["sm120_compiled_ffn_graph_breaks"] == 0
    assert stats["sm120_compiled_ffn_prewarm_min_cosine"] == 0.99999


def test_native_graph_batched_step_forwards_bmm_route_observer(monkeypatch) -> None:
    runner = object.__new__(NativeGraphRunner)
    runner.single = False
    runner.batch_size = 2
    runner.hidden = 4
    runner.num_layers = 2
    runner.token_ids = torch.tensor([0, 1], dtype=torch.long)
    runner.embeddings = torch.randn(3, 4)
    runner.packs = [0, 1]
    runner.state = [None, None]
    runner.xpa = [None, None]
    runner.xpf = [None, None]
    runner.sparse_ffn_out = [None, None]
    runner.elapsed = None
    runner.v_first = torch.zeros(2, 4)
    runner.norm_weight = torch.ones(4)
    runner.norm_bias = torch.zeros(4)
    runner.head = torch.nn.Linear(4, 3, bias=False)
    runner.logits = torch.empty(2, 3)
    runner.copy_from_cache_calls = 0
    runner.copy_from_cache_fast_skips = 0
    runner.bind_cache_calls = 0
    runner.bind_cache_fast_skips = 0
    runner.ada_wagv_bmm_requested = True
    runner.sm120_wagv_bmm_g_requested = True
    runner.sm120_compiled_ffn_requested = True
    runner._decode_route_layers = {
        "ada_wagv_bmm_selected": set(),
        "ada_wagv_bmm_effective": set(),
        "sm120_wagv_bmm_g_selected": set(),
        "sm120_wagv_bmm_g_effective": set(),
        "sm120_compiled_ffn_selected": set(),
        "sm120_compiled_ffn_effective": set(),
    }

    def fake_block(
        hidden,
        state,
        xpa,
        xpf,
        v_first,
        layer_index,
        sparse_out,
        elapsed,
        advance_elapsed,
        route_observer,
    ):
        route_observer("ada_wagv_bmm_selected", layer_index)
        route_observer("ada_wagv_bmm_effective", layer_index)
        route_observer("sm120_wagv_bmm_g_selected", layer_index)
        route_observer("sm120_wagv_bmm_g_effective", layer_index)
        route_observer("sm120_compiled_ffn_selected", layer_index)
        route_observer("sm120_compiled_ffn_effective", layer_index)
        return hidden

    monkeypatch.setattr(native_graph_runtime_module, "_block_ip_batched", fake_block)
    runner._one_step()

    stats = runner.copy_stats()
    assert stats["ada_wagv_bmm_selected_layers"] == [0, 1]
    assert stats["ada_wagv_bmm_effective_layers"] == [0, 1]
    assert stats["ada_wagv_bmm_full_model_effective"] is True
    assert stats["sm120_wagv_bmm_g_selected_layers"] == [0, 1]
    assert stats["sm120_wagv_bmm_g_effective_layers"] == [0, 1]
    assert stats["sm120_wagv_bmm_g_full_model_effective"] is True
    assert stats["sm120_compiled_ffn_selected_layers"] == [0, 1]
    assert stats["sm120_compiled_ffn_effective_layers"] == [0, 1]
    assert stats["sm120_compiled_ffn_full_model_effective"] is True


def test_native_graph_requested_sm120_route_is_fail_closed_before_capture() -> None:
    runner = object.__new__(NativeGraphRunner)
    runner.num_layers = 2
    runner.sm120_wagv_bmm_g_requested = True
    runner.sm120_compiled_ffn_requested = False
    runner._decode_route_layers = {
        "sm120_wagv_bmm_g_selected": {0, 1},
        "sm120_wagv_bmm_g_effective": {0},
    }

    with pytest.raises(RuntimeError, match="fallback is forbidden"):
        runner._require_requested_sm120_routes()

    runner._decode_route_layers["sm120_wagv_bmm_g_effective"].add(1)
    runner._require_requested_sm120_routes()

    runner.sm120_compiled_ffn_requested = True
    runner._decode_route_layers["sm120_compiled_ffn_selected"] = {0, 1}
    runner._decode_route_layers["sm120_compiled_ffn_effective"] = {0}
    with pytest.raises(RuntimeError, match="SM120_COMPILED_FFN"):
        runner._require_requested_sm120_routes()


@pytest.mark.parametrize(
    ("missing_route", "error_match"),
    (
        ("sm120_wagv_bmm_g", "SM120_WAGV_BMM_G"),
        ("sm120_compiled_ffn", "SM120_COMPILED_FFN"),
    ),
)
def test_native_graph_capture_missing_sm120_layer_never_constructs_graph(
    monkeypatch,
    missing_route: str,
    error_match: str,
) -> None:
    events: list[str] = []
    _patch_raw_cuda_capture(monkeypatch, events)
    monkeypatch.setattr(
        native_graph_runtime_module,
        "prewarm_sm120_compiled_ffn",
        lambda _packs, _batch_size: object(),
    )
    monkeypatch.setattr(native_graph_runtime_module, "prewarm_ada_sparse_ffn", None)

    runner = object.__new__(NativeGraphRunner)
    runner.num_layers = 2
    runner.packs = [None, None]
    runner.batch_size = 8
    runner.device = torch.device("cpu")
    runner.sm120_wagv_bmm_g_requested = missing_route == "sm120_wagv_bmm_g"
    runner.sm120_compiled_ffn_requested = missing_route == "sm120_compiled_ffn"
    runner._decode_route_layers = {
        "sm120_wagv_bmm_g_selected": set(),
        "sm120_wagv_bmm_g_effective": set(),
        "sm120_compiled_ffn_selected": set(),
        "sm120_compiled_ffn_effective": set(),
    }

    def one_step() -> None:
        events.append("warmup")
        runner._decode_route_layers[f"{missing_route}_selected"].update((0, 1))
        runner._decode_route_layers[f"{missing_route}_effective"].add(0)

    runner._one_step = one_step
    original_gate = runner._require_requested_sm120_routes

    def checked_gate() -> None:
        events.append("gate")
        original_gate()

    runner._require_requested_sm120_routes = checked_gate

    with pytest.raises(RuntimeError, match=error_match):
        runner._capture()

    assert events == ["warmup", "warmup", "warmup", "gate"]
    assert not hasattr(runner, "graph")


@pytest.mark.parametrize("requested", (False, True))
def test_native_graph_capture_gate_precedes_constructor_and_capture(
    monkeypatch,
    requested: bool,
) -> None:
    events: list[str] = []
    _patch_raw_cuda_capture(monkeypatch, events)
    monkeypatch.setattr(
        native_graph_runtime_module,
        "prewarm_sm120_compiled_ffn",
        lambda _packs, _batch_size: object(),
    )
    monkeypatch.setattr(native_graph_runtime_module, "prewarm_ada_sparse_ffn", None)

    runner = object.__new__(NativeGraphRunner)
    runner.num_layers = 2
    runner.packs = [None, None]
    runner.batch_size = 8
    runner.device = torch.device("cpu")
    runner.sm120_wagv_bmm_g_requested = requested
    runner.sm120_compiled_ffn_requested = requested
    runner._decode_route_layers = {
        "sm120_wagv_bmm_g_selected": set(),
        "sm120_wagv_bmm_g_effective": set(),
        "sm120_compiled_ffn_selected": set(),
        "sm120_compiled_ffn_effective": set(),
    }

    def one_step() -> None:
        events.append("captured" if "capture" in events else "warmup")
        if requested:
            for route in ("sm120_wagv_bmm_g", "sm120_compiled_ffn"):
                runner._decode_route_layers[f"{route}_selected"].update((0, 1))
                runner._decode_route_layers[f"{route}_effective"].update((0, 1))

    runner._one_step = one_step
    original_gate = runner._require_requested_sm120_routes

    def checked_gate() -> None:
        events.append("gate")
        original_gate()

    runner._require_requested_sm120_routes = checked_gate
    runner._capture()

    assert events == [
        "warmup",
        "warmup",
        "warmup",
        "gate",
        "construct",
        "capture",
        "captured",
    ]
    assert runner.graph is not None


def test_native_graph_single_step_preserves_block_ip_signature(monkeypatch) -> None:
    runner = object.__new__(NativeGraphRunner)
    runner.single = True
    runner.hidden = 4
    runner.num_layers = 1
    runner.token_ids = torch.tensor([0], dtype=torch.long)
    runner.embeddings = torch.randn(3, 4)
    runner.packs = ["pack"]
    runner.state = [None]
    runner.xpa = [None]
    runner.xpf = [None]
    runner.sparse_ffn_out = [None]
    runner.elapsed = None
    runner.v_first = torch.zeros(4)
    runner.norm_weight = torch.ones(4)
    runner.norm_bias = torch.zeros(4)
    runner.head = torch.nn.Linear(4, 3, bias=False)
    runner.logits = torch.empty(3)
    observed: list[tuple[object, ...]] = []

    def fake_block(*args):
        observed.append(args)
        return args[0]

    monkeypatch.setattr(native_graph_runtime_module, "_block_ip", fake_block)
    runner._one_step()

    assert len(observed) == 1
    # hidden/state/xpa/xpf/v_first/pack/sparse/elapsed/advance: no observer.
    assert len(observed[0]) == 9
    assert observed[0][-1] is True


if __name__ == "__main__":
    test_native_cache_graph_binding_is_invalidated_by_mutation()
    test_native_graph_never_routes_on_cpu_or_training()
    test_native_graph_cache_management_surface()
    test_native_fast_token_cpu_contract_matches_forward()
    test_native_fast_token_rejects_invalid_usage()
    test_native_graph_replay_can_borrow_logits_buffer()
