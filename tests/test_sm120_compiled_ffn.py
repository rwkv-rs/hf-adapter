from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch
import pytest

from bench import bench_cross_model_speed as cross_bench
from bench.bench_cross_model_speed import rwkv_native_graph_decode_route
from rwkv7_hf import modeling_rwkv7, native_jit, sm120_compiled_ffn as compiled_ffn
from rwkv7_hf.native_graph_runtime import NativeGraphRunner


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


def test_contract_is_exact_sm86_sm89_sm120_fp16_b8_24_layers() -> None:
    assert compiled_ffn.sm120_compiled_ffn_contract(
        batch_size=8,
        hidden_size=1024,
        num_layers=24,
        dtype_name="torch.float16",
        capability=(8, 6),
    )
    assert compiled_ffn.sm120_compiled_ffn_contract(
        batch_size=8,
        hidden_size=1024,
        num_layers=24,
        dtype_name="torch.float16",
        capability=(8, 9),
    )
    assert compiled_ffn.sm120_compiled_ffn_contract(
        batch_size=8,
        hidden_size=2048,
        num_layers=24,
        dtype_name="fp16",
        capability=(12, 0),
    )
    for override in (
        {"batch_size": 1},
        {"hidden_size": 4096},
        {"num_layers": 23},
        {"dtype_name": "torch.bfloat16"},
        {"capability": (8, 0)},
    ):
        values = {
            "batch_size": 8,
            "hidden_size": 1024,
            "num_layers": 24,
            "dtype_name": "torch.float16",
            "capability": (12, 0),
        }
        values.update(override)
        assert not compiled_ffn.sm120_compiled_ffn_contract(**values)


def test_prepare_rejects_cpu_without_attempting_compile() -> None:
    up = torch.empty(4096, 1024, dtype=torch.float16)
    down = torch.empty(1024, 4096, dtype=torch.float16)
    packs = [(index, up, down, None) for index in range(24)]
    with pytest.raises(RuntimeError, match="supports only SM86/SM89/SM120"):
        compiled_ffn.prepare_sm120_compiled_ffn(packs, 8)


def test_prepare_rejects_non_24_layer_sequence_first() -> None:
    up = torch.empty(4096, 1024, dtype=torch.float16)
    down = torch.empty(1024, 4096, dtype=torch.float16)
    with pytest.raises(RuntimeError, match="exactly layers 0..23"):
        compiled_ffn.prepare_sm120_compiled_ffn([(0, up, down, None)], 8)


def test_execute_without_preparation_fails_instead_of_falling_back(
    monkeypatch,
) -> None:
    compiled_ffn.clear_sm120_compiled_ffn_cache()
    monkeypatch.setattr(compiled_ffn, "_capability", lambda _device: (12, 0))
    x = torch.zeros(8, 1024, dtype=torch.float16)
    up = torch.zeros(4096, 1024, dtype=torch.float16)
    down = torch.zeros(1024, 4096, dtype=torch.float16)
    residual = torch.zeros_like(x)
    with pytest.raises(RuntimeError, match="before successful prewarm"):
        compiled_ffn.sm120_compiled_ffn(x, up, down, residual)


def test_execute_rejects_non_exact_or_noncontiguous_inputs_before_dispatch(
    monkeypatch,
) -> None:
    compiled_ffn.clear_sm120_compiled_ffn_cache()
    monkeypatch.setattr(compiled_ffn, "_capability", lambda _device: (12, 0))
    up = torch.zeros(4096, 1024, dtype=torch.float16)
    down = torch.zeros(1024, 4096, dtype=torch.float16)
    residual = torch.zeros(8, 1024, dtype=torch.float16)
    with pytest.raises(RuntimeError, match="unsupported decode input"):
        compiled_ffn.sm120_compiled_ffn(
            torch.zeros(8, 1023, dtype=torch.float16),
            up,
            down,
            torch.zeros(8, 1023, dtype=torch.float16),
        )

    x_noncontiguous = torch.zeros(8, 2048, dtype=torch.float16)[:, ::2]
    assert tuple(x_noncontiguous.shape) == (8, 1024)
    assert not x_noncontiguous.is_contiguous()
    with pytest.raises(RuntimeError, match="contiguous FFN input"):
        compiled_ffn.sm120_compiled_ffn(x_noncontiguous, up, down, residual)

    residual_noncontiguous = torch.zeros(8, 2048, dtype=torch.float16)[:, ::2]
    with pytest.raises(RuntimeError, match="exact contiguous"):
        compiled_ffn.sm120_compiled_ffn(
            torch.zeros(8, 1024, dtype=torch.float16),
            up,
            down,
            residual_noncontiguous,
        )


def test_compile_disable_is_a_hard_error(monkeypatch) -> None:
    monkeypatch.delenv("TORCHDYNAMO_DISABLE", raising=False)
    monkeypatch.setenv("TORCH_COMPILE_DISABLE", "1")
    with pytest.raises(RuntimeError, match="TORCH_COMPILE_DISABLE=1"):
        compiled_ffn._resolve_compile()


def test_compile_counter_gate_requires_one_graph_then_zero_recompiles() -> None:
    compiled_ffn._require_compile_counter_delta(
        (10, 3),
        (11, 3),
        expected_unique_graphs=1,
        context="initial compile",
    )
    compiled_ffn._require_compile_counter_delta(
        (11, 3),
        (11, 3),
        expected_unique_graphs=0,
        context="24-layer reuse",
    )
    with pytest.raises(RuntimeError, match="recompiled across layer weights"):
        compiled_ffn._require_compile_counter_delta(
            (11, 3),
            (12, 3),
            expected_unique_graphs=0,
            context="callable recompiled across layer weights",
        )
    with pytest.raises(RuntimeError, match="graph_break_delta=1"):
        compiled_ffn._require_compile_counter_delta(
            (11, 3),
            (11, 4),
            expected_unique_graphs=0,
            context="24-layer reuse",
        )


def _passing_preparation() -> compiled_ffn.CompiledFFNPreparation:
    return compiled_ffn.CompiledFFNPreparation(
        hidden_size=1024,
        batch_size=8,
        layer_indices=tuple(range(24)),
        min_cosine=0.99999,
        max_abs_diff=0.015625,
        argmax_all_equal=True,
        all_finite=True,
    )


def test_preparation_stats_expose_compile_reuse_and_numeric_evidence() -> None:
    stats = compiled_ffn.sm120_compiled_ffn_preparation_stats(_passing_preparation())
    assert stats["sm120_compiled_ffn_compile_effective"] is True
    assert stats["sm120_compiled_ffn_compile_reused"] is True
    assert stats["sm120_compiled_ffn_unique_graphs"] == 1
    assert stats["sm120_compiled_ffn_graph_breaks"] == 0
    assert stats["sm120_compiled_ffn_compile_mode"] == "max-autotune-no-cudagraphs"
    assert stats["sm120_compiled_ffn_prewarm_all_finite"] is True
    assert stats["sm120_compiled_ffn_prewarm_min_cosine"] == 0.99999
    assert stats["sm120_compiled_ffn_prewarm_argmax_all_equal"] is True
    assert stats["sm120_compiled_ffn_prewarm_max_abs_diff"] == 0.015625
    assert stats["sm120_compiled_ffn_prewarm_layer_indices"] == list(range(24))
    assert stats["sm120_compiled_ffn_prewarm_layer_count"] == 24

    missing = compiled_ffn.sm120_compiled_ffn_preparation_stats(None)
    assert all(value is None for value in missing.values())


def test_weight_contract_rejects_sparse_relayout() -> None:
    up = torch.empty(4096, 1024, dtype=torch.float16)
    packed = torch.empty(4096, 1024, dtype=torch.float16)
    down = packed.transpose(0, 1)
    assert not down.is_contiguous()
    with pytest.raises(RuntimeError, match="already relaid out"):
        compiled_ffn._validate_weight_pair(
            up,
            down,
            hidden_size=1024,
            device=torch.device("cpu"),
            dtype=torch.float16,
            layer_index=0,
        )


def test_explicit_route_suppresses_irreversible_sparse_weight_relayout(
    monkeypatch,
) -> None:
    monkeypatch.setenv("RWKV7_NATIVE_GRAPH_SM120_COMPILED_FFN", "1")
    assert not native_jit._native_graph_sparse_ffn_low_memory_pack_enabled()


def test_compile_and_lazy_warmup_precede_raw_cuda_graph_capture() -> None:
    source = inspect.getsource(NativeGraphRunner._capture)
    prewarm = source.index("prewarm_sm120_compiled_ffn")
    side_stream = source.index("torch.cuda.Stream")
    raw_capture = source.index("torch.cuda.graph")
    assert prewarm < side_stream < raw_capture

    factory_source = inspect.getsource(
        modeling_rwkv7.RWKV7ForCausalLM._rwkv7_native_graph_runner_current_device
    )
    factory_prewarm = factory_source.index("_native_graph_prewarm_sm120_compiled_ffn")
    factory_capture = factory_source.index("_RWKV7NativeGraphBatchedTokenRunner")
    assert factory_prewarm < factory_capture


def test_hf_wrapper_reports_all_24_selected_and_effective_layers() -> None:
    runner = object.__new__(modeling_rwkv7._RWKV7NativeGraphBatchedTokenRunner)
    runner.packs = [None] * 24
    runner.copy_from_cache_calls = 0
    runner.copy_from_cache_fast_skips = 0
    runner.bind_cache_calls = 0
    runner.bind_cache_fast_skips = 0
    runner.ada_wagv_bmm_requested = False
    runner.sm120_wagv_bmm_g_requested = False
    runner.sm120_compiled_ffn_requested = True
    runner.sm120_compiled_ffn_preparation = _passing_preparation()
    runner._decode_route_layers = {
        "ada_wagv_bmm_selected": set(),
        "ada_wagv_bmm_effective": set(),
        "sm120_wagv_bmm_g_selected": set(),
        "sm120_wagv_bmm_g_effective": set(),
        "sm120_compiled_ffn_selected": set(range(24)),
        "sm120_compiled_ffn_effective": set(range(24)),
    }

    stats = runner.copy_stats()
    assert stats["sm120_compiled_ffn_requested"] is True
    assert stats["sm120_compiled_ffn_selected_layers"] == list(range(24))
    assert stats["sm120_compiled_ffn_effective_layers"] == list(range(24))
    assert stats["sm120_compiled_ffn_effective_layer_count"] == 24
    assert stats["sm120_compiled_ffn_full_model_effective"] is True
    assert stats["sm120_compiled_ffn_compile_effective"] is True
    assert stats["sm120_compiled_ffn_compile_reused"] is True
    assert stats["sm120_compiled_ffn_unique_graphs"] == 1
    assert stats["sm120_compiled_ffn_graph_breaks"] == 0


def test_hf_batched_runner_requested_sm120_wagv_route_is_fail_closed() -> None:
    runner = object.__new__(modeling_rwkv7._RWKV7NativeGraphBatchedTokenRunner)
    runner.packs = [None] * 2
    runner.sm120_wagv_bmm_g_requested = True
    runner.sm120_compiled_ffn_requested = False
    runner._decode_route_layers = {
        "sm120_wagv_bmm_g_selected": {0, 1},
        "sm120_wagv_bmm_g_effective": {1},
    }

    with pytest.raises(RuntimeError, match="fallback is forbidden"):
        runner._require_requested_sm120_routes()

    runner._decode_route_layers["sm120_wagv_bmm_g_effective"].add(0)
    runner._require_requested_sm120_routes()

    runner.sm120_compiled_ffn_requested = True
    runner._decode_route_layers["sm120_compiled_ffn_selected"] = {0, 1}
    runner._decode_route_layers["sm120_compiled_ffn_effective"] = {1}
    with pytest.raises(RuntimeError, match="SM120_COMPILED_FFN"):
        runner._require_requested_sm120_routes()


@pytest.mark.parametrize(
    ("missing_route", "error_match"),
    (
        ("sm120_wagv_bmm_g", "SM120_WAGV_BMM_G"),
        ("sm120_compiled_ffn", "SM120_COMPILED_FFN"),
    ),
)
def test_hf_batched_capture_missing_sm120_layer_never_constructs_graph(
    monkeypatch,
    missing_route: str,
    error_match: str,
) -> None:
    events: list[str] = []
    _patch_raw_cuda_capture(monkeypatch, events)

    runner = object.__new__(modeling_rwkv7._RWKV7NativeGraphBatchedTokenRunner)
    runner.packs = [None, None]
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
def test_hf_batched_capture_gate_precedes_constructor_and_capture(
    monkeypatch,
    requested: bool,
) -> None:
    events: list[str] = []
    _patch_raw_cuda_capture(monkeypatch, events)

    runner = object.__new__(modeling_rwkv7._RWKV7NativeGraphBatchedTokenRunner)
    runner.packs = [None, None]
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


def test_hf_wrapper_missing_layer_is_not_full_model_effective() -> None:
    runner = object.__new__(modeling_rwkv7._RWKV7NativeGraphBatchedTokenRunner)
    runner.packs = [None] * 24
    runner.copy_from_cache_calls = 0
    runner.copy_from_cache_fast_skips = 0
    runner.bind_cache_calls = 0
    runner.bind_cache_fast_skips = 0
    runner.ada_wagv_bmm_requested = False
    runner.sm120_wagv_bmm_g_requested = False
    runner.sm120_compiled_ffn_requested = True
    runner.sm120_compiled_ffn_preparation = _passing_preparation()
    runner._decode_route_layers = {
        "ada_wagv_bmm_selected": set(),
        "ada_wagv_bmm_effective": set(),
        "sm120_wagv_bmm_g_selected": set(),
        "sm120_wagv_bmm_g_effective": set(),
        "sm120_compiled_ffn_selected": set(range(24)),
        "sm120_compiled_ffn_effective": set(range(23)),
    }

    stats = runner.copy_stats()
    assert stats["sm120_compiled_ffn_selected"] is True
    assert stats["sm120_compiled_ffn_effective"] is True
    assert stats["sm120_compiled_ffn_effective_layer_count"] == 23
    assert stats["sm120_compiled_ffn_full_model_effective"] is False


def test_env_unset_reports_compiled_route_all_false(monkeypatch) -> None:
    monkeypatch.delenv("RWKV7_NATIVE_GRAPH_SM120_COMPILED_FFN", raising=False)
    monkeypatch.setattr(
        modeling_rwkv7,
        "_rwkv7_kernel_policy",
        lambda: SimpleNamespace(sm120_compiled_ffn=False),
    )
    assert not modeling_rwkv7._native_graph_sm120_compiled_ffn_requested()

    runner = object.__new__(modeling_rwkv7._RWKV7NativeGraphBatchedTokenRunner)
    runner.packs = [None] * 24
    runner.copy_from_cache_calls = 0
    runner.copy_from_cache_fast_skips = 0
    runner.bind_cache_calls = 0
    runner.bind_cache_fast_skips = 0
    runner.ada_wagv_bmm_requested = False
    runner.sm120_wagv_bmm_g_requested = False
    runner.sm120_compiled_ffn_requested = False
    runner.sm120_compiled_ffn_preparation = None
    runner._decode_route_layers = {
        "ada_wagv_bmm_selected": set(),
        "ada_wagv_bmm_effective": set(),
        "sm120_wagv_bmm_g_selected": set(),
        "sm120_wagv_bmm_g_effective": set(),
        "sm120_compiled_ffn_selected": set(),
        "sm120_compiled_ffn_effective": set(),
    }
    stats = runner.copy_stats()
    for name in (
        "requested",
        "selected",
        "effective",
        "full_model_effective",
    ):
        assert stats[f"sm120_compiled_ffn_{name}"] is False
    assert stats["sm120_compiled_ffn_selected_layers"] == []
    assert stats["sm120_compiled_ffn_effective_layers"] == []
    assert stats["sm120_compiled_ffn_effective_layer_count"] == 0
    assert stats["sm120_compiled_ffn_compile_effective"] is None


def test_hf_factory_prewarm_failure_builds_no_runner_and_caches_nothing(
    monkeypatch,
) -> None:
    owner = SimpleNamespace(
        model=SimpleNamespace(
            embeddings=SimpleNamespace(weight=torch.empty(2, dtype=torch.float16))
        )
    )
    packs = [(0, 12, 64)]
    built: list[int] = []

    monkeypatch.setattr(
        modeling_rwkv7,
        "_native_graph_sm120_compiled_ffn_requested",
        lambda: True,
    )

    def fail_prewarm(_packs, _batch_size):
        raise RuntimeError("compile failed before capture")

    monkeypatch.setattr(
        modeling_rwkv7,
        "_native_graph_prewarm_sm120_compiled_ffn",
        fail_prewarm,
    )
    monkeypatch.setattr(
        modeling_rwkv7,
        "_RWKV7NativeGraphBatchedTokenRunner",
        lambda *_args, **_kwargs: built.append(1),
    )

    with pytest.raises(RuntimeError, match="compile failed before capture"):
        modeling_rwkv7.RWKV7ForCausalLM._rwkv7_native_graph_runner_current_device(
            owner, packs, 8
        )
    assert built == []
    assert list(owner._rwkv7_native_graph_runner_cache) == []


def test_cross_model_row_exports_exact_runner_compiled_route() -> None:
    class Model:
        @staticmethod
        def rwkv7_native_graph_runner_copy_stats():
            return {
                "runners": [
                    {"batch_size": 1, "sm120_compiled_ffn_effective": False},
                    {
                        "batch_size": 8,
                        "sm120_compiled_ffn_requested": True,
                        "sm120_compiled_ffn_selected": True,
                        "sm120_compiled_ffn_effective": True,
                        "sm120_compiled_ffn_selected_layers": list(range(24)),
                        "sm120_compiled_ffn_effective_layers": list(range(24)),
                        "sm120_compiled_ffn_effective_layer_count": 24,
                        "sm120_compiled_ffn_full_model_effective": True,
                        "sm120_compiled_ffn_compile_effective": True,
                        "sm120_compiled_ffn_compile_reused": True,
                        "sm120_compiled_ffn_unique_graphs": 1,
                        "sm120_compiled_ffn_graph_breaks": 0,
                        "sm120_compiled_ffn_compile_mode": (
                            "max-autotune-no-cudagraphs"
                        ),
                        "sm120_compiled_ffn_prewarm_all_finite": True,
                        "sm120_compiled_ffn_prewarm_min_cosine": 0.99999,
                        "sm120_compiled_ffn_prewarm_argmax_all_equal": True,
                        "sm120_compiled_ffn_prewarm_max_abs_diff": 0.015625,
                        "sm120_compiled_ffn_prewarm_layer_indices": list(range(24)),
                        "sm120_compiled_ffn_prewarm_layer_count": 24,
                    },
                ]
            }

    route = rwkv_native_graph_decode_route(Model(), 8)
    assert route["rwkv_native_graph_sm120_compiled_ffn_requested"] is True
    assert route["rwkv_native_graph_sm120_compiled_ffn_selected"] is True
    assert route["rwkv_native_graph_sm120_compiled_ffn_effective"] is True
    assert route["rwkv_native_graph_sm120_compiled_ffn_effective_layer_count"] == 24
    assert route["rwkv_native_graph_sm120_compiled_ffn_full_model_effective"] is True
    assert route["rwkv_native_graph_sm120_compiled_ffn_compile_effective"] is True
    assert route["rwkv_native_graph_sm120_compiled_ffn_compile_reused"] is True
    assert route["rwkv_native_graph_sm120_compiled_ffn_unique_graphs"] == 1
    assert route["rwkv_native_graph_sm120_compiled_ffn_graph_breaks"] == 0
    assert route["rwkv_native_graph_sm120_compiled_ffn_prewarm_all_finite"] is True
    assert route["rwkv_native_graph_sm120_compiled_ffn_prewarm_min_cosine"] == 0.99999
    assert (
        route["rwkv_native_graph_sm120_compiled_ffn_prewarm_argmax_all_equal"] is True
    )


def test_cross_model_row_keeps_required_route_fields_when_runner_is_missing() -> None:
    route = rwkv_native_graph_decode_route(object(), 8)
    required = {
        "rwkv_native_graph_sm120_compiled_ffn_requested",
        "rwkv_native_graph_sm120_compiled_ffn_selected",
        "rwkv_native_graph_sm120_compiled_ffn_effective",
        "rwkv_native_graph_sm120_compiled_ffn_selected_layers",
        "rwkv_native_graph_sm120_compiled_ffn_effective_layers",
        "rwkv_native_graph_sm120_compiled_ffn_effective_layer_count",
        "rwkv_native_graph_sm120_compiled_ffn_full_model_effective",
        "rwkv_native_graph_sm120_compiled_ffn_compile_effective",
        "rwkv_native_graph_sm120_compiled_ffn_compile_reused",
        "rwkv_native_graph_sm120_compiled_ffn_unique_graphs",
        "rwkv_native_graph_sm120_compiled_ffn_graph_breaks",
        "rwkv_native_graph_sm120_compiled_ffn_compile_mode",
        "rwkv_native_graph_sm120_compiled_ffn_prewarm_all_finite",
        "rwkv_native_graph_sm120_compiled_ffn_prewarm_min_cosine",
        "rwkv_native_graph_sm120_compiled_ffn_prewarm_argmax_all_equal",
        "rwkv_native_graph_sm120_compiled_ffn_prewarm_max_abs_diff",
        "rwkv_native_graph_sm120_compiled_ffn_prewarm_layer_indices",
        "rwkv_native_graph_sm120_compiled_ffn_prewarm_layer_count",
    }
    assert required.issubset(route)
    assert all(route[name] is None for name in required)


def test_backend_probe_can_preserve_full_b8_greedy_semantics(
    monkeypatch, tmp_path
) -> None:
    logits = torch.zeros(8, 1, 5)
    logits[:, :, 3] = 1.0

    def fake_forward(_args, _model, probe_ids):
        assert tuple(probe_ids.shape) == (8, 4)
        return SimpleNamespace(logits=logits.clone(), past_key_values="state")

    def fake_step(token, state):
        assert tuple(token.shape) == (8, 1)
        assert state == "state"
        return SimpleNamespace(logits=logits.clone(), past_key_values=state)

    observed_batch_sizes: list[int] = []
    monkeypatch.setattr(cross_bench, "forward_prefill", fake_forward)
    monkeypatch.setattr(
        cross_bench,
        "step_function",
        lambda _model, _kind, batch_size: (
            observed_batch_sizes.append(int(batch_size)) or fake_step,
            "native_graph",
        ),
    )
    output = tmp_path / "b8_probe.pt"
    args = SimpleNamespace(
        model_kind="rwkv",
        model_pair="0.4b",
        model_size_label="0.4B",
        model="test-rwkv-model",
        probe_batch_size=8,
        probe_tokens=3,
        probe_output=str(output),
        qwen_backend="auto",
    )
    metadata = cross_bench.save_backend_probe(
        args,
        SimpleNamespace(config=SimpleNamespace(vocab_size=64)),
        torch.arange(32, dtype=torch.long).reshape(8, 4),
    )
    payload = torch.load(output, map_location="cpu", weights_only=True)
    assert observed_batch_sizes == [8]
    assert metadata["probe_batch_size"] == 8
    assert metadata["probe_distinct_batch_prompts"] is True
    assert tuple(payload["input_ids"].shape) == (8, 4)
    assert len({tuple(row.tolist()) for row in payload["input_ids"]}) == 8
    assert tuple(payload["prompt_logits"].shape) == (8, 5)
    assert tuple(payload["final_logits"].shape) == (8, 5)
    assert tuple(payload["greedy_tokens"].shape) == (3, 8)
    assert torch.equal(payload["greedy_tokens"], torch.full((3, 8), 3))
