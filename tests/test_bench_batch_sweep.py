from __future__ import annotations

import torch

from bench.bench_batch_sweep import (
    current_bench_case,
    effective_fused_norm_mix,
    effective_fused_recurrent_raw,
    last_fast_prefill_backend,
    load_tokenizer,
    native_prefill_route,
    native_graph_state_route,
    timed,
)


def test_effective_fused_norm_mix_uses_shape_aware_runtime_route(monkeypatch) -> None:
    import sys
    from types import ModuleType

    package = ModuleType("bench_fake_model")
    native_jit = ModuleType("bench_fake_model.native_jit")
    calls: list[tuple[int, int]] = []

    def enabled(rows: int, hidden_size: int) -> bool:
        calls.append((rows, hidden_size))
        return (hidden_size, rows) == (2048, 8)

    native_jit._native_graph_fused_norm_mix_enabled = enabled
    monkeypatch.setitem(sys.modules, "bench_fake_model", package)
    monkeypatch.setitem(sys.modules, "bench_fake_model.native_jit", native_jit)
    config = type("Config", (), {"hidden_size": 2048})()
    model_type = type("Model", (), {"config": config})
    model_type.__module__ = "bench_fake_model.modeling"
    model = model_type()

    assert effective_fused_norm_mix(model, 4) is False
    assert effective_fused_norm_mix(model, 8) is True
    assert calls == [(4, 2048), (8, 2048)]


def test_repo_code_tokenizer_does_not_require_checkpoint_python(monkeypatch) -> None:
    import rwkv7_hf.tokenization_rwkv7 as tokenizer_module

    sentinel = object()
    calls: list[str] = []

    def from_pretrained(path: str):
        calls.append(path)
        return sentinel

    monkeypatch.setattr(
        tokenizer_module.RWKV7Tokenizer,
        "from_pretrained",
        from_pretrained,
    )
    args = type("Args", (), {"code_source": "repo", "hf_dir": "weights"})()
    assert load_tokenizer(args) is sentinel
    assert calls == ["weights"]


def test_effective_fused_recurrent_raw_uses_shape_aware_runtime_route(monkeypatch) -> None:
    import sys
    from types import ModuleType

    package = ModuleType("bench_fake_raw_model")
    native_jit = ModuleType("bench_fake_raw_model.native_jit")
    native_jit._native_graph_fused_recurrent_raw_enabled = (
        lambda rows, hidden: (hidden, rows) == (1024, 4)
    )
    monkeypatch.setitem(sys.modules, "bench_fake_raw_model", package)
    monkeypatch.setitem(sys.modules, "bench_fake_raw_model.native_jit", native_jit)
    config = type("Config", (), {"hidden_size": 1024})()
    model_type = type("Model", (), {"config": config})
    model_type.__module__ = "bench_fake_raw_model.modeling"
    model = model_type()

    assert effective_fused_recurrent_raw(model, 4) is True
    assert effective_fused_recurrent_raw(model, 8) is False


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


def test_last_fast_prefill_backend_prefers_public_native_api() -> None:
    class Model:
        _rwkv7_native_model_last_prefill_backend = "stale"

        @staticmethod
        def rwkv7_last_fast_prefill_backend() -> str:
            return "native_prefill_graph"

    assert last_fast_prefill_backend(Model()) == "native_prefill_graph"


def test_bench_case_environment_is_available_to_result_rows(monkeypatch) -> None:
    monkeypatch.setenv("RWKV7_BENCH_CASE", "candidate")
    assert current_bench_case() == "candidate"


def test_native_prefill_route_reports_candidate_feature_effective_flags() -> None:
    config = type("Config", (), {"hidden_size": 1024, "num_hidden_layers": 24})()
    model = type(
        "Model",
        (),
        {
            "config": config,
            "_rwkv7_native_prefill_sequence_ffn_effective": True,
            "_rwkv7_native_prefill_fp16_accum_ffn_key_effective": True,
            "rwkv7_last_fast_prefill_backend": lambda self: "native_prefill_graph",
        },
    )()

    route = native_prefill_route(model, 8, 512)
    assert route["prefill_sequence_ffn_effective"] is True
    assert route["prefill_fp16_accum_ffn_key_effective"] is True
