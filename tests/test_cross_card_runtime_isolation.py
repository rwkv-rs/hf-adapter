from __future__ import annotations

from types import SimpleNamespace

import rwkv7_hf.native_model as native_model
import rwkv7_hf.modeling_rwkv7 as modeling


def _quant_config():
    return SimpleNamespace(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_skip_modules=[],
    )


def test_bnb_load_policy_uses_explicit_device_map_gpu(monkeypatch) -> None:
    seen = []

    def policy(*, device=None, **_kwargs):
        seen.append(device)
        return SimpleNamespace(bnb_skip_policy="memory", bnb_int8_threshold=1.25)

    monkeypatch.setattr(native_model, "current_kernel_policy", policy)
    qconfig = _quant_config()
    kwargs = {
        "device_map": "cuda:1",
        "quantization_config": qconfig,
        "config": SimpleNamespace(num_hidden_layers=1),
    }
    native_model.NativeRWKV7ForCausalLM._rwkv7_prepare_bnb_kwargs(
        "unused",
        kwargs,
    )
    assert seen == ["cuda:1", "cuda:1"]
    assert qconfig.llm_int8_threshold == 1.25


def test_bnb_load_policy_is_conservative_for_mixed_device_map(monkeypatch) -> None:
    def policy(**_kwargs):
        raise AssertionError("mixed device_map must not inherit one card's policy")

    monkeypatch.setattr(native_model, "current_kernel_policy", policy)
    qconfig = _quant_config()
    kwargs = {
        "device_map": {"model.embeddings": 0, "lm_head": 1},
        "quantization_config": qconfig,
        "config": SimpleNamespace(num_hidden_layers=1),
    }
    effective, _ = native_model.NativeRWKV7ForCausalLM._rwkv7_prepare_bnb_kwargs(
        "unused",
        kwargs,
    )
    assert effective == "memory"
    assert qconfig.llm_int8_threshold == 6.0


def test_prefill_blas_scope_restores_previous_process_backend(monkeypatch) -> None:
    state = {"value": "default"}
    calls = []
    missing = object()

    def preferred(value=missing):
        if value is missing:
            return state["value"]
        calls.append(value)
        state["value"] = value

    monkeypatch.setattr(
        modeling,
        "_native_prefill_blas_target",
        lambda *_args, **_kwargs: "cublaslt",
    )
    monkeypatch.setattr(
        modeling.torch.backends.cuda,
        "preferred_blas_library",
        preferred,
    )
    with modeling._native_prefill_blas_scope(128, "cuda:1"):
        assert state["value"] == "cublaslt"
    assert state["value"] == "default"
    assert calls == ["cublaslt", "default"]
