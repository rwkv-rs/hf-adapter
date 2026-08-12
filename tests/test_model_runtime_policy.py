from __future__ import annotations

import ast
from types import SimpleNamespace

import pytest
import torch

from rwkv7_hf import model_runtime_policy, native_model


def test_native_model_keeps_historical_policy_patch_surface(monkeypatch) -> None:
    policy = SimpleNamespace(
        bnb_skip_policy="decode_hot",
        bnb_int8_threshold=5.0,
        prefill_graph=True,
        prefill_graph_model_shapes=((4096, 61, 8, 512),),
        prefill_graph_cache_size=3,
        native_external_quant_prefill_graph=True,
    )
    monkeypatch.setattr(native_model, "current_kernel_policy", lambda **_kwargs: policy)
    monkeypatch.setattr(native_model, "_native_jit_prefill", object())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for name in (
        "RWKV7_BNB_SKIP_POLICY",
        "RWKV7_BNB_INT8_THRESHOLD",
        "RWKV7_NATIVE_PREFILL_GRAPH",
        "RWKV7_NATIVE_PREFILL_GRAPH_CACHE_SIZE",
        "RWKV7_NATIVE_PREFILL_EXTERNAL_QUANT_GRAPH",
    ):
        monkeypatch.delenv(name, raising=False)

    assert native_model._bnb_skip_policy() == "decode_hot"
    assert native_model._bnb_int8_threshold_override() == 5.0
    assert native_model._native_prefill_graph_enabled(8, 512, 4096, 61)
    assert native_model._native_prefill_graph_cache_size() == 3
    assert native_model._native_prefill_external_quant_graph_enabled()

    monkeypatch.delenv("RWKV7_NATIVE_MODEL_BACKEND", raising=False)
    monkeypatch.delenv("RWKV7_FAST_TOKEN_BACKEND", raising=False)
    monkeypatch.setattr(native_model, "_native_model_jit_enabled", lambda: False)
    assert native_model._native_model_backend_requested() == "eager"


def test_fast_token_backend_is_native_model_compatibility_alias(monkeypatch) -> None:
    monkeypatch.delenv("RWKV7_NATIVE_MODEL_BACKEND", raising=False)
    monkeypatch.setenv("RWKV7_FAST_TOKEN_BACKEND", "native_jit")
    assert model_runtime_policy.native_model_backend_requested() == "native_jit"

    monkeypatch.setenv("RWKV7_NATIVE_MODEL_BACKEND", "native_graph")
    assert model_runtime_policy.native_model_backend_requested() == "native_graph"


def test_unrelated_wrapper_backend_does_not_override_native_default(monkeypatch) -> None:
    monkeypatch.delenv("RWKV7_NATIVE_MODEL_BACKEND", raising=False)
    monkeypatch.setenv("RWKV7_FAST_TOKEN_BACKEND", "fla")
    assert (
        model_runtime_policy.native_model_backend_requested(
            jit_enabled_fn=lambda: True,
        )
        == "auto"
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("default", "memory"),
        ("hybrid", "decode_hot"),
        ("o_proj", "output_hot"),
        ("throughput", "prefill_hot"),
        ("rk_dense", "decode_rk"),
        ("no_quant", "dense"),
        ("unknown", "memory"),
    ],
)
def test_bnb_policy_aliases_remain_stable(value: str, expected: str) -> None:
    assert model_runtime_policy.bnb_skip_policy(value) == expected


def test_runtime_policy_has_no_model_or_kernel_implementation_imports() -> None:
    source = model_runtime_policy.__file__
    assert source is not None
    tree = ast.parse(open(source, encoding="utf-8").read())
    imported_modules = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }

    assert not any(name.endswith("native_model") for name in imported_modules)
    assert not any(name.endswith("native_jit") for name in imported_modules)
    assert not any("fused_" in name for name in imported_modules)
    assert not any("sm70_" in name for name in imported_modules)
