from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import torch

from rwkv7_hf import native_jit, native_jit_prefill
from rwkv7_hf.native_model import NativeRWKV7Config, NativeRWKV7ForCausalLM


ROOT = Path(__file__).resolve().parents[1]


def _top_level_functions(relative: str) -> set[str]:
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _tiny_model() -> NativeRWKV7ForCausalLM:
    torch.manual_seed(811)
    return NativeRWKV7ForCausalLM(
        NativeRWKV7Config(
            vocab_size=23,
            hidden_size=8,
            attention_hidden_size=8,
            num_heads=2,
            head_dim=4,
            num_hidden_layers=2,
            intermediate_size=16,
            decay_low_rank_dim=3,
            a_low_rank_dim=3,
            gate_low_rank_dim=3,
            v_low_rank_dim=3,
        )
    ).eval()


def test_prefill_execution_ownership_moves_behind_one_call_boundary() -> None:
    facade = _top_level_functions("rwkv7_hf/native_jit.py")
    implementation = _top_level_functions("rwkv7_hf/native_jit_prefill.py")
    moved = {
        "_native_prefill_linear",
        "_native_prefill_scan",
        "_prefill_current_device",
        "_native_prefill_stacked_rkv_weights",
    }

    assert moved.isdisjoint(facade)
    assert moved <= implementation
    for name in moved:
        assert getattr(native_jit, name).__wrapped__ is getattr(native_jit_prefill, name)


def test_prefill_execution_matches_eager_and_preserves_cache_handoff() -> None:
    model = _tiny_model()
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    packs, _, _, _ = native_jit.extract(model)

    with torch.inference_mode():
        reference = model(ids, use_cache=True).logits
        logits, state, xpa, xpf = native_jit.prefill(
            model,
            ids,
            packs,
            logits_to_keep=0,
        )

    torch.testing.assert_close(logits, reference, atol=1e-6, rtol=1e-6)
    assert state[0].shape == (2, 2, 4, 4)
    assert xpa[0].shape == xpf[0].shape == (2, 8)


def test_full_prefill_fp16_accumulation_is_scoped_and_reported(monkeypatch) -> None:
    matmul = SimpleNamespace(allow_fp16_accumulation=False)
    monkeypatch.setattr(native_jit_prefill.torch.backends.cuda, "matmul", matmul)
    monkeypatch.setattr(
        native_jit_prefill,
        "_native_prefill_global_fp16_accum_enabled",
        lambda *_args: True,
    )
    monkeypatch.setattr(
        native_jit_prefill,
        "_native_prefill_block_fp16_accum_enabled",
        lambda *_args: False,
    )
    observed = []

    def fake_impl(*_args, **_kwargs):
        observed.append(matmul.allow_fp16_accumulation)
        return "sentinel"

    monkeypatch.setattr(native_jit_prefill, "_prefill_current_device_impl", fake_impl)
    model = SimpleNamespace(
        model=SimpleNamespace(
            embeddings=SimpleNamespace(weight=torch.empty(1, dtype=torch.float16))
        )
    )
    packs = [(None, None, None, None, None, None, None, torch.empty(8))]

    result = native_jit_prefill._prefill_current_device(
        model,
        torch.ones((8, 512), dtype=torch.long),
        packs,
    )

    assert result == "sentinel"
    assert observed == [True]
    assert matmul.allow_fp16_accumulation is False
    assert model._rwkv7_native_prefill_global_fp16_accum_effective is True


def test_full_prefill_fp16_accumulation_restores_backend_after_error(monkeypatch) -> None:
    matmul = SimpleNamespace(allow_fp16_accumulation=False)
    monkeypatch.setattr(native_jit_prefill.torch.backends.cuda, "matmul", matmul)
    monkeypatch.setattr(
        native_jit_prefill,
        "_native_prefill_global_fp16_accum_enabled",
        lambda *_args: True,
    )
    monkeypatch.setattr(
        native_jit_prefill,
        "_native_prefill_block_fp16_accum_enabled",
        lambda *_args: False,
    )

    def fail_impl(*_args, **_kwargs):
        assert matmul.allow_fp16_accumulation is True
        raise RuntimeError("boom")

    monkeypatch.setattr(native_jit_prefill, "_prefill_current_device_impl", fail_impl)
    model = SimpleNamespace(
        model=SimpleNamespace(
            embeddings=SimpleNamespace(weight=torch.empty(1, dtype=torch.float16))
        )
    )
    packs = [(None, None, None, None, None, None, None, torch.empty(8))]

    try:
        native_jit_prefill._prefill_current_device(
            model,
            torch.ones((8, 512), dtype=torch.long),
            packs,
        )
    except RuntimeError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("expected scoped prefill failure")
    assert matmul.allow_fp16_accumulation is False


def test_block_prefill_fp16_accumulation_is_scoped_and_reported(monkeypatch) -> None:
    matmul = SimpleNamespace(allow_fp16_accumulation=False)
    monkeypatch.setattr(native_jit_prefill.torch.backends.cuda, "matmul", matmul)
    monkeypatch.setattr(
        native_jit_prefill,
        "_native_prefill_global_fp16_accum_enabled",
        lambda *_args: False,
    )
    monkeypatch.setattr(
        native_jit_prefill,
        "_native_prefill_block_fp16_accum_enabled",
        lambda *_args: True,
    )
    observed = []

    def fake_impl(*_args, **kwargs):
        observed.append((matmul.allow_fp16_accumulation, kwargs["block_fp16_accum"]))
        return "sentinel"

    monkeypatch.setattr(native_jit_prefill, "_prefill_current_device_impl", fake_impl)
    model = SimpleNamespace(
        model=SimpleNamespace(
            embeddings=SimpleNamespace(weight=torch.empty(1, dtype=torch.float16))
        )
    )
    packs = [(None, None, None, None, None, None, None, torch.empty(8))]

    result = native_jit_prefill._prefill_current_device(
        model,
        torch.ones((8, 512), dtype=torch.long),
        packs,
    )

    assert result == "sentinel"
    assert observed == [(False, True)]
    assert matmul.allow_fp16_accumulation is False
    assert model._rwkv7_native_prefill_global_fp16_accum_effective is False
    assert model._rwkv7_native_prefill_block_fp16_accum_effective is True


def test_prefill_execution_module_is_shipped_with_remote_adapter() -> None:
    from scripts.adapter_manifest import ADAPTER_FILES

    assert "native_jit_prefill.py" in ADAPTER_FILES
    entrypoint = (ROOT / "rwkv7_hf" / "native_model.py").read_text(encoding="utf-8")
    assert "from .native_jit_prefill import" in entrypoint
