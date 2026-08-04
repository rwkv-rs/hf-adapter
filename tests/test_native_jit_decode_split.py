from __future__ import annotations

import ast
from pathlib import Path

import torch

from rwkv7_hf import native_jit, native_jit_decode
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
    torch.manual_seed(821)
    return NativeRWKV7ForCausalLM(
        NativeRWKV7Config(
            vocab_size=29,
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


def test_decode_ownership_moves_without_token_loop_wrappers() -> None:
    facade = _top_level_functions("rwkv7_hf/native_jit.py")
    implementation = _top_level_functions("rwkv7_hf/native_jit_decode.py")
    moved = {
        "step",
        "step_batched",
        "_block_ip",
        "_block_ip_batched",
        "cuda_graph_decode",
        "greedy_graph",
    }

    assert moved.isdisjoint(facade)
    assert moved <= implementation
    for name in moved:
        assert getattr(native_jit, name) is getattr(native_jit_decode, name)


def test_dense_jit_decode_or_eager_fallback_matches_eager_for_b1_and_b2(monkeypatch) -> None:
    model = _tiny_model()
    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])

    with torch.inference_mode():
        monkeypatch.setenv("RWKV7_NATIVE_MODEL_BACKEND", "eager")
        eager = model(ids, use_cache=True).logits[:, -1]
        prefix = model(ids[:, :2], use_cache=True)
        monkeypatch.setenv("RWKV7_NATIVE_MODEL_BACKEND", "native_jit")
        jit = model(
            ids[:, 2:],
            past_key_values=prefix.past_key_values,
            use_cache=True,
        ).logits[:, -1]

    torch.testing.assert_close(jit, eager, atol=1e-6, rtol=1e-6)
    assert torch.equal(jit.argmax(dim=-1), eager.argmax(dim=-1))
    expected_backend = "native_jit" if model._native_jit_packs() is not None else "eager"
    assert model.rwkv7_native_model_last_decode_backend() == expected_backend


def test_decode_module_is_shipped_with_remote_adapter() -> None:
    from scripts.adapter_manifest import ADAPTER_FILES

    assert "native_jit_decode.py" in ADAPTER_FILES
    entrypoint = (ROOT / "rwkv7_hf" / "native_model.py").read_text(encoding="utf-8")
    assert "from .native_jit_decode import" in entrypoint
