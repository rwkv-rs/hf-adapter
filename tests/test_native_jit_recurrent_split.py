from __future__ import annotations

import ast
from pathlib import Path

import torch

from rwkv7_hf import native_jit, native_jit_recurrent


ROOT = Path(__file__).resolve().parents[1]


def _top_level_functions(relative: str) -> set[str]:
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_recurrent_ownership_moves_without_hot_path_wrappers() -> None:
    facade = _top_level_functions("rwkv7_hf/native_jit.py")
    implementation = _top_level_functions("rwkv7_hf/native_jit_recurrent.py")
    moved = {
        "_native_graph_fused_recurrent_enabled",
        "_recurrent_update_unbatched",
        "_recurrent_update_batched",
    }

    assert moved.isdisjoint(facade)
    assert moved <= implementation
    for name in moved:
        assert getattr(native_jit, name) is getattr(native_jit_recurrent, name)


def test_batched_eager_recurrent_update_matches_explicit_math(monkeypatch) -> None:
    monkeypatch.setenv("RWKV7_NATIVE_GRAPH_FUSED_RECURRENT", "0")
    torch.manual_seed(827)
    batch, heads, width = 2, 2, 4
    shape = (batch, heads * width)
    r, w, k, v, kk, a = [torch.randn(shape) for _ in range(6)]
    state = torch.randn(batch, heads, width, width)

    out, new_state = native_jit._recurrent_update_batched(
        r, w, k, v, kk, a, state, batch, heads, width
    )
    vk = v.view(batch, heads, width, 1) @ k.view(batch, heads, 1, width)
    ab = (-kk).view(batch, heads, width, 1) @ (
        kk * a
    ).view(batch, heads, 1, width)
    expected_state = state * w.view(batch, heads, 1, width) + state @ ab.float() + vk.float()
    expected = (
        expected_state.to(r.dtype) @ r.view(batch, heads, width, 1)
    ).view(batch, heads * width)

    torch.testing.assert_close(new_state, expected_state)
    torch.testing.assert_close(out, expected)


def test_recurrent_module_and_runtime_dependency_lists_are_clean() -> None:
    from scripts.adapter_manifest import ADAPTER_FILES
    from rwkv7_hf import (
        native_jit_decode,
        native_jit_prefill,
        native_jit_prefill_runtime_policy,
    )

    assert "native_jit_recurrent.py" in ADAPTER_FILES
    assert len(native_jit_prefill_runtime_policy._RUNTIME_NAMES) == 33
    assert len(native_jit_prefill._RUNTIME_NAMES) == 69
    assert len(native_jit_decode._RUNTIME_NAMES) == 52
    entrypoint = (ROOT / "rwkv7_hf" / "native_model.py").read_text(encoding="utf-8")
    assert "from .native_jit_recurrent import" in entrypoint
