from __future__ import annotations

import ast
from pathlib import Path

from rwkv7_hf import native_jit, native_jit_graph_dispatch


ROOT = Path(__file__).resolve().parents[1]


def _top_level_functions(relative: str) -> set[str]:
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_graph_dispatch_ownership_moves_without_hot_path_wrappers() -> None:
    facade = _top_level_functions("rwkv7_hf/native_jit.py")
    implementation = _top_level_functions("rwkv7_hf/native_jit_graph_dispatch.py")
    moved = {
        "_native_graph_linear_dispatch",
        "_native_graph_ffn_dispatch",
        "_native_graph_rkv_project",
        "_native_graph_rkv_policy",
        "_native_graph_fused_norm_mix_enabled",
        "prewarm_ada_sparse_ffn",
    }

    assert moved.isdisjoint(facade)
    assert moved <= implementation
    for name in moved:
        assert getattr(native_jit, name) is getattr(native_jit_graph_dispatch, name)


def test_graph_rkv_policy_and_shape_gate_remain_bound_to_facade(monkeypatch) -> None:
    monkeypatch.setenv("RWKV7_NATIVE_GRAPH_RKV_POLICY", "vkwr_auto")
    monkeypatch.setenv("RWKV7_NATIVE_GRAPH_RKV_MIN_HIDDEN", "1024")
    monkeypatch.setenv("RWKV7_NATIVE_GRAPH_RKV_MAX_ROWS", "8")

    assert native_jit._native_graph_rkv_policy() == "vkwr_auto"
    assert native_jit._native_graph_vkwr_rkv_dispatch(1, 2048)
    assert not native_jit._native_graph_vkwr_rkv_dispatch(2, 2048)
    assert native_jit._native_graph_vkwr_rkv_dispatch(8, 2048)
    assert not native_jit._native_graph_vkwr_rkv_dispatch(16, 2048)
    assert not native_jit._native_graph_vkwr_rkv_dispatch(8, 512)


def test_graph_dispatch_module_is_shipped_with_remote_adapter() -> None:
    from scripts.adapter_manifest import ADAPTER_FILES

    assert "native_jit_graph_dispatch.py" in ADAPTER_FILES
    entrypoint = (ROOT / "rwkv7_hf" / "native_model.py").read_text(encoding="utf-8")
    assert "from .native_jit_graph_dispatch import" in entrypoint
