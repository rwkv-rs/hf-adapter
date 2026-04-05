from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

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


def test_ada_wagv_row_limit_is_policy_and_override_gated(monkeypatch) -> None:
    monkeypatch.setattr(
        native_jit_graph_dispatch,
        "_kernel_policy",
        lambda: SimpleNamespace(ada_wagv_lora=True, ada_wagv_lora_max_rows=8),
    )
    monkeypatch.setattr(native_jit_graph_dispatch, "ada_wagv_lora", object())
    monkeypatch.setattr(
        native_jit_graph_dispatch,
        "ada_wagv_lora_should_use",
        lambda rows, hidden, rank: 1 <= rows <= 8 and hidden >= 1024 and rank <= 512,
    )
    monkeypatch.delenv("RWKV7_NATIVE_GRAPH_ADA_WAGV_LORA_MAX_ROWS", raising=False)

    assert native_jit_graph_dispatch._native_graph_ada_wagv_lora_enabled(8, 2048, 128)
    assert not native_jit_graph_dispatch._native_graph_ada_wagv_lora_enabled(9, 2048, 128)

    monkeypatch.setenv("RWKV7_NATIVE_GRAPH_ADA_WAGV_LORA_MAX_ROWS", "4")
    assert not native_jit_graph_dispatch._native_graph_ada_wagv_lora_enabled(8, 2048, 128)
    assert native_jit_graph_dispatch._native_graph_ada_wagv_lora_enabled(4, 2048, 128)


def test_ada_wagv_bmm_is_exact_batch_and_policy_gated(monkeypatch) -> None:
    monkeypatch.setattr(
        native_jit_graph_dispatch,
        "_kernel_policy",
        lambda: SimpleNamespace(ada_wagv_bmm=True),
    )
    monkeypatch.setattr(native_jit_graph_dispatch, "ada_wagv_bmm", object())
    monkeypatch.setattr(
        native_jit_graph_dispatch,
        "ada_wagv_bmm_should_use",
        lambda rows, hidden, rank: rows == 8 and hidden >= 1024 and rank <= 512,
    )
    monkeypatch.delenv("RWKV7_NATIVE_GRAPH_ADA_WAGV_BMM", raising=False)

    assert native_jit_graph_dispatch._native_graph_ada_wagv_bmm_enabled(8, 2048, 256)
    assert not native_jit_graph_dispatch._native_graph_ada_wagv_bmm_enabled(4, 2048, 256)
    monkeypatch.setenv("RWKV7_NATIVE_GRAPH_ADA_WAGV_BMM", "0")
    assert not native_jit_graph_dispatch._native_graph_ada_wagv_bmm_enabled(8, 2048, 256)


def test_wavg_lora_launch_policy_can_specialize_batch_eight(monkeypatch) -> None:
    monkeypatch.setattr(
        native_jit_graph_dispatch,
        "_kernel_policy",
        lambda: SimpleNamespace(
            wavg_lora_blocks=(32, 64, 256),
            wavg_lora_num_warps=8,
            wavg_lora_b8_blocks=(32, 32, 256),
            wavg_lora_b8_num_warps=4,
        ),
    )
    for name in (
        "RWKV7_NATIVE_GRAPH_FUSED_WAVG_LORA_BLOCK_M",
        "RWKV7_NATIVE_GRAPH_FUSED_WAVG_LORA_BLOCK_R",
        "RWKV7_NATIVE_GRAPH_FUSED_WAVG_LORA_BLOCK_K",
        "RWKV7_NATIVE_GRAPH_FUSED_WAVG_LORA_NUM_WARPS",
    ):
        monkeypatch.delenv(name, raising=False)

    assert native_jit_graph_dispatch._native_graph_fused_wavg_lora_blocks(4) == (32, 64, 256)
    assert native_jit_graph_dispatch._native_graph_fused_wavg_lora_num_warps(4) == 8
    assert native_jit_graph_dispatch._native_graph_fused_wavg_lora_blocks(8) == (32, 32, 256)
    assert native_jit_graph_dispatch._native_graph_fused_wavg_lora_num_warps(8) == 4

    monkeypatch.setenv("RWKV7_NATIVE_GRAPH_FUSED_WAVG_LORA_BLOCK_R", "16")
    monkeypatch.setenv("RWKV7_NATIVE_GRAPH_FUSED_WAVG_LORA_NUM_WARPS", "2")
    assert native_jit_graph_dispatch._native_graph_fused_wavg_lora_blocks(8) == (32, 16, 256)
    assert native_jit_graph_dispatch._native_graph_fused_wavg_lora_num_warps(8) == 2


def test_graph_dispatch_module_is_shipped_with_remote_adapter() -> None:
    from scripts.adapter_manifest import ADAPTER_FILES

    assert "native_jit_graph_dispatch.py" in ADAPTER_FILES
    entrypoint = (ROOT / "rwkv7_hf" / "native_model.py").read_text(encoding="utf-8")
    assert "from .native_jit_graph_dispatch import" in entrypoint
