from __future__ import annotations

import ast
from pathlib import Path

import torch

from rwkv7_hf import native_jit, native_jit_packing
from rwkv7_hf.native_jit_packing import _should_stack_rkv
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
    return NativeRWKV7ForCausalLM(
        NativeRWKV7Config(
            vocab_size=19,
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


def test_packing_implementation_ownership_and_state_aliases() -> None:
    facade = _top_level_functions("rwkv7_hf/native_jit.py")
    implementation = _top_level_functions("rwkv7_hf/native_jit_packing.py")

    assert {"extract_dense_packs", "extract_graph_packs"} <= implementation
    assert {"init_state", "init_batched_from_packs"} <= implementation
    assert {"extract_dense_packs", "extract_graph_packs"}.isdisjoint(facade)
    assert native_jit._init is native_jit_packing.init_state
    assert native_jit._init_batched_from_packs is native_jit_packing.init_batched_from_packs


def test_rkv_pack_capacity_gate_keeps_7p2b_copy_free_by_default() -> None:
    assert _should_stack_rkv("vkwr_auto", 2560, 2560)
    assert not _should_stack_rkv("vkwr_auto", 4096, 2560)
    assert _should_stack_rkv("vkwr_auto", 4096, 4096)
    assert not _should_stack_rkv("manual", 2048, 2560)


def test_dense_and_graph_pack_contracts_are_preserved() -> None:
    model = _tiny_model()
    dense, dense_h, dense_n, dense_eps = native_jit.extract(model)
    graph, graph_h, graph_n, graph_eps = native_jit.extract_graph(model)

    assert len(dense) == len(graph) == model.config.num_hidden_layers
    assert (dense_h, dense_n, dense_eps) == (graph_h, graph_n, graph_eps)
    assert all(len(pack) == 41 for pack in dense)
    assert all(len(pack) == 41 for pack in graph)

    state, xpa, xpf, v_first = native_jit._init(
        model,
        torch.device("cpu"),
        torch.float32,
    )
    assert len(state) == len(xpa) == len(xpf) == 2
    assert state[0].shape == (2, 4, 4)
    assert xpa[0].shape == xpf[0].shape == (8,)
    assert v_first.shape == (8,)


def test_packing_module_is_shipped_with_remote_adapter() -> None:
    from scripts.adapter_manifest import ADAPTER_FILES

    assert "native_jit_packing.py" in ADAPTER_FILES
    entrypoint = (ROOT / "rwkv7_hf" / "native_model.py").read_text(encoding="utf-8")
    assert "from .native_jit_packing import" in entrypoint
