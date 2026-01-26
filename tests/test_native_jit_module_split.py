from __future__ import annotations

import ast
from pathlib import Path

import torch
import torch.nn.functional as F

from rwkv7_hf import native_jit, native_jit_linear


ROOT = Path(__file__).resolve().parents[1]


def _top_level_definitions(relative: str) -> set[str]:
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_linear_helper_ownership_moves_without_hot_path_wrappers() -> None:
    facade = _top_level_definitions("rwkv7_hf/native_jit.py")
    implementation = _top_level_definitions("rwkv7_hf/native_jit_linear.py")

    assert {
        "_linear_module",
        "_graph_linear_operand",
        "_graph_linear_is_dense",
        "_graph_linear_shape",
        "_native_graph_relayout_ffn_value_weight",
    }.isdisjoint(facade)
    assert {
        "linear_module",
        "graph_linear_operand",
        "graph_linear_is_dense",
        "graph_linear_shape",
        "relayout_ffn_value_weight",
    } <= implementation
    assert native_jit._linear_module is native_jit_linear.linear_module
    assert native_jit._graph_linear_operand is native_jit_linear.graph_linear_operand
    assert native_jit._graph_linear_is_dense is native_jit_linear.graph_linear_is_dense
    assert native_jit._graph_linear_shape is native_jit_linear.graph_linear_shape
    assert (
        native_jit._native_graph_relayout_ffn_value_weight
        is native_jit_linear.relayout_ffn_value_weight
    )


def test_dense_and_callable_quant_linear_contract() -> None:
    dense = torch.nn.Linear(4, 3, bias=True)
    x = torch.randn(2, 4)
    assert torch.equal(
        native_jit._linear_module(dense, x),
        F.linear(x, dense.weight, dense.bias),
    )
    assert native_jit._graph_linear_operand(dense) is dense.weight
    assert native_jit._graph_linear_is_dense(dense.weight)
    assert native_jit._graph_linear_shape(dense.weight) == (3, 4)

    class PackedLinear(torch.nn.Module):
        in_features = 4
        out_features = 3

        def forward(self, value):
            return value[:, :3] + 1

    packed = PackedLinear()
    assert native_jit._graph_linear_operand(packed) is packed
    assert not native_jit._graph_linear_is_dense(packed)
    assert native_jit._graph_linear_shape(packed) == (3, 4)
    assert torch.equal(native_jit._linear_module(packed, x), packed(x))


def test_sparse_relayout_preserves_parameter_contract() -> None:
    linear = torch.nn.Linear(6, 4, bias=False)
    original = linear.weight.detach().clone()
    keys_before = tuple(linear.state_dict())

    weight = native_jit._native_graph_relayout_ffn_value_weight(linear)

    assert tuple(linear.state_dict()) == keys_before
    assert weight.shape == original.shape
    assert torch.equal(weight, original)
    assert not weight.is_contiguous()
    assert weight.transpose(0, 1).is_contiguous()
    assert getattr(linear, "_rwkv7_sparse_low_memory_layout") is True
    assert native_jit._native_graph_relayout_ffn_value_weight(linear) is weight


def test_try_relayout_wrapper_keeps_native_jit_patch_surface(monkeypatch) -> None:
    sentinel = object()
    seen = {}

    def replacement(_module):
        return sentinel

    def try_impl(module, *, relayout_fn):
        seen["module"] = module
        seen["relayout_fn"] = relayout_fn
        return True

    monkeypatch.setattr(
        native_jit,
        "_native_graph_relayout_ffn_value_weight",
        replacement,
    )
    monkeypatch.setattr(native_jit, "_try_relayout_ffn_value_weight", try_impl)
    module = object()

    assert native_jit._native_graph_try_relayout_ffn_value_weight(module)
    assert seen == {"module": module, "relayout_fn": replacement}


def test_native_jit_linear_is_shipped_with_remote_adapter() -> None:
    from scripts.adapter_manifest import ADAPTER_FILES

    assert "native_jit_linear.py" in ADAPTER_FILES
    entrypoint = (ROOT / "rwkv7_hf" / "native_model.py").read_text(encoding="utf-8")
    assert "from .native_jit_linear import" in entrypoint
