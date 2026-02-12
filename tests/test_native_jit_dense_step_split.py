from __future__ import annotations

import ast
from pathlib import Path

import torch

from rwkv7_hf import native_jit, native_jit_dense_step


ROOT = Path(__file__).resolve().parents[1]


def _top_level_functions(relative: str) -> set[str]:
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_dense_step_ownership_moves_without_hot_path_wrappers() -> None:
    facade = _top_level_functions("rwkv7_hf/native_jit.py")
    implementation = _top_level_functions("rwkv7_hf/native_jit_dense_step.py")
    moved = {"block_step", "block_step_batched"}

    assert moved.isdisjoint(facade)
    assert moved <= implementation
    assert native_jit.block_step is native_jit_dense_step.block_step
    assert native_jit.block_step_batched is native_jit_dense_step.block_step_batched
    assert isinstance(native_jit.block_step, torch.jit.ScriptFunction)
    assert isinstance(native_jit.block_step_batched, torch.jit.ScriptFunction)


def test_dense_step_module_is_shipped_with_remote_adapter() -> None:
    from scripts.adapter_manifest import ADAPTER_FILES

    assert "native_jit_dense_step.py" in ADAPTER_FILES
    entrypoint = (ROOT / "rwkv7_hf" / "native_model.py").read_text(encoding="utf-8")
    assert "from .native_jit_dense_step import" in entrypoint
