from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from rwkv7_hf import native_jit, native_jit_prefill_runtime_policy


ROOT = Path(__file__).resolve().parents[1]


def _top_level_functions(relative: str) -> set[str]:
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_prefill_runtime_policy_ownership_moves_behind_compatibility_wrappers() -> None:
    facade = _top_level_functions("rwkv7_hf/native_jit.py")
    implementation = _top_level_functions(
        "rwkv7_hf/native_jit_prefill_runtime_policy.py"
    )
    moved = {
        "_native_prefill_fused_scan_enabled",
        "_native_prefill_self_chunk_enabled",
        "_native_prefill_fused_shift_mix_enabled",
        "_native_prefill_fused_state_prep_enabled",
        "_native_prefill_fused_output_enabled",
        "_native_prefill_fused_sequence_ffn_enabled",
        "_native_prefill_stacked_rkv_enabled",
    }

    assert moved.isdisjoint(facade)
    assert moved <= implementation
    for name in moved:
        wrapper = getattr(native_jit, name)
        assert wrapper.__wrapped__ is getattr(native_jit_prefill_runtime_policy, name)


def test_prefill_runtime_policy_keeps_facade_override_surface(monkeypatch) -> None:
    policy = SimpleNamespace(
        fused_prefill_scan=True,
        prefill_scan_model_shapes=((2048, 24, 8, 512),),
    )
    monkeypatch.setattr(native_jit, "_kernel_policy", lambda: policy)
    monkeypatch.setattr(native_jit, "fused_recurrent_scan", object())
    monkeypatch.setattr(native_jit, "fused_recurrent_scan_available", lambda: True)
    monkeypatch.delenv("RWKV7_NATIVE_PREFILL_FUSED_SCAN", raising=False)

    assert native_jit._native_prefill_fused_scan_enabled(8, 512, 2048, 24)
    assert not native_jit._native_prefill_fused_scan_enabled(1, 128, 2048, 24)


def test_prefill_runtime_policy_module_is_shipped_with_remote_adapter() -> None:
    from scripts.adapter_manifest import ADAPTER_FILES

    assert "native_jit_prefill_runtime_policy.py" in ADAPTER_FILES
    entrypoint = (ROOT / "rwkv7_hf" / "native_model.py").read_text(encoding="utf-8")
    assert "from .native_jit_prefill_runtime_policy import" in entrypoint
