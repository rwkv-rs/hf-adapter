from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from rwkv7_hf import native_jit, native_jit_prefill_policy


ROOT = Path(__file__).resolve().parents[1]


def _imports(relative: str) -> set[str]:
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    modules = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    modules.update(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    return modules


def test_prefill_policy_module_is_torch_and_kernel_independent() -> None:
    imports = _imports("rwkv7_hf/native_jit_prefill_policy.py")
    assert "torch" not in imports
    assert not any("kernel" in name for name in imports)
    assert not any("triton" in name for name in imports)


def test_model_shape_policy_and_environment_contract(monkeypatch) -> None:
    policy = SimpleNamespace(prefill_scan_model_shapes=((4096, 61, 8, 512),))
    monkeypatch.setattr(native_jit, "_kernel_policy", lambda: policy)
    monkeypatch.delenv("RWKV7_TEST_MODEL_SHAPES", raising=False)

    assert native_jit._native_prefill_model_shape_selected(
        "RWKV7_TEST_MODEL_SHAPES",
        "prefill_scan_model_shapes",
        8,
        512,
        4096,
        61,
    )
    assert not native_jit._native_prefill_model_shape_selected(
        "RWKV7_TEST_MODEL_SHAPES",
        "prefill_scan_model_shapes",
        1,
        128,
        4096,
        61,
    )

    monkeypatch.setenv("RWKV7_TEST_MODEL_SHAPES", "2048x24x1x128")
    assert native_jit._native_prefill_model_shape_selected(
        "RWKV7_TEST_MODEL_SHAPES",
        "prefill_scan_model_shapes",
        1,
        128,
        2048,
        24,
    )
    monkeypatch.setenv("RWKV7_TEST_MODEL_SHAPES", "invalid")
    with pytest.raises(ValueError, match="must contain HxLxBxT tuples"):
        native_jit._native_prefill_model_shape_selected(
            "RWKV7_TEST_MODEL_SHAPES",
            "prefill_scan_model_shapes",
            1,
            128,
            2048,
            24,
        )


def test_model_shape_policy_accepts_bounded_dynamic_profiles(monkeypatch) -> None:
    policy = SimpleNamespace(
        prefill_scan_model_shapes=((2048, 24, 4, 512),),
        prefill_scan_model_profiles=((2048, 24, 8, 4096, 16384),),
    )
    monkeypatch.setattr(native_jit, "_kernel_policy", lambda: policy)
    monkeypatch.delenv("RWKV7_TEST_MODEL_SHAPES", raising=False)

    selected = native_jit._native_prefill_model_shape_selected
    args = ("RWKV7_TEST_MODEL_SHAPES", "prefill_scan_model_shapes")
    assert selected(*args, 3, 129, 2048, 24)
    assert selected(*args, 8, 2048, 2048, 24)
    assert selected(*args, 3, 4096, 2048, 24)
    assert not selected(*args, 9, 128, 2048, 24)
    assert not selected(*args, 5, 4096, 2048, 24)
    assert not selected(*args, 3, 4097, 2048, 24)
    assert not selected(*args, 3, 129, 2560, 32)

    # An explicit exact-shape environment override remains fail-closed and
    # does not silently inherit the policy's dynamic profile.
    monkeypatch.setenv("RWKV7_TEST_MODEL_SHAPES", "2048x24x4x512")
    assert selected(*args, 4, 512, 2048, 24)
    assert not selected(*args, 3, 129, 2048, 24)


def test_self_chunk_size_and_tiles_keep_native_jit_policy_patch(monkeypatch) -> None:
    policy = SimpleNamespace(
        prefill_self_chunk_size=16,
        prefill_self_chunk_shape_sizes=((8, 512, 64),),
        prefill_self_chunk_h_tile_shapes=((8, 512, 4, 2),),
    )
    monkeypatch.setattr(native_jit, "_kernel_policy", lambda: policy)
    monkeypatch.delenv("RWKV7_NATIVE_PREFILL_SELF_CHUNK_SIZE", raising=False)

    assert native_jit._native_prefill_self_chunk_size(8, 512) == 64
    assert native_jit._native_prefill_self_chunk_size(1, 128) == 16
    assert native_jit._native_prefill_self_chunk_h_tiles(8, 512) == (4, 2)
    assert native_jit._native_prefill_self_chunk_h_tiles(1, 128) is None


def test_self_chunk_shape_eligibility_promotes_exact_shapes() -> None:
    policy = SimpleNamespace(
        prefill_self_chunk_model_shapes=((4096, 61, 8, 512),),
        prefill_self_chunk_model_shapes_only=True,
    )
    assert native_jit_prefill_policy.self_chunk_shape_eligible(
        policy=policy,
        tokens=512,
        head_dim=64,
        batch_size=8,
        hidden_size=4096,
        num_layers=61,
        min_tokens=1024,
        raw_model_shapes=None,
    )
    assert not native_jit_prefill_policy.self_chunk_shape_eligible(
        policy=policy,
        tokens=512,
        head_dim=64,
        batch_size=1,
        hidden_size=4096,
        num_layers=61,
        min_tokens=1024,
        raw_model_shapes=None,
    )


def test_prefill_policy_is_shipped_with_remote_adapter() -> None:
    from scripts.adapter_manifest import ADAPTER_FILES

    assert "native_jit_prefill_policy.py" in ADAPTER_FILES
    entrypoint = (ROOT / "rwkv7_hf" / "native_model.py").read_text(encoding="utf-8")
    assert "from .native_jit_prefill_policy import" in entrypoint
