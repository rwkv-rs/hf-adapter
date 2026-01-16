from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _top_level_definitions(relative: str) -> set[str]:
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_native_model_ownership_is_split_from_entrypoint() -> None:
    entrypoint = _top_level_definitions("rwkv7_hf/native_model.py")
    backbone = _top_level_definitions("rwkv7_hf/model_backbone.py")
    config = _top_level_definitions("rwkv7_hf/model_config.py")
    cache = _top_level_definitions("rwkv7_hf/model_cache.py")
    layers = _top_level_definitions("rwkv7_hf/model_layers.py")

    assert "NativeRWKV7Config" not in entrypoint
    assert "NativeRWKV7Cache" not in entrypoint
    assert "NativeRWKV7Attention" not in entrypoint
    assert "NativeRWKV7FFN" not in entrypoint
    assert "NativeRWKV7Layer" not in entrypoint
    assert "NativeRWKV7Model" not in entrypoint
    assert "NativeRWKV7Config" in config
    assert "NativeRWKV7Cache" in cache
    assert "NativeRWKV7Model" in backbone
    assert {
        "NativeRWKV7Attention",
        "NativeRWKV7FFN",
        "NativeRWKV7Layer",
    } <= layers

    entrypoint_text = (ROOT / "rwkv7_hf" / "native_model.py").read_text(
        encoding="utf-8"
    )
    assert "from .model_config import NativeRWKV7Config" in entrypoint_text
    assert "from .model_cache import (" in entrypoint_text
    assert "from .model_layers import (" in entrypoint_text
    assert "from .model_backbone import (" in entrypoint_text


def test_public_import_identity_and_remote_module_name_stay_stable() -> None:
    import rwkv7_hf
    from rwkv7_hf.model_backbone import NativeRWKV7Model as ModelImplementation
    from rwkv7_hf.model_cache import NativeRWKV7Cache as CacheImplementation
    from rwkv7_hf.model_config import NativeRWKV7Config as ConfigImplementation
    from rwkv7_hf.model_layers import (
        NativeRWKV7Attention as AttentionImplementation,
        NativeRWKV7FFN as FFNImplementation,
        NativeRWKV7Layer as LayerImplementation,
    )
    from rwkv7_hf.native_model import (
        NativeRWKV7Attention,
        NativeRWKV7Cache,
        NativeRWKV7Config,
        NativeRWKV7FFN,
        NativeRWKV7Layer,
        NativeRWKV7Model,
    )

    assert NativeRWKV7Model is ModelImplementation
    assert NativeRWKV7Config is ConfigImplementation
    assert NativeRWKV7Cache is CacheImplementation
    assert NativeRWKV7Attention is AttentionImplementation
    assert NativeRWKV7FFN is FFNImplementation
    assert NativeRWKV7Layer is LayerImplementation
    assert rwkv7_hf.NativeRWKV7Config is NativeRWKV7Config
    assert rwkv7_hf.NativeRWKV7Cache is NativeRWKV7Cache
    assert NativeRWKV7Config.__module__ == "rwkv7_hf.native_model"
    assert NativeRWKV7Cache.__module__ == "rwkv7_hf.native_model"
    assert NativeRWKV7Attention.__module__ == "rwkv7_hf.native_model"
    assert NativeRWKV7FFN.__module__ == "rwkv7_hf.native_model"
    assert NativeRWKV7Layer.__module__ == "rwkv7_hf.native_model"
    assert NativeRWKV7Model.__module__ == "rwkv7_hf.native_model"


def test_split_files_are_in_remote_adapter_manifest() -> None:
    from scripts.adapter_manifest import ADAPTER_FILES

    assert "model_config.py" in ADAPTER_FILES
    assert "model_cache.py" in ADAPTER_FILES
    assert "model_layers.py" in ADAPTER_FILES
    assert "model_backbone.py" in ADAPTER_FILES
