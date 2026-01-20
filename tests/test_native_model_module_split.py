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


def _class_methods(relative: str, class_name: str) -> set[str]:
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return {
        node.name
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_native_model_ownership_is_split_from_entrypoint() -> None:
    entrypoint = _top_level_definitions("rwkv7_hf/native_model.py")
    backbone = _top_level_definitions("rwkv7_hf/model_backbone.py")
    config = _top_level_definitions("rwkv7_hf/model_config.py")
    generation = _top_level_definitions("rwkv7_hf/model_generation.py")
    fast_api = _top_level_definitions("rwkv7_hf/model_fast_api.py")
    cache = _top_level_definitions("rwkv7_hf/model_cache.py")
    layers = _top_level_definitions("rwkv7_hf/model_layers.py")
    prefill_graph = _top_level_definitions("rwkv7_hf/model_prefill_graph.py")
    quantization = _top_level_definitions("rwkv7_hf/model_quantization.py")
    speculative = _top_level_definitions("rwkv7_hf/model_speculative.py")
    causal_lm_methods = _class_methods(
        "rwkv7_hf/native_model.py", "NativeRWKV7ForCausalLM"
    )
    speculative_methods = _class_methods(
        "rwkv7_hf/model_speculative.py", "_NativeSpeculativeGenerationMixin"
    )
    quantization_methods = _class_methods(
        "rwkv7_hf/model_quantization.py", "_NativeQuantizationMixin"
    )
    generation_methods = _class_methods(
        "rwkv7_hf/model_generation.py", "_NativeGenerationContractMixin"
    )
    fast_api_methods = _class_methods(
        "rwkv7_hf/model_fast_api.py", "_NativeFastAPIMixin"
    )

    assert "NativeRWKV7Config" not in entrypoint
    assert "NativeRWKV7Cache" not in entrypoint
    assert "NativeRWKV7Attention" not in entrypoint
    assert "NativeRWKV7FFN" not in entrypoint
    assert "NativeRWKV7Layer" not in entrypoint
    assert "NativeRWKV7Model" not in entrypoint
    assert "_NativePrefillGraphRunner" not in entrypoint
    assert "NativeRWKV7Config" in config
    assert "_NativeGenerationContractMixin" in generation
    assert "_NativeFastAPIMixin" in fast_api
    assert "NativeRWKV7Cache" in cache
    assert "NativeRWKV7Model" in backbone
    assert "_NativePrefillGraphRunner" in prefill_graph
    assert "_NativeQuantizationMixin" in quantization
    assert "_NativeSpeculativeGenerationMixin" in speculative
    assert "rwkv7_speculative_generate" not in causal_lm_methods
    assert "rwkv7_speculative_generate" in speculative_methods
    assert {
        "_reorder_cache",
        "prepare_inputs_for_generation",
    }.isdisjoint(causal_lm_methods)
    assert {
        "_reorder_cache",
        "prepare_inputs_for_generation",
    } <= generation_methods
    assert {
        "rwkv7_native_model_last_decode_backend",
        "rwkv7_native_model_last_prefill_backend",
        "rwkv7_prefill_native",
        "rwkv7_prefill_chunks",
        "rwkv7_last_fast_token_backend",
        "rwkv7_last_fast_prefill_backend",
        "rwkv7_warmup_fast_token",
        "rwkv7_forward_token",
        "rwkv7_forward_one",
    }.isdisjoint(causal_lm_methods)
    assert {
        "rwkv7_native_model_last_decode_backend",
        "rwkv7_native_model_last_prefill_backend",
        "rwkv7_prefill_native",
        "rwkv7_prefill_chunks",
        "rwkv7_last_fast_token_backend",
        "rwkv7_last_fast_prefill_backend",
        "rwkv7_warmup_fast_token",
        "rwkv7_forward_token",
        "rwkv7_forward_one",
    } <= fast_api_methods
    assert {
        "_rwkv7_bnb_concrete_skip_modules",
        "rwkv7_bnb_skip_modules",
        "_rwkv7_prepare_bnb_kwargs",
        "from_pretrained",
        "apply_native_mm_quantization_from_config",
        "_clear_native_jit_pack_cache",
    }.isdisjoint(causal_lm_methods)
    assert {
        "_rwkv7_bnb_concrete_skip_modules",
        "rwkv7_bnb_skip_modules",
        "_rwkv7_prepare_bnb_kwargs",
        "from_pretrained",
        "apply_native_mm_quantization_from_config",
        "_clear_native_jit_pack_cache",
    } <= quantization_methods
    assert {
        "NativeRWKV7Attention",
        "NativeRWKV7FFN",
        "NativeRWKV7Layer",
    } <= layers

    entrypoint_text = (ROOT / "rwkv7_hf" / "native_model.py").read_text(
        encoding="utf-8"
    )
    assert "from .model_config import NativeRWKV7Config" in entrypoint_text
    assert "from .model_generation import _NativeGenerationContractMixin" in entrypoint_text
    assert "from .model_fast_api import _NativeFastAPIMixin" in entrypoint_text
    assert "from .model_cache import (" in entrypoint_text
    assert "from .model_layers import (" in entrypoint_text
    assert "from .model_backbone import (" in entrypoint_text
    assert "from .model_prefill_graph import _NativePrefillGraphRunner" in entrypoint_text
    assert "from .model_quantization import _NativeQuantizationMixin" in entrypoint_text
    assert "from .model_speculative import _NativeSpeculativeGenerationMixin" in entrypoint_text


def test_public_import_identity_and_remote_module_name_stay_stable() -> None:
    import rwkv7_hf
    from rwkv7_hf.model_backbone import NativeRWKV7Model as ModelImplementation
    from rwkv7_hf.model_cache import NativeRWKV7Cache as CacheImplementation
    from rwkv7_hf.model_config import NativeRWKV7Config as ConfigImplementation
    from rwkv7_hf.model_generation import _NativeGenerationContractMixin
    from rwkv7_hf.model_fast_api import _NativeFastAPIMixin
    from rwkv7_hf.model_layers import (
        NativeRWKV7Attention as AttentionImplementation,
        NativeRWKV7FFN as FFNImplementation,
        NativeRWKV7Layer as LayerImplementation,
    )
    from rwkv7_hf.model_prefill_graph import (
        _NativePrefillGraphRunner as PrefillGraphImplementation,
    )
    from rwkv7_hf.model_quantization import _NativeQuantizationMixin
    from rwkv7_hf.model_speculative import _NativeSpeculativeGenerationMixin
    from rwkv7_hf.native_model import (
        NativeRWKV7Attention,
        NativeRWKV7Cache,
        NativeRWKV7Config,
        NativeRWKV7FFN,
        NativeRWKV7Layer,
        NativeRWKV7Model,
        NativeRWKV7ForCausalLM,
        _NativePrefillGraphRunner,
    )

    assert issubclass(NativeRWKV7ForCausalLM, _NativeGenerationContractMixin)
    assert issubclass(NativeRWKV7ForCausalLM, _NativeFastAPIMixin)
    assert issubclass(NativeRWKV7ForCausalLM, _NativeQuantizationMixin)
    assert issubclass(NativeRWKV7ForCausalLM, _NativeSpeculativeGenerationMixin)
    assert (
        NativeRWKV7ForCausalLM.from_pretrained.__func__
        is _NativeQuantizationMixin.from_pretrained.__func__
    )
    assert (
        NativeRWKV7ForCausalLM.apply_native_mm_quantization_from_config
        is _NativeQuantizationMixin.apply_native_mm_quantization_from_config
    )
    assert (
        NativeRWKV7ForCausalLM.rwkv7_speculative_generate
        is _NativeSpeculativeGenerationMixin.rwkv7_speculative_generate
    )
    assert (
        NativeRWKV7ForCausalLM._reorder_cache
        is _NativeGenerationContractMixin._reorder_cache
    )
    assert (
        NativeRWKV7ForCausalLM.prepare_inputs_for_generation
        is _NativeGenerationContractMixin.prepare_inputs_for_generation
    )
    assert (
        NativeRWKV7ForCausalLM.rwkv7_forward_token
        is _NativeFastAPIMixin.rwkv7_forward_token
    )
    assert (
        NativeRWKV7ForCausalLM.rwkv7_prefill_chunks
        is _NativeFastAPIMixin.rwkv7_prefill_chunks
    )
    assert _NativePrefillGraphRunner is PrefillGraphImplementation
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
    assert _NativePrefillGraphRunner.__module__ == "rwkv7_hf.native_model"


def test_split_files_are_in_remote_adapter_manifest() -> None:
    from scripts.adapter_manifest import ADAPTER_FILES

    assert "model_config.py" in ADAPTER_FILES
    assert "model_fast_api.py" in ADAPTER_FILES
    assert "model_cache.py" in ADAPTER_FILES
    assert "model_generation.py" in ADAPTER_FILES
    assert "model_layers.py" in ADAPTER_FILES
    assert "model_backbone.py" in ADAPTER_FILES
    assert "model_prefill_graph.py" in ADAPTER_FILES
    assert "model_quantization.py" in ADAPTER_FILES
    assert "model_speculative.py" in ADAPTER_FILES
