#!/usr/bin/env python3
# coding=utf-8
"""FLA-free import smoke for the default native RWKV-7 backend.

This test blocks any ``fla`` import and verifies that:
1. remote-code config/modeling modules are still importable,
2. ``NativeRWKV7Cache`` falls back to the HF ``Cache`` base, and
3. converted native Auto* metadata loads ``NativeRWKV7ForCausalLM`` without a
   backend-selection environment variable.

Usage:
  python tests/test_native_fla_free_import.py
  python tests/test_native_fla_free_import.py --model <hf_dir>
"""
from __future__ import annotations

import argparse
import importlib
import importlib.abc
import os
import shutil
import sys
import tempfile
from pathlib import Path


class BlockFla(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "fla" or fullname.startswith("fla."):
            raise ImportError("blocked fla for fla-free native backend test")
        return None


def _clear_modules() -> None:
    for name in list(sys.modules):
        if name == "fla" or name.startswith("fla.") or name == "rwkv7_hf" or name.startswith("rwkv7_hf."):
            del sys.modules[name]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="")
    args = ap.parse_args()

    os.environ.pop("RWKV7_NATIVE_MODEL", None)
    sys.meta_path.insert(0, BlockFla())
    _clear_modules()

    from transformers.cache_utils import Cache

    native_model = importlib.import_module("rwkv7_hf.native_model")
    package = importlib.import_module("rwkv7_hf")
    assert package.RWKV7Config is native_model.NativeRWKV7Config
    assert package.RWKV7Model is native_model.NativeRWKV7Model
    assert package.RWKV7ForCausalLM is native_model.NativeRWKV7ForCausalLM
    assert package.RWKV7StateCache is native_model.NativeRWKV7Cache
    cache = native_model.NativeRWKV7Cache.from_legacy_cache(None, seen_tokens=7)
    assert isinstance(cache, Cache), type(cache)
    assert cache.get_seq_length() == 7

    config_mod = importlib.import_module("rwkv7_hf.configuration_rwkv7")
    cfg = config_mod.RWKV7Config(hidden_size=8, num_hidden_layers=1, vocab_size=16)
    assert cfg.model_type == "rwkv7_hf_adapter"
    assert cfg.hidden_size == 8

    modeling = importlib.import_module("rwkv7_hf.modeling_rwkv7")
    assert modeling._FLA_IMPORT_ERROR is not None
    kernel_policy = importlib.import_module("rwkv7_hf.kernel_policy")

    class FakeQuantConfig:
        llm_int8_skip_modules = ["existing"]

    quant_config = FakeQuantConfig()
    kwargs = {"quantization_config": quant_config, "rwkv7_bnb_skip_policy": "memory", "config": cfg}
    policy, prepared_quant_config = modeling.RWKV7ForCausalLM._rwkv7_prepare_bnb_kwargs("unused", kwargs)
    assert policy == "memory"
    assert prepared_quant_config is quant_config
    assert kwargs["quantization_config"] is quant_config
    assert "rwkv7_bnb_skip_policy" not in kwargs
    assert "existing" in quant_config.llm_int8_skip_modules
    assert "lm_head" in quant_config.llm_int8_skip_modules
    assert "model.layers.0.attn.w_lora.lora.0" in quant_config.llm_int8_skip_modules
    assert "model.layers.0.attn.g_lora.lora.2" in quant_config.llm_int8_skip_modules

    old_policy = modeling.current_kernel_policy
    try:
        os.environ.pop("RWKV7_NATIVE_MODEL", None)
        modeling.current_kernel_policy = lambda **_: kernel_policy.policy_for_profile(
            kernel_policy.classify_gpu("NVIDIA GeForce GTX 1080 Ti", (6, 1))
        )
        assert modeling._native_model_backend_requested() is True

        os.environ["RWKV7_NATIVE_MODEL"] = "0"
        assert modeling._native_model_backend_requested() is False

        os.environ["RWKV7_NATIVE_MODEL"] = "1"
        assert modeling._native_model_backend_requested() is True
    finally:
        modeling.current_kernel_policy = old_policy

    if args.model:
        # Exercise the real HF remote-code path too: copy the worktree code
        # beside symlinked checkpoint files so AutoModel imports this PR's
        # modeling/config files while FLA remains blocked.
        src = Path(args.model).resolve()
        temporary_model = tempfile.TemporaryDirectory(
            prefix="rwkv7_fla_free_model_", dir=str(src.parent)
        )
        tmp = Path(temporary_model.name)
        code_dir = Path(modeling.__file__).resolve().parent
        for item in src.iterdir():
            target = tmp / item.name
            if item.is_dir():
                continue
            if item.suffix in {".safetensors", ".bin", ".pth"}:
                os.link(item, target)
            else:
                shutil.copy2(item, target)
        for py_file in code_dir.glob("*.py"):
            target = tmp / py_file.name
            if target.exists() or target.is_symlink():
                target.unlink()
            shutil.copy2(py_file, target)

        from transformers import AutoModelForCausalLM

        config_path = tmp / "config.json"
        config = __import__("json").loads(config_path.read_text(encoding="utf-8"))
        config["architectures"] = ["NativeRWKV7ForCausalLM"]
        config["model_type"] = "rwkv7_native"
        config["auto_map"] = {
            "AutoConfig": "native_model.NativeRWKV7Config",
            "AutoModel": "native_model.NativeRWKV7Model",
            "AutoModelForCausalLM": "native_model.NativeRWKV7ForCausalLM",
        }
        config_path.write_text(
            __import__("json").dumps(config, indent=2) + "\n", encoding="utf-8"
        )

        from transformers import AutoConfig, AutoModelForCausalLM

        loaded_config = AutoConfig.from_pretrained(tmp, trust_remote_code=True)
        assert loaded_config.__class__.__name__ == "NativeRWKV7Config"
        model = AutoModelForCausalLM.from_pretrained(
            tmp, trust_remote_code=True, torch_dtype="auto"
        )
        assert model.__class__.__name__ == "NativeRWKV7ForCausalLM", type(model)
        del model
        temporary_model.cleanup()

    print("NATIVE FLA-FREE IMPORT PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
