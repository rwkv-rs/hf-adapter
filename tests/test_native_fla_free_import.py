#!/usr/bin/env python3
# coding=utf-8
"""Fla-free import smoke for the opt-in native RWKV-7 backend.

This test blocks any ``fla`` import and verifies that:
1. remote-code config/modeling modules are still importable,
2. ``NativeRWKV7Cache`` falls back to the HF ``Cache`` base, and
3. ``RWKV7_NATIVE_MODEL=1`` can route the wrapper class to
   ``NativeRWKV7ForCausalLM.from_pretrained`` when a model directory is given.

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

    os.environ["RWKV7_NATIVE_MODEL"] = "1"
    sys.meta_path.insert(0, BlockFla())
    _clear_modules()

    from transformers.cache_utils import Cache

    native_model = importlib.import_module("rwkv7_hf.native_model")
    cache = native_model.NativeRWKV7Cache.from_legacy_cache(None, seen_tokens=7)
    assert isinstance(cache, Cache), type(cache)
    assert cache.get_seq_length() == 7

    config_mod = importlib.import_module("rwkv7_hf.configuration_rwkv7")
    cfg = config_mod.RWKV7Config(hidden_size=8, num_hidden_layers=1, vocab_size=16)
    assert cfg.model_type == "rwkv7_hf_adapter"
    assert cfg.hidden_size == 8

    modeling = importlib.import_module("rwkv7_hf.modeling_rwkv7")
    assert modeling._FLA_IMPORT_ERROR is not None

    if args.model:
        # Exercise the real HF remote-code path too: copy the worktree code
        # beside symlinked checkpoint files so AutoModel imports this PR's
        # modeling/config files while FLA remains blocked.
        src = Path(args.model).resolve()
        tmp = Path(tempfile.mkdtemp(prefix="rwkv7_fla_free_model_"))
        code_dir = Path(modeling.__file__).resolve().parent
        for item in src.iterdir():
            target = tmp / item.name
            if item.is_dir():
                os.symlink(item, target, target_is_directory=True)
            else:
                os.symlink(item, target)
        for py_file in code_dir.glob("*.py"):
            target = tmp / py_file.name
            if target.exists() or target.is_symlink():
                target.unlink()
            shutil.copy2(py_file, target)

        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(tmp, trust_remote_code=True, torch_dtype="auto")
        assert model.__class__.__name__ == "NativeRWKV7ForCausalLM", type(model)

    print("NATIVE FLA-FREE IMPORT PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
