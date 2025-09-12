#!/usr/bin/env python3
# coding=utf-8
"""Sync latest RWKV-7 HF adapter remote-code files into converted model dirs.

Converted checkpoints carry copies of ``configuration_rwkv7.py``,
``modeling_rwkv7.py``, tokenizer code, and native helper modules so standard
``AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`` can load
them without installing this repository.  As the adapter evolves, older
converted dirs need those small Python files refreshed without rewriting large
``model.safetensors`` weights.  This helper performs that code-only sync and
refreshes the Auto* metadata in ``config.json``.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ADAPTER_FILES = [
    "configuration_rwkv7.py",
    "modeling_rwkv7.py",
    "tokenization_rwkv7.py",
    "native_jit.py",
    "native.py",
]


def sync_one(model_dir: Path, *, dry_run: bool = False) -> dict:
    root = Path(__file__).resolve().parents[1]
    src_dir = root / "rwkv7_hf"
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model dir not found: {model_dir}")
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")

    copied = []
    for name in ADAPTER_FILES:
        src = src_dir / name
        dst = model_dir / name
        if not src.exists():
            raise FileNotFoundError(f"adapter source missing: {src}")
        copied.append(str(dst))
        if not dry_run:
            shutil.copyfile(src, dst)

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["architectures"] = ["RWKV7ForCausalLM"]
    cfg["model_type"] = "rwkv7_hf_adapter"
    cfg["auto_map"] = {
        "AutoConfig": "configuration_rwkv7.RWKV7Config",
        "AutoModel": "modeling_rwkv7.RWKV7Model",
        "AutoModelForCausalLM": "modeling_rwkv7.RWKV7ForCausalLM",
    }
    if not dry_run:
        cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return {"model_dir": str(model_dir), "copied": copied, "dry_run": dry_run}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dirs", nargs="+", help="Converted HF model directories to refresh")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for item in args.model_dirs:
        print(json.dumps(sync_one(Path(item), dry_run=args.dry_run), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
