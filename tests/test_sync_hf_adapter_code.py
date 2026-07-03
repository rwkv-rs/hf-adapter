#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

from scripts.sync_hf_adapter_code import ADAPTER_FILES, sync_one


def _converter_adapter_files() -> list[str]:
    """AST-extract the copy_adapter_files list from convert_rwkv7_to_hf.py."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "convert_rwkv7_to_hf.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "copy_adapter_files":
            for child in ast.walk(node):
                if isinstance(child, ast.For) and isinstance(child.iter, ast.List):
                    values = []
                    for item in child.iter.elts:
                        if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
                            break
                        values.append(item.value)
                    else:
                        return values
    raise AssertionError("could not find adapter file list in convert_rwkv7_to_hf.py")


def _relative_import_files(path: Path) -> set[str]:
    """Relative-import (level==1) module filenames referenced by ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            out.add(node.module.split(".", 1)[0] + ".py")
    return out


def _assert_adapter_file_closure() -> None:
    """Every runtime module transitively imported by the shipped adapter files
    must itself be shipped, else ``trust_remote_code`` load breaks. Catches the
    dplr_*/fused_norm_mix/fused_prefill/native_quant_mm4/mm8 drift. Does NOT
    force optional non-runtime modules (e.g. ``sglang_quant``) to ship."""
    root = Path(__file__).resolve().parents[1] / "rwkv7_hf"
    known = set(ADAPTER_FILES)
    pending = list(ADAPTER_FILES)
    seen: set[str] = set()
    missing: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        for rel in _relative_import_files(root / name):
            if rel not in known:
                missing.add(rel)
            elif rel not in seen:
                pending.append(rel)
    assert not missing, f"adapter files import unshipped modules: {sorted(missing)}"


def main() -> int:
    # Converted model dirs must include every runtime remote-code module the
    # shipped files transitively import, and the converter and sync lists must
    # stay aligned. (Does not force optional non-runtime files like sglang_quant.)
    _assert_adapter_file_closure()
    assert _converter_adapter_files() == ADAPTER_FILES, "convert list != sync list"

    with tempfile.TemporaryDirectory() as td:
        model_dir = Path(td) / "rwkv7-g1d-0.4b-hf"
        model_dir.mkdir()
        weight = model_dir / "model.safetensors"
        weight.write_bytes(b"do-not-touch")
        (model_dir / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["OldModel"],
                    "model_type": "old_rwkv7",
                    "auto_map": {"AutoModelForCausalLM": "old.Model"},
                }
            )
            + "\n",
            encoding="utf-8",
        )

        result = sync_one(model_dir)
        assert result["model_dir"] == str(model_dir)
        assert result["dry_run"] is False
        for name in ADAPTER_FILES:
            assert (model_dir / name).exists(), name
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        assert cfg["architectures"] == ["RWKV7ForCausalLM"]
        assert cfg["model_type"] == "rwkv7_hf_adapter"
        assert cfg["auto_map"]["AutoModelForCausalLM"] == "modeling_rwkv7.RWKV7ForCausalLM"
        assert weight.read_bytes() == b"do-not-touch"

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
