#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

from scripts.adapter_manifest import ADAPTER_FILES, LEGACY_REMOTE_CODE_FILES
from scripts.sync_hf_adapter_code import sync_one


def _converter_uses_shared_manifest() -> bool:
    """Confirm the converter imports and iterates the shared manifest."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "convert_rwkv7_to_hf.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    imports_manifest = any(
        isinstance(node, ast.ImportFrom)
        and node.module in {"scripts.adapter_manifest", "adapter_manifest"}
        and any(alias.name == "ADAPTER_FILES" for alias in node.names)
        for node in ast.walk(tree)
    )
    iterates_manifest = any(
        isinstance(node, ast.For)
        and isinstance(node.iter, ast.Name)
        and node.iter.id == "ADAPTER_FILES"
        for node in ast.walk(tree)
    )
    return imports_manifest and iterates_manifest


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


def _assert_remote_code_direct_import_closure() -> None:
    """Transformers dynamic-module caching is shallow on some releases.

    Every dependency reached from either AutoModel entrypoint must therefore
    also appear as a direct relative import there (sentinel imports inside
    ``if False`` count for discovery without importing optional kernels).
    """

    root = Path(__file__).resolve().parents[1] / "rwkv7_hf"
    for entrypoint_name in ("native_model.py",):
        direct = _relative_import_files(root / entrypoint_name)
        pending = list(direct)
        transitive: set[str] = set()
        while pending:
            name = pending.pop()
            if name in transitive or not (root / name).exists():
                continue
            transitive.add(name)
            pending.extend(_relative_import_files(root / name) - transitive)
        missing_direct = transitive - direct
        assert not missing_direct, (
            f"{entrypoint_name} has transitive-only trust_remote_code dependencies; "
            f"add non-executed direct imports for: {sorted(missing_direct)}"
        )


def main() -> int:
    # Converted model dirs must include every runtime remote-code module the
    # shipped files transitively import, and the converter and sync lists must
    # stay aligned. (Does not force optional non-runtime files like sglang_quant.)
    _assert_adapter_file_closure()
    _assert_remote_code_direct_import_closure()
    assert _converter_uses_shared_manifest(), "converter does not use shared adapter manifest"

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
        for name in LEGACY_REMOTE_CODE_FILES:
            (model_dir / name).write_text("stale FLA remote code\n", encoding="utf-8")

        result = sync_one(model_dir)
        assert result["model_dir"] == str(model_dir)
        assert result["dry_run"] is False
        for name in ADAPTER_FILES:
            assert (model_dir / name).exists(), name
        assert sorted(Path(path).name for path in result["removed"]) == sorted(
            LEGACY_REMOTE_CODE_FILES
        )
        for name in LEGACY_REMOTE_CODE_FILES:
            assert not (model_dir / name).exists(), name
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        assert cfg["architectures"] == ["NativeRWKV7ForCausalLM"]
        assert cfg["model_type"] == "rwkv7_native"
        assert cfg["auto_map"] == {
            "AutoConfig": "native_model.NativeRWKV7Config",
            "AutoModel": "native_model.NativeRWKV7Model",
            "AutoModelForCausalLM": "native_model.NativeRWKV7ForCausalLM",
        }
        assert weight.read_bytes() == b"do-not-touch"

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
