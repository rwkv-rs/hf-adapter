#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path, PurePosixPath

from scripts.adapter_manifest import (
    ADAPTER_FILES,
    LEGACY_REMOTE_CODE_FILES,
    copy_manifest_files,
    normalize_manifest_path,
    remove_manifest_files,
    validate_manifest_paths,
)
from scripts.sync_hf_adapter_code import sync_one


def _converter_uses_shared_manifest() -> bool:
    """Confirm the converter imports and calls the shared manifest helpers."""
    script = Path(__file__).resolve().parents[1] / "scripts" / "convert_rwkv7_to_hf.py"
    tree = ast.parse(script.read_text(encoding="utf-8"))
    imported: set[str] = set()
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in {
            "scripts.adapter_manifest",
            "adapter_manifest",
        }:
            imported.update(alias.name for alias in node.names)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"copy_manifest_files", "remove_manifest_files"}
        ):
            called.add(node.func.id)
    required = {
        "ADAPTER_FILES",
        "LEGACY_REMOTE_CODE_FILES",
        "copy_manifest_files",
        "remove_manifest_files",
    }
    return required <= imported and {
        "copy_manifest_files",
        "remove_manifest_files",
    } <= called


def _relative_import_files(root: Path, path: Path) -> set[str]:
    """Manifest-relative Python files referenced by relative imports."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    current = path.relative_to(root)
    package = current.parent.parts
    out: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level < 1:
            continue
        climb = node.level - 1
        if climb > len(package):
            continue
        base = list(package[: len(package) - climb if climb else len(package)])
        modules: list[list[str]] = []
        if node.module:
            modules.append(base + node.module.split("."))
        else:
            modules.extend(base + alias.name.split(".") for alias in node.names)
        for module in modules:
            file_candidate = root.joinpath(*module).with_suffix(".py")
            package_candidate = root.joinpath(*module, "__init__.py")
            if file_candidate.is_file():
                out.add(file_candidate.relative_to(root).as_posix())
            elif package_candidate.is_file():
                out.add(package_candidate.relative_to(root).as_posix())
            else:
                out.add(PurePosixPath(*module).with_suffix(".py").as_posix())
    return out


def _assert_adapter_file_closure() -> None:
    """Every module used by the optional bundled layout must be shipped.

    Missing transitive files break self-contained ``trust_remote_code`` loads.
    This does not force optional non-runtime modules such as ``sglang_quant``
    into the bundle.
    """
    root = Path(__file__).resolve().parents[1] / "rwkv7_hf"
    known = set(ADAPTER_FILES)
    pending = [name for name in ADAPTER_FILES if PurePosixPath(name).suffix == ".py"]
    seen: set[str] = set()
    missing: set[str] = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        for rel in _relative_import_files(root, root / name):
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
        direct = _relative_import_files(root, root / entrypoint_name)
        nested_direct = sorted(name for name in direct if "/" in name)
        assert not nested_direct, (
            "Transformers dynamic-module discovery does not reliably resolve "
            f"nested relative imports from {entrypoint_name}: {nested_direct}"
        )
        pending = list(direct)
        transitive: set[str] = set()
        while pending:
            name = pending.pop()
            # Type-only imports can point back to the remote-code entrypoint.
            # The entrypoint is already present, so it is not a dependency that
            # Transformers needs to discover or copy again.
            if name == entrypoint_name or name in transitive or not (root / name).exists():
                continue
            transitive.add(name)
            pending.extend(_relative_import_files(root, root / name) - transitive)
        missing_direct = transitive - direct
        assert not missing_direct, (
            f"{entrypoint_name} has transitive-only trust_remote_code dependencies; "
            f"add non-executed direct imports for: {sorted(missing_direct)}"
        )


def test_nested_manifest_copy_remove_and_validation() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        source = root / "source"
        destination = root / "destination"
        dry_destination = root / "dry"
        (source / "model").mkdir(parents=True)
        (source / "runtime" / "cuda").mkdir(parents=True)
        (source / "model" / "__init__.py").write_text("MODEL = 1\n", encoding="utf-8")
        (source / "runtime" / "cuda" / "decode.py").write_text(
            "DECODE = 1\n", encoding="utf-8"
        )
        names = ("model/__init__.py", "runtime/cuda/decode.py")

        copied = copy_manifest_files(source, destination, names)
        assert [
            path.relative_to(destination.resolve()).as_posix() for path in copied
        ] == list(names)
        for name in names:
            assert (destination / name).read_bytes() == (source / name).read_bytes()

        dry_copied = copy_manifest_files(
            source, dry_destination, names, dry_run=True
        )
        assert len(dry_copied) == 2
        assert not dry_destination.exists()

        removed = remove_manifest_files(destination, names)
        assert [
            path.relative_to(destination.resolve()).as_posix() for path in removed
        ] == list(names)
        assert all(not (destination / name).exists() for name in names)

        outside = root / "outside"
        outside.mkdir()
        (source / "linked").mkdir()
        (source / "linked" / "payload.py").write_text(
            "PAYLOAD = 1\n", encoding="utf-8"
        )
        linked_destination = destination / "linked"
        try:
            linked_destination.symlink_to(outside, target_is_directory=True)
        except OSError:
            pass  # Windows may not grant symlink creation to the test process.
        else:
            try:
                copy_manifest_files(
                    source, destination, ("linked/payload.py",)
                )
            except ValueError:
                pass
            else:
                raise AssertionError("destination symlink escape should fail")
            assert not (outside / "payload.py").exists()

    assert normalize_manifest_path("model/config.py").as_posix() == "model/config.py"
    for unsafe in (
        "",
        ".",
        "./model.py",
        "../model.py",
        "model/../model.py",
        "/tmp/model.py",
        "model\\config.py",
        "model//config.py",
    ):
        try:
            normalize_manifest_path(unsafe)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe manifest path accepted: {unsafe!r}")
    try:
        validate_manifest_paths(("model/config.py", "model/config.py"))
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate manifest destination should fail")


def test_adapter_manifest_closure_and_sync() -> None:
    # Explicit bundled model dirs must include every runtime remote-code module
    # reached by their entrypoint. The default thin layout does not use this
    # manifest, but bundled conversion and the legacy sync tool must stay aligned.
    _assert_adapter_file_closure()
    _assert_remote_code_direct_import_closure()
    validate_manifest_paths(ADAPTER_FILES)
    assert any("/" in name for name in ADAPTER_FILES), (
        "production manifest no longer exercises nested destination copying"
    )
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
        assert (model_dir / "remote_code" / "__init__.py").is_file()
        assert sorted(Path(path).name for path in result["removed"]) == sorted(
            LEGACY_REMOTE_CODE_FILES
        )
        for name in LEGACY_REMOTE_CODE_FILES:
            assert not (model_dir / name).exists(), name
        cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        assert cfg["architectures"] == ["NativeRWKV7ForCausalLM"]
        assert cfg["model_type"] == "rwkv7_native"
        assert cfg["rwkv7_hf_adapter_layout"] == "bundled"
        assert "rwkv7_hf_runtime_version" not in cfg
        assert cfg["auto_map"] == {
            "AutoConfig": "native_model.NativeRWKV7Config",
            "AutoModel": "native_model.NativeRWKV7Model",
            "AutoModelForCausalLM": "native_model.NativeRWKV7ForCausalLM",
        }
        assert weight.read_bytes() == b"do-not-touch"

        dry_dir = Path(td) / "dry-run-model"
        dry_dir.mkdir()
        dry_config = {
            "architectures": ["OldModel"],
            "model_type": "old_rwkv7",
            "auto_map": {"AutoModelForCausalLM": "old.Model"},
        }
        (dry_dir / "config.json").write_text(
            json.dumps(dry_config) + "\n", encoding="utf-8"
        )
        (dry_dir / LEGACY_REMOTE_CODE_FILES[0]).write_text(
            "keep during dry run\n", encoding="utf-8"
        )
        dry_result = sync_one(dry_dir, dry_run=True)
        assert dry_result["dry_run"] is True
        assert dry_result["copied"]
        assert dry_result["removed"]
        assert not (dry_dir / "remote_code").exists()
        assert (dry_dir / LEGACY_REMOTE_CODE_FILES[0]).is_file()
        assert json.loads(
            (dry_dir / "config.json").read_text(encoding="utf-8")
        ) == dry_config


def main() -> int:
    test_nested_manifest_copy_remove_and_validation()
    test_adapter_manifest_closure_and_sync()
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
