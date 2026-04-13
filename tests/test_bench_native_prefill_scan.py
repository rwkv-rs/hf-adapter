from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.bench_native_prefill_scan import (
    native_jit_packs,
    prepare_model_dir,
    reported_effective_model_path,
)


@pytest.mark.parametrize(
    "name",
    ["_native_jit_packs", "_native_graph_packs", "_rwkv7_native_jit_packs"],
)
def test_native_jit_packs_supports_all_model_interfaces(name: str) -> None:
    sentinel = object()
    model = type("Model", (), {name: lambda self: sentinel})()
    assert native_jit_packs(model) is sentinel


def test_native_jit_packs_prefers_native_model_interface() -> None:
    model = type(
        "Model",
        (),
        {
            "_native_jit_packs": lambda self: "native",
            "_rwkv7_native_jit_packs": lambda self: "legacy",
        },
    )()
    assert native_jit_packs(model) == "native"


def test_native_jit_packs_rejects_unknown_model() -> None:
    with pytest.raises(AttributeError, match="projection packs"):
        native_jit_packs(object())


def test_repo_code_prefill_overlay_selects_canonical_native_model(tmp_path: Path) -> None:
    source = tmp_path / "converted"
    source.mkdir()
    source_config = {
        "architectures": ["RWKV7ForCausalLM"],
        "model_type": "rwkv7_hf_adapter",
        "auto_map": {"AutoModelForCausalLM": "modeling_rwkv7.RWKV7ForCausalLM"},
    }
    (source / "config.json").write_text(json.dumps(source_config), encoding="utf-8")
    (source / "model.safetensors").write_bytes(b"weights")

    prepared, temporary = prepare_model_dir(str(source), code_source="repo")
    try:
        migrated = json.loads((Path(prepared) / "config.json").read_text(encoding="utf-8"))
        assert migrated["architectures"] == ["NativeRWKV7ForCausalLM"]
        assert migrated["model_type"] == "rwkv7_native"
        assert migrated["auto_map"]["AutoModelForCausalLM"] == (
            "native_model.NativeRWKV7ForCausalLM"
        )
        assert json.loads((source / "config.json").read_text(encoding="utf-8")) == source_config
        assert (Path(prepared) / "model.safetensors").read_bytes() == b"weights"
    finally:
        assert temporary is not None
        temporary.cleanup()


def test_repo_code_result_hides_temporary_absolute_overlay_path() -> None:
    args = SimpleNamespace(
        code_source="repo",
        model="relative-model",
        effective_model_path=r"D:\private\rwkv7_repo_code_model_random",
    )
    assert reported_effective_model_path(args) == "<temporary-repo-code-overlay>"


def test_existing_code_result_reports_the_effective_model_path() -> None:
    args = SimpleNamespace(
        code_source="existing",
        model="relative-model",
        effective_model_path="prepared-model",
    )
    assert reported_effective_model_path(args) == "prepared-model"
