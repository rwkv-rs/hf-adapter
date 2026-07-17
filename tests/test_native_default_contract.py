from __future__ import annotations

import json
from pathlib import Path

import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_native_default_architecture_contract_is_explicit() -> None:
    text = (ROOT / "docs" / "architecture" / "NATIVE_DEFAULT_BACKEND.md").read_text(
        encoding="utf-8"
    )

    for required in (
        "NativeRWKV7ForCausalLM",
        "FLA reference backend",
        "must not import FLA",
        "Qwen full-FLA",
        "RWKV7_NATIVE_MODEL",
        "41.68 tok/s",
        "226.3 tok/s",
    ):
        assert required in text


def test_official_train_temp_shell_recipe_is_an_acceptance_gate() -> None:
    text = (ROOT / "docs" / "ACCEPTANCE.md").read_text(encoding="utf-8")

    for required in (
        "demo-training-prepare.sh",
        "demo-training-run.sh",
        "micro_bsz=1",
        "micro_bsz=16",
        "FFN2688",
        "ctx_len=512",
        "lr_init=6e-4",
        "lr_final=6e-5",
        "adam_eps=1e-18",
        "weight_decay=0.001",
        "grad_cp=1",
        "deepspeed_stage_2",
        "magic_prime=2926181",
        "train_temp_official_x070_12x768_b16.json",
    ):
        assert required in text


def test_native_default_performance_gate_rejects_a_flag_only_migration() -> None:
    text = (ROOT / "docs" / "performance" / "FUSED_BACKEND.md").read_text(
        encoding="utf-8"
    )

    assert "native-default-5070" in text
    assert "0.95x" in text
    assert "wrapper-hosted native_graph" in text
    assert "pure-native" in text


def test_base_and_cuda_install_profiles_do_not_require_fla() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    base = project["dependencies"]
    extras = project["optional-dependencies"]

    assert not any("flash-linear-attention" in item for item in base)
    assert not any("flash-linear-attention" in item for item in extras["cuda"])
    assert any("flash-linear-attention" in item for item in extras["fla-reference"])


def test_package_public_classes_are_native() -> None:
    import rwkv7_hf
    from rwkv7_hf.native_model import NativeRWKV7Config, NativeRWKV7ForCausalLM, NativeRWKV7Model

    assert rwkv7_hf.RWKV7Config is NativeRWKV7Config
    assert rwkv7_hf.RWKV7Model is NativeRWKV7Model
    assert rwkv7_hf.RWKV7ForCausalLM is NativeRWKV7ForCausalLM


def test_repo_code_benchmark_overlay_migrates_config_without_touching_source(tmp_path) -> None:
    from bench.bench_native_quant_e2e_decode import prepare_model_dir

    source = tmp_path / "converted"
    source.mkdir()
    source_config = {
        "architectures": ["RWKV7ForCausalLM"],
        "model_type": "rwkv7_hf_adapter",
        "auto_map": {"AutoModelForCausalLM": "modeling_rwkv7.RWKV7ForCausalLM"},
    }
    (source / "config.json").write_text(json.dumps(source_config), encoding="utf-8")
    (source / "model.safetensors").write_bytes(b"weights")

    prepared, temporary = prepare_model_dir(str(source), "repo")
    try:
        migrated = json.loads((Path(prepared) / "config.json").read_text(encoding="utf-8"))
        original = json.loads((source / "config.json").read_text(encoding="utf-8"))
        assert migrated["architectures"] == ["NativeRWKV7ForCausalLM"]
        assert migrated["model_type"] == "rwkv7_native"
        assert migrated["auto_map"]["AutoModelForCausalLM"] == (
            "native_model.NativeRWKV7ForCausalLM"
        )
        assert original == source_config
        assert (Path(prepared) / "model.safetensors").read_bytes() == b"weights"
    finally:
        assert temporary is not None
        temporary.cleanup()
