from __future__ import annotations

from pathlib import Path


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
        "micro_bsz=16",
        "ctx_len=512",
        "lr_init=6e-4",
        "lr_final=6e-5",
        "adam_eps=1e-18",
        "weight_decay=0.001",
        "grad_cp=1",
        "deepspeed_stage_2",
        "magic_prime=2926181",
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
