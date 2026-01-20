from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ROOT / "docs" / "BACKENDS.md"


def test_backend_lifecycle_has_a_versioned_removal_window() -> None:
    text = BACKENDS.read_text(encoding="utf-8")
    assert "## Versioned lifecycle and deprecation policy" in text
    assert "remove it no earlier than `X.(Y+2)`" in text
    assert "Silent removal is forbidden" in text
    assert "Security or correctness emergencies" in text


def test_current_experimental_and_compatibility_surfaces_are_registered() -> None:
    text = BACKENDS.read_text(encoding="utf-8")
    for surface in (
        "`native_model.NativeRWKV7*`",
        "Flat converted-model remote-code dependency namespace",
        "Old module paths kept as import shims",
        "`RWKV7_NATIVE_MODEL` selector",
        "`RWKV7_NATIVE_MODEL_BACKEND` and `RWKV7_NATIVE_MODEL_JIT`",
        "Historical FLA-backed RWKV wrapper",
    ):
        assert surface in text
    assert "| 0.6 | 0.8 |" in text
    assert "scripts/sync_hf_adapter_code.py MODEL" in text
