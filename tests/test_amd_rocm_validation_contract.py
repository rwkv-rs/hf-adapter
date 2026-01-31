from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_amd_runner_targets_decoupled_native_hf_adapter() -> None:
    runner = (ROOT / "bench" / "run_amd_rocm_hf_validation.sh").read_text(
        encoding="utf-8"
    )

    assert "native_model.NativeRWKV7ForCausalLM" in runner
    assert "test_native_model_module_split.py" in runner
    assert "model_fast_api.py" in runner
    assert 'PYTHONPATH="${ROOT_DIR}' in runner
    assert "legacy FLA wrapper was not removed" in runner
    assert "RWKV7_NATIVE_MODEL=" not in runner
    assert "rwkv7_hf/modeling_rwkv7.py" not in runner


def test_amd_validation_doc_does_not_promote_unmeasured_kernels() -> None:
    doc = (ROOT / "docs" / "validation" / "AMD_ROCM_HF_VALIDATION.md").read_text(
        encoding="utf-8"
    )

    assert "fully native HF" in doc
    assert "not an Albatross-parity or quantized-speed claim" in doc
    assert "HIP-native W8/W4" in doc
