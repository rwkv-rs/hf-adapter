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
    assert "model_runtime_policy.py" in runner
    assert 'PYTHONPATH="${ROOT_DIR}' in runner
    assert "bench_native_graph_policy_ab.py" in runner
    assert "test_native_quant_mm8_policy.py" in runner
    assert "test_native_quant_mm4_policy.py" in runner
    assert '"${GPU_ARCH}" == "gfx1100"' in runner
    assert "legacy FLA wrapper was not removed" in runner
    assert "RWKV7_NATIVE_MODEL=" not in runner
    assert "rwkv7_hf/modeling_rwkv7.py" not in runner


def test_amd_validation_doc_does_not_promote_unmeasured_kernels() -> None:
    doc = (ROOT / "docs" / "validation" / "AMD_ROCM_HF_VALIDATION.md").read_text(
        encoding="utf-8"
    )

    assert "fully native HF" in doc
    assert "not an Albatross-parity or full-model quantized-speed claim" in doc
    assert "Exact-gfx1100 decode promotion" in doc
    assert "fails closed" in doc
    assert "Exact-gfx1100 output-head W8/W4 decode" in doc
    assert "all 40 output-head rows" in doc
    assert "Full-model memory quantization remains open" in doc
    assert "amd_gfx1100_full_close_20260730" in doc
