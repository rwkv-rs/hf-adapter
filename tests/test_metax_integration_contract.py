from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from examples import generate
from rwkv7_hf.native_model import NativeRWKV7Config, NativeRWKV7ForCausalLM
from scripts.adapter_manifest import ADAPTER_FILES


ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "f2653e20250821ec48534e5e08b07d59effb985c"


def test_metax_runtime_is_in_converted_checkpoint_manifest() -> None:
    assert "metax_runtime.py" in ADAPTER_FILES
    facade = (ROOT / "rwkv7_hf" / "native_model.py").read_text(encoding="utf-8")
    assert "_native_metax_runtime_dependency_sentinel" in facade


def test_public_package_exposes_import_safe_metax_api() -> None:
    import rwkv7_hf

    for name in (
        "configure_metax_defaults",
        "enable_metax",
        "metax_available",
        "metax_memory_stats",
        "metax_synchronize",
    ):
        assert callable(getattr(rwkv7_hf, name))


def test_generate_auto_selects_exact_metax_before_generic_cuda(monkeypatch) -> None:
    metax_device = SimpleNamespace(type="cuda")
    monkeypatch.setattr(generate, "_metax_available", lambda: True)
    monkeypatch.setattr(generate, "_enable_metax", lambda: metax_device)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert generate.resolve_device("auto") is metax_device
    assert generate.resolve_device("cuda") is metax_device
    assert generate.resolve_dtype("auto", metax_device) is torch.float16


def test_fp16_zero_key_norm_stays_finite_on_native_eager(monkeypatch) -> None:
    monkeypatch.setenv("RWKV7_NATIVE_MODEL_BACKEND", "eager")
    monkeypatch.setenv("RWKV7_NATIVE_MODEL_JIT", "0")
    config = NativeRWKV7Config(
        vocab_size=31,
        hidden_size=8,
        num_hidden_layers=2,
        head_dim=4,
        intermediate_size=16,
        decay_low_rank_dim=3,
        gate_low_rank_dim=3,
        a_low_rank_dim=3,
        v_low_rank_dim=3,
        use_cache=True,
    )
    torch.manual_seed(20260726)
    model = NativeRWKV7ForCausalLM(config).eval().half()
    with torch.no_grad():
        for layer in model.model.layers:
            layer.attn.k_proj.weight.zero_()
    with torch.inference_mode():
        output = model(torch.tensor([[1, 2, 3]], dtype=torch.long), use_cache=True)
    assert torch.isfinite(output.logits).all()


def test_docs_pin_source_and_keep_current_main_rerun_open() -> None:
    document = (ROOT / "docs" / "hardware" / "METAX_C500.md").read_text(
        encoding="utf-8"
    )
    assert SOURCE_COMMIT in document
    assert "current-main" in document
    assert "not a performance" in document
