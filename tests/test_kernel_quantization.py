from __future__ import annotations

import importlib
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def source_kernel_package(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "kernels"))
    for name in tuple(sys.modules):
        if name == "rwkv7_kernels" or name.startswith("rwkv7_kernels."):
            sys.modules.pop(name)


class TinyQuantModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 8, bias=False)

    def forward(self, value):
        return self.proj(value)


class TinyA8W8Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 8, bias=False)
        self.lm_head = torch.nn.Linear(8, 16, bias=False)


def test_a8w8_int_mm_padding_matches_backend_tile_contract():
    a8w8 = importlib.import_module("rwkv7_kernels.nvidia.native_quant_a8w8")

    assert a8w8._int_mm_padded_rows(1, hip=False) == 32
    assert a8w8._int_mm_padded_rows(17, hip=False) == 32
    assert a8w8._int_mm_padded_rows(32, hip=False) == 32
    assert a8w8._int_mm_padded_rows(33, hip=False) == 64
    assert a8w8._int_mm_padded_rows(1, hip=True) == 17
    assert a8w8._int_mm_padded_rows(33, hip=True) == 33


def test_a8w8_tiled_w8a16_is_scoped_to_the_output_head():
    a8w8 = importlib.import_module("rwkv7_kernels.nvidia.native_quant_a8w8")
    model = TinyA8W8Model().eval()

    replaced = a8w8.quantize_model_a8w8(model, min_params=0, policy="memory")

    assert replaced == 2
    assert isinstance(model.proj, a8w8.A8W8Linear)
    assert isinstance(model.lm_head, a8w8.A8W8Linear)
    assert model.proj.tiled_w8a16 is False
    assert model.lm_head.tiled_w8a16 is True
    assert "small_rows=batched_w8a16" in repr(model.proj)
    assert "small_rows=tiled_w8a16" in repr(model.lm_head)


@pytest.mark.parametrize(
    ("method", "module_name"),
    [("native_w8", "MM8Linear"), ("native_w4", "MM4Linear")],
)
def test_native_quantization_is_structural_and_package_owned(method, module_name):
    quantization = importlib.import_module("rwkv7_kernels.quantization")
    torch.manual_seed(127)
    model = TinyQuantModel().eval()
    value = torch.randn(2, 3, 8)

    report = quantization.quantize_model(
        model,
        method,
        min_params=0,
        fused=False,
    )
    actual = model(value)

    assert type(model.proj).__name__ == module_name
    assert report["replaced_modules"] == 1
    assert report["graph_cache_invalidated"] is True
    assert torch.isfinite(actual).all()
    assert actual.shape == (2, 3, 8)
    assert quantization.quantization_report(model) == report
    decode_supported, decode_reason = quantization.quantization_graph_support(
        model, phase="decode"
    )
    prefill_supported, prefill_reason = quantization.quantization_graph_support(
        model, phase="prefill"
    )
    assert decode_supported and "stable graph outputs" in decode_reason
    assert prefill_supported and "stable graph outputs" in prefill_reason
    assert not any(name.startswith("_rwkv7_native_mm_") for name in vars(model))


def test_bitsandbytes_adapter_uses_standard_hf_configuration():
    quantization = importlib.import_module("rwkv7_kernels.quantization")
    config_type = type("Config", (), {"num_hidden_layers": 2})
    config = quantization.prepare_bitsandbytes_config(
        "bnb8",
        config=config_type(),
        policy="decode_rk",
        int8_threshold=5.5,
    )
    assert config.load_in_8bit
    assert not config.load_in_4bit
    assert config.llm_int8_threshold == 5.5
    assert "lm_head" in config.llm_int8_skip_modules
    assert "model.layers.0.attn.w_lora.lora.0" in config.llm_int8_skip_modules
    assert "model.layers.1.attn.r_proj" in config.llm_int8_skip_modules
    assert "model.layers.1.attn.k_proj" in config.llm_int8_skip_modules
    assert "model.layers.1.attn.v_proj" not in config.llm_int8_skip_modules


def test_bitsandbytes_adoption_rejects_an_unquantized_model():
    quantization = importlib.import_module("rwkv7_kernels.quantization")
    with pytest.raises(RuntimeError, match="prepare_bitsandbytes_config"):
        quantization.quantize_model(TinyQuantModel(), "bnb8")


def test_external_quantization_graphs_fail_closed_without_device_policy(monkeypatch):
    quantization = importlib.import_module("rwkv7_kernels.quantization")
    policy_type = type(
        "Policy",
        (),
        {
            "native_external_quant_graph": False,
            "native_external_quant_prefill_graph": False,
        },
    )
    kernel_policy = importlib.import_module("rwkv7_kernels.nvidia.kernel_policy")
    monkeypatch.setattr(kernel_policy, "current_kernel_policy", lambda **_: policy_type())
    model = TinyQuantModel().eval()
    quantization._REPORTS[model] = {"method": "bnb4"}

    for phase in ("prefill", "decode"):
        supported, reason = quantization.quantization_graph_support(
            model, phase=phase
        )
        assert not supported
        assert "no validated" in reason


def test_external_quantization_graph_override_is_explicit(monkeypatch):
    quantization = importlib.import_module("rwkv7_kernels.quantization")
    policy_type = type(
        "Policy",
        (),
        {
            "native_external_quant_graph": False,
            "native_external_quant_prefill_graph": False,
        },
    )
    kernel_policy = importlib.import_module("rwkv7_kernels.nvidia.kernel_policy")
    monkeypatch.setattr(kernel_policy, "current_kernel_policy", lambda **_: policy_type())
    monkeypatch.setenv("RWKV7_NATIVE_GRAPH_EXTERNAL_QUANT", "1")
    model = TinyQuantModel().eval()
    quantization._REPORTS[model] = {"method": "torchao_w8"}

    supported, reason = quantization.quantization_graph_support(
        model, phase="decode"
    )
    assert supported
    assert "exact-device" in reason
    with pytest.raises(ValueError, match="prefill or decode"):
        quantization.quantization_graph_support(model, phase="training")
