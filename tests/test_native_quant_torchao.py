from __future__ import annotations

import sys
import types

import pytest
import torch

from rwkv7_hf import native_quant_torchao as qao


class TinyModel(torch.nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.proj = torch.nn.Linear(8, 8, bias=False, dtype=dtype)
        self.small = torch.nn.Linear(2, 2, bias=False, dtype=dtype)
        self._rwkv7_native_graph_pack_cache = object()
        self._rwkv7_native_graph_runner_cache = object()


def fake_api(calls):
    def quantize_(module, config):
        calls.append((module, config))

    return quantize_, lambda: "w8", lambda group_size: ("w4", group_size)


def test_torchao_017_config_api_fallback(monkeypatch) -> None:
    torchao = types.ModuleType("torchao")
    quantization = types.ModuleType("torchao.quantization")
    quantize_module = types.ModuleType("torchao.quantization.quantize_")
    workflows = types.ModuleType("torchao.quantization.quantize_.workflows")
    int4_workflow = types.ModuleType("torchao.quantization.quantize_.workflows.int4")
    packing = types.ModuleType(
        "torchao.quantization.quantize_.workflows.int4.int4_packing_format"
    )

    class Int4WeightOnlyConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Int8WeightOnlyConfig:
        pass

    class Int4PackingFormat:
        TILE_PACKED_TO_4D = "tile-4d"

    sentinel_quantize = object()
    quantization.Int4WeightOnlyConfig = Int4WeightOnlyConfig
    quantization.Int8WeightOnlyConfig = Int8WeightOnlyConfig
    quantization.quantize_ = sentinel_quantize
    packing.Int4PackingFormat = Int4PackingFormat
    for name, module in (
        ("torchao", torchao),
        ("torchao.quantization", quantization),
        ("torchao.quantization.quantize_", quantize_module),
        ("torchao.quantization.quantize_.workflows", workflows),
        ("torchao.quantization.quantize_.workflows.int4", int4_workflow),
        (
            "torchao.quantization.quantize_.workflows.int4.int4_packing_format",
            packing,
        ),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    quantize_, int8_weight_only, int4_weight_only = qao._torchao_api()
    config = int4_weight_only(group_size=64)
    assert quantize_ is sentinel_quantize
    assert isinstance(int8_weight_only(), Int8WeightOnlyConfig)
    assert config.kwargs == {
        "group_size": 64,
        "int4_packing_format": "tile-4d",
        "version": 2,
    }


def test_torchao_w8_selection_and_cache_invalidation(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(qao, "_torchao_api", lambda: fake_api(calls))
    model = TinyModel()
    replaced = qao.quantize_model_torchao_w8(model, min_params=8, policy="memory")
    assert replaced == 1
    assert calls == [(model.proj, "w8")]
    assert model._rwkv7_native_mm_quantization == "torchao_w8"
    assert not hasattr(model, "_rwkv7_native_graph_pack_cache")
    assert not hasattr(model, "_rwkv7_native_graph_runner_cache")


def test_torchao_w4_requires_bf16(monkeypatch) -> None:
    monkeypatch.setattr(qao, "_torchao_api", lambda: fake_api([]))
    with pytest.raises(ValueError, match="requires a bf16 model"):
        qao.quantize_model_torchao_w4(TinyModel(torch.float16), min_params=8)


def test_torchao_w4_forwards_group_size(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(qao, "_torchao_api", lambda: fake_api(calls))
    model = TinyModel(torch.bfloat16)
    assert qao.quantize_model_torchao_w4(model, min_params=8, group_size=64) == 1
    assert calls == [(model.proj, ("w4", 64))]


def test_torchao_w4_fp16_speed_policy_wraps_head(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(qao, "_torchao_api", lambda: fake_api(calls))
    model = TinyModel(torch.float16)
    model.lm_head = torch.nn.Linear(8, 8, bias=False, dtype=torch.float16)

    replaced = qao.quantize_model_torchao_w4(model, min_params=8, policy="speed")

    assert replaced == 1
    assert isinstance(model.lm_head, qao.TorchAOW4FP16Linear)
    assert model.lm_head.inner.weight.dtype == torch.bfloat16
    assert model._rwkv7_native_mm_quantization == "torchao_w4_fp16_head"
    x = torch.randn(2, 8, dtype=torch.float16)
    assert model.lm_head(x).dtype == torch.float16


def test_torchao_w4_5090_speed_shape_gate_is_exact() -> None:
    gate = qao._torchao_w4_5090_speed_shape_supported
    enabled = ((16384, 4096), (4096, 16384))
    assert gate(
        "model.layers.0.ffn.key",
        (16384, 4096),
        torch.bfloat16,
        (12, 0),
        enabled,
    )
    assert gate(
        "model.layers.0.ffn.value",
        (4096, 16384),
        torch.bfloat16,
        (12, 0),
        enabled,
    )
    assert not gate(
        "model.layers.0.attn.r_proj",
        (4096, 4096),
        torch.bfloat16,
        (12, 0),
        enabled,
    )
    assert not gate(
        "model.layers.0.ffn.key",
        (16384, 4096),
        torch.float16,
        (12, 0),
        enabled,
    )
    assert not gate(
        "model.layers.0.ffn.key",
        (16384, 4096),
        torch.bfloat16,
        (8, 9),
        enabled,
    )
    assert not gate(
        "model.layers.0.ffn.key",
        (16384, 4096),
        torch.bfloat16,
        (12, 0),
        (),
    )


def test_torchao_w4_speed_policy_adds_only_measured_5090_ffn(monkeypatch) -> None:
    class FakeMarlinW4Linear(torch.nn.Module):
        def __init__(self, inner, *, group_size):
            super().__init__()
            self.inner = inner
            self.group_size = group_size

    class FFN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.key = torch.nn.Linear(4, 16, bias=False, dtype=torch.bfloat16)
            self.value = torch.nn.Linear(16, 4, bias=False, dtype=torch.bfloat16)

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.ffn = FFN()
            self.other = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([Layer()])
            self.lm_head = torch.nn.Linear(4, 32, bias=False, dtype=torch.bfloat16)

    calls = []
    monkeypatch.setattr(qao, "_torchao_api", lambda: fake_api(calls))
    fake_marlin = types.ModuleType("rwkv7_hf.native_quant_marlin")
    fake_marlin.MarlinW4Linear = FakeMarlinW4Linear
    monkeypatch.setitem(sys.modules, "rwkv7_hf.native_quant_marlin", fake_marlin)
    monkeypatch.setattr(
        qao,
        "_torchao_w4_5090_speed_module",
        lambda name, _module: name.endswith(".ffn.key") or name.endswith(".ffn.value"),
    )
    model = Model()
    replaced = qao.quantize_model_torchao_w4(model, min_params=1, policy="speed")

    assert replaced == 3
    assert len(calls) == 1
    assert calls[0][0] is model.lm_head
    assert isinstance(model.model.layers[0].ffn.key, FakeMarlinW4Linear)
    assert isinstance(model.model.layers[0].ffn.value, FakeMarlinW4Linear)
    assert model._rwkv7_native_mm_quantization == "marlin_w4_5090_hybrid"
    assert model._rwkv7_native_mm_exact_5090_speed_modules == 2
