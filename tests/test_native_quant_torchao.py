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
