from __future__ import annotations

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
