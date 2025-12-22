#!/usr/bin/env python3
# coding=utf-8
"""CPU-only regression tests for the T4 native quantization patch."""
from __future__ import annotations

import pytest
import torch

import rwkv7_hf.native_quant_mm4 as native_quant_mm4
import rwkv7_hf.sm70_quant as sm70_quant
from rwkv7_hf.native_quant_mm4 import quantize_model_mm4
from rwkv7_hf.native_quant_mm8 import quantize_model_mm8


def test_mm4_group128_nondivisible_k_falls_back_safely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        native_quant_mm4,
        "is_sm7x_quant_device",
        lambda _device=None: True,
    )
    linear = torch.nn.Linear(192, 24, bias=False).eval()
    quantized = native_quant_mm4.MM4Linear(
        linear,
        fused=False,
        group_size=128,
    ).eval()

    assert quantized.group_size == 128
    assert not quantized.groupwise
    assert quantized.sm70_rowwise
    assert hasattr(quantized, "packed_row")
    assert not hasattr(quantized, "packed_group")

    inputs = torch.randn(3, 192)
    output = quantized(inputs)
    assert output.shape == (3, 24)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize(
    ("quantize", "kwargs", "expected_type"),
    [
        (quantize_model_mm8, {}, "MM8Linear"),
        (quantize_model_mm4, {"group_size": 0}, "MM4Linear"),
    ],
)
def test_native_quantization_invalidates_graph_and_jit_caches(
    quantize, kwargs, expected_type: str
) -> None:
    model = torch.nn.Sequential(torch.nn.Linear(16, 16, bias=False))
    cache_attrs = (
        "_rwkv7_native_jit_pack_cache",
        "_rwkv7_native_model_jit_pack_cache",
        "_rwkv7_native_graph_pack_cache",
        "_rwkv7_native_graph_runner_cache",
        "_rwkv7_native_prefill_graph_runner_cache",
        "_rwkv7_native_prefill_graph_hot_runner",
    )
    for name in cache_attrs:
        setattr(model, name, object())

    replaced = quantize(
        model,
        min_params=1,
        fused=False,
        policy="memory",
        **kwargs,
    )

    assert replaced == 1
    assert type(model[0]).__name__ == expected_type
    assert all(not hasattr(model, name) for name in cache_attrs)


def test_sm75_quantization_requires_t4_name_and_rejects_rtx_2080(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sm70_quant.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sm70_quant.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        sm70_quant.torch.cuda,
        "get_device_capability",
        lambda _index: (7, 5),
    )
    monkeypatch.setattr(
        sm70_quant.torch.cuda,
        "get_device_name",
        lambda _index: "GeForce RTX 2080 Ti",
    )
    assert not sm70_quant.is_sm7x_quant_device()

    monkeypatch.setattr(
        sm70_quant.torch.cuda,
        "get_device_name",
        lambda _index: "NVIDIA T4",
    )
    assert sm70_quant.is_sm7x_quant_device()
