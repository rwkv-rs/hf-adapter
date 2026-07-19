from __future__ import annotations

import torch

from rwkv7_hf.native_quant_mm4 import MM4Linear, _mm4_batched_dot_device_supported
from rwkv7_hf.native_quant_mm8 import (
    MM8Linear,
    _mm8_batched_gemv_max_rows_for_capability,
    _mm8_prefill_dequant_tile,
)
from rwkv7_hf.native_quant_policy import should_quantize_linear


def test_mm4_batched_dot_exact_device_policy() -> None:
    assert _mm4_batched_dot_device_supported(8, 6, "NVIDIA GeForce RTX 3090")
    assert _mm4_batched_dot_device_supported(8, 9, "NVIDIA GeForce RTX 4090")
    assert _mm4_batched_dot_device_supported(12, 0, "NVIDIA GeForce RTX 5090")
    assert not _mm4_batched_dot_device_supported(8, 9, "NVIDIA GeForce RTX 4070")
    assert not _mm4_batched_dot_device_supported(8, 0, "NVIDIA A100")


def test_quant_prefill_workspace_is_exact_sm70_only() -> None:
    dense = torch.nn.Linear(16, 32, bias=False)
    mm4 = MM4Linear(dense, fused=False)
    mm8 = MM8Linear(dense, fused=False)

    assert mm4.rwkv7_prefill_dequant_shape(128) is None
    assert mm8.rwkv7_prefill_dequant_shape(128) is None


def test_mm4_cpu_pack_can_target_sm70_deployment(monkeypatch) -> None:
    monkeypatch.setenv("RWKV7_SM70_TARGET_PACK", "1")
    dense = torch.nn.Linear(16, 32, bias=False, dtype=torch.float16)
    mm4 = MM4Linear(dense, fused=False)
    assert mm4.sm70_rowwise
    assert hasattr(mm4, "packed_row")
    assert hasattr(mm4, "row_scale")


def test_mm8_prefill_dequant_tile_tracks_ffn_orientation(monkeypatch) -> None:
    for name in (
        "RWKV7_SM70_MM8_DEQUANT_BLOCK_N",
        "RWKV7_SM70_MM8_DEQUANT_BLOCK_M",
        "RWKV7_SM70_MM8_DEQUANT_WARPS",
    ):
        monkeypatch.delenv(name, raising=False)
    assert _mm8_prefill_dequant_tile(2048, 8192) == (32, 64, 4)
    assert _mm8_prefill_dequant_tile(8192, 2048) == (32, 16, 4)
    assert _mm8_prefill_dequant_tile(2048, 2048) == (32, 32, 4)

    monkeypatch.setenv("RWKV7_SM70_MM8_DEQUANT_BLOCK_N", "64")
    monkeypatch.setenv("RWKV7_SM70_MM8_DEQUANT_BLOCK_M", "32")
    monkeypatch.setenv("RWKV7_SM70_MM8_DEQUANT_WARPS", "8")
    assert _mm8_prefill_dequant_tile(2048, 8192) == (64, 32, 8)


def test_mm8_batched_gemv_is_promoted_only_for_exact_sm70() -> None:
    assert _mm8_batched_gemv_max_rows_for_capability(7, 0) == 16
    assert _mm8_batched_gemv_max_rows_for_capability(7, 5) == 0
    assert _mm8_batched_gemv_max_rows_for_capability(8, 0) == 0
    assert _mm8_batched_gemv_max_rows_for_capability(12, 0) == 0


def test_balanced_policy_selects_head_and_configured_leading_ffn(monkeypatch) -> None:
    monkeypatch.delenv("RWKV7_NATIVE_MM_BALANCED_FFN_LAYERS", raising=False)
    assert should_quantize_linear("lm_head", 10, min_params=1, policy="balanced")
    assert should_quantize_linear(
        "model.layers.0.ffn.key", 10, min_params=1, policy="balanced"
    )
    assert should_quantize_linear(
        "model.layers.0.ffn.value", 10, min_params=1, policy="balanced"
    )
    assert not should_quantize_linear(
        "model.layers.1.ffn.key", 10, min_params=1, policy="balanced"
    )
    monkeypatch.setenv("RWKV7_NATIVE_MM_BALANCED_FFN_LAYERS", "2")
    assert should_quantize_linear(
        "model.layers.1.ffn.value", 10, min_params=1, policy="balanced"
    )
    assert not should_quantize_linear(
        "model.layers.2.ffn.value", 10, min_params=1, policy="balanced"
    )
