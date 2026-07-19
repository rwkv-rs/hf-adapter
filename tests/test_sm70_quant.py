from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from rwkv7_hf.sm70_quant import (
    SM70_W4_BN_TN_CHOICES,
    _cccl_include_paths,
    build_error,
    is_sm70,
    quantize_w4_groupwise,
    quantize_w4_row,
    quantize_w8_row,
    sm70_w4_bn_tn_config,
    sm70_w4_group_bn_tn_config,
    w4_groupwise_linear,
    w4_linear_add,
    w4_linear_relu2,
    w4_linear,
    w8_linear,
)


def test_cccl_include_path_discovery(monkeypatch, tmp_path) -> None:
    cccl = tmp_path / "cccl"
    (cccl / "nv").mkdir(parents=True)
    (cccl / "nv" / "target").write_text("", encoding="utf-8")
    monkeypatch.setenv("CUDA_CCCL_INCLUDE_PATH", str(cccl))
    assert str(cccl.resolve()) in _cccl_include_paths()


def test_row_quantized_layout_and_cpu_fallback() -> None:
    torch.manual_seed(70)
    weight = torch.randn(48, 32)
    x = torch.randn(3, 32)
    q8, s8 = quantize_w8_row(weight)
    q4, s4, inputs = quantize_w4_row(weight)
    assert q8.shape == weight.shape and q8.dtype == torch.int8
    assert q4.shape == (48, 16) and q4.dtype == torch.uint8
    assert s8.shape == s4.shape == (48,)
    assert inputs == 32
    ref = F.linear(x, weight)
    y8 = w8_linear(x, q8, s8)
    y4 = w4_linear(x, q4, s4, 48, inputs)
    assert F.cosine_similarity(y8.flatten(), ref.flatten(), dim=0) >= 0.9999
    assert F.cosine_similarity(y4.flatten(), ref.flatten(), dim=0) >= 0.99


def test_sm70_w4_bn_tn_config_is_fail_closed(monkeypatch) -> None:
    monkeypatch.delenv("RWKV7_SM70_W4_BN", raising=False)
    monkeypatch.delenv("RWKV7_SM70_W4_TN", raising=False)
    assert sm70_w4_bn_tn_config() == (8, 1)
    assert sm70_w4_bn_tn_config(4, 2048, 8192) == (4, 1)
    assert sm70_w4_bn_tn_config(4, 8192, 2048) == (8, 1)
    assert sm70_w4_bn_tn_config(4, 10240, 2560) == (16, 1)
    assert len(SM70_W4_BN_TN_CHOICES) == 13

    monkeypatch.setenv("RWKV7_SM70_W4_BN", "16")
    monkeypatch.setenv("RWKV7_SM70_W4_TN", "4")
    assert sm70_w4_bn_tn_config() == (16, 4)

    monkeypatch.setenv("RWKV7_SM70_W4_TN", "8")
    with pytest.raises(ValueError, match="unsupported sm70 W4 BN/TN pair"):
        sm70_w4_bn_tn_config()


def test_sm70_w4_group_bn_tn_config_is_independent(monkeypatch) -> None:
    monkeypatch.setenv("RWKV7_SM70_W4_BN", "4")
    monkeypatch.setenv("RWKV7_SM70_W4_TN", "1")
    monkeypatch.delenv("RWKV7_SM70_W4_GROUP_BN", raising=False)
    monkeypatch.delenv("RWKV7_SM70_W4_GROUP_TN", raising=False)
    assert sm70_w4_group_bn_tn_config() == (8, 1)
    assert sm70_w4_group_bn_tn_config(1, 4096, 65536) == (16, 1)
    assert sm70_w4_group_bn_tn_config(8, 2560, 65536, 256) == (32, 1)
    monkeypatch.setenv("RWKV7_SM70_W4_GROUP_BN", "16")
    monkeypatch.setenv("RWKV7_SM70_W4_GROUP_TN", "2")
    assert sm70_w4_group_bn_tn_config() == (16, 2)
    monkeypatch.setenv("RWKV7_SM70_W4_GROUP_TN", "8")
    with pytest.raises(ValueError, match="unsupported sm70 groupwise W4 BN/TN"):
        sm70_w4_group_bn_tn_config()


def test_groupwise_w4_layout_quality_and_cpu_fallback() -> None:
    torch.manual_seed(72)
    weight = torch.randn(24, 256)
    x = torch.randn(8, 256)
    packed, scales, inputs = quantize_w4_groupwise(weight, group_size=128)
    assert packed.shape == (24, 128)
    assert scales.shape == (24, 2)
    assert inputs == 256
    got = w4_groupwise_linear(
        x, packed, scales, 24, inputs, group_size=128
    )
    ref = F.linear(x, weight)
    row_packed, row_scales, _ = quantize_w4_row(weight)
    row = w4_linear(x, row_packed, row_scales, 24, inputs)
    group_cosine = F.cosine_similarity(got.flatten(), ref.flatten(), dim=0)
    row_cosine = F.cosine_similarity(row.flatten(), ref.flatten(), dim=0)
    assert group_cosine >= 0.99
    assert group_cosine > row_cosine


def test_groupwise_w4_rejects_unsupported_layouts() -> None:
    weight = torch.randn(4, 128)
    with pytest.raises(ValueError, match="group_size=128"):
        quantize_w4_groupwise(weight, group_size=64)
    with pytest.raises(ValueError, match="K divisible by group_size"):
        quantize_w4_groupwise(torch.randn(4, 192), group_size=128)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="exact sm_70 CUDA device required",
)
def test_sm70_dp4a_batch_reuse_and_forward_into() -> None:
    torch.manual_seed(71)
    x = torch.randn(8, 768, device="cuda", dtype=torch.float16) * 0.1
    weight = torch.randn(4096, 768, device="cuda", dtype=torch.float16) * 0.03
    q8, s8 = quantize_w8_row(weight)
    q4, s4, inputs = quantize_w4_row(weight)
    ref = F.linear(x, weight).float()
    out8 = torch.empty((8, 4096), device="cuda", dtype=torch.float16)
    out4 = torch.empty_like(out8)
    returned8 = w8_linear(x, q8, s8, out=out8)
    returned4 = w4_linear(x, q4, s4, 4096, inputs, out=out4)
    residual = torch.randn_like(out4)
    residual_before = residual.clone()
    expected_add = out4 + residual
    relu2 = w4_linear_relu2(x, q4, s4, 4096, inputs)
    added = w4_linear_add(x, q4, s4, residual, 4096, inputs)
    torch.cuda.synchronize()
    assert is_sm70(x.device) and build_error() is None
    assert returned8.data_ptr() == out8.data_ptr()
    assert returned4.data_ptr() == out4.data_ptr()
    torch.testing.assert_close(relu2, torch.relu(out4) ** 2, rtol=0.0, atol=0.0)
    torch.testing.assert_close(added, expected_add, rtol=0.0, atol=0.0)
    torch.testing.assert_close(residual, residual_before, rtol=0.0, atol=0.0)
    assert F.cosine_similarity(out8.float(), ref, dim=-1).min() >= 0.999
    lo = (q4 & 15).to(x.dtype) - 8
    hi = (q4 >> 4).to(x.dtype) - 8
    dequant4 = torch.empty_like(weight)
    dequant4[:, 0::2] = lo
    dequant4[:, 1::2] = hi
    dequant4.mul_(s4[:, None])
    kernel_reference4 = F.linear(x, dequant4).float()
    assert F.cosine_similarity(out4.float(), kernel_reference4, dim=-1).min() >= 0.9999
    assert F.cosine_similarity(out4.float(), ref, dim=-1).min() >= 0.989
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = w4_linear(x, q4, s4, 4096, inputs, out=out4)
    graph.replay()
    torch.cuda.synchronize()
    assert captured.data_ptr() == out4.data_ptr()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="exact sm_70 CUDA device required",
)
@pytest.mark.parametrize("group_size", (128, 256))
def test_sm70_groupwise_w4_matches_dequantized_oracle(group_size: int) -> None:
    torch.manual_seed(73)
    x = torch.randn(17, 2048, device="cuda", dtype=torch.float16) * 0.1
    weight = torch.randn(1024, 2048, device="cuda", dtype=torch.float16) * 0.03
    packed, scales, inputs = quantize_w4_groupwise(weight, group_size=group_size)
    out = torch.empty((17, 1024), device="cuda", dtype=torch.float16)
    got = w4_groupwise_linear(
        x,
        packed,
        scales,
        1024,
        inputs,
        group_size=group_size,
        out=out,
    )
    lo = (packed & 15).to(x.dtype) - 8
    hi = (packed >> 4).to(x.dtype) - 8
    dequant = torch.empty_like(weight)
    dequant[:, 0::2] = lo
    dequant[:, 1::2] = hi
    dequant.mul_(scales.repeat_interleave(group_size, dim=1))
    oracle = F.linear(x, dequant).float()
    torch.cuda.synchronize()
    assert got.data_ptr() == out.data_ptr()
    assert F.cosine_similarity(got.float(), oracle, dim=-1).min() >= 0.9999
