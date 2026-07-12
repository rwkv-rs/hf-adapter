from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from rwkv7_hf.sm70_quant import (
    build_error,
    is_sm70,
    quantize_w4_row,
    quantize_w8_row,
    w4_linear,
    w4_linear_relu2,
    w8_linear,
)


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
    y4_relu2 = w4_linear_relu2(x, q4, s4, 48, inputs)
    assert F.cosine_similarity(y8.flatten(), ref.flatten(), dim=0) >= 0.9999
    assert F.cosine_similarity(y4.flatten(), ref.flatten(), dim=0) >= 0.99
    torch.testing.assert_close(y4_relu2, torch.relu(y4) ** 2, rtol=0, atol=0)


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
    fused_relu2 = w4_linear_relu2(x, q4, s4, 4096, inputs)
    torch.cuda.synchronize()
    assert is_sm70(x.device) and build_error() is None
    assert returned8.data_ptr() == out8.data_ptr()
    assert returned4.data_ptr() == out4.data_ptr()
    assert F.cosine_similarity(out8.float(), ref, dim=-1).min() >= 0.999
    lo = (q4 & 15).to(x.dtype) - 8
    hi = (q4 >> 4).to(x.dtype) - 8
    dequant4 = torch.empty_like(weight)
    dequant4[:, 0::2] = lo
    dequant4[:, 1::2] = hi
    dequant4.mul_(s4[:, None])
    kernel_reference4 = F.linear(x, dequant4).float()
    assert F.cosine_similarity(out4.float(), kernel_reference4, dim=-1).min() >= 0.9999
    assert F.cosine_similarity(
        fused_relu2.float(), (torch.relu(out4) ** 2).float(), dim=-1,
    ).min() >= 0.9999
    assert F.cosine_similarity(out4.float(), ref, dim=-1).min() >= 0.989
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = w4_linear(x, q4, s4, 4096, inputs, out=out4)
    graph.replay()
    torch.cuda.synchronize()
    assert captured.data_ptr() == out4.data_ptr()
