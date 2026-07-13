import pytest
import torch
import torch.nn.functional as F

from rwkv7_hf.native_quant_mm4 import dequantize_mm4, quantize_mm4
from rwkv7_hf.native_quant_mm4_groupwise import (
    MM4GroupwiseLinear,
    dequantize_groupwise_mm4,
    groupwise_mm4_gemv_available,
    groupwise_mm4_matmul,
    groupwise_mm4_storage_bytes,
    quantize_groupwise_mm4,
    quantize_model_mm4_groupwise,
)


def test_groupwise_mm4_roundtrip_and_matmul_oracle():
    torch.manual_seed(7)
    weight = torch.randn(70, 18, dtype=torch.float32)
    values = quantize_groupwise_mm4(weight, group_size=32)
    packed, scales, biases, k_orig, n_orig, _, _, group_size = values
    dense = dequantize_groupwise_mm4(
        packed, scales, biases, k_orig, n_orig, group_size
    )
    assert dense.shape == weight.shape
    x = torch.randn(3, weight.shape[0])
    torch.testing.assert_close(
        groupwise_mm4_matmul(
            x, packed, scales, biases, k_orig, n_orig, group_size
        ),
        x @ dense,
    )
    assert groupwise_mm4_storage_bytes(packed, scales, biases) < weight.numel() * 4


def test_groupwise_mm4_reduces_local_range_error():
    torch.manual_seed(11)
    chunks = []
    for scale in (0.01, 0.1, 1.0, 10.0):
        chunks.append(torch.randn(32, 64) * scale)
    weight = torch.cat(chunks, dim=0)
    current = quantize_mm4(weight)
    current_dense = dequantize_mm4(*current[:5], current[5], out_dtype=weight.dtype)
    grouped = quantize_groupwise_mm4(weight, group_size=32)
    grouped_dense = dequantize_groupwise_mm4(
        grouped[0], grouped[1], grouped[2], grouped[3], grouped[4], grouped[7]
    )
    current_mse = F.mse_loss(current_dense, weight)
    grouped_mse = F.mse_loss(grouped_dense, weight)
    assert grouped_mse < current_mse


def test_groupwise_linear_and_model_replacement():
    torch.manual_seed(13)
    linear = torch.nn.Linear(64, 32, bias=True)
    quant = MM4GroupwiseLinear(linear, group_size=32)
    x = torch.randn(2, 64)
    expected = groupwise_mm4_matmul(
        x,
        quant.packed,
        quant.scales,
        quant.biases,
        quant.k_orig,
        quant.n_orig,
        quant.group_size,
    ) + linear.bias
    torch.testing.assert_close(quant(x), expected)

    model = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.ReLU())
    replaced = quantize_model_mm4_groupwise(model, min_params=0, group_size=32)
    assert replaced == 1
    assert isinstance(model[0], MM4GroupwiseLinear)
    assert model._rwkv7_native_mm_quantization == "mm4_groupwise"


@pytest.mark.skipif(
    not torch.cuda.is_available() or not groupwise_mm4_gemv_available("cuda"),
    reason="requires CUDA and Triton",
)
@pytest.mark.parametrize("group_size", [32, 128])
def test_groupwise_fused_cuda_matches_dequantized_oracle(group_size):
    torch.manual_seed(20260713)
    linear = torch.nn.Linear(
        256, 384, bias=False, device="cuda", dtype=torch.float16
    )
    module = MM4GroupwiseLinear(linear, group_size=group_size, fused=True)
    dense = dequantize_groupwise_mm4(
        module.packed,
        module.scales,
        module.biases,
        module.k_orig,
        module.n_orig,
        module.group_size,
        out_dtype=torch.float16,
    )
    for rows in (1, 4):
        x = torch.randn(rows, 256, device="cuda", dtype=torch.float16)
        expected = x @ dense
        actual = module(x)
        cosine = F.cosine_similarity(
            expected.float().flatten(), actual.float().flatten(), dim=0
        ).item()
        assert cosine >= 0.99999
        assert torch.allclose(actual, expected, rtol=2e-2, atol=2e-2)
        assert torch.allclose(
            module.rwkv7_forward_relu2(x),
            torch.relu(expected) ** 2,
            rtol=3e-2,
            atol=3e-2,
        )
        residual = torch.randn_like(expected)
        assert torch.allclose(
            module.rwkv7_forward_add(x, residual),
            expected + residual,
            rtol=2e-2,
            atol=2e-2,
        )
