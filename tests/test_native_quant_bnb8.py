from __future__ import annotations

import pytest
import torch

from rwkv7_hf.native_quant_bnb8 import (
    fused_bnb8_attn_sequence_mix_quant,
    fused_bnb8_ffn_sequence_mix_quant,
    fused_bnb8_relu_square_quant,
    fused_bnb8_relu_square_quant_available,
)
from rwkv7_hf.fused_time_mix import fused_attn_sequence_shift_mix
from rwkv7_hf.fused_time_mix import fused_ffn_sequence_shift_mix


def test_cpu_runtime_reports_optional_fusion_availability() -> None:
    assert isinstance(fused_bnb8_relu_square_quant_available(), bool)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_relu_square_quant_matches_bitsandbytes() -> None:
    pytest.importorskip("bitsandbytes")
    torch.manual_seed(17)
    x = torch.randn(8, 4096, device="cuda", dtype=torch.float16)
    reference_input = torch.relu(x) ** 2
    expected_q, expected_scale, _ = torch.ops.bitsandbytes.int8_vectorwise_quant.default(
        reference_input,
        0.0,
    )
    actual_q, actual_scale = fused_bnb8_relu_square_quant(x)
    assert torch.equal(actual_q, expected_q)
    assert torch.equal(actual_scale, expected_scale)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_relu_square_quant_supports_rank_three() -> None:
    pytest.importorskip("bitsandbytes")
    torch.manual_seed(19)
    x = torch.randn(2, 4, 1024, device="cuda", dtype=torch.float16)
    expected_q, expected_scale, _ = torch.ops.bitsandbytes.int8_vectorwise_quant.default(
        (torch.relu(x) ** 2).reshape(-1, x.shape[-1]),
        0.0,
    )
    actual_q, actual_scale = fused_bnb8_relu_square_quant(x)
    assert actual_q.shape == expected_q.shape
    assert torch.equal(actual_q, expected_q)
    assert torch.equal(actual_scale, expected_scale)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_rkv_mix_quant_matches_materialized_bitsandbytes_path() -> None:
    pytest.importorskip("bitsandbytes")
    torch.manual_seed(23)
    batch, tokens, hidden = 2, 8, 1024
    x = torch.randn(batch, tokens, hidden, device="cuda", dtype=torch.float16)
    initial = torch.randn(batch, hidden, device="cuda", dtype=torch.float16)
    mixes = [torch.rand(hidden, device="cuda", dtype=torch.float16) for _ in range(6)]
    reference = fused_attn_sequence_shift_mix(x, initial, *mixes)
    actual = fused_bnb8_attn_sequence_mix_quant(x, initial, *mixes)

    # W/V/A/G and the recurrent shift state are the streams that remain
    # materialized for the low-rank branches.
    for candidate, expected in zip(actual[6:11], (reference[1], reference[3], reference[4], reference[5], reference[6])):
        assert torch.equal(candidate, expected)
    for index, expected in enumerate((reference[0], reference[2], reference[3])):
        expected_q, expected_scale, _ = torch.ops.bitsandbytes.int8_vectorwise_quant.default(
            expected.reshape(-1, hidden),
            0.0,
        )
        assert torch.equal(actual[2 * index], expected_q)
        assert torch.equal(actual[2 * index + 1], expected_scale)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_ffn_mix_quant_matches_materialized_bitsandbytes_path() -> None:
    pytest.importorskip("bitsandbytes")
    torch.manual_seed(29)
    batch, tokens, hidden = 2, 8, 1024
    x = torch.randn(batch, tokens, hidden, device="cuda", dtype=torch.float16)
    initial = torch.randn(batch, hidden, device="cuda", dtype=torch.float16)
    mix = torch.rand(hidden, device="cuda", dtype=torch.float16)
    mixed, expected_next = fused_ffn_sequence_shift_mix(x, initial, mix)
    expected_q, expected_scale, _ = torch.ops.bitsandbytes.int8_vectorwise_quant.default(
        mixed.reshape(-1, hidden),
        0.0,
    )
    actual_q, actual_scale, actual_next = fused_bnb8_ffn_sequence_mix_quant(x, initial, mix)
    assert torch.equal(actual_q, expected_q)
    assert torch.equal(actual_scale, expected_scale)
    assert torch.equal(actual_next, expected_next)
