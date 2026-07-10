#!/usr/bin/env python3
from __future__ import annotations

import torch
import pytest

from rwkv7_hf.ada_lora import ada_wagv_lora, ada_wagv_lora_available, ada_wagv_lora_should_use


def test_shape_policy() -> None:
    assert ada_wagv_lora_should_use(1, 1024, 64)
    assert ada_wagv_lora_should_use(4, 4096, 512)
    assert not ada_wagv_lora_should_use(8, 1024, 64)
    assert not ada_wagv_lora_should_use(1, 768, 64)


def test_cpu_fallback_shapes_and_values() -> None:
    torch.manual_seed(7)
    rows, hidden = 2, 32
    ranks = (8, 6, 4, 5)
    x = [torch.randn(rows, hidden) for _ in range(4)]
    down = [torch.randn(rank, hidden) for rank in ranks]
    up = [torch.randn(hidden, rank) for rank in ranks]
    w0, a0, v0 = (torch.randn(hidden) for _ in range(3))
    v = torch.randn(rows, hidden)
    v_first = torch.randn(rows, hidden)
    outputs = ada_wagv_lora(
        *x, *down, *up, w0, a0, v0, v, v_first, force_fallback=True
    )
    assert len(outputs) == 4
    assert all(tuple(item.shape) == (rows, hidden) for item in outputs)
    assert all(torch.isfinite(item).all() for item in outputs)


@pytest.mark.parametrize("dtype,max_abs", [(torch.float16, 0.02), (torch.bfloat16, 0.03)])
def test_ada_cuda_matches_fallback_for_fp16_and_bf16(dtype, max_abs) -> None:
    if not torch.cuda.is_available() or not ada_wagv_lora_available("cuda"):
        pytest.skip("Ada sm_89 CUDA kernel is unavailable")
    torch.manual_seed(11)
    rows, hidden = 1, 1024
    ranks = (64, 64, 128, 64)
    x = [torch.randn(rows, hidden, device="cuda", dtype=dtype) for _ in range(4)]
    down = [torch.randn(rank, hidden, device="cuda", dtype=dtype) * 0.02 for rank in ranks]
    up = [torch.randn(hidden, rank, device="cuda", dtype=dtype) * 0.02 for rank in ranks]
    w0, a0, v0 = (torch.randn(hidden, device="cuda", dtype=dtype) * 0.02 for _ in range(3))
    v = torch.randn(rows, hidden, device="cuda", dtype=dtype)
    v_first = torch.randn(rows, hidden, device="cuda", dtype=dtype)
    with torch.inference_mode():
        reference = ada_wagv_lora(
            *x, *down, *up, w0, a0, v0, v, v_first, force_fallback=True
        )
        actual = ada_wagv_lora(*x, *down, *up, w0, a0, v0, v, v_first)
    for expected, observed in zip(reference, actual):
        assert torch.allclose(expected.float(), observed.float(), atol=max_abs, rtol=0.01)
        cosine = torch.nn.functional.cosine_similarity(
            expected.float().flatten().unsqueeze(0),
            observed.float().flatten().unsqueeze(0),
        ).item()
        assert cosine >= 0.9999
