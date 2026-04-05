#!/usr/bin/env python3
from __future__ import annotations

import torch
import pytest

from rwkv7_hf.ada_lora import (
    ada_wag_lora,
    ada_wagv_bmm,
    ada_wagv_bmm_should_use,
    ada_wagv_lora,
    ada_wagv_lora_available,
    ada_wagv_lora_should_use,
)


def test_shape_policy() -> None:
    assert ada_wagv_lora_should_use(1, 1024, 64)
    assert ada_wagv_lora_should_use(4, 4096, 512)
    assert ada_wagv_lora_should_use(8, 1024, 64)
    assert not ada_wagv_lora_should_use(9, 1024, 64)
    assert not ada_wagv_lora_should_use(1, 768, 64)
    assert ada_wagv_bmm_should_use(8, 1024, 128)
    assert not ada_wagv_bmm_should_use(4, 1024, 128)
    assert not ada_wagv_bmm_should_use(8, 768, 128)


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


def test_cpu_fallback_can_fuse_a_sigmoid_and_skip_v() -> None:
    torch.manual_seed(8)
    rows, hidden = 2, 32
    ranks = (8, 6, 4, 5)
    x = [torch.randn(rows, hidden) for _ in range(4)]
    down = [torch.randn(rank, hidden) for rank in ranks]
    up = [torch.randn(hidden, rank) for rank in ranks]
    w0, a0, v0 = (torch.randn(hidden) for _ in range(3))
    v = torch.randn(rows, hidden)
    v_first = torch.randn(rows, hidden)
    reference = ada_wagv_lora(
        *x, *down, *up, w0, a0, v0, v, v_first, force_fallback=True
    )
    fused = ada_wagv_lora(
        *x, *down, *up, w0, a0, v0, v, v_first,
        sigmoid_a=True, compute_v=False, force_fallback=True,
    )
    torch.testing.assert_close(fused[0], reference[0])
    torch.testing.assert_close(fused[1], torch.sigmoid(reference[1]))
    torch.testing.assert_close(fused[2], reference[2])
    torch.testing.assert_close(fused[3], v)


def test_wag_only_cpu_fallback_matches_independent_linears() -> None:
    torch.manual_seed(9)
    rows, hidden = 8, 32
    ranks = (8, 6, 4)
    xw, xa, xg = (torch.randn(rows, hidden) for _ in range(3))
    w1, a1, g1 = (torch.randn(rank, hidden) for rank in ranks)
    w2, a2, g2 = (torch.randn(hidden, rank) for rank in ranks)
    w0, a0 = (torch.randn(hidden) for _ in range(2))
    actual = ada_wag_lora(
        xw, xa, xg, w1, a1, g1, w2, a2, g2, w0, a0,
        force_fallback=True,
    )
    expected = (
        torch.nn.functional.linear(torch.tanh(torch.nn.functional.linear(xw, w1)), w2, w0),
        torch.nn.functional.linear(torch.nn.functional.linear(xa, a1), a2, a0),
        torch.nn.functional.linear(torch.sigmoid(torch.nn.functional.linear(xg, g1)), g2),
    )
    for observed, reference in zip(actual, expected):
        torch.testing.assert_close(observed, reference)


def test_bmm_cpu_fallback_matches_existing_grouped_path() -> None:
    torch.manual_seed(10)
    rows, hidden = 8, 32
    ranks = (8, 6, 4, 5)
    x = [torch.randn(rows, hidden) for _ in range(4)]
    down = [torch.randn(rank, hidden) for rank in ranks]
    up = [torch.randn(hidden, rank) for rank in ranks]
    w0, a0, v0 = (torch.randn(hidden) for _ in range(3))
    v = torch.randn(rows, hidden)
    v_first = torch.randn(rows, hidden)
    expected = ada_wagv_lora(
        *x, *down, *up, w0, a0, v0, v, v_first,
        sigmoid_a=True, force_fallback=True,
    )
    actual = ada_wagv_bmm(
        *x, *down, *up, w0, a0, v0, v, v_first,
        sigmoid_a=True,
    )
    for observed, reference in zip(actual, expected):
        torch.testing.assert_close(observed, reference)


def test_ada_b8_bmm_cuda_matches_fallback() -> None:
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (8, 9)
    ):
        pytest.skip("exact sm_89 B8 tensor-core route is unavailable")
    torch.manual_seed(12)
    rows, hidden = 8, 1024
    ranks = (64, 64, 128, 32)
    x = [
        torch.randn(rows, hidden, device="cuda", dtype=torch.float16)
        for _ in range(4)
    ]
    wav = torch.stack((x[0], x[1], x[3]))
    x = [wav[0], wav[1], x[2], wav[2]]
    down = [
        torch.randn(rank, hidden, device="cuda", dtype=torch.float16) * 0.02
        for rank in ranks
    ]
    up = [
        torch.randn(hidden, rank, device="cuda", dtype=torch.float16) * 0.02
        for rank in ranks
    ]
    w0, a0, v0 = (
        torch.randn(hidden, device="cuda", dtype=torch.float16) * 0.02
        for _ in range(3)
    )
    v = torch.randn(rows, hidden, device="cuda", dtype=torch.float16)
    v_first = torch.randn(rows, hidden, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        expected = ada_wagv_lora(
            *x, *down, *up, w0, a0, v0, v, v_first,
            sigmoid_a=True, force_fallback=True,
        )
        actual = ada_wagv_bmm(
            *x, *down, *up, w0, a0, v0, v, v_first,
            sigmoid_a=True,
        )
        cached = getattr(down[0], "_rwkv7_ada_wagv_bmm_pack", None)
        repeated = ada_wagv_bmm(
            *x, *down, *up, w0, a0, v0, v, v_first,
            sigmoid_a=True,
        )
        expected_no_v = ada_wagv_lora(
            x[0], x[1], x[2], x[3],
            down[0], down[1], down[2], down[2],
            up[0], up[1], up[2], up[2],
            w0, a0, a0, v, v,
            sigmoid_a=True, compute_v=False, force_fallback=True,
        )
        actual_no_v = ada_wagv_bmm(
            x[0], x[1], x[2], x[3],
            down[0], down[1], down[2], down[2],
            up[0], up[1], up[2], up[2],
            w0, a0, a0, v, v,
            sigmoid_a=True, compute_v=False,
        )
    assert isinstance(cached, tuple) and len(cached) == 4
    assert getattr(down[0], "_rwkv7_ada_wagv_bmm_pack", None) is cached
    for reference, observed, second in zip(expected, actual, repeated):
        assert torch.allclose(
            reference.float(), observed.float(), atol=0.03, rtol=0.01
        )
        torch.testing.assert_close(observed, second)
    assert isinstance(getattr(down[0], "_rwkv7_ada_wa_bmm_pack", None), tuple)
    for reference, observed in zip(expected_no_v, actual_no_v):
        assert torch.allclose(
            reference.float(), observed.float(), atol=0.03, rtol=0.01
        )


@pytest.mark.parametrize("dtype,max_abs", [(torch.float16, 0.02), (torch.bfloat16, 0.03)])
def test_ada_cuda_matches_fallback_for_fp16_and_bf16(dtype, max_abs) -> None:
    if not torch.cuda.is_available() or not ada_wagv_lora_available("cuda"):
        pytest.skip("sm_89/sm_120 small-row CUDA kernel is unavailable")
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
