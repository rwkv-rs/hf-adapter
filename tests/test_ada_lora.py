#!/usr/bin/env python3
from __future__ import annotations

import torch
import pytest

from rwkv7_hf.ada_lora import (
    _ada_wagv_bmm_pack,
    _stack_sm120_wagv_inputs,
    ada_wag_lora,
    ada_wagv_bmm,
    ada_wagv_bmm_available,
    ada_wagv_bmm_should_use,
    ada_wagv_lora,
    ada_wagv_lora_available,
    ada_wagv_lora_should_use,
    sm120_wagv_bmm_g_available,
    sm120_wagv_bmm_g_should_use,
)


def test_shape_policy() -> None:
    assert ada_wagv_lora_should_use(1, 1024, 64)
    assert ada_wagv_lora_should_use(4, 4096, 512)
    assert ada_wagv_lora_should_use(8, 1024, 64)
    assert not ada_wagv_lora_should_use(9, 1024, 64)
    assert not ada_wagv_lora_should_use(1, 768, 64)
    assert ada_wagv_bmm_should_use(8, 1024, 128)
    assert ada_wagv_bmm_should_use(8, 2560, 480)
    assert not ada_wagv_bmm_should_use(4, 1024, 128)
    assert not ada_wagv_bmm_should_use(8, 768, 128)
    assert not ada_wagv_bmm_should_use(8, 4096, 480)
    assert sm120_wagv_bmm_g_should_use(8, 1024, 128)
    assert sm120_wagv_bmm_g_should_use(8, 2048, 512)
    assert not sm120_wagv_bmm_g_should_use(4, 1024, 128)
    assert not sm120_wagv_bmm_g_should_use(8, 2560, 128)
    assert not sm120_wagv_bmm_g_should_use(8, 2048, 513)


@pytest.mark.parametrize(
    ("capability", "expected"),
    [((8, 6), True), ((8, 9), True), ((12, 0), True), ((8, 0), False)],
)
def test_lora_extension_device_policy_admits_exact_sm86_probe(
    monkeypatch, capability: tuple[int, int], expected: bool
) -> None:
    monkeypatch.setattr(
        "rwkv7_hf.ada_lora._small_row_capability",
        lambda device=None: capability,
    )
    assert ada_wagv_lora_available("cuda", build=False) is expected


@pytest.mark.parametrize(
    ("capability", "expected"),
    [((8, 6), True), ((8, 9), True), ((12, 0), True), ((12, 1), False)],
)
def test_bmm_device_policy_is_exact_capability_gated(
    monkeypatch, capability: tuple[int, int], expected: bool
) -> None:
    monkeypatch.setattr(
        "rwkv7_hf.ada_lora._small_row_capability",
        lambda device=None: capability,
    )
    assert ada_wagv_bmm_available("cuda") is expected


@pytest.mark.parametrize(
    ("capability", "has_triton", "expected"),
    [
        ((12, 0), True, True),
        ((12, 0), False, False),
        ((8, 6), True, True),
        ((8, 9), True, True),
        ((12, 1), True, False),
    ],
)
def test_sm120_bmm_g_requires_exact_device_and_triton(
    monkeypatch,
    capability: tuple[int, int],
    has_triton: bool,
    expected: bool,
) -> None:
    monkeypatch.setattr(
        "rwkv7_hf.ada_lora._small_row_capability",
        lambda device=None: capability,
    )
    monkeypatch.setattr("rwkv7_hf.ada_lora._HAS_TRITON", has_triton)
    assert sm120_wagv_bmm_g_available("cuda") is expected


def test_sm120_bmm_g_pack_order_and_cache() -> None:
    hidden = 8
    w1 = torch.full((2, hidden), 1.0)
    a1 = torch.full((3, hidden), 2.0)
    g1 = torch.full((4, hidden), 3.0)
    v1 = torch.full((1, hidden), 4.0)
    w2 = torch.full((hidden, 2), 5.0)
    a2 = torch.full((hidden, 3), 6.0)
    g2 = torch.full((hidden, 4), 7.0)
    v2 = torch.full((hidden, 1), 8.0)
    w0 = torch.full((hidden,), 9.0)
    a0 = torch.full((hidden,), 10.0)
    v0 = torch.full((hidden,), 11.0)

    down, up_t, bias = _ada_wagv_bmm_pack(
        w1,
        a1,
        v1,
        w2,
        a2,
        v2,
        w0,
        a0,
        v0,
        include_v=True,
        include_g=True,
        g1=g1,
        g2=g2,
    )
    # Packed group order is V/W/A/G to match the six-slot norm/mix view.
    assert tuple(down.shape) == (4, 4, hidden)
    assert torch.equal(down[0, :1], v1)
    assert torch.equal(down[1, :2], w1)
    assert torch.equal(down[2, :3], a1)
    assert torch.equal(down[3, :4], g1)
    assert torch.equal(up_t[0, :1], v2.transpose(0, 1))
    assert torch.equal(up_t[1, :2], w2.transpose(0, 1))
    assert torch.equal(up_t[2, :3], a2.transpose(0, 1))
    assert torch.equal(up_t[3, :4], g2.transpose(0, 1))
    assert torch.equal(bias[:, 0], torch.stack((v0, w0, a0, torch.zeros_like(w0))))
    cached = getattr(w1, "_rwkv7_sm120_wagv_bmm_g_pack")
    repeated = _ada_wagv_bmm_pack(
        w1,
        a1,
        v1,
        w2,
        a2,
        v2,
        w0,
        a0,
        v0,
        include_v=True,
        include_g=True,
        g1=g1,
        g2=g2,
    )
    assert getattr(w1, "_rwkv7_sm120_wagv_bmm_g_pack") is cached
    assert all(left is right for left, right in zip((down, up_t, bias), repeated))


def test_sm120_bmm_g_reuses_six_slot_norm_mix_storage() -> None:
    rows, hidden = 8, 16
    backing = torch.arange(6 * rows * hidden).reshape(6, rows, hidden)
    xr, xk, xv, xw, xa, xg = backing
    grouped = _stack_sm120_wagv_inputs(xw, xa, xg, xv, include_v=True)
    assert grouped.untyped_storage().data_ptr() == backing.untyped_storage().data_ptr()
    assert torch.equal(grouped, torch.stack((xv, xw, xa, xg)))
    grouped_no_v = _stack_sm120_wagv_inputs(xw, xa, xg, xv, include_v=False)
    assert (
        grouped_no_v.untyped_storage().data_ptr()
        == backing.untyped_storage().data_ptr()
    )
    assert torch.equal(grouped_no_v, torch.stack((xw, xa, xg)))


def test_bmm_require_mode_fails_closed_on_fallback() -> None:
    torch.manual_seed(13)
    rows, hidden = 8, 32
    ranks = (8, 6, 4, 5)
    x = [torch.randn(rows, hidden) for _ in range(4)]
    down = [torch.randn(rank, hidden) for rank in ranks]
    up = [torch.randn(hidden, rank) for rank in ranks]
    w0, a0, v0 = (torch.randn(hidden) for _ in range(3))
    v = torch.randn(rows, hidden)
    v_first = torch.randn(rows, hidden)

    with pytest.raises(RuntimeError, match="exact device/dtype/layout contract"):
        ada_wagv_bmm(
            *x,
            *down,
            *up,
            w0,
            a0,
            v0,
            v,
            v_first,
            sigmoid_a=True,
            require_bmm=True,
        )


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
    outputs = ada_wagv_lora(*x, *down, *up, w0, a0, v0, v, v_first, force_fallback=True)
    assert len(outputs) == 4
    assert all(tuple(item.shape) == (rows, hidden) for item in outputs)
    assert all(torch.isfinite(item).all() for item in outputs)


def test_required_extension_rejects_fallback() -> None:
    torch.manual_seed(71)
    rows, hidden = 1, 32
    ranks = (8, 6, 4, 5)
    x = [torch.randn(rows, hidden) for _ in range(4)]
    down = [torch.randn(rank, hidden) for rank in ranks]
    up = [torch.randn(hidden, rank) for rank in ranks]
    w0, a0, v0 = (torch.randn(hidden) for _ in range(3))
    v = torch.randn(rows, hidden)
    v_first = torch.randn(rows, hidden)
    with pytest.raises(RuntimeError, match="fallback is forbidden"):
        ada_wagv_lora(
            *x,
            *down,
            *up,
            w0,
            a0,
            v0,
            v,
            v_first,
            force_fallback=True,
            require_extension=True,
        )


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
        *x,
        *down,
        *up,
        w0,
        a0,
        v0,
        v,
        v_first,
        sigmoid_a=True,
        compute_v=False,
        force_fallback=True,
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
        xw,
        xa,
        xg,
        w1,
        a1,
        g1,
        w2,
        a2,
        g2,
        w0,
        a0,
        force_fallback=True,
    )
    expected = (
        torch.nn.functional.linear(
            torch.tanh(torch.nn.functional.linear(xw, w1)), w2, w0
        ),
        torch.nn.functional.linear(torch.nn.functional.linear(xa, a1), a2, a0),
        torch.nn.functional.linear(
            torch.sigmoid(torch.nn.functional.linear(xg, g1)), g2
        ),
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
        *x,
        *down,
        *up,
        w0,
        a0,
        v0,
        v,
        v_first,
        sigmoid_a=True,
        force_fallback=True,
    )
    actual = ada_wagv_bmm(
        *x,
        *down,
        *up,
        w0,
        a0,
        v0,
        v,
        v_first,
        sigmoid_a=True,
    )
    for observed, reference in zip(actual, expected):
        torch.testing.assert_close(observed, reference)


def test_ada_b8_bmm_cuda_matches_fallback() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in {
        (8, 9),
        (12, 0),
    }:
        pytest.skip("exact sm_89/sm_120 B8 tensor-core route is unavailable")
    torch.manual_seed(12)
    rows, hidden = 8, 1024
    ranks = (64, 64, 128, 32)
    x = [
        torch.randn(rows, hidden, device="cuda", dtype=torch.float16) for _ in range(4)
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
        torch.randn(hidden, device="cuda", dtype=torch.float16) * 0.02 for _ in range(3)
    )
    v = torch.randn(rows, hidden, device="cuda", dtype=torch.float16)
    v_first = torch.randn(rows, hidden, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        expected = ada_wagv_lora(
            *x,
            *down,
            *up,
            w0,
            a0,
            v0,
            v,
            v_first,
            sigmoid_a=True,
            force_fallback=True,
        )
        actual = ada_wagv_bmm(
            *x,
            *down,
            *up,
            w0,
            a0,
            v0,
            v,
            v_first,
            sigmoid_a=True,
        )
        cached = getattr(down[0], "_rwkv7_ada_wagv_bmm_pack", None)
        repeated = ada_wagv_bmm(
            *x,
            *down,
            *up,
            w0,
            a0,
            v0,
            v,
            v_first,
            sigmoid_a=True,
        )
        expected_no_v = ada_wagv_lora(
            x[0],
            x[1],
            x[2],
            x[3],
            down[0],
            down[1],
            down[2],
            down[2],
            up[0],
            up[1],
            up[2],
            up[2],
            w0,
            a0,
            a0,
            v,
            v,
            sigmoid_a=True,
            compute_v=False,
            force_fallback=True,
        )
        actual_no_v = ada_wagv_bmm(
            x[0],
            x[1],
            x[2],
            x[3],
            down[0],
            down[1],
            down[2],
            down[2],
            up[0],
            up[1],
            up[2],
            up[2],
            w0,
            a0,
            a0,
            v,
            v,
            sigmoid_a=True,
            compute_v=False,
        )
    assert isinstance(cached, tuple) and len(cached) == 4
    assert getattr(down[0], "_rwkv7_ada_wagv_bmm_pack", None) is cached
    for reference, observed, second in zip(expected, actual, repeated):
        assert torch.allclose(reference.float(), observed.float(), atol=0.03, rtol=0.01)
        cosine = torch.nn.functional.cosine_similarity(
            reference.float().reshape(rows, -1),
            observed.float().reshape(rows, -1),
            dim=-1,
        ).min()
        assert float(cosine) >= 0.9999
        torch.testing.assert_close(observed, second)
    assert isinstance(getattr(down[0], "_rwkv7_ada_wa_bmm_pack", None), tuple)
    for reference, observed in zip(expected_no_v, actual_no_v):
        assert torch.allclose(reference.float(), observed.float(), atol=0.03, rtol=0.01)
        cosine = torch.nn.functional.cosine_similarity(
            reference.float().reshape(rows, -1),
            observed.float().reshape(rows, -1),
            dim=-1,
        ).min()
        assert float(cosine) >= 0.9999


@pytest.mark.parametrize(
    ("layer_name", "compute_v"),
    [("layer0", False), ("later_layer", True)],
)
def test_sm120_b8_bmm_g_zero_copy_cuda_matches_fallback(
    layer_name: str,
    compute_v: bool,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip(
            "CUDA is required for the exact-SM86/SM89/SM120 W/A/G/V route"
        )
    if torch.cuda.get_device_capability() not in {(8, 6), (8, 9), (12, 0)}:
        pytest.skip(
            "exact sm_86, sm_89, or sm_120 GPU is required for the W/A/G/V route"
        )
    if not sm120_wagv_bmm_g_available("cuda"):
        pytest.skip(
            "exact-SM86/SM89/SM120 W/A/G/V Triton epilogues are unavailable"
        )

    rows, hidden = 8, 1024
    ranks = (64, 64, 128, 32)
    generator = torch.Generator(device="cuda").manual_seed(1200 + int(compute_v))

    # Match the fused norm/mix contract exactly: R/K/V/W/A/G are six adjacent
    # [B,H] views, so layer 0 can reuse W/A/G and later layers V/W/A/G without
    # allocating a stack before the grouped BMM.
    norm_mix = torch.randn(
        6,
        rows,
        hidden,
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    _, _, xv, xw, xa, xg = norm_mix.unbind(0)
    rank_in = [
        torch.randn(
            rank,
            hidden,
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        * 0.02
        for rank in ranks
    ]
    rank_out = [
        torch.randn(
            hidden,
            rank,
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        * 0.02
        for rank in ranks
    ]
    w0 = torch.zeros(hidden, device="cuda", dtype=torch.float16)
    a0 = torch.zeros_like(w0)
    v0 = torch.zeros_like(w0)
    w0[17] = 4.0
    a0[23] = 4.0
    rank_out[2][41].add_(2.0)
    v = (
        torch.randn(
            rows,
            hidden,
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        * 0.1
    )
    v_first = (
        torch.randn(
            rows,
            hidden,
            device="cuda",
            dtype=torch.float16,
            generator=generator,
        )
        * 0.1
    )
    v[:, 31] = 4.0
    v_first[:, 31] = 4.0

    mixed = _stack_sm120_wagv_inputs(
        xw,
        xa,
        xg,
        xv,
        include_v=compute_v,
    )
    first = xv if compute_v else xw
    group_count = 4 if compute_v else 3
    assert mixed.is_contiguous()
    assert tuple(mixed.shape) == (group_count, rows, hidden)
    assert mixed.untyped_storage().data_ptr() == norm_mix.untyped_storage().data_ptr()
    assert mixed.data_ptr() == first.data_ptr()
    assert int(mixed.storage_offset()) == int(first.storage_offset())

    packed_down, packed_up_t, packed_bias = _ada_wagv_bmm_pack(
        rank_in[0],
        rank_in[1],
        rank_in[3],
        rank_out[0],
        rank_out[1],
        rank_out[3],
        w0,
        a0,
        v0,
        include_v=compute_v,
        include_g=True,
        g1=rank_in[2],
        g2=rank_out[2],
    )
    assert tuple(packed_down.shape) == (group_count, max(ranks), hidden)
    assert tuple(packed_up_t.shape) == (group_count, max(ranks), hidden)
    assert tuple(packed_bias.shape) == (group_count, 1, hidden)
    assert packed_down.is_contiguous()
    assert tuple(packed_up_t.stride()) == (
        hidden * max(ranks),
        1,
        max(ranks),
    )
    assert packed_up_t.transpose(1, 2).is_contiguous()
    assert packed_bias.is_contiguous()
    cache_name = (
        "_rwkv7_sm120_wagv_bmm_g_pack" if compute_v else "_rwkv7_sm120_wag_bmm_g_pack"
    )
    cached = getattr(rank_in[0], cache_name, None)
    assert isinstance(cached, tuple) and len(cached) == 4
    assert cached[1] is packed_down
    assert cached[2] is packed_up_t
    assert cached[3] is packed_bias

    with torch.inference_mode():
        reference = ada_wagv_lora(
            xw,
            xa,
            xg,
            xv,
            *rank_in,
            *rank_out,
            w0,
            a0,
            v0,
            v,
            v_first,
            sigmoid_a=True,
            compute_v=compute_v,
            force_fallback=True,
        )
        observed = ada_wagv_bmm(
            xw,
            xa,
            xg,
            xv,
            *rank_in,
            *rank_out,
            w0,
            a0,
            v0,
            v,
            v_first,
            sigmoid_a=True,
            compute_v=compute_v,
            include_g=True,
            require_bmm=True,
            require_zero_copy=True,
        )
    assert getattr(rank_in[0], cache_name, None) is cached

    for output_name, expected, actual in zip(
        ("w", "a", "g", "v"),
        reference,
        observed,
        strict=True,
    ):
        assert torch.isfinite(expected).all(), f"{layer_name}/{output_name} reference"
        assert torch.isfinite(actual).all(), f"{layer_name}/{output_name} actual"
        row_cosine = torch.nn.functional.cosine_similarity(
            expected.float(),
            actual.float(),
            dim=-1,
        )
        assert torch.isfinite(row_cosine).all()
        assert float(row_cosine.min()) >= 0.9999, (
            layer_name,
            output_name,
            row_cosine.tolist(),
        )
        assert torch.equal(expected.argmax(dim=-1), actual.argmax(dim=-1)), (
            layer_name,
            output_name,
        )


@pytest.mark.parametrize(
    "dtype,max_abs", [(torch.float16, 0.02), (torch.bfloat16, 0.03)]
)
def test_ada_cuda_matches_fallback_for_fp16_and_bf16(dtype, max_abs) -> None:
    if not torch.cuda.is_available() or not ada_wagv_lora_available("cuda"):
        pytest.skip("sm_89/sm_120 small-row CUDA kernel is unavailable")
    torch.manual_seed(11)
    rows, hidden = 1, 1024
    ranks = (64, 64, 128, 64)
    x = [torch.randn(rows, hidden, device="cuda", dtype=dtype) for _ in range(4)]
    down = [
        torch.randn(rank, hidden, device="cuda", dtype=dtype) * 0.02 for rank in ranks
    ]
    up = [
        torch.randn(hidden, rank, device="cuda", dtype=dtype) * 0.02 for rank in ranks
    ]
    w0, a0, v0 = (
        torch.randn(hidden, device="cuda", dtype=dtype) * 0.02 for _ in range(3)
    )
    v = torch.randn(rows, hidden, device="cuda", dtype=dtype)
    v_first = torch.randn(rows, hidden, device="cuda", dtype=dtype)
    with torch.inference_mode():
        reference = ada_wagv_lora(
            *x, *down, *up, w0, a0, v0, v, v_first, force_fallback=True
        )
        actual = ada_wagv_lora(*x, *down, *up, w0, a0, v0, v, v_first)
    for expected, observed in zip(reference, actual):
        assert torch.allclose(
            expected.float(), observed.float(), atol=max_abs, rtol=0.01
        )
        cosine = torch.nn.functional.cosine_similarity(
            expected.float().flatten().unsqueeze(0),
            observed.float().flatten().unsqueeze(0),
        ).item()
        assert cosine >= 0.9999
