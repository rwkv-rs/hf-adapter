from __future__ import annotations

import torch
import torch.nn.functional as F

from rwkv7_hf.sm70_wagv import sm70_orig_linear, sm70_orig_rkv, sm70_wagv_lora


def test_wagv_fallback_matches_reference() -> None:
    torch.manual_seed(70)
    rows, hidden, rank = 3, 32, 8
    xw, xa, xg, xv = (torch.randn(rows, hidden) for _ in range(4))
    w1, a1, g1, v1 = (torch.randn(rank, hidden) for _ in range(4))
    w2, a2, g2, v2 = (torch.randn(hidden, rank) for _ in range(4))
    w0, a0, v0 = (torch.randn(hidden) for _ in range(3))
    value, value_first = torch.randn(rows, hidden), torch.randn(rows, hidden)
    got = sm70_wagv_lora(
        xw,
        xa,
        xg,
        xv,
        w1,
        a1,
        g1,
        v1,
        w2,
        a2,
        g2,
        v2,
        w0,
        a0,
        v0,
        value,
        value_first,
        force_fallback=True,
    )
    gate = torch.sigmoid(F.linear(F.linear(xv, v1), v2, v0))
    expected = (
        F.linear(torch.tanh(F.linear(xw, w1)), w2, w0),
        F.linear(F.linear(xa, a1), a2, a0),
        F.linear(torch.sigmoid(F.linear(xg, g1)), g2),
        value + (value_first - value) * gate,
    )
    for actual, reference in zip(got, expected):
        assert torch.equal(actual, reference)


def test_orig_linear_fallback_for_unsupported_shape() -> None:
    torch.manual_seed(71)
    x = torch.randn(2, 64)
    weights = [torch.randn(64, 64) for _ in range(3)]
    assert torch.equal(sm70_orig_linear(x, weights[0]), F.linear(x, weights[0]))
    got = sm70_orig_rkv(x, x + 1, x - 1, *weights)
    refs = (
        F.linear(x, weights[0]),
        F.linear(x + 1, weights[1]),
        F.linear(x - 1, weights[2]),
    )
    for actual, reference in zip(got, refs):
        assert torch.equal(actual, reference)
