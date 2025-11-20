from __future__ import annotations

import torch
import torch.nn.functional as F

from rwkv7_hf.fused_ffn import fused_sequence_ffn


def test_sequence_ffn_fallback_matches_reference() -> None:
    torch.manual_seed(7007)
    batch, tokens, hidden, intermediate = 2, 4, 8, 16
    x = torch.randn(batch, tokens, hidden)
    prev = torch.randn(batch, hidden)
    mix = torch.rand(hidden)
    key = torch.randn(intermediate, hidden)
    value = torch.randn(hidden, intermediate)

    prev_seq = torch.cat([prev[:, None, :], x[:, :-1, :]], dim=1)
    shifted = x + (prev_seq - x) * mix.view(1, 1, hidden)
    expected = F.linear(torch.relu(F.linear(shifted, key)) ** 2, value)

    got, next_state = fused_sequence_ffn(x, prev, mix, key, value, force_fallback=True)

    torch.testing.assert_close(got, expected)
    torch.testing.assert_close(next_state, x[:, -1, :])


def test_sequence_ffn_accepts_singleton_sequence_prev_state() -> None:
    torch.manual_seed(7008)
    x = torch.randn(1, 3, 8)
    prev = torch.randn(1, 1, 8)
    mix = torch.rand(1, 1, 8)
    key = torch.randn(16, 8)
    value = torch.randn(8, 16)

    got, next_state = fused_sequence_ffn(x, prev, mix, key, value, force_fallback=True)

    assert got.shape == x.shape
    assert next_state.shape == (1, 8)
