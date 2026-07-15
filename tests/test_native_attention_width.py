from __future__ import annotations

import pytest
import torch

from rwkv7_hf.native_model import NativeRWKV7Config, NativeRWKV7ForCausalLM


def test_native_attention_width_can_exceed_residual_hidden_size() -> None:
    config = NativeRWKV7Config(
        vocab_size=31,
        hidden_size=8,
        attention_hidden_size=16,
        num_heads=4,
        head_dim=4,
        num_hidden_layers=2,
        intermediate_size=16,
        decay_low_rank_dim=3,
        a_low_rank_dim=3,
        gate_low_rank_dim=3,
        v_low_rank_dim=3,
    )
    model = NativeRWKV7ForCausalLM(config)
    attention = model.model.layers[0].attn
    assert attention.r_proj.weight.shape == (16, 8)
    assert attention.k_proj.weight.shape == (16, 8)
    assert attention.v_proj.weight.shape == (16, 8)
    assert attention.o_proj.weight.shape == (8, 16)
    assert attention.g_norm.weight.shape == (16,)

    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    result = model(input_ids=input_ids, labels=input_ids, use_cache=True)
    assert result.logits.shape == (2, 3, 31)
    assert result.loss is not None and torch.isfinite(result.loss)
    result.loss.backward()
    state, xpa, xpf, v_first = result.past_key_values
    assert state[0].shape == (2, 4, 4, 4)
    assert xpa[0].shape == (2, 8)
    assert xpf[0].shape == (2, 8)
    assert v_first.shape == (2, 16)


def test_native_attention_width_must_match_recurrent_heads() -> None:
    with pytest.raises(ValueError, match=r"num_heads \* head_dim"):
        NativeRWKV7Config(
            hidden_size=8,
            attention_hidden_size=12,
            num_heads=2,
            head_dim=4,
        )
