from __future__ import annotations

import pytest

from rwkv7_hf.configuration_rwkv7 import RWKV7Config


@pytest.fixture
def tiny_config():
    return RWKV7Config(
        vocab_size=64,
        hidden_size=16,
        attention_hidden_size=16,
        num_hidden_layers=2,
        num_heads=2,
        head_dim=8,
        intermediate_size=32,
        decay_low_rank_dim=4,
        gate_low_rank_dim=4,
        a_low_rank_dim=4,
        v_low_rank_dim=4,
        pad_token_id=0,
        eos_token_id=0,
        bos_token_id=1,
    )
