from __future__ import annotations

import torch

from rwkv7_hf.native_model import (
    NativeRWKV7Config,
    NativeRWKV7ForCausalLM,
    _zero3_pad_native_training_batch,
)


def _tiny_model() -> NativeRWKV7ForCausalLM:
    return NativeRWKV7ForCausalLM(
        NativeRWKV7Config(
            vocab_size=32,
            hidden_size=8,
            num_hidden_layers=1,
            head_dim=4,
            intermediate_size=16,
            decay_low_rank_dim=2,
            gate_low_rank_dim=2,
            a_low_rank_dim=2,
            v_low_rank_dim=2,
        )
    )


def test_zero3_variable_length_batch_is_globally_padded(monkeypatch) -> None:
    torch.manual_seed(20260722)
    model = _tiny_model()
    first_param = next(model.parameters())
    first_param.ds_id = 0
    first_param.ds_status = object()
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def fake_all_reduce(length, op=None):
        length.fill_(6)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    labels = input_ids.clone()
    padded_ids, _, padded_mask, padded_labels, local_length = _zero3_pad_native_training_batch(
        model,
        input_ids,
        None,
        None,
        labels,
        pad_token_id=0,
    )

    assert local_length == 4
    assert padded_ids.tolist() == [[1, 2, 3, 4, 0, 0]]
    assert padded_mask.tolist() == [[1, 1, 1, 1, 0, 0]]
    assert padded_labels.tolist() == [[1, 2, 3, 4, -100, -100]]
