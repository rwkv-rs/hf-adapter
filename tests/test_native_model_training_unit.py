#!/usr/bin/env python3
# coding=utf-8
"""CPU unit coverage for the experimental native RWKV-7 CausalLM training API.

This intentionally uses a tiny random config so it can run without converted
weights, FLA, CUDA, or model files. The checkpoint-level equivalence tests stay
in ``tests/test_native_model.py``; this file verifies the HF/PEFT-facing API
surface that training stacks expect from a CausalLM fallback.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from rwkv7_hf.native_model import NativeRWKV7Config, NativeRWKV7ForCausalLM


def build_tiny_model() -> NativeRWKV7ForCausalLM:
    torch.manual_seed(1234)
    cfg = NativeRWKV7Config(
        vocab_size=23,
        hidden_size=8,
        num_hidden_layers=2,
        head_dim=4,
        intermediate_size=16,
        decay_low_rank_dim=3,
        gate_low_rank_dim=3,
        a_low_rank_dim=3,
        v_low_rank_dim=3,
        use_cache=True,
    )
    return NativeRWKV7ForCausalLM(cfg)


def main() -> int:
    model = build_tiny_model()
    input_ids = torch.tensor(
        [
            [1, 2, 3, 4, 5],
            [6, 5, 4, 3, 2],
        ],
        dtype=torch.long,
    )
    labels = input_ids.clone()
    labels[0, 2] = -100

    assert model.get_input_embeddings() is model.model.embeddings
    assert model.get_output_embeddings() is model.lm_head

    out = model(input_ids=input_ids, labels=labels, use_cache=False)
    assert out.loss is not None
    assert out.past_key_values is None
    assert out.logits.shape == (2, 5, 23)
    expected = F.cross_entropy(
        out.logits[:, :-1, :].contiguous().view(-1, 23).float(),
        labels[:, 1:].contiguous().view(-1),
        ignore_index=-100,
    )
    assert torch.allclose(out.loss, expected), (out.loss.item(), expected.item())
    out.loss.backward()
    assert model.get_input_embeddings().weight.grad is not None
    assert model.get_output_embeddings().weight.grad is not None
    assert torch.isfinite(model.get_input_embeddings().weight.grad).all()
    assert torch.isfinite(model.get_output_embeddings().weight.grad).all()

    tuple_out = model(input_ids=input_ids, labels=labels, return_dict=False)
    assert len(tuple_out) == 2
    assert tuple_out[0].shape == ()
    assert tuple_out[1].shape == (2, 5, 23)

    with torch.no_grad():
        cached = model(input_ids=input_ids[:, :3], use_cache=True)
        assert cached.logits.shape == (2, 1, 23)
        assert cached.past_key_values is not None

    try:
        model(input_ids=input_ids, labels=labels[:, :4])
    except ValueError as exc:
        assert "same shape" in str(exc)
    else:
        raise AssertionError("mismatched labels should raise ValueError")

    try:
        model(input_ids=input_ids[:, :1], labels=labels[:, :1], past_key_values=cached.past_key_values)
    except ValueError as exc:
        assert "past_key_values" in str(exc)
    else:
        raise AssertionError("labels with past_key_values should raise ValueError")

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
