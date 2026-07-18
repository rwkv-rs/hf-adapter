from __future__ import annotations

import pytest
import torch

from rwkv7_hf.native_model import NativeRWKV7Config, NativeRWKV7ForCausalLM


def build_tiny_model() -> NativeRWKV7ForCausalLM:
    torch.manual_seed(2026)
    config = NativeRWKV7Config(
        vocab_size=31,
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
    return NativeRWKV7ForCausalLM(config).eval()


def assert_cache_close(left, right) -> None:
    assert left.get_seq_length() == right.get_seq_length()
    for left_group, right_group in zip(left, right, strict=True):
        if isinstance(left_group, list):
            for left_tensor, right_tensor in zip(left_group, right_group, strict=True):
                assert torch.allclose(left_tensor, right_tensor)
        else:
            assert torch.allclose(left_group, right_group)


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 32])
def test_native_chunked_prefill_matches_full_cpu_forward(chunk_size: int) -> None:
    model = build_tiny_model()
    input_ids = torch.tensor(
        [[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]],
        dtype=torch.long,
    )
    with torch.inference_mode():
        full = model(input_ids, use_cache=True, logits_to_keep=1)
        chunked = model.rwkv7_prefill_chunks(
            input_ids,
            chunk_size=chunk_size,
            logits_to_keep=1,
        )
        next_token = full.logits.argmax(dim=-1)
        full_next = model(
            next_token,
            past_key_values=full.past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )
        chunked_next = model(
            next_token,
            past_key_values=chunked.past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )

    assert torch.allclose(chunked.logits, full.logits)
    assert torch.allclose(chunked_next.logits, full_next.logits)
    assert_cache_close(chunked.past_key_values, full.past_key_values)


def test_native_chunked_prefill_supports_mask_past_and_tuple_result() -> None:
    model = build_tiny_model()
    input_ids = torch.tensor([[0, 1, 2, 3, 4], [0, 5, 6, 7, 8]], dtype=torch.long)
    attention_mask = torch.tensor([[0, 1, 1, 1, 1], [0, 1, 1, 1, 1]], dtype=torch.long)
    with torch.inference_mode():
        full = model(input_ids, attention_mask=attention_mask, use_cache=True, logits_to_keep=1)
        chunked = model.rwkv7_prefill_chunks(
            input_ids,
            attention_mask=attention_mask,
            chunk_size=2,
            logits_to_keep=1,
        )
        prefix = model(input_ids[:, :2], attention_mask=attention_mask[:, :2], use_cache=True)
        continued = model(
            input_ids[:, 2:],
            attention_mask=attention_mask[:, 2:],
            past_key_values=prefix.past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )
        chunked_continued = model.rwkv7_prefill_chunks(
            input_ids[:, 2:],
            attention_mask=attention_mask[:, 2:],
            past_key_values=prefix.past_key_values,
            chunk_size=1,
            logits_to_keep=1,
            return_dict=False,
        )

    assert torch.allclose(chunked.logits, full.logits)
    assert chunked.past_key_values.get_seq_length() == input_ids.shape[1]
    assert isinstance(chunked_continued, tuple) and len(chunked_continued) == 2
    assert torch.allclose(chunked_continued[0], continued.logits)
    assert chunked_continued[1].get_seq_length() == input_ids.shape[1]


def test_native_chunked_prefill_validates_contract() -> None:
    model = build_tiny_model()
    ids = torch.tensor([[1, 2]], dtype=torch.long)
    with pytest.raises(ValueError, match="chunk_size"):
        model.rwkv7_prefill_chunks(ids, chunk_size=0)
    with pytest.raises(ValueError, match="same"):
        model.rwkv7_prefill_chunks(ids, attention_mask=torch.ones(1, 1, dtype=torch.long))
    model.train()
    with pytest.raises(RuntimeError, match="inference-only"):
        model.rwkv7_prefill_chunks(ids)
