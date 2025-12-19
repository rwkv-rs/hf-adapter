#!/usr/bin/env python3
# coding=utf-8
"""Upstream-style API contract checks for the native RWKV-7 modules.

This is intentionally tiny and CPU-only.  It covers surfaces that Transformers
model common tests usually probe before the native backend is moved toward an
upstream-style ``transformers.models.rwkv7`` layout.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig

from rwkv7_hf.native_model import NativeRWKV7Config, NativeRWKV7ForCausalLM


def build_tiny_model() -> NativeRWKV7ForCausalLM:
    torch.manual_seed(20260705)
    cfg = NativeRWKV7Config(
        vocab_size=29,
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
    return NativeRWKV7ForCausalLM(cfg).eval()


def expect_unsupported_attentions(fn) -> None:
    try:
        fn()
    except NotImplementedError as exc:
        assert "attention maps" in str(exc)
    else:
        raise AssertionError("output_attentions=True should be rejected explicitly")


def main() -> int:
    default_config = NativeRWKV7Config()
    assert default_config.vocab_size == 65536
    assert default_config.tie_word_embeddings is False
    assert default_config.num_attention_heads == default_config.num_heads
    assert NativeRWKV7Config(vocab_size=17).vocab_size == 17
    alias_config = NativeRWKV7Config(hidden_size=16, head_dim=4, num_attention_heads=4)
    assert alias_config.num_heads == 4
    assert alias_config.num_attention_heads == 4
    alias_config_with_null_heads = NativeRWKV7Config(hidden_size=16, head_dim=4, num_heads=None, num_attention_heads=4)
    assert alias_config_with_null_heads.num_heads == 4
    assert alias_config_with_null_heads.num_attention_heads == 4
    roundtrip_config = NativeRWKV7Config.from_dict(alias_config.to_dict())
    assert roundtrip_config.model_type == "rwkv7_native"
    assert roundtrip_config.vocab_size == alias_config.vocab_size
    assert roundtrip_config.num_attention_heads == alias_config.num_attention_heads
    assert roundtrip_config.tie_word_embeddings is False
    bnb_kwargs = {
        "config": roundtrip_config,
        "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
        "rwkv7_bnb_skip_policy": "memory",
    }
    bnb_policy, bnb_config = NativeRWKV7ForCausalLM._rwkv7_prepare_bnb_kwargs(
        "unused-local-model",
        bnb_kwargs,
    )
    assert bnb_policy == "memory"
    assert "rwkv7_bnb_skip_policy" not in bnb_kwargs
    assert bnb_config is bnb_kwargs["quantization_config"]
    assert "lm_head" in bnb_config.llm_int8_skip_modules
    assert r".*_lora\.lora\.[02]" in bnb_config.llm_int8_skip_modules
    assert "model.layers.0.attn.w_lora.lora.0" in bnb_config.llm_int8_skip_modules
    model = build_tiny_model()
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    token_type_ids = torch.zeros_like(input_ids)
    head_mask = torch.ones(model.config.num_hidden_layers, dtype=torch.float32)
    position_ids = torch.arange(input_ids.shape[1], dtype=torch.long).unsqueeze(0).expand_as(input_ids)
    cache_position = torch.arange(input_ids.shape[1], dtype=torch.long)
    assert model.main_input_name == "input_ids"
    assert model.model.main_input_name == "input_ids"
    assert model.base_model is model.model
    assert model.model.base_model is model.model
    assert model.can_generate()
    assert model.config.tie_word_embeddings is False
    assert model.get_input_embeddings().weight is not model.get_output_embeddings().weight
    base_embeddings = model.model.get_input_embeddings()
    assert model.model.resize_token_embeddings(model.config.vocab_size) is base_embeddings
    try:
        model.model.resize_token_embeddings(model.config.vocab_size + 1)
    except NotImplementedError as exc:
        assert "fixed official trie vocabulary" in str(exc)
    else:
        raise AssertionError("bare native model should reject RWKV vocab resize")
    model.tie_weights()
    assert model.get_input_embeddings().weight is not model.get_output_embeddings().weight

    with torch.no_grad():
        base_out = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=True,
            output_hidden_states=True,
        )
    assert base_out.last_hidden_state.shape == (2, 3, model.config.hidden_size)
    assert base_out.past_key_values.get_seq_length() == 3
    assert len(base_out.hidden_states) == model.config.num_hidden_layers + 1
    assert base_out.hidden_states[0].shape == (2, 3, model.config.hidden_size)
    assert base_out.hidden_states[-1].shape == base_out.last_hidden_state.shape
    assert torch.allclose(base_out.hidden_states[-1], base_out.last_hidden_state)

    inputs_embeds = model.get_input_embeddings()(input_ids)
    with torch.no_grad():
        base_from_embeds = model.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        base_from_token_type_ids = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        base_from_head_mask = model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        causal_from_ids = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        causal_from_token_type_ids = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        causal_from_head_mask = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        causal_from_position_ids = model(input_ids=input_ids, position_ids=position_ids, cache_position=cache_position)
        causal_from_embeds = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        causal_embeds_prefill = model(inputs_embeds=inputs_embeds[:, :2], use_cache=True)
        causal_embeds_decode = model(
            inputs_embeds=inputs_embeds[:, 2:3],
            past_key_values=causal_embeds_prefill.past_key_values,
            use_cache=True,
        )
        causal_keep_one = model(input_ids=input_ids, logits_to_keep=1)
        causal_keep_two = model(input_ids=input_ids, num_logits_to_keep=2)
        causal_keep_positions = model(input_ids=input_ids, logits_to_keep=torch.tensor([0, 2], dtype=torch.long))
        native_prefill = model.rwkv7_prefill_native(input_ids, logits_to_keep=1)
        native_prefill_tuple = model.rwkv7_prefill_native(input_ids, logits_to_keep=1, return_dict=False)
        assert model.rwkv7_last_fast_prefill_backend() == "native_eager"
        chunked_prefill = model.rwkv7_prefill_chunks(input_ids, chunk_size=1, logits_to_keep=1)
        chunked_prefill_tuple = model.rwkv7_prefill_chunks(
            input_ids,
            chunk_size=2,
            logits_to_keep=1,
            return_dict=False,
        )
        speculative = model.rwkv7_speculative_generate(
            input_ids[:1],
            model,
            max_new_tokens=3,
            draft_tokens=2,
            return_stats=True,
        )
        greedy = model.generate(
            input_ids[:1],
            max_new_tokens=3,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
        )
        causal_1d = model(
            input_ids=input_ids[0],
            attention_mask=torch.ones(input_ids.shape[1], dtype=torch.long),
            logits_to_keep=1,
        )
        embedded_loss = model(inputs_embeds=inputs_embeds, labels=input_ids)
        decode = model(
            input_ids[:, -1:],
            past_key_values=causal_from_ids.past_key_values.clone(),
            use_cache=True,
            output_hidden_states=True,
        )
        decode_full_mask = model(
            input_ids[:, -1:],
            past_key_values=causal_from_ids.past_key_values.clone(),
            attention_mask=torch.ones((2, input_ids.shape[1] + 1), dtype=torch.long),
            use_cache=True,
        )
        prefill_one = model(input_ids[:, :1], use_cache=True)
        prefill_cache_for_multi = prefill_one.past_key_values.clone()
        prefill_state_before = [state.clone() for state in prefill_cache_for_multi._state]
        cached_multi = model(
            input_ids[:, 1:],
            past_key_values=prefill_cache_for_multi,
            use_cache=True,
        )
        prefill_no_cache = prefill_one.past_key_values.clone()
        prefill_no_cache_state_before = [state.clone() for state in prefill_no_cache._state]
        cached_multi_no_return_cache = model(
            input_ids[:, 1:],
            past_key_values=prefill_no_cache,
            use_cache=False,
        )
        cached_multi_full_mask = model(
            input_ids[:, 1:],
            past_key_values=prefill_one.past_key_values.clone(),
            attention_mask=torch.ones((2, input_ids.shape[1]), dtype=torch.long),
            use_cache=True,
        )
        base_prefill_one = model.model(input_ids[:, :1], use_cache=True)
        base_prefill_cache_for_multi = base_prefill_one.past_key_values.clone()
        base_prefill_state_before = [state.clone() for state in base_prefill_cache_for_multi._state]
        base_cached_multi = model.model(
            input_ids[:, 1:],
            past_key_values=base_prefill_cache_for_multi,
            use_cache=True,
        )
        base_embeds_prefill = model.model(inputs_embeds=inputs_embeds[:, :2], use_cache=True)
        base_embeds_decode = model.model(
            inputs_embeds=inputs_embeds[:, 2:3],
            past_key_values=base_embeds_prefill.past_key_values,
            use_cache=True,
        )
        base_prefill_no_cache = base_prefill_one.past_key_values.clone()
        base_prefill_no_cache_state_before = [state.clone() for state in base_prefill_no_cache._state]
        base_cached_multi_no_return_cache = model.model(
            input_ids[:, 1:],
            past_key_values=base_prefill_no_cache,
            use_cache=False,
        )
        base_decode_full_mask = model.model(
            input_ids[:, -1:],
            past_key_values=base_out.past_key_values.clone(),
            attention_mask=torch.ones((2, input_ids.shape[1] + 1), dtype=torch.long),
            use_cache=True,
        )
    assert torch.allclose(base_from_embeds.last_hidden_state, base_out.last_hidden_state)
    assert torch.allclose(base_from_token_type_ids.last_hidden_state, base_out.last_hidden_state)
    assert torch.allclose(base_from_head_mask.last_hidden_state, base_out.last_hidden_state)
    assert torch.allclose(causal_from_embeds.logits, causal_from_ids.logits)
    assert torch.allclose(causal_embeds_decode.logits, causal_from_ids.logits[:, 2:3])
    assert torch.allclose(causal_from_token_type_ids.logits, causal_from_ids.logits)
    assert torch.allclose(causal_from_head_mask.logits, causal_from_ids.logits)
    assert torch.allclose(causal_from_position_ids.logits, causal_from_ids.logits)
    assert causal_keep_one.logits.shape == (2, 1, model.config.vocab_size)
    assert torch.allclose(causal_keep_one.logits, causal_from_ids.logits[:, -1:])
    assert causal_keep_two.logits.shape == (2, 2, model.config.vocab_size)
    assert torch.allclose(causal_keep_two.logits, causal_from_ids.logits[:, -2:])
    assert torch.allclose(causal_keep_positions.logits, causal_from_ids.logits[:, [0, 2]])
    assert torch.allclose(native_prefill.logits, causal_keep_one.logits)
    assert native_prefill.past_key_values.get_seq_length() == input_ids.shape[1]
    assert torch.allclose(native_prefill_tuple[0], causal_keep_one.logits)
    assert native_prefill_tuple[1].get_seq_length() == input_ids.shape[1]
    assert torch.allclose(chunked_prefill.logits, causal_keep_one.logits)
    assert chunked_prefill.past_key_values.get_seq_length() == input_ids.shape[1]
    assert torch.allclose(chunked_prefill_tuple[0], causal_keep_one.logits)
    assert chunked_prefill_tuple[1].get_seq_length() == input_ids.shape[1]
    assert torch.equal(speculative["sequences"], greedy)
    assert speculative["stats"]["generated_tokens"] == 3
    assert speculative["stats"]["accepted_tokens"] == 3
    assert causal_1d.logits.shape == (1, 1, model.config.vocab_size)
    assert torch.allclose(causal_1d.logits, causal_from_ids.logits[:1, -1:])
    assert embedded_loss.loss is not None
    assert torch.isfinite(embedded_loss.loss)
    assert len(causal_from_ids.hidden_states) == model.config.num_hidden_layers + 1
    assert causal_from_ids.hidden_states[-1].shape == (2, 3, model.config.hidden_size)
    assert decode.logits.shape == (2, 1, model.config.vocab_size)
    assert torch.allclose(decode_full_mask.logits, decode.logits)
    assert cached_multi.logits.shape == (2, 2, model.config.vocab_size)
    assert cached_multi.past_key_values.get_seq_length() == input_ids.shape[1]
    assert torch.allclose(cached_multi.logits, causal_from_ids.logits[:, 1:])
    assert cached_multi_no_return_cache.past_key_values is None
    assert torch.allclose(cached_multi_no_return_cache.logits, cached_multi.logits)
    assert prefill_cache_for_multi.get_seq_length() == 1
    assert prefill_no_cache.get_seq_length() == 1
    for before, after in zip(prefill_state_before, prefill_cache_for_multi._state, strict=False):
        assert torch.equal(before, after)
    for before, after in zip(prefill_no_cache_state_before, prefill_no_cache._state, strict=False):
        assert torch.equal(before, after)
    assert torch.allclose(cached_multi_full_mask.logits, cached_multi.logits)
    assert base_cached_multi.last_hidden_state.shape == (2, 2, model.config.hidden_size)
    assert base_cached_multi.past_key_values.get_seq_length() == input_ids.shape[1]
    assert torch.allclose(base_cached_multi.last_hidden_state, base_out.last_hidden_state[:, 1:])
    assert torch.allclose(base_embeds_decode.last_hidden_state, base_out.last_hidden_state[:, 2:3])
    assert base_cached_multi_no_return_cache.past_key_values is None
    assert torch.allclose(base_cached_multi_no_return_cache.last_hidden_state, base_cached_multi.last_hidden_state)
    assert base_prefill_cache_for_multi.get_seq_length() == 1
    assert base_prefill_no_cache.get_seq_length() == 1
    for before, after in zip(base_prefill_state_before, base_prefill_cache_for_multi._state, strict=False):
        assert torch.equal(before, after)
    for before, after in zip(base_prefill_no_cache_state_before, base_prefill_no_cache._state, strict=False):
        assert torch.equal(before, after)
    assert base_decode_full_mask.last_hidden_state.shape == (2, 1, model.config.hidden_size)
    assert decode.hidden_states[-1].shape == (2, 1, model.config.hidden_size)
    try:
        model(input_ids[:1, -1:], past_key_values=causal_from_ids.past_key_values.clone(), use_cache=True)
    except ValueError as exc:
        assert "cache batch size" in str(exc)
    else:
        raise AssertionError("CausalLM should reject cache/input batch-size mismatch")
    try:
        model.model(input_ids[:1, -1:], past_key_values=base_out.past_key_values.clone(), use_cache=True)
    except ValueError as exc:
        assert "cache batch size" in str(exc)
    else:
        raise AssertionError("base model should reject cache/input batch-size mismatch")
    assert model.get_output_embeddings() is model.lm_head
    original_decoder = model.get_decoder()
    original_embeddings = model.get_input_embeddings()
    original_lm_head = model.get_output_embeddings()
    model._rwkv7_native_model_jit_pack_cache = ("sentinel", object())
    model.set_input_embeddings(torch.nn.Embedding(model.config.vocab_size, model.config.hidden_size))
    assert not hasattr(model, "_rwkv7_native_model_jit_pack_cache")
    model.set_input_embeddings(original_embeddings)
    model._rwkv7_native_model_jit_pack_cache = ("sentinel", object())
    model.set_output_embeddings(torch.nn.Linear(model.config.hidden_size, model.config.vocab_size, bias=False))
    assert not hasattr(model, "_rwkv7_native_model_jit_pack_cache")
    model.set_output_embeddings(original_lm_head)
    replacement_decoder = type(original_decoder)(model.config)
    model._rwkv7_native_model_jit_pack_cache = ("sentinel", object())
    model.set_decoder(replacement_decoder)
    assert model.get_decoder() is replacement_decoder
    assert not hasattr(model, "_rwkv7_native_model_jit_pack_cache")
    model.set_decoder(original_decoder)
    assert model.get_decoder() is original_decoder

    padded_ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 0]], dtype=torch.long)
    padded_mask = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.long)
    compact_rows = [
        torch.tensor([[1, 2, 3]], dtype=torch.long),
        torch.tensor([[4, 5, 6]], dtype=torch.long),
    ]
    with torch.no_grad():
        padded_base = model.model(input_ids=padded_ids, attention_mask=padded_mask, use_cache=True)
        padded = model(input_ids=padded_ids, attention_mask=padded_mask, use_cache=True)
        compact = [model(input_ids=row, use_cache=True) for row in compact_rows]
        compact_base = [model.model(input_ids=row, use_cache=True) for row in compact_rows]
    assert padded_base.last_hidden_state.shape == (2, 4, model.config.hidden_size)
    assert padded.past_key_values.get_seq_length() == padded_ids.shape[1]
    for row_idx, valid_out in enumerate(compact):
        assert torch.allclose(padded.logits[row_idx, -1], valid_out.logits[0, -1], atol=1e-6)
        assert torch.allclose(
            padded_base.last_hidden_state[row_idx, -1],
            compact_base[row_idx].last_hidden_state[0, -1],
            atol=1e-6,
        )
    next_token = torch.cat([out.logits[:, -1:].argmax(dim=-1) for out in compact], dim=0)
    with torch.no_grad():
        padded_next = model(input_ids=next_token, past_key_values=padded.past_key_values, use_cache=True)
        compact_next = [
            model(
                input_ids=next_token[row_idx : row_idx + 1],
                past_key_values=compact[row_idx].past_key_values,
                use_cache=True,
            )
            for row_idx in range(2)
        ]
    for row_idx, valid_next in enumerate(compact_next):
        assert torch.allclose(padded_next.logits[row_idx], valid_next.logits[0], atol=1e-6)

    prepared = model.prepare_inputs_for_generation(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        head_mask=head_mask,
        use_cache=None,
        logits_to_keep=1,
        cache_position=cache_position,
    )
    assert prepared["input_ids"].shape == input_ids.shape
    assert prepared["attention_mask"].shape == attention_mask.shape
    assert torch.equal(prepared["token_type_ids"], token_type_ids)
    assert prepared["head_mask"] is head_mask
    assert prepared["use_cache"] is True
    assert prepared["logits_to_keep"] == 1
    assert torch.equal(prepared["cache_position"], cache_position)
    prepared_embeds = model.prepare_inputs_for_generation(
        input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=True,
    )
    assert "input_ids" not in prepared_embeds
    assert torch.equal(prepared_embeds["inputs_embeds"], inputs_embeds)
    prepared_cached = model.prepare_inputs_for_generation(
        input_ids,
        past_key_values=causal_from_ids.past_key_values,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        head_mask=head_mask,
        position_ids=position_ids,
        cache_position=cache_position,
        num_logits_to_keep=1,
        use_cache=True,
    )
    assert prepared_cached["input_ids"].shape == (2, 1)
    assert prepared_cached["attention_mask"].shape == (2, 1)
    assert prepared_cached["token_type_ids"].shape == (2, 1)
    assert torch.equal(prepared_cached["token_type_ids"], token_type_ids[:, -1:])
    assert prepared_cached["head_mask"] is head_mask
    assert prepared_cached["position_ids"].shape == (2, 1)
    assert torch.equal(prepared_cached["position_ids"], position_ids[:, -1:])
    assert prepared_cached["cache_position"].shape == (1,)
    assert torch.equal(prepared_cached["cache_position"], cache_position[-1:])
    assert prepared_cached["num_logits_to_keep"] == 1
    prepared_cached_1d = model.prepare_inputs_for_generation(
        input_ids[0],
        past_key_values=causal_1d.past_key_values,
        attention_mask=torch.ones(input_ids.shape[1], dtype=torch.long),
        token_type_ids=torch.zeros(input_ids.shape[1], dtype=torch.long),
        position_ids=torch.arange(input_ids.shape[1], dtype=torch.long),
        cache_position=torch.arange(input_ids.shape[1], dtype=torch.long),
        use_cache=True,
    )
    assert prepared_cached_1d["input_ids"].shape == (1,)
    assert torch.equal(prepared_cached_1d["input_ids"], input_ids[0, -1:])
    assert prepared_cached_1d["attention_mask"].shape == (1,)
    assert prepared_cached_1d["token_type_ids"].shape == (1,)
    assert prepared_cached_1d["position_ids"].shape == (1,)
    assert prepared_cached_1d["cache_position"].shape == (1,)
    prepared_cached_scalar_pos = model.prepare_inputs_for_generation(
        input_ids[:, -1:],
        past_key_values=causal_from_ids.past_key_values,
        position_ids=torch.tensor(input_ids.shape[1] - 1, dtype=torch.long),
        cache_position=torch.tensor(input_ids.shape[1] - 1, dtype=torch.long),
        use_cache=True,
    )
    assert prepared_cached_scalar_pos["position_ids"].shape == (1,)
    assert prepared_cached_scalar_pos["cache_position"].shape == (1,)
    assert int(prepared_cached_scalar_pos["position_ids"].item()) == input_ids.shape[1] - 1
    assert int(prepared_cached_scalar_pos["cache_position"].item()) == input_ids.shape[1] - 1
    prepared_cached_embeds = model.prepare_inputs_for_generation(
        None,
        inputs_embeds=inputs_embeds,
        past_key_values=causal_from_ids.past_key_values,
        attention_mask=attention_mask,
        use_cache=True,
    )
    assert "input_ids" not in prepared_cached_embeds
    assert prepared_cached_embeds["inputs_embeds"].shape == (2, 1, model.config.hidden_size)
    assert torch.equal(prepared_cached_embeds["inputs_embeds"], inputs_embeds[:, -1:])
    assert prepared_cached_embeds["attention_mask"].shape == (2, 1)

    tuple_out = model.model(input_ids=input_ids, return_dict=False)
    assert tuple_out[0].shape == (2, 3, model.config.hidden_size)
    causal_tuple = model(input_ids=input_ids, return_dict=False)
    assert causal_tuple[0].shape == (2, 3, model.config.vocab_size)
    model.config.return_dict = False
    base_tuple_default = model.model(input_ids=input_ids)
    causal_tuple_default = model(input_ids=input_ids)
    assert isinstance(base_tuple_default, tuple)
    assert isinstance(causal_tuple_default, tuple)
    model.config.return_dict = True
    model.config.output_hidden_states = True
    assert model(input_ids=input_ids).hidden_states is not None
    model.config.output_hidden_states = False
    model.config.use_cache = False
    assert model(input_ids=input_ids).past_key_values is None
    assert model.model(input_ids=input_ids).past_key_values is None
    model.config.use_cache = True

    with tempfile.TemporaryDirectory(prefix="rwkv7_native_contract_") as tmp:
        out_dir = Path(tmp)
        model.save_pretrained(out_dir)
        saved_config = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
        assert saved_config["auto_map"]["AutoConfig"] == "native_model.NativeRWKV7Config"
        assert saved_config["auto_map"]["AutoModel"] == "native_model.NativeRWKV7Model"
        assert saved_config["auto_map"]["AutoModelForCausalLM"] == "native_model.NativeRWKV7ForCausalLM"
        assert saved_config["tie_word_embeddings"] is False
        assert (out_dir / "native_model.py").exists()
        reloaded_config = AutoConfig.from_pretrained(out_dir, trust_remote_code=True)
        assert reloaded_config.__class__.__name__ == "NativeRWKV7Config"
        reloaded_base = AutoModel.from_pretrained(out_dir, trust_remote_code=True).eval()
        reloaded = AutoModelForCausalLM.from_pretrained(out_dir, trust_remote_code=True).eval()
        with torch.no_grad():
            reloaded_base_hidden = reloaded_base(input_ids=input_ids).last_hidden_state
            reloaded_logits = reloaded(input_ids=input_ids, logits_to_keep=1).logits
        assert reloaded_base.__class__.__name__ == "NativeRWKV7Model"
        assert reloaded.__class__.__name__ == "NativeRWKV7ForCausalLM"
        assert torch.allclose(reloaded_base_hidden, base_out.last_hidden_state)
        assert torch.allclose(reloaded_logits, causal_from_ids.logits[:, -1:])
        reloaded_base.gradient_checkpointing_enable()
        assert getattr(reloaded_base, "is_gradient_checkpointing", True)

    expect_unsupported_attentions(lambda: model.model(input_ids=input_ids, output_attentions=True))
    expect_unsupported_attentions(lambda: model(input_ids=input_ids, output_attentions=True))
    try:
        model(input_ids=input_ids, logits_to_keep=1, num_logits_to_keep=2)
    except ValueError as exc:
        assert "must match" in str(exc)
    else:
        raise AssertionError("conflicting logits_to_keep aliases should raise ValueError")

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
