#!/usr/bin/env python3
# coding=utf-8
"""CPU/no-CUDA generation smoke for the experimental native RWKV-7 CausalLM.

This uses a tiny random config and does not require converted weights, FLA, CUDA,
or external model files. It guards the CPU fallback GenerationMixin path that
upstream Transformers / AMD / no-GPU contributors depend on.
"""
from __future__ import annotations

import types

import torch
from transformers.cache_utils import DynamicCache

from rwkv7_hf.native_model import NativeRWKV7Cache, NativeRWKV7Config, NativeRWKV7ForCausalLM


def build_tiny_model() -> NativeRWKV7ForCausalLM:
    torch.manual_seed(2026)
    cfg = NativeRWKV7Config(
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
    return NativeRWKV7ForCausalLM(cfg).eval()


class WrappedHead(torch.nn.Module):
    """Module-only output head, matching native mm8/mm4 Linear replacements."""

    def __init__(self, linear: torch.nn.Linear):
        super().__init__()
        self.linear = linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def main() -> int:
    model = build_tiny_model()
    embeddings = model.get_input_embeddings()
    assert model.resize_token_embeddings(model.config.vocab_size) is embeddings
    try:
        model.resize_token_embeddings(model.config.vocab_size + 1)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("native model should reject RWKV vocab resize")

    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    batch_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    with torch.no_grad():
        batch_out = model(batch_ids, use_cache=True)
        batch_clone = batch_out.past_key_values.clone()
        assert batch_clone is not batch_out.past_key_values
        assert batch_clone._state is not batch_out.past_key_values._state
        assert batch_out.past_key_values.get_batch_size() == 2
        assert batch_out.past_key_values.get_seq_length(0) == 3
        assert batch_out.past_key_values.get_seq_length(99) == 0
        assert batch_out.past_key_values.seen_tokens == 3
        assert batch_out.past_key_values.is_initialized is True
        assert batch_out.past_key_values.is_sliding is False
        assert batch_out.past_key_values.max_batch_size == 2
        assert batch_out.past_key_values.max_cache_len == -1
        assert batch_out.past_key_values.get_mask_sizes(torch.arange(2, dtype=torch.long)) == (5, 0)
        assert batch_out.past_key_values.get_mask_sizes(2, 0) == (5, 0)
        assert batch_out.past_key_values.get_mask_sizes(2, 99) == (2, 0)
        assert batch_out.past_key_values.get_mask_sizes(torch.tensor(0, dtype=torch.long)) == (4, 0)
        assert batch_out.past_key_values.get_mask_sizes(torch.ones((1, 2), dtype=torch.long)) == (5, 0)
        assert batch_out.past_key_values.get_mask_sizes(None) == (3, 0)
        metrics = batch_out.past_key_values.rwkv7_cache_metrics()
        assert metrics["seen_tokens"] == 3
        assert metrics["batch_size"] == 2
        assert len(batch_out.past_key_values) == 4
        assert batch_out.past_key_values.layers == []
        assert "NativeRWKV7Cache(seen_tokens=3, batch_size=2, layers=2)" == repr(batch_out.past_key_values)
        assert batch_out.past_key_values[0] is batch_out.past_key_values._state
        assert batch_out.past_key_values[3] is batch_out.past_key_values._v_first
        assert batch_out.past_key_values[:2] == (
            batch_out.past_key_values._state,
            batch_out.past_key_values._xpa,
        )
        state_view = batch_out.past_key_values.states
        assert len(state_view) == model.config.num_hidden_layers
        assert state_view[0]["recurrent_state"] is batch_out.past_key_values._state[0]
        assert state_view[0]["attn_state"] is batch_out.past_key_values._xpa[0]
        assert state_view[0]["ffn_state"] is batch_out.past_key_values._xpf[0]
        assert state_view[0]["conv_state"] is None
        detached = batch_out.past_key_values.detach(inplace=False)
        assert detached is not batch_out.past_key_values
        assert detached._state[0].grad_fn is None
        assert detached.states[0]["recurrent_state"].grad_fn is None
        moved = detached.to(device="cpu", dtype=torch.float64, inplace=False)
        assert moved is not detached
        assert moved._state[0].dtype == torch.float64
        assert moved.states[0]["recurrent_state"].dtype == torch.float64
        compact = batch_out.past_key_values.batch_select([1], inplace=False)
        assert compact.get_batch_size() == 1
        assert compact.rwkv7_cache_metrics()["batch_select_calls"] == 1
        compact_alias = batch_out.past_key_values.compact([1], inplace=False)
        assert compact_alias.get_batch_size() == 1
        assert compact_alias.rwkv7_cache_metrics()["batch_select_calls"] == 1
        assert NativeRWKV7Cache().has_previous_state() is False
        assert batch_out.past_key_values.has_previous_state() is True
        assert batch_out.past_key_values.has_previous_state(0) is True
        assert batch_out.past_key_values.has_previous_state(99) is False
        repeated = batch_out.past_key_values.clone()
        repeated_v_first = repeated._v_first.clone()
        assert repeated.batch_repeat_interleave(2) is repeated
        assert repeated.get_batch_size() == 4
        assert torch.equal(repeated._v_first, repeated_v_first.repeat_interleave(2, dim=0))
        repeated_select_source = repeated._v_first.clone()
        assert repeated.batch_select_indices(torch.tensor([3, 0], dtype=torch.long)) is repeated
        assert repeated.get_batch_size() == 2
        assert torch.equal(repeated._v_first, repeated_select_source.index_select(0, torch.tensor([3, 0])))
        assert repeated.rwkv7_cache_metrics()["batch_repeat_interleave_calls"] == 1
        assert repeated.rwkv7_cache_metrics()["batch_select_indices_calls"] == 1
        no_crop = batch_out.past_key_values.clone()
        assert no_crop.crop(no_crop.get_seq_length()) is no_crop
        assert no_crop.get_seq_length() == 3
        cleared_by_negative_crop = batch_out.past_key_values.clone()
        assert cleared_by_negative_crop.crop(-3) is cleared_by_negative_crop
        assert cleared_by_negative_crop.get_seq_length() == 0
        assert cleared_by_negative_crop.get_batch_size() is None
        positive_crop = batch_out.past_key_values.clone()
        try:
            positive_crop.crop(1)
        except NotImplementedError as exc:
            assert "shorter positive prefix" in str(exc)
        else:
            raise AssertionError("native recurrent cache should reject partial positive crop")
        try:
            batch_out.past_key_values.update(None, None, 0)
        except NotImplementedError as exc:
            assert "not a Transformer KV cache" in str(exc)
        else:
            raise AssertionError("native recurrent cache should reject KV update")
        unsupported_cache_calls = [
            (lambda: batch_out.past_key_values.update_recurrent_state(None, 0), "forward"),
            (lambda: batch_out.past_key_values.update_conv_state(None, 0), "convolution"),
            (lambda: batch_out.past_key_values.update_indexer(None, 0), "indexer"),
            (lambda: batch_out.past_key_values.early_initialization(2, 1, 1, torch.float32, torch.device("cpu")), "early-initialized"),
            (lambda: batch_out.past_key_values.offload(0), "offload"),
            (lambda: batch_out.past_key_values.prefetch(0), "restore"),
        ]
        for fn, expected in unsupported_cache_calls:
            try:
                fn()
            except NotImplementedError as exc:
                assert expected in str(exc)
            else:
                raise AssertionError(f"native recurrent cache should reject unsupported cache call: {expected}")
        reset = batch_out.past_key_values.clone()
        reset.reset()
        assert reset.get_seq_length() == 0
        assert reset.get_batch_size() is None
        assert reset.seen_tokens == 0
        assert reset.is_initialized is False
        assert reset.max_batch_size is None
        assert reset.max_cache_len == -1
        assert NativeRWKV7Cache.from_legacy_cache((), seen_tokens=0).get_seq_length() == 0
        assert NativeRWKV7Cache.from_legacy_cache(DynamicCache(config=model.config)).get_seq_length() == 0
        assert NativeRWKV7Cache.from_legacy_cache(batch_out.past_key_values) is batch_out.past_key_values
        legacy_cache = batch_out.past_key_values.to_legacy_cache()
        assert isinstance(legacy_cache, tuple)
        assert len(legacy_cache) == 4
        assert legacy_cache.get_seq_length() == 3
        assert legacy_cache.seen_tokens == 3
        legacy_cache.seen_tokens = 4
        assert legacy_cache.get_seq_length() == 4
        legacy_cache.seen_tokens = 3
        assert legacy_cache.get_seq_length(99) == 0
        round_trip_cache = NativeRWKV7Cache.from_legacy_cache(legacy_cache)
        assert round_trip_cache.get_seq_length() == 3
        assert round_trip_cache.get_batch_size() == 2
        explicit_round_trip_cache = NativeRWKV7Cache.from_legacy_cache(legacy_cache, seen_tokens=1)
        assert explicit_round_trip_cache.get_seq_length() == 3
        legacy_forward = model(batch_ids[:, :2], use_cache=True, return_legacy_cache=True)
        assert isinstance(legacy_forward.past_key_values, tuple)
        assert not isinstance(legacy_forward.past_key_values, NativeRWKV7Cache)
        assert legacy_forward.past_key_values.get_seq_length() == 2
        legacy_forward_decode = model(
            batch_ids[:, 2:3],
            past_key_values=legacy_forward.past_key_values,
            use_cache=True,
            return_legacy_cache=False,
        )
        assert isinstance(legacy_forward_decode.past_key_values, NativeRWKV7Cache)
        assert legacy_forward_decode.past_key_values.get_seq_length() == 3
        assert torch.allclose(legacy_forward_decode.logits, batch_out.logits[:, 2:3])
        legacy_tuple = model(batch_ids[:, :2], use_cache=True, return_legacy_cache=True, return_dict=False)
        assert len(legacy_tuple) == 2
        assert legacy_tuple[0].shape == (2, 2, model.config.vocab_size)
        assert isinstance(legacy_tuple[1], tuple)
        assert not isinstance(legacy_tuple[1], NativeRWKV7Cache)
        assert legacy_tuple[1].get_seq_length() == 2
        legacy_decode = model(torch.tensor([[7], [8]], dtype=torch.long), past_key_values=legacy_cache, use_cache=True)
        native_decode = model(
            torch.tensor([[7], [8]], dtype=torch.long),
            past_key_values=batch_out.past_key_values.clone(),
            use_cache=True,
        )
        base_legacy_decode = model.model(
            input_ids=torch.tensor([[7], [8]], dtype=torch.long),
            past_key_values=legacy_cache,
            use_cache=True,
        )
        assert torch.allclose(legacy_decode.logits, native_decode.logits)
        assert legacy_decode.past_key_values.get_seq_length() == 4
        assert base_legacy_decode.past_key_values.get_seq_length() == 4
        try:
            NativeRWKV7Cache.from_legacy_cache((legacy_cache[0], legacy_cache[1]))
        except TypeError:
            pass
        else:
            raise AssertionError("native cache should reject malformed legacy caches")
        original_state = batch_out.past_key_values._state[0].clone()
        model(
            torch.tensor([[7]], dtype=torch.long),
            past_key_values=batch_clone.select_batch(torch.tensor([0], dtype=torch.long), inplace=False),
            use_cache=True,
        )
        assert torch.equal(batch_out.past_key_values._state[0], original_state)
        flat_out = model(torch.tensor([7], dtype=torch.long), past_key_values=batch_out.past_key_values.select_batch(torch.tensor([0], dtype=torch.long), inplace=False), use_cache=True)
    assert flat_out.logits.shape[:2] == (1, 1)
    old_first = batch_out.past_key_values._v_first.clone()
    beam_idx = torch.tensor([1, 0], dtype=torch.long)
    legacy_reordered = model._reorder_cache(batch_out.past_key_values.to_legacy_cache(), beam_idx)
    assert isinstance(legacy_reordered, tuple)
    assert not isinstance(legacy_reordered, NativeRWKV7Cache)
    assert legacy_reordered.get_seq_length() == batch_out.past_key_values.get_seq_length()
    assert torch.equal(legacy_reordered[3], old_first.index_select(0, beam_idx))
    reordered = model._reorder_cache(batch_out.past_key_values, beam_idx)
    assert reordered is batch_out.past_key_values
    assert torch.equal(reordered._v_first, old_first.index_select(0, beam_idx))
    model.gradient_checkpointing_enable()
    assert getattr(model, "is_gradient_checkpointing", True)
    assert model._supports_default_dynamic_cache() is False

    # Some GenerationMixin versions pre-create a DynamicCache for unknown model
    # classes. Native RWKV recurrent state is not a KV cache, so an empty
    # DynamicCache must be treated as no cache and run a full prompt prefill.
    with torch.no_grad():
        empty_dynamic = DynamicCache(config=model.config)
        dyn_out = model(input_ids, past_key_values=empty_dynamic, use_cache=True)
        ref_out = model(input_ids, use_cache=True)
        base_dyn = model.model(input_ids=input_ids, past_key_values=DynamicCache(config=model.config), use_cache=True)
        base_ref = model.model(input_ids=input_ids, use_cache=True)
    assert dyn_out.logits.shape == ref_out.logits.shape == (1, input_ids.shape[1], model.config.vocab_size)
    assert torch.allclose(dyn_out.logits, ref_out.logits)
    assert dyn_out.past_key_values.get_seq_length() == input_ids.shape[1]
    assert torch.allclose(base_dyn.last_hidden_state, base_ref.last_hidden_state)
    assert model._reorder_cache(DynamicCache(config=model.config), torch.tensor([0], dtype=torch.long)) is None

    # Native mm8/mm4 output heads are module replacements without a dense
    # `.weight`; the native fallback must call the head module instead.
    wrapped_model = build_tiny_model()
    wrapped_model.lm_head = WrappedHead(wrapped_model.lm_head)
    with torch.no_grad():
        wrapped = wrapped_model(torch.tensor([[1, 2]], dtype=torch.long), use_cache=True)
        wrapped_decode = wrapped_model(
            torch.tensor([[3]], dtype=torch.long),
            past_key_values=wrapped.past_key_values,
            use_cache=True,
        )
    assert wrapped.logits.shape == (1, 2, wrapped_model.config.vocab_size)
    assert wrapped_decode.logits.shape == (1, 1, wrapped_model.config.vocab_size)

    with torch.no_grad():
        input_embeds = model.get_input_embeddings()(input_ids)
        embeds_only_out = model.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=2,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=None,
        )
        ids_and_embeds_out = model.generate(
            input_ids=input_ids,
            inputs_embeds=input_embeds,
            max_new_tokens=2,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=None,
        )
    assert embeds_only_out.shape == (1, 2), tuple(embeds_only_out.shape)
    assert ids_and_embeds_out.shape == (1, input_ids.shape[1] + 2), tuple(ids_and_embeds_out.shape)
    assert torch.equal(ids_and_embeds_out[:, : input_ids.shape[1]], input_ids)
    with torch.no_grad():
        token_type_out = model.generate(
            input_ids,
            token_type_ids=torch.zeros_like(input_ids),
            max_new_tokens=2,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=None,
        )
        token_type_embeds_out = model.generate(
            input_ids=input_ids,
            inputs_embeds=input_embeds,
            token_type_ids=torch.zeros_like(input_ids),
            max_new_tokens=1,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=None,
        )
    assert token_type_out.shape == (1, input_ids.shape[1] + 2), tuple(token_type_out.shape)
    assert token_type_embeds_out.shape == (1, input_ids.shape[1] + 1), tuple(token_type_embeds_out.shape)
    assert torch.equal(token_type_out[:, : input_ids.shape[1]], input_ids)
    assert torch.equal(token_type_embeds_out[:, : input_ids.shape[1]], input_ids)
    with torch.no_grad():
        head_mask_out = model.generate(
            input_ids,
            head_mask=torch.ones(model.config.num_hidden_layers, dtype=torch.float32),
            max_new_tokens=1,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=None,
        )
    assert head_mask_out.shape == (1, input_ids.shape[1] + 1), tuple(head_mask_out.shape)
    assert torch.equal(head_mask_out[:, : input_ids.shape[1]], input_ids)

    calls: list[tuple[tuple[int, ...], bool, bool, tuple[int, ...] | None]] = []
    original_forward = model.forward

    def counted_forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=None, **kwargs):
        mask_shape = tuple(attention_mask.shape) if isinstance(attention_mask, torch.Tensor) else None
        calls.append((tuple(input_ids.shape), past_key_values is not None, bool(use_cache), mask_shape))
        return original_forward(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

    model.forward = types.MethodType(counted_forward, model)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=3,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=None,
        )

    assert out.shape == (1, 7), tuple(out.shape)
    assert torch.equal(out[:, : input_ids.shape[1]], input_ids)
    assert calls, "generate should call forward"
    assert calls[0] == ((1, input_ids.shape[1]), False, True, (1, input_ids.shape[1])), calls
    assert all(shape == (1, 1) and has_cache and use_cache for shape, has_cache, use_cache, _ in calls[1:]), calls

    with torch.no_grad():
        legacy_flag_out = model.generate(
            input_ids,
            max_new_tokens=1,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=None,
            return_dict_in_generate=True,
            return_legacy_cache=True,
        )
        native_flag_out = model.generate(
            input_ids,
            max_new_tokens=1,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=None,
            return_dict_in_generate=True,
            return_legacy_cache=False,
        )
    assert legacy_flag_out.sequences.shape == native_flag_out.sequences.shape == (1, 5)
    assert isinstance(legacy_flag_out.past_key_values, tuple)
    assert hasattr(legacy_flag_out.past_key_values, "get_seq_length")
    assert isinstance(native_flag_out.past_key_values, NativeRWKV7Cache)
    assert hasattr(native_flag_out.past_key_values, "get_seq_length")

    with torch.no_grad():
        hidden_out = model.generate(
            input_ids,
            max_new_tokens=2,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=None,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
        )
    assert hidden_out.sequences.shape == (1, input_ids.shape[1] + 2)
    assert len(hidden_out.scores) == 2
    assert hidden_out.scores[0].shape == (1, model.config.vocab_size)
    assert len(hidden_out.hidden_states) == 2
    assert len(hidden_out.hidden_states[0]) == model.config.num_hidden_layers + 1
    assert hidden_out.hidden_states[0][-1].shape == (1, input_ids.shape[1], model.config.hidden_size)
    assert hidden_out.hidden_states[1][-1].shape == (1, 1, model.config.hidden_size)
    assert hidden_out.past_key_values.get_seq_length() == input_ids.shape[1] + 1
    with torch.no_grad():
        logits_out = model.generate(
            input_ids,
            max_new_tokens=2,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=None,
            return_dict_in_generate=True,
            output_logits=True,
        )
        no_cache_hidden_out = model.generate(
            input_ids,
            max_new_tokens=2,
            do_sample=False,
            use_cache=False,
            pad_token_id=0,
            eos_token_id=None,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
        )
    assert len(logits_out.logits) == 2
    assert logits_out.logits[0].shape == (1, model.config.vocab_size)
    assert logits_out.scores is None
    assert logits_out.past_key_values.get_seq_length() == input_ids.shape[1] + 1
    assert no_cache_hidden_out.past_key_values is None
    assert len(no_cache_hidden_out.scores) == 2
    assert no_cache_hidden_out.hidden_states[0][-1].shape == (1, input_ids.shape[1], model.config.hidden_size)
    assert no_cache_hidden_out.hidden_states[1][-1].shape == (1, input_ids.shape[1] + 1, model.config.hidden_size)
    try:
        model.generate(
            input_ids,
            max_new_tokens=1,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=None,
            output_attentions=True,
        )
    except NotImplementedError as exc:
        assert "attention maps" in str(exc)
    else:
        raise AssertionError("generate(output_attentions=True) should be rejected explicitly")

    with torch.no_grad():
        beam_out = model.generate(
            input_ids,
            max_new_tokens=2,
            num_beams=2,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=None,
        )
        legacy_beam_out = model.generate(
            input_ids,
            max_new_tokens=1,
            num_beams=2,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=None,
            return_dict_in_generate=True,
            return_legacy_cache=True,
        )
    assert beam_out.shape == (1, 6), tuple(beam_out.shape)
    assert torch.equal(beam_out[:, : input_ids.shape[1]], input_ids)
    assert legacy_beam_out.sequences.shape == (1, 5), tuple(legacy_beam_out.sequences.shape)
    assert isinstance(legacy_beam_out.past_key_values, tuple)
    assert hasattr(legacy_beam_out.past_key_values, "get_seq_length")

    padded_ids = torch.tensor([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=torch.long)
    padded_mask = torch.tensor([[0, 1, 1, 1], [0, 1, 1, 1]], dtype=torch.long)
    calls.clear()
    with torch.no_grad():
        padded_out = model.generate(
            padded_ids,
            attention_mask=padded_mask,
            max_new_tokens=2,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=None,
        )
    assert padded_out.shape == (2, 6), tuple(padded_out.shape)
    assert torch.equal(padded_out[:, : padded_ids.shape[1]], padded_ids)
    assert calls[0] == ((2, padded_ids.shape[1]), False, True, tuple(padded_mask.shape)), calls
    assert all(shape == (2, 1) and has_cache and use_cache for shape, has_cache, use_cache, _ in calls[1:]), calls
    assert all(mask_shape is None or mask_shape[0] == 2 for *_, mask_shape in calls), calls
    print("NATIVE CPU GENERATE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
