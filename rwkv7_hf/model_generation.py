# coding=utf-8
"""Hugging Face generation input and recurrent-cache contracts for RWKV-7."""
from __future__ import annotations

import torch

from .model_cache import (
    NativeRWKV7Cache,
    _cache_seen,
    _native_cache_tuple_or_none,
    _native_last_token_slice,
)
from .recurrent_state import recurrent_state_layout_of


class _NativeGenerationContractMixin:
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx: torch.LongTensor):
        """Beam/select helper for batched native recurrent caches."""
        native_cache = _native_cache_tuple_or_none(past_key_values)
        if native_cache is None:
            return None
        if hasattr(past_key_values, "reorder_cache"):
            return past_key_values.reorder_cache(beam_idx)
        state, xpa, xpf, v_first = native_cache
        index = beam_idx.to(v_first.device)
        seen = _cache_seen(past_key_values)
        reordered = NativeRWKV7Cache(
            [s.index_select(0, index.to(s.device)) for s in state],
            [x.index_select(0, index.to(x.device)) for x in xpa],
            [x.index_select(0, index.to(x.device)) for x in xpf],
            v_first.index_select(0, index),
            seen_tokens=seen,
            state_layout=recurrent_state_layout_of(native_cache),
        )
        return reordered.to_legacy_cache() if isinstance(past_key_values, tuple) else reordered

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds: torch.Tensor | None = None,
        token_type_ids=None,
        head_mask=None,
        return_legacy_cache: bool | None = None,
        **kwargs,
    ):
        # Ensure GenerationMixin gets a cache on the first step. Earlier H1 code
        # only enabled cache after a cache already existed, causing full-prefix
        # recomputation on every greedy token.
        native_cache = _native_cache_tuple_or_none(past_key_values)
        model_inputs = {}
        if native_cache is not None:
            if input_ids is not None:
                model_inputs["input_ids"] = _native_last_token_slice(input_ids)
            elif inputs_embeds is not None:
                model_inputs["inputs_embeds"] = _native_last_token_slice(inputs_embeds)
            else:
                model_inputs["input_ids"] = input_ids
        elif inputs_embeds is not None:
            model_inputs["inputs_embeds"] = inputs_embeds
        else:
            model_inputs["input_ids"] = input_ids
        use_cache = kwargs.get("use_cache", True)
        if use_cache is None:
            use_cache = True
        model_inputs["past_key_values"] = past_key_values
        model_inputs["use_cache"] = use_cache
        if return_legacy_cache is not None:
            model_inputs["return_legacy_cache"] = return_legacy_cache
        if head_mask is not None:
            model_inputs["head_mask"] = head_mask
        if token_type_ids is not None:
            if native_cache is not None:
                token_type_ids = _native_last_token_slice(token_type_ids)
            model_inputs["token_type_ids"] = token_type_ids
        if kwargs.get("attention_mask") is not None:
            attention_mask = kwargs["attention_mask"]
            model_inputs["attention_mask"] = _native_last_token_slice(attention_mask) if native_cache is not None else attention_mask
        if "logits_to_keep" in kwargs:
            model_inputs["logits_to_keep"] = kwargs["logits_to_keep"]
        if "num_logits_to_keep" in kwargs:
            model_inputs["num_logits_to_keep"] = kwargs["num_logits_to_keep"]
        if "output_hidden_states" in kwargs:
            model_inputs["output_hidden_states"] = kwargs["output_hidden_states"]
        if "output_attentions" in kwargs:
            model_inputs["output_attentions"] = kwargs["output_attentions"]
        if "return_dict" in kwargs:
            model_inputs["return_dict"] = kwargs["return_dict"]
        if "position_ids" in kwargs:
            position_ids = kwargs["position_ids"]
            if native_cache is not None:
                position_ids = _native_last_token_slice(position_ids)
            model_inputs["position_ids"] = position_ids
        if "cache_position" in kwargs:
            cache_position = kwargs["cache_position"]
            if native_cache is not None:
                cache_position = _native_last_token_slice(cache_position)
            model_inputs["cache_position"] = cache_position
        return model_inputs
