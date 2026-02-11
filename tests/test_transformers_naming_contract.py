#!/usr/bin/env python3
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from rwkv7_hf.configuration_rwkv7 import RWKV7Config
from rwkv7_hf.model_config import NativeRWKV7Config
from rwkv7_hf.modeling_rwkv7 import (
    RWKV7ForCausalLM,
    _RWKV7ForCausalLM as _FLARWKV7ForCausalLM,
    _resolve_wrapper_logits_to_keep,
)


EXPECTED_FORWARD_PARAMETERS = (
    "self",
    "input_ids",
    "attention_mask",
    "inputs_embeds",
    "past_key_values",
    "labels",
    "shift_labels",
    "use_cache",
    "output_attentions",
    "output_hidden_states",
    "return_dict",
    "logits_to_keep",
    "position_ids",
    "cache_position",
    "kwargs",
)


def test_optimized_wrapper_exposes_transformers_forward_signature() -> None:
    parameters = inspect.signature(RWKV7ForCausalLM.forward).parameters
    assert tuple(parameters) == EXPECTED_FORWARD_PARAMETERS
    assert parameters["input_ids"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["labels"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert parameters["kwargs"].kind is inspect.Parameter.VAR_KEYWORD
    assert not any(parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in parameters.values())


def test_deprecated_logits_keyword_compares_tensor_values_safely() -> None:
    selector = torch.tensor([1, 3])
    assert _resolve_wrapper_logits_to_keep(selector, selector.clone()) is selector

    with pytest.raises(ValueError, match="logits_to_keep and num_logits_to_keep must match"):
        _resolve_wrapper_logits_to_keep(torch.tensor([1, 3]), torch.tensor([0, 2]))


def test_explicit_forward_preserves_fla_keyword_dispatch(monkeypatch) -> None:
    captured: dict[str, object] = {}
    sentinel = object()

    def fake_fla_forward(_self, **kwargs):
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(_FLARWKV7ForCausalLM, "forward", fake_fla_forward)
    model = object.__new__(RWKV7ForCausalLM)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(use_cache=False)

    input_ids = torch.tensor([[1, 2]])
    position_ids = torch.tensor([[0, 1]])
    cache_position = torch.tensor([0, 1])
    output = model(
        input_ids=input_ids,
        use_cache=False,
        num_logits_to_keep=1,
        position_ids=position_ids,
        cache_position=cache_position,
    )

    assert output is sentinel
    assert captured["input_ids"] is input_ids
    assert captured["logits_to_keep"] == 1
    assert captured["position_ids"] is position_ids
    assert captured["cache_position"] is cache_position


@pytest.mark.parametrize("config_type", [NativeRWKV7Config, RWKV7Config])
@pytest.mark.parametrize(
    "head_kwargs",
    [{"num_heads": 4}, {"num_attention_heads": 4}],
    ids=["rwkv-name", "transformers-name"],
)
def test_head_count_aliases_match_and_round_trip(config_type, head_kwargs) -> None:
    config = config_type(
        hidden_size=16,
        attention_hidden_size=16,
        head_dim=4,
        num_hidden_layers=1,
        intermediate_size=32,
        **head_kwargs,
    )
    assert config.num_heads == 4
    assert config.num_attention_heads == 4

    serialized = config.to_dict()
    assert serialized["num_heads"] == 4
    assert serialized["num_attention_heads"] == 4
    restored = config_type.from_dict(serialized)
    assert restored.num_heads == 4
    assert restored.num_attention_heads == 4


@pytest.mark.parametrize("config_type", [NativeRWKV7Config, RWKV7Config])
def test_head_count_aliases_reject_conflicts(config_type) -> None:
    with pytest.raises(ValueError, match="num_heads and num_attention_heads must match"):
        config_type(
            hidden_size=16,
            attention_hidden_size=16,
            head_dim=4,
            num_heads=4,
            num_attention_heads=2,
            num_hidden_layers=1,
            intermediate_size=32,
        )
