from __future__ import annotations

import inspect
import os
from importlib.metadata import version

import torch
from packaging.version import Version
from peft import LoraConfig, get_peft_model
from trl import DPOConfig, DPOTrainer, GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

from rwkv7_hf.native_model import NativeRWKV7Config, NativeRWKV7ForCausalLM


SUPPORTED_RANGES = {
    "transformers": (Version("5.12.1"), Version("6")),
    "peft": (Version("0.19.1"), Version("1")),
    "trl": (Version("1.7"), Version("2")),
}


def test_resolved_hf_ecosystem_versions_are_inside_the_declared_range() -> None:
    resolved = {name: Version(version(name)) for name in SUPPORTED_RANGES}
    for name, actual in resolved.items():
        minimum, maximum = SUPPORTED_RANGES[name]
        assert minimum <= actual < maximum, (name, actual, minimum, maximum)

    if os.environ.get("RWKV7_HF_COMPAT_LANE") == "minimum":
        assert resolved == {
            "transformers": Version("5.12.1"),
            "peft": Version("0.19.1"),
            "trl": Version("1.7.0"),
        }


def _tiny_model() -> NativeRWKV7ForCausalLM:
    return NativeRWKV7ForCausalLM(
        NativeRWKV7Config(
            vocab_size=23,
            hidden_size=8,
            num_hidden_layers=2,
            head_dim=4,
            intermediate_size=16,
            decay_low_rank_dim=3,
            gate_low_rank_dim=3,
            a_low_rank_dim=3,
            v_low_rank_dim=3,
            use_cache=False,
        )
    )


def test_peft_lora_wraps_and_backpropagates_through_the_native_hf_model() -> None:
    model = get_peft_model(
        _tiny_model(),
        LoraConfig(
            task_type="CAUSAL_LM",
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            bias="none",
            target_modules=["r_proj", "k_proj", "v_proj", "o_proj", "key", "value"],
        ),
    )
    ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)
    output = model(input_ids=ids, labels=ids, use_cache=False)
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_trl_sft_dpo_grpo_public_constructor_contract_is_available() -> None:
    pairs = (
        (SFTTrainer, SFTConfig),
        (DPOTrainer, DPOConfig),
        (GRPOTrainer, GRPOConfig),
    )
    for trainer, config in pairs:
        parameters = inspect.signature(trainer.__init__).parameters
        assert {"model", "args", "train_dataset"} <= set(parameters)
        assert inspect.isclass(config)
