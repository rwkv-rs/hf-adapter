from __future__ import annotations

import os
from importlib.metadata import version
from pathlib import Path

import torch
from datasets import Dataset
from packaging.version import Version
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from trl import DPOConfig, DPOTrainer, GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer

from rwkv7_hf.native_model import NativeRWKV7Config, NativeRWKV7ForCausalLM


ROOT = Path(__file__).resolve().parents[1]


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


def test_compat_lane_imports_the_installed_wheel() -> None:
    if os.environ.get("RWKV7_EXPECT_INSTALLED") != "1":
        return
    import rwkv7_hf

    module_path = Path(rwkv7_hf.__file__).resolve()
    assert not module_path.is_relative_to(ROOT), (
        f"compatibility lane imported the source checkout instead of the wheel: {module_path}"
    )


def _tiny_model() -> NativeRWKV7ForCausalLM:
    return NativeRWKV7ForCausalLM(
        NativeRWKV7Config(
            vocab_size=32,
            hidden_size=8,
            num_hidden_layers=2,
            head_dim=4,
            intermediate_size=16,
            decay_low_rank_dim=3,
            gate_low_rank_dim=3,
            a_low_rank_dim=3,
            v_low_rank_dim=3,
            use_cache=False,
            pad_token_id=0,
            eos_token_id=1,
            bos_token_id=1,
        )
    )


def _tiny_tokenizer() -> PreTrainedTokenizerFast:
    vocabulary = {"[PAD]": 0, "[EOS]": 1, "[UNK]": 2}
    vocabulary.update(
        {
            token: index
            for index, token in enumerate(
                ["hello", "world", "rwkv", "is", "fast", "good", "bad", "tiny"],
                start=3,
            )
        }
    )
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        model_max_length=16,
    )


def _trainable_snapshot(model) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _assert_finite_step_updated(trainer) -> None:
    before = _trainable_snapshot(trainer.model)
    result = trainer.train()
    after = _trainable_snapshot(trainer.model)
    assert torch.isfinite(torch.tensor(float(result.training_loss)))
    assert any(
        not torch.equal(before[name], after[name])
        for name in before.keys() & after.keys()
    ), f"{type(trainer).__name__} completed without updating a trainable parameter"


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


def test_native_model_exposes_a_transformers_tensor_parallel_plan() -> None:
    model = _tiny_model()
    assert model.supports_tp_plan
    assert model.tp_plan == {
        "model.embeddings": "embedding_rowwise",
        "model.layers.*.attn.r_proj": "colwise_gather_output",
        "model.layers.*.attn.k_proj": "colwise_gather_output",
        "model.layers.*.attn.v_proj": "colwise_gather_output",
        "model.layers.*.attn.o_proj": "rowwise_split_input",
        "model.layers.*.attn.w_lora.lora.*": "colwise_gather_output",
        "model.layers.*.attn.a_lora.lora.*": "colwise_gather_output",
        "model.layers.*.attn.g_lora.lora.*": "colwise_gather_output",
        "model.layers.*.attn.v_lora.lora.*": "colwise_gather_output",
        "model.layers.*.ffn.key": "colwise",
        "model.layers.*.ffn.value": "rowwise",
        "lm_head": "colwise_gather_output",
    }
    model._tp_size = 2
    assert model._rwkv7_has_tensor_parallel()
    # Packed single-device kernels read raw full matrices and must fail closed
    # once Transformers owns sharded modules and communication hooks.
    assert model._native_jit_packs() is None


def test_trl_sft_runs_a_real_tiny_optimizer_step(tmp_path) -> None:
    trainer = SFTTrainer(
        model=_tiny_model(),
        args=SFTConfig(
            output_dir=str(tmp_path / "sft"),
            max_steps=1,
            per_device_train_batch_size=1,
            learning_rate=1e-3,
            logging_strategy="no",
            save_strategy="no",
            report_to=[],
            disable_tqdm=True,
            use_cpu=True,
            bf16=False,
            fp16=False,
            dataloader_pin_memory=False,
            gradient_checkpointing=False,
            max_length=8,
            dataset_text_field="text",
            packing=False,
            loss_type="nll",
        ),
        train_dataset=Dataset.from_dict({"text": ["hello world", "rwkv is fast"]}),
        processing_class=_tiny_tokenizer(),
    )
    _assert_finite_step_updated(trainer)


def test_trl_dpo_runs_a_real_tiny_optimizer_step(tmp_path) -> None:
    trainer = DPOTrainer(
        model=_tiny_model(),
        ref_model=_tiny_model(),
        args=DPOConfig(
            output_dir=str(tmp_path / "dpo"),
            max_steps=1,
            per_device_train_batch_size=1,
            learning_rate=1e-3,
            logging_strategy="no",
            save_strategy="no",
            report_to=[],
            disable_tqdm=True,
            use_cpu=True,
            bf16=False,
            fp16=False,
            dataloader_pin_memory=False,
            gradient_checkpointing=False,
            remove_unused_columns=False,
            max_length=8,
        ),
        train_dataset=Dataset.from_dict(
            {
                "prompt": ["rwkv is", "hello"],
                "chosen": [" fast", " world"],
                "rejected": [" bad", " bad"],
            }
        ),
        processing_class=_tiny_tokenizer(),
    )
    _assert_finite_step_updated(trainer)


def test_trl_grpo_runs_a_real_tiny_optimizer_step(tmp_path) -> None:
    def reward(completions, **kwargs):
        del kwargs
        return [float(index % 2) for index, _ in enumerate(completions)]

    trainer = GRPOTrainer(
        model=_tiny_model(),
        reward_funcs=reward,
        args=GRPOConfig(
            output_dir=str(tmp_path / "grpo"),
            max_steps=1,
            per_device_train_batch_size=2,
            learning_rate=1e-3,
            logging_strategy="no",
            save_strategy="no",
            report_to=[],
            disable_tqdm=True,
            use_cpu=True,
            bf16=False,
            fp16=False,
            dataloader_pin_memory=False,
            gradient_checkpointing=False,
            remove_unused_columns=False,
            max_completion_length=2,
            num_generations=2,
            generation_batch_size=2,
        ),
        train_dataset=Dataset.from_dict(
            {"prompt": ["hello", "rwkv", "tiny", "world"]}
        ),
        processing_class=_tiny_tokenizer(),
    )
    _assert_finite_step_updated(trainer)
