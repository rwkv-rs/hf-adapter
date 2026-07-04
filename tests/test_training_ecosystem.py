from __future__ import annotations

from pathlib import Path

import torch
import pytest
import transformers
from packaging.version import Version


if Version(transformers.__version__) < Version("4.56.2"):
    pytest.skip(
        "the pinned TRL training stack requires Transformers >=4.56.2",
        allow_module_level=True,
    )


datasets = pytest.importorskip("datasets")
peft = pytest.importorskip("peft")
trl = pytest.importorskip("trl")
tokenizers = pytest.importorskip("tokenizers")

from transformers import PreTrainedTokenizerFast, default_data_collator

from rwkv7_hf.modeling_rwkv7 import RWKV7ForCausalLM


def test_grpo_example_disables_unselected_vllm_before_trainer_import():
    source = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "finetune"
        / "grpo_lora.py"
    ).read_text(encoding="utf-8")
    disable = source.index("trl_import_utils._vllm_available = False")
    trainer_import = source.index("from trl import GRPOConfig, GRPOTrainer")
    assert disable < trainer_import


def test_peft_lora_and_trl_sft_one_step(tmp_path, tiny_config):
    raw = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {"<pad>": 0, "<eos>": 1, "<unk>": 2, "a": 3, "b": 4},
            unk_token="<unk>",
        )
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    model = RWKV7ForCausalLM(tiny_config)
    lora = peft.get_peft_model(
        model,
        peft.LoraConfig(
            task_type=peft.TaskType.CAUSAL_LM,
            r=2,
            lora_alpha=4,
            target_modules=["r_proj", "k_proj", "v_proj", "o_proj", "key", "value"],
        ),
    )
    dataset = datasets.Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3, 4], [4, 3, 2, 1]],
            "labels": [[1, 2, 3, 4], [4, 3, 2, 1]],
            "attention_mask": [[1, 1, 1, 1], [1, 1, 1, 1]],
        }
    )
    args = trl.SFTConfig(
        output_dir=str(tmp_path),
        use_cpu=True,
        bf16=False,
        fp16=False,
        max_steps=1,
        per_device_train_batch_size=2,
        report_to="none",
        save_strategy="no",
        disable_tqdm=True,
        gradient_checkpointing=False,
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
    )
    trainer = trl.SFTTrainer(
        model=lora,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=default_data_collator,
    )
    before = {
        name: parameter.detach().clone()
        for name, parameter in trainer.model.named_parameters()
        if parameter.requires_grad
    }
    result = trainer.train()
    assert result.global_step == 1
    assert torch.isfinite(torch.tensor(result.training_loss))
    assert any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in trainer.model.named_parameters()
        if name in before
    )
