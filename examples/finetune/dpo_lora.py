#!/usr/bin/env python3
from __future__ import annotations

import argparse

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from common import (
    ReproCallback,
    checkpoint_inventory,
    common_arguments,
    deterministic_subset,
    lora_config,
    prepare_run,
    report_target,
    run_captured,
    record_wandb,
    snapshot_trainable,
    validate_adapter_reload,
    validate_parameter_change,
    validate_resume,
)

DATASET = "HuggingFaceH4/ultrafeedback_binarized"
REVISION = "3949bf5f8c17c394422ccfab0c31ea9c20bdeb85"


def text(messages):
    return "\n".join(
        ("User" if row["role"] == "user" else "Assistant")
        + ": "
        + row["content"]
        for row in messages
    )


def render(example):
    return {
        "prompt": text(example["chosen"][:-1]),
        "chosen": example["chosen"][-1]["content"],
        "rejected": example["rejected"][-1]["content"],
    }


def main():
    parser = argparse.ArgumentParser(description="RWKV-7 LoRA DPO example")
    common_arguments(parser)
    args = parser.parse_args()
    output = prepare_run(args, DATASET, REVISION)
    train = load_dataset(DATASET, revision=REVISION, split="train_prefs")
    evaluation = load_dataset(DATASET, revision=REVISION, split="test_prefs")
    train = deterministic_subset(train, args.train_samples, args.seed, output, "train")
    evaluation = deterministic_subset(evaluation, args.eval_samples, args.seed, output, "eval")
    train = train.map(render, remove_columns=train.column_names)
    evaluation = evaluation.map(render, remove_columns=evaluation.column_names)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.model_revision, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.model_revision, trust_remote_code=True
    )
    model.config.use_cache = False
    callback = ReproCallback(output)
    config = DPOConfig(
        output_dir=str(output),
        seed=args.seed,
        max_length=args.max_length,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        eval_strategy="no",
        save_steps=25,
        logging_steps=1,
        report_to=report_target(args),
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=config,
        train_dataset=train,
        eval_dataset=evaluation,
        processing_class=tokenizer,
        peft_config=lora_config(),
        callbacks=[callback],
    )
    before = snapshot_trainable(trainer.model)
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trained_model = trainer.accelerator.unwrap_model(
        trainer.model, keep_fp32_wrapper=False
    )
    adapter_dir = output / "adapter-final"
    trained_model.save_pretrained(adapter_dir)
    trainer.save_metrics("train", result.metrics)
    trainer.save_metrics("eval", trainer.evaluate())
    callback.write_status(trainer.state.global_step)
    validate_resume(args.resume_from_checkpoint, trainer.state.global_step, output)
    validate_parameter_change(trained_model, before, output)
    validate_adapter_reload(
        trained_model,
        model_id=args.model,
        model_revision=args.model_revision,
        adapter_dir=adapter_dir,
        output=output,
    )
    record_wandb(output)
    checkpoint_inventory(output)


if __name__ == "__main__":
    raise SystemExit(run_captured(main))
