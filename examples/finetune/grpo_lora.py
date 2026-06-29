#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

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

DATASET = "openai/gsm8k"
REVISION = "740312add88f781978c0658806c59bc2815b9866"


def final_answer(value: str) -> str:
    match = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", value)
    return match.group(1).replace(",", "") if match else ""


def render(example):
    return {
        "prompt": [
            {
                "role": "user",
                "content": example["question"] + "\nShow your work and finish with #### answer.",
            }
        ],
        "answer": final_answer(example["answer"]),
    }


def correctness_reward(completions, answer, **kwargs):
    del kwargs
    rewards = []
    for completion, expected in zip(completions, answer):
        if isinstance(completion, list):
            completion = completion[-1]["content"]
        rewards.append(1.0 if final_answer(str(completion)) == str(expected) else 0.0)
    return rewards


def main():
    parser = argparse.ArgumentParser(description="RWKV-7 LoRA GRPO example")
    common_arguments(parser)
    args = parser.parse_args()
    output = prepare_run(args, DATASET, REVISION)
    train = load_dataset(DATASET, "main", revision=REVISION, split="train").map(render)
    evaluation = load_dataset(DATASET, "main", revision=REVISION, split="test").map(render)
    train = deterministic_subset(train, args.train_samples, args.seed, output, "train")
    evaluation = deterministic_subset(evaluation, args.eval_samples, args.seed, output, "eval")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.model_revision, trust_remote_code=True
    )
    max_prompt_length = args.max_length // 2

    def truncate_prompt(example):
        content = example["prompt"][0]["content"]
        token_ids = tokenizer.encode(content, add_special_tokens=False)
        example["prompt"][0]["content"] = tokenizer.decode(
            token_ids[:max_prompt_length], skip_special_tokens=False
        )
        return example

    train = train.map(truncate_prompt)
    evaluation = evaluation.map(truncate_prompt)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.model_revision, trust_remote_code=True
    )
    model.config.use_cache = False
    callback = ReproCallback(output)
    config = GRPOConfig(
        output_dir=str(output),
        seed=args.seed,
        max_completion_length=args.max_length // 2,
        max_steps=args.max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_generations=2,
        gradient_checkpointing=True,
        save_steps=25,
        logging_steps=1,
        report_to=report_target(args),
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[correctness_reward],
        args=config,
        train_dataset=train,
        eval_dataset=evaluation,
        processing_class=tokenizer,
        peft_config=lora_config(),
        callbacks=[callback],
    )
    before = snapshot_trainable(trainer.model)
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    adapter_dir = output / "adapter-final"
    trainer.save_model(adapter_dir)
    trainer.save_metrics("train", result.metrics)
    trainer.save_metrics("eval", trainer.evaluate())
    callback.write_status(trainer.state.global_step)
    validate_resume(args.resume_from_checkpoint, trainer.state.global_step, output)
    validate_parameter_change(trainer.model, before, output)
    validate_adapter_reload(
        trainer.model,
        model_id=args.model,
        model_revision=args.model_revision,
        adapter_dir=adapter_dir,
        output=output,
    )
    record_wandb(output)
    checkpoint_inventory(output)


if __name__ == "__main__":
    raise SystemExit(run_captured(main))
