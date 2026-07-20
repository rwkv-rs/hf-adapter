#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# The canonical example deliberately uses Transformers generation, not vLLM.
# TRL 0.20 probes any system-wide vLLM installation while importing
# GRPOTrainer and then imports private vLLM symbols even when use_vllm=False.
# A stale unrelated vLLM package can therefore break this example before the
# arguments are parsed. Mark that optional integration unavailable before
# TRL's lazy trainer import; the selected native generation path is unchanged.
from trl import import_utils as trl_import_utils

trl_import_utils._vllm_available = False
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

from common import (  # noqa: E402
    ReproCallback,
    attach_lora_adapters,
    checkpoint_inventory,
    common_arguments,
    deterministic_subset,
    gradient_checkpointing_kwargs,
    model_load_kwargs,
    prepare_run,
    report_target,
    run_captured,
    record_wandb,
    snapshot_trainable,
    trainer_precision_flags,
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


def format_reward(completions, **kwargs):
    """Small dense reward that makes the cold-start example trainable.

    Exact GSM8K correctness remains the primary reward.  A randomly initialized
    or very small base model can otherwise receive an all-zero group forever,
    so this auxiliary term rewards non-empty, lexically varied reasoning and
    the requested numeric answer format.
    """

    del kwargs
    rewards = []
    for completion in completions:
        if isinstance(completion, list):
            completion = completion[-1]["content"]
        text = str(completion).strip()
        words = set(text.split())
        # Do not saturate the lexical term at the short completion length: a
        # saturated reward gives every sampled answer the same value and GRPO
        # then has zero within-group advantage.  These small dense terms favor
        # varied reasoning, numeric work, and mathematical structure while the
        # exact-answer reward remains dominant.
        score = min(len(words), 100) / 1000.0
        score += min(len(set(text)), 100) / 10000.0
        score += min(sum(character.isdigit() for character in text), 20) / 1000.0
        score += min(sum(character in "+-*/=" for character in text), 20) / 2000.0
        if "####" in text:
            score += 0.1
        if final_answer(text):
            score += 0.1
        rewards.append(score)
    return rewards


def main():
    parser = argparse.ArgumentParser(description="RWKV-7 LoRA GRPO example")
    common_arguments(parser)
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=64,
        help="Maximum sampled answer length within --max-length",
    )
    args = parser.parse_args()
    if not 0 < args.max_completion_length < args.max_length:
        parser.error("--max-completion-length must be between 1 and --max-length - 1")
    output = prepare_run(args, DATASET, REVISION)
    train = load_dataset(DATASET, "main", revision=REVISION, split="train")
    evaluation = load_dataset(DATASET, "main", revision=REVISION, split="test")
    train = deterministic_subset(train, args.train_samples, args.seed, output, "train")
    evaluation = deterministic_subset(evaluation, args.eval_samples, args.seed, output, "eval")
    train = train.map(render, remove_columns=train.column_names)
    evaluation = evaluation.map(render, remove_columns=evaluation.column_names)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.model_revision, trust_remote_code=True
    )
    max_prompt_length = args.max_length - args.max_completion_length

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
        args.model,
        **model_load_kwargs(args),
    )
    model = attach_lora_adapters(model, args)
    model.config.use_cache = False
    callback = ReproCallback(output)
    config = GRPOConfig(
        output_dir=str(output),
        seed=args.seed,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=2,
        # GRPO's sampled rollout is inference, so use the canonical recurrent
        # cache there.  The model config remains use_cache=False for the
        # gradient-checkpointed policy-loss forward/backward below.
        generation_kwargs={"use_cache": True},
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs(),
        save_steps=25,
        logging_steps=1,
        report_to=report_target(args),
        **trainer_precision_flags(),
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[correctness_reward, format_reward],
        args=config,
        train_dataset=train,
        eval_dataset=evaluation,
        processing_class=tokenizer,
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
