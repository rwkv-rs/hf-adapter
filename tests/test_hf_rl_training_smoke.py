#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import tempfile
from typing import Any

# FLA backward can hit Dynamo/Triton issues on the V100 test box.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
from datasets import Dataset as HFDataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPTS = [
    "User: Say hello.\n\nAssistant:",
    "User: Count to two.\n\nAssistant:",
]


def ensure_trl_fsdp_compat() -> None:
    """Patch older torch builds for newer TRL imports when needed."""
    try:
        import torch.distributed.fsdp as fsdp
    except Exception:
        return
    if not hasattr(fsdp, "FSDPModule"):
        class FSDPModule:  # pragma: no cover - import shim only
            pass
        fsdp.FSDPModule = FSDPModule


def load_base_model(model_path: str, device: str, attn_mode: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        device_map=device if device.startswith("cuda") else None,
    )
    model.config.use_cache = False
    model.config.fuse_cross_entropy = False
    model.config.use_l2warp = False
    model.config.attn_mode = attn_mode
    for layer in getattr(model.model, "layers", []):
        attn = getattr(layer, "attn", None)
        if hasattr(attn, "mode"):
            attn.mode = attn_mode
    return model


def lora_config() -> LoraConfig:
    return LoraConfig(
        task_type="CAUSAL_LM",
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["r_proj", "k_proj", "v_proj", "o_proj", "key", "value"],
    )


def run_dpo(args: argparse.Namespace) -> float:
    ensure_trl_fsdp_compat()
    from trl import DPOConfig, DPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_base_model(args.model, args.device, args.attn_mode)
    dataset = HFDataset.from_dict(
        {
            "prompt": PROMPTS,
            "chosen": [" Hello!", " one two."],
            "rejected": [" Goodbye.", " three four."],
        }
    )
    with tempfile.TemporaryDirectory(prefix="rwkv7_dpo_smoke_") as out_dir:
        train_args = DPOConfig(
            output_dir=out_dir,
            max_steps=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            fp16=args.device.startswith("cuda"),
            bf16=False,
            gradient_checkpointing=False,
            optim="adamw_torch",
            remove_unused_columns=False,
            max_length=args.max_length,
        )
        trainer = DPOTrainer(
            model=model,
            args=train_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=lora_config(),
        )
        result = trainer.train()
    loss = float(result.training_loss)
    assert math.isfinite(loss), result.training_loss
    print("trl_dpo_train_loss", loss)
    return loss


def reward_func(prompts: list[Any], completions: list[Any], **_: Any) -> list[float]:
    # Constant rewards keep this smoke deterministic and focused on trainer/model
    # compatibility rather than reward modeling quality.
    return [1.0 for _ in completions]


def run_grpo(args: argparse.Namespace) -> float:
    ensure_trl_fsdp_compat()
    from trl import GRPOConfig, GRPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_base_model(args.model, args.device, args.attn_mode)
    dataset = HFDataset.from_dict({"prompt": PROMPTS})
    with tempfile.TemporaryDirectory(prefix="rwkv7_grpo_smoke_") as out_dir:
        train_args = GRPOConfig(
            output_dir=out_dir,
            max_steps=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            fp16=args.device.startswith("cuda"),
            bf16=False,
            gradient_checkpointing=False,
            optim="adamw_torch",
            remove_unused_columns=False,
            max_completion_length=args.grpo_max_completion_length,
            num_generations=2,
            generation_batch_size=2,
        )
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_func,
            args=train_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=lora_config(),
        )
        result = trainer.train()
    loss = float(result.training_loss)
    assert math.isfinite(loss), result.training_loss
    print("trl_grpo_train_loss", loss)
    return loss


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-mode", default="fused_recurrent", choices=["chunk", "fused_recurrent"])
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--grpo-max-completion-length", type=int, default=2)
    ap.add_argument("--backend", choices=["dpo", "grpo", "both"], default="both")
    args = ap.parse_args()
    if args.backend in {"dpo", "both"}:
        run_dpo(args)
    if args.backend in {"grpo", "both"}:
        run_grpo(args)
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
