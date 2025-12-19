#!/usr/bin/env python3
# coding=utf-8
"""TRL GRPO smoke for NativeRWKV7ForCausalLM (fla-free RL training path).

GRPO needs generation during training (rollouts) + reward + advantage, so it
exercises native generate() inside the training loop -- the last piece of the
bounty PEFT+RL requirement on the fla-free path (DPO is covered by
test_native_dpo_smoke.py). FLA-backed GRPO fails on cards where FLA backward
is blocked (Blackwell sm_120); native unblocks it.

Gate: finite GRPO loss AND trainable (LoRA) parameters update.

  python tests/test_native_grpo_smoke.py --model <hf_dir>
"""
from __future__ import annotations

import argparse
import math
import tempfile
from typing import Any

import torch
from datasets import Dataset as HFDataset
from peft import LoraConfig
from transformers import AutoTokenizer

# Keep this smoke independent of a partially-installed DeepSpeed package and of
# TRL importing the newer torch FSDPModule symbol on older torch (mirrors the
# FLA rl-smoke shims).
try:
    import accelerate.utils.other as _accelerate_other

    _accelerate_other.is_deepspeed_available = lambda: False
except Exception:
    pass
try:
    import torch.distributed.fsdp as _torch_fsdp

    if not hasattr(_torch_fsdp, "FSDPModule"):
        class _RWKV7FSDPModuleCompat(torch.nn.Module):
            pass

        _torch_fsdp.FSDPModule = _RWKV7FSDPModuleCompat
except Exception:
    pass

from rwkv7_hf.native_model import NativeRWKV7ForCausalLM

PROMPTS = ["The quick brown fox", "Once upon a time", "RWKV is a linear model"]


def reward_func(prompts: list[Any], completions: list[Any], **_: Any) -> list[float]:
    # Deterministic non-constant rewards -> non-zero GRPO advantage/update.
    return [float(i % 2) for i, _ in enumerate(completions)]


def trainable_snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--max-steps", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-completion-length", type=int, default=8)
    args = ap.parse_args()

    dt = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or 0

    model = NativeRWKV7ForCausalLM.from_pretrained(
        args.model, torch_dtype=dt, device_map="cuda"
    )
    peft_config = LoraConfig(
        task_type="CAUSAL_LM", r=4, lora_alpha=8, lora_dropout=0.0,
        target_modules=["r_proj", "v_proj", "o_proj"],
    )

    from trl import GRPOConfig, GRPOTrainer

    dataset = HFDataset.from_dict({"prompt": PROMPTS * 4})

    with tempfile.TemporaryDirectory(prefix="native_grpo_") as out_dir:
        train_args = GRPOConfig(
            output_dir=out_dir,
            max_steps=args.max_steps,
            per_device_train_batch_size=max(2, args.batch_size),
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            fp16=(args.dtype == "fp16"),
            bf16=(args.dtype == "bf16"),
            gradient_checkpointing=False,
            optim="adamw_torch",
            remove_unused_columns=False,
            max_completion_length=args.max_completion_length,
            num_generations=2,
            generation_batch_size=2,
        )
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_func,
            args=train_args,
            train_dataset=dataset,
            processing_class=tok,
            peft_config=peft_config,
        )
        before = trainable_snapshot(trainer.model)
        result = trainer.train()

    loss = float(result.training_loss)
    after = trainable_snapshot(trainer.model)
    delta = max((before[n] - after[n]).abs().max().item() for n in before)
    log = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    print(f"[native-grpo] loss history: {[round(x, 4) for x in log]}")
    print(f"[native-grpo] train_loss={loss:.4f}, max_trainable_delta={delta:.6f}")
    ok = math.isfinite(loss) and delta > 0.0
    print("NATIVE GRPO PASS" if ok else "NATIVE GRPO FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
