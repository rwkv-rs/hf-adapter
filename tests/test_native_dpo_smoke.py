#!/usr/bin/env python3
# coding=utf-8
"""TRL DPO smoke for NativeRWKV7ForCausalLM (fla-free RL training path).

The FLA-backed DPO/GRPO smoke (`test_hf_rl_training_smoke.py`) fails on cards
where FLA backward is blocked (Blackwell sm_120: FLA DPLR chunk backward needs
128KB shared mem > 5070's 99KB). The native path's HF Trainer works after the
Cache-contract + module-call fixes on this branch; this test checks that TRL
**DPOTrainer** also runs end-to-end on native, unblocking the bounty's
PEFT+RL requirement on such cards.

Gate: finite DPO loss AND trainable (LoRA) parameters update.

  python tests/test_native_dpo_smoke.py --model <hf_dir>
"""
from __future__ import annotations

import argparse
import math
import tempfile

import torch
from datasets import Dataset as HFDataset
from peft import LoraConfig
from transformers import AutoTokenizer

from rwkv7_hf.native_model import NativeRWKV7ForCausalLM

try:  # Keep this smoke independent of a partially-installed DeepSpeed package.
    import accelerate.utils.other as _accelerate_other

    _accelerate_other.is_deepspeed_available = lambda: False
except Exception:
    pass

try:  # TRL versions may import the newer torch FSDPModule symbol on older torch.
    import torch.distributed.fsdp as _torch_fsdp

    if not hasattr(_torch_fsdp, "FSDPModule"):
        class _RWKV7FSDPModuleCompat(torch.nn.Module):
            pass

        _torch_fsdp.FSDPModule = _RWKV7FSDPModuleCompat
except Exception:
    pass

PROMPTS = ["The quick brown fox", "Once upon a time"]


def trainable_snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--max-steps", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=48)
    args = ap.parse_args()

    dt = torch.float32 if args.dtype == "fp32" else torch.float16
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or 0

    model = NativeRWKV7ForCausalLM.from_pretrained(
        args.model, torch_dtype=dt, device_map="cuda"
    )
    # NativeRWKV7FFN exposes key/value nn.Linear too, so the same target set
    # as the FLA RL smoke works.
    peft_config = LoraConfig(
        task_type="CAUSAL_LM", r=4, lora_alpha=8, lora_dropout=0.0,
        target_modules=["r_proj", "k_proj", "v_proj", "o_proj", "key", "value"],
    )

    from trl import DPOConfig, DPOTrainer

    dataset = HFDataset.from_dict(
        {
            "prompt": PROMPTS * 4,
            "chosen": [" jumps.", " in a land,"] * 4,
            "rejected": [" sleeps.", " far away,"] * 4,
        }
    )

    with tempfile.TemporaryDirectory(prefix="native_dpo_") as out_dir:
        train_args = DPOConfig(
            output_dir=out_dir,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            fp16=(args.dtype == "fp16"),
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
            processing_class=tok,
            peft_config=peft_config,
        )
        before = trainable_snapshot(trainer.model)
        result = trainer.train()

    loss = float(result.training_loss)
    after = trainable_snapshot(trainer.model)
    delta = max(
        (before[n] - after[n]).abs().max().item() for n in before
    )
    log = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    print(f"[native-dpo] loss history: {[round(x, 4) for x in log]}")
    print(f"[native-dpo] train_loss={loss:.4f}, max_trainable_delta={delta:.6f}")
    ok = math.isfinite(loss) and delta > 0.0
    print("NATIVE DPO PASS" if ok else "NATIVE DPO FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
