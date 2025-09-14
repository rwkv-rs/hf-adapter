#!/usr/bin/env python3
# coding=utf-8
"""TRL SFTTrainer smoke for NativeRWKV7ForCausalLM (fla-free PEFT path).

This is the native/no-FLA counterpart to ``test_hf_training_smoke.py --backend trl``.
It exercises the standard TRL ``SFTTrainer`` + PEFT LoRA integration through the
native model's HF CausalLM contract: attention_mask is accepted, labels produce
all-token logits/loss, and LoRA modules update.

Gate: finite SFT loss AND trainable LoRA parameters update.

  python tests/test_native_sft_smoke.py --model <hf_dir>
"""
from __future__ import annotations

import argparse
import math
import tempfile

import torch
from datasets import Dataset as HFDataset
from peft import LoraConfig
from transformers import AutoTokenizer

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

from rwkv7_hf.native_model import NativeRWKV7ForCausalLM

PROMPTS = [
    "User: Say hello.\n\nAssistant: Hello!",
    "User: Count to three.\n\nAssistant: one two three.",
    "RWKV is a linear recurrent language model.",
]


def device_map_for(device: str):
    if not device.startswith("cuda"):
        return None
    if ":" in device:
        return {"": int(device.split(":", 1)[1])}
    return "cuda"


def trainable_snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--max-steps", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-length", type=int, default=48)
    args = ap.parse_args()

    dt = torch.float32 if args.dtype == "fp32" else torch.float16
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = NativeRWKV7ForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dt,
        device_map=device_map_for(args.device),
    )
    model.config.use_cache = False

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["r_proj", "k_proj", "v_proj", "o_proj", "key", "value"],
    )

    from trl import SFTConfig, SFTTrainer

    dataset = HFDataset.from_dict({"text": PROMPTS * 4})

    with tempfile.TemporaryDirectory(prefix="native_sft_") as out_dir:
        train_args = SFTConfig(
            output_dir=out_dir,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            fp16=(args.dtype == "fp16" and args.device.startswith("cuda")),
            bf16=False,
            gradient_checkpointing=False,
            optim="adamw_torch",
            max_length=args.max_length,
            dataset_text_field="text",
            packing=False,
            # TRL 1.7 defaults to chunked_nll on some installs, which patches
            # the inner backbone and expects a standalone `NativeRWKV7Model`
            # hidden-state forward. This smoke targets the standard CausalLM
            # SFT path used by most HF scripts, so keep loss computation on the
            # model's normal labels -> loss forward contract.
            loss_type="nll",
        )
        trainer = SFTTrainer(
            model=model,
            args=train_args,
            train_dataset=dataset,
            processing_class=tok,
            peft_config=peft_config,
        )
        before = trainable_snapshot(trainer.model)
        assert before, "expected LoRA/trainable parameters"
        result = trainer.train()

    loss = float(result.training_loss)
    after = trainable_snapshot(trainer.model)
    delta = max((before[n] - after[n]).abs().max().item() for n in before)
    log = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    print(f"[native-sft] loss history: {[round(x, 4) for x in log]}")
    print(f"[native-sft] train_loss={loss:.4f}, max_trainable_delta={delta:.6f}")
    ok = math.isfinite(loss) and delta > 0.0
    print("NATIVE SFT PASS" if ok else "NATIVE SFT FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
