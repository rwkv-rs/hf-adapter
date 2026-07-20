#!/usr/bin/env python3
# coding=utf-8
"""HF Trainer smoke for NativeRWKV7ForCausalLM (fla-free path, PEFT LoRA).

Proves the native model runs the FULL HF ``Trainer`` loop (not just forward/
backward) with PEFT LoRA — so it can replace the FLA wrapper for training on
cards where FLA backward is blocked (e.g. Blackwell sm_120: FLA DPLR chunk
backward needs 128KB shared mem > 5070's 99KB). Depends on the native path's
HF Cache contract (NativeRWKV7Cache) and the module-call PERT fix
(attn_step[_batched] calls layer.r_proj(x) so PEFT's LoraLinear is invoked).

Gate: loss decreases over a few steps AND trainable (LoRA) params actually update.

  python tests/test_native_trainer_smoke.py --model <hf_dir>
"""
from __future__ import annotations

import argparse
import tempfile

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, Trainer, TrainingArguments

from rwkv7_hf.native_model import NativeRWKV7ForCausalLM
from rwkv7_hf.training_precision import peft_trainer_precision_kwargs

try:  # Keep this smoke independent of a partially-installed DeepSpeed package.
    import accelerate.utils.other as _accelerate_other

    _accelerate_other.is_deepspeed_available = lambda: False
except Exception:
    pass

PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Once upon a time, in a faraway land,",
    "User: Hello!\n\nAssistant: Hi there!",
    "The capital of France is Paris.",
    "import torch\nx = torch.randn(",
    "RWKV is a linear recurrent model.",
]


class FixedLenCollator:
    def __call__(self, feats):
        ids = torch.stack([f["input_ids"] for f in feats])
        return {"input_ids": ids, "labels": ids.clone()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--length", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    dt = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.dtype]
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = NativeRWKV7ForCausalLM.from_pretrained(
        args.model, torch_dtype=dt, device_map="cuda"
    )

    lc = LoraConfig(
        r=8, lora_alpha=16, target_modules=["r_proj", "v_proj", "o_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lc)

    rows = []
    for p in PROMPTS * 8:
        ids = tok(p, add_special_tokens=False).input_ids[: args.length]
        if len(ids) < args.length:
            ids = ids + [0] * (args.length - len(ids))
        rows.append({"input_ids": torch.tensor(ids, dtype=torch.long)})

    before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    precision_kwargs = peft_trainer_precision_kwargs(args.dtype)

    targs = TrainingArguments(
        output_dir=tempfile.mkdtemp(prefix="native_trainer_"),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        **precision_kwargs,
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=rows, data_collator=FixedLenCollator()
    )
    trainer.train()

    after = {n: p for n, p in model.named_parameters() if p.requires_grad}
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    updated = sum(1 for n in before if not torch.equal(before[n], after[n]))

    print(f"[native-trainer] loss history: {[round(x, 4) for x in losses]}")
    print(f"[native-trainer] trainable params updated: {updated}/{len(before)}")
    ok = len(losses) >= 2 and losses[-1] < losses[0] and updated > 0
    print("NATIVE TRAINER PASS" if ok else "NATIVE TRAINER FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
