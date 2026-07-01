#!/usr/bin/env python3
# coding=utf-8
"""HF Trainer checkpoint/resume smoke for NativeRWKV7ForCausalLM + PEFT LoRA.

This covers the production-ish Trainer lifecycle that is not exercised by a
single uninterrupted training call: save a checkpoint, create a fresh native
model + LoRA wrapper, then resume from the saved checkpoint and continue.

Gate: checkpoint exists, resumed Trainer reaches the requested global step,
finite loss is reported, and LoRA parameters update.

  python tests/test_native_trainer_resume_smoke.py --model <hf_dir>
"""
from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, Trainer, TrainingArguments

from rwkv7_hf.native_model import NativeRWKV7ForCausalLM

try:  # Keep this smoke independent of a partially-installed DeepSpeed package.
    import accelerate.utils.other as _accelerate_other

    _accelerate_other.is_deepspeed_available = lambda: False
except Exception:
    pass

PROMPTS = [
    "User: Say hello.\n\nAssistant: Hello!",
    "User: Count to three.\n\nAssistant: one two three.",
    "RWKV uses recurrent state instead of a KV cache.",
    "The capital of France is Paris.",
]


class FixedLenCollator:
    def __call__(self, feats):
        ids = torch.stack([f["input_ids"] for f in feats])
        return {"input_ids": ids, "labels": ids.clone()}


def device_map_for(device: str):
    if not device.startswith("cuda"):
        return None
    if ":" in device:
        return {"": int(device.split(":", 1)[1])}
    return "cuda"


def build_rows(tok, length: int, repeats: int = 8):
    rows = []
    for p in PROMPTS * repeats:
        ids = tok(p, add_special_tokens=False).input_ids[:length]
        if len(ids) < length:
            ids = ids + [tok.pad_token_id or 0] * (length - len(ids))
        rows.append({"input_ids": torch.tensor(ids, dtype=torch.long)})
    return rows


def load_lora_native(model_dir: str, dtype: torch.dtype, device: str):
    model = NativeRWKV7ForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        device_map=device_map_for(device),
    )
    model.config.use_cache = False
    lc = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["r_proj", "k_proj", "v_proj", "o_proj", "key", "value"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lc)


def trainable_snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def max_delta(before, model) -> float:
    out = 0.0
    for n, p in model.named_parameters():
        if p.requires_grad and n in before:
            out = max(out, float((before[n] - p.detach()).abs().max().item()))
    return out


def latest_checkpoint(out_dir: str) -> str:
    ckpts = sorted(Path(out_dir).glob("checkpoint-*"), key=lambda p: int(p.name.rsplit("-", 1)[1]))
    assert ckpts, f"no checkpoints in {out_dir}"
    return str(ckpts[-1])


def make_args(out_dir: str, max_steps: int, batch_size: int, dtype: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=out_dir,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        logging_steps=1,
        save_strategy="steps",
        save_steps=1,
        save_total_limit=3,
        # Older torch builds (<2.6) are blocked by recent Transformers from
        # torch.load'ing optimizer .pt state. The checkpoint/resume contract we
        # need here is model/adapter + TrainerState continuity, so make the
        # smoke portable by not saving optimizer/scheduler state.
        save_only_model=True,
        report_to=[],
        remove_unused_columns=False,
        fp16=(dtype == "fp16"),
        bf16=False,
        dataloader_num_workers=0,
        gradient_checkpointing=False,
        optim="adamw_torch",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--first-steps", type=int, default=2)
    ap.add_argument("--resume-steps", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--length", type=int, default=32)
    args = ap.parse_args()
    assert args.resume_steps > args.first_steps, "resume-steps must exceed first-steps"

    dtype = torch.float32 if args.dtype == "fp32" else torch.float16
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rows = build_rows(tok, args.length)
    collator = FixedLenCollator()

    with tempfile.TemporaryDirectory(prefix="native_trainer_resume_") as out_dir:
        model = load_lora_native(args.model, dtype, args.device)
        before_first = trainable_snapshot(model)
        trainer = Trainer(
            model=model,
            args=make_args(out_dir, args.first_steps, args.batch_size, args.dtype),
            train_dataset=rows,
            data_collator=collator,
        )
        first_result = trainer.train()
        first_delta = max_delta(before_first, model)
        ckpt = latest_checkpoint(out_dir)

        resumed_model = load_lora_native(args.model, dtype, args.device)
        before_resume = trainable_snapshot(resumed_model)
        resumed_trainer = Trainer(
            model=resumed_model,
            args=make_args(out_dir, args.resume_steps, args.batch_size, args.dtype),
            train_dataset=rows,
            data_collator=collator,
        )
        resume_result = resumed_trainer.train(resume_from_checkpoint=ckpt)
        resume_delta = max_delta(before_resume, resumed_model)
        global_step = int(resumed_trainer.state.global_step)

    first_loss = float(first_result.training_loss)
    resume_loss = float(resume_result.training_loss)
    ok = (
        math.isfinite(first_loss)
        and math.isfinite(resume_loss)
        and first_delta > 0.0
        and resume_delta > 0.0
        and global_step == args.resume_steps
    )
    print(
        f"[native-trainer-resume] checkpoint={Path(ckpt).name}, "
        f"first_loss={first_loss:.4f}, resume_loss={resume_loss:.4f}, "
        f"first_delta={first_delta:.6f}, resume_delta={resume_delta:.6f}, "
        f"global_step={global_step}"
    )
    print("NATIVE TRAINER RESUME PASS" if ok else "NATIVE TRAINER RESUME FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
