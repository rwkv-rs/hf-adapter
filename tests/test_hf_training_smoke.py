#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from typing import Any

# FLA backward can hit Dynamo/Triton issues on the V100 test box.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import math

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


PROMPTS = [
    "User: Say hello.\n\nAssistant: Hello!",
    "User: Count to three.\n\nAssistant: one two three.",
]


class TokenDataset(Dataset):
    def __init__(self, tokenizer, max_length: int):
        self.rows = []
        for text in PROMPTS:
            enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
            row = {k: v[0] for k, v in enc.items()}
            row["labels"] = row["input_ids"].clone()
            self.rows.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.rows[idx].items()}


@dataclass
class CausalCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        labels = [f["labels"] for f in features]
        inputs = [{k: v for k, v in f.items() if k != "labels"} for f in features]
        batch = self.tokenizer.pad(inputs, return_tensors="pt")
        label_batch = self.tokenizer.pad({"input_ids": labels}, return_tensors="pt")["input_ids"]
        label_batch[label_batch == self.tokenizer.pad_token_id] = -100
        batch["labels"] = label_batch
        return batch


def load_lora_model(model_path: str, device: str, attn_mode: str):
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
    lora_cfg = LoraConfig(
        task_type="CAUSAL_LM",
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["r_proj", "k_proj", "v_proj", "o_proj", "key", "value"],
    )
    return get_peft_model(model, lora_cfg)


def run_trainer(args) -> None:
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_lora_model(args.model, args.device, args.attn_mode)
    dataset = TokenDataset(tok, args.max_length)
    collator = CausalCollator(tok)
    with tempfile.TemporaryDirectory(prefix="rwkv7_trainer_smoke_") as out_dir:
        train_args = TrainingArguments(
            output_dir=out_dir,
            max_steps=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            remove_unused_columns=False,
            fp16=args.device.startswith("cuda"),
            bf16=False,
            dataloader_num_workers=0,
            gradient_checkpointing=False,
            optim="adamw_torch",
        )
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=dataset,
            data_collator=collator,
            processing_class=tok,
        )
        result = trainer.train()
    assert math.isfinite(float(result.training_loss)), result.training_loss
    print("trainer_train_loss", result.training_loss)


def run_trl(args) -> None:
    try:
        from datasets import Dataset as HFDataset
        from trl import SFTConfig, SFTTrainer
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("TRL SFT smoke requires `datasets` and `trl`") from exc

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_lora_model(args.model, args.device, args.attn_mode)
    dataset = HFDataset.from_dict({"text": PROMPTS})
    with tempfile.TemporaryDirectory(prefix="rwkv7_trl_sft_smoke_") as out_dir:
        sft_args = SFTConfig(
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
            max_length=args.max_length,
            dataset_text_field="text",
            packing=False,
        )
        trainer = SFTTrainer(model=model, args=sft_args, train_dataset=dataset, processing_class=tok)
        result = trainer.train()
    assert math.isfinite(float(result.training_loss)), result.training_loss
    print("trl_sft_train_loss", result.training_loss)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-mode", default="fused_recurrent", choices=["chunk", "fused_recurrent"])
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--backend", choices=["trainer", "trl", "both"], default="both")
    args = ap.parse_args()
    if args.backend in {"trainer", "both"}:
        run_trainer(args)
    if args.backend in {"trl", "both"}:
        run_trl(args)
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
