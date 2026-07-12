#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

# This smoke only resumes checkpoints it creates in a temporary directory.
# Disable the recent Transformers torch.load guard for older torch runtimes.
try:
    import transformers.trainer as _hf_trainer
    import transformers.utils.import_utils as _hf_import_utils

    _hf_import_utils.check_torch_load_is_safe = lambda: None
    _hf_trainer.check_torch_load_is_safe = lambda: None
except Exception:
    pass

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


PROMPTS = [
    "User: Say hello.\n\nAssistant: Hello!",
    "User: Count to three.\n\nAssistant: one two three.",
]
TRAIN_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def infer_model_size_label(model_path: str, explicit: str = "") -> str | None:
    if explicit:
        return explicit.lower()
    match = re.search(r"(\d+(?:\.\d+)?b)", Path(model_path).name.lower())
    return match.group(1) if match else None


def model_metadata(args: argparse.Namespace, model) -> dict[str, Any]:
    cfg = getattr(model, "config", None)
    return {
        "model_name": Path(args.model).name,
        "model_size_label": infer_model_size_label(args.model, args.model_size_label),
        "hf_model_dir": args.model,
        "hidden_size": getattr(cfg, "hidden_size", None),
        "intermediate_size": getattr(cfg, "intermediate_size", None),
        "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
        "head_dim": getattr(cfg, "head_dim", None),
        "num_heads": getattr(cfg, "num_heads", None),
    }


class TokenDataset(Dataset):
    def __init__(self, tokenizer, max_length: int, repeats: int = 1):
        self.rows = []
        for text in PROMPTS * max(1, int(repeats)):
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


def device_name(device: str) -> str:
    return torch.cuda.get_device_name(0) if device.startswith("cuda") and torch.cuda.is_available() else device


def runtime_metadata(model) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]

    def git(*args: str) -> str | None:
        proc = subprocess.run(
            ["git", *args], cwd=root, text=True, capture_output=True, check=False
        )
        return proc.stdout.strip() if proc.returncode == 0 else None

    chip = None
    if platform.system() == "Darwin":
        proc = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
            capture_output=True,
            check=False,
        )
        chip = proc.stdout.strip() or None
    dirty = git("status", "--porcelain")
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "chip": chip,
        "python": platform.python_version(),
        "torch_version": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "parameter_device": str(next(model.parameters()).device),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(dirty) if dirty is not None else None,
        "process_isolated": True,
    }


def train_torch_dtype(train_dtype: str) -> torch.dtype:
    return TRAIN_DTYPES[train_dtype]


def use_fp16(device: str, train_dtype: str) -> bool:
    return device.startswith("cuda") and train_dtype == "fp16"


def use_bf16(device: str, train_dtype: str) -> bool:
    return device.startswith("cuda") and train_dtype == "bf16"


def append_rows(path: str, rows: list[dict[str, Any]]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def trainable_snapshot(model) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().float().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def max_trainable_delta(before: dict[str, torch.Tensor], model) -> float:
    max_delta = 0.0
    for name, param in model.named_parameters():
        if not param.requires_grad or name not in before:
            continue
        delta = (param.detach().float().cpu() - before[name]).abs().max().item()
        max_delta = max(max_delta, float(delta))
    return max_delta


def load_lora_model(model_path: str, device: str, attn_mode: str, train_dtype: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=train_torch_dtype(train_dtype),
        device_map=device if device.startswith("cuda") else None,
    )
    model.config.use_cache = False
    model.config.fuse_cross_entropy = False
    model.config.use_l2warp = False
    model.config.attn_mode = attn_mode
    if not device.startswith("cuda"):
        model = model.to(device)
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


def latest_checkpoint(out_dir: str) -> str:
    ckpts = sorted(Path(out_dir).glob("checkpoint-*"), key=lambda p: int(p.name.rsplit("-", 1)[1]))
    if not ckpts:
        raise RuntimeError(f"no checkpoints in {out_dir}")
    return str(ckpts[-1])


def release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def make_args(args: argparse.Namespace, out_dir: str, max_steps: int) -> TrainingArguments:
    return TrainingArguments(
        output_dir=out_dir,
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=1e-4,
        logging_steps=1,
        save_strategy="steps",
        save_steps=1,
        save_total_limit=3,
        report_to=[],
        remove_unused_columns=False,
        fp16=use_fp16(args.device, args.train_dtype),
        bf16=use_bf16(args.device, args.train_dtype),
        dataloader_num_workers=0,
        gradient_checkpointing=False,
        optim="adamw_torch",
    )


def run_resume(args: argparse.Namespace) -> dict[str, Any]:
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    dataset = TokenDataset(tok, args.max_length, repeats=args.dataset_repeats)
    collator = CausalCollator(tok)

    out_dir = tempfile.mkdtemp(prefix="rwkv7_hf_trainer_resume_")
    try:
        model = load_lora_model(args.model, args.device, args.attn_mode, args.train_dtype)
        before_first = trainable_snapshot(model)
        trainer = Trainer(
            model=model,
            args=make_args(args, out_dir, args.first_steps),
            train_dataset=dataset,
            data_collator=collator,
            processing_class=tok,
        )
        first_result = trainer.train()
        first_loss = float(first_result.training_loss)
        first_delta = max_trainable_delta(before_first, model)
        first_step = int(trainer.state.global_step)
        ckpt = latest_checkpoint(out_dir)
        checkpoint_name = Path(ckpt).name
        for rng_file in Path(ckpt).glob("rng_state*.pth"):
            rng_file.unlink(missing_ok=True)

        del trainer, model, before_first, first_result
        release_cuda()

        resumed_model = load_lora_model(args.model, args.device, args.attn_mode, args.train_dtype)
        before_resume = trainable_snapshot(resumed_model)
        resumed_trainer = Trainer(
            model=resumed_model,
            args=make_args(args, out_dir, args.resume_steps),
            train_dataset=dataset,
            data_collator=collator,
            processing_class=tok,
        )
        resume_result = resumed_trainer.train(resume_from_checkpoint=ckpt)
        resume_loss = float(resume_result.training_loss)
        resume_delta = max_trainable_delta(before_resume, resumed_model)
        global_step = int(resumed_trainer.state.global_step)
        metrics = dict(getattr(resume_result, "metrics", {}) or {})
        metadata_model = resumed_model
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)

    ok = (
        math.isfinite(first_loss)
        and math.isfinite(resume_loss)
        and first_delta > 0.0
        and resume_delta > 0.0
        and first_step == args.first_steps
        and global_step == args.resume_steps
    )
    row = {
        "axis": "checkpoint_resume_smoke",
        "backend": "hf_adapter",
        "trainer_backend": "trainer_resume",
        "status": "pass" if ok else "fail",
        "dtype": args.train_dtype,
        "train_dtype": args.train_dtype,
        "device": device_name(args.device),
        **model_metadata(args, metadata_model),
        "attn_mode": args.attn_mode,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "first_steps": args.first_steps,
        "resume_steps": args.resume_steps,
        "dataset_repeats": args.dataset_repeats,
        "max_length": args.max_length,
        "checkpoint": checkpoint_name,
        "first_loss": first_loss,
        "resume_loss": resume_loss,
        "train_runtime_s": float(metrics["train_runtime"]) if "train_runtime" in metrics else None,
        "first_max_trainable_delta": first_delta,
        "resume_max_trainable_delta": resume_delta,
        "global_step": global_step,
        **runtime_metadata(metadata_model),
    }
    if not ok:
        raise AssertionError(row)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-size-label", default="", help="Optional size label such as 0.4b; inferred from --model when omitted")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-mode", default="fused_recurrent", choices=["chunk", "fused_recurrent"])
    ap.add_argument("--max-length", type=int, default=32)
    ap.add_argument("--train-dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    ap.add_argument("--first-steps", type=int, default=1)
    ap.add_argument("--resume-steps", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=1)
    ap.add_argument("--dataset-repeats", type=int, default=4)
    ap.add_argument("--results", default="")
    args = ap.parse_args()
    if args.device == "mps":
        os.environ.setdefault("RWKV7_NATIVE_MODEL", "1")
    if args.resume_steps <= args.first_steps:
        raise ValueError("resume-steps must exceed first-steps")
    row = run_resume(args)
    append_rows(args.results, [row])
    print(json.dumps(row, ensure_ascii=False))
    print("HF TRAINER RESUME PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
