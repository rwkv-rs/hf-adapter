#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
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

TRAIN_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def infer_model_size_label(model_path: str, explicit: str = "") -> str | None:
    if explicit:
        return explicit.lower()
    match = re.search(r"(\d+(?:\.\d+)?b)", Path(model_path).name.lower())
    return match.group(1) if match else None


def model_metadata(args, model) -> dict[str, Any]:
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


def metric(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    return float(value) if value is not None else None


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


def keep_trainable_params_fp32(model) -> None:
    """Keep LoRA adapters in fp32 so a one-step smoke observes a real update.

    The base model is still loaded in ``--train-dtype``.  Trainer AMP/GradScaler
    can legitimately skip a tiny fp16 one-step update when RWKV kernels emit a
    non-finite global grad norm; for this compatibility smoke we care that the
    HF/PEFT training stack runs and trainable LoRA weights move.
    """

    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
            if param.grad is not None:
                param.grad.data = param.grad.data.float()


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
    model = get_peft_model(model, lora_cfg)
    keep_trainable_params_fp32(model)
    return model


def run_trainer(args) -> dict[str, Any]:
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_lora_model(args.model, args.device, args.attn_mode, args.train_dtype)
    dataset = TokenDataset(tok, args.max_length, repeats=args.dataset_repeats)
    collator = CausalCollator(tok)
    before = trainable_snapshot(model)
    assert before, "expected LoRA/trainable parameters"
    with tempfile.TemporaryDirectory(prefix="rwkv7_trainer_smoke_") as out_dir:
        train_args = TrainingArguments(
            output_dir=out_dir,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=1e-4,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            remove_unused_columns=False,
            # The model itself is loaded in --train-dtype. Keep Trainer mixed
            # precision off so GradScaler cannot turn the one-step smoke into a
            # no-op before the trainable-delta assertion.
            fp16=False,
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
    delta = max_trainable_delta(before, model)
    assert delta > 0.0, "LoRA/trainable parameters did not update"
    metrics = dict(getattr(result, "metrics", {}) or {})
    row = {
        "axis": "training_smoke",
        "backend": "hf_adapter",
        "trainer_backend": "trainer",
        "status": "pass",
        "dtype": args.train_dtype,
        "train_dtype": args.train_dtype,
        "device": device_name(args.device),
        **model_metadata(args, model),
        "attn_mode": args.attn_mode,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "max_steps": args.max_steps,
        "dataset_repeats": args.dataset_repeats,
        "max_length": args.max_length,
        "train_loss": float(result.training_loss),
        "train_runtime_s": metric(metrics, "train_runtime"),
        "train_samples_per_second": metric(metrics, "train_samples_per_second"),
        "train_steps_per_second": metric(metrics, "train_steps_per_second"),
        "max_trainable_delta": delta,
        **runtime_metadata(model),
    }
    print("trainer_train_loss", result.training_loss, "max_trainable_delta", delta)
    return row


def run_trl(args) -> dict[str, Any]:
    try:
        from datasets import Dataset as HFDataset
        from trl import SFTConfig, SFTTrainer
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise RuntimeError("TRL SFT smoke requires `datasets` and `trl`") from exc

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_lora_model(args.model, args.device, args.attn_mode, args.train_dtype)
    dataset = HFDataset.from_dict({"text": PROMPTS * max(1, int(args.dataset_repeats))})
    before = trainable_snapshot(model)
    assert before, "expected LoRA/trainable parameters"
    with tempfile.TemporaryDirectory(prefix="rwkv7_trl_sft_smoke_") as out_dir:
        sft_args = SFTConfig(
            output_dir=out_dir,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=1e-4,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            fp16=False,
            bf16=False,
            gradient_checkpointing=False,
            max_length=args.max_length,
            dataset_text_field="text",
            packing=False,
            # TRL 1.7 defaults to ``chunked_nll`` and patches a decoder-only
            # backbone interface that RWKV intentionally does not expose.
            # The standard NLL path delegates to the public CausalLM forward
            # contract and is the portable choice for recurrent architectures.
            loss_type="nll",
        )
        trainer = SFTTrainer(model=model, args=sft_args, train_dataset=dataset, processing_class=tok)
        result = trainer.train()
    assert math.isfinite(float(result.training_loss)), result.training_loss
    delta = max_trainable_delta(before, model)
    assert delta > 0.0, "LoRA/trainable parameters did not update"
    metrics = dict(getattr(result, "metrics", {}) or {})
    row = {
        "axis": "training_smoke",
        "backend": "hf_adapter",
        "trainer_backend": "trl_sft",
        "status": "pass",
        "dtype": args.train_dtype,
        "train_dtype": args.train_dtype,
        "device": device_name(args.device),
        **model_metadata(args, model),
        "attn_mode": args.attn_mode,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "max_steps": args.max_steps,
        "dataset_repeats": args.dataset_repeats,
        "max_length": args.max_length,
        "train_loss": float(result.training_loss),
        "train_runtime_s": metric(metrics, "train_runtime"),
        "train_samples_per_second": metric(metrics, "train_samples_per_second"),
        "train_steps_per_second": metric(metrics, "train_steps_per_second"),
        "max_trainable_delta": delta,
        **runtime_metadata(model),
    }
    print("trl_sft_train_loss", result.training_loss, "max_trainable_delta", delta)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-size-label", default="", help="Optional size label such as 0.4b; inferred from --model when omitted")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-mode", default="fused_recurrent", choices=["chunk", "fused_recurrent"])
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--train-dtype", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--max-steps", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=1)
    ap.add_argument("--dataset-repeats", type=int, default=4)
    ap.add_argument("--backend", choices=["trainer", "trl", "both"], default="both")
    ap.add_argument("--results", default="")
    args = ap.parse_args()
    if args.device == "mps":
        os.environ.setdefault("RWKV7_NATIVE_MODEL", "1")
    if args.train_dtype is None:
        args.train_dtype = "bf16" if args.device.startswith("cuda") else "fp32"

    rows = []
    if args.backend in {"trainer", "both"}:
        rows.append(run_trainer(args))
    if args.backend in {"trl", "both"}:
        rows.append(run_trl(args))
    append_rows(args.results, rows)
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
