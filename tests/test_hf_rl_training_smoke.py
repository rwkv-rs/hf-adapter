#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from pathlib import Path
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


def device_name(device: str) -> str:
    return torch.cuda.get_device_name(0) if device.startswith("cuda") and torch.cuda.is_available() else device


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
    """Keep LoRA adapters in fp32 for deterministic one-step smoke updates."""

    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
            if param.grad is not None:
                param.grad.data = param.grad.data.float()


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


def load_base_model(model_path: str, device: str, attn_mode: str, train_dtype: str):
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


def run_dpo(args: argparse.Namespace) -> dict[str, Any]:
    ensure_trl_fsdp_compat()
    from trl import DPOConfig, DPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_base_model(args.model, args.device, args.attn_mode, args.train_dtype)
    dataset = HFDataset.from_dict(
        {
            "prompt": PROMPTS * max(1, int(args.dataset_repeats)),
            "chosen": [" Hello!", " one two."] * max(1, int(args.dataset_repeats)),
            "rejected": [" Goodbye.", " three four."] * max(1, int(args.dataset_repeats)),
        }
    )
    with tempfile.TemporaryDirectory(prefix="rwkv7_dpo_smoke_") as out_dir:
        train_args = DPOConfig(
            output_dir=out_dir,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=1e-4,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            # The model itself is loaded in --train-dtype. Keep Trainer mixed
            # precision off so the one-step LoRA update check is not skipped by
            # AMP loss scaling on tiny smoke batches.
            fp16=False,
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
        keep_trainable_params_fp32(trainer.model)
        before = trainable_snapshot(trainer.model)
        assert before, "expected LoRA/trainable parameters"
        result = trainer.train()
    loss = float(result.training_loss)
    assert math.isfinite(loss), result.training_loss
    delta = max_trainable_delta(before, trainer.model)
    assert delta > 0.0, "DPO LoRA/trainable parameters did not update"
    metrics = dict(getattr(result, "metrics", {}) or {})
    row = {
        "axis": "training_smoke",
        "backend": "hf_adapter",
        "trainer_backend": "trl_dpo",
        "status": "pass",
        "dtype": args.train_dtype,
        "train_dtype": args.train_dtype,
        "device": device_name(args.device),
        **model_metadata(args, trainer.model),
        "attn_mode": args.attn_mode,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "max_steps": args.max_steps,
        "dataset_repeats": args.dataset_repeats,
        "max_length": args.max_length,
        "train_loss": loss,
        "train_runtime_s": metric(metrics, "train_runtime"),
        "train_samples_per_second": metric(metrics, "train_samples_per_second"),
        "train_steps_per_second": metric(metrics, "train_steps_per_second"),
        "max_trainable_delta": delta,
    }
    print("trl_dpo_train_loss", loss, "max_trainable_delta", delta)
    return row


def reward_func(prompts: list[Any], completions: list[Any], **_: Any) -> list[float]:
    # Deterministic non-constant rewards keep the smoke focused on trainer/model
    # compatibility while still producing a non-zero GRPO advantage/update.
    return [float(i % 2) for i, _ in enumerate(completions)]


def run_grpo(args: argparse.Namespace) -> dict[str, Any]:
    ensure_trl_fsdp_compat()
    from trl import GRPOConfig, GRPOTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_base_model(args.model, args.device, args.attn_mode, args.train_dtype)
    dataset = HFDataset.from_dict({"prompt": PROMPTS * max(1, int(args.dataset_repeats))})
    with tempfile.TemporaryDirectory(prefix="rwkv7_grpo_smoke_") as out_dir:
        train_args = GRPOConfig(
            output_dir=out_dir,
            max_steps=args.max_steps,
            per_device_train_batch_size=max(2, args.batch_size),
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=1e-4,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            fp16=False,
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
        keep_trainable_params_fp32(trainer.model)
        before = trainable_snapshot(trainer.model)
        assert before, "expected LoRA/trainable parameters"
        result = trainer.train()
    loss = float(result.training_loss)
    assert math.isfinite(loss), result.training_loss
    delta = max_trainable_delta(before, trainer.model)
    assert delta > 0.0, "GRPO LoRA/trainable parameters did not update"
    metrics = dict(getattr(result, "metrics", {}) or {})
    row = {
        "axis": "training_smoke",
        "backend": "hf_adapter",
        "trainer_backend": "trl_grpo",
        "status": "pass",
        "dtype": args.train_dtype,
        "train_dtype": args.train_dtype,
        "device": device_name(args.device),
        **model_metadata(args, trainer.model),
        "attn_mode": args.attn_mode,
        "batch_size": max(2, args.batch_size),
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": max(2, args.batch_size) * args.gradient_accumulation_steps,
        "max_steps": args.max_steps,
        "dataset_repeats": args.dataset_repeats,
        "max_completion_length": args.grpo_max_completion_length,
        "train_loss": loss,
        "train_runtime_s": metric(metrics, "train_runtime"),
        "train_samples_per_second": metric(metrics, "train_samples_per_second"),
        "train_steps_per_second": metric(metrics, "train_steps_per_second"),
        "max_trainable_delta": delta,
    }
    print("trl_grpo_train_loss", loss, "max_trainable_delta", delta)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-size-label", default="", help="Optional size label such as 0.4b; inferred from --model when omitted")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-mode", default="fused_recurrent", choices=["chunk", "fused_recurrent"])
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--train-dtype", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--grpo-max-completion-length", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=1)
    ap.add_argument("--dataset-repeats", type=int, default=4)
    ap.add_argument("--backend", choices=["dpo", "grpo", "both"], default="both")
    ap.add_argument("--results", default="")
    args = ap.parse_args()
    if args.train_dtype is None:
        args.train_dtype = "bf16" if args.device.startswith("cuda") else "fp32"

    rows = []
    if args.backend in {"dpo", "both"}:
        rows.append(run_dpo(args))
    if args.backend in {"grpo", "both"}:
        rows.append(run_grpo(args))
    append_rows(args.results, rows)
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
