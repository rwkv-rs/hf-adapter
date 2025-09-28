#!/usr/bin/env python3
# coding=utf-8
"""Apple Silicon / MPS real-model PEFT+Trainer smoke for RWKV-7 HF dirs.

Unlike ``test_apple_silicon_trainer_smoke.py`` this loads a converted HF model
(directory passed by --model).  It is intentionally tiny (short sequence, one
step) but proves that an actual RWKV-7 checkpoint can run PEFT LoRA backward and
HF Trainer on Apple MPS.  Non-Apple hosts emit a skip row unless --require-apple
is set.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import platform
import re
import subprocess
import tempfile
import time
from importlib import metadata
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("RWKV7_NATIVE_MODEL", "1")
os.environ.setdefault("RWKV7_FAST_FORWARD", "0")
os.environ.setdefault("RWKV7_FAST_CACHE", "0")
os.environ.setdefault("RWKV7_FAST_TOKEN_BACKEND", "native_jit")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROMPTS = [
    "User: Hi.\n\nAssistant:",
    "User: Count to two.\n\nAssistant: one two.",
]
LORA_TARGETS = ["r_proj", "k_proj", "v_proj", "o_proj", "key", "value"]
GRPO_LORA_TARGETS = ["r_proj", "v_proj", "o_proj"]


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def append_result(path: str, row: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def emit(path: str, row: dict[str, Any]) -> None:
    print(json.dumps(row, ensure_ascii=False))
    append_result(path, row)


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "missing"


def darwin_sysctl(name: str) -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", name], text=True).strip()
    except Exception:
        return "unknown"


def apple_memory_gb() -> int | str:
    raw = darwin_sysctl("hw.memsize")
    try:
        return round(int(raw) / 1024 / 1024 / 1024)
    except Exception:
        return "unknown"


def infer_model_size_label(model_path: str, explicit: str = "") -> str:
    if explicit:
        return explicit.lower()
    match = re.search(r"(\d+(?:\.\d+)?b)", Path(model_path).name.lower())
    return match.group(1) if match else "unknown"


def choose_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("requested --device mps but MPS is unavailable")
        return requested
    return "mps" if torch.backends.mps.is_available() else "cpu"


def dtype_for(torch: Any, name: str) -> Any:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def tensor_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}


def keep_trainable_params_fp32(model: Any) -> None:
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
            if param.grad is not None:
                param.grad.data = param.grad.data.float()


def trainable_snapshot(model: Any) -> dict[str, Any]:
    return {name: p.detach().float().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def trainable_changed_l1(before: dict[str, Any], model: Any) -> float:
    total = 0.0
    for name, param in model.named_parameters():
        if name in before:
            total += float((param.detach().float().cpu() - before[name]).abs().sum())
    return total


def trainable_grad_l1(model: Any) -> float:
    total = 0.0
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            total += float(param.grad.detach().float().abs().sum().cpu())
    return total


def mps_memory(torch: Any) -> dict[str, int]:
    if not hasattr(torch, "mps"):
        return {}
    out: dict[str, int] = {}
    for name in ["current_allocated_memory", "driver_allocated_memory", "recommended_max_memory"]:
        fn = getattr(torch.mps, name, None)
        if fn is None:
            continue
        try:
            out[f"mps_{name}_bytes"] = int(fn())
        except Exception:
            pass
    return out


def release(torch: Any, *objs: Any) -> None:
    del objs
    gc.collect()
    if hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def patch_trl_runtime(torch: Any) -> None:
    """Keep TRL imports independent of optional DeepSpeed/FSDP quirks."""

    try:
        import accelerate.utils.other as accelerate_other

        accelerate_other.is_deepspeed_available = lambda: False
    except Exception:
        pass
    try:
        import torch.distributed.fsdp as torch_fsdp

        if not hasattr(torch_fsdp, "FSDPModule"):
            class RWKV7FSDPModuleCompat(torch.nn.Module):
                pass

            torch_fsdp.FSDPModule = RWKV7FSDPModuleCompat
    except Exception:
        pass


def configure_base_model(model: Any, args: argparse.Namespace, use_cache: bool) -> Any:
    model.config.use_cache = use_cache
    model.config.fuse_cross_entropy = False
    model.config.use_l2warp = False
    if hasattr(model.config, "attn_mode"):
        model.config.attn_mode = args.attn_mode
    for layer in getattr(getattr(model, "model", None), "layers", []):
        attn = getattr(layer, "attn", None)
        if hasattr(attn, "mode"):
            attn.mode = args.attn_mode
    return model


def lora_config(args: argparse.Namespace, targets: list[str] | None = None):
    from peft import LoraConfig

    return LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        target_modules=targets or LORA_TARGETS,
    )


def load_base_model(args: argparse.Namespace, torch: Any, device: str, dtype: Any, *, use_cache: bool = False):
    from transformers import AutoModelForCausalLM

    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=None,
    )
    configure_base_model(model, args, use_cache=use_cache)
    model.to(device)
    model.train()
    return model, time.perf_counter() - t0


def load_lora_model(args: argparse.Namespace, torch: Any, device: str, dtype: Any):
    from peft import get_peft_model

    model, load_s = load_base_model(args, torch, device, dtype, use_cache=False)
    model = get_peft_model(model, lora_config(args))
    keep_trainable_params_fp32(model)
    model.train()
    return model, load_s


class PromptDataset:
    def __init__(self, tokenizer: Any, max_length: int, repeats: int):
        self.rows = []
        for text in PROMPTS * max(1, int(repeats)):
            enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
            row = {k: v[0] for k, v in enc.items()}
            row["labels"] = row["input_ids"].clone()
            self.rows.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {k: v.clone() for k, v in self.rows[idx].items()}


def collate(tokenizer: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [r["labels"] for r in rows]
    inputs = [{k: v for k, v in r.items() if k != "labels"} for r in rows]
    batch = tokenizer.pad(inputs, return_tensors="pt")
    label_batch = tokenizer.pad({"input_ids": labels}, return_tensors="pt")["input_ids"]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    label_batch[label_batch == pad_id] = -100
    batch["labels"] = label_batch
    return batch


def base_row(args: argparse.Namespace, model: Any, device: str, dtype: Any) -> dict[str, Any]:
    cfg = getattr(model, "config", None)
    return {
        "model": Path(args.model).name,
        "model_size_label": infer_model_size_label(args.model, args.model_size_label),
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "attn_mode": args.attn_mode,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "hidden_size": getattr(cfg, "hidden_size", None),
        "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
        "head_dim": getattr(cfg, "head_dim", None),
        "backend_class": model.__class__.__name__,
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "total_params": int(sum(p.numel() for p in model.parameters())),
    }


def run_manual_lora(args: argparse.Namespace, torch: Any, tokenizer: Any, device: str, dtype: Any) -> tuple[dict[str, Any], Any]:
    model, load_s = load_lora_model(args, torch, device, dtype)
    before = trainable_snapshot(model)
    if not before:
        raise AssertionError("expected LoRA/trainable parameters")
    batch = tokenizer(PROMPTS[0], truncation=True, max_length=args.max_length, return_tensors="pt")
    batch = tensor_to_device(batch, device)
    labels = batch["input_ids"].clone()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    t0 = time.perf_counter()
    out = model(**batch, labels=labels, use_cache=False)
    loss = float(out.loss.detach().cpu())
    if not math.isfinite(loss):
        raise AssertionError(f"non-finite manual LoRA loss {loss}")
    out.loss.backward()
    grad_l1 = trainable_grad_l1(model)
    if grad_l1 <= 0:
        raise AssertionError("expected non-zero LoRA gradients")
    opt.step()
    opt.zero_grad(set_to_none=True)
    changed = trainable_changed_l1(before, model)
    if changed <= 0:
        raise AssertionError("expected LoRA parameter update")
    elapsed = time.perf_counter() - t0
    row = {
        "axis": "apple_silicon_model_peft_lora_train",
        "status": "pass",
        **base_row(args, model, device, dtype),
        "prompt_tokens": int(batch["input_ids"].shape[1]),
        "loss": round(loss, 6),
        "grad_l1": round(grad_l1, 6),
        "changed_l1": round(changed, 6),
        "load_s": round(load_s, 4),
        "elapsed_s": round(elapsed, 4),
        **mps_memory(torch),
    }
    return row, model


def run_trainer_lora(args: argparse.Namespace, torch: Any, tokenizer: Any, device: str, dtype: Any) -> tuple[dict[str, Any], Any]:
    from transformers import Trainer, TrainingArguments

    model, load_s = load_lora_model(args, torch, device, dtype)
    before = trainable_snapshot(model)
    if not before:
        raise AssertionError("expected LoRA/trainable parameters")
    dataset = PromptDataset(tokenizer, args.max_length, args.dataset_repeats)
    targs = TrainingArguments(
        output_dir=tempfile.mkdtemp(prefix="apple_model_lora_trainer_"),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        disable_tqdm=True,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        use_cpu=False,
        fp16=False,
        bf16=False,
        optim="adamw_torch",
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=dataset,
        data_collator=lambda rows: collate(tokenizer, rows),
        processing_class=tokenizer,
    )
    t0 = time.perf_counter()
    result = trainer.train()
    elapsed = time.perf_counter() - t0
    loss = float(result.training_loss)
    if not math.isfinite(loss):
        raise AssertionError(f"non-finite Trainer LoRA loss {loss}")
    changed = trainable_changed_l1(before, model)
    if changed <= 0:
        raise AssertionError("expected Trainer LoRA parameter update")
    metrics = dict(getattr(result, "metrics", {}) or {})
    row = {
        "axis": "apple_silicon_model_peft_lora_trainer",
        "status": "pass",
        **base_row(args, model, device, dtype),
        "dataset_rows": len(dataset),
        "training_loss": round(loss, 6),
        "changed_l1": round(changed, 6),
        "load_s": round(load_s, 4),
        "elapsed_s": round(elapsed, 4),
        "train_runtime_s": metrics.get("train_runtime"),
        "train_samples_per_second": metrics.get("train_samples_per_second"),
        "train_steps_per_second": metrics.get("train_steps_per_second"),
        **mps_memory(torch),
    }
    return row, model


def run_trl_sft_lora(args: argparse.Namespace, torch: Any, tokenizer: Any, device: str, dtype: Any) -> tuple[dict[str, Any], Any]:
    patch_trl_runtime(torch)
    from datasets import Dataset as HFDataset
    from trl import SFTConfig, SFTTrainer

    model, load_s = load_lora_model(args, torch, device, dtype)
    dataset = HFDataset.from_dict({"text": PROMPTS * max(1, int(args.dataset_repeats))})
    with tempfile.TemporaryDirectory(prefix="apple_model_lora_sft_") as out_dir:
        sft_args = SFTConfig(
            output_dir=out_dir,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=1,
            learning_rate=args.lr,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            remove_unused_columns=True,
            disable_tqdm=True,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            use_cpu=False,
            fp16=False,
            bf16=False,
            gradient_checkpointing=False,
            optim="adamw_torch",
            max_length=args.max_length,
            dataset_text_field="text",
            packing=False,
            # TRL 1.7 may default to chunked loss paths that patch the inner
            # backbone. This smoke validates the standard CausalLM labels ->
            # loss contract used by HF/PEFT scripts.
            loss_type="nll",
        )
        trainer = SFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
        before = trainable_snapshot(trainer.model)
        if not before:
            raise AssertionError("expected TRL SFT LoRA/trainable parameters")
        t0 = time.perf_counter()
        result = trainer.train()
        elapsed = time.perf_counter() - t0
    loss = float(result.training_loss)
    if not math.isfinite(loss):
        raise AssertionError(f"non-finite TRL SFT LoRA loss {loss}")
    changed = trainable_changed_l1(before, trainer.model)
    if changed <= 0:
        raise AssertionError("expected TRL SFT LoRA parameter update")
    metrics = dict(getattr(result, "metrics", {}) or {})
    row = {
        "axis": "apple_silicon_model_peft_lora_trl_sft",
        "status": "pass",
        **base_row(args, trainer.model, device, dtype),
        "dataset_rows": int(dataset.num_rows),
        "training_loss": round(loss, 6),
        "changed_l1": round(changed, 6),
        "load_s": round(load_s, 4),
        "elapsed_s": round(elapsed, 4),
        "train_runtime_s": metrics.get("train_runtime"),
        "train_samples_per_second": metrics.get("train_samples_per_second"),
        "train_steps_per_second": metrics.get("train_steps_per_second"),
        **mps_memory(torch),
    }
    return row, trainer.model


def run_trl_dpo_lora(args: argparse.Namespace, torch: Any, tokenizer: Any, device: str, dtype: Any) -> tuple[dict[str, Any], Any]:
    patch_trl_runtime(torch)
    from datasets import Dataset as HFDataset
    from trl import DPOConfig, DPOTrainer

    model, load_s = load_base_model(args, torch, device, dtype, use_cache=False)
    dataset = HFDataset.from_dict(
        {
            "prompt": PROMPTS * max(1, int(args.dataset_repeats)),
            "chosen": [" Hello!", " one two."] * max(1, int(args.dataset_repeats)),
            "rejected": [" Bye.", " three four."] * max(1, int(args.dataset_repeats)),
        }
    )
    with tempfile.TemporaryDirectory(prefix="apple_model_lora_dpo_") as out_dir:
        dpo_args = DPOConfig(
            output_dir=out_dir,
            max_steps=args.max_steps,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=1,
            learning_rate=args.lr,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            remove_unused_columns=False,
            disable_tqdm=True,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            use_cpu=False,
            fp16=False,
            bf16=False,
            gradient_checkpointing=False,
            optim="adamw_torch",
            max_length=max(16, args.max_length),
        )
        trainer = DPOTrainer(
            model=model,
            args=dpo_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=lora_config(args),
        )
        keep_trainable_params_fp32(trainer.model)
        before = trainable_snapshot(trainer.model)
        if not before:
            raise AssertionError("expected TRL DPO LoRA/trainable parameters")
        t0 = time.perf_counter()
        result = trainer.train()
        elapsed = time.perf_counter() - t0
    loss = float(result.training_loss)
    if not math.isfinite(loss):
        raise AssertionError(f"non-finite TRL DPO LoRA loss {loss}")
    changed = trainable_changed_l1(before, trainer.model)
    if changed <= 0:
        raise AssertionError("expected TRL DPO LoRA parameter update")
    metrics = dict(getattr(result, "metrics", {}) or {})
    row = {
        "axis": "apple_silicon_model_peft_lora_trl_dpo",
        "status": "pass",
        **base_row(args, trainer.model, device, dtype),
        "dataset_rows": int(dataset.num_rows),
        "dpo_max_length": max(16, args.max_length),
        "training_loss": round(loss, 6),
        "changed_l1": round(changed, 6),
        "load_s": round(load_s, 4),
        "elapsed_s": round(elapsed, 4),
        "train_runtime_s": metrics.get("train_runtime"),
        "train_samples_per_second": metrics.get("train_samples_per_second"),
        "train_steps_per_second": metrics.get("train_steps_per_second"),
        **mps_memory(torch),
    }
    return row, trainer.model


def grpo_reward_func(prompts: list[Any], completions: list[Any], **_: Any) -> list[float]:
    # Deterministic non-constant rewards keep the smoke focused on
    # Trainer/model compatibility while still producing a parameter update.
    return [float(i % 2) for i, _ in enumerate(completions)]


def run_trl_grpo_lora(args: argparse.Namespace, torch: Any, tokenizer: Any, device: str, dtype: Any) -> tuple[dict[str, Any], Any]:
    patch_trl_runtime(torch)
    from datasets import Dataset as HFDataset
    from trl import GRPOConfig, GRPOTrainer

    model, load_s = load_base_model(args, torch, device, dtype, use_cache=True)
    dataset = HFDataset.from_dict({"prompt": ["Hi", "Count"] * max(1, int(args.dataset_repeats))})
    with tempfile.TemporaryDirectory(prefix="apple_model_lora_grpo_") as out_dir:
        grpo_args = GRPOConfig(
            output_dir=out_dir,
            max_steps=args.max_steps,
            per_device_train_batch_size=max(2, args.batch_size),
            gradient_accumulation_steps=1,
            learning_rate=args.lr,
            logging_steps=1,
            save_strategy="no",
            report_to=[],
            remove_unused_columns=False,
            disable_tqdm=True,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            use_cpu=False,
            fp16=False,
            bf16=False,
            gradient_checkpointing=False,
            optim="adamw_torch",
            max_completion_length=args.grpo_max_completion_length,
            num_generations=2,
            generation_batch_size=2,
        )
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=grpo_reward_func,
            args=grpo_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=lora_config(args, GRPO_LORA_TARGETS),
        )
        keep_trainable_params_fp32(trainer.model)
        before = trainable_snapshot(trainer.model)
        if not before:
            raise AssertionError("expected TRL GRPO LoRA/trainable parameters")
        t0 = time.perf_counter()
        result = trainer.train()
        elapsed = time.perf_counter() - t0
    loss = float(result.training_loss)
    if not math.isfinite(loss):
        raise AssertionError(f"non-finite TRL GRPO LoRA loss {loss}")
    changed = trainable_changed_l1(before, trainer.model)
    if changed <= 0:
        raise AssertionError("expected TRL GRPO LoRA parameter update")
    metrics = dict(getattr(result, "metrics", {}) or {})
    row = {
        "axis": "apple_silicon_model_peft_lora_trl_grpo",
        "status": "pass",
        **base_row(args, trainer.model, device, dtype),
        "batch_size": max(2, args.batch_size),
        "dataset_rows": int(dataset.num_rows),
        "max_completion_length": args.grpo_max_completion_length,
        "training_loss": round(loss, 6),
        "changed_l1": round(changed, 6),
        "load_s": round(load_s, 4),
        "elapsed_s": round(elapsed, 4),
        "train_runtime_s": metrics.get("train_runtime"),
        "train_samples_per_second": metrics.get("train_samples_per_second"),
        "train_steps_per_second": metrics.get("train_steps_per_second"),
        **mps_memory(torch),
    }
    return row, trainer.model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Converted RWKV-7 HF model directory")
    ap.add_argument("--model-size-label", default="")
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--attn-mode", default="chunk", choices=["chunk", "fused_recurrent"])
    ap.add_argument("--max-length", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=1)
    ap.add_argument("--dataset-repeats", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grpo-max-completion-length", type=int, default=1)
    ap.add_argument("--lora-r", type=int, default=4)
    ap.add_argument("--lora-alpha", type=int, default=8)
    ap.add_argument("--backend", choices=["manual", "trainer", "trl_sft", "trl_dpo", "trl_grpo", "rl", "both", "all"], default="both")
    ap.add_argument("--results", default="")
    ap.add_argument("--require-apple", action="store_true")
    ap.add_argument("--require-peft", action="store_true")
    ap.add_argument("--require-trl", action="store_true")
    args = ap.parse_args()

    if not is_apple_silicon():
        row = {
            "axis": "apple_silicon_model_training_smoke",
            "status": "skip",
            "reason": "not Darwin/arm64",
            "platform": platform.platform(),
            "machine": platform.machine(),
            "model": Path(args.model).name,
        }
        emit(args.results, row)
        if args.require_apple:
            raise SystemExit(2)
        return 0

    if importlib.util.find_spec("peft") is None:
        row = {"axis": "apple_silicon_model_training_smoke", "status": "skip", "reason": "peft missing"}
        emit(args.results, row)
        if args.require_peft:
            raise SystemExit(3)
        return 0

    needs_trl = args.backend in {"trl_sft", "trl_dpo", "trl_grpo", "rl", "all"}
    if needs_trl and (importlib.util.find_spec("trl") is None or importlib.util.find_spec("datasets") is None):
        row = {"axis": "apple_silicon_model_trl_sft", "status": "skip", "reason": "trl or datasets missing"}
        emit(args.results, row)
        if args.require_trl:
            raise SystemExit(4)
        return 0

    import torch
    from transformers import AutoTokenizer

    device = choose_device(torch, args.device)
    dtype = dtype_for(torch, args.dtype)
    env = {
        "axis": "apple_silicon_model_training_env",
        "status": "info",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "chip": darwin_sysctl("machdep.cpu.brand_string"),
        "memory_gb": apple_memory_gb(),
        "torch": getattr(torch, "__version__", "unknown"),
        "transformers": package_version("transformers"),
        "peft": package_version("peft"),
        "trl": package_version("trl"),
        "datasets": package_version("datasets"),
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "device": device,
        "dtype": args.dtype,
        "model": Path(args.model).name,
        "model_size_label": infer_model_size_label(args.model, args.model_size_label),
        **mps_memory(torch),
    }
    emit(args.results, env)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rows: list[dict[str, Any]] = []
    if args.backend in {"manual", "both", "all"}:
        row, model = run_manual_lora(args, torch, tokenizer, device, dtype)
        rows.append(row)
        emit(args.results, row)
        del model
        release(torch)
    if args.backend in {"trainer", "both", "all"}:
        row, model = run_trainer_lora(args, torch, tokenizer, device, dtype)
        rows.append(row)
        emit(args.results, row)
        del model
        release(torch)
    if args.backend in {"trl_sft", "all"}:
        row, model = run_trl_sft_lora(args, torch, tokenizer, device, dtype)
        rows.append(row)
        emit(args.results, row)
        del model
        release(torch)
    if args.backend in {"trl_dpo", "rl", "all"}:
        row, model = run_trl_dpo_lora(args, torch, tokenizer, device, dtype)
        rows.append(row)
        emit(args.results, row)
        del model
        release(torch)
    if args.backend in {"trl_grpo", "rl", "all"}:
        row, model = run_trl_grpo_lora(args, torch, tokenizer, device, dtype)
        rows.append(row)
        emit(args.results, row)
        del model
        release(torch)

    print("APPLE SILICON MODEL TRAINING SMOKE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
