#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Keep the V100 training smoke path out of Dynamo/Triton compile trouble.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
# Some pip/conda CUDA runtimes expose torch CUDA without a full CUDA_HOME.
# DeepSpeed ZeRO does not need the fp_quantizer CUDA_HOME compatibility probe,
# so keep this smoke runnable in those environments.
os.environ.setdefault("DS_IGNORE_CUDA_DETECTION", "1")


def ensure_single_process_distributed_env() -> None:
    """Avoid DeepSpeed falling back to mpi4py for one-process smoke runs."""
    if any(k in os.environ for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK")):
        return
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")


PROMPTS = [
    "User: Say hello.\n\nAssistant: Hello!",
    "User: Count to three.\n\nAssistant: one two three.",
]

TRAIN_DTYPES = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}


def infer_model_size_label(model_path: str, explicit: str = "") -> str | None:
    if explicit:
        return explicit.lower()
    match = re.search(r"(\d+(?:\.\d+)?b)", Path(model_path).name.lower())
    return match.group(1) if match else None


def model_metadata(args: argparse.Namespace, model: Any | None = None) -> dict[str, Any]:
    cfg = getattr(model, "config", None)
    return {
        "model_name": Path(args.model).name,
        "model_size_label": infer_model_size_label(args.model, getattr(args, "model_size_label", "")),
        "hf_model_dir": args.model,
        "hidden_size": getattr(cfg, "hidden_size", None),
        "intermediate_size": getattr(cfg, "intermediate_size", None),
        "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
        "head_dim": getattr(cfg, "head_dim", None),
        "num_heads": getattr(cfg, "num_heads", None),
    }


def optional_torch() -> Any | None:
    if importlib.util.find_spec("torch") is None:
        return None
    try:
        import torch
    except Exception:
        return None
    return torch


def require_training_deps() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    return torch, LoraConfig, get_peft_model, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def single_group_trainer_cls(Trainer: Any, torch: Any) -> Any:
    class SingleGroupTrainer(Trainer):
        def create_optimizer(self):
            if self.optimizer is None:
                # Keep ZeRO smoke optimizer state aligned with one trainable group.
                params = [p for p in self.model.parameters() if p.requires_grad]
                self.optimizer = torch.optim.AdamW(params, lr=self.args.learning_rate)
            return self.optimizer

    return SingleGroupTrainer


def missing_training_deps() -> list[str]:
    missing = []
    for name in ("torch", "peft", "transformers"):
        if importlib.util.find_spec(name) is None:
            missing.append(name)
    return missing


def deepspeed_import_error() -> str | None:
    try:
        import deepspeed  # noqa: F401
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


class TokenDataset:
    def __init__(self, tokenizer, max_length: int, repeats: int = 1):
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


@dataclass
class CausalCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        labels = [f["labels"] for f in features]
        inputs = [{k: v for k, v in f.items() if k != "labels"} for f in features]
        batch = self.tokenizer.pad(inputs, return_tensors="pt")
        label_batch = self.tokenizer.pad({"input_ids": labels}, return_tensors="pt")["input_ids"]
        label_batch[label_batch == self.tokenizer.pad_token_id] = -100
        batch["labels"] = label_batch
        return batch


def device_name() -> str:
    torch = optional_torch()
    if torch is None:
        return "unknown-no-torch"
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def cuda_device_count() -> int:
    torch = optional_torch()
    if torch is None or not torch.cuda.is_available():
        return 0
    return int(torch.cuda.device_count())


def append_rows(path: str, rows: list[dict[str, Any]]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def train_torch_dtype(torch: Any, train_dtype: str) -> Any:
    return getattr(torch, TRAIN_DTYPES[train_dtype])


def use_fp16(torch: Any, train_dtype: str) -> bool:
    return torch.cuda.is_available() and train_dtype == "fp16"


def use_bf16(torch: Any, train_dtype: str) -> bool:
    return torch.cuda.is_available() and train_dtype == "bf16"


def materialize_trainable_param(param) -> Any | None:
    """Return a full CPU fp32 copy, including DeepSpeed ZeRO-3 shards."""
    try:
        if hasattr(param, "ds_id"):
            from deepspeed.utils import safe_get_full_fp32_param

            full = safe_get_full_fp32_param(param)
            if full is not None:
                return full.detach().float().cpu().clone()
    except Exception:
        pass
    if int(param.numel()) == 0:
        return None
    return param.detach().float().cpu().clone()


def trainable_snapshot(model) -> dict[str, Any]:
    out = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        value = materialize_trainable_param(param)
        if value is not None:
            out[name] = value
    return out


def max_trainable_delta(before: dict[str, Any], model) -> float:
    max_delta = 0.0
    compared = 0
    for name, param in model.named_parameters():
        if not param.requires_grad or name not in before:
            continue
        value = materialize_trainable_param(param)
        if value is None or tuple(value.shape) != tuple(before[name].shape):
            continue
        delta = (value - before[name]).abs().max().item()
        max_delta = max(max_delta, float(delta))
        compared += 1
    if compared == 0:
        raise RuntimeError("No trainable parameters could be compared after DeepSpeed training")
    return max_delta


def load_lora_model(model_path: str, attn_mode: str, train_dtype: str):
    torch, LoraConfig, get_peft_model, AutoModelForCausalLM, _, _, _ = require_training_deps()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=train_torch_dtype(torch, train_dtype),
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
    # DeepSpeed broadcasts rank 0 parameters during initialization. Give every
    # rank the same LoRA initialization first so the post-step delta measures
    # optimizer work rather than that expected initialization broadcast.
    torch.manual_seed(20260703)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260703)
    return get_peft_model(model, lora_cfg)


def metric(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    return float(value) if value is not None else None


def zero_config_path(config_dir: Path, stage: int) -> Path:
    path = config_dir / f"zero{stage}.json"
    if not path.exists():
        raise FileNotFoundError(f"DeepSpeed config not found: {path}")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    actual = int((cfg.get("zero_optimization") or {}).get("stage", -1))
    if actual != stage:
        raise ValueError(f"{path} has zero stage {actual}, expected {stage}")
    return path


def skip_row(args: argparse.Namespace, stage: int, reason: str) -> dict[str, Any]:
    return {
        "axis": "deepspeed_training_smoke",
        "backend": "hf_adapter",
        "trainer_backend": f"trainer_zero{stage}",
        "zero_stage": stage,
        "status": "skip",
        "reason": reason,
        "dtype": args.train_dtype,
        "train_dtype": args.train_dtype,
        "device": device_name(),
        **model_metadata(args),
        "cuda_device_count": cuda_device_count(),
        "distributed_world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
    }


def run_stage(args: argparse.Namespace, stage: int) -> dict[str, Any]:
    config_path = zero_config_path(Path(args.config_dir), stage)
    ensure_single_process_distributed_env()
    if importlib.util.find_spec("deepspeed") is None:
        if args.optional:
            return skip_row(args, stage, "deepspeed missing")
        raise RuntimeError("DeepSpeed is not installed")
    ds_error = deepspeed_import_error()
    if ds_error:
        if args.optional:
            return skip_row(args, stage, f"deepspeed import failed: {ds_error}")
        raise RuntimeError(f"DeepSpeed import failed: {ds_error}")
    missing = missing_training_deps()
    if missing:
        if args.optional:
            return skip_row(args, stage, f"training dependencies missing: {','.join(missing)}")
        raise RuntimeError(f"Training dependencies are not installed: {', '.join(missing)}")

    torch, _, _, _, AutoTokenizer, Trainer, TrainingArguments = require_training_deps()
    Trainer = single_group_trainer_cls(Trainer, torch)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_lora_model(args.model, args.attn_mode, args.train_dtype)
    dataset = TokenDataset(tok, args.max_length, repeats=args.dataset_repeats)
    collator = CausalCollator(tok)
    before = trainable_snapshot(model)
    assert before, "expected LoRA/trainable parameters"

    with tempfile.TemporaryDirectory(prefix=f"rwkv7_zero{stage}_smoke_") as out_dir:
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
            fp16=use_fp16(torch, args.train_dtype),
            bf16=use_bf16(torch, args.train_dtype),
            dataloader_num_workers=0,
            gradient_checkpointing=False,
            deepspeed=str(config_path),
        )
        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=dataset,
            data_collator=collator,
            processing_class=tok,
        )
        result = trainer.train()

    loss = float(result.training_loss)
    assert math.isfinite(loss), result.training_loss
    delta = max_trainable_delta(before, model)
    assert delta > 0.0, "DeepSpeed LoRA/trainable parameters did not update"
    metrics = dict(getattr(result, "metrics", {}) or {})
    return {
        "axis": "deepspeed_training_smoke",
        "backend": "hf_adapter",
        "trainer_backend": f"trainer_zero{stage}",
        "zero_stage": stage,
        "status": "pass",
        "dtype": args.train_dtype,
        "train_dtype": args.train_dtype,
        "device": device_name(),
        **model_metadata(args, model),
        "cuda_device_count": cuda_device_count(),
        "distributed_world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
        "attn_mode": args.attn_mode,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "max_steps": args.max_steps,
        "dataset_repeats": args.dataset_repeats,
        "max_length": args.max_length,
        "deepspeed_config": str(config_path),
        "train_loss": loss,
        "train_runtime_s": metric(metrics, "train_runtime"),
        "train_samples_per_second": metric(metrics, "train_samples_per_second"),
        "train_steps_per_second": metric(metrics, "train_steps_per_second"),
        "max_trainable_delta": delta,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-size-label", default="", help="Optional size label such as 0.4b; inferred from --model when omitted")
    ap.add_argument("--config-dir", default="configs/deepspeed")
    ap.add_argument("--zero-stage", choices=["2", "3", "both"], default="both")
    ap.add_argument("--attn-mode", default="fused_recurrent", choices=["chunk", "fused_recurrent"])
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--train-dtype", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--max-steps", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=1)
    ap.add_argument("--dataset-repeats", type=int, default=4)
    ap.add_argument("--optional", action="store_true", help="Append skip rows instead of failing when deepspeed is unavailable")
    ap.add_argument("--results", default="")
    args = ap.parse_args()
    if args.train_dtype is None:
        torch = optional_torch()
        args.train_dtype = "bf16" if torch is not None and torch.cuda.is_available() else "fp32"

    stages = [2, 3] if args.zero_stage == "both" else [int(args.zero_stage)]
    rows = [run_stage(args, stage) for stage in stages]
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    append_rows(args.results, rows)
    print("PASS" if all(row["status"] in {"pass", "skip"} for row in rows) else "FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
