#!/usr/bin/env python3
# coding=utf-8
"""Apple Silicon / MPS HF Trainer smoke for the native RWKV-7 backend.

This is a tiny synthetic test: no model files and no datasets dependency. It
proves that HF Trainer can drive the native backend on MPS, and optionally that
PEFT LoRA parameters update under Trainer. Non-Apple hosts emit a skip row unless
--require-apple is set.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import subprocess
import tempfile
import time
from importlib import metadata
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("RWKV7_NATIVE_MODEL", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def append_result(path: str, row: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")



def mps_backend(torch: Any) -> Any | None:
    return getattr(getattr(torch, "backends", None), "mps", None)


def mps_is_available(torch: Any) -> bool:
    mps = mps_backend(torch)
    if mps is None or not hasattr(mps, "is_available"):
        return False
    try:
        return bool(mps.is_available())
    except Exception:
        return False


def mps_is_built(torch: Any) -> bool:
    mps = mps_backend(torch)
    if mps is None or not hasattr(mps, "is_built"):
        return False
    try:
        return bool(mps.is_built())
    except Exception:
        return False


def choose_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        if requested == "mps" and not mps_is_available(torch):
            raise RuntimeError("requested --device mps but MPS is unavailable")
        return requested
    return "mps" if mps_is_available(torch) else "cpu"


def dtype_for(torch: Any, name: str) -> Any:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


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


def build_tiny_model(torch: Any, device: str, dtype: Any):
    from rwkv7_hf.native_model import NativeRWKV7Config, NativeRWKV7ForCausalLM

    torch.manual_seed(20260704)
    cfg = NativeRWKV7Config(
        vocab_size=43,
        hidden_size=16,
        num_hidden_layers=2,
        head_dim=4,
        intermediate_size=32,
        decay_low_rank_dim=4,
        gate_low_rank_dim=4,
        a_low_rank_dim=4,
        v_low_rank_dim=4,
        use_cache=False,
    )
    model = NativeRWKV7ForCausalLM(cfg).to(device=device, dtype=dtype)
    model.config.use_cache = False
    return model


def attach_peft_lora(model: Any):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        task_type="CAUSAL_LM",
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        target_modules=["r_proj", "k_proj", "v_proj", "o_proj", "key", "value"],
    )
    return get_peft_model(model, cfg)


class ToyDataset:
    def __init__(self, torch: Any, vocab_size: int, length: int, rows: int):
        self.rows = []
        for i in range(rows):
            ids = (torch.arange(length, dtype=torch.long) + i) % vocab_size
            self.rows.append({"input_ids": ids, "labels": ids.clone()})

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.rows[idx]


def collate(torch: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "input_ids": torch.stack([r["input_ids"] for r in rows], dim=0),
        "labels": torch.stack([r["labels"] for r in rows], dim=0),
    }


def trainable_snapshot(model: Any) -> dict[str, Any]:
    return {name: p.detach().float().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def changed_l1(before: dict[str, Any], model: Any) -> float:
    total = 0.0
    for name, param in model.named_parameters():
        if name in before:
            total += float((param.detach().float().cpu() - before[name]).abs().sum())
    return total


def run_trainer(torch: Any, model: Any, args: argparse.Namespace, kind: str) -> dict[str, Any]:
    from transformers import Trainer, TrainingArguments

    dataset = ToyDataset(torch, model.config.vocab_size, args.length, rows=max(args.batch_size * args.max_steps * 2, 4))
    before = trainable_snapshot(model)
    tmpdir = tempfile.mkdtemp(prefix=f"apple_{kind}_trainer_")
    targs = TrainingArguments(
        output_dir=tmpdir,
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
        use_cpu=False,
        fp16=False,
        bf16=False,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=dataset,
        data_collator=lambda rows: collate(torch, rows),
    )
    start = time.perf_counter()
    result = trainer.train()
    elapsed = time.perf_counter() - start
    changed = changed_l1(before, model)
    if changed <= 0:
        raise AssertionError(f"{kind}: expected trainable parameter update")
    loss = float(result.training_loss)
    if not (loss == loss and loss < float("inf")):
        raise AssertionError(f"{kind}: non-finite training loss {loss}")
    return {
        "axis": f"apple_silicon_{kind}_trainer",
        "status": "pass",
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "length": args.length,
        "training_loss": round(loss, 6),
        "changed_l1": round(changed, 6),
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "elapsed_s": round(elapsed, 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--length", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--results", default="")
    ap.add_argument("--require-apple", action="store_true")
    ap.add_argument("--require-peft", action="store_true")
    ap.add_argument("--skip-peft", action="store_true")
    args = ap.parse_args()

    if not is_apple_silicon():
        row = {
            "axis": "apple_silicon_trainer_smoke",
            "status": "skip",
            "reason": "not Darwin/arm64",
            "platform": platform.platform(),
            "machine": platform.machine(),
        }
        print(json.dumps(row, ensure_ascii=False))
        append_result(args.results, row)
        if args.require_apple:
            raise SystemExit(2)
        return 0

    import torch

    device = choose_device(torch, args.device)
    dtype = dtype_for(torch, args.dtype)
    env = {
        "axis": "apple_silicon_trainer_env",
        "status": "info",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "chip": darwin_sysctl("machdep.cpu.brand_string"),
        "memory_gb": apple_memory_gb(),
        "torch": getattr(torch, "__version__", "unknown"),
        "transformers": package_version("transformers"),
        "peft": package_version("peft"),
        "mps_built": mps_is_built(torch),
        "mps_available": mps_is_available(torch),
        "device": device,
        "dtype": args.dtype,
        "peft_available": importlib.util.find_spec("peft") is not None,
    }
    print(json.dumps(env, ensure_ascii=False))
    append_result(args.results, env)

    model = build_tiny_model(torch, device, dtype)
    row = run_trainer(torch, model, args, "native")
    row.update({"device": device, "dtype": str(dtype).replace("torch.", "")})
    print(json.dumps(row, ensure_ascii=False))
    append_result(args.results, row)

    if not args.skip_peft:
        if importlib.util.find_spec("peft") is None:
            row = {"axis": "apple_silicon_peft_trainer", "status": "skip", "reason": "peft missing"}
            print(json.dumps(row, ensure_ascii=False))
            append_result(args.results, row)
            if args.require_peft:
                raise SystemExit(3)
        else:
            peft_model = attach_peft_lora(build_tiny_model(torch, device, dtype))
            row = run_trainer(torch, peft_model, args, "peft_lora")
            row.update({"device": device, "dtype": str(dtype).replace("torch.", "")})
            print(json.dumps(row, ensure_ascii=False))
            append_result(args.results, row)

    print("APPLE SILICON TRAINER SMOKE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
