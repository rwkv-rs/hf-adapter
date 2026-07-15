#!/usr/bin/env python3
"""Real-checkpoint Apple MPS acceptance for common PEFT methods.

LoRA, AdaLoRA, IA3 and prompt tuning exercise train/save/reload. Prefix
tuning is rejected deliberately: PEFT's transformer-style prefix encoder
produces a KV ``DynamicCache`` while RWKV-7 consumes a recurrent state cache.
The rejection itself is part of the public capability contract.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import torch
from peft import (
    AdaLoraConfig,
    IA3Config,
    LoraConfig,
    PeftModel,
    PrefixTuningConfig,
    PromptTuningConfig,
    get_peft_model,
)
from transformers import AutoTokenizer

from rwkv7_hf.native_model import NativeRWKV7ForCausalLM


LINEAR_TARGETS = ["r_proj", "k_proj", "v_proj", "o_proj", "key", "value"]


def git(root: Path, *args: str) -> str | None:
    proc = subprocess.run(
        ["git", *args], cwd=root, text=True, capture_output=True, check=False
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def runtime_metadata(root: Path) -> dict[str, Any]:
    chip = None
    if platform.system() == "Darwin":
        proc = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True,
            capture_output=True,
            check=False,
        )
        chip = proc.stdout.strip() or None
    dirty = git(root, "status", "--porcelain", "--untracked-files=no")
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "chip": chip,
        "python": platform.python_version(),
        "torch_version": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "git_commit": git(root, "rev-parse", "HEAD"),
        "git_dirty": bool(dirty) if dirty is not None else None,
        "process_isolated": True,
    }


def emit(path: str, row: dict[str, Any]) -> None:
    print(json.dumps(row, ensure_ascii=False))
    if path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def release() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def load_native(model_path: str, device: str) -> NativeRWKV7ForCausalLM:
    model = NativeRWKV7ForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32
    )
    model.config.use_cache = False
    return model.to(device)


def config_for(method: str, model) -> Any:
    common = {"task_type": "CAUSAL_LM"}
    if method == "lora":
        return LoraConfig(
            **common,
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=LINEAR_TARGETS,
        )
    if method == "adalora":
        return AdaLoraConfig(
            **common,
            target_r=2,
            init_r=4,
            total_step=4,
            tinit=0,
            tfinal=0,
            deltaT=1,
            target_modules=LINEAR_TARGETS,
        )
    if method == "ia3":
        return IA3Config(
            **common,
            target_modules=["k_proj", "v_proj", "value"],
            feedforward_modules=["value"],
        )
    hidden = int(model.config.hidden_size)
    layers = int(model.config.num_hidden_layers)
    heads = int(model.config.num_heads)
    if method == "prompt_tuning":
        return PromptTuningConfig(
            **common,
            num_virtual_tokens=4,
            token_dim=hidden,
            num_layers=layers,
            num_attention_heads=heads,
        )
    if method == "prefix_tuning":
        return PrefixTuningConfig(
            **common,
            num_virtual_tokens=4,
            token_dim=hidden,
            num_layers=layers,
            num_attention_heads=heads,
            encoder_hidden_size=hidden,
        )
    raise ValueError(f"unknown method: {method}")


def trainable_snapshot(model) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().float().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def max_delta(before: dict[str, torch.Tensor], model) -> float:
    return max(
        (
            float((param.detach().float().cpu() - before[name]).abs().max().item())
            for name, param in model.named_parameters()
            if param.requires_grad and name in before
        ),
        default=0.0,
    )


def logits(model, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        out = model(**batch, use_cache=False)
    got = out.logits.detach().float().cpu()
    if not bool(torch.isfinite(got).all()):
        raise AssertionError("non-finite logits")
    return got


def run_supported(
    method: str,
    args: argparse.Namespace,
    tokenizer,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    base = load_native(args.model, args.device)
    model = get_peft_model(base, config_for(method, base)).to(args.device)
    parameter_device = str(next(model.parameters()).device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    before = trainable_snapshot(model)
    if not before or trainable <= 0:
        raise AssertionError(f"{method}: no trainable parameters")

    encoded = tokenizer(args.text, return_tensors="pt", add_special_tokens=False)
    batch = {key: value.to(args.device) for key, value in encoded.items()}
    labels = batch["input_ids"].clone()
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad), lr=args.learning_rate
    )
    optimizer.zero_grad(set_to_none=True)
    out = model(**batch, labels=labels, use_cache=False)
    loss = float(out.loss.detach().cpu())
    if not math.isfinite(loss):
        raise AssertionError(f"{method}: non-finite loss")
    out.loss.backward()
    grad_norm_sq = sum(
        float(param.grad.detach().float().pow(2).sum().cpu())
        for param in model.parameters()
        if param.requires_grad and param.grad is not None
    )
    optimizer.step()
    if method == "adalora":
        model.base_model.update_and_allocate(1)
    update = max_delta(before, model)
    ref = logits(model, batch)

    with tempfile.TemporaryDirectory(prefix=f"rwkv7_{method}_") as adapter_dir:
        model.save_pretrained(adapter_dir)
        artifact_files = sorted(path.name for path in Path(adapter_dir).iterdir())
        del optimizer, out, model, base, before
        release()

        fresh = load_native(args.model, args.device)
        reloaded = PeftModel.from_pretrained(fresh, adapter_dir).to(args.device)
        restored = logits(reloaded, batch)
        reload_max_abs = float((ref - restored).abs().max().item())

    ok = (
        parameter_device.startswith(args.device)
        and grad_norm_sq > 0.0
        and update > 0.0
        and reload_max_abs <= args.max_logit_diff
        and "adapter_config.json" in artifact_files
        and "adapter_model.safetensors" in artifact_files
    )
    row = {
        "axis": "mps_peft_method",
        "status": "pass" if ok else "fail",
        "support_status": "supported",
        "method": method,
        "model": Path(args.model).name,
        "model_path": args.model,
        "device": args.device,
        "parameter_device": parameter_device,
        "dtype": "fp32",
        "trainable_parameters": trainable,
        "train_loss": loss,
        "gradient_l2": math.sqrt(grad_norm_sq),
        "max_trainable_delta": update,
        "reload_logits_max_abs": reload_max_abs,
        "max_logit_diff": args.max_logit_diff,
        "artifact_files": artifact_files,
        **metadata,
    }
    del fresh, reloaded
    release()
    if not ok:
        raise AssertionError(row)
    return row


def run_prefix_contract(
    args: argparse.Namespace,
    tokenizer,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    base = load_native(args.model, args.device)
    model = get_peft_model(base, config_for("prefix_tuning", base)).to(args.device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoded = tokenizer(args.text, return_tensors="pt", add_special_tokens=False)
    batch = {key: value.to(args.device) for key, value in encoded.items()}
    reason = ""
    rejected = False
    try:
        model(**batch, labels=batch["input_ids"].clone(), use_cache=False)
    except TypeError as exc:
        reason = str(exc)
        rejected = "NativeRWKV7Cache" in reason and "recurrent cache" in reason
    row = {
        "axis": "mps_peft_method",
        "status": "pass" if rejected else "fail",
        "support_status": "unsupported",
        "method": "prefix_tuning",
        "model": Path(args.model).name,
        "model_path": args.model,
        "device": args.device,
        "parameter_device": str(next(model.parameters()).device),
        "dtype": "fp32",
        "trainable_parameters": trainable,
        "expected_rejection": rejected,
        "technical_reason": reason,
        "capability_contract": (
            "PEFT prefix tuning emits transformer KV DynamicCache; RWKV-7 requires "
            "NativeRWKV7Cache recurrent state. Prompt tuning is the supported virtual-token method."
        ),
        **metadata,
    }
    del model, base
    release()
    if not rejected:
        raise AssertionError(row)
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="mps", choices=["mps", "cpu"])
    parser.add_argument("--text", default="User: Explain RWKV briefly.\n\nAssistant: Recurrent state.")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-logit-diff", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--results", default="")
    args = parser.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required but unavailable")
    root = Path(__file__).resolve().parents[1]
    metadata = runtime_metadata(root)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    methods = ["lora", "adalora", "ia3", "prompt_tuning"]
    rows = [run_supported(method, args, tokenizer, metadata) for method in methods]
    rows.append(run_prefix_contract(args, tokenizer, metadata))
    for row in rows:
        emit(args.results, row)
    summary = {
        "axis": "mps_peft_methods_summary",
        "status": "pass",
        "model": Path(args.model).name,
        "supported_methods": methods,
        "unsupported_methods": ["prefix_tuning"],
        "pass_rows": sum(row["status"] == "pass" for row in rows),
        "total_rows": len(rows),
        **metadata,
    }
    emit(args.results, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
