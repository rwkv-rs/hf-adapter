#!/usr/bin/env python3
# coding=utf-8
"""Apple Silicon / MPS training smoke for the native RWKV-7 backend.

This is intentionally tiny and weight-free. It proves that the FLA-free native
model can compute a CausalLM loss, backpropagate on MPS, and optionally accept
PEFT LoRA adapters on Apple Silicon. On non-Apple hosts it emits a skip row
unless --require-apple is set.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("RWKV7_NATIVE_MODEL", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")


from apple_silicon_utils import (
    append_result,
    choose_device,
    dtype_for,
    is_apple_silicon,
    mps_is_available,
    mps_is_built,
)


def build_tiny_model(torch: Any, device: str, dtype: Any):
    from rwkv7_hf.native_model import NativeRWKV7Config, NativeRWKV7ForCausalLM

    torch.manual_seed(20260704)
    cfg = NativeRWKV7Config(
        vocab_size=41,
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


def grad_l1(model: Any) -> float:
    total = 0.0
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            total += float(param.grad.detach().float().abs().sum().cpu())
    return total


def changed_l1(before: dict[str, Any], model: Any) -> float:
    total = 0.0
    for name, param in model.named_parameters():
        if name in before:
            total += float((param.detach().float().cpu() - before[name]).abs().sum())
    return total


def train_one_step(torch: Any, model: Any, device: str, lr: float, batch_size: int, length: int) -> dict[str, Any]:
    model.train()
    input_ids = (torch.arange(batch_size * length, device=device, dtype=torch.long).view(batch_size, length) % model.config.vocab_size)
    labels = input_ids.clone()
    before = {name: p.detach().float().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    t0 = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    out = model(input_ids=input_ids, labels=labels, use_cache=False)
    loss = out.loss
    if loss is None or not torch.isfinite(loss.detach()).item():
        raise AssertionError(f"non-finite loss: {loss}")
    loss.backward()
    grad = grad_l1(model)
    if grad <= 0:
        raise AssertionError("expected non-zero gradient")
    opt.step()
    changed = changed_l1(before, model)
    if changed <= 0:
        raise AssertionError("expected trainable parameter update")
    elapsed = time.perf_counter() - t0
    return {
        "loss": round(float(loss.detach().cpu()), 6),
        "grad_l1": round(grad, 6),
        "changed_l1": round(changed, 6),
        "elapsed_s": round(elapsed, 4),
    }


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--length", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--results", default="")
    ap.add_argument("--require-apple", action="store_true")
    ap.add_argument("--require-peft", action="store_true")
    ap.add_argument("--skip-peft", action="store_true")
    args = ap.parse_args()

    if not is_apple_silicon():
        row = {
            "axis": "apple_silicon_training_smoke",
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
        "axis": "apple_silicon_training_env",
        "status": "info",
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": getattr(torch, "__version__", "unknown"),
        "mps_built": mps_is_built(torch),
        "mps_available": mps_is_available(torch),
        "device": device,
        "dtype": args.dtype,
        "peft_available": importlib.util.find_spec("peft") is not None,
    }
    print(json.dumps(env, ensure_ascii=False))
    append_result(args.results, env)

    base = build_tiny_model(torch, device, dtype)
    metrics = train_one_step(torch, base, device, args.lr, args.batch_size, args.length)
    row = {
        "axis": "apple_silicon_tiny_train",
        "status": "pass",
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "batch_size": args.batch_size,
        "length": args.length,
        **metrics,
    }
    print(json.dumps(row, ensure_ascii=False))
    append_result(args.results, row)

    if not args.skip_peft:
        if importlib.util.find_spec("peft") is None:
            row = {"axis": "apple_silicon_peft_lora_train", "status": "skip", "reason": "peft missing"}
            print(json.dumps(row, ensure_ascii=False))
            append_result(args.results, row)
            if args.require_peft:
                raise SystemExit(3)
        else:
            peft_model = attach_peft_lora(build_tiny_model(torch, device, dtype))
            metrics = train_one_step(torch, peft_model, device, args.lr, args.batch_size, args.length)
            trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
            row = {
                "axis": "apple_silicon_peft_lora_train",
                "status": "pass",
                "device": device,
                "dtype": str(dtype).replace("torch.", ""),
                "batch_size": args.batch_size,
                "length": args.length,
                "trainable_params": int(trainable),
                **metrics,
            }
            print(json.dumps(row, ensure_ascii=False))
            append_result(args.results, row)

    print("APPLE SILICON TRAINING SMOKE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
