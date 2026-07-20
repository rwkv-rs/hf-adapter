#!/usr/bin/env python3
# coding=utf-8
"""PEFT LoRA save/load/merge smoke for NativeRWKV7ForCausalLM.

The plain native PEFT smoke proves gradients flow. This test covers the user
workflow that usually follows training:

1. wrap native RWKV-7 with PEFT LoRA,
2. update adapter weights,
3. ``save_pretrained`` the adapter,
4. reload the adapter on a fresh native base model,
5. ``merge_and_unload`` and verify logits stay aligned.

Gate: adapter reload and merged model both reproduce the trained adapter logits.

  python tests/test_native_peft_save_load_merge.py --model <hf_dir>
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import re
import subprocess
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoTokenizer

from rwkv7_hf.native_model import NativeRWKV7ForCausalLM
from rwkv7_hf.training_precision import merged_logits_pass


def device_map_for(device: str):
    if not device.startswith("cuda"):
        return None
    if ":" in device:
        return {"": int(device.split(":", 1)[1])}
    return "cuda"


def first_param_device(model) -> torch.device:
    return next(model.parameters()).device


def load_native(model_dir: str, dtype: torch.dtype, device: str):
    model = NativeRWKV7ForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        device_map=device_map_for(device),
    )
    model.config.use_cache = False
    if not device.startswith("cuda"):
        model = model.to(device)
    return model


def lora_config() -> LoraConfig:
    return LoraConfig(
        task_type="CAUSAL_LM",
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        target_modules=["r_proj", "k_proj", "v_proj", "o_proj", "key", "value"],
    )


def release_cuda(*objs) -> None:
    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def infer_model_size_label(model_path: str, explicit: str = "") -> str | None:
    if explicit:
        return explicit.lower()
    match = re.search(r"(\d+(?:\.\d+)?b)", Path(model_path).name.lower())
    return match.group(1) if match else None


def runtime_metadata(parameter_device: str) -> dict:
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
    dirty = git("status", "--porcelain", "--untracked-files=no")
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "chip": chip,
        "python": platform.python_version(),
        "torch_version": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "parameter_device": parameter_device,
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(dirty) if dirty is not None else None,
        "process_isolated": True,
    }


def logits_for(model, tokenizer, text: str) -> torch.Tensor:
    model.eval()
    dev = first_param_device(model)
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    enc = {k: v.to(dev) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc, use_cache=False)
    logits = out.logits.detach().float().cpu()
    assert logits.isfinite().all(), "non-finite logits"
    return logits


def logit_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    diff = (reference - candidate).abs()
    lhs = reference.reshape(-1).double()
    rhs = candidate.reshape(-1).double()
    cosine = float(F.cosine_similarity(lhs.unsqueeze(0), rhs.unsqueeze(0)).item())
    top1 = reference.argmax(dim=-1) == candidate.argmax(dim=-1)
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "cosine": cosine,
        "top1_match_rate": float(top1.float().mean().item()),
    }


def generate_tail(model, tokenizer, text: str, tokens: int = 2) -> list[int]:
    model.eval()
    dev = first_param_device(model)
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    enc = {k: v.to(dev) for k, v in enc.items()}
    with torch.no_grad():
        generated = model.generate(
            **enc,
            max_new_tokens=tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or 0,
            eos_token_id=None,
        )
    return generated[0, -tokens:].detach().cpu().tolist()


def train_adapter_step(model, tokenizer, text: str, lr: float, steps: int) -> float:
    model.train()
    dev = first_param_device(model)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    last_loss = float("nan")
    for _ in range(steps):
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(dev) for k, v in enc.items()}
        labels = enc["input_ids"].clone()
        opt.zero_grad(set_to_none=True)
        out = model(**enc, labels=labels, use_cache=False)
        loss = out.loss
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())
    return last_loss


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-size-label", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-logit-diff", type=float, default=1e-4)
    ap.add_argument("--fp16-merge-max-abs", type=float, default=0.5)
    ap.add_argument("--fp16-merge-max-mean-abs", type=float, default=0.05)
    ap.add_argument("--fp16-merge-min-cosine", type=float, default=0.9999)
    ap.add_argument("--results", default="")
    args = ap.parse_args()

    dtype = torch.float32 if args.dtype == "fp32" else torch.float16
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    base = load_native(args.model, dtype, args.device)
    base_logits = logits_for(base, tok, "User: Hello.\n\nAssistant:")
    model = get_peft_model(base, lora_config())
    parameter_device = str(first_param_device(model))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable > 0, "expected LoRA trainable parameters"

    loss = train_adapter_step(
        model,
        tok,
        "User: Teach RWKV in one sentence.\n\nAssistant: RWKV keeps recurrent state.",
        args.lr,
        args.steps,
    )
    assert math.isfinite(loss), loss
    ref_logits = logits_for(model, tok, "User: Hello.\n\nAssistant:")
    trained_tail = generate_tail(model, tok, "User: Hello.\n\nAssistant:")
    trained_vs_base_diff = float((ref_logits - base_logits).abs().max().item())

    with tempfile.TemporaryDirectory(prefix="native_peft_adapter_") as adapter_dir:
        model.save_pretrained(adapter_dir)

        # Keep only one full base model resident at a time. This lets the same
        # save/load/merge contract run on 32GB V100s for 2.9B+ checkpoints.
        del model, base
        release_cuda()

        fresh = load_native(args.model, dtype, args.device)
        reloaded = PeftModel.from_pretrained(fresh, adapter_dir)
        reload_logits = logits_for(reloaded, tok, "User: Hello.\n\nAssistant:")
        reload_metrics = logit_metrics(ref_logits, reload_logits)

        # Exercise reversible in-place merge before the destructive
        # merge_and_unload path. Both directions must preserve the trained
        # adapter function within the declared tolerance.
        reloaded.merge_adapter()
        merged_inplace_logits = logits_for(reloaded, tok, "User: Hello.\n\nAssistant:")
        merged_inplace_metrics = logit_metrics(ref_logits, merged_inplace_logits)
        reloaded.unmerge_adapter()
        unmerged_logits = logits_for(reloaded, tok, "User: Hello.\n\nAssistant:")
        unmerge_metrics = logit_metrics(ref_logits, unmerged_logits)

        reloaded_tail = generate_tail(reloaded, tok, "User: Hello.\n\nAssistant:")

        merged = reloaded.merge_and_unload()
        merge_logits = logits_for(merged, tok, "User: Hello.\n\nAssistant:")
        merge_metrics = logit_metrics(ref_logits, merge_logits)
        merged_tail = generate_tail(merged, tok, "User: Hello.\n\nAssistant:")

        del fresh, reloaded, merged
        release_cuda()

        # Exercise GenerationMixin in a second fresh process-style load before
        # declaring the serialized adapter usable.
        reloaded_for_generate = PeftModel.from_pretrained(
            load_native(args.model, dtype, args.device), adapter_dir
        ).eval()
        generated_tail = generate_tail(reloaded_for_generate, tok, "User: Hello.\n\nAssistant:")

    greedy_match = trained_tail == reloaded_tail == generated_tail == merged_tail
    # Merging FP32 LoRA deltas into FP16 base weights necessarily rounds the
    # base tensor, and merge->unmerge cannot be bitwise reversible. Gate that
    # production path by bounded error, direction, and exact per-position
    # argmax rather than applying the FP32 serialization threshold to it.
    bounded_merge = all(
        merged_logits_pass(
            metrics,
            dtype=args.dtype,
            strict_max_abs=args.max_logit_diff,
            fp16_max_abs=args.fp16_merge_max_abs,
            fp16_max_mean_abs=args.fp16_merge_max_mean_abs,
            fp16_min_cosine=args.fp16_merge_min_cosine,
        )
        for metrics in (merged_inplace_metrics, unmerge_metrics, merge_metrics)
    )
    ok = (
        trained_vs_base_diff > 0.0
        and reload_metrics["max_abs"] <= args.max_logit_diff
        and bounded_merge
        and greedy_match
    )
    row = {
        "axis": "adapter_roundtrip",
        "status": "pass" if ok else "fail",
        "backend": "native_hf_peft",
        "model": Path(args.model).name,
        "model_path": args.model,
        "model_size_label": infer_model_size_label(args.model, args.model_size_label),
        "device": args.device,
        "dtype": args.dtype,
        "steps": args.steps,
        "learning_rate": args.lr,
        "train_loss": loss,
        "trainable_parameters": trainable,
        "trained_vs_base_logits_max_abs": trained_vs_base_diff,
        "reload_logits": reload_metrics,
        "merge_inplace_logits": merged_inplace_metrics,
        "unmerge_logits": unmerge_metrics,
        "merge_logits": merge_metrics,
        # Stable aliases retained for existing result consumers.
        "reload_logits_max_abs": reload_metrics["max_abs"],
        "merge_inplace_logits_max_abs": merged_inplace_metrics["max_abs"],
        "unmerge_logits_max_abs": unmerge_metrics["max_abs"],
        "merge_logits_max_abs": merge_metrics["max_abs"],
        "max_logit_diff": args.max_logit_diff,
        "fp16_merge_thresholds": {
            "max_abs": args.fp16_merge_max_abs,
            "max_mean_abs": args.fp16_merge_max_mean_abs,
            "min_cosine": args.fp16_merge_min_cosine,
            "top1_match_rate": 1.0,
        },
        "trained_tail": trained_tail,
        "reloaded_tail": reloaded_tail,
        "generated_tail": generated_tail,
        "merged_tail": merged_tail,
        "generated_tokens": len(generated_tail),
        "greedy_token_match": greedy_match,
        **runtime_metadata(parameter_device),
    }
    if args.results:
        out = Path(args.results)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False))
    print(
        f"[native-peft-save-load-merge] train_loss={loss:.4f}, "
        f"trainable={trainable}, reload_diff={reload_metrics['max_abs']:.8f}, "
        f"merge_diff={merge_metrics['max_abs']:.8f}, generated_tail={generated_tail}"
    )
    print("NATIVE PEFT SAVE/LOAD/MERGE PASS" if ok else "NATIVE PEFT SAVE/LOAD/MERGE FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
