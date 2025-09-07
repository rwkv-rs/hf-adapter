#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

# FLA backward currently trips torch.compile/Triton on the V100 test box unless Dynamo is disabled.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-mode", default="fused_recurrent", choices=["chunk", "fused_recurrent"])
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    cfg.attn_mode = args.attn_mode
    cfg.use_cache = False
    cfg.fuse_cross_entropy = False
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        config=cfg,
        torch_dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
        device_map=args.device if args.device.startswith("cuda") else None,
    )
    lora_cfg = LoraConfig(
        task_type="CAUSAL_LM",
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        target_modules=["r_proj", "k_proj", "v_proj", "o_proj", "key", "value"],
    )
    model = get_peft_model(model, lora_cfg)
    model.train()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print("trainable", trainable, "total", total, "pct", round(100 * trainable / total, 4))

    batch = tok("User: Hello!\n\nAssistant: Hello there.", return_tensors="pt")
    if args.device.startswith("cuda"):
        batch = {k: v.cuda() for k, v in batch.items()}
    labels = batch["input_ids"].clone()
    out = model(**batch, labels=labels, use_cache=False)
    print("loss", float(out.loss.detach().cpu()))
    out.loss.backward()
    nonzero = []
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            norm = float(p.grad.float().norm().detach().cpu())
            if norm != 0.0:
                nonzero.append((n, norm))
    print("nonzero_grad_count", len(nonzero))
    for n, norm in nonzero[:5]:
        print("nonzero_grad", n, norm)
    assert nonzero, "No non-zero LoRA gradients observed"


if __name__ == "__main__":
    main()
