#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--prompt", default="User: Hello!\n\nAssistant:")
    args = ap.parse_args()

    if args.device.startswith("npu"):
        from rwkv7_hf.ascend_runtime import enable_ascend

        enable_ascend(args.device, backend="eager", required=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    accelerator = args.device.startswith(("cuda", "npu", "musa", "mps"))
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float16 if accelerator else torch.float32,
        device_map=args.device if args.device.startswith("cuda") else None,
    ).eval()
    if not args.device.startswith("cuda"):
        model.to(args.device)
    enc = tok(args.prompt, return_tensors="pt")
    enc = {k: v.to(args.device) for k, v in enc.items()}

    with torch.inference_mode():
        t0 = time.time()
        out = model(**enc, use_cache=True)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        elif args.device.startswith("npu"):
            torch.npu.synchronize()
        elif args.device.startswith("musa"):
            torch.musa.synchronize()
        print("logits_shape", tuple(out.logits.shape))
        print("top5", out.logits[0, -1].float().topk(5).indices.tolist())
        print("forward_sec", round(time.time() - t0, 4))
        gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        elif args.device.startswith("npu"):
            torch.npu.synchronize()
        elif args.device.startswith("musa"):
            torch.musa.synchronize()
    getter = getattr(model, "rwkv7_last_fast_token_backend", None)
    if callable(getter):
        print("generate_fast_token_backend", getter())
    print("generated_ids_shape", tuple(gen.shape))
    print("decoded_BEGIN")
    print(tok.decode(gen[0].tolist(), skip_special_tokens=True))
    print("decoded_END")


if __name__ == "__main__":
    main()
