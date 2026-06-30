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

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
        device_map=args.device if args.device.startswith("cuda") else None,
    ).eval()
    enc = tok(args.prompt, return_tensors="pt")
    if args.device.startswith("cuda"):
        enc = {k: v.cuda() for k, v in enc.items()}

    with torch.inference_mode():
        t0 = time.time()
        out = model(**enc, use_cache=True)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        print("logits_shape", tuple(out.logits.shape))
        print("top5", out.logits[0, -1].float().topk(5).indices.tolist())
        print("forward_sec", round(time.time() - t0, 4))
        gen = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
    print("generated_ids_shape", tuple(gen.shape))
    print("decoded_BEGIN")
    print(tok.decode(gen[0].tolist(), skip_special_tokens=True))
    print("decoded_END")


if __name__ == "__main__":
    main()
