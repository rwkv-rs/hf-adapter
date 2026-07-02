#!/usr/bin/env python3
# coding=utf-8
"""Persistence round-trip for native int8 (mm8).

Save an fp16 model with ``config.use_native_mm8=True`` -> reload ->
``from_pretrained`` auto-quantizes eligible linears into MM8Linear, and the
output matches the original fp16 (int8 is a deterministic function of fp16).

Run: python tests/test_native_mm8_persist.py --model <hf_dir>
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import tempfile

os.environ.setdefault("RWKV_V7_ON", "1")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cos-min", type=float, default=0.999)
    args = ap.parse_args()

    repo_rwkv7_hf = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "rwkv7_hf")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ids = tok("The quick brown fox jumps over the lazy dog.",
              return_tensors="pt", add_special_tokens=False).input_ids.cuda()

    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float16, device_map="cuda").eval()
    with torch.no_grad():
        ref = model(ids).logits[0, -1].float().cpu()

    model.config.use_native_mm8 = True
    model.config.native_mm8_min_params = 8_000_000
    tmp = tempfile.mkdtemp(prefix="mm8_persist_")
    try:
        model.save_pretrained(tmp)
        tok.save_pretrained(tmp)
        # remote-code deps (incl native_quant_mm8) must be in the saved dir
        for f in glob.glob(os.path.join(repo_rwkv7_hf, "*.py")):
            shutil.copy(f, tmp)

        del model
        torch.cuda.empty_cache()
        reloaded = AutoModelForCausalLM.from_pretrained(
            tmp, trust_remote_code=True, torch_dtype=torch.float16, device_map="cuda").eval()

        n_mm8 = sum(1 for mod in reloaded.modules() if type(mod).__name__ == "MM8Linear")
        with torch.no_grad():
            out = reloaded(ids).logits[0, -1].float().cpu()
        cos = F.cosine_similarity(ref.unsqueeze(0), out.unsqueeze(0)).item()
        flag_read = bool(getattr(reloaded.config, "use_native_mm8", False))
        print(f"flag read={flag_read} | {n_mm8} MM8Linear on reload | cos vs fp16={cos:.6f}", flush=True)
        ok = flag_read and n_mm8 >= 1 and cos >= args.cos_min
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if not ok:
        print("FAIL", flush=True)
        return 1
    print("PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
