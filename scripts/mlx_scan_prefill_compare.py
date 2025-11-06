#!/usr/bin/env python3
# coding=utf-8
"""Compare MLX token-major prefill with opt-in WKV scan prefill on a real HF model."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


def _append_jsonl(path: str | None, row: dict[str, Any]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", help="HF RWKV-7 model directory")
    p.add_argument("--prompt", default="User: Compare RWKV scan prefill. Assistant:")
    p.add_argument("--prompt-target-chars", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--dtype", default="fp16")
    p.add_argument("--quantization", default="mm4")
    p.add_argument("--quant-min-params", type=int, default=4_000_000)
    p.add_argument("--quant-rkv-min-params", type=int, default=0)
    p.add_argument("--quant-backend", default="auto")
    p.add_argument("--wkv-backend", default="metal")
    p.add_argument("--results", default="")
    return p.parse_args()


def _make_prompt(seed: str, target_chars: int) -> str:
    if target_chars <= 0:
        raise ValueError("--prompt-target-chars must be positive")
    repeats = (target_chars + len(seed) - 1) // len(seed)
    return (seed * repeats)[:target_chars]


def main() -> int:
    args = _parse_args()

    import mlx.core as mx
    from transformers import AutoTokenizer

    from rwkv7_hf.mlx_model import load_mlx_rwkv7_model

    prompt = _make_prompt(args.prompt, args.prompt_target_chars)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not input_ids:
        raise ValueError("tokenizer produced no prompt tokens")
    batch_ids = [input_ids]

    common = dict(
        dtype=args.dtype,
        quantization=args.quantization,
        quant_min_params=args.quant_min_params,
        quant_rkv_min_params=args.quant_rkv_min_params,
        quant_backend=args.quant_backend,
        wkv_backend=args.wkv_backend,
    )

    old_flag = os.environ.get("RWKV7_MLX_WKV_SCAN_PREFILL")
    os.environ["RWKV7_MLX_WKV_SCAN_PREFILL"] = "0"
    token_model = load_mlx_rwkv7_model(args.model, **common)
    t0 = time.perf_counter()
    token_logits, token_state = token_model.prefill(batch_ids)
    token_prefill_s = time.perf_counter() - t0
    token_gen, token_final_state = token_model.decode_greedy(
        token_logits, token_state, max_new_tokens=args.max_new_tokens
    )

    os.environ["RWKV7_MLX_WKV_SCAN_PREFILL"] = "1"
    scan_model = load_mlx_rwkv7_model(args.model, **common)
    t0 = time.perf_counter()
    scan_logits, scan_state = scan_model.prefill(batch_ids)
    scan_prefill_s = time.perf_counter() - t0
    scan_gen, scan_final_state = scan_model.decode_greedy(
        scan_logits, scan_state, max_new_tokens=args.max_new_tokens
    )

    mx.eval(token_logits, scan_logits, token_gen, scan_gen)
    logit_diff = float(mx.max(mx.abs(token_logits.astype(mx.float32) - scan_logits.astype(mx.float32))))
    state_diffs = []
    for a, b in zip(token_state.recurrent_state, scan_state.recurrent_state, strict=True):
        mx.eval(a, b)
        state_diffs.append(float(mx.max(mx.abs(a - b))))
    gen_equal = bool(mx.all(token_gen == scan_gen).item())

    if old_flag is None:
        os.environ.pop("RWKV7_MLX_WKV_SCAN_PREFILL", None)
    else:
        os.environ["RWKV7_MLX_WKV_SCAN_PREFILL"] = old_flag

    row = {
        "axis": "mlx_scan_prefill_compare",
        "status": "pass" if gen_equal and logit_diff < 0.25 else "fail",
        "model": Path(args.model).name,
        "model_path": str(args.model),
        "prompt_target_chars": int(args.prompt_target_chars),
        "prompt_eval_tokens": int(len(input_ids)),
        "max_new_tokens": int(args.max_new_tokens),
        "dtype": args.dtype,
        "quantization": args.quantization,
        "quant_backend": args.quant_backend,
        "wkv_backend": args.wkv_backend,
        "token_prefill_s": round(float(token_prefill_s), 6),
        "scan_prefill_s": round(float(scan_prefill_s), 6),
        "speedup_scan_vs_token_prefill": round(float(token_prefill_s / scan_prefill_s), 6) if scan_prefill_s > 0 else None,
        "token_prefill_tok_s": round(float(len(input_ids) / token_prefill_s), 6) if token_prefill_s > 0 else None,
        "scan_prefill_tok_s": round(float(len(input_ids) / scan_prefill_s), 6) if scan_prefill_s > 0 else None,
        "max_abs_logits": round(float(logit_diff), 8),
        "max_abs_state": round(float(max(state_diffs or [0.0])), 8),
        "generated_equal": gen_equal,
        "token_generated_preview": [int(x) for x in token_gen.reshape(-1)[:16].tolist()],
        "scan_generated_preview": [int(x) for x in scan_gen.reshape(-1)[:16].tolist()],
        "token_wkv_counts": token_model.telemetry().get("wkv_backend_counts"),
        "scan_wkv_counts": scan_model.telemetry().get("wkv_backend_counts"),
        "scan_wkv_scan_counts": scan_model.telemetry().get("wkv_scan_prefill_counts"),
    }
    print(json.dumps(row, ensure_ascii=False))
    _append_jsonl(args.results, row)
    return 0 if row["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
