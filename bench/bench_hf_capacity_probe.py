#!/usr/bin/env python3
"""Fail-closed single-GPU fit probe for local Hugging Face checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time
from typing import Any


DTYPES = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}
SEED = "The quick brown fox jumps over the lazy dog. " * 256


def checkpoint_payload_bytes(model_dir: Path) -> int | None:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        total = index.get("metadata", {}).get("total_size")
        if total is not None:
            return int(total)
    shards = list(model_dir.glob("*.safetensors"))
    if shards:
        return sum(path.stat().st_size for path in shards)
    return None


def classify_result(expected: str, observed: str) -> tuple[str, str | None]:
    if expected == observed:
        return "pass", None
    return "fail", f"expected {expected}, observed {observed}"


def prepare_model_dir(
    model_dir: Path,
    *,
    model_kind: str,
    code_source: str,
) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if model_kind != "rwkv" or code_source == "model":
        return model_dir, None
    try:
        from bench.bench_cross_model_speed import prepare_rwkv_model_dir
    except ModuleNotFoundError:
        from bench_cross_model_speed import prepare_rwkv_model_dir

    effective, temporary = prepare_rwkv_model_dir(str(model_dir), "repo")
    return Path(effective), temporary


def append_row(path: str, row: dict[str, Any]) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-kind", choices=("rwkv", "qwen35"), required=True)
    parser.add_argument("--model-size-label", required=True)
    parser.add_argument("--code-source", choices=("model", "repo"), default="repo")
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument(
        "--expect",
        choices=("fit", "capacity-limit"),
        required=True,
        help="Make the observed fit boundary fail closed.",
    )
    parser.add_argument("--results", default="")
    args = parser.parse_args()

    if args.batch_size < 1 or args.prompt_tokens < 1 or args.max_new_tokens < 1:
        parser.error("batch, prompt, and generation lengths must be positive")
    model_dir = Path(args.model).resolve()
    if not model_dir.is_dir():
        parser.error("--model must be a local Hugging Face model directory")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, DTYPES[args.dtype])
    device_name = args.device
    total_vram_mb = None
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            parser.error("CUDA is required for a CUDA capacity probe")
        device_name = torch.cuda.get_device_name(0)
        total_vram_mb = round(torch.cuda.get_device_properties(0).total_memory / 2**20, 1)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    effective_dir, temporary = prepare_model_dir(
        model_dir,
        model_kind=args.model_kind,
        code_source=args.code_source,
    )
    model = None
    observed = "error"
    error_type = None
    logits_finite = False
    cache_type = None
    generated_tokens = 0
    load_s = None
    peak_vram_mb = None
    started = time.perf_counter()
    try:
        tokenizer = AutoTokenizer.from_pretrained(effective_dir, trust_remote_code=True)
        load_started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            effective_dir,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=args.device if args.device.startswith("cuda") else None,
            low_cpu_mem_usage=True,
        ).eval()
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        load_s = round(time.perf_counter() - load_started, 3)
        encoded = tokenizer(SEED, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded.input_ids[:, : args.prompt_tokens].to(next(model.parameters()).device)
        input_ids = input_ids.repeat(args.batch_size, 1)
        with torch.inference_mode():
            try:
                output = model(input_ids, use_cache=True, logits_to_keep=1)
            except TypeError:
                output = model(input_ids, use_cache=True)
            logits_finite = bool(torch.isfinite(output.logits).all().item())
            cache_type = type(output.past_key_values).__name__
            generated = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        generated_tokens = int(generated.shape[1] - input_ids.shape[1])
        observed = "fit" if logits_finite and generated_tokens == args.max_new_tokens else "error"
    except torch.cuda.OutOfMemoryError:
        observed = "capacity-limit"
        error_type = "cuda_out_of_memory"
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            observed = "capacity-limit"
            error_type = "cuda_out_of_memory"
        else:
            error_type = type(exc).__name__
    except Exception as exc:  # Keep unexpected loader failures distinct from capacity.
        error_type = type(exc).__name__
    finally:
        if args.device.startswith("cuda"):
            peak_vram_mb = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        if temporary is not None:
            temporary.cleanup()

    status, error = classify_result(args.expect, observed)
    row = {
        "axis": "hf_single_gpu_capacity",
        "status": status,
        "expected": args.expect,
        "observed": observed,
        "error": error,
        "error_type": error_type,
        "device": device_name,
        "total_vram_mb": total_vram_mb,
        "model_kind": args.model_kind,
        "model_size_label": args.model_size_label,
        "model_name": model_dir.name,
        "checkpoint_payload_bytes": checkpoint_payload_bytes(model_dir),
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "prompt_tokens": args.prompt_tokens,
        "max_new_tokens": args.max_new_tokens,
        "generated_tokens": generated_tokens,
        "logits_finite": logits_finite,
        "cache_type": cache_type,
        "load_s": load_s,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "peak_vram_mb": peak_vram_mb,
    }
    print(json.dumps(row, indent=2, ensure_ascii=False), flush=True)
    append_row(args.results, row)
    return 0 if status == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
