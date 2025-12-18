#!/usr/bin/env python3
# coding=utf-8
"""End-to-end native and TorchAO W8/W4 decode benchmark for the HF adapter.

This complements the isolated RKV/GEMV native-quant microbenchmarks by applying
``quantize_model_mm8`` / ``quantize_model_mm4`` to a loaded HF model and timing
the normal serving decode path.  It records both footprint and decode speed vs
an fp16 baseline so card-validation issues can distinguish memory savings from
actual end-to-end speed wins.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("RWKV_V7_ON", "1")
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
SEED = "The quick brown fox jumps over the lazy dog. " * 256


def infer_model_size_label(hf_dir: str, explicit: str = "") -> str | None:
    if explicit:
        return explicit.lower()
    match = re.search(r"(\d+(?:\.\d+)?b)", Path(hf_dir).name.lower())
    return match.group(1) if match else None


def cuda_sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def device_name(device: str) -> str:
    return torch.cuda.get_device_name(0) if device.startswith("cuda") else device


def peak_mb(device: str) -> float | None:
    if not device.startswith("cuda"):
        return None
    return round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)


def _tensor_payload_bytes(tensor, seen: set[int]) -> int:
    """Count wrapper-subclass payloads instead of their logical dense shape."""

    ident = id(tensor)
    if ident in seen:
        return 0
    seen.add(ident)
    flatten = getattr(tensor, "__tensor_flatten__", None)
    if callable(flatten) and type(tensor) not in {torch.Tensor, torch.nn.Parameter}:
        try:
            names = flatten()[0]
            payload = 0
            for name in names:
                value = getattr(tensor, name)
                if isinstance(value, torch.Tensor):
                    payload += _tensor_payload_bytes(value, seen)
            if payload:
                return payload
        except Exception:
            pass
    return int(tensor.numel()) * int(tensor.element_size())


def module_footprint_mb(model) -> float:
    total = 0
    seen: set[int] = set()
    for tensor in list(model.parameters()) + list(model.buffers()):
        total += _tensor_payload_bytes(tensor, seen)
    return round(total / 1024 / 1024, 1)


def model_metadata(args, model) -> dict[str, Any]:
    cfg = getattr(model, "config", None)
    return {
        "model_name": Path(args.hf_dir).name,
        "model_size_label": infer_model_size_label(args.hf_dir, args.model_size_label),
        "hf_model_dir": args.hf_dir,
        "hidden_size": getattr(cfg, "hidden_size", None),
        "intermediate_size": getattr(cfg, "intermediate_size", None),
        "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
        "head_dim": getattr(cfg, "head_dim", None),
        "num_heads": getattr(cfg, "num_heads", None),
    }


def prepare_model_dir(
    model_path: str, code_source: str
) -> tuple[str, tempfile.TemporaryDirectory[str] | None]:
    """Overlay current repo code without copying checkpoint payloads."""

    if code_source == "model":
        return model_path, None
    source = Path(model_path).resolve()
    repo_code = Path(__file__).resolve().parents[1] / "rwkv7_hf"
    if not source.is_dir():
        raise ValueError("--code-source repo requires a local converted model directory")
    temporary = tempfile.TemporaryDirectory(
        prefix="rwkv7_native_quant_repo_code_", dir=source.parent
    )
    target = Path(temporary.name)
    for item in source.iterdir():
        if item.name == "__pycache__" or item.suffix == ".py":
            continue
        link = target / item.name
        if item.name == "config.json":
            shutil.copy2(item, link)
            continue
        try:
            link.symlink_to(item, target_is_directory=item.is_dir())
        except OSError:
            if item.is_dir():
                shutil.copytree(item, link)
            else:
                os.link(item, link)
    for py_file in repo_code.glob("*.py"):
        shutil.copy2(py_file, target / py_file.name)
    from scripts.sync_hf_adapter_code import sync_one

    sync_one(target)
    return str(target), temporary


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def baseline_path(args) -> Path | None:
    if not args.baseline_dir:
        return None
    key = args.baseline_key or "_".join(
        [
            infer_model_size_label(args.hf_dir, args.model_size_label) or Path(args.hf_dir).name,
            Path(args.hf_dir).name,
            f"dtype-{args.dtype}",
            f"attn-{args.attn_mode}",
            f"fast-{args.fast_token_backend}",
            f"bsz-{args.batch_size}",
            f"prompt-{args.prompt_tokens}",
            f"decode-{args.decode_tokens}",
            f"min-{args.min_params}",
            f"policy-{args.policy}",
        ]
    )
    return Path(args.baseline_dir) / f"{_safe_slug(key)}.pt"


def save_baseline(
    args, row: dict[str, Any], prompt_logits, final_logits, greedy_tokens
) -> None:
    path = baseline_path(args)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "row": row,
            "prompt_logits": prompt_logits.cpu(),
            "final_logits": final_logits.cpu(),
            "greedy_tokens": greedy_tokens,
            "next_token": int(row["next_token"]),
            "prefill_tokps_total": float(row["prefill_tokps_total"]),
            "decode_tokps_total": float(row["decode_tokps_total"]),
            "model_footprint_mb": float(row["model_footprint_mb"]),
        },
        path,
    )
    print(f"saved fp16 baseline -> {path}", flush=True)


def load_baseline(args) -> dict[str, Any] | None:
    path = baseline_path(args)
    if path is None:
        return None
    if not path.exists():
        if args.allow_missing_baseline:
            return None
        raise FileNotFoundError(f"missing fp16 baseline: {path}")
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch without weights_only
        return torch.load(path, map_location="cpu")


def set_attn_mode(model, attn_mode: str) -> None:
    model.config.attn_mode = attn_mode
    for layer in getattr(model.model, "layers", []):
        attn = getattr(layer, "attn", None)
        if hasattr(attn, "mode"):
            attn.mode = attn_mode


def encode(tok, prompt_tokens: int, bsz: int, device: str) -> torch.Tensor:
    ids = tok(SEED, return_tensors="pt", add_special_tokens=False).input_ids[:, :prompt_tokens]
    ids = ids.repeat(bsz, 1)
    return ids.to(device) if device.startswith("cuda") else ids


def last_fast_token_backend(model):
    getter = getattr(model, "rwkv7_last_fast_token_backend", None)
    if callable(getter):
        return getter()
    return getattr(model, "_rwkv7_last_fast_token_backend", None)


def last_native_model_decode_backend(model):
    getter = getattr(model, "rwkv7_native_model_last_decode_backend", None)
    if callable(getter):
        return getter()
    return getattr(model, "_rwkv7_native_model_last_decode_backend", None)


def quantize_model(
    model,
    quantization: str,
    min_params: int,
    policy: str,
    *,
    mm4_group_size: int = 0,
    mm4_group_policy: str = "all",
    group_size: int = 128,
    quantize_head: bool | None = None,
    marlin_skip_last_layers: int | None = None,
) -> tuple[int, dict[str, int]]:
    if quantization == "none":
        return 0, count_modules(model)
    if quantization == "mm8":
        from rwkv7_hf.native_quant_mm8 import quantize_model_mm8
        replaced = quantize_model_mm8(model, min_params=min_params, fused=True, policy=policy)
    elif quantization == "mm4":
        from rwkv7_hf.native_quant_mm4 import quantize_model_mm4
        replaced = quantize_model_mm4(
            model,
            min_params=min_params,
            policy=policy,
            group_size=mm4_group_size,
            group_policy=mm4_group_policy,
        )
    elif quantization in {"torchao_w8", "torchao_w4"}:
        from rwkv7_hf.native_quant_torchao import quantize_model_torchao

        replaced = quantize_model_torchao(
            model,
            quantization,
            min_params=min_params,
            policy=policy,
            group_size=int(group_size),
            quantize_head=quantize_head,
            marlin_skip_last_layers=marlin_skip_last_layers,
        )
    elif quantization == "a8w8":
        from rwkv7_hf.native_quant_a8w8 import quantize_model_a8w8

        replaced = quantize_model_a8w8(
            model,
            min_params=min_params,
            policy=policy,
        )
    else:  # pragma: no cover
        raise ValueError(quantization)
    return int(replaced), count_modules(model)


def count_modules(model) -> dict[str, int]:
    counts = {
        "linear_dense": 0,
        "mm8": 0,
        "mm4": 0,
        "a8w8": 0,
        "marlin_w4": 0,
        "torchao_w8": 0,
        "torchao_w4": 0,
    }
    for mod in model.modules():
        name = type(mod).__name__
        if isinstance(mod, torch.nn.Linear):
            weight = getattr(mod, "weight", None)
            impl = getattr(weight, "tensor_impl", None)
            weight_type = type(weight).__name__
            if (
                hasattr(impl, "packed_weight")
                or weight_type == "Int4TilePackedTo4dTensor"
                or (hasattr(weight, "qdata") and hasattr(weight, "scale_and_zero"))
            ):
                counts["torchao_w4"] += 1
            elif hasattr(impl, "int_data"):
                counts["torchao_w8"] += 1
            else:
                counts["linear_dense"] += 1
        elif name == "MM8Linear":
            counts["mm8"] += 1
        elif name == "MM4Linear":
            counts["mm4"] += 1
        elif name == "A8W8Linear":
            counts["a8w8"] += 1
        elif name == "MarlinW4Linear":
            counts["marlin_w4"] += 1
    return counts


def load_model(args, dtype, *, load_on_cpu: bool = False):
    os.environ["RWKV7_FAST_TOKEN_BACKEND"] = args.fast_token_backend
    if args.fast_cache != "auto":
        os.environ["RWKV7_FAST_CACHE"] = "1" if args.fast_cache == "true" else "0"
    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "device_map": None
        if load_on_cpu
        else (args.device if args.device.startswith("cuda") else None),
    }
    if load_on_cpu:
        load_kwargs["low_cpu_mem_usage"] = True
    model_dir = getattr(args, "effective_hf_dir", args.hf_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, **load_kwargs).eval()
    if args.fuse_norm != "auto":
        desired = args.fuse_norm == "true"
        actual = bool(getattr(model.config, "fuse_norm", False))
        if actual != desired:
            raise ValueError(f"Loaded model config has fuse_norm={actual}; expected {desired}")
    set_attn_mode(model, args.attn_mode)
    return model


def validate_quantize_before_device(args) -> None:
    """Keep CPU-first packing explicit and limited to supported fresh-process lanes."""

    if not args.quantize_before_device:
        return
    if not args.device.startswith("cuda"):
        raise ValueError("--quantize-before-device requires a CUDA target device")
    memory_quantizations = {"mm8", "mm4"}
    speed_quantizations = {"a8w8"}
    if args.single_quantization not in memory_quantizations | speed_quantizations:
        raise ValueError(
            "--quantize-before-device requires --single-quantization "
            "mm8, mm4, or a8w8"
        )
    if args.single_quantization in memory_quantizations and args.policy != "memory":
        raise ValueError("CPU-first MM8/MM4 requires --policy memory")
    if args.single_quantization in speed_quantizations and args.policy != "speed":
        raise ValueError("CPU-first A8W8 requires --policy speed")
    if args.paired_baseline:
        raise ValueError(
            "--quantize-before-device cannot measure an in-process fp16 baseline"
        )
    if not args.allow_missing_baseline:
        raise ValueError(
            "--quantize-before-device requires --allow-missing-baseline"
        )


def benchmark_decode(args, tok, model, ids):
    fast_fn = getattr(model, "rwkv7_forward_token", None)
    if fast_fn is None and ids.shape[0] == 1:
        fast_fn = getattr(model, "rwkv7_forward_one", None)
    if fast_fn is None:
        def step_fn(token_ids, *, past_key_values):
            return model(token_ids, past_key_values=past_key_values, use_cache=True, logits_to_keep=1)

        step_backend = "module_call"
    else:
        def step_fn(token_ids, *, past_key_values):
            return fast_fn(token_ids, past_key_values=past_key_values)

        step_backend = last_fast_token_backend(model) or os.environ.get("RWKV7_FAST_TOKEN_BACKEND", "auto")

    samples = []
    # Exclude one-time import/compile/cache construction from paired steady
    # prefill timing, just as decode excludes its warmup steps.
    with torch.inference_mode():
        model(ids, use_cache=True, logits_to_keep=1)
    cuda_sync(args.device)
    for _repeat in range(args.timing_repeats):
        with torch.inference_mode():
            cuda_sync(args.device)
            prefill_start = time.time()
            out = model(ids, use_cache=True, logits_to_keep=1)
            cuda_sync(args.device)
            prefill_sec = time.time() - prefill_start
            state = out.past_key_values
            nxt = out.logits[:, -1:].argmax(dim=-1)
            prompt_logits = out.logits[:, -1].float().detach().cpu()
            greedy_tokens = []
            for _ in range(args.warmup):
                out = step_fn(nxt, past_key_values=state)
                state = out.past_key_values
                nxt = out.logits[:, -1:].argmax(dim=-1)
                greedy_tokens.append(nxt.detach().cpu())
            cuda_sync(args.device)
            t0 = time.time()
            for _ in range(args.decode_tokens):
                out = step_fn(nxt, past_key_values=state)
                state = out.past_key_values
                nxt = out.logits[:, -1:].argmax(dim=-1)
            cuda_sync(args.device)
            dt = time.time() - t0
            final_logits = out.logits[:, -1].float().detach().cpu()
        samples.append(
            {
                "prefill_sec": prefill_sec,
                "prefill_tokps_total": round(
                    (ids.shape[0] * ids.shape[1]) / prefill_sec,
                    1,
                ),
                "decode_sec": dt,
                "decode_tokps_total": round((ids.shape[0] * args.decode_tokens) / dt, 1),
                "decode_tokps_per_seq": round(args.decode_tokens / dt, 1),
                "decode_ms_per_step": round(1000 * dt / args.decode_tokens, 3),
                "prompt_logits": prompt_logits,
                "final_logits": final_logits,
                "greedy_tokens": torch.cat(greedy_tokens, dim=1).tolist(),
                "next_token": int(nxt[0, -1].detach().cpu()),
                "fast_token_backend_effective": step_backend,
                "native_model_decode_backend_effective": last_native_model_decode_backend(model),
                "cache_type": type(state).__name__ if state is not None else None,
            }
        )
    selected = sorted(samples, key=lambda item: float(item["decode_sec"]))[len(samples) // 2]
    prefill_selected = sorted(
        samples,
        key=lambda item: float(item["prefill_sec"]),
    )[len(samples) // 2]
    selected["prefill_sec"] = prefill_selected["prefill_sec"]
    selected["prefill_tokps_total"] = prefill_selected["prefill_tokps_total"]
    selected["timing_repeats"] = len(samples)
    selected["prefill_tokps_samples"] = [
        float(item["prefill_tokps_total"]) for item in samples
    ]
    selected["decode_tokps_samples"] = [float(item["decode_tokps_total"]) for item in samples]
    greedy_hashes = [
        hashlib.sha256(
            json.dumps(item["greedy_tokens"], separators=(",", ":")).encode("ascii")
        ).hexdigest()
        for item in samples
    ]
    selected["greedy_tokens_sha256"] = greedy_hashes[samples.index(selected)]
    selected["greedy_repeat_sha256"] = greedy_hashes
    selected["greedy_repeat_deterministic"] = len(set(greedy_hashes)) == 1
    return selected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--code-source", choices=("repo", "model"), default="repo")
    ap.add_argument("--model-size-label", default="")
    ap.add_argument("--dtype", default="fp16", choices=sorted(DTYPES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn-mode", default="fused_recurrent", choices=["chunk", "fused_recurrent"])
    ap.add_argument("--fuse-norm", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--fast-cache", choices=["auto", "true", "false"], default="true")
    ap.add_argument("--fast-token-backend", choices=["auto", "fla", "native_jit", "native_graph"], default="native_graph")
    quantization_choices = ["none", "mm8", "mm4", "a8w8", "torchao_w8", "torchao_w4"]
    ap.add_argument("--quantizations", nargs="+", choices=quantization_choices, default=["none", "mm8", "mm4"])
    ap.add_argument(
        "--single-quantization",
        choices=quantization_choices,
        default=None,
        help="Run exactly one quantization in this process. Useful for fresh-process 7B+ rows.",
    )
    ap.add_argument("--min-params", type=int, default=8_000_000)
    ap.add_argument(
        "--mm4-group-size",
        type=int,
        choices=(0, 128, 256),
        default=0,
        help="native MM4 scale group: 0=row-wise, 128/256=groupwise V100 routes",
    )
    ap.add_argument(
        "--mm4-group-policy",
        choices=(
            "all",
            "lm_head",
            "ffn_key",
            "ffn_value",
            "lm_head_and_key",
            "lm_head_and_value",
        ),
        default="all",
        help="modules that receive groupwise scales; other MM4 modules stay row-wise",
    )
    ap.add_argument(
        "--group-size",
        type=int,
        choices=(32, 64, 128),
        default=128,
        help="TorchAO/Marlin W4 group size; ignored by non-W4 backends",
    )
    ap.add_argument(
        "--quantize-head",
        choices=("auto", "true", "false"),
        default="auto",
        help="auto uses the exact-card model profile; true/false overrides it",
    )
    ap.add_argument(
        "--marlin-skip-last-layers",
        type=int,
        default=-1,
        help="-1 uses the exact-card profile; otherwise leave this many final FFN layers dense",
    )
    ap.add_argument(
        "--policy",
        default="memory",
        choices=["memory", "speed"],
        help="native MM module-selection policy: memory=all size-gated linears, speed=lm_head only",
    )
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--prompt-tokens", type=int, default=32)
    ap.add_argument("--decode-tokens", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--timing-repeats", type=int, default=1, help="Independent decode measurements; report the median run")
    ap.add_argument("--baseline-dir", default="", help="Directory for fp16 baseline logits/tokps used by fresh quant-only runs")
    ap.add_argument("--baseline-key", default="", help="Optional explicit baseline-cache key shared by fp16/mm8/mm4 subprocesses")
    ap.add_argument(
        "--paired-baseline",
        action="store_true",
        help="Measure a dense baseline in the same fresh process immediately before quantizing; removes cross-process clock noise",
    )
    ap.add_argument("--allow-missing-baseline", action="store_true", help="Emit quant-only rows with null ratios when fp16 OOM/no baseline")
    ap.add_argument(
        "--quantize-before-device",
        action="store_true",
        help=(
            "Load dense weights on CPU, pack either native MM8/MM4 memory lanes "
            "or the A8W8 speed lane, then move only the packed model to "
            "CUDA. Requires one supported quantization and --allow-missing-baseline."
        ),
    )
    ap.add_argument("--results", default=str(Path(__file__).parent / "results.jsonl"))
    args = ap.parse_args()
    if args.timing_repeats < 1:
        ap.error("--timing-repeats must be >= 1")
    try:
        validate_quantize_before_device(args)
    except ValueError as exc:
        ap.error(str(exc))
    if args.single_quantization is not None:
        args.quantizations = [args.single_quantization]
    else:
        args.quantizations = list(dict.fromkeys(["none", *args.quantizations]))

    dtype = DTYPES[args.dtype]
    args.effective_hf_dir, code_overlay = prepare_model_dir(
        args.hf_dir, args.code_source
    )
    tok = AutoTokenizer.from_pretrained(args.effective_hf_dir, trust_remote_code=True)
    ids = encode(tok, args.prompt_tokens, args.batch_size, args.device)
    rows = []
    baseline_prompt = None
    baseline_final = None
    baseline_next = None
    baseline_greedy = None
    baseline_prefill_tokps = None
    baseline_tokps = None
    baseline_footprint = None
    out_path = Path(args.results) if args.results else None
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    for quantization in args.quantizations:
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        model = load_model(args, dtype, load_on_cpu=args.quantize_before_device)
        if args.quantize_before_device:
            baseline_footprint = module_footprint_mb(model)
        if quantization != "none" and args.paired_baseline:
            dense_footprint = module_footprint_mb(model)
            dense_res = benchmark_decode(args, tok, model, ids)
            baseline_prompt = dense_res["prompt_logits"]
            baseline_final = dense_res["final_logits"]
            baseline_next = dense_res["next_token"]
            baseline_greedy = dense_res["greedy_tokens"]
            baseline_prefill_tokps = dense_res["prefill_tokps_total"]
            baseline_tokps = dense_res["decode_tokps_total"]
            baseline_footprint = dense_footprint
        replaced, module_counts = quantize_model(
            model,
            quantization,
            args.min_params,
            args.policy,
            mm4_group_size=args.mm4_group_size,
            mm4_group_policy=args.mm4_group_policy,
            group_size=args.group_size,
            quantize_head=(
                None if args.quantize_head == "auto" else args.quantize_head == "true"
            ),
            marlin_skip_last_layers=(
                None
                if args.marlin_skip_last_layers < 0
                else args.marlin_skip_last_layers
            ),
        )
        if args.quantize_before_device:
            gc.collect()
            model.to(args.device)
        footprint = module_footprint_mb(model)
        # Measure steady-state inference memory, not temporary fp32 tensors
        # created while quantizing a dense checkpoint at process startup.
        # Production deployments normally load an already packed checkpoint.
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        res = benchmark_decode(args, tok, model, ids)
        prompt_logits_for_baseline = res["prompt_logits"]
        final_logits_for_baseline = res["final_logits"]
        greedy_tokens_for_baseline = res["greedy_tokens"]
        if quantization == "none":
            baseline_prompt = res.pop("prompt_logits")
            baseline_final = res.pop("final_logits")
            baseline_next = res["next_token"]
            baseline_greedy = res.pop("greedy_tokens")
            baseline_prefill_tokps = res["prefill_tokps_total"]
            baseline_tokps = res["decode_tokps_total"]
            baseline_footprint = footprint
            prompt_cos = final_cos = 1.0
            same_next = True
            same_greedy = True
            first_greedy_mismatch = None
            prefill_speed_ratio = 1.0
            speed_ratio = 1.0
            footprint_ratio = 1.0
        else:
            if baseline_prompt is None or baseline_final is None:
                cached_baseline = load_baseline(args)
                if cached_baseline is not None:
                    baseline_prompt = cached_baseline["prompt_logits"]
                    baseline_final = cached_baseline["final_logits"]
                    baseline_next = int(cached_baseline["next_token"])
                    baseline_greedy = cached_baseline.get("greedy_tokens")
                    baseline_prefill_tokps = float(
                        cached_baseline.get(
                            "prefill_tokps_total",
                            cached_baseline.get("row", {}).get("prefill_tokps_total", 0.0),
                        )
                    ) or None
                    baseline_tokps = float(cached_baseline["decode_tokps_total"])
                    baseline_footprint = float(cached_baseline["model_footprint_mb"])
            prompt_logits = res.pop("prompt_logits")
            final_logits = res.pop("final_logits")
            greedy_tokens = res.pop("greedy_tokens")
            if baseline_prompt is None or baseline_final is None:
                if not args.allow_missing_baseline:
                    raise RuntimeError("quantized run has no in-process or cached fp16 baseline")
                prompt_cos = final_cos = None
                same_next = None
                same_greedy = None
                first_greedy_mismatch = None
                prefill_speed_ratio = None
                speed_ratio = None
                footprint_ratio = (
                    float(footprint) / float(baseline_footprint)
                    if baseline_footprint is not None
                    else None
                )
            else:
                prompt_cos = F.cosine_similarity(baseline_prompt.flatten().unsqueeze(0), prompt_logits.flatten().unsqueeze(0)).item()
                final_cos = F.cosine_similarity(baseline_final.flatten().unsqueeze(0), final_logits.flatten().unsqueeze(0)).item()
                same_next = int(res["next_token"]) == int(baseline_next)
                same_greedy = (
                    greedy_tokens == baseline_greedy
                    if baseline_greedy is not None
                    else None
                )
                first_greedy_mismatch = None
                if same_greedy is False:
                    flat_baseline = [token for row in baseline_greedy for token in row]
                    flat_quant = [token for row in greedy_tokens for token in row]
                    first_greedy_mismatch = next(
                        index
                        for index, (baseline_token, quant_token) in enumerate(
                            zip(flat_baseline, flat_quant)
                        )
                        if baseline_token != quant_token
                    )
                prefill_speed_ratio = (
                    float(res["prefill_tokps_total"]) / float(baseline_prefill_tokps)
                    if baseline_prefill_tokps is not None
                    else None
                )
                speed_ratio = float(res["decode_tokps_total"]) / float(baseline_tokps)
                footprint_ratio = float(footprint) / float(baseline_footprint)
        sm70_active = bool(
            quantization == "mm4"
            and args.device.startswith("cuda")
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability(args.device) == (7, 0)
        )
        row = {
            "axis": "native_quant_e2e_decode",
            "backend": "hf_adapter",
            "code_source": args.code_source,
            "effective_hf_dir": args.effective_hf_dir,
            "status": "pass",
            "quantization": quantization,
            "native_mm4_group_size": args.mm4_group_size if quantization == "mm4" else None,
            "native_mm4_group_policy": (
                args.mm4_group_policy if quantization == "mm4" else None
            ),
            "sm70_w4_tuning_policy": (
                "environment_override"
                if sm70_active
                and any(
                    name in os.environ
                    for name in (
                        "RWKV7_SM70_W4_BN",
                        "RWKV7_SM70_W4_TN",
                        "RWKV7_SM70_W4_GROUP_BN",
                        "RWKV7_SM70_W4_GROUP_TN",
                    )
                )
                else "exact_v100_auto_20260716"
                if sm70_active
                else None
            ),
            "sm70_w4_fused_epilogue": (
                os.environ.get("RWKV7_SM70_W4_FUSED_EPILOGUE", "0").strip().lower()
                not in {"", "0", "false", "no", "off"}
                if sm70_active
                else None
            ),
            "sm70_w4_block_n": (
                int(os.environ["RWKV7_SM70_W4_BN"])
                if sm70_active and "RWKV7_SM70_W4_BN" in os.environ
                else None
            ),
            "sm70_w4_thread_n": (
                int(os.environ["RWKV7_SM70_W4_TN"])
                if sm70_active and "RWKV7_SM70_W4_TN" in os.environ
                else None
            ),
            "sm70_w4_group_block_n": (
                int(os.environ["RWKV7_SM70_W4_GROUP_BN"])
                if sm70_active
                and "RWKV7_SM70_W4_GROUP_BN" in os.environ
                else None
            ),
            "sm70_w4_group_thread_n": (
                int(os.environ["RWKV7_SM70_W4_GROUP_TN"])
                if sm70_active
                and "RWKV7_SM70_W4_GROUP_TN" in os.environ
                else None
            ),
            "dtype": args.dtype,
            "device": device_name(args.device),
            **model_metadata(args, model),
            "attn_mode": args.attn_mode,
            "fuse_norm": getattr(model.config, "fuse_norm", None),
            "fast_cache": os.environ.get("RWKV7_FAST_CACHE", "1") not in {"0", "false", "False", "no", "off"},
            "batch_size": args.batch_size,
            "prompt_tokens": int(ids.shape[1]),
            "decode_tokens": args.decode_tokens,
            "min_params": args.min_params,
            "quant_group_size": int(args.group_size),
            "quantized_head": bool(
                getattr(model, "_rwkv7_native_mm_quantized_head", False)
            ),
            "marlin_skip_last_layers": int(
                getattr(model, "_rwkv7_native_mm_marlin_skip_last_layers", 0)
            ),
            "native_mm_policy": args.policy,
            "paired_baseline": bool(args.paired_baseline and quantization != "none"),
            "quantize_before_device": bool(args.quantize_before_device),
            "replaced_modules": replaced,
            "module_counts": module_counts,
            "native_mm_kernel": getattr(model, "_rwkv7_native_mm_exact_5090_kernel", None),
            "fused_relu2_ffn_modules": int(
                getattr(model, "_rwkv7_native_mm_fused_relu2_ffn_modules", 0)
            ),
            "model_footprint_mb": footprint,
            "baseline_prefill_tokps_total": round(float(baseline_prefill_tokps), 1) if baseline_prefill_tokps is not None else None,
            "baseline_decode_tokps_total": round(float(baseline_tokps), 1) if baseline_tokps is not None else None,
            "baseline_model_footprint_mb": round(float(baseline_footprint), 1) if baseline_footprint is not None else None,
            "footprint_ratio_vs_fp16": round(footprint_ratio, 4) if footprint_ratio is not None else None,
            "prefill_speed_ratio_vs_fp16": round(prefill_speed_ratio, 4) if prefill_speed_ratio is not None else None,
            "decode_speed_ratio_vs_fp16": round(speed_ratio, 4) if speed_ratio is not None else None,
            "prompt_logits_cos_vs_fp16": round(float(prompt_cos), 8) if prompt_cos is not None else None,
            "final_logits_cos_vs_fp16": round(float(final_cos), 8) if final_cos is not None else None,
            "same_next_token_as_fp16": bool(same_next) if same_next is not None else None,
            "same_greedy_tokens_as_fp16": (
                bool(same_greedy) if same_greedy is not None else None
            ),
            "first_greedy_mismatch_flat_index": first_greedy_mismatch,
            "peak_vram_mb": peak_mb(args.device),
            **{k: v for k, v in res.items() if k not in {"prompt_logits", "final_logits"}},
        }
        if quantization == "none":
            save_baseline(
                args,
                row,
                prompt_logits_for_baseline,
                final_logits_for_baseline,
                greedy_tokens_for_baseline,
            )
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)
        del model
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        if out_path is not None:
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"appended 1 row -> {out_path}", flush=True)

    if code_overlay is not None:
        code_overlay.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
