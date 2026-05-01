#!/usr/bin/env python3
# coding=utf-8
"""End-to-end A/B for Ada grouped W/A/G/V low-rank decode.

The default axis changes only ``RWKV7_NATIVE_GRAPH_ADA_WAGV_LORA`` between
captures. ``--axis ada_wagv_bmm`` instead holds that grouped route on and
changes only the B8 tensor-core BMM switch. The
``ada_wagv_bmm_from_default`` axis compares the ungrouped fallback directly
with grouped BMM; use it when a card's current policy does not already enable
grouped W/A/G/V at B8. ``sm120_wagv_bmm_g`` holds the proven B8 BMM baseline
on and toggles the exact-SM120 padded W/A/G/V + fused-epilogue + six-slot
norm/mix route. Sparse FFN and the Ada exact-row linear probe stay disabled,
while output/recurrent-output, raw recurrent, and decode norm/mix routes remain
enabled. It records correctness, cache telemetry, latency, throughput, and
peak memory.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("RWKV_V7_ON", "1")
os.environ.setdefault("RWKV7_FAST_TOKEN_BACKEND", "native_graph")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rwkv7_hf.native_model import NativeRWKV7ForCausalLM

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
SEED = "The quick brown fox jumps over the lazy dog. " * 128
_FALSE_VALUES = {"0", "false", "False", "no", "off"}


def cuda_sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def device_name(device: str) -> str:
    return torch.cuda.get_device_name(0) if device.startswith("cuda") else device


def peak_mb(device: str) -> float | None:
    if not device.startswith("cuda"):
        return None
    return round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)


def model_metadata(args: argparse.Namespace, model: Any) -> dict[str, Any]:
    cfg = getattr(model, "config", None)
    return {
        "model_name": Path(args.hf_dir).name,
        "hidden_size": getattr(cfg, "hidden_size", None),
        "intermediate_size": getattr(cfg, "intermediate_size", None),
        "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
        "head_dim": getattr(cfg, "head_dim", None),
        "num_heads": getattr(cfg, "num_heads", None),
    }


def wagv_extension_status(model: Any, device: str) -> dict[str, Any]:
    """Report whether the selected model package can build the grouped extension."""

    package = model.__class__.__module__.rsplit(".", 1)[0]
    try:
        module = importlib.import_module(package + ".ada_lora")
        available = bool(module.ada_wagv_lora_available(device, build=True))
        error = module.ada_wagv_lora_build_error(device)
    except Exception as exc:
        available = False
        error = repr(exc)
    return {
        "wagv_extension_available": available,
        "wagv_extension_error": error,
    }


def wagv_bmm_route_status(
    model: Any,
    batch_size: int,
    *,
    route_prefix: str = "ada_wagv_bmm",
) -> dict[str, Any]:
    """Read the grouped-BMM route actually captured by one graph runner."""

    empty = {
        "requested": None,
        "selected": None,
        "effective": None,
        "selected_layers": [],
        "effective_layers": [],
        "effective_layer_count": 0,
        "full_model_effective": None,
    }
    getter = getattr(model, "rwkv7_native_graph_runner_copy_stats", None)
    if not callable(getter):
        return empty
    try:
        runners = getter().get("runners", [])
    except Exception:
        return empty
    match = next(
        (
            row
            for row in reversed(runners)
            if int(row.get("batch_size", -1)) == int(batch_size)
        ),
        None,
    )
    if not isinstance(match, dict):
        return empty
    return {
        "requested": match.get(f"{route_prefix}_requested"),
        "selected": match.get(f"{route_prefix}_selected"),
        "effective": match.get(f"{route_prefix}_effective"),
        "selected_layers": list(match.get(f"{route_prefix}_selected_layers", [])),
        "effective_layers": list(match.get(f"{route_prefix}_effective_layers", [])),
        "effective_layer_count": int(
            match.get(f"{route_prefix}_effective_layer_count", 0)
        ),
        "full_model_effective": match.get(f"{route_prefix}_full_model_effective"),
    }


def wagv_bmm_route_pass(
    status: dict[str, Any],
    *,
    requested: bool,
    num_layers: int,
) -> bool:
    """Fail closed when a requested BMM capture silently selected a fallback."""

    if status.get("requested") is not bool(requested):
        return False
    if not requested:
        return not bool(status.get("selected")) and not bool(status.get("effective"))
    expected_layers = list(range(int(num_layers)))
    return bool(
        status.get("selected")
        and status.get("effective")
        and status.get("full_model_effective")
        and status.get("selected_layers") == expected_layers
        and status.get("effective_layers") == expected_layers
        and int(status.get("effective_layer_count", 0)) == int(num_layers)
    )


def greedy_match_summary(
    reference: list[int], candidate: list[int]
) -> tuple[bool, int, int]:
    """Return fail-closed greedy alignment with an unambiguous boolean result."""

    reference_total = len(reference)
    match_count = sum(
        int(left == right) for left, right in zip(reference, candidate, strict=False)
    )
    all_match = bool(
        reference_total > 0
        and len(candidate) == reference_total
        and match_count == reference_total
    )
    return all_match, match_count, reference_total


def logits_pair_metrics(
    reference: torch.Tensor, candidate: torch.Tensor, *, batch_size: int
) -> dict[str, Any]:
    """Return finite, cosine, and max-error evidence without JSON NaN/Inf."""

    finite = bool(
        tuple(reference.shape) == tuple(candidate.shape)
        and torch.isfinite(reference).all().item()
        and torch.isfinite(candidate).all().item()
    )
    if not finite:
        return {"finite": False, "min_cosine": None, "max_abs_diff": None}
    reference_f = reference.float().reshape(int(batch_size), -1)
    candidate_f = candidate.float().reshape(int(batch_size), -1)
    return {
        "finite": True,
        "min_cosine": float(
            torch.nn.functional.cosine_similarity(reference_f, candidate_f, dim=-1)
            .min()
            .cpu()
        ),
        "max_abs_diff": float((reference_f - candidate_f).abs().max().cpu()),
    }


def set_attn_mode(model, attn_mode: str) -> None:
    model.config.attn_mode = attn_mode
    for layer in getattr(model.model, "layers", []):
        attn = getattr(layer, "attn", None)
        if hasattr(attn, "mode"):
            attn.mode = attn_mode


def load_model(args: argparse.Namespace, dtype: torch.dtype):
    if args.fast_cache != "auto":
        os.environ["RWKV7_FAST_CACHE"] = "1" if args.fast_cache == "true" else "0"
    os.environ["RWKV7_FAST_TOKEN_BACKEND"] = "native_graph"
    os.environ["RWKV7_NATIVE_GRAPH_CACHE_SIZE"] = str(args.native_graph_cache_size)
    model_cls = (
        NativeRWKV7ForCausalLM if args.code_source == "repo" else AutoModelForCausalLM
    )
    load_kwargs = {
        "torch_dtype": dtype,
        "device_map": args.device if args.device.startswith("cuda") else None,
    }
    if args.code_source == "model":
        load_kwargs["trust_remote_code"] = True
    model = model_cls.from_pretrained(
        args.hf_dir,
        **load_kwargs,
    ).eval()
    if args.fuse_norm != "auto":
        desired = args.fuse_norm == "true"
        actual = bool(getattr(model.config, "fuse_norm", False))
        if actual != desired:
            raise ValueError(
                f"Loaded model config has fuse_norm={actual}; use a converted model dir with fuse_norm={desired}"
            )
    set_attn_mode(model, args.attn_mode)
    for name in ("rwkv7_forward_token", "rwkv7_clear_native_graph_cache"):
        if not hasattr(model, name):
            raise ValueError(f"Loaded model does not expose {name}")
    return model


def encode(tok, prompt_tokens: int, batch_size: int, device: str) -> torch.Tensor:
    ids = tok(SEED, return_tensors="pt", add_special_tokens=False).input_ids[
        :, :prompt_tokens
    ]
    ids = ids.repeat(batch_size, 1)
    return ids.to(device) if device.startswith("cuda") else ids


def prefill(model, ids: torch.Tensor):
    out = model(ids, use_cache=True, logits_to_keep=1)
    token = out.logits[:, -1:].argmax(dim=-1)
    return token, out.past_key_values


def wagv_mode_flags(axis: str, enabled: bool) -> tuple[bool, bool, bool]:
    """Return grouped-WAGV, BMM, and exact-SM120 flags for one capture.

    ``ada_wagv_bmm`` isolates the BMM implementation behind an already-on
    grouped route. ``ada_wagv_bmm_from_default`` is the production-policy
    comparison for cards whose B8 fallback is still ungrouped.
    """

    if axis == "ada_wagv_bmm":
        return True, bool(enabled), False
    if axis == "ada_wagv_bmm_from_default":
        return bool(enabled), bool(enabled), False
    if axis == "ada_wagv_lora":
        return bool(enabled), False, False
    if axis == "sm120_wagv_bmm_g":
        return True, True, bool(enabled)
    raise ValueError(f"unsupported WAGV benchmark axis: {axis!r}")


def run_mode(
    model, token: torch.Tensor, base_state, args: argparse.Namespace, *, enabled: bool
) -> dict[str, Any]:
    grouped, bmm, sm120_g = wagv_mode_flags(args.axis, enabled)
    os.environ["RWKV7_NATIVE_GRAPH_ADA_WAGV_LORA"] = "1" if grouped else "0"
    os.environ["RWKV7_NATIVE_GRAPH_ADA_WAGV_BMM"] = "1" if bmm else "0"
    os.environ["RWKV7_NATIVE_GRAPH_SM120_WAGV_BMM_G"] = "1" if sm120_g else "0"
    os.environ["RWKV7_NATIVE_GRAPH_RKV_POLICY"] = args.rkv_policy
    os.environ["RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN"] = "0"
    os.environ["RWKV7_NATIVE_GRAPH_ADA_LINEAR"] = "0"
    os.environ["RWKV7_NATIVE_GRAPH_FUSED_PROJECTION"] = "0"
    os.environ["RWKV7_NATIVE_GRAPH_FUSED_WAVG_LORA"] = "0"
    os.environ["RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX"] = "1"
    os.environ["RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX_NUM_WARPS"] = str(args.num_warps)
    os.environ["RWKV7_NATIVE_GRAPH_FUSED_RECURRENT_RAW"] = "1"
    # Hold already-promoted decode fusions constant across both captures.
    os.environ["RWKV7_NATIVE_GRAPH_FUSED_RECURRENT_OUTPUT"] = "1"
    os.environ["RWKV7_NATIVE_GRAPH_FUSED_OUTPUT"] = "1"
    model.rwkv7_clear_native_graph_cache()
    if hasattr(model, "rwkv7_reset_native_graph_cache_stats"):
        model.rwkv7_reset_native_graph_cache_stats()
    state = base_state.clone()
    tok = token.clone()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        first = model.rwkv7_forward_token(tok, past_key_values=state)
        effective_backend = getattr(
            model, "rwkv7_last_fast_token_backend", lambda: None
        )()
        # Correctness is collected outside the timed region so device-to-host
        # token copies do not hide a small fused-kernel delta.
        state = base_state.clone()
        tok = token.clone()
        greedy_tokens: list[int] = []
        last_logits = None
        for _ in range(args.correctness_steps):
            out = model.rwkv7_forward_token(tok, past_key_values=state)
            last_logits = out.logits.detach().clone()
            tok = out.logits[:, -1:].argmax(dim=-1)
            greedy_tokens.extend(int(v) for v in tok.detach().cpu().reshape(-1))
        if last_logits is None:
            raise RuntimeError("correctness_steps must be at least one")
        state = base_state.clone()
        tok = token.clone()
        for _ in range(args.warmup):
            out = model.rwkv7_forward_token(tok, past_key_values=state)
            if not args.fixed_token:
                tok = out.logits[:, -1:].argmax(dim=-1)
        cuda_sync(args.device)
        t0 = time.perf_counter()
        for _ in range(args.steps):
            out = model.rwkv7_forward_token(tok, past_key_values=state)
            if not args.fixed_token:
                tok = out.logits[:, -1:].argmax(dim=-1)
            else:
                tok = token
        cuda_sync(args.device)
    ms_per_step = (time.perf_counter() - t0) * 1000.0 / float(args.steps)
    stats = (
        model.rwkv7_native_graph_cache_stats()
        if hasattr(model, "rwkv7_native_graph_cache_stats")
        else {}
    )
    bmm_route = wagv_bmm_route_status(model, args.batch_size)
    sm120_g_route = wagv_bmm_route_status(
        model, args.batch_size, route_prefix="sm120_wagv_bmm_g"
    )
    return {
        "effective_backend": effective_backend,
        "first_logits": first.logits.detach().clone(),
        "last_logits": last_logits,
        "ms_per_step": ms_per_step,
        "tokps_total": 1000.0 * int(token.numel()) / ms_per_step
        if ms_per_step > 0
        else None,
        "greedy_tokens": greedy_tokens,
        "cache_stats": stats,
        "bmm_requested_by_mode": bmm,
        "bmm_route": bmm_route,
        "sm120_g_requested_by_mode": sm120_g,
        "sm120_g_route": sm120_g_route,
        "peak_vram_mb": peak_mb(args.device),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--dtype", default="fp16", choices=sorted(DTYPES))
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--code-source",
        choices=("model", "repo"),
        default="model",
        help="load checkpoint-bundled remote code or the current repository implementation",
    )
    ap.add_argument(
        "--attn-mode", default="fused_recurrent", choices=["chunk", "fused_recurrent"]
    )
    ap.add_argument("--fuse-norm", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--fast-cache", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument(
        "--axis",
        choices=(
            "ada_wagv_lora",
            "ada_wagv_bmm",
            "ada_wagv_bmm_from_default",
            "sm120_wagv_bmm_g",
        ),
        default="ada_wagv_lora",
    )
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--correctness-steps", type=int, default=32)
    ap.add_argument("--fixed-token", action="store_true")
    ap.add_argument("--num-warps", type=int, choices=[1, 2, 4, 8], default=4)
    ap.add_argument(
        "--rkv-policy",
        choices=("manual", "vkwr_auto"),
        default="vkwr_auto",
        help="hold the R/K/V projection route explicit across both captures",
    )
    ap.add_argument("--native-graph-cache-size", type=int, default=8)
    ap.add_argument("--results", default=str(Path(__file__).parent / "results.jsonl"))
    args = ap.parse_args()

    if args.correctness_steps < 1:
        ap.error("--correctness-steps must be at least 1")

    dtype = DTYPES[args.dtype]
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    tok = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)
    model = load_model(args, dtype)
    extension_status = wagv_extension_status(model, args.device)
    ids = encode(tok, args.prompt_tokens, args.batch_size, args.device)
    with torch.inference_mode():
        os.environ["RWKV7_NATIVE_GRAPH_ADA_WAGV_LORA"] = "0"
        token, base_state = prefill(model, ids)
        baseline = run_mode(model, token, base_state, args, enabled=False)
        fused = run_mode(model, token, base_state, args, enabled=True)

    first_metrics = logits_pair_metrics(
        baseline["first_logits"], fused["first_logits"], batch_size=args.batch_size
    )
    last_metrics = logits_pair_metrics(
        baseline["last_logits"], fused["last_logits"], batch_size=args.batch_size
    )
    greedy_match, greedy_match_count, greedy_total = greedy_match_summary(
        baseline["greedy_tokens"], fused["greedy_tokens"]
    )
    num_layers = int(getattr(model.config, "num_hidden_layers", 0))
    baseline_bmm_route_pass = wagv_bmm_route_pass(
        baseline["bmm_route"],
        requested=bool(baseline["bmm_requested_by_mode"]),
        num_layers=num_layers,
    )
    fused_bmm_route_pass = wagv_bmm_route_pass(
        fused["bmm_route"],
        requested=bool(fused["bmm_requested_by_mode"]),
        num_layers=num_layers,
    )
    baseline_sm120_route_pass = wagv_bmm_route_pass(
        baseline["sm120_g_route"],
        requested=bool(baseline["sm120_g_requested_by_mode"]),
        num_layers=num_layers,
    )
    fused_sm120_route_pass = wagv_bmm_route_pass(
        fused["sm120_g_route"],
        requested=bool(fused["sm120_g_requested_by_mode"]),
        num_layers=num_layers,
    )
    correctness_pass = bool(
        greedy_match
        and first_metrics["finite"]
        and last_metrics["finite"]
        and first_metrics["min_cosine"] is not None
        and first_metrics["min_cosine"] >= 0.9999
        and last_metrics["min_cosine"] is not None
        and last_metrics["min_cosine"] >= 0.9999
        and baseline_bmm_route_pass
        and fused_bmm_route_pass
        and baseline_sm120_route_pass
        and fused_sm120_route_pass
    )
    row = {
        "axis": f"native_graph_{args.axis}",
        "backend": "hf_adapter",
        "status": "pass" if correctness_pass else "fail",
        "dtype": args.dtype,
        "device": device_name(args.device),
        "code_source": args.code_source,
        **model_metadata(args, model),
        "attn_mode": args.attn_mode,
        "fuse_norm": getattr(model.config, "fuse_norm", None),
        "fast_cache": os.environ.get("RWKV7_FAST_CACHE", "1") not in _FALSE_VALUES,
        "batch_size": args.batch_size,
        "prompt_tokens": int(ids.shape[1]),
        "steps": args.steps,
        "correctness_steps": args.correctness_steps,
        "fixed_token": args.fixed_token,
        "num_warps": args.num_warps,
        "rkv_policy": args.rkv_policy,
        **extension_status,
        "baseline_effective_backend": baseline["effective_backend"],
        "fused_effective_backend": fused["effective_backend"],
        "baseline_ada_wagv_bmm_requested": baseline["bmm_route"]["requested"],
        "baseline_ada_wagv_bmm_selected": baseline["bmm_route"]["selected"],
        "baseline_ada_wagv_bmm_effective": baseline["bmm_route"]["effective"],
        "baseline_ada_wagv_bmm_effective_layers": baseline["bmm_route"][
            "effective_layers"
        ],
        "baseline_ada_wagv_bmm_route_pass": baseline_bmm_route_pass,
        "fused_ada_wagv_bmm_requested": fused["bmm_route"]["requested"],
        "fused_ada_wagv_bmm_selected": fused["bmm_route"]["selected"],
        "fused_ada_wagv_bmm_effective": fused["bmm_route"]["effective"],
        "fused_ada_wagv_bmm_effective_layers": fused["bmm_route"]["effective_layers"],
        "fused_ada_wagv_bmm_route_pass": fused_bmm_route_pass,
        "baseline_sm120_wagv_bmm_g_requested": baseline["sm120_g_route"]["requested"],
        "baseline_sm120_wagv_bmm_g_selected": baseline["sm120_g_route"]["selected"],
        "baseline_sm120_wagv_bmm_g_effective": baseline["sm120_g_route"]["effective"],
        "baseline_sm120_wagv_bmm_g_effective_layers": baseline["sm120_g_route"][
            "effective_layers"
        ],
        "baseline_sm120_wagv_bmm_g_route_pass": baseline_sm120_route_pass,
        "fused_sm120_wagv_bmm_g_requested": fused["sm120_g_route"]["requested"],
        "fused_sm120_wagv_bmm_g_selected": fused["sm120_g_route"]["selected"],
        "fused_sm120_wagv_bmm_g_effective": fused["sm120_g_route"]["effective"],
        "fused_sm120_wagv_bmm_g_effective_layers": fused["sm120_g_route"][
            "effective_layers"
        ],
        "fused_sm120_wagv_bmm_g_route_pass": fused_sm120_route_pass,
        "baseline_ms_per_step": round(float(baseline["ms_per_step"]), 4),
        "fused_ms_per_step": round(float(fused["ms_per_step"]), 4),
        "speedup": round(
            float(baseline["ms_per_step"]) / float(fused["ms_per_step"]), 4
        ),
        "baseline_tokps_total": round(float(baseline["tokps_total"]), 1),
        "fused_tokps_total": round(float(fused["tokps_total"]), 1),
        "logits_finite_first_step": first_metrics["finite"],
        "max_abs_diff_first_step": first_metrics["max_abs_diff"],
        "min_cosine_first_step": first_metrics["min_cosine"],
        "logits_finite_last_step": last_metrics["finite"],
        "max_abs_diff_last_step": last_metrics["max_abs_diff"],
        "min_cosine_last_step": last_metrics["min_cosine"],
        "greedy_match": greedy_match,
        "greedy_match_count": greedy_match_count,
        "greedy_total": greedy_total,
        "correctness_pass": correctness_pass,
        "baseline_cache_stats": baseline["cache_stats"],
        "fused_cache_stats": fused["cache_stats"],
        "baseline_peak_vram_mb": baseline["peak_vram_mb"],
        "fused_peak_vram_mb": fused["peak_vram_mb"],
        "vram_delta_mb": (
            None
            if baseline["peak_vram_mb"] is None or fused["peak_vram_mb"] is None
            else round(
                float(fused["peak_vram_mb"]) - float(baseline["peak_vram_mb"]), 1
            )
        ),
    }
    print(json.dumps(row, indent=2, ensure_ascii=False), flush=True)
    if args.results:
        out = Path(args.results)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nappended 1 row -> {out}", flush=True)
    return 0 if correctness_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
