#!/usr/bin/env python3
# coding=utf-8
"""Capture and compare true sequence prefill against pinned official v3a.

Each engine runs in a separate process so a 32 GiB card only holds one model.
The capture includes the public prefill result, FP16 recurrent/shift state,
last-token residual output after every layer, and the first cached decode step.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

import torch

# Running ``python scripts/...`` otherwise resolves an older site-packages
# adapter before the checkout under test.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:  # noqa: E402
    from scripts.compare_official_native_inference import (
        DEFAULT_PROMPT,
        load_official,
        metrics_pass as inference_metrics_pass,
        metrics_pass_official_envelope,
        sha256_file,
        tensor_metrics,
    )
except ImportError:  # direct ``python scripts/...`` execution
    from compare_official_native_inference import (
        DEFAULT_PROMPT,
        load_official,
        metrics_pass as inference_metrics_pass,
        metrics_pass_official_envelope,
        sha256_file,
        tensor_metrics,
    )


THRESHOLDS = {
    "logits": {"min_cosine": 0.9999, "max_abs": 0.125},
    "first_decode_logits": {
        "min_cosine": 0.9999,
        "max_abs": 0.125,
        "fp16_max_ulps_at_max": 4.0,
        "fp16_max_abs_tail": 0.1875,
        "fp16_max_fraction_over_abs": 5.0e-5,
    },
    "state": {
        "min_cosine": 0.9999,
        "max_abs": 4.0,
        "max_mean_abs": 0.001,
        "outlier_abs": 1.0,
        "max_outlier_fraction": 1.0e-5,
    },
    "xpa": {"min_cosine": 0.9999, "max_abs": 0.125},
    "xpf": {"min_cosine": 0.9999, "max_abs": 0.125},
    "layer_outputs": {"min_cosine": 0.9999, "max_abs": 0.25},
}


def metric_pass(metrics: dict[str, Any], kind: str) -> bool:
    if kind == "first_decode_logits":
        return inference_metrics_pass(metrics, "logits")
    if kind in {"xpa", "xpf"}:
        return inference_metrics_pass(metrics, kind)
    threshold = THRESHOLDS[kind]
    passed = bool(
        metrics["finite"]
        and metrics["cosine"] >= threshold["min_cosine"]
        and metrics["max_abs"] <= threshold["max_abs"]
    )
    if "max_mean_abs" in threshold:
        passed = passed and metrics["mean_abs"] <= threshold["max_mean_abs"]
    if "max_outlier_fraction" in threshold:
        passed = passed and (
            metrics.get("outlier_fraction", 1.0)
            <= threshold["max_outlier_fraction"]
        )
    return bool(passed)


def metric_pass_official_self_envelope(
    name: str,
    metrics: dict[str, Any],
    envelope: dict[str, Any] | None,
    *,
    multiplier: float,
) -> bool:
    """Limit process-variance relief to the first cached decode logits."""

    if name != "first_decode_logits":
        return False
    return metrics_pass_official_envelope(
        metrics,
        envelope,
        multiplier=multiplier,
    )


def native_runtime_environment() -> dict[str, str]:
    """Capture explicit Native runtime controls needed to replay a row."""

    return {
        name: os.environ[name]
        for name in sorted(os.environ)
        if name.startswith("RWKV7_NATIVE_")
    }


def prompt_ids(args: argparse.Namespace) -> torch.Tensor:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.hf_dir, trust_remote_code=True)
    text = DEFAULT_PROMPT * max(2, args.prompt_tokens // 128 + 2)
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
    if int(ids.shape[1]) < args.prompt_tokens:
        raise RuntimeError("deterministic prompt text did not produce enough tokens")
    return ids[:, : args.prompt_tokens].repeat(args.batch_size, 1)


def cuda_time(call, *, warmup: int, repeats: int) -> tuple[Any, list[float]]:
    result = None
    for _ in range(warmup):
        result = call()
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = call()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
    return result, times


def cuda_time_discard(call, *, warmup: int, repeats: int) -> list[float]:
    """Time independent requests without retaining an unused output cache."""

    for _ in range(warmup):
        result = call()
        del result
    torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = call()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
        del result
    return times


def timing_summary(times_ms: list[float], tokens: int) -> dict[str, Any]:
    median_ms = statistics.median(times_ms)
    return {
        "times_ms": times_ms,
        "median_ms": median_ms,
        "aggregate_tokps": tokens * 1000.0 / median_ms,
    }


def native_snapshot(cache: Any) -> dict[str, torch.Tensor]:
    return {
        "state": torch.stack([value.detach().cpu() for value in cache._state]),
        "xpa": torch.stack([value.detach().cpu() for value in cache._xpa]),
        "xpf": torch.stack([value.detach().cpu() for value in cache._xpf]),
    }


def capture_native(args: argparse.Namespace) -> dict[str, Any]:
    from rwkv7_hf.native_model import NativeRWKV7ForCausalLM

    ids = prompt_ids(args).to(args.device)
    model = NativeRWKV7ForCausalLM.from_pretrained(
        args.hf_dir,
        torch_dtype=torch.float16,
        device_map=args.device,
    ).eval()
    torch.cuda.reset_peak_memory_stats()

    static_ids = ids.clone() if args.native_cuda_graph else ids

    def run_prefill():
        return model(static_ids, use_cache=True, logits_to_keep=1)

    with torch.inference_mode():
        old_capture = os.environ.get("RWKV7_NATIVE_PREFILL_CAPTURE_LAYER_OUTPUTS")
        os.environ["RWKV7_NATIVE_PREFILL_CAPTURE_LAYER_OUTPUTS"] = "0"
        try:
            if args.native_cuda_graph:
                warmup_stream = torch.cuda.Stream()
                warmup_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(warmup_stream):
                    for _ in range(args.warmup):
                        run_prefill()
                torch.cuda.current_stream().wait_stream(warmup_stream)
                torch.cuda.synchronize()

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = run_prefill()
                graph.replay()
                torch.cuda.synchronize()
                times = []
                for _ in range(args.repeats):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    times.append(float(start.elapsed_time(end)))
            else:
                times = cuda_time_discard(
                    run_prefill,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )

            # Keep audit copies outside the production timing scope, matching
            # the official capture path below. The final run still exercises
            # the same public HF forward and cache handoff.
            os.environ["RWKV7_NATIVE_PREFILL_CAPTURE_LAYER_OUTPUTS"] = "1"
            output = run_prefill()
            torch.cuda.synchronize()
        finally:
            if old_capture is None:
                os.environ.pop("RWKV7_NATIVE_PREFILL_CAPTURE_LAYER_OUTPUTS", None)
            else:
                os.environ["RWKV7_NATIVE_PREFILL_CAPTURE_LAYER_OUTPUTS"] = old_capture

        cache = output.past_key_values
        prefill_state = native_snapshot(cache)
        prefill_state_dtype = str(cache._state[0].dtype)
        layer_values = getattr(model, "_rwkv7_native_prefill_layer_outputs", None)
        if not layer_values:
            raise RuntimeError(
                "Native prefill did not expose benchmark layer outputs: "
                f"backend={model.rwkv7_native_model_last_prefill_backend()}, "
                f"fp16_effective={getattr(model, '_rwkv7_native_prefill_fp16_recurrent_effective', None)}"
            )
        layers = torch.stack([value.detach().cpu() for value in layer_values])
        first_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        first_decode = model.rwkv7_forward_token(
            first_token,
            past_key_values=cache,
            return_dict=True,
            copy_logits=True,
        )
        torch.cuda.synchronize()
        first_decode_state_dtype = str(cache._state[0].dtype)

    return {
        "engine": "native_hf_sequence",
        "source_revision": args.native_source_revision,
        "batch_size": args.batch_size,
        "prompt_tokens": args.prompt_tokens,
        "prompt_ids": ids.cpu(),
        "precision": "fp16_state_fp16_io",
        "timing_scope": (
            "fixed_shape_cuda_graph_replay_including_cache_initialization"
            if args.native_cuda_graph
            else "public_hf_forward_including_cache_initialization"
        ),
        "execution": "cuda_graph" if args.native_cuda_graph else "eager",
        "timing": timing_summary(times, int(ids.numel())),
        "prefill_backend": model.rwkv7_native_model_last_prefill_backend(),
        "fp16_recurrent_effective": bool(
            getattr(model, "_rwkv7_native_prefill_fp16_recurrent_effective", False)
        ),
        "stacked_rkv_effective": bool(
            getattr(model, "_rwkv7_native_prefill_stacked_rkv_effective", False)
        ),
        "wavg_lora_effective": bool(
            getattr(model, "_rwkv7_native_prefill_wavg_lora_effective", False)
        ),
        "sequence_ffn_effective": bool(
            getattr(model, "_rwkv7_native_prefill_sequence_ffn_effective", False)
        ),
        "fp16_accum_ffn_key_effective": bool(
            getattr(
                model,
                "_rwkv7_native_prefill_fp16_accum_ffn_key_effective",
                False,
            )
        ),
        "prefill_state_dtype": prefill_state_dtype,
        "first_decode_state_dtype": first_decode_state_dtype,
        "seen_tokens": int(args.prompt_tokens),
        "logits": output.logits[:, -1].detach().cpu(),
        "first_token": first_token.cpu(),
        "first_decode_logits": first_decode.logits[:, -1].detach().cpu(),
        "first_decode_token": first_decode.logits[:, -1].argmax(dim=-1).cpu(),
        "first_decode_backend": model.rwkv7_native_model_last_decode_backend(),
        "runtime_env": native_runtime_environment(),
        "prefill": prefill_state,
        "layer_outputs": layers,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
    }


def install_official_layer_capture(model: Any, module: Any):
    boundaries = {
        int(model.z[f"blocks.{layer}.ln1.weight"].data_ptr()): layer - 1
        for layer in range(1, int(module.L))
    }
    original_add_ln = model.add_ln
    original_add_last_ln = model.add_last_ln
    capture: dict[str, Any] = {"active": False, "values": {}}

    def add_ln(x, residual, weight, bias):
        result = original_add_ln(x, residual, weight, bias)
        if capture["active"]:
            layer = boundaries.get(int(weight.data_ptr()))
            if layer is not None:
                capture["values"][layer] = result[0][:, -1].detach().clone()
        return result

    def add_last_ln(x, residual, weight, bias):
        if capture["active"]:
            capture["values"][int(module.L) - 1] = model.add(x, residual)[:, -1].detach().clone()
        return original_add_last_ln(x, residual, weight, bias)

    model.add_ln = add_ln
    model.add_last_ln = add_last_ln
    return capture


def official_snapshot(state: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "state": state[1].detach().cpu(),
        "xpa": state[0][:, 0].detach().cpu(),
        "xpf": state[0][:, 1].detach().cpu(),
    }


def capture_official(args: argparse.Namespace) -> dict[str, Any]:
    ids = prompt_ids(args).to(args.device)
    model, revision, verification = load_official(args)
    module = sys.modules[args.official_module]
    layer_capture = install_official_layer_capture(model, module)
    torch.cuda.reset_peak_memory_stats()

    def run_prefill():
        state = model.zero_state(args.batch_size)
        logits = model.forward(ids, state)
        return logits, state

    with torch.inference_mode():
        _, times = cuda_time(run_prefill, warmup=args.warmup, repeats=args.repeats)
        layer_capture["values"] = {}
        layer_capture["active"] = True
        output, state = run_prefill()
        torch.cuda.synchronize()
        layer_capture["active"] = False
        values = layer_capture["values"]
        if sorted(values) != list(range(int(module.L))):
            raise RuntimeError(f"official layer capture is incomplete: {sorted(values)}")
        layers = torch.stack([values[index].cpu() for index in range(int(module.L))])
        prefill_state = official_snapshot(state)
        seen_tokens = int(state[2][0].item())
        first_token = output.argmax(dim=-1, keepdim=True)
        first_decode_logits = model.forward(first_token, state).view(args.batch_size, -1)
        torch.cuda.synchronize()

    return {
        "engine": "official_v3a_sequence",
        "source_revision": revision,
        "source_verification": verification,
        "runtime": {
            "wkv": "fp16",
            "emb": args.official_emb,
            "batched_rkv": args.official_batched_rkv,
            "cmix_sparse": args.official_cmix_sparse,
            "lowrank_weight": args.official_lowrank_weight,
            "orig_linear_groups": args.official_orig_linear_groups,
        },
        "batch_size": args.batch_size,
        "prompt_tokens": args.prompt_tokens,
        "prompt_ids": ids.cpu(),
        "precision": "fp16_state_fp16_io",
        "timing_scope": "official_forward_with_preallocated_zero_state",
        "timing": timing_summary(times, int(ids.numel())),
        "state_dtype": str(state[1].dtype),
        "seen_tokens": seen_tokens,
        "logits": output.detach().cpu(),
        "first_token": first_token.cpu(),
        "first_decode_logits": first_decode_logits.detach().cpu(),
        "first_decode_token": first_decode_logits.argmax(dim=-1).cpu(),
        "prefill": prefill_state,
        "layer_outputs": layers,
        "peak_vram_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
    }


def compare(args: argparse.Namespace) -> dict[str, Any]:
    native = torch.load(args.native_capture, map_location="cpu", weights_only=False)
    official = torch.load(args.official_capture, map_location="cpu", weights_only=False)
    for key in ("batch_size", "prompt_tokens", "precision"):
        if native[key] != official[key]:
            raise ValueError(f"capture metadata mismatch for {key}")
    if official["source_revision"] != args.official_commit:
        raise ValueError("official capture does not match the required pinned commit")
    if not torch.equal(native["prompt_ids"], official["prompt_ids"]):
        raise ValueError("prompt token IDs differ")
    official_self_envelope = (
        json.loads(Path(args.official_self_envelope).read_text(encoding="utf-8"))
        if args.official_self_envelope
        else None
    )
    if official_self_envelope is not None:
        for key, expected in (
            ("axis", "official_prefill_self_repeat"),
            ("official_commit", official["source_revision"]),
            ("precision", native["precision"]),
            ("batch_size", native["batch_size"]),
            ("prompt_tokens", native["prompt_tokens"]),
        ):
            if official_self_envelope.get(key) != expected:
                raise ValueError(f"official prefill self-envelope mismatch for {key}")

    metrics: dict[str, Any] = {}
    for name in ("logits", "first_decode_logits", "layer_outputs"):
        item = tensor_metrics(native[name], official[name])
        item["standard_threshold_pass"] = metric_pass(item, name)
        item["official_self_envelope_pass"] = metric_pass_official_self_envelope(
            name,
            item,
            (official_self_envelope or {}).get("envelope", {}).get(name),
            multiplier=args.official_envelope_multiplier,
        )
        item["threshold_pass"] = bool(
            item["standard_threshold_pass"]
            or item["official_self_envelope_pass"]
        )
        metrics[name] = item
    for name in ("state", "xpa", "xpf"):
        item = tensor_metrics(native["prefill"][name], official["prefill"][name])
        if name == "state":
            difference = (
                native["prefill"][name].float()
                - official["prefill"][name].float()
            ).abs()
            outlier_abs = float(THRESHOLDS["state"]["outlier_abs"])
            outlier_count = int((difference > outlier_abs).sum().item())
            item.update(
                {
                    "outlier_abs": outlier_abs,
                    "outlier_count": outlier_count,
                    "outlier_fraction": outlier_count / difference.numel(),
                }
            )
        item["threshold_pass"] = metric_pass(item, name)
        metrics[name] = item

    native_tokps = float(native["timing"]["aggregate_tokps"])
    official_tokps = float(official["timing"]["aggregate_tokps"])
    quality_pass = bool(
        all(item["threshold_pass"] for item in metrics.values())
        and torch.equal(native["first_token"], official["first_token"])
        and torch.equal(native["first_decode_token"], official["first_decode_token"])
        and native["seen_tokens"] == official["seen_tokens"]
        and native["fp16_recurrent_effective"]
        and native["prefill_state_dtype"] == "torch.float16"
        and native["first_decode_state_dtype"] == "torch.float16"
    )
    report = {
        "axis": "official_native_sequence_prefill_alignment",
        "status": "pass" if quality_pass else "fail",
        "quality_pass": quality_pass,
        "performance_gate_pass": native_tokps >= official_tokps,
        "batch_size": native["batch_size"],
        "prompt_tokens": native["prompt_tokens"],
        "precision": native["precision"],
        "official_commit": official["source_revision"],
        "native_capture_sha256": sha256_file(args.native_capture),
        "official_capture_sha256": sha256_file(args.official_capture),
        "thresholds": THRESHOLDS,
        "metrics": metrics,
        "first_token_exact": bool(torch.equal(native["first_token"], official["first_token"])),
        "first_decode_token_exact": bool(
            torch.equal(native["first_decode_token"], official["first_decode_token"])
        ),
        "native": {
            "source_revision": native["source_revision"],
            "timing": native["timing"],
            "peak_vram_mb": native["peak_vram_mb"],
            "execution": native.get("execution", "eager"),
            "backend": native["prefill_backend"],
            "first_decode_backend": native["first_decode_backend"],
            "prefill_state_dtype": native["prefill_state_dtype"],
            "first_decode_state_dtype": native["first_decode_state_dtype"],
            "stacked_rkv_effective": native["stacked_rkv_effective"],
            "wavg_lora_effective": native["wavg_lora_effective"],
            "sequence_ffn_effective": native["sequence_ffn_effective"],
            "fp16_accum_ffn_key_effective": native[
                "fp16_accum_ffn_key_effective"
            ],
            "runtime_env": native.get("runtime_env", {}),
        },
        "official": {
            "source_revision": official["source_revision"],
            "timing": official["timing"],
            "peak_vram_mb": official["peak_vram_mb"],
            "runtime": official.get("runtime", {}),
            "source_verification": official.get("source_verification", {}),
        },
        "native_over_official_tokps": native_tokps / official_tokps,
    }
    if official_self_envelope is not None:
        report["official_self_envelope"] = {
            "axis": official_self_envelope.get("axis"),
            "official_commit": official_self_envelope.get("official_commit"),
            "precision": official_self_envelope.get("precision"),
            "batch_size": official_self_envelope.get("batch_size"),
            "prompt_tokens": official_self_envelope.get("prompt_tokens"),
            "multiplier": args.official_envelope_multiplier,
            "envelope": official_self_envelope.get("envelope"),
        }
    return report


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=("capture-native", "capture-official", "compare"))
    ap.add_argument("--hf-dir")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--output")
    ap.add_argument("--native-source-revision", default="working-tree")
    ap.add_argument("--native-cuda-graph", action="store_true")
    ap.add_argument("--official-dir")
    ap.add_argument("--official-model")
    ap.add_argument("--official-module", default="rwkv7_fast_v3a")
    ap.add_argument("--official-commit", default="cc57df475465c6cacd42ecd4f2f05a588ee5473b")
    ap.add_argument("--official-source-manifest")
    ap.add_argument("--official-emb", choices=("gpu", "cpu"), default="gpu")
    ap.add_argument(
        "--official-batched-rkv", choices=("auto", "on", "off"), default="off"
    )
    ap.add_argument(
        "--official-cmix-sparse", choices=("auto", "no-fc", "off"), default="no-fc"
    )
    ap.add_argument(
        "--official-lowrank-weight",
        choices=("orig", "transpose", "both"),
        default="both",
    )
    ap.add_argument(
        "--official-orig-linear-groups", default="att_c2c,ffn_key,head"
    )
    ap.add_argument("--native-capture")
    ap.add_argument("--official-capture")
    ap.add_argument("--official-self-envelope", default="")
    ap.add_argument("--official-envelope-multiplier", type=float, default=1.25)
    return ap


def main() -> int:
    args = parser().parse_args()
    if args.mode == "capture-native":
        result = capture_native(args)
    elif args.mode == "capture-official":
        result = capture_official(args)
    else:
        result = compare(args)
    if not args.output:
        raise ValueError("--output is required")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "compare":
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["status"] == "pass" else 1
    torch.save(result, output)
    print(json.dumps({
        "engine": result["engine"],
        "batch_size": result["batch_size"],
        "prompt_tokens": result["prompt_tokens"],
        "timing": result["timing"],
        "peak_vram_mb": result["peak_vram_mb"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
