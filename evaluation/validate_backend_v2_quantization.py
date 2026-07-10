#!/usr/bin/env python3
"""Validate every optional quant adapter through reference and backend-v2."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from common import environment, git_revision, model_fingerprint, sha256_file


METHODS = (
    "native_w8",
    "native_w4",
    "a8w8",
    "torchao_w8",
    "torchao_w4",
    "marlin_w4",
    "marlin_bntn_w4",
    "bnb8",
    "bnb4",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--method", action="append", choices=METHODS, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=17)
    parser.add_argument("--greedy-tokens", type=int, default=8)
    parser.add_argument("--min-params", type=int, default=1_000_000)
    parser.add_argument("--policy", choices=("memory", "speed"), default="memory")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--code-sha")
    parser.add_argument("--hf-wheel", type=Path)
    parser.add_argument("--kernel-wheel", type=Path)
    return parser.parse_args()


def route(optimized: bool) -> None:
    os.environ["RWKV7_BACKEND"] = "optimized" if optimized else "reference"
    os.environ["RWKV7_MODEL_KERNEL_IMPL"] = "native" if optimized else "auto"
    os.environ["RWKV7_KERNEL_IMPL"] = "auto"


def metric(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    candidate = candidate.detach().float().reshape(-1)
    reference = reference.detach().float().reshape(-1)
    delta = (candidate - reference).abs()
    candidate_norm = float(candidate.norm())
    reference_norm = float(reference.norm())
    cosine = (
        1.0
        if candidate_norm == 0.0 and reference_norm == 0.0
        else float(
            torch.nn.functional.cosine_similarity(
                candidate.unsqueeze(0), reference.unsqueeze(0)
            )
        )
    )
    return {
        "finite": bool(torch.isfinite(candidate).all() and torch.isfinite(reference).all()),
        "cosine": cosine,
        "max_abs": float(delta.max()) if delta.numel() else 0.0,
        "mean_abs": float(delta.mean()) if delta.numel() else 0.0,
        "relative_l2": float(delta.norm()) / max(reference_norm, 1.0e-12),
    }


def model_dtype(method: str) -> torch.dtype:
    return (
        torch.bfloat16
        if method in {"torchao_w4", "marlin_w4", "marlin_bntn_w4"}
        else torch.float16
    )


def load_candidate(path: Path, method: str, args: argparse.Namespace):
    from rwkv7_hf.configuration_rwkv7 import RWKV7Config
    from rwkv7_hf.modeling_rwkv7 import RWKV7ForCausalLM
    from rwkv7_kernels.quantization import (
        prepare_bitsandbytes_config,
        quantization_report,
        quantize_model,
    )

    dtype = model_dtype(method)
    if method in {"bnb8", "bnb4"}:
        config = RWKV7Config.from_pretrained(path)
        bnb = prepare_bitsandbytes_config(
            method,
            config=config,
            policy="decode_hot" if args.policy == "speed" else "memory",
            compute_dtype=dtype,
        )
        model = RWKV7ForCausalLM.from_pretrained(
            path,
            dtype=dtype,
            quantization_config=bnb,
            device_map={"": 0},
        ).eval()
        report = quantize_model(model, method, policy=args.policy)
        return model, report

    model = RWKV7ForCausalLM.from_pretrained(path, dtype=dtype).cuda().eval()
    report = quantize_model(
        model,
        method,
        min_params=args.min_params,
        policy=args.policy,
        group_size=args.group_size,
        production_bn_tn=method == "marlin_bntn_w4",
    )
    assert quantization_report(model) == report
    return model, report


def cache_metric(candidate, reference) -> dict[str, Any]:
    rows = [
        metric(left, right)
        for left, right in zip(candidate.recurrent_state, reference.recurrent_state)
    ]
    return {
        "layers": rows,
        "passed": all(
            row["finite"] and row["cosine"] >= 0.9999 and row["max_abs"] <= 0.15
            for row in rows
        ),
    }


def validate_method(
    path: Path, method: str, args: argparse.Namespace, ids: torch.Tensor
) -> dict[str, Any]:
    from rwkv7_hf.modeling_rwkv7 import RWKV7ForCausalLM
    from rwkv7_hf.ops_rwkv7 import get_last_model_route

    dtype = model_dtype(method)
    dense = RWKV7ForCausalLM.from_pretrained(path, dtype=dtype).cuda().eval()
    with torch.inference_mode():
        route(False)
        dense_output = dense(input_ids=ids, use_cache=True)
    del dense
    torch.cuda.empty_cache()

    model, quantization = load_candidate(path, method, args)
    with torch.inference_mode():
        route(False)
        quant_reference = model(input_ids=ids, use_cache=True)
        route(True)
        quant_native = model(input_ids=ids, use_cache=True)
    actual_route = get_last_model_route()
    backend_logits = metric(quant_native.logits, quant_reference.logits)
    backend_cache = cache_metric(
        quant_native.past_key_values, quant_reference.past_key_values
    )
    quality_logits = metric(quant_reference.logits, dense_output.logits)

    prompt = ids[:1, : min(8, ids.shape[1])]
    generate_kwargs = {
        "max_new_tokens": args.greedy_tokens,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": 0,
        "eos_token_id": None,
    }
    with torch.inference_mode():
        route(False)
        reference_tokens = model.generate(prompt, **generate_kwargs)
        route(True)
        native_tokens = model.generate(prompt, **generate_kwargs)
    greedy_equal = bool(torch.equal(native_tokens, reference_tokens))
    quality_limit = 0.995 if method in {"native_w8", "a8w8", "torchao_w8", "bnb8"} else 0.97
    passed = bool(
        int(quantization["replaced_modules"]) > 0
        and backend_logits["finite"]
        and backend_logits["cosine"] >= 0.9999
        and backend_logits["max_abs"] <= 0.15
        and backend_cache["passed"]
        and quality_logits["finite"]
        and quality_logits["cosine"] >= quality_limit
        and greedy_equal
        and actual_route
        and str(actual_route.get("implementation", "")).startswith(
            "native-nvidia-prefill-v2["
        )
    )
    result = {
        "method": method,
        "dtype": str(dtype),
        "status": "passed" if passed else "failed",
        "quantization": quantization,
        "backend_logits": backend_logits,
        "backend_cache": backend_cache,
        "dense_quality_logits": quality_logits,
        "greedy_equal_between_quant_routes": greedy_equal,
        "route": actual_route,
    }
    del model
    torch.cuda.empty_cache()
    return result


def main() -> int:
    args = arguments()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    methods = tuple(args.method or METHODS)
    path = args.model.expanduser().resolve()
    # All methods see identical token IDs; every RWKV vocabulary contains this
    # conservative range.
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    ids = torch.randint(
        1,
        1024,
        (args.batch, args.tokens),
        generator=generator,
        device="cuda",
    )
    results = []
    capability = tuple(torch.cuda.get_device_capability())
    for method in methods:
        try:
            results.append(validate_method(path, method, args, ids))
        except Exception as exc:
            expected_unsupported = method == "marlin_bntn_w4" and capability != (12, 0)
            results.append({
                "method": method,
                "status": "not_applicable" if expected_unsupported else "failed",
                "expected_supported": not expected_unsupported,
                "device_capability": capability,
                "error": f"{type(exc).__name__}: {exc}",
            })
    wheels = {}
    for name, wheel in (("rwkv7_hf", args.hf_wheel), ("rwkv7_kernels", args.kernel_wheel)):
        if wheel is not None:
            wheels[name] = {"path": str(wheel), "sha256": sha256_file(wheel)}
    report = {
        "schema": "rwkv7-backend-v2-quantization-validation-v1",
        "status": (
            "passed"
            if all(row["status"] in {"passed", "not_applicable"} for row in results)
            else "failed"
        ),
        "code_sha": args.code_sha or git_revision(Path(__file__).resolve().parents[1]),
        "environment": environment(),
        "model": model_fingerprint(path),
        "wheels": wheels,
        "methods": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "status": report["status"]}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
