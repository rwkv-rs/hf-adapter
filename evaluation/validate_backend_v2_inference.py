#!/usr/bin/env python3
"""Validate the strict backend-v2 model route against the readable HF model."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import torch

from common import environment, git_revision, model_fingerprint, sha256_file


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, help="label=path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--batch", action="append", type=int, default=[])
    parser.add_argument("--tokens", action="append", type=int, default=[])
    parser.add_argument("--decode-steps", type=int, default=4)
    parser.add_argument("--greedy-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--code-sha")
    parser.add_argument("--hf-wheel", type=Path)
    parser.add_argument("--kernel-wheel", type=Path)
    return parser.parse_args()


def parse_model(value: str) -> tuple[str, Path]:
    label, path = value.split("=", 1)
    return label, Path(path).expanduser().resolve()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[name]


def metric(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    left = candidate.detach().float().reshape(-1)
    right = reference.detach().float().reshape(-1)
    delta = (left - right).abs()
    cosine = float(
        torch.nn.functional.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0))
    )
    return {
        "finite": bool(torch.isfinite(left).all() and torch.isfinite(right).all()),
        "cosine": cosine,
        "max_abs": float(delta.max()) if delta.numel() else 0.0,
        "mean_abs": float(delta.mean()) if delta.numel() else 0.0,
        "argmax_same": bool(
            torch.equal(
                candidate.detach().float().argmax(dim=-1),
                reference.detach().float().argmax(dim=-1),
            )
        ),
    }


def metric_passed(row: dict[str, Any], dtype: torch.dtype) -> bool:
    if not row["finite"]:
        return False
    if dtype == torch.float32:
        return row["max_abs"] <= 1.0e-4
    limit = 0.15 if dtype == torch.float16 else 0.30
    return row["cosine"] >= 0.9999 and row["max_abs"] <= limit


def route_mode(optimized: bool) -> None:
    os.environ["RWKV7_BACKEND"] = "optimized" if optimized else "reference"
    os.environ["RWKV7_KERNEL_IMPL"] = "auto"
    os.environ["RWKV7_MODEL_KERNEL_IMPL"] = "native" if optimized else "auto"


def last_route() -> dict[str, Any] | None:
    from rwkv7_hf.ops_rwkv7 import get_last_model_route

    return get_last_model_route()


def cache_rows(candidate, reference, dtype: torch.dtype) -> dict[str, Any]:
    rows = []
    for candidate_state, reference_state in zip(
        candidate.recurrent_state, reference.recurrent_state
    ):
        rows.append(metric(candidate_state, reference_state))
    passed = all(metric_passed(row, dtype) for row in rows)
    return {
        "passed": passed,
        "layers": rows,
        "candidate_seen_tokens": int(candidate.seen_tokens),
        "reference_seen_tokens": int(reference.seen_tokens),
    }


def run_model(
    label: str,
    path: Path,
    *,
    dtype: torch.dtype,
    device: torch.device,
    batches: tuple[int, ...],
    tokens: tuple[int, ...],
    decode_steps: int,
    greedy_tokens: int,
    seed: int,
) -> dict[str, Any]:
    from rwkv7_hf.modeling_rwkv7 import RWKV7ForCausalLM

    model = RWKV7ForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval()
    vocab = int(model.config.vocab_size)
    generator = torch.Generator(device=device).manual_seed(seed)
    cases: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for batch in batches:
        for sequence in tokens:
            ids = torch.randint(
                1,
                vocab,
                (batch, sequence),
                generator=generator,
                device=device,
            )
            with torch.inference_mode():
                route_mode(False)
                reference = model(input_ids=ids, use_cache=True, logits_to_keep=0)
                route_mode(True)
                candidate = model(input_ids=ids, use_cache=True, logits_to_keep=0)
            logits = metric(candidate.logits, reference.logits)
            state = cache_rows(candidate.past_key_values, reference.past_key_values, dtype)
            route = last_route()
            passed = bool(
                metric_passed(logits, dtype)
                and state["passed"]
                and route
                and route.get("selected") == "optimized"
                and str(route.get("implementation", "")).startswith(
                    "native-nvidia-prefill-v2["
                )
            )
            row = {
                "case": f"b{batch}-t{sequence}",
                "passed": passed,
                "logits": logits,
                "cache": state,
                "route": route,
            }
            cases.append(row)
            if not passed:
                failures.append(row)

    # One batch deliberately mixes right and left padding.
    sequence = max(tokens[0], 17)
    ids = torch.randint(1, vocab, (2, sequence), generator=generator, device=device)
    mask = torch.ones(2, sequence, dtype=torch.bool, device=device)
    mask[0, -3:] = False
    mask[1, :4] = False
    with torch.inference_mode():
        route_mode(False)
        reference = model(input_ids=ids, attention_mask=mask, use_cache=True)
        route_mode(True)
        candidate = model(input_ids=ids, attention_mask=mask, use_cache=True)
    logits = metric(candidate.logits, reference.logits)
    state = cache_rows(candidate.past_key_values, reference.past_key_values, dtype)
    route = last_route()
    padding_passed = bool(
        metric_passed(logits, dtype)
        and state["passed"]
        and route
        and "masked_compact" in str(route.get("implementation"))
    )
    padding_row = {
        "case": "mixed-left-right-padding",
        "passed": padding_passed,
        "logits": logits,
        "cache": state,
        "route": route,
    }
    cases.append(padding_row)
    if not padding_passed:
        failures.append(padding_row)

    # Teacher-forced cached decode compares every state transition.
    prompt = torch.randint(1, vocab, (2, 17), generator=generator, device=device)
    continuation = torch.randint(
        1, vocab, (2, decode_steps), generator=generator, device=device
    )
    with torch.inference_mode():
        route_mode(False)
        reference = model(input_ids=prompt, use_cache=True)
        reference_cache = reference.past_key_values
        reference_rows = []
        for index in range(decode_steps):
            output = model(
                input_ids=continuation[:, index : index + 1],
                past_key_values=reference_cache,
                use_cache=True,
            )
            reference_cache = output.past_key_values
            reference_rows.append(output.logits)

        route_mode(True)
        candidate = model(input_ids=prompt, use_cache=True)
        candidate_cache = candidate.past_key_values
        candidate_rows = []
        routes = []
        for index in range(decode_steps):
            output = model(
                input_ids=continuation[:, index : index + 1],
                past_key_values=candidate_cache,
                use_cache=True,
            )
            candidate_cache = output.past_key_values
            candidate_rows.append(output.logits)
            routes.append(last_route())
    decode_logits = metric(torch.cat(candidate_rows, 1), torch.cat(reference_rows, 1))
    decode_state = cache_rows(candidate_cache, reference_cache, dtype)
    decode_passed = bool(
        metric_passed(decode_logits, dtype)
        and decode_state["passed"]
        and all(
            route
            and str(route.get("implementation", "")).startswith(
                "native-nvidia-fused-decode-v2["
            )
            for route in routes
        )
    )
    decode_row = {
        "case": "teacher-forced-cached-decode",
        "passed": decode_passed,
        "logits": decode_logits,
        "cache": decode_state,
        "routes": routes,
    }
    cases.append(decode_row)
    if not decode_passed:
        failures.append(decode_row)

    generation_prompt = prompt[:1, :8]
    generation_kwargs = {
        "max_new_tokens": greedy_tokens,
        "do_sample": False,
        "use_cache": True,
        "pad_token_id": 0,
        "eos_token_id": None,
    }
    with torch.inference_mode():
        route_mode(False)
        reference_greedy = model.generate(generation_prompt, **generation_kwargs)
        route_mode(True)
        candidate_greedy = model.generate(generation_prompt, **generation_kwargs)
    greedy_passed = bool(torch.equal(candidate_greedy, reference_greedy))
    greedy_row = {
        "case": "greedy-generation",
        "passed": greedy_passed,
        "tokens": greedy_tokens,
        "equal": greedy_passed,
        "route": last_route(),
    }
    cases.append(greedy_row)
    if not greedy_passed:
        failures.append(greedy_row)

    with torch.inference_mode():
        route_mode(False)
        reference_beam = model.generate(
            generation_prompt,
            max_new_tokens=4,
            num_beams=2,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=None,
        )
        route_mode(True)
        candidate_beam = model.generate(
            generation_prompt,
            max_new_tokens=4,
            num_beams=2,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
            eos_token_id=None,
        )
    beam_passed = bool(torch.equal(candidate_beam, reference_beam))
    beam_row = {
        "case": "beam-generation",
        "passed": beam_passed,
        "equal": beam_passed,
        "route": last_route(),
    }
    cases.append(beam_row)
    if not beam_passed:
        failures.append(beam_row)

    del model
    torch.cuda.empty_cache()
    return {
        "label": label,
        "model": model_fingerprint(path),
        "status": "passed" if not failures else "failed",
        "cases": cases,
        "failures": failures,
    }


def main() -> int:
    args = arguments()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")
    dtype = dtype_from_name(args.dtype)
    specs = [parse_model(value) for value in args.model]
    batches = tuple(args.batch or (1, 4, 8))
    tokens = tuple(args.tokens or (17, 128))
    torch.manual_seed(args.seed)
    reports = [
        run_model(
            label,
            path,
            dtype=dtype,
            device=device,
            batches=batches,
            tokens=tokens,
            decode_steps=args.decode_steps,
            greedy_tokens=args.greedy_tokens,
            seed=args.seed + index,
        )
        for index, (label, path) in enumerate(specs)
    ]
    wheel_hashes = {}
    for name, path in (("rwkv7_hf", args.hf_wheel), ("rwkv7_kernels", args.kernel_wheel)):
        if path is not None:
            wheel_hashes[name] = {"path": str(path), "sha256": sha256_file(path)}
    report = {
        "schema": "rwkv7-backend-v2-inference-validation-v1",
        "status": "passed" if all(row["status"] == "passed" for row in reports) else "failed",
        "code_sha": args.code_sha or git_revision(Path(__file__).resolve().parents[1]),
        "dtype": args.dtype,
        "batches": batches,
        "tokens": tokens,
        "environment": environment(),
        "wheels": wheel_hashes,
        "models": reports,
    }
    if not all(math.isfinite(value) for value in (float(len(reports)),)):
        raise AssertionError("unreachable non-finite report")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "status": report["status"]}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
