#!/usr/bin/env python3
"""Validate reference/native RWKV-7 against one pinned FLA checkout.

The script records three distinct lanes.  The optimized lane is accepted only
when the actual whole-model trace names a backend-v2 prefill/decode/training
implementation; requesting an environment selector is not route evidence.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from common import environment, git_revision, model_fingerprint, sha256_file
from fla_common import (
    activate_fla_source,
    compare_states,
    gradient_metrics,
    gradient_rows_passed,
    metric_passed,
    recurrent_states,
    state_rows_passed,
    tensor_metric,
    write_json,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", required=True, help="label=path")
    parser.add_argument("--fla-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--batch", action="append", type=int, default=[])
    parser.add_argument("--tokens", action="append", type=int, default=[])
    parser.add_argument("--decode-steps", type=int, default=16)
    parser.add_argument("--greedy-tokens", type=int, default=64)
    parser.add_argument(
        "--training-model",
        help="label from --model used for BF16 full-gradient three-way parity",
    )
    parser.add_argument("--training-batch", type=int, default=1)
    parser.add_argument("--training-tokens", type=int, default=16)
    parser.add_argument(
        "--training-mode",
        choices=("native", "skip-not-applicable"),
        default="native",
        help="SM70 uses skip-not-applicable because train_temp requires BF16/sm80.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--code-sha")
    parser.add_argument("--hf-wheel", type=Path)
    parser.add_argument("--kernel-wheel", type=Path)
    return parser.parse_args()


def parse_models(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        label, path = value.split("=", 1)
        if label in result:
            raise ValueError(f"duplicate model label: {label}")
        result[label] = Path(path).expanduser().resolve()
    return result


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[name]


def route_mode(optimized: bool) -> None:
    os.environ["RWKV7_BACKEND"] = "optimized" if optimized else "reference"
    os.environ["RWKV7_KERNEL_IMPL"] = "auto"
    os.environ["RWKV7_MODEL_KERNEL_IMPL"] = "native" if optimized else "auto"


def last_model_route() -> dict[str, Any] | None:
    from rwkv7_hf.ops_rwkv7 import get_last_model_route

    return get_last_model_route()


def route_is(route: dict[str, Any] | None, phase: str) -> bool:
    if not route or route.get("selected") != "optimized":
        return False
    implementation = str(route.get("implementation", ""))
    expected = {
        "prefill": "native-nvidia-prefill-v2[",
        "decode": "native-nvidia-fused-decode-v2[",
        "training": "native-nvidia-train-temp-autograd-v2",
    }[phase]
    return implementation.startswith(expected)


def recurrent_route_is_optimized(route: dict[str, Any] | None) -> bool:
    return bool(
        route
        and route.get("selected") == "optimized"
        and route.get("implementation")
        in {
            "native-triton-rank1-scan-v1",
            "torch-cuda-graph-reference-v1",
        }
    )


def run_operator_parity(
    dtype: torch.dtype,
    batches: tuple[int, ...],
    lengths: tuple[int, ...],
    seed: int,
) -> dict[str, Any]:
    from fla.ops.rwkv7 import chunk_rwkv7
    from rwkv7_hf.ops_rwkv7 import (
        get_last_recurrent_route,
        rwkv7_recurrent,
        rwkv7_recurrent_reference,
    )

    rows = []
    for batch in batches:
        for length in lengths:
            generator = torch.Generator(device="cuda").manual_seed(
                seed + batch * 1000 + length
            )
            shape = (batch, length, 2, 64)
            base = {
                name: torch.randn(
                    shape, device="cuda", dtype=dtype, generator=generator
                )
                * 0.1
                for name in ("r", "k", "v", "a", "b")
            }
            base["w"] = -(
                torch.rand(shape, device="cuda", dtype=dtype, generator=generator) * 0.5
                + 0.1
            )
            base["state"] = (
                torch.randn(
                    (batch, 2, 64, 64),
                    device="cuda",
                    dtype=torch.float32,
                    generator=generator,
                )
                * 0.01
            )
            with torch.inference_mode():
                reference_output, reference_state = rwkv7_recurrent_reference(
                    base["r"],
                    base["w"].exp(),
                    base["k"],
                    base["v"],
                    base["a"],
                    base["b"],
                    base["state"],
                )
                route_mode(True)
                optimized_output, optimized_state = rwkv7_recurrent(
                    base["r"],
                    base["w"].exp(),
                    base["k"],
                    base["v"],
                    base["a"],
                    base["b"],
                    base["state"],
                )
                optimized_route = get_last_recurrent_route()
                fla_output, fla_state = chunk_rwkv7(
                    r=base["r"],
                    w=base["w"],
                    k=base["k"],
                    v=base["v"],
                    a=base["a"],
                    b=base["b"],
                    initial_state=base["state"],
                    output_final_state=True,
                )
            comparisons = {}
            for name, output, state in (
                ("optimized", optimized_output, optimized_state),
                ("fla", fla_output, fla_state),
            ):
                output_row = tensor_metric(output.cpu(), reference_output.cpu())
                state_row = tensor_metric(state.cpu(), reference_state.cpu())
                comparisons[name] = {
                    "passed": metric_passed(output_row, dtype)
                    and metric_passed(state_row, dtype),
                    "output": output_row,
                    "state": state_row,
                }

            gradient_lanes = {}
            for name in ("reference", "fla"):
                values = {
                    key: value.detach().clone().requires_grad_(True)
                    for key, value in base.items()
                }
                if name == "reference":
                    output, state = rwkv7_recurrent_reference(
                        values["r"],
                        values["w"].exp(),
                        values["k"],
                        values["v"],
                        values["a"],
                        values["b"],
                        values["state"],
                    )
                else:
                    output, state = chunk_rwkv7(
                        r=values["r"],
                        w=values["w"],
                        k=values["k"],
                        v=values["v"],
                        a=values["a"],
                        b=values["b"],
                        initial_state=values["state"],
                        output_final_state=True,
                    )
                loss = output.float().square().mean() + state.float().square().mean()
                loss.backward()
                gradient_lanes[name] = {
                    key: value.grad.detach().cpu() for key, value in values.items()
                }
            gradients = gradient_metrics(
                gradient_lanes["fla"], gradient_lanes["reference"]
            )
            gradient_passed = gradient_rows_passed(gradients, dtype)
            passed = bool(
                comparisons["optimized"]["passed"]
                and comparisons["fla"]["passed"]
                and recurrent_route_is_optimized(optimized_route)
                and gradient_passed
            )
            rows.append(
                {
                    "case": f"b{batch}-t{length}",
                    "passed": passed,
                    "comparisons": comparisons,
                    "optimized_route": optimized_route,
                    "fla_vs_reference_gradients": gradients,
                    "gradient_passed": gradient_passed,
                }
            )
            del (
                base,
                reference_output,
                reference_state,
                optimized_output,
                optimized_state,
            )
            del fla_output, fla_state, gradient_lanes
    return {"passed": all(row["passed"] for row in rows), "cases": rows}


def clean_model(path: Path, dtype: torch.dtype):
    from rwkv7_hf.modeling_rwkv7 import RWKV7ForCausalLM

    return RWKV7ForCausalLM.from_pretrained(path, dtype=dtype).cuda().eval()


def fla_model(path: Path, dtype: torch.dtype, *, training: bool = False):
    from fla.models.rwkv7.configuration_rwkv7 import RWKV7Config
    from fla.models.rwkv7.modeling_rwkv7 import RWKV7ForCausalLM

    config = RWKV7Config.from_pretrained(path)
    # Keep the mathematical contract explicit.  These flags otherwise make
    # the comparison dependent on whichever optional FLA fusions are installed.
    config.fuse_norm = False
    config.fuse_cross_entropy = False
    config.fuse_linear_cross_entropy = False
    config.use_l2warp = False
    model = RWKV7ForCausalLM.from_pretrained(path, config=config, dtype=dtype).cuda()
    return model.train() if training else model.eval()


def manual_cached(model, prompt: torch.Tensor, continuation: torch.Tensor):
    output = model(input_ids=prompt, use_cache=True, logits_to_keep=0)
    cache = output.past_key_values
    logits = []
    for index in range(int(continuation.shape[1])):
        output = model(
            input_ids=continuation[:, index : index + 1],
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=0,
        )
        cache = output.past_key_values
        logits.append(output.logits.detach().cpu())
    return torch.cat(logits, dim=1), cache


def manual_greedy(model, prompt: torch.Tensor, count: int):
    output = model(input_ids=prompt, use_cache=True, logits_to_keep=0)
    cache = output.past_key_values
    token = output.logits[:, -1].argmax(-1, keepdim=True)
    generated = [token.detach().cpu()]
    routes = [last_model_route()]
    for _ in range(count - 1):
        output = model(
            input_ids=token,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=0,
        )
        cache = output.past_key_values
        token = output.logits[:, -1].argmax(-1, keepdim=True)
        generated.append(token.detach().cpu())
        routes.append(last_model_route())
    return torch.cat(generated, dim=1), routes


def cache_snapshot(cache: Any) -> list[torch.Tensor]:
    return [value.detach().float().cpu() for value in recurrent_states(cache)]


def collect_clean_lane(
    model,
    inputs: dict[str, tuple[torch.Tensor, torch.Tensor | None]],
    prompt: torch.Tensor,
    continuation: torch.Tensor,
    greedy_prompt: torch.Tensor,
    greedy_tokens: int,
    *,
    optimized: bool,
) -> dict[str, Any]:
    route_mode(optimized)
    rows: dict[str, Any] = {"forward": {}, "routes": {"prefill": {}, "decode": []}}
    with torch.inference_mode():
        for name, (ids, mask) in inputs.items():
            output = model(
                input_ids=ids,
                attention_mask=mask,
                use_cache=True,
                logits_to_keep=0,
            )
            rows["forward"][name] = {
                "logits": output.logits.detach().cpu(),
                "cache": cache_snapshot(output.past_key_values),
            }
            rows["routes"]["prefill"][name] = last_model_route()
        cached, cache = manual_cached(model, prompt, continuation)
        rows["cached_logits"] = cached
        rows["cached_cache"] = cache_snapshot(cache)
        rows["routes"]["decode"].append(last_model_route())
        generated, greedy_routes = manual_greedy(model, greedy_prompt, greedy_tokens)
        rows["greedy"] = generated
        rows["routes"]["greedy"] = greedy_routes
    return rows


def collect_fla_lane(
    model,
    inputs: dict[str, tuple[torch.Tensor, torch.Tensor | None]],
    prompt: torch.Tensor,
    continuation: torch.Tensor,
    greedy_prompt: torch.Tensor,
    greedy_tokens: int,
) -> dict[str, Any]:
    rows: dict[str, Any] = {"forward": {}}
    with torch.inference_mode():
        for name, (ids, mask) in inputs.items():
            output = model(
                input_ids=ids,
                attention_mask=mask,
                use_cache=True,
                logits_to_keep=0,
            )
            rows["forward"][name] = {
                "logits": output.logits.detach().cpu(),
                "cache": cache_snapshot(output.past_key_values),
            }
        rows["cached_logits"], cached_cache = manual_cached(model, prompt, continuation)
        rows["cached_cache"] = cache_snapshot(cached_cache)
        rows["greedy"], _ = manual_greedy(model, greedy_prompt, greedy_tokens)
    return rows


def compare_lanes(
    candidate: dict[str, Any], reference: dict[str, Any], dtype: torch.dtype
) -> dict[str, Any]:
    forward = {}
    for name in reference["forward"]:
        logits = tensor_metric(
            candidate["forward"][name]["logits"],
            reference["forward"][name]["logits"],
        )
        states = compare_states(
            candidate["forward"][name]["cache"],
            reference["forward"][name]["cache"],
        )
        forward[name] = {
            "passed": metric_passed(logits, dtype, logits=True)
            and state_rows_passed(states, dtype),
            "logits": logits,
            "states": states,
        }
    cached_logits = tensor_metric(
        candidate["cached_logits"], reference["cached_logits"]
    )
    cached_states = compare_states(candidate["cached_cache"], reference["cached_cache"])
    greedy_equal = bool(torch.equal(candidate["greedy"], reference["greedy"]))
    passed = (
        all(row["passed"] for row in forward.values())
        and metric_passed(cached_logits, dtype, logits=True)
        and state_rows_passed(cached_states, dtype)
        and greedy_equal
    )
    return {
        "passed": passed,
        "forward": forward,
        "cached_decode": {
            "passed": metric_passed(cached_logits, dtype, logits=True)
            and state_rows_passed(cached_states, dtype),
            "logits": cached_logits,
            "states": cached_states,
        },
        "greedy": {
            "passed": greedy_equal,
            "candidate": candidate["greedy"].tolist(),
            "reference": reference["greedy"].tolist(),
        },
    }


def release_lane_tensors(lane: dict[str, Any]) -> None:
    for row in lane["forward"].values():
        row.pop("cache", None)
        row.pop("logits", None)
    lane.pop("cached_cache", None)
    lane.pop("cached_logits", None)
    lane.pop("greedy", None)


def run_inference_model(
    label: str,
    path: Path,
    dtype: torch.dtype,
    batches: tuple[int, ...],
    lengths: tuple[int, ...],
    decode_steps: int,
    greedy_tokens: int,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    model = clean_model(path, dtype)
    vocab = int(model.config.vocab_size)
    inputs: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = {
        f"b{batch}-t{length}": (
            torch.randint(
                1, vocab, (batch, length), device="cuda", generator=generator
            ),
            None,
        )
        for batch in batches
        for length in lengths
    }
    padding_ids = torch.randint(
        1, vocab, (2, max(17, min(lengths))), device="cuda", generator=generator
    )
    padding_mask = torch.ones_like(padding_ids, dtype=torch.bool)
    padding_mask[0, -3:] = False
    padding_mask[1, :4] = False
    inputs["mixed-left-right-padding"] = (padding_ids, padding_mask)
    prompt = torch.randint(1, vocab, (1, 17), device="cuda", generator=generator)
    continuation = torch.randint(
        1, vocab, (1, decode_steps), device="cuda", generator=generator
    )
    greedy_prompt = torch.randint(1, vocab, (1, 17), device="cuda", generator=generator)
    reference = collect_clean_lane(
        model,
        inputs,
        prompt,
        continuation,
        greedy_prompt,
        greedy_tokens,
        optimized=False,
    )
    optimized = collect_clean_lane(
        model,
        inputs,
        prompt,
        continuation,
        greedy_prompt,
        greedy_tokens,
        optimized=True,
    )
    optimized_routes_passed = all(
        route_is(route, "prefill") for route in optimized["routes"]["prefill"].values()
    ) and all(
        route_is(route, "decode")
        for route in (optimized["routes"]["decode"] + optimized["routes"]["greedy"][1:])
    )
    optimized_comparison = compare_lanes(optimized, reference, dtype)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    fla = fla_model(path, dtype)
    fla_rows = collect_fla_lane(
        fla, inputs, prompt, continuation, greedy_prompt, greedy_tokens
    )
    fla_comparison = compare_lanes(fla_rows, reference, dtype)
    del fla
    gc.collect()
    torch.cuda.empty_cache()

    release_lane_tensors(reference)
    release_lane_tensors(optimized)
    release_lane_tensors(fla_rows)
    return {
        "label": label,
        "model": model_fingerprint(path),
        "passed": bool(
            optimized_comparison["passed"]
            and optimized_routes_passed
            and fla_comparison["passed"]
        ),
        "optimized_vs_reference": optimized_comparison,
        "fla_vs_reference": fla_comparison,
        "optimized_routes_passed": optimized_routes_passed,
        "optimized_routes": optimized["routes"],
    }


def run_training_lane(kind: str, path: Path, ids: torch.Tensor, labels: torch.Tensor):
    dtype = torch.bfloat16
    if kind == "fla":
        model = fla_model(path, dtype, training=True)
    else:
        model = clean_model(path, dtype).train()
        route_mode(kind == "optimized")
    model.zero_grad(set_to_none=True)
    output = model(
        input_ids=ids,
        labels=labels,
        use_cache=False,
        logits_to_keep=0,
    )
    # Independently recompute the standard HF shifted loss so the comparison
    # cannot silently inherit a backend-specific auxiliary loss.
    shifted = F.cross_entropy(
        output.logits[:, :-1].float().reshape(-1, output.logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )
    shifted.backward()
    gradients = {
        name: parameter.grad.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    row = {
        "logits": output.logits.detach().cpu(),
        "loss": shifted.detach().cpu(),
        "gradients": gradients,
        "route": None if kind == "fla" else last_model_route(),
    }
    del output, model
    gc.collect()
    torch.cuda.empty_cache()
    return row


def run_training(path: Path, batch: int, tokens: int, seed: int) -> dict[str, Any]:
    if tokens % 16:
        raise ValueError("native training sequence length must be divisible by 16")
    probe = clean_model(path, torch.bfloat16)
    vocab = int(probe.config.vocab_size)
    del probe
    torch.cuda.empty_cache()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.randint(1, vocab, (batch, tokens), device="cuda", generator=generator)
    labels = ids.clone()
    labels[0, tokens // 2] = -100
    lanes = {
        name: run_training_lane(name, path, ids, labels)
        for name in ("reference", "optimized", "fla")
    }
    comparisons = {}
    for name in ("optimized", "fla"):
        logits = tensor_metric(lanes[name]["logits"], lanes["reference"]["logits"])
        loss = tensor_metric(lanes[name]["loss"], lanes["reference"]["loss"])
        gradients = gradient_metrics(
            lanes[name]["gradients"], lanes["reference"]["gradients"]
        )
        comparisons[name] = {
            "passed": metric_passed(logits, torch.bfloat16, logits=True)
            and loss["finite"]
            and loss["max_abs"] <= 0.02
            and gradient_rows_passed(gradients, torch.bfloat16),
            "logits": logits,
            "loss": loss,
            "gradients": gradients,
        }
    optimized_route_passed = route_is(lanes["optimized"]["route"], "training")
    for lane in lanes.values():
        lane.pop("logits")
        lane.pop("loss")
        lane.pop("gradients")
    return {
        "passed": bool(
            optimized_route_passed
            and comparisons["optimized"]["passed"]
            and comparisons["fla"]["passed"]
        ),
        "batch": batch,
        "tokens": tokens,
        "optimized_route_passed": optimized_route_passed,
        "routes": {name: row["route"] for name, row in lanes.items()},
        "comparisons": comparisons,
    }


def main() -> int:
    args = arguments()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    models = parse_models(args.model)
    dtype = dtype_from_name(args.dtype)
    fla = activate_fla_source(args.fla_source)
    batches = tuple(args.batch or (1, 4))
    lengths = tuple(args.tokens or (1, 17, 128))
    operator = run_operator_parity(
        dtype,
        batches,
        lengths,
        args.seed + 100_000,
    )
    inference = []
    for index, (label, path) in enumerate(models.items()):
        inference.append(
            run_inference_model(
                label,
                path,
                dtype,
                batches,
                lengths,
                args.decode_steps,
                args.greedy_tokens,
                args.seed + index * 1000,
            )
        )
    training_label = args.training_model or next(iter(models))
    if training_label not in models:
        raise ValueError(f"unknown --training-model label: {training_label}")
    if args.training_mode == "native":
        training = run_training(
            models[training_label],
            args.training_batch,
            args.training_tokens,
            args.seed + 50_000,
        )
        training = {
            "status": "passed" if training["passed"] else "failed",
            **training,
        }
    else:
        training = {
            "status": "not_applicable",
            "passed": True,
            "batch": args.training_batch,
            "tokens": args.training_tokens,
            "device_capability": tuple(torch.cuda.get_device_capability()),
            "reason": (
                "migrated train_temp is BF16 and requires sm80 or newer; "
                "operator input/state gradients remain covered above"
            ),
        }
    wheels = {}
    for name, path in (
        ("rwkv7_hf", args.hf_wheel),
        ("rwkv7_kernels", args.kernel_wheel),
    ):
        if path is not None:
            path = path.expanduser().resolve()
            wheels[name] = {"path": str(path), "sha256": sha256_file(path)}
    passed = (
        operator["passed"]
        and all(row["passed"] for row in inference)
        and training["passed"]
    )
    report = {
        "schema": "rwkv7-backend-v2-three-way-parity-v1",
        "status": "passed" if passed else "failed",
        "code_sha": args.code_sha or git_revision(Path(__file__).resolve().parents[1]),
        "dtype": args.dtype,
        "fla": fla,
        "environment": environment(),
        "wheels": wheels,
        "settings": {
            "batches": batches,
            "tokens": lengths,
            "decode_steps": args.decode_steps,
            "greedy_tokens": args.greedy_tokens,
            "seed": args.seed,
            "training_mode": args.training_mode,
        },
        "operator": operator,
        "inference": inference,
        "training": {"model": training_label, **training},
    }
    write_json(args.output, report)
    print(json.dumps({"output": str(args.output), "status": report["status"]}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
