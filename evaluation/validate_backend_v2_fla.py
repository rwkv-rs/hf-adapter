#!/usr/bin/env python3
"""Validate reference/optimized RWKV-7 against one pinned FLA checkout.

The script records three distinct lanes.  The optimized lane is accepted only
when inference names its actual whole-model implementation and training names
the readable model plus every selected tensor leaf; requesting an environment
selector is not route evidence.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

# Fix pinned FLA/TorchInductor compilation to one worker.  The default
# 24-process pool was observed to hit its 300-second atexit TimeoutExpired
# after the validation JSON had already been written.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import torch
import torch.nn.functional as F

from common import environment, git_revision, model_fingerprint, sha256_file
from fla_common import (
    activate_fla_source,
    annotate_metric,
    annotate_state_rows,
    compare_states,
    gradient_metrics,
    gradient_rows_passed,
    recurrent_states,
    state_rows_aspirational_passed,
    state_rows_release_passed,
    tensor_metric,
    write_json,
)
from training_metrics import (
    adaptive_fast_domain_expected,
    classify_candidate_and_fla_reference_results,
    full_model_reference_release_envelope,
    global_gradient_metric,
    gradient_parameter_summary,
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
        choices=("reference", "adaptive", "native", "skip-not-applicable"),
        default="adaptive",
        help=(
            "adaptive validates the formal optional-kernel training program "
            "with shape-local reference fallback; reference forces the clean "
            "PyTorch baseline; native is a deprecated alias; SM70 may use "
            "skip-not-applicable"
        ),
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
    # Operator and inference validation must not inherit a training selector
    # from the parent shell or an earlier training lane in this process.
    os.environ["RWKV7_TRAINING_KERNEL_IMPL"] = "auto"


def three_way_validation_status(
    *,
    candidate_reference_release_passed: bool,
    candidate_reference_aspirational_passed: bool,
    route_passed: bool,
    fla_reference_release_passed: bool,
    fla_reference_aspirational_passed: bool,
) -> dict[str, Any]:
    """Separate blocking candidate gates from stricter numeric diagnostics."""

    candidate_gate = {
        "role": "release-gate-blocking",
        "passed": bool(candidate_reference_release_passed),
    }
    candidate_diagnostic = {
        "role": "diagnostic-non-blocking",
        "passed": bool(candidate_reference_aspirational_passed),
    }
    route_gate = {
        "role": "release-gate-blocking",
        "passed": bool(route_passed),
    }
    fla_diagnostic = {
        "role": "diagnostic-non-blocking",
        # Preserve the historical strict-diagnostic meaning of ``passed``.
        # The two explicit keys are authoritative for new consumers.
        "passed": bool(fla_reference_aspirational_passed),
        "release_passed": bool(fla_reference_release_passed),
        "aspirational_passed": bool(fla_reference_aspirational_passed),
    }
    return {
        "passed": bool(candidate_gate["passed"] and route_gate["passed"]),
        "candidate_reference_release_gate": candidate_gate,
        "candidate_reference_aspirational_diagnostic": candidate_diagnostic,
        "route_release_gate": route_gate,
        "fla_reference_diagnostic": fla_diagnostic,
    }


def canonical_training_mode(value: str) -> str:
    return "adaptive" if value == "native" else value


def training_route_mode(optimized: bool, training_mode: str) -> None:
    os.environ["RWKV7_KERNEL_IMPL"] = "auto"
    os.environ["RWKV7_MODEL_KERNEL_IMPL"] = "auto"
    if optimized and training_mode == "adaptive":
        os.environ["RWKV7_BACKEND"] = "auto"
        os.environ["RWKV7_TRAINING_KERNEL_IMPL"] = "adaptive"
    elif optimized and training_mode == "reference":
        os.environ["RWKV7_BACKEND"] = "auto"
        os.environ["RWKV7_TRAINING_KERNEL_IMPL"] = "auto"
    else:
        os.environ["RWKV7_BACKEND"] = "reference"
        os.environ["RWKV7_TRAINING_KERNEL_IMPL"] = "auto"


def last_model_route() -> dict[str, Any] | None:
    from rwkv7_hf.ops_rwkv7 import get_last_model_route

    return get_last_model_route()


def last_training_routes() -> dict[str, Any]:
    from rwkv7_hf.ops_rwkv7 import (
        get_last_linear_route,
        get_last_mix6_route,
        get_last_model_route,
        get_last_recurrent_route,
        get_last_training_program_route,
    )

    return {
        "model": get_last_model_route(),
        "recurrent": get_last_recurrent_route(),
        "linear": get_last_linear_route(),
        "mix6": get_last_mix6_route(),
        "program": get_last_training_program_route(),
    }


def route_is(route: dict[str, Any] | None, phase: str) -> bool:
    if not route or route.get("selected") != "optimized":
        return False
    implementation = str(route.get("implementation", ""))
    expected = {
        "prefill": "native-nvidia-prefill-v2[",
        "decode": "native-nvidia-fused-decode-v2[",
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
                output_row = annotate_metric(
                    tensor_metric(output.cpu(), reference_output.cpu()),
                    dtype,
                    logits=False,
                )
                state_row = annotate_metric(
                    tensor_metric(state.cpu(), reference_state.cpu()),
                    dtype,
                    logits=False,
                )
                release_passed = bool(
                    output_row["release_passed"] and state_row["release_passed"]
                )
                aspirational_passed = bool(
                    output_row["aspirational_passed"]
                    and state_row["aspirational_passed"]
                )
                comparisons[name] = {
                    "passed": release_passed,
                    "release_passed": release_passed,
                    "aspirational_passed": aspirational_passed,
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
            status = three_way_validation_status(
                candidate_reference_release_passed=comparisons["optimized"][
                    "release_passed"
                ],
                candidate_reference_aspirational_passed=comparisons["optimized"][
                    "aspirational_passed"
                ],
                route_passed=recurrent_route_is_optimized(optimized_route),
                fla_reference_release_passed=(
                    comparisons["fla"]["release_passed"] and gradient_passed
                ),
                fla_reference_aspirational_passed=(
                    comparisons["fla"]["aspirational_passed"] and gradient_passed
                ),
            )
            rows.append(
                {
                    "case": f"b{batch}-t{length}",
                    **status,
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
    return {
        "passed": all(row["passed"] for row in rows),
        "candidate_reference_release_gate": {
            "role": "release-gate-blocking",
            "passed": all(
                row["candidate_reference_release_gate"]["passed"] for row in rows
            ),
        },
        "candidate_reference_aspirational_diagnostic": {
            "role": "diagnostic-non-blocking",
            "passed": all(
                row["candidate_reference_aspirational_diagnostic"]["passed"]
                for row in rows
            ),
        },
        "route_release_gate": {
            "role": "release-gate-blocking",
            "passed": all(row["route_release_gate"]["passed"] for row in rows),
        },
        "fla_reference_diagnostic": {
            "role": "diagnostic-non-blocking",
            "passed": all(
                row["fla_reference_diagnostic"]["aspirational_passed"]
                for row in rows
            ),
            "release_passed": all(
                row["fla_reference_diagnostic"]["release_passed"] for row in rows
            ),
            "aspirational_passed": all(
                row["fla_reference_diagnostic"]["aspirational_passed"]
                for row in rows
            ),
        },
        "cases": rows,
    }


def clean_model(path: Path, dtype: torch.dtype):
    from rwkv7_hf.modeling_rwkv7 import RWKV7ForCausalLM

    return RWKV7ForCausalLM.from_pretrained(path, torch_dtype=dtype).cuda().eval()


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
    model = RWKV7ForCausalLM.from_pretrained(
        path, config=config, torch_dtype=dtype
    ).cuda()
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
        logits = annotate_metric(
            tensor_metric(
                candidate["forward"][name]["logits"],
                reference["forward"][name]["logits"],
            ),
            dtype,
            logits=True,
        )
        states = annotate_state_rows(
            compare_states(
                candidate["forward"][name]["cache"],
                reference["forward"][name]["cache"],
            ),
            dtype,
        )
        release_passed = bool(
            logits["release_passed"]
            and state_rows_release_passed(states, dtype)
        )
        aspirational_passed = bool(
            logits["aspirational_passed"]
            and state_rows_aspirational_passed(states, dtype)
        )
        forward[name] = {
            "passed": release_passed,
            "release_passed": release_passed,
            "aspirational_passed": aspirational_passed,
            "logits": logits,
            "states": states,
        }
    cached_logits = annotate_metric(
        tensor_metric(candidate["cached_logits"], reference["cached_logits"]),
        dtype,
        logits=True,
    )
    cached_states = annotate_state_rows(
        compare_states(candidate["cached_cache"], reference["cached_cache"]),
        dtype,
    )
    greedy_equal = bool(torch.equal(candidate["greedy"], reference["greedy"]))
    cached_release_passed = bool(
        cached_logits["release_passed"]
        and state_rows_release_passed(cached_states, dtype)
    )
    cached_aspirational_passed = bool(
        cached_logits["aspirational_passed"]
        and state_rows_aspirational_passed(cached_states, dtype)
    )
    release_passed = (
        all(row["release_passed"] for row in forward.values())
        and cached_release_passed
        and greedy_equal
    )
    aspirational_passed = (
        all(row["aspirational_passed"] for row in forward.values())
        and cached_aspirational_passed
        and greedy_equal
    )
    return {
        "passed": release_passed,
        "release_passed": release_passed,
        "aspirational_passed": aspirational_passed,
        "forward": forward,
        "cached_decode": {
            "passed": cached_release_passed,
            "release_passed": cached_release_passed,
            "aspirational_passed": cached_aspirational_passed,
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
    status = three_way_validation_status(
        candidate_reference_release_passed=optimized_comparison[
            "release_passed"
        ],
        candidate_reference_aspirational_passed=optimized_comparison[
            "aspirational_passed"
        ],
        route_passed=optimized_routes_passed,
        fla_reference_release_passed=fla_comparison["release_passed"],
        fla_reference_aspirational_passed=fla_comparison[
            "aspirational_passed"
        ],
    )
    return {
        "label": label,
        "model": model_fingerprint(path),
        **status,
        "optimized_vs_reference": optimized_comparison,
        "fla_vs_reference": fla_comparison,
        "optimized_routes_passed": optimized_routes_passed,
        "optimized_routes": optimized["routes"],
    }


def run_training_lane(
    kind: str,
    path: Path,
    ids: torch.Tensor,
    labels: torch.Tensor,
    training_mode: str,
):
    dtype = torch.bfloat16
    if kind == "fla":
        model = fla_model(path, dtype, training=True)
    else:
        model = clean_model(path, dtype).train()
        training_route_mode(kind == "optimized", training_mode)
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
        "route": None if kind == "fla" else last_training_routes(),
    }
    del output, model
    gc.collect()
    torch.cuda.empty_cache()
    return row


def compare_full_model_training_lane(
    candidate: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    """Compare one lane with the shared full-model reference envelope."""

    logits = tensor_metric(candidate["logits"], reference["logits"])
    loss = tensor_metric(candidate["loss"], reference["loss"])
    gradients = gradient_metrics(candidate["gradients"], reference["gradients"])
    global_gradient = global_gradient_metric(
        candidate["gradients"], reference["gradients"]
    )
    reference_release_envelope = full_model_reference_release_envelope(
        {
            "logits": logits,
            "loss": loss,
            "global_gradient": global_gradient,
        }
    )
    return {
        "passed": reference_release_envelope["passed"],
        "release_passed": reference_release_envelope["passed"],
        # Full-model training uses BF16 and has no separate FP16-logit
        # max-absolute target, so this lane's aspirational outcome is the same
        # fixed reference envelope.
        "aspirational_passed": reference_release_envelope["passed"],
        "reference_release_envelope": reference_release_envelope,
        "strict_named_parameter_diagnostic_passed": gradient_rows_passed(
            gradients, torch.bfloat16
        ),
        "logits": logits,
        "loss": loss,
        "gradients": gradients,
        "global_gradient": global_gradient,
        "gradient_parameter_summary": gradient_parameter_summary(gradients),
    }


def run_training(
    path: Path,
    batch: int,
    tokens: int,
    seed: int,
    training_mode: str,
) -> dict[str, Any]:
    probe = clean_model(path, torch.bfloat16)
    vocab = int(probe.config.vocab_size)
    del probe
    torch.cuda.empty_cache()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.randint(1, vocab, (batch, tokens), device="cuda", generator=generator)
    labels = ids.clone()
    labels[0, tokens // 2] = -100
    lanes = {
        name: run_training_lane(name, path, ids, labels, training_mode)
        for name in ("reference", "optimized", "fla")
    }
    comparisons = {
        name: compare_full_model_training_lane(lanes[name], lanes["reference"])
        for name in ("optimized", "fla")
    }
    routes = lanes["optimized"]["route"] or {}
    model_route = routes.get("model") or {}
    recurrent_route = routes.get("recurrent") or {}
    linear_route = routes.get("linear") or {}
    mix6_route = routes.get("mix6") or {}
    program_route = routes.get("program") or {}
    if training_mode == "reference":
        optimized_route_passed = bool(
            model_route.get("selected") == "reference"
            and model_route.get("phase") == "training"
            and model_route.get("implementation") == "torch-reference-model-v1"
            and recurrent_route.get("selected") == "reference"
            and recurrent_route.get("implementation") == "torch-reference-v1"
            and linear_route.get("selected") == "reference"
            and linear_route.get("implementation") == "torch-reference-linear-v1"
            and mix6_route.get("selected") == "reference"
            and mix6_route.get("implementation") == "torch-reference-mix6-v1"
            and program_route.get("selected") == "reference"
            and program_route.get("implementation")
            == "torch-reference-training-program-v1"
        )
    else:
        fast_domain = adaptive_fast_domain_expected(batch=batch, tokens=tokens)
        recurrent_implementation = (
            "native-nvidia-rwkv7-factorized-recurrent-training-v1"
            if fast_domain
            else "torch-cuda-rwkv7-batched-matrix-recurrent-training-v1"
        )
        linear_route_passed = (
            linear_route.get("selected") == "optimized"
            and linear_route.get("implementation")
            == "torch-cuda-rwkv7-flattened-linear-training-v1"
            if fast_domain
            else linear_route.get("selected") == "reference"
            and linear_route.get("implementation") == "torch-reference-linear-v1"
        )
        optimized_route_passed = bool(
            model_route.get("selected") == "reference"
            and model_route.get("phase") == "training"
            and model_route.get("implementation") == "torch-reference-model-v1"
            and recurrent_route.get("selected") == "optimized"
            and recurrent_route.get("implementation") == recurrent_implementation
            and linear_route_passed
            and mix6_route.get("selected") == "optimized"
            and mix6_route.get("implementation")
            == "native-nvidia-rwkv7-mix6-training-v1"
            and program_route.get("selected")
            == ("optimized" if fast_domain else "reference")
            and program_route.get("implementation")
            == "native-nvidia-rwkv7-adaptive-training-program-v1"
        )
    for lane in lanes.values():
        lane.pop("logits")
        lane.pop("loss")
        lane.pop("gradients")
    numerical_roles = classify_candidate_and_fla_reference_results(
        comparisons["optimized"], comparisons["fla"]
    )
    candidate_reference_release_gate = numerical_roles[
        "candidate_reference_release_gate"
    ]
    fla_reference_diagnostic = {
        **numerical_roles["fla_reference_diagnostic"],
        "release_passed": numerical_roles["fla_reference_diagnostic"]["passed"],
        "aspirational_passed": comparisons["fla"]["aspirational_passed"],
    }
    return {
        "passed": bool(
            optimized_route_passed and candidate_reference_release_gate["passed"]
        ),
        "batch": batch,
        "tokens": tokens,
        "optimized_route_passed": optimized_route_passed,
        "route_release_gate": {
            "role": "release-gate-blocking",
            "passed": optimized_route_passed,
        },
        "candidate_reference_release_gate": candidate_reference_release_gate,
        "candidate_reference_aspirational_diagnostic": {
            "role": "diagnostic-non-blocking",
            "passed": comparisons["optimized"]["aspirational_passed"],
        },
        "fla_reference_diagnostic": fla_reference_diagnostic,
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
    training_mode = canonical_training_mode(args.training_mode)
    if training_mode in {"reference", "adaptive"}:
        training = run_training(
            models[training_label],
            args.training_batch,
            args.training_tokens,
            args.seed + 50_000,
            training_mode,
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
                "native BF16 training leaves require sm80 or newer; the "
                "readable HF training loop remains available"
            ),
            "candidate_reference_release_gate": {
                "role": "release-gate-blocking",
                "passed": True,
                "status": "not_applicable",
            },
            "candidate_reference_aspirational_diagnostic": {
                "role": "diagnostic-non-blocking",
                "passed": True,
                "status": "not_applicable",
            },
            "route_release_gate": {
                "role": "release-gate-blocking",
                "passed": True,
                "status": "not_applicable",
            },
            "fla_reference_diagnostic": {
                "role": "diagnostic-non-blocking",
                "passed": True,
                "release_passed": True,
                "aspirational_passed": True,
                "status": "not_applicable",
            },
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
    candidate_aspirational_passed = bool(
        operator["candidate_reference_aspirational_diagnostic"]["passed"]
        and all(
            row["candidate_reference_aspirational_diagnostic"]["passed"]
            for row in inference
        )
        and (
            training.get("candidate_reference_aspirational_diagnostic", {}).get(
                "passed", True
            )
            if training_mode in {"reference", "adaptive"}
            else True
        )
    )
    fla_release_diagnostics_passed = bool(
        operator["fla_reference_diagnostic"]["release_passed"]
        and all(
            row["fla_reference_diagnostic"]["release_passed"]
            for row in inference
        )
        and (
            training.get("fla_reference_diagnostic", {}).get(
                "release_passed", True
            )
            if training_mode in {"reference", "adaptive"}
            else True
        )
    )
    fla_aspirational_diagnostics_passed = bool(
        operator["fla_reference_diagnostic"]["aspirational_passed"]
        and all(
            row["fla_reference_diagnostic"]["aspirational_passed"]
            for row in inference
        )
        and (
            training.get("fla_reference_diagnostic", {}).get(
                "aspirational_passed", True
            )
            if training_mode in {"reference", "adaptive"}
            else True
        )
    )
    report = {
        "schema": "rwkv7-backend-v2-three-way-validation-v3",
        "status": "passed" if passed else "failed",
        "release_gates": {
            "role": "blocking",
            "passed": passed,
            "operator": operator["passed"],
            "inference": all(row["passed"] for row in inference),
            "training": training["passed"],
        },
        "candidate_aspirational_diagnostics": {
            "role": "diagnostic-non-blocking",
            "passed": candidate_aspirational_passed,
        },
        "fla_diagnostics": {
            "role": "diagnostic-non-blocking",
            "complete": True,
            "passed_release_envelope": fla_release_diagnostics_passed,
            "passed_strict_envelope": fla_aspirational_diagnostics_passed,
        },
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
            "training_mode": training_mode,
            "requested_training_mode": args.training_mode,
        },
        "numeric_envelopes": {
            "release": {
                "fp32": "rtol=1e-4,atol=1e-5",
                "fp16_cosine_min": 0.9999,
                "bf16_cosine_min": 0.999,
                "finite_required": True,
                "fp16_logits_max_abs": "reported-not-blocking",
            },
            "aspirational": {
                "fp32": "rtol=1e-4,atol=1e-5",
                "low_precision_cosine_min": 0.9999,
                "fp16_logits_max_abs": 0.15,
            },
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
