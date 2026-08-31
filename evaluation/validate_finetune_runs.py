#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


EXPECTED_DATASETS = {
    "sft": (
        "HuggingFaceH4/ultrachat_200k",
        "8049631c405ae6576f93f445c6b8166f76f5505a",
    ),
    "dpo": (
        "HuggingFaceH4/ultrafeedback_binarized",
        "3949bf5f8c17c394422ccfab0c31ea9c20bdeb85",
    ),
    "grpo": (
        "openai/gsm8k",
        "740312add88f781978c0658806c59bc2815b9866",
    ),
}
EXPECTED_TARGETS = ["r_proj", "k_proj", "v_proj", "o_proj", "key", "value"]
READABLE_MODEL_IMPLEMENTATION = "torch-reference-model-v1"
MATRIX_RECURRENT_IMPLEMENTATION = (
    "torch-cuda-rwkv7-batched-matrix-recurrent-training-v1"
)
FACTORIZED_RECURRENT_IMPLEMENTATION = (
    "native-nvidia-rwkv7-factorized-recurrent-training-v1"
)
FLATTENED_LINEAR_IMPLEMENTATION = "torch-cuda-rwkv7-flattened-linear-training-v1"
MIX6_IMPLEMENTATION = "native-nvidia-rwkv7-mix6-training-v1"
ADAPTIVE_TRAINING_PROGRAM_IMPLEMENTATION = (
    "native-nvidia-rwkv7-adaptive-training-program-v1"
)
HISTORICAL_WHOLE_MODEL_IMPLEMENTATION = "native-nvidia-official-training-autograd-v2"

REFERENCE_RECURRENT_IMPLEMENTATION = "torch-reference-v1"
REFERENCE_LINEAR_IMPLEMENTATION = "torch-reference-linear-v1"
REFERENCE_MIX6_IMPLEMENTATION = "torch-reference-mix6-v1"
REFERENCE_PROGRAM_IMPLEMENTATION = "torch-reference-training-program-v1"


def _positive_trace_implementations(
    kernel_trace: dict, key: str, failures: list[str]
) -> set[str]:
    """Return executed implementations from one validated trace counter."""

    counter = kernel_trace.get(key)
    if not isinstance(counter, dict):
        failures.append(f"kernel trace {key} is not an object")
        return set()
    implementations: set[str] = set()
    for implementation, count in counter.items():
        if (
            not isinstance(implementation, str)
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
        ):
            failures.append(f"kernel trace {key} contains an invalid counter")
            continue
        if count:
            implementations.add(implementation)
    return implementations


def validate_reference_finetune_route_evidence(
    routes: list[dict], kernel_trace: dict
) -> dict:
    """Require one complete readable HF training program.

    Optional inference may occur during adapter save/reload in the same
    process, so the process-wide trace is used only to reject known optimized
    *training* leaves. Per-optimizer-step boundary records prove the actual
    model/program/recurrent/linear/Mix6 training route.
    """

    failures: list[str] = []
    if not isinstance(routes, list) or any(not isinstance(row, dict) for row in routes):
        return {
            "passed": False,
            "failures": ["backend routes are not a list of objects"],
            "observed": {},
        }
    if not isinstance(kernel_trace, dict):
        return {
            "passed": False,
            "failures": ["kernel route trace is not an object"],
            "observed": {},
        }
    if kernel_trace.get("schema") != "rwkv7-kernel-route-trace-v2":
        failures.append("missing versioned process-wide kernel route trace")
    if kernel_trace.get("requested_training_policy") != "auto":
        failures.append("kernel trace training policy does not match reference")

    expected = {
        "model": READABLE_MODEL_IMPLEMENTATION,
        "program": REFERENCE_PROGRAM_IMPLEMENTATION,
        "recurrent": REFERENCE_RECURRENT_IMPLEMENTATION,
        "linear": REFERENCE_LINEAR_IMPLEMENTATION,
        "mix6": REFERENCE_MIX6_IMPLEMENTATION,
    }
    optimizer_routes = [
        row
        for row in routes
        if row.get("event") == "pre_optimizer_step" and row.get("boundary") in expected
    ]
    observed: dict[str, list[str]] = {}
    for boundary, implementation in expected.items():
        rows = [row for row in optimizer_routes if row.get("boundary") == boundary]
        observed[boundary] = sorted(
            {str(row.get("implementation")) for row in rows}
        )
        if not rows:
            failures.append(f"training did not record the reference {boundary} route")
        if any(
            row.get("selected") != "reference"
            or (boundary != "program" and row.get("phase") != "training")
            or row.get("implementation") != implementation
            for row in rows
        ):
            failures.append(f"training recorded a non-reference {boundary} route")

    for key in (
        "actual_recurrent_calls",
        "actual_linear_calls",
        "actual_mix6_calls",
    ):
        actual = _positive_trace_implementations(kernel_trace, key, failures)
        if actual:
            failures.append(
                f"formal reference training executed optional diagnostic routes: "
                f"{sorted(actual)}"
            )

    actual_model = _positive_trace_implementations(
        kernel_trace, "actual_model_calls", failures
    )
    if HISTORICAL_WHOLE_MODEL_IMPLEMENTATION in actual_model or any(
        row.get("implementation") == HISTORICAL_WHOLE_MODEL_IMPLEMENTATION
        for row in routes
    ):
        failures.append("whole-model diagnostic training unexpectedly executed")
    return {
        "passed": not failures,
        "failures": failures,
        "observed": observed,
    }


def validate_adaptive_finetune_route_evidence(
    routes: list[dict], kernel_trace: dict
) -> dict:
    """Validate legal adaptive routes without inventing a fast-path hit.

    Canonical SFT/DPO/GRPO batches are chosen by their trainers, not by the
    optional kernel.  They may therefore remain on the exact/reference program
    when their shape, padding, state, or autograd provenance is outside the
    certified B4/T128 fast domain.  This integration gate accepts every legal
    adaptive leaf or fallback observed during the run.  The separate formal
    training report, not a canonical finetune, proves the complete B4/T128
    atomic fast program.
    """

    failures: list[str] = []
    if not isinstance(routes, list) or any(not isinstance(row, dict) for row in routes):
        return {
            "passed": False,
            "failures": ["backend routes are not a list of objects"],
            "fast_program_observed": False,
            "fast_program_route_observed": False,
            "fast_program_inferred_from_complete_leaf_trace": False,
            "fallback_program_observed": False,
            "observed": {"recurrent": [], "linear": [], "mix6": []},
        }
    if not isinstance(kernel_trace, dict):
        return {
            "passed": False,
            "failures": ["kernel route trace is not an object"],
            "fast_program_observed": False,
            "fast_program_route_observed": False,
            "fast_program_inferred_from_complete_leaf_trace": False,
            "fallback_program_observed": False,
            "observed": {"recurrent": [], "linear": [], "mix6": []},
        }
    if kernel_trace.get("schema") != "rwkv7-kernel-route-trace-v2":
        failures.append("missing versioned process-wide kernel route trace")
    if kernel_trace.get("requested_training_policy") != "adaptive":
        failures.append("kernel trace training policy does not match adaptive")

    optimizer_routes = [
        row
        for row in routes
        if row.get("event") == "pre_optimizer_step"
        and row.get("boundary") in {"model", "program", "recurrent", "linear", "mix6"}
    ]
    readable_model = any(
        row.get("boundary") == "model"
        and row.get("selected") == "reference"
        and row.get("phase") == "training"
        and row.get("implementation") == READABLE_MODEL_IMPLEMENTATION
        for row in optimizer_routes
    )
    if not readable_model:
        failures.append("training did not retain the readable HF layer loop")
    invalid_model_routes = [
        row
        for row in optimizer_routes
        if row.get("boundary") == "model"
        and not (
            row.get("selected") == "reference"
            and row.get("phase") == "training"
            and row.get("implementation") == READABLE_MODEL_IMPLEMENTATION
        )
    ]
    if invalid_model_routes:
        failures.append("training recorded an invalid model route")

    program_routes = [
        row for row in optimizer_routes if row.get("boundary") == "program"
    ]
    invalid_program_routes = [
        row
        for row in program_routes
        if row.get("implementation") != ADAPTIVE_TRAINING_PROGRAM_IMPLEMENTATION
        or row.get("selected") not in {"optimized", "reference"}
    ]
    if not program_routes:
        failures.append("training did not record the adaptive program decision")
    if invalid_program_routes:
        failures.append("training recorded an invalid adaptive program route")
    explicit_fast_program = any(
        row.get("selected") == "optimized" for row in program_routes
    )
    fallback_program = any(row.get("selected") == "reference" for row in program_routes)

    recurrent_routes = [
        row for row in optimizer_routes if row.get("boundary") == "recurrent"
    ]
    linear_routes = [row for row in optimizer_routes if row.get("boundary") == "linear"]
    mix6_routes = [row for row in optimizer_routes if row.get("boundary") == "mix6"]

    allowed_routes = {
        "recurrent": {
            ("optimized", MATRIX_RECURRENT_IMPLEMENTATION),
            ("optimized", FACTORIZED_RECURRENT_IMPLEMENTATION),
            ("reference", REFERENCE_RECURRENT_IMPLEMENTATION),
        },
        "linear": {
            ("optimized", FLATTENED_LINEAR_IMPLEMENTATION),
            ("reference", REFERENCE_LINEAR_IMPLEMENTATION),
        },
        "mix6": {
            ("optimized", MIX6_IMPLEMENTATION),
            ("reference", REFERENCE_MIX6_IMPLEMENTATION),
        },
    }
    for boundary, rows in (
        ("recurrent", recurrent_routes),
        ("linear", linear_routes),
        ("mix6", mix6_routes),
    ):
        if not rows:
            failures.append(f"training did not record a {boundary} route")
        if any(
            (row.get("selected"), row.get("implementation"))
            not in allowed_routes[boundary]
            for row in rows
        ):
            failures.append(f"training recorded an invalid {boundary} route")

    actual_model = _positive_trace_implementations(
        kernel_trace, "actual_model_calls", failures
    )
    actual_recurrent = _positive_trace_implementations(
        kernel_trace, "actual_recurrent_calls", failures
    )
    actual_linear = _positive_trace_implementations(
        kernel_trace, "actual_linear_calls", failures
    )
    actual_mix6 = _positive_trace_implementations(
        kernel_trace, "actual_mix6_calls", failures
    )
    route_recurrent = {str(row.get("implementation")) for row in recurrent_routes}
    route_linear = {str(row.get("implementation")) for row in linear_routes}
    route_mix6 = {str(row.get("implementation")) for row in mix6_routes}
    observed_recurrent = route_recurrent | actual_recurrent
    observed_linear = route_linear | actual_linear
    observed_mix6 = route_mix6 | actual_mix6

    historical_route = any(
        row.get("implementation") == HISTORICAL_WHOLE_MODEL_IMPLEMENTATION
        for row in routes
    )
    if HISTORICAL_WHOLE_MODEL_IMPLEMENTATION in actual_model or historical_route:
        failures.append("whole-model diagnostic training unexpectedly executed")

    allowed_implementations = {
        "recurrent": {
            MATRIX_RECURRENT_IMPLEMENTATION,
            FACTORIZED_RECURRENT_IMPLEMENTATION,
            REFERENCE_RECURRENT_IMPLEMENTATION,
        },
        "linear": {
            FLATTENED_LINEAR_IMPLEMENTATION,
            REFERENCE_LINEAR_IMPLEMENTATION,
        },
        "mix6": {MIX6_IMPLEMENTATION, REFERENCE_MIX6_IMPLEMENTATION},
    }
    for boundary, observed in (
        ("recurrent", observed_recurrent),
        ("linear", observed_linear),
        ("mix6", observed_mix6),
    ):
        if not observed:
            failures.append(f"adaptive training did not execute a {boundary} boundary")
        unexpected = observed - allowed_implementations[boundary]
        if unexpected:
            failures.append(
                f"adaptive training executed unknown {boundary} implementations: "
                f"{sorted(unexpected)}"
            )

    # The adaptive program is atomic.  A recorded fast decision is valid only
    # when all three coupled optimized leaves were observed somewhere in the
    # run.  Conversely, a run that observed only fallback decisions must not
    # contain either fast-only recurrent or linear execution.  Finetune runs
    # may legitimately contain both decisions across different batches, so
    # validate each observed decision rather than forcing one global mode.
    fast_leaf_implementations = {
        FACTORIZED_RECURRENT_IMPLEMENTATION,
        FLATTENED_LINEAR_IMPLEMENTATION,
        MIX6_IMPLEMENTATION,
    }
    observed_fast_leaf_implementations = {
        implementation
        for implementation, observed in (
            (FACTORIZED_RECURRENT_IMPLEMENTATION, observed_recurrent),
            (FLATTENED_LINEAR_IMPLEMENTATION, observed_linear),
            (MIX6_IMPLEMENTATION, observed_mix6),
        )
        if implementation in observed
    }
    complete_fast_leaf_trace = (
        observed_fast_leaf_implementations == fast_leaf_implementations
    )
    fast_only_leaf_observed = bool(
        observed_fast_leaf_implementations
        & {FACTORIZED_RECURRENT_IMPLEMENTATION, FLATTENED_LINEAR_IMPLEMENTATION}
    )
    partial_fast_leaf_trace = fast_only_leaf_observed and not complete_fast_leaf_trace
    # Preference trainers can run several forwards before one optimizer
    # callback.  The callback sees the final ContextVar decision, while the
    # process-wide leaf counters retain every executed fast boundary.  A
    # complete three-leaf trace is therefore valid evidence that an atomic
    # fast program ran; a partial trace is never accepted as an inference.
    inferred_fast_program = complete_fast_leaf_trace and not explicit_fast_program
    fast_program_observed = explicit_fast_program or inferred_fast_program

    if explicit_fast_program:
        required_fast = (
            (
                FACTORIZED_RECURRENT_IMPLEMENTATION,
                observed_recurrent,
                "factorized recurrent",
            ),
            (FLATTENED_LINEAR_IMPLEMENTATION, observed_linear, "flattened linear"),
            (MIX6_IMPLEMENTATION, observed_mix6, "Mix6"),
        )
        for implementation, observed, label in required_fast:
            if implementation not in observed:
                failures.append(
                    f"optimized adaptive program did not execute the {label} boundary"
                )
    elif partial_fast_leaf_trace:
        failures.append(
            "fast training leaves were only partially observed without an "
            "optimized adaptive program decision"
        )
    if fallback_program:
        if not observed_recurrent.intersection(
            {MATRIX_RECURRENT_IMPLEMENTATION, REFERENCE_RECURRENT_IMPLEMENTATION}
        ):
            failures.append(
                "adaptive fallback did not execute an exact/reference recurrence"
            )
        if REFERENCE_LINEAR_IMPLEMENTATION not in observed_linear:
            failures.append("adaptive fallback did not retain reference linears")
    if not fast_program_observed:
        if FACTORIZED_RECURRENT_IMPLEMENTATION in observed_recurrent:
            failures.append(
                "factorized recurrent executed without an optimized adaptive program"
            )
        if FLATTENED_LINEAR_IMPLEMENTATION in observed_linear:
            failures.append(
                "flattened linear executed without an optimized adaptive program"
            )

    return {
        "passed": not failures,
        "failures": failures,
        "fast_program_observed": fast_program_observed,
        "fast_program_route_observed": explicit_fast_program,
        "fast_program_inferred_from_complete_leaf_trace": inferred_fast_program,
        "fallback_program_observed": fallback_program,
        "observed": {
            "recurrent": sorted(observed_recurrent),
            "linear": sorted(observed_linear),
            "mix6": sorted(observed_mix6),
        },
    }


def read(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_run(
    path: Path,
    name: str,
    expected_step: int,
    *,
    require_backend_v2_routes: bool,
    require_training_candidate: str | None,
) -> dict:
    exit_status = read(path / "exit_status.json")
    checks = read(path / "training_checks.json")
    reload = read(path / "adapter_reload.json")
    inventory = read(path / "checkpoint_inventory.json")
    changed = read(path / "changed_parameters.json")
    config = read(path / "resolved_config.json")
    environment = read(path / "environment.json")
    model = read(path / "model_provenance.json")
    fingerprints = read(path / "dataset_fingerprints.json")
    artifacts = read(path / "artifact_provenance.json")
    routes = read(path / "backend_routes.json")
    kernel_trace_path = path / "kernel_route_trace.json"
    kernel_trace = read(kernel_trace_path) if kernel_trace_path.is_file() else {}
    metrics_path = path / "metrics.jsonl"
    metrics = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    numeric = [
        float(value)
        for row in metrics
        for value in row.values()
        if isinstance(value, (int, float))
    ]
    failures = []
    if exit_status.get("returncode") != 0:
        failures.append("nonzero exit")
    if not checks.get("finite_loss"):
        failures.append("no finite loss")
    if not checks.get("nonzero_gradient"):
        failures.append("no nonzero gradient")
    if int(checks.get("global_step", -1)) != expected_step:
        failures.append(f"global_step != {expected_step}")
    if not reload.get("close"):
        failures.append("adapter reload mismatch")
    if not changed:
        failures.append("no changed parameters")
    if not inventory:
        failures.append("empty checkpoint inventory")
    if not metrics or any(not math.isfinite(value) for value in numeric):
        failures.append("missing or non-finite metrics")
    expected_dataset, expected_revision = EXPECTED_DATASETS[name]
    expected_config = {
        "seed": 42,
        "max_length": 512,
        "train_samples": 1024,
        "eval_samples": 128,
        "max_steps": 100,
        "gradient_accumulation_steps": 1,
        "report_to": "none",
        "dataset_name": expected_dataset,
        "dataset_revision": expected_revision,
        "target_modules": EXPECTED_TARGETS,
    }
    if name == "grpo":
        expected_config["max_completion_length"] = 64
    adaptive_route_evidence = None
    reference_route_evidence = None
    if require_training_candidate == "adaptive":
        adaptive_route_evidence = validate_adaptive_finetune_route_evidence(
            routes, kernel_trace
        )
        failures.extend(adaptive_route_evidence["failures"])
        if not checks.get("readable_model_loop"):
            failures.append("training checks did not retain the readable HF layer loop")
        if checks.get("adaptive_fast_program") is not adaptive_route_evidence.get(
            "fast_program_route_observed"
        ):
            failures.append("training checks disagree with the adaptive fast decision")
        if checks.get("adaptive_program_fallback") is not adaptive_route_evidence.get(
            "fallback_program_observed"
        ):
            failures.append(
                "training checks disagree with the adaptive fallback decision"
            )
        if checks.get("historical_whole_model_diagnostic") is not False:
            failures.append("historical whole-model diagnostic route is not false")
    elif require_training_candidate == "reference":
        reference_route_evidence = validate_reference_finetune_route_evidence(
            routes, kernel_trace
        )
        failures.extend(reference_route_evidence["failures"])
        if not checks.get("readable_model_loop"):
            failures.append("training checks did not retain the readable HF layer loop")
        if checks.get("adaptive_fast_program") is not False:
            failures.append("reference training claimed an adaptive fast program")
        if checks.get("historical_whole_model_diagnostic") is not False:
            failures.append("historical whole-model diagnostic route is not false")
    if require_training_candidate is not None:
        expected_config.update(
            {
                "torch_dtype": "bfloat16",
                "lora_dtype": "model",
            }
        )
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected_config.items()
        if config.get(key) != value
    }
    if mismatches:
        failures.append(f"non-canonical resolved config: {mismatches}")
    if (
        environment.get("transformers") != "4.56.2"
        or environment.get("trl") != "0.20.0"
    ):
        failures.append("unexpected canonical Transformers/TRL environment")
    if not config.get("source_revision") or not config.get("code_sha"):
        failures.append("missing source revision")
    if not model.get("resolved_revision") or not model.get("files", {}).get(
        "model.safetensors"
    ):
        failures.append("missing model/weight provenance")
    if not all(
        fingerprints.get(split, {}).get("selected") for split in ("train", "eval")
    ):
        failures.append("missing deterministic dataset fingerprints")
    if require_backend_v2_routes:
        if not artifacts.get("rwkv7_hf", {}).get("sha256"):
            failures.append("missing rwkv7-hf wheel SHA256")
        if not artifacts.get("rwkv7_kernels", {}).get("sha256"):
            failures.append("missing rwkv7-kernels wheel SHA256")
        readable_model = any(
            row.get("event") == "pre_optimizer_step"
            and row.get("boundary") == "model"
            and row.get("selected") == "reference"
            and row.get("phase") == "training"
            and row.get("implementation") == READABLE_MODEL_IMPLEMENTATION
            for row in routes
        )
        if not readable_model or not checks.get("readable_model_loop"):
            failures.append("training did not prove the readable HF model loop")
        if checks.get("historical_whole_model_diagnostic") is not False:
            failures.append("historical whole-model diagnostic route is not false")
    if (
        require_training_candidate in {"matrix", "factorized"}
    ):
        if kernel_trace.get("schema") != "rwkv7-kernel-route-trace-v2":
            failures.append("missing versioned process-wide kernel route trace")
        if kernel_trace.get("requested_training_policy") != require_training_candidate:
            failures.append("kernel trace training policy does not match the request")
        model_reference = any(
            row.get("event") == "pre_optimizer_step"
            and row.get("boundary") == "model"
            and row.get("selected") == "reference"
            and row.get("phase") == "training"
            and row.get("implementation") == READABLE_MODEL_IMPLEMENTATION
            for row in routes
        )
        recurrent_implementations = {
            "matrix": MATRIX_RECURRENT_IMPLEMENTATION,
            "factorized": FACTORIZED_RECURRENT_IMPLEMENTATION,
        }
        expected_recurrents = (
            set(recurrent_implementations.values())
            if require_training_candidate == "adaptive"
            else {recurrent_implementations[require_training_candidate]}
        )
        observed_recurrents = {
            str(row.get("implementation"))
            for row in routes
            if row.get("event") == "pre_optimizer_step"
            and row.get("boundary") == "recurrent"
            and row.get("selected") == "optimized"
            and row.get("implementation") in expected_recurrents
        }
        traced_recurrents = {
            str(implementation)
            for implementation, count in kernel_trace.get(
                "actual_recurrent_calls", {}
            ).items()
            if int(count) > 0 and implementation in expected_recurrents
        }
        observed_recurrents.update(traced_recurrents)
        recurrent_candidate = bool(observed_recurrents)
        matrix_recurrent = (
            any(
                row.get("event") == "pre_optimizer_step"
                and row.get("boundary") == "recurrent"
                and row.get("selected") == "optimized"
                and row.get("implementation") == recurrent_implementations["matrix"]
                for row in routes
            )
            or int(
                kernel_trace.get("actual_recurrent_calls", {}).get(
                    recurrent_implementations["matrix"], 0
                )
            )
            > 0
        )
        factorized_recurrent = (
            any(
                row.get("event") == "pre_optimizer_step"
                and row.get("boundary") == "recurrent"
                and row.get("selected") == "optimized"
                and row.get("implementation") == recurrent_implementations["factorized"]
                for row in routes
            )
            or int(
                kernel_trace.get("actual_recurrent_calls", {}).get(
                    recurrent_implementations["factorized"], 0
                )
            )
            > 0
        )
        if not model_reference or not checks.get("readable_model_loop"):
            failures.append("training did not retain the readable HF layer loop")
        recurrent_check = bool(
            (matrix_recurrent and checks.get("matrix_recurrent_leaf"))
            or (factorized_recurrent and checks.get("factorized_recurrent_leaf"))
        )
        if not recurrent_candidate or not recurrent_check:
            failures.append("training did not execute the requested recurrent boundary")
        linear_reference = any(
            row.get("event") == "pre_optimizer_step"
            and row.get("boundary") == "linear"
            and row.get("selected") == "reference"
            and row.get("implementation") == "torch-reference-linear-v1"
            for row in routes
        )
        flattened_linear = (
            any(
                row.get("event") == "pre_optimizer_step"
                and row.get("boundary") == "linear"
                and row.get("selected") == "optimized"
                and row.get("implementation") == FLATTENED_LINEAR_IMPLEMENTATION
                for row in routes
            )
            or int(
                kernel_trace.get("actual_linear_calls", {}).get(
                    FLATTENED_LINEAR_IMPLEMENTATION, 0
                )
            )
            > 0
        )
        reference_mix6 = any(
            row.get("event") == "pre_optimizer_step"
            and row.get("boundary") == "mix6"
            and row.get("selected") == "reference"
            and row.get("implementation") == "torch-reference-mix6-v1"
            for row in routes
        )
        optimized_mix6 = (
            any(
                row.get("event") == "pre_optimizer_step"
                and row.get("boundary") == "mix6"
                and row.get("selected") == "optimized"
                and row.get("implementation") == MIX6_IMPLEMENTATION
                for row in routes
            )
            or int(
                kernel_trace.get("actual_mix6_calls", {}).get(MIX6_IMPLEMENTATION, 0)
            )
            > 0
        )
        if require_training_candidate == "matrix":
            if not linear_reference:
                failures.append("matrix training did not retain reference linears")
            if not reference_mix6:
                failures.append("matrix training did not retain reference Mix6")
        elif require_training_candidate == "factorized":
            if not flattened_linear or not checks.get("flattened_linear_leaf"):
                failures.append(
                    "factorized CUDA training did not execute the flattened linear boundary"
                )
            if not optimized_mix6 or not checks.get("mix6_leaf"):
                failures.append(
                    "factorized CUDA training did not execute the Mix6 boundary"
                )
        else:
            if matrix_recurrent and not linear_reference:
                failures.append(
                    "adaptive matrix fallback did not retain reference linears"
                )
            if factorized_recurrent and not (flattened_linear or linear_reference):
                failures.append(
                    "adaptive factorized route did not record a linear boundary"
                )
            if not optimized_mix6 or not checks.get("mix6_leaf"):
                failures.append("adaptive training did not execute the Mix6 boundary")
        if factorized_recurrent and flattened_linear and optimized_mix6:
            if not checks.get("clean_leaf_training"):
                failures.append("clean accelerated training leaves were not reconciled")
        historical_trace_count = int(
            kernel_trace.get("actual_model_calls", {}).get(
                HISTORICAL_WHOLE_MODEL_IMPLEMENTATION, 0
            )
        )
        if (
            checks.get("historical_whole_model_diagnostic") is not False
            or historical_trace_count
        ):
            failures.append("whole-model diagnostic training unexpectedly executed")
    if name == "grpo" and not any(
        float(row.get("reward_std", 0.0)) > 0.0 for row in metrics
    ):
        failures.append("GRPO never observed nonzero within-group reward variance")
    return {
        "path": str(path),
        "global_step": checks.get("global_step"),
        "metrics": len(metrics),
        "changed_parameters": len(changed),
        "inventory_files": len(inventory),
        "adapter_reload_max_abs": reload.get("max_abs"),
        "backend_routes": routes,
        "kernel_route_trace": kernel_trace,
        "adaptive_route_evidence": adaptive_route_evidence,
        "reference_route_evidence": reference_route_evidence,
        "artifacts": artifacts,
        "status": "passed" if not failures else "failed",
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate canonical SFT/DPO/GRPO artifacts"
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--require-backend-v2-routes", action="store_true")
    parser.add_argument(
        "--require-training-candidate",
        choices=("reference", "adaptive", "matrix", "factorized"),
    )
    args = parser.parse_args()
    runs = {
        name: validate_run(
            args.result_dir / name,
            name,
            100,
            require_backend_v2_routes=args.require_backend_v2_routes,
            require_training_candidate=args.require_training_candidate,
        )
        for name in ("sft", "dpo", "grpo")
    }
    resume = read(args.result_dir / "sft-resume" / "resume_check.json")
    resume_exit = read(args.result_dir / "sft-resume" / "exit_status.json")
    wandb = read(args.result_dir / "sft-wandb-offline" / "wandb.json")
    wandb_exit = read(args.result_dir / "sft-wandb-offline" / "exit_status.json")
    ancillary = {
        "resume": {
            "passed": bool(resume.get("advanced"))
            and resume_exit.get("returncode") == 0,
            **resume,
        },
        "wandb_offline": {
            "passed": wandb_exit.get("returncode") == 0 and bool(wandb.get("enabled")),
            **wandb,
        },
    }
    passed = all(row["status"] == "passed" for row in runs.values()) and all(
        row["passed"] for row in ancillary.values()
    )
    report = {
        "status": "passed" if passed else "failed",
        "runs": runs,
        "ancillary": ancillary,
    }
    (args.result_dir / "validation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
