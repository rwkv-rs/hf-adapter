"""Release-time contract for actual optional-backend route evidence.

Inference may use the optional backend. Formal Hugging Face training currently
uses one complete readable reference program because API v4 cannot bind and
preflight every concrete training leaf before the layer loop. Keeping this
rule in one helper prevents provenance, asset verification, and the generated
public Issue from promoting private leaf diagnostics as a production route.
"""

from __future__ import annotations

from typing import Any


REQUIRED_ROUTE_PHASES = ("prefill", "decode", "training", "quantization")
SELECTOR_ONLY_ROUTES = frozenset(
    {
        "auto",
        "graph",
        "native",
        "optimized",
        "reference",
        "triton",
    }
)

READABLE_TRAINING_MODEL_ROUTE = "torch-reference-model-v1"
REFERENCE_TRAINING_PROGRAM_ROUTE = "torch-reference-training-program-v1"
REFERENCE_TRAINING_RECURRENT_ROUTE = "torch-reference-v1"
REFERENCE_TRAINING_LINEAR_ROUTE = "torch-reference-linear-v1"
REFERENCE_TRAINING_MIX6_ROUTE = "torch-reference-mix6-v1"
REQUIRED_REFERENCE_TRAINING_ROUTES = frozenset(
    {
        READABLE_TRAINING_MODEL_ROUTE,
        REFERENCE_TRAINING_PROGRAM_ROUTE,
        REFERENCE_TRAINING_RECURRENT_ROUTE,
        REFERENCE_TRAINING_LINEAR_ROUTE,
        REFERENCE_TRAINING_MIX6_ROUTE,
    }
)

# Retained solely to reject or label historical/isolated diagnostics.
ADAPTIVE_TRAINING_PROGRAM_ROUTE = "native-nvidia-rwkv7-adaptive-training-program-v1"
DIAGNOSTIC_OPTIMIZED_TRAINING_LEAF_ROUTES = frozenset(
    {
        "native-nvidia-rwkv7-factorized-recurrent-training-v1",
        "torch-cuda-rwkv7-flattened-linear-training-v1",
        "native-nvidia-rwkv7-mix6-training-v1",
    }
)
HISTORICAL_WHOLE_MODEL_TRAINING_ROUTE = "native-nvidia-official-training-autograd-v2"
DIAGNOSTIC_TRAINING_ROUTES = frozenset(
    {
        ADAPTIVE_TRAINING_PROGRAM_ROUTE,
        *DIAGNOSTIC_OPTIMIZED_TRAINING_LEAF_ROUTES,
        "torch-cuda-rwkv7-batched-matrix-recurrent-training-v1",
    }
)
FORMAL_REFERENCE_BACKEND_ENVIRONMENT = {
    "RWKV7_BACKEND": "auto",
    "RWKV7_KERNEL_IMPL": "auto",
    "RWKV7_MODEL_KERNEL_IMPL": "auto",
    "RWKV7_TRAINING_KERNEL_IMPL": "auto",
}


def route_values(value: Any) -> list[str]:
    """Normalize one route field while rejecting selectors and empty values."""

    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        values = value
    else:
        raise ValueError("actual route evidence must be a string or list of strings")
    if not values or any(not item.strip() for item in values):
        raise ValueError("actual route evidence must not be empty")
    if any(item.strip().lower() in SELECTOR_ONLY_ROUTES for item in values):
        raise ValueError("requested selector is not actual route evidence")
    return values


def validate_training_routes(value: Any) -> list[str]:
    """Require the exact complete readable training program."""

    normalized = route_values(value)
    training = set(normalized)

    historical = sorted(
        route
        for route in training
        if route == HISTORICAL_WHOLE_MODEL_TRAINING_ROUTE
        or route.startswith(f"{HISTORICAL_WHOLE_MODEL_TRAINING_ROUTE}[")
    )
    if historical:
        raise ValueError(
            "historical whole-model train-temp route is not formal HF training "
            f"evidence: {historical}"
        )
    missing = sorted(REQUIRED_REFERENCE_TRAINING_ROUTES - training)
    if missing:
        raise ValueError(
            "training route evidence lacks the complete reference program: "
            f"{missing}"
        )
    diagnostics = sorted(training & DIAGNOSTIC_TRAINING_ROUTES)
    if diagnostics:
        raise ValueError(
            "formal reference training contains optional diagnostic routes: "
            f"{diagnostics}"
        )
    unknown = sorted(training - REQUIRED_REFERENCE_TRAINING_ROUTES)
    if unknown:
        raise ValueError(f"training route evidence contains unknown routes: {unknown}")
    return normalized


def validate_formal_reference_environment(environment: Any) -> dict[str, str]:
    """Require the fail-closed selector state used by formal HF training."""

    if not isinstance(environment, dict):
        raise ValueError("formal reference environment is missing")
    backend = environment.get("backend_environment")
    if not isinstance(backend, dict):
        raise ValueError("formal reference backend environment is missing")
    actual = {name: backend.get(name) for name in FORMAL_REFERENCE_BACKEND_ENVIRONMENT}
    if actual != FORMAL_REFERENCE_BACKEND_ENVIRONMENT:
        raise ValueError(
            "formal reference backend environment differs: "
            f"expected={FORMAL_REFERENCE_BACKEND_ENVIRONMENT} actual={actual}"
        )
    return actual


def validate_actual_routes(routes: Any) -> dict[str, list[str]]:
    """Validate and normalize the complete release route matrix.

    Training evidence must prove the complete readable reference program and
    contain no optimized/diagnostic leaf. Historical diagnostics remain useful
    but are not admissible HF release routes.
    """

    if not isinstance(routes, dict):
        raise ValueError("actual route evidence is missing")
    normalized: dict[str, list[str]] = {}
    for phase in REQUIRED_ROUTE_PHASES:
        if phase not in routes or routes[phase] in (None, [], ""):
            raise ValueError(f"actual {phase} route evidence is missing")
        normalized[phase] = route_values(routes[phase])
    normalized["training"] = validate_training_routes(normalized["training"])
    return normalized
