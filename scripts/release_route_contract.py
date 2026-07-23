"""Release-time contract for actual optional-backend route evidence.

The training model boundary is intentionally not an optimized whole-model
forward.  Standard Hugging Face training executes the readable layer loop and
may accelerate only the three independent tensor leaves named below.  Keeping
this rule in one release helper prevents provenance, asset verification, and
the generated public Issue from drifting to different interpretations.
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
REQUIRED_TRAINING_LEAF_ROUTES = frozenset(
    {
        "native-nvidia-rwkv7-factorized-recurrent-training-v1",
        "torch-cuda-rwkv7-flattened-linear-training-v1",
        "native-nvidia-rwkv7-mix6-training-v1",
    }
)
HISTORICAL_WHOLE_MODEL_TRAINING_ROUTE = "native-nvidia-train-temp-autograd-v2"
TRAINING_FALLBACK_ROUTES = frozenset(
    {
        "torch-cuda-rwkv7-batched-matrix-recurrent-training-v1",
        "torch-reference-v1",
        "torch-reference-linear-v1",
        "torch-reference-mix6-v1",
    }
)
FORMAL_ADAPTIVE_BACKEND_ENVIRONMENT = {
    "RWKV7_BACKEND": "auto",
    "RWKV7_KERNEL_IMPL": "auto",
    "RWKV7_MODEL_KERNEL_IMPL": "auto",
    "RWKV7_TRAINING_KERNEL_IMPL": "adaptive",
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
    """Validate formal readable-loop training route evidence."""

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
    if READABLE_TRAINING_MODEL_ROUTE not in training:
        raise ValueError("training route evidence lacks the readable HF model loop")
    missing_leaves = sorted(REQUIRED_TRAINING_LEAF_ROUTES - training)
    if missing_leaves:
        raise ValueError(
            "training route evidence lacks required independent kernel leaves: "
            f"{missing_leaves}"
        )
    allowed = {
        READABLE_TRAINING_MODEL_ROUTE,
        *REQUIRED_TRAINING_LEAF_ROUTES,
        *TRAINING_FALLBACK_ROUTES,
    }
    unknown = sorted(training - allowed)
    if unknown:
        raise ValueError(f"training route evidence contains unknown routes: {unknown}")
    return normalized


def validate_formal_adaptive_environment(environment: Any) -> dict[str, str]:
    """Require the fail-closed selector state used by formal HF training."""

    if not isinstance(environment, dict):
        raise ValueError("formal adaptive environment is missing")
    backend = environment.get("backend_environment")
    if not isinstance(backend, dict):
        raise ValueError("formal adaptive backend environment is missing")
    actual = {name: backend.get(name) for name in FORMAL_ADAPTIVE_BACKEND_ENVIRONMENT}
    if actual != FORMAL_ADAPTIVE_BACKEND_ENVIRONMENT:
        raise ValueError(
            "formal adaptive backend environment differs: "
            f"expected={FORMAL_ADAPTIVE_BACKEND_ENVIRONMENT} actual={actual}"
        )
    return actual


def validate_actual_routes(routes: Any) -> dict[str, list[str]]:
    """Validate and normalize the complete release route matrix.

    Training evidence must prove the readable model boundary and every clean
    accelerated leaf.  The historical whole-model train-temp runtime remains
    useful for archived diagnostics, but it is not an admissible HF release
    route and must never be promoted by provenance.
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
