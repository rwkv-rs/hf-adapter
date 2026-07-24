"""Shared full-model training metrics for RWKV-7 evaluation harnesses.

This module is evaluation-only. It compares the complete optimizer update
without concatenating every parameter into another large tensor, while callers
retain all named per-parameter diagnostics in their result bundle.
"""

from __future__ import annotations

from typing import Any

import torch


MODEL_LOGITS_COSINE_MIN = 0.9999
MODEL_LOSS_MAX_ABS = 0.01
MODEL_GRADIENT_COSINE_MIN = 0.9995
MODEL_GRADIENT_RELATIVE_L2_MAX = 0.025
NAMED_GRADIENT_COSINE_MIN_DIAGNOSTIC = 0.999
NAMED_GRADIENT_RELATIVE_L2_MAX_DIAGNOSTIC = 0.02

# Keep the independent evaluator oracle deliberately small until additional
# dense shapes have passed the same multi-seed full-model gate.  The runtime
# owns a matching policy in ``rwkv7_kernels.training_dispatcher``; validators
# do not ask the implementation under test which route they should expect.
ADAPTIVE_FAST_DOMAIN_BATCH = 4
ADAPTIVE_FAST_DOMAIN_TOKENS = 128


def adaptive_fast_domain_expected(
    *,
    batch: int,
    tokens: int,
    fully_active: bool = True,
    initial_state_zero: bool = True,
    token_aligned: bool | None = None,
) -> bool:
    """Return the independently declared adaptive fast-route envelope."""

    if token_aligned is None:
        token_aligned = tokens > 0 and tokens % 16 == 0
    return bool(
        fully_active
        and initial_state_zero
        and token_aligned
        and batch == ADAPTIVE_FAST_DOMAIN_BATCH
        and tokens == ADAPTIVE_FAST_DOMAIN_TOKENS
    )


def full_model_reference_release_envelope(
    comparison: dict[str, Any],
) -> dict[str, Any]:
    """Apply one fixed reference envelope to a full-model training lane.

    Candidate and comparison implementations are judged independently against
    the readable reference model.  This deliberately avoids a pairwise
    ordering between several roundoff metrics: one lane may have a smaller
    causal-loss delta while another has a closer optimizer-gradient vector,
    and both are valid when they satisfy the same published limits.
    """

    logits = comparison.get("logits") or {}
    loss = comparison.get("loss") or {}
    global_gradient = comparison.get("global_gradient") or {}
    components = {
        "logits": bool(
            logits.get("finite")
            and float(logits.get("cosine", float("-inf"))) >= MODEL_LOGITS_COSINE_MIN
        ),
        "causal_loss": bool(
            loss.get("finite")
            and float(loss.get("max_abs", float("inf"))) <= MODEL_LOSS_MAX_ABS
        ),
        "optimizer_gradient_vector": global_gradient_passed(global_gradient),
    }
    return {
        "passed": all(components.values()),
        "comparison_target": "readable-reference",
        "acceptance_basis": "fixed-full-model-reference-envelope",
        "components": components,
        "thresholds": {
            "logits_cosine_min": MODEL_LOGITS_COSINE_MIN,
            "causal_loss_max_abs": MODEL_LOSS_MAX_ABS,
            "optimizer_gradient_cosine_min": MODEL_GRADIENT_COSINE_MIN,
            "optimizer_gradient_relative_l2_max": (MODEL_GRADIENT_RELATIVE_L2_MAX),
        },
    }


def classify_candidate_and_fla_reference_results(
    candidate: dict[str, Any],
    fla: dict[str, Any],
) -> dict[str, Any]:
    """Separate the blocking candidate gate from the external FLA diagnostic.

    Both lanes retain the same readable-reference measurements.  Only the
    implementation under test is a release gate: pinned FLA is an independent
    third-party comparator whose BF16 drift must remain visible but cannot
    invalidate a candidate that satisfies the published reference envelope.
    """

    candidate_envelope = full_model_reference_release_envelope(candidate)
    fla_envelope = full_model_reference_release_envelope(fla)
    return {
        "passed": candidate_envelope["passed"],
        "candidate_reference_release_gate": {
            **candidate_envelope,
            "role": "release-gate-blocking",
        },
        "fla_reference_diagnostic": {
            **fla_envelope,
            "role": "diagnostic-non-blocking",
        },
    }


def checkpoint_input_hash_gate(
    cases: list[dict[str, Any]],
    *,
    key_fields: tuple[str, ...],
) -> dict[str, Any]:
    """Prove checkpoint-on/off cases consumed byte-identical token IDs."""

    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in cases:
        key = tuple(row[field] for field in key_fields)
        group = groups.setdefault(key, {"hashes": set(), "modes": set()})
        group["hashes"].add(str(row["input_ids_sha256"]))
        group["modes"].add(bool(row["checkpointing"]))
    rows = []
    for key, group in sorted(groups.items(), key=lambda item: repr(item[0])):
        hashes = sorted(group["hashes"])
        modes = sorted(group["modes"])
        rows.append(
            {
                "key": dict(zip(key_fields, key, strict=True)),
                "hashes": hashes,
                "checkpointing_modes": modes,
                "passed": len(hashes) == 1 and modes == [False, True],
            }
        )
    return {
        "passed": bool(rows) and all(row["passed"] for row in rows),
        "groups": rows,
    }


def global_gradient_metric(
    candidate: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Compare the complete optimizer update without concatenating tensors."""

    candidate_only = sorted(set(candidate) - set(reference))
    reference_only = sorted(set(reference) - set(candidate))
    common = sorted(set(candidate) & set(reference))
    shape_mismatch = {
        name: {
            "candidate": list(candidate[name].shape),
            "reference": list(reference[name].shape),
        }
        for name in common
        if candidate[name].shape != reference[name].shape
    }
    comparable = [name for name in common if name not in shape_mismatch]
    dot = torch.zeros((), dtype=torch.float64)
    candidate_square = torch.zeros((), dtype=torch.float64)
    reference_square = torch.zeros((), dtype=torch.float64)
    delta_square = torch.zeros((), dtype=torch.float64)
    max_abs = 0.0
    finite = True
    elements = 0
    for name in comparable:
        left = candidate[name].detach().float().reshape(-1)
        right = reference[name].detach().float().reshape(-1)
        delta = left - right
        finite = finite and bool(
            torch.isfinite(left).all()
            and torch.isfinite(right).all()
            and torch.isfinite(delta).all()
        )
        dot += (left * right).sum(dtype=torch.float64)
        candidate_square += (left * left).sum(dtype=torch.float64)
        reference_square += (right * right).sum(dtype=torch.float64)
        delta_square += (delta * delta).sum(dtype=torch.float64)
        if delta.numel():
            max_abs = max(max_abs, float(delta.abs().max()))
        elements += int(delta.numel())
    denominator = candidate_square.sqrt() * reference_square.sqrt()
    cosine = (
        1.0
        if float(denominator) == 0.0
        and float(candidate_square + reference_square) == 0.0
        else float(dot / denominator.clamp_min(1.0e-30))
    )
    reference_norm = reference_square.sqrt().clamp_min(1.0e-30)
    return {
        "finite": finite,
        "candidate_only": candidate_only,
        "reference_only": reference_only,
        "shape_mismatch": shape_mismatch,
        "parameter_count": len(comparable),
        "element_count": elements,
        "cosine": cosine,
        "relative_l2": float(delta_square.sqrt() / reference_norm),
        "candidate_to_reference_norm": float(candidate_square.sqrt() / reference_norm),
        "max_abs": max_abs,
    }


def global_gradient_passed(
    metric: dict[str, Any],
    *,
    cosine_min: float = MODEL_GRADIENT_COSINE_MIN,
    relative_l2_max: float = MODEL_GRADIENT_RELATIVE_L2_MAX,
) -> bool:
    """Apply the full-model optimizer-update acceptance contract.

    A named all-zero gradient vector is valid, but an empty or structurally
    incomplete vector is not.  Keeping the structural checks here prevents
    individual validators from accidentally treating an empty intersection as
    a perfect comparison.
    """

    return bool(
        metric.get("finite")
        and int(metric.get("parameter_count", 0)) > 0
        and int(metric.get("element_count", 0)) > 0
        and not metric.get("candidate_only")
        and not metric.get("reference_only")
        and not metric.get("shape_mismatch")
        and float(metric.get("cosine", float("-inf"))) >= cosine_min
        and float(metric.get("relative_l2", float("inf"))) <= relative_l2_max
    )


def gradient_parameter_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Summarize named-gradient spread while retaining every row in JSON."""

    rows = report["parameters"]
    if not rows:
        return {
            "parameter_count": 0,
            "strict_parameter_count": 0,
            "strict_parameter_fraction": 0.0,
        }
    strict = [
        name
        for name, row in rows.items()
        if row["finite"]
        and row["cosine"] >= NAMED_GRADIENT_COSINE_MIN_DIAGNOSTIC
        and row["relative_l2"] <= NAMED_GRADIENT_RELATIVE_L2_MAX_DIAGNOSTIC
    ]
    relative = sorted(float(row["relative_l2"]) for row in rows.values())
    cosine = sorted(float(row["cosine"]) for row in rows.values())

    def percentile(values: list[float], fraction: float) -> float:
        index = round((len(values) - 1) * fraction)
        return values[index]

    return {
        "parameter_count": len(rows),
        "strict_parameter_count": len(strict),
        "strict_parameter_fraction": len(strict) / len(rows),
        "relative_l2_median": percentile(relative, 0.5),
        "relative_l2_p95": percentile(relative, 0.95),
        "relative_l2_p99": percentile(relative, 0.99),
        "relative_l2_max": relative[-1],
        "cosine_min": cosine[0],
        "cosine_p01": percentile(cosine, 0.01),
        "cosine_median": percentile(cosine, 0.5),
    }


__all__ = [
    "ADAPTIVE_FAST_DOMAIN_BATCH",
    "ADAPTIVE_FAST_DOMAIN_TOKENS",
    "MODEL_GRADIENT_COSINE_MIN",
    "MODEL_GRADIENT_RELATIVE_L2_MAX",
    "MODEL_LOGITS_COSINE_MIN",
    "MODEL_LOSS_MAX_ABS",
    "NAMED_GRADIENT_COSINE_MIN_DIAGNOSTIC",
    "NAMED_GRADIENT_RELATIVE_L2_MAX_DIAGNOSTIC",
    "adaptive_fast_domain_expected",
    "classify_candidate_and_fla_reference_results",
    "checkpoint_input_hash_gate",
    "full_model_reference_release_envelope",
    "global_gradient_metric",
    "global_gradient_passed",
    "gradient_parameter_summary",
]
