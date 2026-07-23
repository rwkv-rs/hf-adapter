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
    "MODEL_GRADIENT_COSINE_MIN",
    "MODEL_GRADIENT_RELATIVE_L2_MAX",
    "MODEL_LOGITS_COSINE_MIN",
    "MODEL_LOSS_MAX_ABS",
    "NAMED_GRADIENT_COSINE_MIN_DIAGNOSTIC",
    "NAMED_GRADIENT_RELATIVE_L2_MAX_DIAGNOSTIC",
    "global_gradient_metric",
    "global_gradient_passed",
    "gradient_parameter_summary",
]
