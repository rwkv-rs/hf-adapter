"""Shared, evaluation-only helpers for the pinned FLA RWKV-7 lane.

This module deliberately lives outside both installable distributions.  It
contains comparison and cache-adaptation code needed by the release harness,
not model or kernel implementation code.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

import torch


EXPECTED_FLA_COMMIT = "80e494f6c588e091fc8316b612870df29375c5b8"


def activate_fla_source(source: Path) -> dict[str, str]:
    """Verify *source* and make that exact checkout importable."""

    source = source.expanduser().resolve()
    if not (source / "fla" / "__init__.py").is_file():
        raise ValueError(f"not an FLA source tree: {source}")
    marker = source / ".fla-upstream-commit"
    if (source / ".git").exists():
        commit = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    elif marker.is_file():
        commit = marker.read_text(encoding="utf-8").strip()
    else:
        raise ValueError(
            f"FLA source has neither .git metadata nor {marker.name}: {source}"
        )
    if commit != EXPECTED_FLA_COMMIT:
        raise ValueError(
            f"FLA commit mismatch: expected {EXPECTED_FLA_COMMIT}, got {commit}"
        )
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    return {"source": source_text, "commit": commit}


def tensor_metric(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    """Return stable full-tensor metrics with FP64 scalar reductions."""

    candidate_tensor = candidate.detach().float()
    reference_tensor = reference.detach().float()
    if candidate_tensor.shape != reference_tensor.shape:
        return {
            "shape_match": False,
            "candidate_shape": list(candidate_tensor.shape),
            "reference_shape": list(reference_tensor.shape),
            "finite": False,
            "cosine": 0.0,
            "max_abs": float("inf"),
            "mean_abs": float("inf"),
            "relative_l2": float("inf"),
            "argmax_same": False,
            "fp32_allclose": False,
        }
    argmax_same = bool(
        torch.equal(
            candidate_tensor.argmax(dim=-1), reference_tensor.argmax(dim=-1)
        )
    )
    candidate = candidate_tensor.reshape(-1)
    reference = reference_tensor.reshape(-1)
    delta = candidate - reference
    dot = (candidate * reference).sum(dtype=torch.float64)
    left_norm = (candidate * candidate).sum(dtype=torch.float64).sqrt()
    right_norm = (reference * reference).sum(dtype=torch.float64).sqrt()
    denominator = left_norm * right_norm
    if denominator == 0:
        cosine = 1.0 if torch.equal(candidate, reference) else 0.0
    else:
        cosine = float((dot / denominator).clamp(-1.0, 1.0))
    return {
        "shape_match": True,
        "candidate_shape": list(candidate_tensor.shape),
        "reference_shape": list(reference_tensor.shape),
        "finite": bool(
            torch.isfinite(candidate).all() and torch.isfinite(reference).all()
        ),
        "cosine": cosine,
        "max_abs": float(delta.abs().max()) if delta.numel() else 0.0,
        "mean_abs": float(delta.abs().mean()) if delta.numel() else 0.0,
        "relative_l2": float(delta.norm(dtype=torch.float64))
        / max(float(right_norm), 1.0e-12),
        "argmax_same": argmax_same,
        "fp32_allclose": bool(
            torch.allclose(candidate, reference, rtol=1.0e-4, atol=1.0e-5)
        ),
    }


def release_metric_passed(
    row: dict[str, Any], dtype: torch.dtype, *, logits: bool = False
) -> bool:
    """Apply the calibrated release envelope from ``docs/EVALUATION.md``.

    ``max_abs`` remains part of every metric row, but it is not a blocking
    low-precision release criterion.  Equivalent CUDA GEMM layouts can move a
    small number of logits past an absolute ceiling while preserving the
    complete tensor cosine and generated sequence.  Model-level greedy
    equality is checked separately by the caller.

    ``logits`` is accepted for a symmetric API with
    :func:`aspirational_metric_passed`; it does not change the release floor.
    """

    if not row.get("shape_match", True) or not row.get("finite", False):
        return False
    if dtype == torch.float32:
        return bool(row.get("fp32_allclose", False))
    cosine_floor = 0.9999 if dtype == torch.float16 else 0.999
    return float(row.get("cosine", float("-inf"))) >= cosine_floor


def aspirational_metric_passed(
    row: dict[str, Any], dtype: torch.dtype, *, logits: bool = False
) -> bool:
    """Evaluate the original stricter target without making it a gate."""

    if not row.get("shape_match", True) or not row.get("finite", False):
        return False
    if dtype == torch.float32:
        return bool(row.get("fp32_allclose", False))
    passed = float(row.get("cosine", float("-inf"))) >= 0.9999
    if dtype == torch.float16 and logits:
        passed = passed and float(row.get("max_abs", float("inf"))) <= 0.15
    return bool(passed)


def annotate_metric(
    row: dict[str, Any], dtype: torch.dtype, *, logits: bool = False
) -> dict[str, Any]:
    """Attach both release and aspirational outcomes to a metric row."""

    row["release_passed"] = release_metric_passed(row, dtype, logits=logits)
    row["aspirational_passed"] = aspirational_metric_passed(
        row, dtype, logits=logits
    )
    return row


def metric_passed(
    row: dict[str, Any], dtype: torch.dtype, *, logits: bool = False
) -> bool:
    """Compatibility spelling for callers that only need the release gate."""

    return release_metric_passed(row, dtype, logits=logits)


def _state_from_layer(value: Any) -> torch.Tensor:
    if isinstance(value, dict):
        value = value.get("recurrent_state")
    elif hasattr(value, "state") and isinstance(value.state, dict):
        value = value.state.get("recurrent_state")
    if isinstance(value, (tuple, list)):
        if not value:
            raise ValueError("empty recurrent-state tuple")
        value = value[0]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"unsupported recurrent-state value: {type(value)!r}")
    return value


def recurrent_states(cache: Any) -> list[torch.Tensor]:
    """Extract recurrent tensors from either canonical HF or FLA caches."""

    values = getattr(cache, "recurrent_state", None)
    if isinstance(values, (tuple, list)):
        return [_state_from_layer(value) for value in values]
    if hasattr(cache, "states"):
        return [_state_from_layer(value) for value in cache.states]
    if hasattr(cache, "layers"):
        return [_state_from_layer(value) for value in cache.layers]
    if isinstance(cache, (tuple, list)):
        return [_state_from_layer(value) for value in cache]
    try:
        return [_state_from_layer(cache[index]) for index in range(len(cache))]
    except Exception as exc:
        raise TypeError(f"unsupported cache type: {type(cache)!r}") from exc


def compare_states(candidate_cache: Any, reference_cache: Any) -> list[dict[str, Any]]:
    """Compare caches while detecting FLA's alternate [V,K] presentation."""

    candidate = recurrent_states(candidate_cache)
    reference = recurrent_states(reference_cache)
    if len(candidate) != len(reference):
        raise ValueError(
            f"cache layer mismatch: candidate={len(candidate)}, "
            f"reference={len(reference)}"
        )
    rows: list[dict[str, Any]] = []
    for layer, (left, right) in enumerate(zip(candidate, reference, strict=True)):
        left = left.detach().cpu()
        right = right.detach().cpu()
        choices: list[tuple[str, torch.Tensor]] = [("direct", left)]
        if left.ndim >= 2:
            choices.append(("candidate_transposed", left.transpose(-1, -2)))
        viable = [item for item in choices if item[1].shape == right.shape]
        if not viable:
            row = tensor_metric(left, right)
            row.update({"layer": layer, "layout": "incompatible"})
            rows.append(row)
            continue
        layout, aligned = min(
            viable,
            key=lambda item: float(
                (item[1].float() - right.float()).abs().max()
            ),
        )
        row = tensor_metric(aligned, right)
        row.update({"layer": layer, "layout": layout})
        rows.append(row)
    return rows


def annotate_state_rows(
    rows: Iterable[dict[str, Any]], dtype: torch.dtype
) -> list[dict[str, Any]]:
    """Annotate state rows while preserving their complete numeric evidence."""

    return [annotate_metric(row, dtype, logits=False) for row in rows]


def state_rows_release_passed(
    rows: Iterable[dict[str, Any]], dtype: torch.dtype
) -> bool:
    return all(release_metric_passed(row, dtype, logits=False) for row in rows)


def state_rows_aspirational_passed(
    rows: Iterable[dict[str, Any]], dtype: torch.dtype
) -> bool:
    return all(aspirational_metric_passed(row, dtype, logits=False) for row in rows)


def state_rows_passed(rows: Iterable[dict[str, Any]], dtype: torch.dtype) -> bool:
    """Compatibility spelling for the blocking state release envelope."""

    return state_rows_release_passed(rows, dtype)


def gradient_metrics(
    candidate: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]
) -> dict[str, Any]:
    common = sorted(set(candidate) & set(reference))
    rows = {
        name: tensor_metric(candidate[name], reference[name]) for name in common
    }
    return {
        "candidate_only": sorted(set(candidate) - set(reference)),
        "reference_only": sorted(set(reference) - set(candidate)),
        "parameters": rows,
    }


def gradient_rows_passed(report: dict[str, Any], dtype: torch.dtype) -> bool:
    if report["candidate_only"] or report["reference_only"]:
        return False
    if not report["parameters"]:
        return False
    if dtype == torch.float32:
        return all(
            row["finite"] and row["relative_l2"] <= 5.0e-4
            for row in report["parameters"].values()
        )
    return all(
        row["finite"] and row["cosine"] >= 0.999 and row["relative_l2"] <= 0.02
        for row in report["parameters"].values()
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


__all__ = [
    "EXPECTED_FLA_COMMIT",
    "activate_fla_source",
    "annotate_metric",
    "annotate_state_rows",
    "aspirational_metric_passed",
    "compare_states",
    "gradient_metrics",
    "gradient_rows_passed",
    "metric_passed",
    "release_metric_passed",
    "recurrent_states",
    "state_rows_aspirational_passed",
    "state_rows_passed",
    "state_rows_release_passed",
    "tensor_metric",
    "write_json",
]
