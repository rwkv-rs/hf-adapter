"""Reusable RWKV-LM train_temp alignment semantics and metrics.

This module intentionally contains no model-loading or benchmark orchestration.
It keeps the loss and optimizer grouping rules small enough to unit test before
running the official and Hugging Face models on a GPU.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any, Literal

import torch
import torch.nn.functional as F


ParameterNaming = Literal["official", "hf"]


class _TrainTempL2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        loss: torch.Tensor,
        logits: torch.Tensor,
        token_count: int,
        factor: float,
    ) -> torch.Tensor:
        ctx.save_for_backward(logits)
        ctx.token_count = int(token_count)
        ctx.factor = float(factor)
        return loss

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        (logits,) = ctx.saved_tensors
        scale = ctx.factor / ctx.token_count
        max_values, max_indices = torch.max(logits, dim=-1, keepdim=True)
        extra = torch.zeros_like(logits)
        extra.scatter_(-1, max_indices, max_values * scale)
        return grad_output, extra * grad_output.to(extra.dtype), None, None


def train_temp_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int = -100,
    l2wrap_factor: float = 1.0e-4,
) -> torch.Tensor:
    """Return train_temp-style causal CE with the L2Wrap logit gradient.

    The official production kernel fuses CE and L2Wrap. Its forward scalar is
    ordinary mean cross entropy; backward adds ``max_logit * 1e-4 / (B*T)`` at
    each token's argmax logit. The denominator remains the dense target count,
    matching train_temp batches, which do not use padded targets.
    """

    if logits.ndim < 2:
        raise ValueError(f"logits must have at least two dimensions, got {tuple(logits.shape)}")
    if tuple(logits.shape[:-1]) != tuple(targets.shape):
        raise ValueError(
            "targets must match logits without the vocabulary dimension: "
            f"{tuple(targets.shape)} != {tuple(logits.shape[:-1])}"
        )
    if targets.numel() <= 0:
        raise ValueError("train_temp_cross_entropy requires at least one target")
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )
    return _TrainTempL2Wrap.apply(loss, logits, int(targets.numel()), float(l2wrap_factor))


def _is_train_temp_w0(name: str, naming: ParameterNaming) -> bool:
    if naming == "official":
        return "att.w0" in name
    if naming == "hf":
        parts = name.split(".")
        return len(parts) >= 6 and parts[-5:] == ["attn", "w_lora", "lora", "2", "bias"]
    raise ValueError(f"unsupported parameter naming: {naming!r}")


def _is_translated_train_temp_low_rank_weight(name: str, naming: ParameterNaming) -> bool:
    if naming != "hf":
        return False
    parts = name.split(".")
    return (
        len(parts) >= 7
        and parts[-5] == "attn"
        and parts[-4] in {"w_lora", "a_lora", "g_lora", "v_lora"}
        and parts[-3] == "lora"
        and parts[-2] in {"0", "2"}
        and parts[-1] == "weight"
    )


def build_train_temp_param_groups(
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    weight_decay: float,
    naming: ParameterNaming,
) -> list[dict[str, Any]]:
    """Classify parameters using the official train_temp optimizer recipe."""

    if naming not in {"official", "hf"}:
        raise ValueError(f"unsupported parameter naming: {naming!r}")
    buckets: dict[str, list[tuple[str, torch.nn.Parameter]]] = {
        "lr_1x": [],
        "lr_2x": [],
        "decay": [],
    }
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        if _is_train_temp_w0(name, naming):
            bucket = "lr_2x"
        elif _is_translated_train_temp_low_rank_weight(name, naming):
            # Official train_temp names these tensors w1/w2, a1/a2, g1/g2,
            # and v1/v2, so its `.weight` decay rule does not select them.
            bucket = "lr_1x"
        elif weight_decay > 0 and ".weight" in name and parameter.squeeze().ndim >= 2:
            bucket = "decay"
        else:
            bucket = "lr_1x"
        buckets[bucket].append((name, parameter))

    groups: list[dict[str, Any]] = []
    for group_name, lr_scale, decay in (
        ("lr_1x", 1.0, 0.0),
        ("lr_2x", 2.0, 0.0),
        ("decay", 1.0, float(weight_decay)),
    ):
        entries = sorted(buckets[group_name], key=lambda item: item[0])
        if not entries:
            continue
        groups.append(
            {
                "group_name": group_name,
                "params": [parameter for _, parameter in entries],
                "param_names": [name for name, _ in entries],
                "weight_decay": decay,
                "my_lr_scale": lr_scale,
            }
        )
    return groups


def compare_tensors(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    """Return stable alignment metrics without accepting invalid tensors."""

    shape_match = tuple(reference.shape) == tuple(candidate.shape)
    finite = bool(torch.isfinite(reference).all().item() and torch.isfinite(candidate).all().item())
    comparable = shape_match and finite and reference.numel() > 0
    result: dict[str, Any] = {
        "shape_match": shape_match,
        "finite": finite,
        "comparable": comparable,
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "numel": int(reference.numel()) if shape_match else None,
        "cosine": None,
        "relative_l2": None,
        "max_abs": None,
        "mean_abs": None,
    }
    if not comparable:
        return result

    ref = reference.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    cand = candidate.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    diff = cand - ref
    if bool(torch.equal(ref, cand)):
        result.update(
            {
                "cosine": 1.0,
                "relative_l2": 0.0,
                "max_abs": 0.0,
                "mean_abs": 0.0,
            }
        )
        return result
    ref_norm = torch.linalg.vector_norm(ref)
    cand_norm = torch.linalg.vector_norm(cand)
    denominator = max(float(ref_norm.item()), torch.finfo(torch.float64).eps)
    if float(ref_norm.item()) == 0.0 and float(cand_norm.item()) == 0.0:
        cosine = 1.0
    elif float(ref_norm.item()) == 0.0 or float(cand_norm.item()) == 0.0:
        cosine = 0.0
    else:
        cosine = float(torch.dot(ref, cand).item() / (ref_norm.item() * cand_norm.item()))
    result.update(
        {
            "cosine": max(-1.0, min(1.0, cosine)),
            "relative_l2": float(torch.linalg.vector_norm(diff).item() / denominator),
            "max_abs": float(diff.abs().max().item()),
            "mean_abs": float(diff.abs().mean().item()),
        }
    )
    if not all(
        math.isfinite(float(result[key]))
        for key in ("cosine", "relative_l2", "max_abs", "mean_abs")
    ):
        result["finite"] = False
        result["comparable"] = False
    return result


__all__ = [
    "build_train_temp_param_groups",
    "compare_tensors",
    "train_temp_cross_entropy",
]
