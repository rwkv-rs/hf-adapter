"""Small correctness primitives used by benchmark gates."""

from __future__ import annotations

from typing import Any


def tensor_comparison(reference: Any, candidate: Any) -> dict[str, float | bool]:
    import torch
    import torch.nn.functional as functional

    ref = reference.float()
    cand = candidate.float()
    return {
        "max_abs_diff": float((ref - cand).abs().max().detach().cpu()),
        "min_cosine": float(
            functional.cosine_similarity(ref, cand, dim=-1).min().detach().cpu()
        ),
        "greedy_match": bool(
            torch.equal(
                ref.argmax(dim=-1).detach().cpu(),
                cand.argmax(dim=-1).detach().cpu(),
            )
        ),
    }
