from __future__ import annotations

import torch
import pytest

from bench.bench_chunked_prefill import alignment_passes, minimum_row_cosine


def test_minimum_row_cosine_uses_each_batch_row() -> None:
    reference = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    candidate = torch.tensor([[2.0, 0.0], [1.0, 1.0]])
    assert minimum_row_cosine(reference, candidate) == pytest.approx(2**-0.5)


def test_alignment_accepts_high_cosine_fp16_scale_drift() -> None:
    assert alignment_passes(
        max_abs_diff=0.1875,
        min_cosine=0.99999,
        greedy_match=True,
        max_diff_limit=0.15,
        min_cosine_limit=0.9999,
    )


def test_alignment_rejects_token_or_direction_mismatch() -> None:
    assert not alignment_passes(0.01, 1.0, False, 0.15, 0.9999)
    assert not alignment_passes(0.2, 0.99, True, 0.15, 0.9999)
