from __future__ import annotations

import pytest

from rwkv7_hf.bn_tn_tuning import BNTNConfig, bn_tn_candidates, select_best_bn_tn


def test_bn_and_tn_are_independent_tile_dimensions() -> None:
    config = BNTNConfig(block_n=256, thread_n=4)
    assert config.threads == 64
    assert config.as_dict() == {"block_n": 256, "thread_n": 4, "threads": 64}


@pytest.mark.parametrize(
    "block_n,thread_n",
    [(0, 1), (64, 0), (100, 3), (64, 4), (2048, 1)],
)
def test_invalid_bn_tn_launches_fail_closed(block_n: int, thread_n: int) -> None:
    with pytest.raises(ValueError):
        BNTNConfig(block_n, thread_n)


def test_candidate_product_keeps_only_whole_warp_launches() -> None:
    got = bn_tn_candidates((64, 128, 256), (1, 2, 4, 8))
    assert got == (
        BNTNConfig(64, 1),
        BNTNConfig(64, 2),
        BNTNConfig(128, 1),
        BNTNConfig(128, 2),
        BNTNConfig(128, 4),
        BNTNConfig(256, 1),
        BNTNConfig(256, 2),
        BNTNConfig(256, 4),
        BNTNConfig(256, 8),
    )


def test_selection_rejects_fast_incorrect_rows() -> None:
    rows = [
        {"block_n": 128, "thread_n": 2, "candidate_ms": 0.8, "cosine_vs_current": 0.95},
        {"block_n": 128, "thread_n": 4, "candidate_ms": 1.0, "cosine_vs_current": 0.9999},
        {"block_n": 256, "thread_n": 8, "candidate_ms": 0.9, "cosine_vs_current": 0.9998},
    ]
    best = select_best_bn_tn(rows)
    assert best is rows[2]
