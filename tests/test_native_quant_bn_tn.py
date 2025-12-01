from __future__ import annotations

import pytest

from rwkv7_hf.native_quant_bn_tn import (
    BNTNGrid,
    RTX5090_DECODE_GRID,
    RTX5090_PREFILL_GRID,
    rtx5090_w4_grid,
    rtx5090_w4_launch_plan,
)


def test_production_grid_has_physical_bn_tn_semantics() -> None:
    assert RTX5090_DECODE_GRID.as_dict() == {
        "block_n": 128,
        "thread_n": 8,
        "tile_k": 128,
        "cuda_threads": 256,
        "stages": 4,
        "logical_output_writers": 16,
    }
    assert RTX5090_PREFILL_GRID.block_n == 256
    assert RTX5090_PREFILL_GRID.logical_output_writers == 32


def test_production_grid_routes_decode_and_prefill_rows() -> None:
    assert rtx5090_w4_grid(1) is RTX5090_DECODE_GRID
    assert rtx5090_w4_grid(8) is RTX5090_DECODE_GRID
    assert rtx5090_w4_grid(16) is RTX5090_DECODE_GRID
    assert rtx5090_w4_grid(17) is RTX5090_PREFILL_GRID
    assert rtx5090_w4_grid(1024) is RTX5090_PREFILL_GRID


def test_launch_plan_covers_mixed_low_row_tail() -> None:
    plan = rtx5090_w4_launch_plan(65, 16384)
    assert [launch.rows for launch in plan.launches] == [64, 1]
    assert [launch.grid.block_n for launch in plan.launches] == [256, 128]
    assert plan.mixed_grid is True
    assert plan.as_dict()["launches"][-1]["thread_n"] == 8


def test_launch_plan_tracks_marlin_output_width_parallelism_cap() -> None:
    up = rtx5090_w4_launch_plan(8193, 16384)
    down = rtx5090_w4_launch_plan(8193, 4096)
    assert [launch.rows for launch in up.launches] == [1024] * 8 + [1]
    assert [launch.rows for launch in down.launches] == [8192, 1]
    assert up.mixed_grid and down.mixed_grid


@pytest.mark.parametrize(
    "kwargs",
    [
        {"block_n": 96, "thread_n": 8, "tile_k": 64, "cuda_threads": 128, "stages": 4},
        {"block_n": 128, "thread_n": 4, "tile_k": 64, "cuda_threads": 128, "stages": 4},
        {"block_n": 128, "thread_n": 8, "tile_k": 32, "cuda_threads": 128, "stages": 4},
        {"block_n": 128, "thread_n": 8, "tile_k": 64, "cuda_threads": 64, "stages": 4},
        {"block_n": 128, "thread_n": 8, "tile_k": 64, "cuda_threads": 128, "stages": 3},
    ],
)
def test_grid_rejects_unemitted_physical_shapes(kwargs) -> None:
    with pytest.raises(ValueError):
        BNTNGrid(**kwargs)


def test_grid_rejects_nonpositive_rows() -> None:
    with pytest.raises(ValueError):
        rtx5090_w4_grid(0)
    with pytest.raises(ValueError):
        rtx5090_w4_launch_plan(1, 0)
