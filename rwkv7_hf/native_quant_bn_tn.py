# coding=utf-8
"""Production BN/TN grid contract for the BF16/W4 Tensor Core backend.

``BN`` and ``TN`` retain their physical meanings here:

* ``BN`` is the number of output columns in one Tensor Core CTA tile.
* ``TN`` is the number of contiguous BF16 output columns written by one
  coalesced epilogue writer.  The vendored Marlin epilogue stores one CUDA
  ``int4`` (16 bytes), therefore ``TN == 8``.

This is intentionally different from Marlin's historical ``thread_n`` name,
which is a CTA tile width rather than a per-thread output width.  The runtime
passes the expected BN/TN to the CUDA launcher and fails closed if the selected
kernel does not implement the promised grid.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BNTNGrid:
    """One physically defined BF16/W4 Tensor Core launch grid."""

    block_n: int
    thread_n: int
    tile_k: int
    cuda_threads: int
    stages: int

    def __post_init__(self) -> None:
        if self.block_n not in (64, 128, 256):
            raise ValueError("BN must be 64, 128, or 256")
        if self.thread_n != 8:
            raise ValueError("the vectorized BF16 epilogue requires TN=8")
        if self.block_n % self.thread_n:
            raise ValueError("BN must be divisible by TN")
        if self.tile_k not in (64, 128):
            raise ValueError("K tile must be 64 or 128")
        if self.cuda_threads not in (128, 256):
            raise ValueError("CUDA thread count must be 128 or 256")
        if self.stages not in (2, 4):
            raise ValueError("pipeline stages must be 2 or 4")

    @property
    def logical_output_writers(self) -> int:
        """Number of TN-wide writers required for one output row."""

        return self.block_n // self.thread_n

    def as_dict(self) -> dict[str, int]:
        return {
            "block_n": self.block_n,
            "thread_n": self.thread_n,
            "tile_k": self.tile_k,
            "cuda_threads": self.cuda_threads,
            "stages": self.stages,
            "logical_output_writers": self.logical_output_writers,
        }


@dataclass(frozen=True)
class BNTNLaunch:
    """One internal Marlin row segment and the grid selected for it."""

    rows: int
    grid: BNTNGrid

    def __post_init__(self) -> None:
        if self.rows <= 0:
            raise ValueError("launch rows must be positive")

    def as_dict(self) -> dict[str, object]:
        return {"rows": self.rows, **self.grid.as_dict()}


@dataclass(frozen=True)
class BNTNLaunchPlan:
    """All internal launches used to execute one logical GEMM."""

    rows: int
    out_features: int
    launches: tuple[BNTNLaunch, ...]

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.out_features <= 0:
            raise ValueError("rows and out_features must be positive")
        if sum(launch.rows for launch in self.launches) != self.rows:
            raise ValueError("launch rows must exactly cover the logical GEMM")

    @property
    def mixed_grid(self) -> bool:
        return len({launch.grid for launch in self.launches}) > 1

    def as_dict(self) -> dict[str, object]:
        return {
            "rows": self.rows,
            "out_features": self.out_features,
            "mixed_grid": self.mixed_grid,
            "launches": [launch.as_dict() for launch in self.launches],
        }


# Exact grid selected by the current Marlin scheduler for the measured SM120
# group-128 U4B8 kernels. Decode and prefill differ because low-row latency and
# high-row CTA reuse have different occupancy optima; device-name policy stays
# in kernel_policy.py.
RTX5090_DECODE_GRID = BNTNGrid(
    block_n=128,
    thread_n=8,
    tile_k=128,
    cuda_threads=256,
    stages=4,
)
RTX5090_PREFILL_GRID = BNTNGrid(
    block_n=256,
    thread_n=8,
    tile_k=64,
    cuda_threads=256,
    stages=4,
)


def rtx5090_w4_grid(rows: int) -> BNTNGrid:
    """Resolve the production grid for one internal Marlin row segment."""

    rows = int(rows)
    if rows <= 0:
        raise ValueError("rows must be positive")
    return RTX5090_DECODE_GRID if rows <= 16 else RTX5090_PREFILL_GRID


def rtx5090_w4_launch_plan(rows: int, out_features: int) -> BNTNLaunchPlan:
    """Resolve every internal launch, including a low-row tail segment.

    Marlin processes large or non-aligned M dimensions in row segments.  A
    logical 65-row GEMM, for example, executes a 64-row BN=256 launch followed
    by a 1-row BN=128 launch.  The CUDA production sentinel validates this same
    rule after the scheduler has formed each segment.
    """

    rows = int(rows)
    out_features = int(out_features)
    if rows <= 0 or out_features <= 0:
        raise ValueError("rows and out_features must be positive")
    max_parallel = 128 if out_features <= 4096 else 16
    max_segment_rows = max_parallel * 4 * 16
    launches: list[BNTNLaunch] = []
    remaining = rows
    while remaining:
        segment_rows = min((remaining // 64) * 64, max_segment_rows)
        if segment_rows == 0:
            segment_rows = remaining
        launches.append(BNTNLaunch(segment_rows, rtx5090_w4_grid(segment_rows)))
        remaining -= segment_rows
    return BNTNLaunchPlan(rows, out_features, tuple(launches))


__all__ = [
    "BNTNGrid",
    "BNTNLaunch",
    "BNTNLaunchPlan",
    "RTX5090_DECODE_GRID",
    "RTX5090_PREFILL_GRID",
    "rtx5090_w4_grid",
    "rtx5090_w4_launch_plan",
]
