# coding=utf-8
"""Shared validation helpers for explicit CUDA ``BN``/``TN`` sweeps.

``BN`` is the number of output columns owned by a CUDA thread block. ``TN``
is the number of output columns accumulated by one CUDA thread.  They are
separate tuning dimensions; the corresponding thread count is ``BN / TN``.

The production Triton kernels do not expose a physical per-thread ``TN`` --
their lane/MMA layout is compiler controlled.  These helpers are therefore
used by the handwritten-CUDA probe and by promotion tooling, not to relabel a
Triton ``num_warps`` sweep as a ``TN`` sweep.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True, order=True)
class BNTNConfig:
    """One legal explicit CUDA output-tile configuration."""

    block_n: int
    thread_n: int

    def __post_init__(self) -> None:
        block_n = int(self.block_n)
        thread_n = int(self.thread_n)
        if block_n <= 0 or thread_n <= 0:
            raise ValueError("BN and TN must be positive")
        if block_n % thread_n:
            raise ValueError("BN must be divisible by TN")
        threads = block_n // thread_n
        if threads < 32 or threads > 1024 or threads % 32:
            raise ValueError("BN/TN must produce 32..1024 CUDA threads in whole warps")

    @property
    def threads(self) -> int:
        return int(self.block_n) // int(self.thread_n)

    def as_dict(self) -> dict[str, int]:
        return {
            "block_n": int(self.block_n),
            "thread_n": int(self.thread_n),
            "threads": self.threads,
        }


def bn_tn_candidates(
    block_ns: Iterable[int] = (64, 128, 256),
    thread_ns: Iterable[int] = (1, 2, 4, 8),
) -> tuple[BNTNConfig, ...]:
    """Return every legal Cartesian-product candidate in stable order."""

    out: list[BNTNConfig] = []
    for block_n in block_ns:
        for thread_n in thread_ns:
            try:
                out.append(BNTNConfig(int(block_n), int(thread_n)))
            except ValueError:
                continue
    return tuple(out)


def select_best_bn_tn(
    rows: Sequence[Mapping[str, object]],
    *,
    latency_key: str = "candidate_ms",
    cosine_key: str = "cosine_vs_current",
    min_cosine: float = 0.999,
) -> Mapping[str, object] | None:
    """Select the lowest-latency correct row from a measured sweep."""

    accepted = []
    for row in rows:
        try:
            latency = float(row[latency_key])
            cosine = float(row[cosine_key])
            BNTNConfig(int(row["block_n"]), int(row["thread_n"]))
        except (KeyError, TypeError, ValueError):
            continue
        if latency > 0.0 and cosine >= float(min_cosine):
            accepted.append((latency, int(row["block_n"]), int(row["thread_n"]), row))
    return min(accepted, default=None, key=lambda item: item[:3])[3] if accepted else None


__all__ = ["BNTNConfig", "bn_tn_candidates", "select_best_bn_tn"]
