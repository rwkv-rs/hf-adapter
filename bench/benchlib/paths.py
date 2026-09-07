"""Stable repository paths for scripts moved below ``bench/``."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


BENCH_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BENCH_ROOT.parent
DEFAULT_RESULTS_PATH = BENCH_ROOT / "_runs" / "results.jsonl"


def default_run_directory(name: str, *, now: datetime | None = None) -> Path:
    """Return a non-root scratch directory for an unpromoted benchmark run."""

    instant = now or datetime.now(timezone.utc)
    stamp = instant.strftime("%Y%m%dT%H%M%SZ")
    return BENCH_ROOT / "_runs" / f"{name}_{stamp}"
