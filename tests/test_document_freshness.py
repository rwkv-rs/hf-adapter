#!/usr/bin/env python3
"""Guard the canonical-vs-historical documentation boundaries."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def main() -> int:
    canonical = [
        "HF_STATUS.md",
        "HF_TODO.md",
        "BENCHMARK.md",
        "docs/ACCEPTANCE.md",
        "docs/HARDWARE_MATRIX.md",
    ]
    for relative in canonical:
        text = read(relative)
        assert "2026-07-16" in text, f"missing current audit date: {relative}"

    for path in sorted((ROOT / "docs/plans").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        assert "Historical" in text or "historical" in text, (
            f"plan lacks lifecycle banner: {path.relative_to(ROOT)}"
        )

    stale_exact = {
        "README.md": [
            "ZeRO3 resume remains a follow-up gap",
            "This is a wrapper-based first stage",
        ],
        "HF_TODO.md": ["### 2a. Verified-FLA Qwen3.5 RTX 5070 comparison"],
    }
    for relative, phrases in stale_exact.items():
        text = read(relative)
        for phrase in phrases:
            assert phrase not in text, f"stale phrase in {relative}: {phrase}"

    todo = read("HF_TODO.md")
    assert "## Current milestone — COMPLETE" in todo
    assert "per-PR template, not a list of outstanding project tasks" in todo

    required_current = [
        "README.md",
        "HF_STATUS.md",
        "BENCHMARK.md",
        "docs/ACCEPTANCE.md",
        "docs/HARDWARE_MATRIX.md",
        "docs/PERFORMANCE.md",
        "docs/validation/V100_HF_VALIDATION.md",
    ]
    for relative in required_current:
        assert "v100_active_b1b8_20260715" in read(relative), (
            f"V100 current artifact missing from {relative}"
        )

    v100_indexed = [
        "AGENTS.md",
        "README.md",
        "HF_STATUS.md",
        "BENCHMARK.md",
        "docs/ACCEPTANCE.md",
        "docs/HARDWARE_MATRIX.md",
        "docs/PERFORMANCE.md",
        "docs/QUANTIZATION.md",
        "docs/performance/FUSED_BACKEND.md",
        "docs/validation/V100_HF_VALIDATION.md",
        "bench/INDEX.md",
    ]
    for relative in v100_indexed:
        assert "v100_acceptance_20260716" in read(relative), (
            f"V100 consolidated index missing from {relative}"
        )

    assert "Strict global audit snapshot" in read(
        "docs/hardware/APPLE_PRODUCTION_ACCEPTANCE.md"
    )
    assert "Dated 2026-07-02 validation snapshot" in read(
        "bench/4090_validation_summary.md"
    )
    assert "Historical investigation" in read(
        "docs/validation/math500_accuracy_parity.md"
    )

    print("DOCUMENT FRESHNESS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
