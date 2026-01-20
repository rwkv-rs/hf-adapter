#!/usr/bin/env python3
"""Guard the canonical-vs-historical documentation boundaries."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def main() -> int:
    canonical_dates = {
        "HF_STATUS.md": "2026-07-26",
        "HF_TODO.md": "2026-07-26",
        "BENCHMARK.md": "2026-07-20",
        "docs/ACCEPTANCE.md": "2026-07-26",
        "docs/HARDWARE_MATRIX.md": "2026-07-20",
    }
    for relative, expected_date in canonical_dates.items():
        text = read(relative)
        assert expected_date in text, f"missing current audit date: {relative}"

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
    assert "## Scope and current boundary" in todo
    assert "current HF milestone is complete" in todo
    assert "4d1de1733b90e99eaf9c104eb73639eb221e3ad2" in todo
    assert "- [x]" not in todo
    assert "PP/TP are closed for the declared HF scope" in todo
    assert "PP/TP and multi-device behavior" not in todo
    assert "accepted RTX 5090 full-MATH500" in todo
    assert "promoted Apple M5 MLX pairs/shapes for raw peak-memory" in todo
    assert "V100 CUDA target/draft speculative artifact" in todo
    assert "Capture true peak-to-peak memory" not in todo
    assert "Add CUDA target/draft end-to-end speed" not in todo
    assert "Add end-user SFT/LoRA/DPO examples" not in todo
    assert "scheduled clean-install CPU plus optional CUDA and Apple" not in todo
    assert "Keep card-specific routing isolated" not in todo
    assert "Leave DFlash and serving-engine scheduler" not in todo
    assert "Test a supported Transformers/PEFT/TRL version range in CI" not in todo
    assert "Publish a release-versioned deprecation window" not in todo
    assert "Add and enforce pytest markers" not in todo
    assert "minimum/current Transformers-PEFT-TRL CI lanes" in todo
    assert "per-PR template, not a list of outstanding project tasks" in todo
    assert "Do not convert the unchecked roadmap" in todo

    status = read("HF_STATUS.md")
    assert "## Completion reporting rule" in status
    assert "no official repository-wide completion percentage" in status
    assert "| PP/TP boundary | **PASS for HF scope**" in status

    acceptance = read("docs/ACCEPTANCE.md")
    assert "## How to report completion" in acceptance
    assert "current HF milestone is complete" in acceptance
    assert "| PP/TP boundary | **PASS for HF scope**" in acceptance

    readme = read("README.md")
    assert "Completion is reported by **named scope**" in readme

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
