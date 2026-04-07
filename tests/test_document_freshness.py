#!/usr/bin/env python3
"""Guard the canonical-vs-historical documentation boundaries."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def main() -> int:
    canonical_dates = {
        "HF_STATUS.md": "2026-08-09",
        "HF_TODO.md": "2026-08-09",
        "BENCHMARK.md": "2026-08-09",
        "docs/ACCEPTANCE.md": "2026-08-09",
        "docs/HARDWARE_MATRIX.md": "2026-08-09",
        "docs/PROJECT_SUMMARY.md": "2026-08-09",
        "docs/RESULTS_INDEX.md": "2026-08-09",
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
        "HF_TODO.md": [
            "### 2a. Verified-FLA Qwen3.5 RTX 5070 comparison",
            "## P0 — Universal production gaps",
        ],
        "HF_STATUS.md": ["7.2B fp16 fits through B4/P128"],
        "BENCHMARK.md": ["7.2B fp16 fits through B4/P128"],
        "docs/ACCEPTANCE.md": ["The next backend promotion replaces"],
    }
    for relative, phrases in stale_exact.items():
        text = read(relative)
        for phrase in phrases:
            assert phrase not in text, f"stale phrase in {relative}: {phrase}"

    todo = read("HF_TODO.md")
    normalized_todo = " ".join(todo.split())
    assert "## Scope and current boundary" in todo
    assert "The current HF milestone is complete" in todo
    assert "there are **no remaining blocking items**".lower() in normalized_todo.lower()
    assert "045bac1b769240facd290e1ac8232e8b1ca39778" in todo
    assert "## Post-release expansion projects" in todo
    assert "- [x]" not in todo
    assert "- [ ]" not in todo
    assert "Dense HF inference PP/TP is closed for the declared scope" in todo
    assert "RWKV-7 HF adapter v0.6.0`: **COMPLETE**" in todo

    status = read("HF_STATUS.md")
    assert "## Completion reporting rule" in status
    assert "no official repository-wide completion percentage" in status
    assert "| HF v0.6 adapter deliverable | **COMPLETE**" in status
    assert "| PP/TP boundary | **PASS for dense HF inference scope**" in status
    assert "4080_7p2b_fp16_state_20260809" in status

    acceptance = read("docs/ACCEPTANCE.md")
    assert "## How to report completion" in acceptance
    assert "current HF milestone is complete" in acceptance
    assert "| HF adapter release scope | **COMPLETE**" in acceptance
    assert "| PP/TP boundary | **PASS for dense HF inference scope**" in acceptance
    assert "4080_7p2b_fp16_state_20260809" in acceptance

    benchmark = read("BENCHMARK.md")
    assert "344.39 tok/s" in benchmark
    assert "12,288/12,288" in benchmark

    bench_index = read("bench/INDEX.md")
    for artifact in (
        "4080_v100_decode_tuning_20260808",
        "4080_b8_projection_bmm_20260809",
        "4080_7p2b_fp16_state_20260809",
    ):
        assert artifact in bench_index, f"current promoted artifact missing: {artifact}"

    project_summary = read("docs/PROJECT_SUMMARY.md")
    results_index = read("docs/RESULTS_INDEX.md")
    for current_doc in (project_summary, results_index):
        assert "045bac1b769240facd290e1ac8232e8b1ca39778" in current_doc
        assert "4080_7p2b_fp16_state_20260809" in current_doc

    changelog = read("CHANGELOG.md")
    assert "## Unreleased" in changelog
    assert "PR #102" in changelog
    assert "yyqdbngt" in changelog

    readme = read("README.md")
    assert "Completion is reported by **named scope**" in readme
    assert "HF adapter `v0.6.0` deliverable is complete" in readme

    readme_zh = read("README_ZH.md")
    assert "RWKV-7 HF Adapter `v0.6.0` 交付范围已经完成" in readme_zh

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
