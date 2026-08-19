#!/usr/bin/env python3
"""Guard the compact current documentation boundary."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def main() -> int:
    current_docs = (
        "HF_STATUS.md",
        "HF_TODO.md",
        "BENCHMARK.md",
        "docs/ACCEPTANCE.md",
        "docs/HARDWARE_MATRIX.md",
        "docs/PERFORMANCE.md",
        "docs/PROJECT_SUMMARY.md",
        "docs/RESULTS_INDEX.md",
    )
    expected_audit_dates = {relative: "2026-08-17" for relative in current_docs}
    for relative in ("HF_STATUS.md", "docs/RESULTS_INDEX.md"):
        expected_audit_dates[relative] = "2026-08-19"
    for relative, expected_date in expected_audit_dates.items():
        assert expected_date in read(relative), relative

    for path in sorted((ROOT / "docs/plans").glob("*.md")):
        text = path.read_text(encoding="utf-8").lower()
        assert "historical" in text, (
            f"plan lacks lifecycle banner: {path.relative_to(ROOT)}"
        )

    status = read("HF_STATUS.md")
    assert "HF v0.7 adapter deliverable | **COMPLETE**" in status
    assert "FLA wrapper/reference | **COMPATIBILITY AND ORACLE ONLY**" in status

    todo = " ".join(read("HF_TODO.md").split()).lower()
    assert "current hf milestone is complete" in todo
    assert "no remaining blocking items" in todo

    acceptance = read("docs/ACCEPTANCE.md")
    assert "Parameter adjustment" in acceptance
    assert "Native is the retained RWKV performance backend" in acceptance

    benchmark = read("BENCHMARK.md")
    assert "344.39 tok/s" in benchmark
    assert "12,288/12,288" in benchmark
    assert "bench/CURRENT_ARTIFACTS.json" in benchmark

    for relative in current_docs:
        text = read(relative)
        assert "v100_active_b1b8_20260715" not in text
        assert "5070_qwen35_full_fla_bsz8_20260714" not in text

    readme = read("README.md")
    assert "Completion is reported by **named scope**" in readme
    assert 'python -m pip install "rwkv7-hf==0.7.0"' in readme

    readme_zh = read("README_ZH.md")
    assert 'python -m pip install "rwkv7-hf==0.7.0"' in readme_zh

    print("DOCUMENT FRESHNESS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
