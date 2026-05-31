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
    expected_audit_dates["HF_STATUS.md"] = "2026-08-20"
    expected_audit_dates["HF_TODO.md"] = "2026-08-20"
    expected_audit_dates["docs/RESULTS_INDEX.md"] = "2026-08-19"
    for relative, expected_date in expected_audit_dates.items():
        assert expected_date in read(relative), relative

    for path in sorted((ROOT / "docs/plans").glob("*.md")):
        text = path.read_text(encoding="utf-8").lower()
        assert "historical" in text, (
            f"plan lacks lifecycle banner: {path.relative_to(ROOT)}"
        )

    status = read("HF_STATUS.md")
    assert "HF v0.8 adapter deliverable | **COMPLETE**" in status
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
    assert 'python -m pip install "rwkv7-hf==0.8.0"' in readme

    readme_zh = read("README_ZH.md")
    assert 'python -m pip install "rwkv7-hf==0.8.0"' in readme_zh

    ordinary_user_docs = (
        "README.md",
        "README_ZH.md",
        "docs/USER_GUIDE.md",
        "docs/USER_GUIDE_ZH.md",
        "docs/PUBLISHED_MODELS.md",
        "docs/PUBLISHED_MODELS_ZH.md",
    )
    for relative in ordinary_user_docs:
        text = read(relative)
        assert 'rwkv7-hf==0.8.0' in text, relative
        assert "rwkv7-hf-doctor" in text, relative
        assert "rwkv7-hf-smoke" in text, relative
        assert "wangyue114514/rwkv7-g1d-0.1b-hf" in text, relative

    for relative in (
        "README.md",
        "README_ZH.md",
        "docs/USER_GUIDE.md",
        "docs/USER_GUIDE_ZH.md",
        "docs/KERNEL_WHEELS.md",
        "docs/KERNEL_WHEELS_ZH.md",
    ):
        text = read(relative)
        assert "rwkv7-hf-kernels recommend" in text, relative
        assert "rwkv7-hf-kernels install" in text, relative

    english_first_run = read("docs/USER_GUIDE.md").split(
        "## 2. Optional: get and convert a model", 1
    )[0]
    chinese_first_run = read("docs/USER_GUIDE_ZH.md").split(
        "## 2. 可选：下载并转换模型", 1
    )[0]
    assert "pip install -e" not in english_first_run
    assert "pip install -e" not in chinese_first_run

    ai_setup = read("docs/AI_ASSISTED_SETUP.md")
    assert "从 PyPI 安装并运行公开 0.1B" in ai_setup
    assert "rwkv7-hf-smoke" in ai_setup
    assert "TASK_ID=first-run" in ai_setup

    print("DOCUMENT FRESHNESS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
