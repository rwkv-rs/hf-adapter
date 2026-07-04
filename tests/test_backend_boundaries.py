#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "rwkv7_hf"

# Keep exact card/chip names in validation evidence, docs, scripts, and the
# centralized policy table.  Core model files should branch on capabilities or
# normalized policy families instead of specific SKUs.
DEVICE_NAME_RE = re.compile(
    r"\b("
    r"V100|A100|A800|A10|A6000|H100|H200|P100|T4|"
    r"RTX\s*(?:20|30|40|50)-?series|RTX\s*(?:4090|4080|4070|5070|5080|5090)|"
    r"GTX\s*(?:10)-?series|GTX\s*(?:1080|1070|1060)|"
    r"Apple\s+M\d+|M\d+\s*(?:Pro|Max|Ultra)|"
    r"Blackwell|Hopper|Volta|Turing|Ampere|Ada|Pascal"
    r")\b",
)

ALLOWED_CORE_FILES = {
    CORE / "kernel_policy.py",
}


def iter_core_python_files() -> list[Path]:
    return sorted(
        path
        for path in CORE.rglob("*.py")
        if "__pycache__" not in path.parts and path not in ALLOWED_CORE_FILES
    )


def test_exact_device_names_stay_out_of_core_model_files() -> None:
    offenders: list[str] = []
    for path in iter_core_python_files():
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            match = DEVICE_NAME_RE.search(line)
            if match:
                offenders.append(f"{path.relative_to(ROOT)}:{line_no}: {match.group(0)}")
    assert not offenders, (
        "Exact hardware names belong in docs/tests/scripts/bench or "
        "rwkv7_hf/kernel_policy.py, not scattered through core model files:\n"
        + "\n".join(offenders)
    )


if __name__ == "__main__":
    test_exact_device_names_stay_out_of_core_model_files()
    print("BACKEND BOUNDARIES PASS")
