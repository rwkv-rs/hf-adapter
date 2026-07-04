#!/usr/bin/env python3
# coding=utf-8
"""Static guards for Apple Silicon / no-FLA packaging and docs."""
from __future__ import annotations

import stat
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_fla_is_optional_dependency() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = data["project"].get("dependencies", [])
    assert "flash-linear-attention" not in deps
    optional = data["project"].get("optional-dependencies", {})
    assert "flash-linear-attention" in optional.get("fla", [])
    assert "flash-linear-attention" in optional.get("cuda", [])


def test_apple_smoke_script_static() -> None:
    script = ROOT / "scripts/run_apple_silicon_smoke.sh"
    assert script.exists()
    assert script.stat().st_mode & stat.S_IXUSR
    subprocess.run(["bash", "-n", str(script)], cwd=ROOT, check=True)
    text = script.read_text(encoding="utf-8")
    assert "RWKV7_NATIVE_MODEL" in text
    assert "PYTORCH_ENABLE_MPS_FALLBACK" in text
    assert "tests/test_apple_silicon_smoke.py" in text


def test_apple_doc_links_entry_points() -> None:
    doc = ROOT / "docs/hardware/APPLE_SILICON.md"
    text = doc.read_text(encoding="utf-8")
    assert "scripts/run_apple_silicon_smoke.sh" in text
    assert "tests/test_apple_silicon_smoke.py" in text
    assert "RafaelUI" in text
    assert "RWKV7_NATIVE_MODEL=1" in text
    assert "MLX" in text
    assert "Metal" in text


def main() -> int:
    test_fla_is_optional_dependency()
    test_apple_smoke_script_static()
    test_apple_doc_links_entry_points()
    print("APPLE SILICON PACKAGING PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
