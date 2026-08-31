from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "RELEASE_SOURCE_FREEZE.json"
EXPLICIT_DISTRIBUTION_INPUTS = {
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "kernels/pyproject.toml",
    "kernels/README.md",
    "kernels/LICENSE",
}
FROZEN_SOURCE_ROOTS = (
    ROOT / "rwkv7_hf",
    ROOT / "rwkv7_hf_tools",
    ROOT / "kernels" / "rwkv7_kernels",
)


def _frozen_source_paths() -> set[str]:
    paths = set(EXPLICIT_DISTRIBUTION_INPUTS)
    for source_root in FROZEN_SOURCE_ROOTS:
        paths.update(
            path.relative_to(ROOT).as_posix()
            for path in source_root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        )
    return paths


def test_release_source_freeze_is_complete_and_byte_exact():
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    assert freeze["schema"] == "rwkv7-release-source-freeze-v1"
    assert freeze["version"] == "1.0.0"
    assert freeze["kernel_api_version"] == 4
    assert freeze["plugin_entrypoint"] == "rwkv7_kernels.execute_optional_v4"

    records = freeze["files"]
    assert len(records) == len({record["path"] for record in records})
    assert {record["path"] for record in records} == _frozen_source_paths()
    for record in records:
        path = ROOT / record["path"]
        assert path.is_file(), record["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
