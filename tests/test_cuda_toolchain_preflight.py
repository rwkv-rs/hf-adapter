from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "preflight_cuda_toolchain",
    ROOT / "evaluation" / "preflight_cuda_toolchain.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(ROOT / "evaluation"))
SPEC.loader.exec_module(MODULE)


def test_cuda_version_parsers():
    assert MODULE.cuda_major_minor("13.0") == (13, 0)
    assert MODULE.cuda_major_minor("12.6+local") == (12, 6)
    assert MODULE.cuda_major_minor(None) is None
    assert MODULE.nvcc_major_minor(
        ["Cuda compilation tools, release 13.0, V13.0.88"]
    ) == (13, 0)
    assert MODULE.nvcc_major_minor(["not a CUDA compiler version"]) is None


def test_compile_probe_records_real_compiler_output(tmp_path: Path):
    nvcc = tmp_path / "nvcc"
    nvcc.write_text(
        "#!/bin/sh\n"
        "while [ $# -gt 0 ]; do\n"
        '  if [ "$1" = -o ]; then shift; printf object > "$1"; fi\n'
        "  shift\n"
        "done\n",
        encoding="utf-8",
    )
    nvcc.chmod(0o755)
    report = MODULE.compile_probe(nvcc, "sm_89")
    assert report["exit_code"] == 0
    assert report["object_bytes"] == 6
    assert len(report["object_sha256"]) == 64
    assert report["command"] == [
        str(nvcc),
        "-arch",
        "sm_89",
        "-c",
        "probe.cu",
        "-o",
        "probe.o",
    ]
