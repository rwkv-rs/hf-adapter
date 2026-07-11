#!/usr/bin/env python3
"""Prove the CUDA compiler used by lazy native-extension validation.

This is an infrastructure gate, not a model benchmark.  It binds the PyTorch
CUDA runtime, ``CUDA_HOME`` compiler, provenance file, target SM, and a real
``nvcc`` object compilation into one small JSON report before GPU work starts.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from common import environment, git_revision, sha256_file


_CUDA_RELEASE = re.compile(r"release\s+(\d+)\.(\d+)", re.IGNORECASE)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--arch",
        help="nvcc target such as sm_89; defaults to the active GPU capability",
    )
    parser.add_argument("--code-sha")
    return parser.parse_args()


def cuda_major_minor(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    match = re.search(r"(\d+)\.(\d+)", value)
    return (int(match.group(1)), int(match.group(2))) if match else None


def nvcc_major_minor(lines: list[str] | None) -> tuple[int, int] | None:
    if not lines:
        return None
    match = _CUDA_RELEASE.search("\n".join(lines))
    return (int(match.group(1)), int(match.group(2))) if match else None


def detected_arch() -> str | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(0)
        return f"sm_{major}{minor}"
    except Exception:
        return None


def compile_probe(nvcc: Path, arch: str) -> dict[str, Any]:
    source = (
        'extern "C" __global__ void rwkv7_toolchain_probe(float *x) {\n'
        "  if (blockIdx.x == 0 && threadIdx.x == 0) x[0] += 1.0f;\n"
        "}\n"
    )
    with tempfile.TemporaryDirectory(prefix="rwkv7-cuda-preflight-") as raw:
        root = Path(raw)
        source_path = root / "probe.cu"
        object_path = root / "probe.o"
        source_path.write_text(source, encoding="utf-8")
        command = [
            str(nvcc),
            "-arch",
            arch,
            "-c",
            str(source_path),
            "-o",
            str(object_path),
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        return {
            "command": [str(nvcc), "-arch", arch, "-c", "probe.cu", "-o", "probe.o"],
            "source_sha256": sha256_file(source_path),
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "object_bytes": object_path.stat().st_size
            if object_path.is_file()
            else None,
            "object_sha256": sha256_file(object_path)
            if object_path.is_file()
            else None,
        }


def build_report(*, arch: str | None, code_sha: str | None) -> dict[str, Any]:
    runtime = environment()
    toolkit = runtime.get("cuda_toolkit") or {}
    torch_cuda = cuda_major_minor(runtime.get("cuda"))
    nvcc_cuda = nvcc_major_minor(toolkit.get("nvcc_version"))
    effective_arch = arch or detected_arch()
    failures: list[str] = []
    if torch_cuda is None:
        failures.append("PyTorch CUDA runtime version is unavailable")
    if nvcc_cuda is None:
        failures.append("nvcc CUDA release is unavailable")
    if torch_cuda is not None and nvcc_cuda is not None and torch_cuda != nvcc_cuda:
        failures.append(
            f"PyTorch CUDA {torch_cuda[0]}.{torch_cuda[1]} does not match "
            f"nvcc {nvcc_cuda[0]}.{nvcc_cuda[1]}"
        )
    if not (toolkit.get("provenance") or {}).get("sha256"):
        failures.append("CUDA toolkit PROVENANCE.txt identity is unavailable")
    if not effective_arch or not re.fullmatch(r"sm_\d{2,3}", effective_arch):
        failures.append("a valid CUDA target architecture is unavailable")

    compilation = None
    nvcc_value = toolkit.get("nvcc")
    if not failures and nvcc_value:
        compilation = compile_probe(Path(nvcc_value), effective_arch)
        if compilation["exit_code"] != 0 or not compilation["object_sha256"]:
            failures.append("nvcc compile probe failed")

    return {
        "schema": "rwkv7-cuda-toolchain-preflight-v1",
        "status": "passed" if not failures else "failed",
        "code_sha": code_sha or git_revision(Path(__file__).resolve().parents[1]),
        "environment": runtime,
        "torch_cuda_major_minor": list(torch_cuda) if torch_cuda else None,
        "nvcc_cuda_major_minor": list(nvcc_cuda) if nvcc_cuda else None,
        "target_arch": effective_arch,
        "compilation": compilation,
        "failures": failures,
    }


def main() -> int:
    args = arguments()
    report = build_report(arch=args.arch, code_sha=args.code_sha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "status": report["status"]}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
