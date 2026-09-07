#!/usr/bin/env python3
"""Build exact-runtime RWKV-7 CUDA extension wheels from adapter sources."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
GENERATED = HERE / ".generated"
PACKAGE = HERE / "rwkv7_kernels"

PROTOCOL = "rwkv7-kernel-package-v1"
SCHEMA_VERSION = 1

EXTENSION_SPECS = {
    "native_wkv_fp16": {
        "source": "native_wkv_fp16.py",
        "cpp": "_CPP_SOURCE",
        "cuda": "_CUDA_SOURCE",
        "cxx_flags": ["-O3"],
        "nvcc_flags": ["-O3", "--use_fast_math", "--extra-device-vectorization"],
        "arches": {"sm70", "sm75", "sm80", "sm86", "sm89", "sm90", "sm120"},
    },
    "sm70_linear": {
        "source": "sm70_linear.py",
        "cpp": "_CPP_SOURCE",
        "cuda": "_CUDA_SOURCE",
        "cxx_flags": ["-O3"],
        "nvcc_flags": ["-O3", "--use_fast_math"],
        "arches": {"sm70"},
    },
    "sm70_wagv": {
        "source": "sm70_wagv.py",
        "cpp": "_CPP_SOURCE",
        "cuda": "_CUDA_SOURCE",
        "cxx_flags": ["-O3"],
        "nvcc_flags": ["-O3", "--use_fast_math", "--extra-device-vectorization"],
        "arches": {"sm70"},
    },
    "sm70_quant": {
        "source": "sm70_quant.py",
        "cpp": "_CPP",
        "cuda": "_CUDA",
        "cxx_flags": ["-O3"],
        "nvcc_flags": ["-O3", "--use_fast_math", "--extra-device-vectorization"],
        "arches": {"sm70", "sm75"},
    },
    "ada_lora": {
        "source": "ada_lora.py",
        "cpp": "_CPP_SOURCE",
        "cuda": "_CUDA_SOURCE",
        "cxx_flags": ["-O3"],
        "nvcc_flags": ["-O3", "--use_fast_math", "--extra-device-vectorization"],
        "arches": {"sm86", "sm89", "sm120"},
    },
    "ada_sparse_ffn": {
        "source": "ada_sparse_ffn.py",
        "cpp": "_CPP_SOURCE",
        "cuda": "_CUDA_SOURCE",
        "cxx_flags": ["-O3"],
        "nvcc_flags": ["-O3", "--use_fast_math", "--extra-device-vectorization"],
        "arches": {"sm70", "sm86", "sm89", "sm120"},
    },
    "blackwell_norm_mix": {
        "source": "blackwell_norm_mix.py",
        "cpp": "_CPP_SOURCE",
        "cuda": "_CUDA_SOURCE",
        "cxx_flags": ["-O3"],
        "nvcc_flags": ["-O3", "--extra-device-vectorization"],
        "arches": {"sm120"},
    },
}


def _project_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if match is None:
        raise RuntimeError("project version missing from pyproject.toml")
    return match.group(1)


def _literal(source: Path, name: str) -> str:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(
            isinstance(target, ast.Name) and target.id == name for target in targets
        ):
            value = ast.literal_eval(node.value)
            if not isinstance(value, str):
                break
            return value
    raise RuntimeError(f"literal {name} not found in {source}")


def _architectures() -> tuple[list[str], list[str]]:
    raw = os.environ.get("RWKV7_KERNEL_ARCH_LIST", "").strip()
    if not raw:
        raise RuntimeError(
            "RWKV7_KERNEL_ARCH_LIST is required (for example: 7.0 or 8.9)"
        )
    cuda_arches: list[str] = []
    manifest_arches: list[str] = []
    for value in re.split(r"[;, ]+", raw):
        if not re.fullmatch(r"\d+\.\d+", value):
            raise RuntimeError(f"invalid CUDA architecture {value!r}")
        cuda_arches.append(value)
        major, minor = value.split(".")
        manifest_arches.append(f"sm{major}{minor}")
    return cuda_arches, manifest_arches


def _selected_extensions(manifest_arches: list[str]) -> list[str]:
    explicit = os.environ.get("RWKV7_KERNEL_EXTENSIONS", "").strip()
    if explicit:
        selected = [value.strip() for value in explicit.split(",") if value.strip()]
    else:
        selected = [
            name
            for name, spec in EXTENSION_SPECS.items()
            if set(manifest_arches) & set(spec["arches"])
        ]
    unknown = sorted(set(selected) - set(EXTENSION_SPECS))
    if unknown:
        raise RuntimeError(f"unknown kernel extensions: {unknown}")
    incompatible = {
        name: sorted(set(manifest_arches) - set(EXTENSION_SPECS[name]["arches"]))
        for name in selected
        if not set(manifest_arches) <= set(EXTENSION_SPECS[name]["arches"])
    }
    if incompatible:
        raise RuntimeError(
            f"extensions do not support requested architectures: {incompatible}"
        )
    if not selected:
        raise RuntimeError("no extensions selected for the requested architecture")
    return selected


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.STDOUT
        ).strip()
    except Exception:
        return None


def _build_manifest(
    *,
    distribution: str,
    kernel_version: str,
    adapter_version: str,
    cuda_arches: list[str],
    manifest_arches: list[str],
    selected: list[str],
) -> dict[str, Any]:
    torch_version = str(torch.__version__)
    torch_series = ".".join(torch_version.split("+", 1)[0].split(".")[:2])
    adapter_minor = ".".join(adapter_version.split(".")[:2])
    next_minor = (
        f"{adapter_version.split('.')[0]}.{int(adapter_version.split('.')[1]) + 1}"
    )
    nvcc = shutil.which("nvcc")
    digest = hashlib.sha256()
    for name in selected:
        path = ROOT / "rwkv7_hf" / str(EXTENSION_SPECS[name]["source"])
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "distribution": distribution,
        "version": kernel_version,
        "adapter_specifier": f">={adapter_version},<{next_minor}",
        "adapter_series": adapter_minor,
        "python_version": platform.python_version(),
        "python_series": f"{sys.version_info.major}.{sys.version_info.minor}",
        "python_abi": getattr(sys.implementation, "cache_tag", None),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "torch_version": torch_version,
        "torch_series": torch_series,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "torch_cxx11_abi": bool(torch._C._GLIBCXX_USE_CXX11_ABI),
        "architectures": manifest_arches,
        "torch_cuda_arch_list": cuda_arches,
        "extensions": {name: f"rwkv7_kernels._C.{name}" for name in selected},
        "source_commit": os.environ.get("RWKV7_KERNEL_SOURCE_COMMIT")
        or _command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
        or "working-tree",
        "source_tree_sha256": digest.hexdigest(),
        "nvcc": nvcc,
        "nvcc_version": _command_output([nvcc, "--version"]) if nvcc else None,
    }


def _prepare_sources(selected: list[str]) -> list[CUDAExtension]:
    GENERATED.mkdir(parents=True, exist_ok=True)
    extensions = []
    for name in selected:
        spec = EXTENSION_SPECS[name]
        source = ROOT / "rwkv7_hf" / str(spec["source"])
        # PyTorch's Ninja emitter derives object names from source stems, so
        # the binding and CUDA translation units must not share one stem.
        cpp = GENERATED / f"{name}_bind.cpp"
        cuda = GENERATED / f"{name}_kernel.cu"
        cpp.write_text(_literal(source, str(spec["cpp"])), encoding="utf-8")
        cuda.write_text(_literal(source, str(spec["cuda"])), encoding="utf-8")
        extensions.append(
            CUDAExtension(
                name=f"rwkv7_kernels._C.{name}",
                sources=[str(cpp), str(cuda)],
                extra_compile_args={
                    "cxx": list(spec["cxx_flags"]),
                    "nvcc": list(spec["nvcc_flags"]),
                },
            )
        )
    return extensions


cuda_arches, manifest_arches = _architectures()
os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(cuda_arches)
selected_extensions = _selected_extensions(manifest_arches)
adapter_version = _project_version()
cuda_version = str(getattr(torch.version, "cuda", "") or "").replace(".", "")
if not cuda_version:
    raise RuntimeError("the kernel wheel must be built with a CUDA-enabled PyTorch")
torch_series = "".join(str(torch.__version__).split("+", 1)[0].split(".")[:2])
arch_tag = manifest_arches[0] if len(manifest_arches) == 1 else "fatbin"
kernel_version = os.environ.get(
    "RWKV7_KERNEL_VERSION",
    f"{adapter_version}+cu{cuda_version}.torch{torch_series}.{arch_tag}",
)
distribution = os.environ.get("RWKV7_KERNEL_DIST_NAME", "rwkv7-kernels")
manifest = _build_manifest(
    distribution=distribution,
    kernel_version=kernel_version,
    adapter_version=adapter_version,
    cuda_arches=cuda_arches,
    manifest_arches=manifest_arches,
    selected=selected_extensions,
)
(PACKAGE / "_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)

setup(
    name=distribution,
    version=kernel_version,
    description="Prebuilt native CUDA kernels for rwkv7-hf",
    long_description=(HERE / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    license="MIT",
    license_files=["LICENSE", "LICENSES/Apache-2.0.txt"],
    python_requires=">=3.10",
    packages=find_packages(),
    package_data={"rwkv7_kernels": ["_manifest.json"]},
    ext_modules=_prepare_sources(selected_extensions),
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    zip_safe=False,
)
