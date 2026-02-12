# coding=utf-8
"""Lazy torch_musa extension build helpers.

This module preserves the MUSA SDK 4.2.0 / torch_musa 2.5.0 build workaround
validated by KakaruHayate/RWKV-MUSA. It never modifies site-packages and is
import-safe when torch_musa is absent.
"""
from __future__ import annotations

import glob
import os
from typing import Any

_BAD_COMPILER_OPTION = "--compiler-options '-fPIC'"
_GOOD_COMPILER_OPTION = "-fPIC"
_PATCHED_MODULE_IDS: set[int] = set()


def _gcc_includes() -> list[str]:
    """Return an installed matching libstdc++ include set for mcc."""

    candidates = sorted(glob.glob("/usr/include/c++/*"), reverse=True)
    for base in candidates:
        version = os.path.basename(base)
        if not version.isdigit():
            continue
        target = f"/usr/include/x86_64-linux-gnu/c++/{version}"
        if os.path.isdir(base) and os.path.isdir(target):
            return [
                "-isystem",
                base,
                "-isystem",
                target,
                "-isystem",
                os.path.join(base, "backward"),
            ]
    return []


def _musa_extension_module() -> Any:
    try:
        import torch_musa  # noqa: F401
        from torch_musa.utils import musa_extension
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "RWKV-7 MUSA kernels require the Moore Threads torch_musa package"
        ) from exc
    return musa_extension


def _install_ninja_flag_patch(musa_extension: Any) -> None:
    """Patch the torch_musa 2.5.0 JIT-only nvcc-style flag mismatch."""

    module_id = id(musa_extension)
    if module_id in _PATCHED_MODULE_IDS:
        return
    original = getattr(musa_extension, "_write_ninja_file", None)
    if original is None:
        raise RuntimeError(
            "unsupported torch_musa extension API: _write_ninja_file is missing"
        )

    def patched_write(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        with open(path, "r", encoding="utf-8") as handle:
            content = handle.read()
        if _BAD_COMPILER_OPTION in content:
            content = content.replace(_BAD_COMPILER_OPTION, _GOOD_COMPILER_OPTION)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(content)
        return result

    musa_extension._write_ninja_file = patched_write
    _PATCHED_MODULE_IDS.add(module_id)


def load_musa_inline(
    name: str,
    cpp_sources,
    musa_sources,
    functions,
    *,
    extra_musa_cflags=None,
    extra_cflags=None,
    verbose: bool = False,
):
    """Build an inline MUSA extension through the validated torch_musa path."""

    musa_extension = _musa_extension_module()
    _install_ninja_flag_patch(musa_extension)
    gcc_includes = _gcc_includes()
    return musa_extension.load_inline(
        name=name,
        cpp_sources=cpp_sources,
        musa_sources=musa_sources,
        functions=functions,
        extra_cflags=(extra_cflags or ["-O3"]) + gcc_includes,
        extra_musa_cflags=(extra_musa_cflags or ["-O3"]) + gcc_includes,
        verbose=verbose,
    )


__all__ = ["load_musa_inline"]
