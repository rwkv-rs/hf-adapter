# coding=utf-8
"""Process-safe environment setup for lazy CUDA extension builds.

``torch.utils.cpp_extension`` reads several process-global environment
variables while importing/building an extension.  Leaving card-specific
values behind makes the next extension build inherit the first GPU's CUDA
architecture, and independent module locks do not prevent two builders from
racing.  Keep all temporary edits under one package-wide lock and restore the
caller's environment exactly after the build finishes.
"""
from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


_BUILD_ENV_LOCK = threading.RLock()
_MISSING = object()
_MANAGED_KEYS = (
    "PATH",
    "CUDA_HOME",
    "TORCH_CUDA_ARCH_LIST",
    "LIBRARY_PATH",
    "LD_LIBRARY_PATH",
)


def _prepend_env_path(name: str, value: str) -> None:
    items = [item for item in os.environ.get(name, "").split(os.pathsep) if item]
    if value not in items:
        os.environ[name] = os.pathsep.join([value, *items])


@contextmanager
def cuda_extension_build_environment(
    *,
    arch_list: str,
    add_runtime_library: bool = True,
) -> Iterator[Path | None]:
    """Temporarily prepare venv CUDA/Ninja paths for one extension build.

    The requested architecture is forced for this build so a value left by a
    different card cannot select the wrong binary. The caller's original
    ``TORCH_CUDA_ARCH_LIST`` is restored afterward. The yielded path is the
    wheel-provided CUDA runtime directory when present.
    """

    with _BUILD_ENV_LOCK:
        previous = {
            key: os.environ.get(key, _MISSING)
            for key in _MANAGED_KEYS
        }
        try:
            # ``absolute`` preserves a venv path where ``resolve`` may jump to
            # /usr/bin and hide the colocated Ninja/NVCC executables.
            python_bin = Path(sys.executable).absolute().parent
            # A thin venv can symlink the base interpreter while leaving build
            # tools installed beside that base interpreter.  Keep the venv
            # first, but expose the base bin as a deployment-safe fallback.
            base_bin = Path(sys.base_prefix).absolute() / "bin"
            if base_bin != python_bin and base_bin.is_dir():
                _prepend_env_path("PATH", str(base_bin))
            _prepend_env_path("PATH", str(python_bin))
            if "CUDA_HOME" not in os.environ:
                nvcc = next(
                    (
                        candidate / "nvcc"
                        for candidate in (python_bin, base_bin)
                        if (candidate / "nvcc").exists()
                    ),
                    None,
                )
                if nvcc is not None:
                    os.environ["CUDA_HOME"] = str(nvcc.parent.parent)
            os.environ["TORCH_CUDA_ARCH_LIST"] = str(arch_list)

            runtime_lib = (
                Path(sys.prefix)
                / "lib"
                / f"python{sys.version_info.major}.{sys.version_info.minor}"
                / "site-packages"
                / "nvidia"
                / "cuda_runtime"
                / "lib"
            )
            if not runtime_lib.is_dir():
                runtime_lib = None
            if add_runtime_library and runtime_lib is not None:
                _prepend_env_path("LIBRARY_PATH", str(runtime_lib))
                _prepend_env_path("LD_LIBRARY_PATH", str(runtime_lib))
            yield runtime_lib
        finally:
            for key, value in previous.items():
                if value is _MISSING:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = str(value)
