# coding=utf-8
"""Optional MetaX C500 runtime helpers for the native RWKV-7 HF backend.

MetaX exposes its MXMACA runtime through PyTorch's CUDA-compatible API.  That
compatibility is useful, but it also means a C500 can otherwise be mistaken for
an unrelated vendor family by capability-only routing.  This module performs exact
device/stack admission and selects the conservative native eager path that was
validated by ``123123213weqw/rwkv7-metax-c500``.

No MetaX package is imported at package-import time.  CPU, NVIDIA CUDA, ROCm,
MUSA, Ascend and Apple users therefore keep their existing import behavior.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import os
import re
import shutil
import subprocess
from typing import Any


SOURCE_REPOSITORY = "123123213weqw/rwkv7-metax-c500"
SOURCE_COMMIT = "f2653e20250821ec48534e5e08b07d59effb985c"

VALIDATED_METAX_DEVICE = "MetaX C500"
VALIDATED_MXMACA_VERSION = "3.5.3.20"
VALIDATED_METAX_TORCH_VERSION = "2.8.0+metax3.5.3.9"
VALIDATED_METAX_TORCH_CUDA_VERSION = "11.6"
VALIDATED_METAX_DRIVER_VERSION = "3.8.30"
VALIDATED_MX_SMI_VERSION = "2.2.12"

_TRUE = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class MetaXRuntimeInfo:
    available: bool
    device_count: int
    device_name: str | None
    torch_version: str | None
    torch_cuda_version: str | None
    mxmaca_version: str | None
    driver_version: str | None
    mx_smi_version: str | None
    device: str
    backend: str
    validated_stack: bool
    validation_status: str
    validation_reason: str
    allow_unvalidated: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_metax_device_name(value: str | None) -> str:
    """Normalize harmless separators without accepting adjacent products."""

    tokens = tuple(
        part
        for part in re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).split()
        if part
    )
    if tokens == ("metax", "c500"):
        return VALIDATED_METAX_DEVICE
    return " ".join(tokens)


def is_metax_c500_name(value: str | None) -> bool:
    return normalize_metax_device_name(value) == VALIDATED_METAX_DEVICE


def allow_unvalidated_metax() -> bool:
    return os.environ.get("RWKV7_ALLOW_UNVALIDATED_METAX", "0").strip().lower() in _TRUE


@lru_cache(maxsize=1)
def _mx_smi_output() -> str:
    executable = shutil.which("mx-smi")
    if executable is None:
        return ""
    try:
        completed = subprocess.run(
            [executable],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part.strip()
    )[-12000:]


def _metadata_value(
    explicit_env: str,
    pattern: str,
    *,
    output: str | None = None,
) -> str | None:
    explicit = os.environ.get(explicit_env)
    if explicit:
        return explicit.strip()
    match = re.search(
        pattern, _mx_smi_output() if output is None else output, re.IGNORECASE
    )
    return match.group(1).strip() if match else None


def detect_mxmaca_version(*, output: str | None = None) -> str | None:
    return _metadata_value(
        "MXMACA_VERSION",
        r"\bMACA\s+Version\s*:\s*([^\s|]+)",
        output=output,
    )


def detect_metax_driver_version(*, output: str | None = None) -> str | None:
    return _metadata_value(
        "METAX_DRIVER_VERSION",
        r"Kernel\s+Mode\s+Driver\s+Version\s*:\s*([^\s|]+)",
        output=output,
    )


def detect_mx_smi_version(*, output: str | None = None) -> str | None:
    return _metadata_value(
        "MX_SMI_VERSION",
        r"mx-smi\s+version\s*:\s*([^\s|]+)",
        output=output,
    )


def validate_metax_stack(
    *,
    device_name: str | None,
    torch_version: str | None,
    torch_cuda_version: str | None,
    mxmaca_version: str | None,
) -> tuple[bool, str]:
    """Validate the exact C500 software row with no family inheritance."""

    observed = {
        "device_name": normalize_metax_device_name(device_name),
        "mxmaca_version": str(mxmaca_version or ""),
        "torch_version": str(torch_version or ""),
        "torch_cuda_version": str(torch_cuda_version or ""),
    }
    expected = {
        "device_name": VALIDATED_METAX_DEVICE,
        "mxmaca_version": VALIDATED_MXMACA_VERSION,
        "torch_version": VALIDATED_METAX_TORCH_VERSION,
        "torch_cuda_version": VALIDATED_METAX_TORCH_CUDA_VERSION,
    }
    mismatches = [
        f"{key}={observed[key]!r} (expected {expected[key]!r})"
        for key in expected
        if observed[key] != expected[key]
    ]
    if mismatches:
        return False, "; ".join(mismatches)
    return True, "exact validated MetaX C500 MXMACA software stack"


def metax_available(device: int | str = 0) -> bool:
    """Return whether an exact C500 is visible through ``torch.cuda``."""

    try:
        import torch

        if not torch.cuda.is_available():
            return False
        index = _device_index(device, torch_module=torch)
        return 0 <= index < int(torch.cuda.device_count()) and is_metax_c500_name(
            torch.cuda.get_device_name(index)
        )
    except Exception:
        return False


def _device_index(device: int | str, *, torch_module) -> int:
    if isinstance(device, bool):
        raise ValueError("MetaX device index cannot be boolean")
    if isinstance(device, int):
        return int(device)
    text = str(device).strip().lower()
    if text in {"metax", "cuda"}:
        return 0
    if text.isdigit():
        return int(text)
    if text.startswith("metax:"):
        return int(text.split(":", 1)[1])
    resolved = torch_module.device(text)
    if resolved.type != "cuda":
        raise ValueError(f"MetaX uses cuda:<index>; got {device!r}")
    return int(resolved.index or 0)


def configure_metax_defaults(*, overwrite: bool = False) -> dict[str, str]:
    """Select the validated FLA/Triton/CUDA-extension-free C500 route."""

    values = {
        "RWKV7_NATIVE_MODEL": "1",
        "RWKV7_NATIVE_MODEL_BACKEND": "eager",
        "RWKV7_NATIVE_MODEL_JIT": "0",
        "RWKV7_FAST_FORWARD": "0",
        "RWKV7_FAST_CACHE": "0",
        "RWKV7_FAST_PREFILL": "0",
        "RWKV7_NATIVE_GRAPH": "0",
        "RWKV7_NATIVE_PREFILL_GRAPH": "0",
    }
    for key, value in values.items():
        if overwrite or key not in os.environ:
            os.environ[key] = value
    return {key: os.environ[key] for key in values}


def enable_metax(
    device: int | str = 0,
    *,
    required: bool = True,
    set_device: bool = True,
) -> MetaXRuntimeInfo:
    """Validate a C500 and configure the conservative native HF backend."""

    configure_metax_defaults()
    import torch

    index = _device_index(device, torch_module=torch)
    cuda_available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count()) if cuda_available else 0
    if required and not cuda_available:
        raise RuntimeError("MetaX C500 was requested, but torch.cuda is unavailable")
    if cuda_available and not 0 <= index < count:
        raise ValueError(
            f"MetaX device index {index} is outside visible device count {count}"
        )
    name = str(torch.cuda.get_device_name(index)) if cuda_available else None
    available = bool(cuda_available and is_metax_c500_name(name))
    if required and not available:
        raise RuntimeError(f"expected exact MetaX C500, got {name!r}")

    output = _mx_smi_output()
    torch_version = str(getattr(torch, "__version__", "")) or None
    torch_cuda_version = (
        str(getattr(getattr(torch, "version", None), "cuda", "") or "") or None
    )
    mxmaca_version = detect_mxmaca_version(output=output)
    validated, reason = validate_metax_stack(
        device_name=name,
        torch_version=torch_version,
        torch_cuda_version=torch_cuda_version,
        mxmaca_version=mxmaca_version,
    )
    override = allow_unvalidated_metax()
    if available and not validated and not override:
        raise RuntimeError(
            "unvalidated MetaX C500 production stack: "
            f"{reason}. Set RWKV7_ALLOW_UNVALIDATED_METAX=1 only for an "
            "explicitly reported experimental run."
        )
    if available and set_device:
        torch.cuda.set_device(index)
    if not available:
        status = "unavailable"
    elif validated:
        status = "validated"
    else:
        status = "unvalidated_override"
    return MetaXRuntimeInfo(
        available=available,
        device_count=count,
        device_name=name,
        torch_version=torch_version,
        torch_cuda_version=torch_cuda_version,
        mxmaca_version=mxmaca_version,
        driver_version=detect_metax_driver_version(output=output),
        mx_smi_version=detect_mx_smi_version(output=output),
        device=f"cuda:{index}",
        backend=os.environ["RWKV7_NATIVE_MODEL_BACKEND"],
        validated_stack=validated,
        validation_status=status,
        validation_reason=reason,
        allow_unvalidated=override,
    )


def synchronize(device: int | str | None = None) -> None:
    import torch

    resolved = None if device is None else _device_index(device, torch_module=torch)
    torch.cuda.synchronize(resolved)


def memory_stats(device: int | str | None = None) -> dict[str, int]:
    requested = 0 if device is None else device
    if not metax_available(requested):
        return {}
    import torch

    resolved = _device_index(requested, torch_module=torch)
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(resolved)),
        "reserved_bytes": int(torch.cuda.memory_reserved(resolved)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(resolved)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(resolved)),
    }


__all__ = [
    "MetaXRuntimeInfo",
    "SOURCE_COMMIT",
    "SOURCE_REPOSITORY",
    "allow_unvalidated_metax",
    "configure_metax_defaults",
    "detect_metax_driver_version",
    "detect_mx_smi_version",
    "detect_mxmaca_version",
    "enable_metax",
    "is_metax_c500_name",
    "memory_stats",
    "metax_available",
    "normalize_metax_device_name",
    "synchronize",
    "validate_metax_stack",
]
