# coding=utf-8
"""Optional Huawei Ascend runtime helpers for the native RWKV-7 HF backend.

The adapter deliberately does not import :mod:`torch_npu` at package import
 time.  Calling :func:`enable_ascend` registers the private-use NPU device and
selects the conservative FLA/CUDA-free native PyTorch route.  Standard HF
``Auto*`` loading remains unchanged; models can then be moved with
``model.to('npu:0')``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import import_module, metadata
from functools import lru_cache
import os
from typing import Any


_FALSE = {"0", "false", "no", "off"}
_TRUE = {"1", "true", "yes", "on"}

VALIDATED_ASCEND_DEVICE = "Ascend910B3"
VALIDATED_ASCEND_CANN_VERSION = "8.5.0"
VALIDATED_ASCEND_TORCH_VERSION = "2.9.0+cpu"
VALIDATED_ASCEND_TORCH_NPU_VERSION = "2.9.0"


@dataclass(frozen=True)
class AscendRuntimeInfo:
    available: bool
    device_count: int
    device_name: str | None
    torch_version: str | None
    torch_npu_version: str | None
    cann_version: str | None
    device: str
    backend: str
    validated_stack: bool
    validation_status: str
    validation_reason: str
    allow_unvalidated: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


@lru_cache(maxsize=1)
def _import_torch_npu_cached():
    try:
        return import_module("torch_npu")
    except (ImportError, OSError, RuntimeError):
        return None


def import_torch_npu(*, required: bool = False):
    """Import torch_npu lazily and return it, or ``None`` when unavailable."""

    module = _import_torch_npu_cached()
    if module is None and required:
        raise RuntimeError(
            "Huawei Ascend was requested but torch_npu could not be imported. "
            "Install a torch/torch_npu pair compatible with the host CANN release."
        ) from None
    return module


def ascend_available() -> bool:
    if import_torch_npu(required=False) is None:
        return False
    import torch

    npu = getattr(torch, "npu", None)
    try:
        return bool(npu is not None and npu.is_available())
    except Exception:
        return False


def normalize_ascend_device_name(value: str | None) -> str:
    """Normalize spelling only; never use substring/family matching."""

    compact = "".join(character for character in str(value or "") if character.isalnum())
    if compact.casefold() == "ascend910b3":
        return VALIDATED_ASCEND_DEVICE
    return compact


def validate_ascend_stack(
    *,
    device_name: str | None,
    torch_version: str | None,
    torch_npu_version: str | None,
    cann_version: str | None,
) -> tuple[bool, str]:
    """Validate the one production evidence row using exact equality."""

    observed = {
        "device_name": normalize_ascend_device_name(device_name),
        "cann_version": str(cann_version or ""),
        "torch_version": str(torch_version or ""),
        "torch_npu_version": str(torch_npu_version or ""),
    }
    expected = {
        "device_name": VALIDATED_ASCEND_DEVICE,
        "cann_version": VALIDATED_ASCEND_CANN_VERSION,
        "torch_version": VALIDATED_ASCEND_TORCH_VERSION,
        "torch_npu_version": VALIDATED_ASCEND_TORCH_NPU_VERSION,
    }
    mismatches = [
        f"{key}={observed[key]!r} (expected {expected[key]!r})"
        for key in expected
        if observed[key] != expected[key]
    ]
    if mismatches:
        return False, "; ".join(mismatches)
    return True, "exact validated Ascend 910B3 software stack"


def allow_unvalidated_ascend() -> bool:
    return os.environ.get("RWKV7_ALLOW_UNVALIDATED_ASCEND", "0").strip().lower() in _TRUE


def detect_cann_version(torch_npu_module=None) -> str | None:
    """Return an explicit/runtime CANN version without family guessing."""

    explicit = os.environ.get("ASCEND_TOOLKIT_VERSION") or _cann_version_from_home()
    if explicit:
        return str(explicit)
    module = torch_npu_module
    if module is None:
        module = import_torch_npu(required=False)
    accessors = (
        getattr(getattr(module, "npu", None), "get_cann_version", None),
        getattr(getattr(module, "_C", None), "_npu_getCANNVersion", None),
    )
    for accessor in accessors:
        if callable(accessor):
            try:
                value = accessor()
            except (AttributeError, RuntimeError, TypeError):
                continue
            if value is not None:
                return str(value)
    return None


def configure_ascend_defaults(*, backend: str = "eager", overwrite: bool = False) -> dict[str, str]:
    """Select a fail-closed native backend suitable for torch_npu.

    ``eager`` is the portable default. ``native_jit`` is the packed pure-torch
    decode route and is useful after a card-local correctness smoke. CUDA graph,
    Triton, FLA, bitsandbytes and CUDA extensions are never enabled here.
    Explicit user environment values win unless ``overwrite=True``.
    """

    aliases = {"torch": "eager", "jit": "native_jit", "auto": "auto"}
    backend = aliases.get(str(backend).strip().lower(), str(backend).strip().lower())
    if backend not in {"eager", "native_jit", "auto"}:
        raise ValueError("Ascend backend must be eager, native_jit, or auto")
    values = {
        "RWKV7_NATIVE_MODEL": "1",
        "RWKV7_NATIVE_MODEL_BACKEND": backend,
        "RWKV7_NATIVE_MODEL_JIT": "0" if backend == "eager" else "1",
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


def enable_ascend(
    device: int | str = 0,
    *,
    backend: str = "eager",
    required: bool = True,
    set_device: bool = True,
) -> AscendRuntimeInfo:
    """Register torch_npu, configure safe defaults and optionally select NPU."""

    configure_ascend_defaults(backend=backend)
    torch_npu_module = import_torch_npu(required=required)
    import torch

    available = ascend_available()
    if required and not available:
        raise RuntimeError("torch_npu imported, but no Huawei Ascend NPU is available")
    text = str(device)
    if text.isdigit():
        text = f"npu:{int(text)}"
    elif text == "npu":
        text = "npu:0"
    if not text.startswith("npu:"):
        raise ValueError(f"Ascend device must be npu:<index>; got {device!r}")
    index = int(text.split(":", 1)[1])
    count = int(torch.npu.device_count()) if available else 0
    if available and not 0 <= index < count:
        raise ValueError(f"Ascend device index {index} is outside visible device count {count}")
    name = str(torch.npu.get_device_name(index)) if available else None
    torch_version = getattr(torch, "__version__", None)
    torch_npu_version = _package_version("torch-npu")
    cann_version = detect_cann_version(torch_npu_module)
    validated, validation_reason = validate_ascend_stack(
        device_name=name,
        torch_version=torch_version,
        torch_npu_version=torch_npu_version,
        cann_version=cann_version,
    )
    override = allow_unvalidated_ascend()
    if available and not validated and not override:
        raise RuntimeError(
            "unvalidated Huawei Ascend production stack: "
            f"{validation_reason}. Set RWKV7_ALLOW_UNVALIDATED_ASCEND=1 only "
            "for an explicitly reported experimental run."
        )
    if available and set_device:
        torch.npu.set_device(text)
    if not available:
        validation_status = "unavailable"
    elif validated:
        validation_status = "validated"
    else:
        validation_status = "unvalidated_override"
    return AscendRuntimeInfo(
        available=available,
        device_count=count,
        device_name=name,
        torch_version=torch_version,
        torch_npu_version=torch_npu_version,
        cann_version=cann_version,
        device=text,
        backend=os.environ["RWKV7_NATIVE_MODEL_BACKEND"],
        validated_stack=validated,
        validation_status=validation_status,
        validation_reason=validation_reason,
        allow_unvalidated=override,
    )


def _cann_version_from_home() -> str | None:
    home = os.environ.get("ASCEND_HOME_PATH") or os.environ.get("ASCEND_TOOLKIT_HOME")
    if not home:
        return None
    base = os.path.basename(home.rstrip("/"))
    return base.removeprefix("cann-") or None


def synchronize(device: int | str | None = None) -> None:
    """Synchronize the current/selected NPU without exposing torch_npu globally."""

    import_torch_npu(required=True)
    import torch

    torch.npu.synchronize(device)


def memory_stats(device: int | str | None = None) -> dict[str, int]:
    """Return stable allocated/reserved NPU memory counters when available."""

    if not ascend_available():
        return {}
    import torch

    result: dict[str, int] = {}
    for label, name in (
        ("allocated_bytes", "memory_allocated"),
        ("reserved_bytes", "memory_reserved"),
        ("max_allocated_bytes", "max_memory_allocated"),
        ("max_reserved_bytes", "max_memory_reserved"),
    ):
        fn = getattr(torch.npu, name, None)
        if fn is not None:
            try:
                result[label] = int(fn(device)) if device is not None else int(fn())
            except (RuntimeError, TypeError):
                pass
    return result
