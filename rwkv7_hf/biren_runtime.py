# coding=utf-8
"""Optional Biren BR106M runtime helpers for the native RWKV-7 HF backend.

BIRENSUPA registers a ``supa`` private-use PyTorch device through
``torch_br``.  The validated BR10x path uses BF16 projections, FP32 recurrent
state, decomposed GroupNorm, and eager execution.  Vendor modules are imported
only when this runtime is explicitly probed or enabled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from importlib import import_module, metadata
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any


SOURCE_REPOSITORY = "yyqdbngt/rwkv7-biren-br106m"
SOURCE_COMMIT = "47322bfaffc2e662fa989863c3fda4d74f02fc32"

VALIDATED_BIREN_DEVICE = "Biren106M"
VALIDATED_BIREN_SDK_VERSION = "1.11.0.0.rc2"
VALIDATED_BIREN_DRIVER_VERSION = "1.11.0"
VALIDATED_BIREN_SUPA_VERSION = "1.11"
VALIDATED_BIREN_TORCH_VERSION = "2.9.0+cu128"
VALIDATED_BIREN_TORCH_BR_VERSION = "1.10.0.20900+br1xx"
DEFAULT_BIREN_ENV_SCRIPT = Path(
    "/usr/local/birensupa/sdk/latest/scripts/brsw_set_env.sh"
)
DEFAULT_BIREN_VERSION_FILE = Path(
    "/usr/local/birensupa/sdk/latest/scripts/project_version.txt"
)

_TRUE = {"1", "true", "yes", "on"}


class BirenDTypeError(ValueError):
    """Raised before an unsupported BR10x floating-point path is dispatched."""


@dataclass(frozen=True)
class BirenRuntimePolicy:
    device_type: str = "supa"
    dtype: Any = None
    backend: str = "eager"
    compile_enabled: bool = False


@dataclass(frozen=True)
class BirenRuntimeInfo:
    available: bool
    device_count: int
    device_name: str | None
    torch_version: str | None
    torch_br_version: str | None
    sdk_version: str | None
    driver_version: str | None
    supa_version: str | None
    brsmi_version: str | None
    device: str
    dtype: str
    state_dtype: str
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
def _import_torch_br_cached():
    try:
        return import_module("torch_br")
    except (ImportError, OSError, RuntimeError):
        return None


def import_torch_br(*, required: bool = False):
    """Register the SUPA private-use backend lazily."""

    module = _import_torch_br_cached()
    if module is None and required:
        raise RuntimeError(
            "Biren SUPA was requested but torch_br could not be imported. "
            "Source the matched BIRENSUPA environment before starting Python."
        ) from None
    return module


def normalize_biren_device_name(value: str | None) -> str:
    """Normalize the exact validated product without accepting neighbors."""

    compact = "".join(
        character for character in str(value or "") if character.isalnum()
    )
    if compact.casefold() == "biren106m":
        return VALIDATED_BIREN_DEVICE
    return compact


def is_biren_br106m_name(value: str | None) -> bool:
    return normalize_biren_device_name(value) == VALIDATED_BIREN_DEVICE


def _normalize_biren_dtype(dtype):
    import torch

    if dtype is None or str(dtype).strip().lower() in {"", "auto", "none"}:
        return torch.bfloat16
    if isinstance(dtype, torch.dtype):
        return dtype
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "torch.bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "torch.float16": torch.float16,
        "fp32": torch.float32,
        "float": torch.float32,
        "float32": torch.float32,
        "torch.float32": torch.float32,
    }
    normalized = aliases.get(str(dtype).strip().lower())
    if normalized is None:
        raise BirenDTypeError(f"unsupported Biren dtype: {dtype!r}")
    return normalized


def validate_biren_model_dtype(dtype, *, device_type: str):
    import torch

    normalized = _normalize_biren_dtype(dtype)
    if str(device_type).split(":", 1)[0] == "supa" and normalized == torch.float16:
        raise BirenDTypeError(
            "Biren BR106M does not support the RWKV-7 float16 GEMM path; "
            "load and run the model with torch.bfloat16"
        )
    return normalized


def biren_runtime_policy(dtype=None) -> BirenRuntimePolicy:
    return BirenRuntimePolicy(
        dtype=validate_biren_model_dtype(dtype, device_type="supa")
    )


def validate_biren_forward_dtype(
    dtype,
    *,
    input_device,
    model_device,
    model_dtype=None,
) -> None:
    device_types = {
        getattr(input_device, "type", str(input_device).split(":", 1)[0]),
        getattr(model_device, "type", str(model_device).split(":", 1)[0]),
    }
    if "supa" in device_types:
        validate_biren_model_dtype(dtype, device_type="supa")
        validate_biren_model_dtype(model_dtype, device_type="supa")


def allow_unvalidated_biren() -> bool:
    return os.environ.get("RWKV7_ALLOW_UNVALIDATED_BIREN", "0").strip().lower() in _TRUE


def detect_biren_sdk_version() -> str | None:
    explicit = os.environ.get("BIRENSUPA_SDK_VERSION")
    if explicit:
        return explicit.strip()
    try:
        return DEFAULT_BIREN_VERSION_FILE.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


@lru_cache(maxsize=1)
def _brsmi_output() -> str:
    executable = shutil.which("brsmi")
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


def detect_biren_driver_versions(*, output: str | None = None) -> dict[str, str | None]:
    text = _brsmi_output() if output is None else output
    match = re.search(
        r"BR-SMI\s+(?P<brsmi>\S+).*Driver Version:\s*(?P<driver>\S+)"
        r".*SUPA Version:\s*(?P<supa>\S+)",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    parsed = match.groupdict() if match else {}
    return {
        "brsmi": os.environ.get("BRSMI_VERSION") or parsed.get("brsmi"),
        "driver": os.environ.get("BIREN_DRIVER_VERSION") or parsed.get("driver"),
        "supa": os.environ.get("BIREN_SUPA_VERSION") or parsed.get("supa"),
    }


def validate_biren_stack(
    *,
    device_name: str | None,
    visible_devices: int | None,
    torch_version: str | None,
    torch_br_version: str | None,
    sdk_version: str | None,
    driver_version: str | None,
    supa_version: str | None,
) -> tuple[bool, str]:
    observed = {
        "device_name": normalize_biren_device_name(device_name),
        "visible_devices": str(visible_devices if visible_devices is not None else ""),
        "sdk_version": str(sdk_version or ""),
        "driver_version": str(driver_version or ""),
        "supa_version": str(supa_version or ""),
        "torch_version": str(torch_version or ""),
        "torch_br_version": str(torch_br_version or ""),
    }
    expected = {
        "device_name": VALIDATED_BIREN_DEVICE,
        "visible_devices": "1",
        "sdk_version": VALIDATED_BIREN_SDK_VERSION,
        "driver_version": VALIDATED_BIREN_DRIVER_VERSION,
        "supa_version": VALIDATED_BIREN_SUPA_VERSION,
        "torch_version": VALIDATED_BIREN_TORCH_VERSION,
        "torch_br_version": VALIDATED_BIREN_TORCH_BR_VERSION,
    }
    mismatches = [
        f"{key}={observed[key]!r} (expected {expected[key]!r})"
        for key in expected
        if observed[key] != expected[key]
    ]
    if mismatches:
        return False, "; ".join(mismatches)
    return True, "exact validated Biren BR106M BIRENSUPA software stack"


def _device_index(device: int | str) -> int:
    if isinstance(device, bool):
        raise ValueError("Biren device index cannot be boolean")
    if isinstance(device, int):
        return int(device)
    text = str(device).strip().lower()
    if text in {"biren", "supa"}:
        return 0
    if text.isdigit():
        return int(text)
    if text.startswith(("biren:", "supa:")):
        return int(text.split(":", 1)[1])
    raise ValueError(f"Biren device must be supa:<index>; got {device!r}")


def biren_available(device: int | str = 0) -> bool:
    if import_torch_br(required=False) is None:
        return False
    try:
        import torch

        supa = getattr(torch, "supa", None)
        if supa is None or not supa.is_available():
            return False
        index = _device_index(device)
        return 0 <= index < int(supa.device_count()) and is_biren_br106m_name(
            supa.get_device_name(index)
        )
    except Exception:
        return False


def configure_biren_defaults(*, overwrite: bool = False) -> dict[str, str]:
    """Select the validated BF16/FP32 eager, no-FLA SUPA route."""

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


def enable_biren(
    device: int | str = 0,
    *,
    required: bool = True,
    set_device: bool = True,
) -> BirenRuntimeInfo:
    """Register torch_br, validate one BR106M, and select eager execution."""

    configure_biren_defaults()
    import_torch_br(required=required)
    import torch

    index = _device_index(device)
    supa = getattr(torch, "supa", None)
    try:
        available = bool(supa is not None and supa.is_available())
    except Exception:
        available = False
    count = int(supa.device_count()) if available else 0
    if required and not available:
        raise RuntimeError("torch_br imported, but no Biren SUPA device is available")
    if available and not 0 <= index < count:
        raise ValueError(
            f"Biren device index {index} is outside visible device count {count}"
        )
    name = str(supa.get_device_name(index)) if available else None
    exact_device = bool(available and is_biren_br106m_name(name))
    if required and not exact_device:
        raise RuntimeError(f"expected exact Biren106M, got {name!r}")

    versions = detect_biren_driver_versions()
    torch_version = str(getattr(torch, "__version__", "")) or None
    torch_br_version = _package_version("torch_br")
    sdk_version = detect_biren_sdk_version()
    validated, reason = validate_biren_stack(
        device_name=name,
        visible_devices=count,
        torch_version=torch_version,
        torch_br_version=torch_br_version,
        sdk_version=sdk_version,
        driver_version=versions["driver"],
        supa_version=versions["supa"],
    )
    override = allow_unvalidated_biren()
    if exact_device and not validated and not override:
        raise RuntimeError(
            "unvalidated Biren BR106M production stack: "
            f"{reason}. Set RWKV7_ALLOW_UNVALIDATED_BIREN=1 only for an "
            "explicitly reported experimental run."
        )
    if exact_device and set_device:
        supa.set_device(f"supa:{index}")
    if not exact_device:
        status = "unavailable"
    elif validated:
        status = "validated"
    else:
        status = "unvalidated_override"
    return BirenRuntimeInfo(
        available=exact_device,
        device_count=count,
        device_name=name,
        torch_version=torch_version,
        torch_br_version=torch_br_version,
        sdk_version=sdk_version,
        driver_version=versions["driver"],
        supa_version=versions["supa"],
        brsmi_version=versions["brsmi"],
        device=f"supa:{index}",
        dtype="bfloat16",
        state_dtype="float32",
        backend=os.environ["RWKV7_NATIVE_MODEL_BACKEND"],
        validated_stack=validated,
        validation_status=status,
        validation_reason=reason,
        allow_unvalidated=override,
    )


def synchronize(device: int | str | None = None) -> None:
    import_torch_br(required=True)
    import torch

    resolved = None if device is None else _device_index(device)
    torch.supa.synchronize(resolved)


def memory_stats(device: int | str | None = None) -> dict[str, int]:
    requested = 0 if device is None else device
    if not biren_available(requested):
        return {}
    import torch

    resolved = _device_index(requested)
    result: dict[str, int] = {}
    for label, name in (
        ("allocated_bytes", "memory_allocated"),
        ("reserved_bytes", "memory_reserved"),
        ("max_allocated_bytes", "max_memory_allocated"),
        ("max_reserved_bytes", "max_memory_reserved"),
    ):
        function = getattr(torch.supa, name, None)
        if callable(function):
            try:
                result[label] = int(function(resolved))
            except (RuntimeError, TypeError):
                pass
    return result


__all__ = [
    "BirenDTypeError",
    "BirenRuntimeInfo",
    "BirenRuntimePolicy",
    "SOURCE_COMMIT",
    "SOURCE_REPOSITORY",
    "allow_unvalidated_biren",
    "biren_available",
    "biren_runtime_policy",
    "configure_biren_defaults",
    "detect_biren_driver_versions",
    "detect_biren_sdk_version",
    "enable_biren",
    "import_torch_br",
    "is_biren_br106m_name",
    "memory_stats",
    "normalize_biren_device_name",
    "synchronize",
    "validate_biren_forward_dtype",
    "validate_biren_model_dtype",
    "validate_biren_stack",
]
