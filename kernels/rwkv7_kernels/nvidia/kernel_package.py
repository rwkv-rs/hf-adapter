# coding=utf-8
"""Discover and load optional prebuilt RWKV-7 kernel packages.

The base :mod:`rwkv7_hf` wheel stays portable.  Native CUDA extensions may be
provided by a separately built ``rwkv7_kernels`` package, selected before the
historical lazy-JIT path.  The package exposes only a JSON manifest at import
time; individual binary modules are imported lazily after compatibility has
been checked.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
import platform
import sys
from copy import deepcopy
from typing import Any

from packaging.specifiers import InvalidSpecifier, SpecifierSet


KERNEL_PACKAGE_PROTOCOL = "rwkv7-kernel-package-v1"
KERNEL_PACKAGE_SCHEMA_VERSION = 1
KERNEL_PACKAGE_MODULE = "rwkv7_kernels"
KERNEL_MODE_ENV = "RWKV7_KERNELS_MODE"
KERNEL_MODES = ("auto", "prebuilt", "jit", "portable")

_RUNTIME_EVENTS: dict[str, dict[str, Any]] = {}


def _version_series(raw: Any, parts: int = 2) -> str | None:
    if raw is None:
        return None
    values = str(raw).split("+", 1)[0].split(".")
    if len(values) < parts or not all(value.isdigit() for value in values[:parts]):
        return None
    return ".".join(values[:parts])


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def kernel_mode() -> str:
    """Return the requested binary/JIT policy.

    ``auto`` prefers a compatible prebuilt package and otherwise retains the
    existing lazy-JIT behavior. ``prebuilt`` forbids JIT fallback, ``jit``
    ignores an installed prebuilt package, and ``portable`` disables both
    native extension mechanisms so tensor/Triton/eager fallbacks remain in
    control.
    """

    mode = os.environ.get(KERNEL_MODE_ENV, "auto").strip().lower()
    if mode not in KERNEL_MODES:
        choices = ", ".join(KERNEL_MODES)
        raise ValueError(f"{KERNEL_MODE_ENV} must be one of {choices}; got {mode!r}")
    return mode


def jit_extensions_allowed() -> bool:
    return kernel_mode() in {"auto", "jit"}


def prebuilt_extensions_allowed() -> bool:
    return kernel_mode() in {"auto", "prebuilt"}


def _safe_call(callable_object: Any, default: Any = None) -> Any:
    try:
        return callable_object() if callable(callable_object) else default
    except Exception:
        return default


def _device_capability(torch_module: Any, device: Any = None) -> tuple[int, int] | None:
    cuda = getattr(torch_module, "cuda", None)
    if not bool(_safe_call(getattr(cuda, "is_available", None), False)):
        return None
    try:
        if isinstance(device, int):
            index = int(device)
        else:
            resolved = torch_module.device("cuda" if device is None else device)
            resolved_type = getattr(resolved, "type", "cuda")
            if resolved_type != "cuda":
                return None
            index = (
                int(_safe_call(getattr(cuda, "current_device", None), 0) or 0)
                if getattr(resolved, "index", None) is None
                else int(resolved.index)
            )
        major, minor = cuda.get_device_capability(index)
        return int(major), int(minor)
    except Exception:
        return None


def _runtime_environment(torch_module: Any, device: Any = None) -> dict[str, Any]:
    torch_version = str(getattr(torch_module, "__version__", "unknown"))
    torch_runtime = getattr(torch_module, "version", None)
    capability = _device_capability(torch_module, device)
    cxx11_abi = _safe_call(
        lambda: bool(getattr(torch_module._C, "_GLIBCXX_USE_CXX11_ABI")), None
    )
    return {
        "adapter_version": _distribution_version("rwkv7-hf"),
        "python_version": platform.python_version(),
        "python_series": f"{sys.version_info.major}.{sys.version_info.minor}",
        "python_abi": getattr(sys.implementation, "cache_tag", None),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "torch_version": torch_version,
        "torch_series": _version_series(torch_version),
        "torch_cuda_version": getattr(torch_runtime, "cuda", None),
        "torch_hip_version": getattr(torch_runtime, "hip", None),
        "torch_cxx11_abi": cxx11_abi,
        "device_capability": list(capability) if capability is not None else None,
        "device_arch": (
            f"sm{capability[0]}{capability[1]}" if capability is not None else None
        ),
    }


def recommended_distribution(environment: dict[str, Any]) -> str | None:
    """Return the stable binary companion distribution name for CUDA."""

    return "rwkv7-kernels" if recommended_build(environment) is not None else None


def recommended_build(environment: dict[str, Any]) -> str | None:
    """Return the deterministic CUDA/Torch/architecture build lane."""

    cuda = _version_series(environment.get("torch_cuda_version"))
    torch_series = _version_series(environment.get("torch_series"))
    arch = environment.get("device_arch")
    if cuda is None or torch_series is None or not arch:
        return None
    cuda_tag = cuda.replace(".", "")
    torch_tag = torch_series.replace(".", "")
    return f"cu{cuda_tag}-torch{torch_tag}-{arch}"


def _load_package_module() -> tuple[Any | None, str | None]:
    try:
        if importlib.util.find_spec(KERNEL_PACKAGE_MODULE) is None:
            return None, None
        module = importlib.import_module(KERNEL_PACKAGE_MODULE)
        return module, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _read_manifest(module: Any) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = getattr(module, "manifest", None)
        manifest = value() if callable(value) else getattr(module, "MANIFEST", None)
        if not isinstance(manifest, dict):
            return None, "rwkv7_kernels did not expose a dictionary manifest"
        return deepcopy(manifest), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _validate_manifest(
    manifest: dict[str, Any], environment: dict[str, Any]
) -> list[str]:
    reasons: list[str] = []
    if manifest.get("schema_version") != KERNEL_PACKAGE_SCHEMA_VERSION:
        reasons.append(
            "kernel manifest schema mismatch: "
            f"{manifest.get('schema_version')!r} != {KERNEL_PACKAGE_SCHEMA_VERSION}"
        )
    if manifest.get("protocol") != KERNEL_PACKAGE_PROTOCOL:
        reasons.append(
            f"kernel protocol mismatch: {manifest.get('protocol')!r} != "
            f"{KERNEL_PACKAGE_PROTOCOL!r}"
        )

    exact_fields = (
        ("python_series", "python_series"),
        ("python_abi", "python_abi"),
        ("platform_system", "platform_system"),
        ("platform_machine", "platform_machine"),
        ("torch_series", "torch_series"),
        ("torch_cuda_version", "torch_cuda_version"),
        ("torch_cxx11_abi", "torch_cxx11_abi"),
    )
    for manifest_name, environment_name in exact_fields:
        expected = manifest.get(manifest_name)
        actual = environment.get(environment_name)
        if expected is None:
            reasons.append(f"kernel manifest is missing required field {manifest_name}")
        elif expected != actual:
            reasons.append(
                f"{manifest_name} mismatch: wheel={expected!r}, runtime={actual!r}"
            )

    if manifest.get("distribution") != "rwkv7-kernels":
        reasons.append(
            "kernel distribution mismatch: "
            f"{manifest.get('distribution')!r} != 'rwkv7-kernels'"
        )
    if not manifest.get("version"):
        reasons.append("kernel manifest is missing required field version")
    extensions = manifest.get("extensions")
    if not isinstance(extensions, dict) or not extensions:
        reasons.append("kernel manifest must declare a non-empty extensions mapping")

    architectures = [str(item).lower() for item in manifest.get("architectures", [])]
    runtime_arch = environment.get("device_arch")
    if not architectures:
        reasons.append("kernel manifest does not declare any GPU architectures")
    elif runtime_arch is None:
        reasons.append("runtime does not expose a CUDA GPU architecture")
    elif runtime_arch.lower() not in architectures:
        reasons.append(
            f"device architecture {runtime_arch} is not present in wheel "
            f"architectures {architectures}"
        )
    if environment.get("torch_hip_version") is not None:
        reasons.append("CUDA kernel wheels are not loaded into a ROCm runtime")
    adapter_specifier = manifest.get("adapter_specifier")
    adapter_version = environment.get("adapter_version")
    if not adapter_specifier:
        reasons.append("kernel manifest is missing required field adapter_specifier")
    elif not adapter_version:
        reasons.append("rwkv7-hf distribution version is unavailable")
    else:
        try:
            if adapter_version not in SpecifierSet(str(adapter_specifier)):
                reasons.append(
                    f"adapter version {adapter_version} does not satisfy kernel wheel "
                    f"requirement {adapter_specifier}"
                )
        except InvalidSpecifier:
            reasons.append(
                f"invalid adapter_specifier in kernel manifest: {adapter_specifier!r}"
            )
    return reasons


def inspect_kernel_package(
    *,
    torch_module: Any | None = None,
    device: Any = None,
    package_module: Any | None = None,
) -> dict[str, Any]:
    """Return a JSON-serializable compatibility report without loading binaries."""

    if torch_module is None:
        import torch as torch_module  # type: ignore[no-redef]

    environment = _runtime_environment(torch_module, device)
    mode = kernel_mode()
    report: dict[str, Any] = {
        "status": "missing",
        "mode": mode,
        "module": KERNEL_PACKAGE_MODULE,
        "environment": environment,
        "recommended_distribution": recommended_distribution(environment),
        "recommended_build": recommended_build(environment),
        "manifest": None,
        "reasons": [],
    }
    if mode in {"jit", "portable"}:
        report["status"] = "disabled"
        report["reasons"] = [f"prebuilt kernels disabled by {KERNEL_MODE_ENV}={mode}"]
        return report

    module_error = None
    if package_module is None:
        package_module, module_error = _load_package_module()
    if package_module is None:
        if module_error is not None:
            report["status"] = "import_error"
            report["reasons"] = [module_error]
        else:
            report["reasons"] = ["rwkv7_kernels is not installed"]
        return report

    manifest, manifest_error = _read_manifest(package_module)
    if manifest is None:
        report["status"] = "invalid"
        report["reasons"] = [manifest_error or "kernel manifest unavailable"]
        return report
    report["manifest"] = manifest
    reasons = _validate_manifest(manifest, environment)
    report["reasons"] = reasons
    report["status"] = "ready" if not reasons else "incompatible"
    return report


def _record_event(name: str, **values: Any) -> None:
    _RUNTIME_EVENTS[name] = {"extension": name, **values}


def load_prebuilt_extension(
    name: str,
    *,
    torch_module: Any | None = None,
    device: Any = None,
) -> Any | None:
    """Load one compatible binary extension, or return ``None`` for fallback."""

    mode = kernel_mode()
    if not prebuilt_extensions_allowed():
        _record_event(name, status="disabled", source=mode)
        return None
    if torch_module is None:
        import torch as torch_module  # type: ignore[no-redef]

    module, module_error = _load_package_module()
    report = inspect_kernel_package(
        torch_module=torch_module, device=device, package_module=module
    )
    if module_error is not None and report["status"] == "missing":
        report["status"] = "import_error"
        report["reasons"] = [module_error]
    if report["status"] != "ready":
        _record_event(
            name,
            status=report["status"],
            source="prebuilt",
            reasons=list(report["reasons"]),
        )
        if mode == "prebuilt":
            details = "; ".join(report["reasons"]) or report["status"]
            raise RuntimeError(
                f"prebuilt RWKV-7 kernel {name!r} unavailable: {details}"
            )
        return None

    manifest = report["manifest"] or {}
    extensions = manifest.get("extensions", {})
    if name not in extensions:
        reason = f"kernel package does not provide extension {name!r}"
        _record_event(
            name, status="missing_extension", source="prebuilt", reasons=[reason]
        )
        if mode == "prebuilt":
            raise RuntimeError(reason)
        return None
    try:
        loader = getattr(module, "load_extension")
        extension = loader(name)
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        _record_event(name, status="import_error", source="prebuilt", reasons=[reason])
        if mode == "prebuilt":
            raise RuntimeError(
                f"prebuilt RWKV-7 kernel {name!r} failed to import: {reason}"
            ) from exc
        return None
    _record_event(
        name,
        status="selected",
        source="prebuilt",
        module=getattr(extension, "__name__", str(extensions[name])),
        distribution=manifest.get("distribution"),
        version=manifest.get("version"),
    )
    return extension


def record_jit_extension(
    name: str, *, selected: bool, error: str | None = None
) -> None:
    values: dict[str, Any] = {
        "status": "selected" if selected else "unavailable",
        "source": "jit",
    }
    if error:
        values["reasons"] = [error]
    _record_event(name, **values)


def kernel_runtime_report(
    *, torch_module: Any | None = None, device: Any = None
) -> dict[str, Any]:
    """Return package compatibility and extension-selection telemetry."""

    package = inspect_kernel_package(torch_module=torch_module, device=device)
    return {
        "mode": kernel_mode(),
        "prebuilt_allowed": prebuilt_extensions_allowed(),
        "jit_allowed": jit_extensions_allowed(),
        "package": package,
        "extensions": deepcopy(_RUNTIME_EVENTS),
    }


def reset_kernel_runtime_events() -> None:
    """Clear selection telemetry (primarily for tests and isolated probes)."""

    _RUNTIME_EVENTS.clear()


__all__ = [
    "KERNEL_MODE_ENV",
    "KERNEL_MODES",
    "KERNEL_PACKAGE_PROTOCOL",
    "KERNEL_PACKAGE_SCHEMA_VERSION",
    "inspect_kernel_package",
    "jit_extensions_allowed",
    "kernel_mode",
    "kernel_runtime_report",
    "load_prebuilt_extension",
    "prebuilt_extensions_allowed",
    "recommended_distribution",
    "recommended_build",
    "record_jit_extension",
    "reset_kernel_runtime_events",
]
