# coding=utf-8
"""Inspect the installed RWKV-7 runtime, accelerator, and kernel policy.

This command is deliberately read-only.  It does not load a checkpoint, build
an extension, capture a CUDA graph, or benchmark a kernel.  Its job is to make
the environment and the policy selected for each visible accelerator explicit
before a user starts a larger model load.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from .kernel_policy import current_adaptation_rule, current_kernel_policy


SCHEMA_VERSION = 1
PREFILL_FEATURES = (
    "fast_prefill",
    "fused_prefill_scan",
    "fused_prefill_self_chunk",
    "prefill_graph",
    "fused_prefill_shift_mix",
    "fused_prefill_state_prep",
    "fused_prefill_state_scan",
    "fused_prefill_output",
    "fused_prefill_residual_gemm",
    "fused_prefill_clampw_scan",
    "fused_prefill_stacked_rkv",
    "fused_prefill_sequence_ffn",
)
DECODE_FEATURES = (
    "fused_recurrent",
    "fused_recurrent_output",
    "fused_recurrent_raw",
    "fused_output",
    "fused_norm_mix",
    "native_graph_fp16_recurrent",
    "native_graph_triton_fp16_state",
    "native_graph_precompute_embedding",
    "sm70_linear",
    "sm70_wagv_lora",
    "ada_linear",
    "ada_wagv_lora",
    "ada_wagv_bmm",
    "sm120_compiled_ffn",
    "ada_sparse_ffn",
    "blackwell_cmix",
    "fused_output_project",
    "fused_projection",
    "fused_wag_lora",
    "fused_wavg_lora",
)
QUANT_FEATURES = (
    "native_external_quant_prefill",
    "native_external_quant_graph",
    "native_external_quant_prefill_graph",
    "native_bnb8_direct",
    "native_bnb8_relu_quant",
    "native_bnb8_rkv_mix_quant",
    "native_bnb8_ffn_mix_quant",
    "a8w8_fused_ffn",
)


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module_status(
    module_name: str, distribution_name: str | None = None
) -> dict[str, Any]:
    available = importlib.util.find_spec(module_name) is not None
    version = (
        _distribution_version(distribution_name or module_name) if available else None
    )
    return {"available": available, "version": version}


def _safe_call(callable_object: Any, default: Any = None) -> Any:
    try:
        return callable_object() if callable(callable_object) else default
    except Exception:
        return default


def _device_memory_bytes(torch_module: Any, index: int | None) -> int | None:
    if index is None:
        return None
    cuda = getattr(torch_module, "cuda", None)
    properties = _safe_call(
        lambda: cuda.get_device_properties(index) if cuda is not None else None
    )
    total_memory = getattr(properties, "total_memory", None)
    return int(total_memory) if total_memory is not None else None


def _torch_binary_compatible(
    torch_module: Any, capability: tuple[int, int] | None
) -> bool | None:
    """Return whether the installed Torch CUDA binaries cover one device."""

    if capability is None:
        return None
    cuda = getattr(torch_module, "cuda", None)
    arch_list = _safe_call(getattr(cuda, "get_arch_list", None), []) or []
    if not arch_list:
        return None
    device_cc = 10 * int(capability[0]) + int(capability[1])
    extract = getattr(cuda, "_extract_arch_version", None)
    compatible = getattr(cuda, "_code_compatible_with_device", None)
    if callable(extract) and callable(compatible):
        try:
            return any(compatible(device_cc, extract(arch)) for arch in arch_list)
        except Exception:
            pass
    special_capabilities = {53, 62, 72, 87, 101}
    for arch in arch_list:
        raw = str(arch).lower().split("_", 1)[-1]
        raw = raw.removesuffix("a").removesuffix("f")
        try:
            code_cc = int(raw)
        except ValueError:
            continue
        if code_cc == device_cc:
            return True
        if (
            code_cc // 10 == device_cc // 10
            and code_cc <= device_cc
            and device_cc not in special_capabilities
        ):
            return True
    return False


def _enabled(policy: Any, names: Sequence[str]) -> list[str]:
    return [name for name in names if bool(getattr(policy, name, False))]


def _policy_report(torch_module: Any, device: Any) -> dict[str, Any]:
    policy = current_kernel_policy(device=device, torch_module=torch_module)
    rule = current_adaptation_rule(device=device, torch_module=torch_module)
    profile = policy.profile
    return {
        "profile": asdict(profile),
        "memory_bytes": _device_memory_bytes(
            torch_module, getattr(profile, "device_index", None)
        ),
        "torch_binary_compatible": _torch_binary_compatible(
            torch_module, getattr(profile, "capability", None)
        ),
        "policy_defaults": {
            "fast_token_backend": policy.fast_token_backend,
            "fast_cache": bool(policy.fast_cache),
            "native_graph_state_dtype": policy.native_graph_state_dtype,
            "prefill": _enabled(policy, PREFILL_FEATURES),
            "decode": _enabled(policy, DECODE_FEATURES),
            "quantization": _enabled(policy, QUANT_FEATURES),
            "quant_policy": policy.quant_policy,
            "notes": policy.notes,
        },
        "adaptation": {
            "status": rule.status,
            "default_stance": rule.default_stance,
            "default_on": list(rule.default_on),
            "default_off": list(rule.default_off),
        },
    }


def _visible_policy_devices(torch_module: Any, requested: str | None) -> list[Any]:
    if requested is not None:
        return [requested]
    cuda = getattr(torch_module, "cuda", None)
    cuda_available = bool(_safe_call(getattr(cuda, "is_available", None), False))
    if cuda_available:
        count = int(_safe_call(getattr(cuda, "device_count", None), 1) or 1)
        return list(range(max(1, count)))
    return [None]


def _toolchain_report(torch_module: Any) -> dict[str, Any]:
    torch_version = getattr(torch_module, "version", None)
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    nvcc = shutil.which("nvcc")
    ninja = shutil.which("ninja")
    package_root = Path(__file__).resolve().parent
    cuda = getattr(torch_module, "cuda", None)
    return {
        "cuda_runtime": getattr(torch_version, "cuda", None),
        "hip_runtime": getattr(torch_version, "hip", None),
        "cuda_home": cuda_home,
        "nvcc": nvcc,
        "ninja": ninja,
        "cuda_extension_build_ready": bool((cuda_home or nvcc) and ninja),
        "triton": _module_status("triton"),
        "torch_cuda_arch_list": list(
            _safe_call(getattr(cuda, "get_arch_list", None), []) or []
        ),
        "packaged_sources": {
            "csrc": (package_root / "csrc").is_dir(),
            "triton_kernels": (package_root / "fused_recurrent_update.py").is_file(),
        },
        "cache_directories": {
            "torch_extensions": os.environ.get(
                "TORCH_EXTENSIONS_DIR", str(Path.home() / ".cache" / "torch_extensions")
            ),
            "triton": os.environ.get(
                "TRITON_CACHE_DIR", str(Path.home() / ".triton" / "cache")
            ),
        },
    }


def collect_diagnostics(
    *, device: str | None = None, torch_module: Any | None = None
) -> dict[str, Any]:
    """Return a JSON-serializable runtime and kernel-policy report."""

    if torch_module is None:
        import torch as torch_module  # type: ignore[no-redef]

    mps = getattr(getattr(torch_module, "backends", None), "mps", None)
    cuda = getattr(torch_module, "cuda", None)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "ready",
        "scope": "environment_and_policy_only",
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": {
            "rwkv7_hf": _distribution_version("rwkv7-hf"),
            "torch": str(getattr(torch_module, "__version__", "unknown")),
            "transformers": _distribution_version("transformers"),
        },
        "accelerators": {
            "cuda_available": bool(
                _safe_call(getattr(cuda, "is_available", None), False)
            ),
            "cuda_device_count": int(
                _safe_call(getattr(cuda, "device_count", None), 0) or 0
            ),
            "mps_available": bool(
                _safe_call(getattr(mps, "is_available", None), False)
            ),
        },
        "toolchain": _toolchain_report(torch_module),
        "devices": [
            _policy_report(torch_module, item)
            for item in _visible_policy_devices(torch_module, device)
        ],
        "notes": [
            "Policy defaults are hardware-level candidates; the actual runtime route also depends on model shape, dtype, batch size, sequence length, optional packages, and environment overrides.",
            "This command never compiles kernels or loads model weights.",
        ],
    }

    warnings: list[str] = []
    if report["accelerators"]["cuda_available"]:
        if not report["toolchain"]["triton"]["available"]:
            warnings.append(
                "CUDA is available but Triton is not importable; Triton routes will fall back."
            )
        if not report["toolchain"]["cuda_extension_build_ready"]:
            warnings.append(
                "A complete local CUDA extension toolchain was not detected; routes that require NVCC and Ninja cannot build, while Torch and Triton routes may still work."
            )
    for item in report["devices"]:
        profile = item["profile"]
        if item["torch_binary_compatible"] is False:
            capability = profile.get("capability")
            warnings.append(
                f"Installed PyTorch CUDA binaries do not support {profile['name']} "
                f"compute capability {capability}; install a matching PyTorch build."
            )
            report["status"] = "not_ready"
        adaptation_status = str(item["adaptation"]["status"]).lower()
        unvalidated_profile = bool(
            profile.get("validation_scope") == "unvalidated"
            and (
                profile.get("family") in {"unknown_cuda", "legacy_cuda"}
                or profile.get("hardware_generation") is not None
            )
        )
        if unvalidated_profile or any(
            marker in adaptation_status
            for marker in ("todo validation", "policy placeholder", "unsupported")
        ):
            warnings.append(
                f"{profile['name']} does not identify an exact validated hardware profile; conservative fallbacks remain enabled."
            )
    report["warnings"] = warnings
    return report


def render_text(report: dict[str, Any]) -> str:
    """Render a compact human-readable doctor report."""

    packages = report["packages"]
    accelerators = report["accelerators"]
    toolchain = report["toolchain"]
    lines = [
        "RWKV-7 Hugging Face runtime doctor",
        f"Status: {report['status'].upper()}",
        f"Python: {report['python']['version']} ({report['platform']['system']} {report['platform']['machine']})",
        "Packages: "
        f"rwkv7-hf={packages['rwkv7_hf'] or 'not-installed'} "
        f"torch={packages['torch']} "
        f"transformers={packages['transformers'] or 'not-installed'}",
        "Accelerators: "
        f"cuda={str(accelerators['cuda_available']).lower()} "
        f"devices={accelerators['cuda_device_count']} "
        f"mps={str(accelerators['mps_available']).lower()}",
        "Toolchain: "
        f"triton={toolchain['triton']['version'] or ('available' if toolchain['triton']['available'] else 'unavailable')} "
        f"nvcc={toolchain['nvcc'] or 'unavailable'} "
        f"ninja={toolchain['ninja'] or 'unavailable'}",
    ]
    for index, item in enumerate(report["devices"]):
        profile = item["profile"]
        policy = item["policy_defaults"]
        capability = profile.get("capability")
        capability_text = (
            f"sm_{capability[0]}{capability[1]}" if capability is not None else "n/a"
        )
        memory = item.get("memory_bytes")
        memory_text = f"{memory / 1024**3:.1f} GiB" if memory is not None else "n/a"
        lines.extend(
            [
                f"Device {index}: {profile['name']}",
                "  Profile: "
                f"vendor={profile['vendor']} family={profile['family']} "
                f"capability={capability_text} memory={memory_text} "
                f"validation={profile['validation_scope']}",
                "  Defaults: "
                f"token_backend={policy['fast_token_backend']} "
                f"cache={str(policy['fast_cache']).lower()} "
                f"state_dtype={policy['native_graph_state_dtype']} "
                f"torch_binary={item['torch_binary_compatible']}",
                "  Prefill candidates: "
                + (", ".join(policy["prefill"]) or "conservative fallback"),
                "  Decode candidates: "
                + (", ".join(policy["decode"]) or "conservative fallback"),
                "  Quant candidates: "
                + (", ".join(policy["quantization"]) or "conservative fallback"),
                f"  Policy note: {policy['notes']}",
            ]
        )
    for warning in report["warnings"]:
        lines.append(f"WARNING: {warning}")
    lines.extend(report["notes"])
    lines.append("RESULT: READY" if report["status"] == "ready" else "RESULT: FAIL")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the RWKV-7 runtime, accelerator, and kernel policy."
    )
    parser.add_argument(
        "--device",
        help="Inspect one torch device such as cuda:0 or mps; default inspects every visible CUDA device.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON."
    )
    parser.add_argument(
        "--output", type=Path, help="Also write the report to this path."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = collect_diagnostics(device=args.device)
    rendered = (
        json.dumps(report, ensure_ascii=False, indent=2)
        if args.json
        else render_text(report)
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["status"] == "ready" else 1


def cli() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    cli()
