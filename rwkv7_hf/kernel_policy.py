# coding=utf-8
"""GPU-aware default kernel policy for RWKV-7 HF/native paths.

The adapter must support many cards, but fused kernels are not universally
profitable or even available.  This module centralizes the *default* policy:

* explicit environment variables always win;
* CUDA generation decides conservative defaults;
* unvalidated/shallow kernels stay off until a per-GPU benchmark row proves
  they should be enabled.

The policy intentionally does not replace benchmarks.  It gives each GPU family
a stable starting point, while AGENTS.md defines the validation gates required
before changing a default.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any


FALSE_VALUES = {"0", "false", "no", "off"}
TRUE_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class GPUProfile:
    """Normalized hardware identity used by the kernel policy."""

    name: str
    vendor: str
    family: str
    capability: tuple[int, int] | None = None
    device_index: int | None = None
    is_cuda: bool = False
    is_hip: bool = False


@dataclass(frozen=True)
class KernelPolicy:
    """Default fused-kernel policy for a GPU profile.

    These are defaults only.  Runtime env vars such as
    ``RWKV7_NATIVE_GRAPH_FUSED_OUTPUT=0`` override them.
    """

    profile: GPUProfile
    fast_token_backend: str = "auto"
    fast_cache: bool = True
    bnb_skip_policy: str = "memory"
    fused_recurrent: bool = False
    fused_prefill_scan: bool = False
    fused_recurrent_output: bool = False
    fused_output: bool = False
    fused_output_project: bool = False
    fused_projection: bool = False
    fused_wag_lora: bool = False
    fused_wavg_lora: bool = False
    output_project_block_m: int = 16
    wag_lora_blocks: tuple[int, int, int] = (64, 64, 64)
    wavg_lora_blocks: tuple[int, int, int] = (64, 64, 64)
    quant_policy: str = "memory_first"
    notes: str = ""


def classify_gpu(name: str | None, capability: tuple[int, int] | None, *, is_hip: bool = False) -> GPUProfile:
    """Classify a GPU without requiring torch/CUDA to be available."""

    gpu_name = (name or "unknown").strip() or "unknown"
    lower = gpu_name.lower()
    if is_hip or any(token in lower for token in ("amd", "radeon", "instinct", "mi250", "mi300")):
        return GPUProfile(name=gpu_name, vendor="amd", family="amd_hip", capability=capability, is_cuda=False, is_hip=True)
    if capability is None:
        return GPUProfile(name=gpu_name, vendor="unknown", family="cpu_or_unknown", capability=None)

    major, minor = int(capability[0]), int(capability[1])
    family = "unknown_cuda"
    if major < 6:
        family = "legacy_cuda"
    elif major == 6:
        family = "pascal"
    elif major == 7 and minor == 0:
        family = "volta"
    elif major == 7:
        family = "turing"
    elif major == 8 and minor == 9:
        family = "ada"
    elif major == 8:
        family = "ampere"
    elif major == 9:
        family = "hopper"
    elif major >= 10 or "rtx 50" in lower or "blackwell" in lower:
        family = "blackwell"
    return GPUProfile(name=gpu_name, vendor="nvidia", family=family, capability=(major, minor), is_cuda=True)


def detect_gpu_profile(device: int | str | None = None, torch_module: Any | None = None) -> GPUProfile:
    """Detect the active GPU profile, falling back to cpu_or_unknown."""

    if torch_module is None:
        try:  # pragma: no cover - optional in CPU-only CI
            import torch as torch_module  # type: ignore[no-redef]
        except Exception:  # pragma: no cover
            torch_module = None
    if torch_module is None:
        return classify_gpu(None, None)

    is_hip = bool(getattr(getattr(torch_module, "version", None), "hip", None))
    cuda = getattr(torch_module, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    if not callable(is_available) or not is_available():
        return classify_gpu(None, None, is_hip=is_hip)

    try:
        index = 0 if device is None else torch_module.device(device).index
        if index is None:
            index = int(cuda.current_device())
    except Exception:
        index = 0
    try:
        name = str(cuda.get_device_name(index))
    except Exception:
        name = "unknown"
    try:
        capability = tuple(int(v) for v in cuda.get_device_capability(index))  # type: ignore[arg-type]
    except Exception:
        capability = None
    profile = classify_gpu(name, capability, is_hip=is_hip)
    return GPUProfile(
        name=profile.name,
        vendor=profile.vendor,
        family=profile.family,
        capability=profile.capability,
        device_index=index,
        is_cuda=profile.is_cuda,
        is_hip=profile.is_hip,
    )


def policy_for_profile(profile: GPUProfile) -> KernelPolicy:
    """Return conservative defaults for a normalized GPU profile."""

    family = profile.family
    if family == "cpu_or_unknown":
        return KernelPolicy(
            profile=profile,
            fused_recurrent_output=True,
            fused_output=True,
            fused_prefill_scan=False,
            notes="no live GPU detected: preserve historical request defaults; runtime availability gates still prevent CUDA use",
        )
    if family in {"amd_hip", "legacy_cuda", "pascal", "unknown_cuda"}:
        return KernelPolicy(
            profile=profile,
            fused_recurrent_output=False,
            fused_output=False,
            notes="compatibility-first: keep experimental Triton/native_graph fusions off until per-card smoke passes",
        )
    if family == "volta":
        return KernelPolicy(
            profile=profile,
            fused_recurrent_output=True,
            fused_output=True,
            fused_prefill_scan=False,
            output_project_block_m=16,
            quant_policy="memory_first_decode_hot_optional",
            notes="V100 baseline: output/recurrent-output fusions are default; projection/output-project/LoRA fusions remain opt-in",
        )
    if family in {"turing", "ampere"}:
        return KernelPolicy(
            profile=profile,
            fused_recurrent_output=True,
            fused_output=True,
            fused_prefill_scan=False,
            output_project_block_m=16,
            notes="CUDA tensor-core generation: use stable output fusions; require local sweep before projection/LoRA defaults",
        )
    if family == "ada":
        return KernelPolicy(
            profile=profile,
            fused_recurrent_output=True,
            fused_output=True,
            fused_prefill_scan=False,
            output_project_block_m=16,
            notes="RTX 40/Ada: stable output fusions on by default; shallow split-K projection stays off",
        )
    if family == "hopper":
        return KernelPolicy(
            profile=profile,
            fused_recurrent_output=True,
            fused_output=True,
            fused_prefill_scan=False,
            output_project_block_m=32,
            notes="Hopper profile: stable output fusions on; H100-specific projection/quant kernels require sweep rows",
        )
    if family == "blackwell":
        return KernelPolicy(
            profile=profile,
            fused_recurrent_output=True,
            fused_output=True,
            fused_prefill_scan=False,
            output_project_block_m=32,
            notes="RTX 50/Blackwell: prefer native/no-FLA compatibility smokes; keep unvalidated projection/LoRA fusions off",
        )
    return KernelPolicy(profile=profile)


def current_kernel_policy(device: int | str | None = None, torch_module: Any | None = None) -> KernelPolicy:
    return policy_for_profile(detect_gpu_profile(device=device, torch_module=torch_module))


def env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    value = raw.strip().lower()
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    return bool(default)


def env_int(name: str, default: int, *, lower: int = 1, upper: int | None = None) -> int:
    raw = os.environ.get(name)
    try:
        value = int(str(raw if raw is not None else default).strip())
    except Exception:
        value = int(default)
    value = max(int(lower), value)
    if upper is not None:
        value = min(int(upper), value)
    return value


def env_blocks(
    names: tuple[str, str, str],
    defaults: tuple[int, int, int],
    uppers: tuple[int, int, int],
) -> tuple[int, int, int]:
    return (
        env_int(names[0], defaults[0], lower=1, upper=uppers[0]),
        env_int(names[1], defaults[1], lower=1, upper=uppers[1]),
        env_int(names[2], defaults[2], lower=1, upper=uppers[2]),
    )
