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
    fast_prefill: bool = False
    bnb_skip_policy: str = "memory"
    fused_recurrent: bool = False
    fused_prefill_scan: bool = False
    prefill_graph: bool = False
    fused_prefill_shift_mix: bool = False
    fused_prefill_state_prep: bool = False
    fused_prefill_state_scan: bool = False
    fused_prefill_state_scan_max_batch: int | None = None
    fused_prefill_output: bool = False
    fused_recurrent_output: bool = False
    fused_recurrent_raw: bool = False
    fused_output: bool = False
    fused_norm_mix: bool = False
    sm70_linear: bool = False
    ada_linear: bool = False
    ada_wagv_lora: bool = False
    ada_sparse_ffn: bool = False
    fused_output_project: bool = False
    fused_projection: bool = False
    fused_wag_lora: bool = False
    fused_wavg_lora: bool = False
    wavg_lora_bsz1_max_hidden: int | None = None
    output_project_block_m: int = 16
    wag_lora_blocks: tuple[int, int, int] = (64, 64, 64)
    wavg_lora_blocks: tuple[int, int, int] = (64, 64, 64)
    quant_policy: str = "memory_first"
    notes: str = ""


@dataclass(frozen=True)
class GPUAdaptationRule:
    """Human-readable contract for adapting and validating one GPU family.

    ``KernelPolicy`` controls runtime defaults.  This rule records the
    card-specific evidence that must exist before those defaults can be
    promoted.  Keep it aligned with the live contract in AGENTS.md.
    """

    family: str
    cards: tuple[str, ...]
    status: str
    default_stance: str
    default_on: tuple[str, ...]
    default_off: tuple[str, ...]
    required_functional: tuple[str, ...]
    required_benchmarks: tuple[str, ...]
    quant_rule: str
    promotion_rule: str


COMMON_FUNCTIONAL_SMOKES = (
    "import_from_pretrained",
    "generate_use_cache",
    "rwkv7_forward_token",
    "batch_cache",
    "dynamic_batch_cache",
    "chunked_prefill",
    "native_graph_decode_greedy_match",
)

COMMON_PERF_BENCHMARKS = (
    "bench_batch_sweep.py bsz=1/2/4/8",
    "bench_native_graph_overhead.py",
    "bench_native_prefill_scan.py when prefill is claimed",
    "native_graph fused-output/recurrent-output A/B",
    "projection/LoRA/layout sweep before projection defaults",
    "W8/W4 footprint + speed rows before quant speed claims",
)


ADAPTATION_RULES: dict[str, GPUAdaptationRule] = {
    "cpu_or_unknown": GPUAdaptationRule(
        family="cpu_or_unknown",
        cards=("CPU", "no live CUDA/HIP device"),
        status="compatibility fallback",
        default_stance="reference-only; runtime availability gates must prevent CUDA kernels",
        default_on=("fast_cache",),
        default_off=("all CUDA/HIP custom kernels",),
        required_functional=("import", "pure torch/native_model smoke where supported"),
        required_benchmarks=("CPU smoke only; no GPU performance claim",),
        quant_rule="do not claim W8/W4 speed without a real accelerator row",
        promotion_rule="never promote GPU defaults from CPU-only evidence",
    ),
    "legacy_cuda": GPUAdaptationRule(
        family="legacy_cuda",
        cards=("pre-Pascal CUDA",),
        status="unsupported performance target",
        default_stance="compatibility-first",
        default_on=("fast_cache",),
        default_off=("native_graph fused Triton kernels", "bnb speed claims"),
        required_functional=COMMON_FUNCTIONAL_SMOKES[:3],
        required_benchmarks=("single-card import/generate smoke",),
        quant_rule="memory-only if a backend loads; no speed target",
        promotion_rule="do not enable fused defaults on legacy CUDA",
    ),
    "unknown_cuda": GPUAdaptationRule(
        family="unknown_cuda",
        cards=("unclassified CUDA GPU",),
        status="policy placeholder",
        default_stance="safe fallback until exact architecture is added",
        default_on=("fast_cache",),
        default_off=("native_graph fused Triton kernels",),
        required_functional=COMMON_FUNCTIONAL_SMOKES,
        required_benchmarks=COMMON_PERF_BENCHMARKS,
        quant_rule="memory-only until exact-card W8/W4 speed rows exist",
        promotion_rule="add an explicit family/card rule before changing defaults",
    ),
    "pascal": GPUAdaptationRule(
        family="pascal",
        cards=("Tesla P100", "GTX 10-series"),
        status="touched; GTX 1080 Ti 0.1B smoke/bnb+native-mm quant speed rows and 0.4B fp16 row exist",
        default_stance="compatibility-first; Pascal lacks the newer tensor-core path",
        default_on=("fast_cache",),
        default_off=("fused_recurrent_output", "fused_output", "projection/LoRA fusions", "fused_prefill_scan"),
        required_functional=(
            "import_from_pretrained",
            "generate_use_cache",
            "default native/no-FLA decode",
            "batch_cache",
            "dynamic_batch_cache",
            "chunked_prefill",
        ),
        required_benchmarks=COMMON_PERF_BENCHMARKS,
        quant_rule="bnb W8/W4 rows are slower than fp16; native mm8/mm4 0.1B lm_head rows pass, broader promotion needs larger exact-card quant rows",
        promotion_rule="require exact-card decode greedy match plus non-negative speed before any default",
    ),
    "volta": GPUAdaptationRule(
        family="volta",
        cards=("Tesla V100-PCIE-32GB", "Tesla V100-SXM"),
        status="current regression baseline",
        default_stance="conservative production-smoke baseline",
        default_on=(
            "fast_cache",
            "fused_recurrent_output",
            "fused_recurrent_raw",
            "fused_output",
            "fused_norm_mix",
            "batch-routed fused_wavg_lora",
            "shape-routed sm70_linear",
            "batch-routed fused prefill",
        ),
        default_off=("fused_recurrent", "fused_output_project", "full projection fusion"),
        required_functional=COMMON_FUNCTIONAL_SMOKES
        + ("HF Trainer", "TRL SFT/DPO/GRPO", "PEFT save/load/merge"),
        required_benchmarks=COMMON_PERF_BENCHMARKS
        + ("training smoke telemetry", "Albatross A/B rows when available"),
        quant_rule="W8/W4 memory rows valid; speed unsolved until native quant beats fp16 on V100",
        promotion_rule="do not change V100 defaults without preserving HF training and decode rows",
    ),
    "turing": GPUAdaptationRule(
        family="turing",
        cards=("Tesla T4", "RTX 20-series"),
        status="TODO validation target",
        default_stance="Volta-safe output fusions only after card-local smoke",
        default_on=("fast_cache", "fused_recurrent_output", "fused_output"),
        default_off=("fused_prefill_scan", "fused_output_project", "projection/LoRA fusions"),
        required_functional=COMMON_FUNCTIONAL_SMOKES,
        required_benchmarks=COMMON_PERF_BENCHMARKS,
        quant_rule="memory-first until exact-card speed rows beat fp16",
        promotion_rule="require bsz sweep and quant rows before performance claims",
    ),
    "ampere": GPUAdaptationRule(
        family="ampere",
        cards=("A100", "A800", "RTX A6000", "A10", "RTX 30-series"),
        status="A100/A800/RTX A6000 validation rows exist; keep conservative Ampere defaults",
        default_stance="stable output fusions; tune larger batch and training paths per card",
        default_on=("fast_cache", "fused_recurrent_output", "fused_output"),
        default_off=("fused_prefill_scan", "fused_output_project", "projection/LoRA fusions"),
        required_functional=COMMON_FUNCTIONAL_SMOKES
        + ("ZeRO-2/ZeRO-3 smoke when training is claimed",),
        required_benchmarks=COMMON_PERF_BENCHMARKS
        + ("larger-batch prefill", "state-cache reuse/hit-rate rows"),
        quant_rule="bnb/native W8/W4 require exact-card footprint and speed telemetry rows; current A800/A6000 rows reduce memory but do not satisfy the quantized-speed gate",
        promotion_rule="do not reuse V100/4090 block sizes without an Ampere sweep",
    ),
    "ada": GPUAdaptationRule(
        family="ada",
        cards=("RTX 4090", "RTX 4080/4070", "RTX 40-series"),
        status="4090 decode optimized; exact-row B2 and grouped W/A/G/V B1/B2/B4 correctness/speed rows pass",
        default_stance="high-end consumer path with shape-routed exact-row and grouped low-rank kernels",
        default_on=(
            "fast_cache", "fused_recurrent_output", "fused_recurrent_raw", "fused_output",
            "fused_norm_mix", "ada_linear for rows=2 and rows=4 hidden projections", "ada_wagv_lora for rows<=4",
        ),
        default_off=("fused_output_project", "generic Triton projection/LoRA fusions", "ada_sparse_ffn", "fused_prefill_scan by default"),
        required_functional=COMMON_FUNCTIONAL_SMOKES,
        required_benchmarks=COMMON_PERF_BENCHMARKS
        + ("fast-prefill TTFT/TPOT rows when RWKV7_FAST_PREFILL is considered",),
        quant_rule="bnb is compatibility/memory baseline; native quant speed needs end-to-end rows",
        promotion_rule="4090 bsz=1/4 min speedup gates must pass before enabling a new fusion",
    ),
    "hopper": GPUAdaptationRule(
        family="hopper",
        cards=("H100", "H200"),
        status="TODO validation target",
        default_stance="expected fast server path, but not tuned until H100 rows exist",
        default_on=("fast_cache", "fused_recurrent_output", "fused_output"),
        default_off=("fused_prefill_scan", "fused_output_project", "projection/LoRA fusions"),
        required_functional=COMMON_FUNCTIONAL_SMOKES
        + ("multi-GPU PP/TP smoke when serving is claimed", "ZeRO-2/ZeRO-3 smoke when training is claimed"),
        required_benchmarks=COMMON_PERF_BENCHMARKS
        + ("larger model rows", "large batch/chunked prefill rows"),
        quant_rule="W8/W4 and FP8-like paths require H100-specific precision/speed rows",
        promotion_rule="do not assume 4090 or Blackwell tile sizes are optimal on H100",
    ),
    "blackwell": GPUAdaptationRule(
        family="blackwell",
        cards=("RTX 5070 Laptop", "RTX 5090", "RTX 5080/5090", "RTX 50-series"),
        status="touched; 5070 Laptop rows and RTX 5090 HF/native-prefill/native-trainer rows exist",
        default_stance="prefer native/no-FLA fallback when FLA kernels fail on 50-series; apply Blackwell Triton/torch.compile compatibility for early sm_120 stacks",
        default_on=("fast_cache", "fused_recurrent_output", "fused_output"),
        default_off=("fused_output_project", "projection/LoRA fusions", "fused_prefill_scan by default"),
        required_functional=COMMON_FUNCTIONAL_SMOKES
        + ("native_model no-FLA training smoke", "bnb W8/W4 functional inference", "triton_compat remote-code import"),
        required_benchmarks=COMMON_PERF_BENCHMARKS
        + ("50-series FLA compatibility row", "native/no-FLA fallback row", "RTX 5090 HF validation runner artifact when claiming 5090"),
        quant_rule="microbench wins are insufficient; require end-to-end decode and quality rows",
        promotion_rule="promote only fusions with exact-card greedy match and min bsz speedup >= 1.0x",
    ),
    "amd_hip": GPUAdaptationRule(
        family="amd_hip",
        cards=("AMD Instinct MI250/MI300", "Radeon ROCm cards"),
        status="compatibility target; TODO validation",
        default_stance="pure PyTorch/native_model first; CUDA/Triton kernels off",
        default_on=("fast_cache",),
        default_off=("CUDA native_graph fused kernels", "bnb CUDA-only speed paths"),
        required_functional=COMMON_FUNCTIONAL_SMOKES
        + ("ROCm import/generate", "pure PyTorch/native_model forward/backward"),
        required_benchmarks=("ROCm smoke rows", "HIP-specific speed rows before parity claims"),
        quant_rule="no AMD quant performance claim until HIP-specific W8/W4 rows exist",
        promotion_rule="add ROCm-specific kernels or proven fallbacks before enabling accelerated defaults",
    ),
}


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
            notes="compatibility-first: keep experimental Triton/native_graph fusions off; Pascal uses native/no-FLA fallback unless overridden",
        )
    if family == "volta":
        return KernelPolicy(
            profile=profile,
            fast_prefill=True,
            fused_recurrent_output=True,
            fused_recurrent_raw=True,
            fused_output=True,
            fused_prefill_scan=True,
            fused_prefill_state_prep=True,
            fused_prefill_state_scan=True,
            fused_prefill_state_scan_max_batch=1,
            fused_prefill_output=True,
            fused_norm_mix=True,
            fused_wavg_lora=True,
            wavg_lora_bsz1_max_hidden=1024,
            sm70_linear=True,
            output_project_block_m=16,
            quant_policy="memory_first_decode_hot_optional",
            notes="V100 baseline: batch-routed split-row prefill and WAVG-LoRA plus shape-routed sm70 linear/RKV, output/recurrent-output, and decode norm/mix fusions are default; WAVG bsz=1 is limited to hidden<=1024 and full projection/output-project remain opt-in",
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
        is_4090 = "4090" in profile.name.lower()
        return KernelPolicy(
            profile=profile,
            fast_prefill=is_4090,
            fused_recurrent_output=True,
            fused_recurrent_raw=True,
            fused_output=True,
            fused_norm_mix=True,
            fused_prefill_scan=is_4090,
            prefill_graph=is_4090,
            fused_prefill_shift_mix=is_4090,
            fused_prefill_state_prep=is_4090,
            fused_prefill_output=is_4090,
            ada_linear=True,
            ada_wagv_lora=True,
            ada_sparse_ffn=False,
            output_project_block_m=16,
            notes="RTX 40/Ada: exact-4090 rows promote fixed-shape prefill graph with split scan/state-prep/output/shift plus raw recurrent decode, decode norm/mix, rows=2 exact linear, rows=4 hidden exact linear, and rows<=4 grouped W/A/G/V LoRA; other Ada cards retain the compatible fallback until measured",
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
            notes="RTX 50/Blackwell: use triton_compat for early sm_120 stacks, prefer native/no-FLA smokes, keep unvalidated projection/LoRA fusions off",
        )
    return KernelPolicy(profile=profile)


def adaptation_rule_for_profile(profile: GPUProfile) -> GPUAdaptationRule:
    """Return the validation/adaptation contract for a normalized GPU profile."""

    return ADAPTATION_RULES.get(profile.family, ADAPTATION_RULES["unknown_cuda"])


def current_adaptation_rule(device: int | str | None = None, torch_module: Any | None = None) -> GPUAdaptationRule:
    return adaptation_rule_for_profile(detect_gpu_profile(device=device, torch_module=torch_module))


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
