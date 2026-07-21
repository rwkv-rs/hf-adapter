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


def is_tesla_t4_name(name: str) -> bool:
    """Match the exact T4 product token without accepting names like T400."""

    normalized = "".join(
        character if character.isalnum() else " "
        for character in str(name).lower()
    )
    return "t4" in normalized.split()


def is_v100_name(name: str) -> bool:
    """Match the exact V100 product token without promoting every sm_70 GPU."""

    normalized = "".join(
        character if character.isalnum() else " "
        for character in str(name).lower()
    )
    return "v100" in normalized.split()


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
    is_mps: bool = False


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
    bnb_int8_threshold: float | None = None
    native_external_quant_prefill: bool = False
    native_external_quant_graph: bool = False
    native_external_quant_prefill_graph: bool = False
    native_bnb8_direct: bool = False
    native_bnb8_relu_quant: bool = False
    native_bnb8_rkv_mix_quant: bool = False
    native_bnb8_ffn_mix_quant: bool = False
    native_bnb8_attn_mix_block: int = 1024
    native_bnb8_ffn_mix_block: int = 1024
    a8w8_gemv_max_rows: int = 1
    mm4_fused_max_rows: int | None = None
    mm4_gemv_block_pairs: int | None = None
    mm4_gemv_block_n: int | None = None
    mm4_dot_min_rows: int | None = None
    mm4_dot_block_b: int | None = None
    mm4_dot_block_pairs: int | None = None
    mm4_dot_block_n: int | None = None
    mm4_dot_warps: int | None = None
    marlin_w4_ffn_shapes: tuple[tuple[int, int], ...] = ()
    # hidden, intermediate, layers, group_size, quantize_head, skip_last_layers
    marlin_w4_model_profiles: tuple[tuple[int, int, int, int, bool, int], ...] = ()
    fused_recurrent: bool = False
    fused_prefill_scan: bool = False
    prefill_scan_model_shapes: tuple[tuple[int, int, int, int], ...] = ()
    fused_prefill_self_chunk: bool = False
    prefill_self_chunk_min_tokens: int = 1024
    prefill_self_chunk_size: int = 16
    prefill_self_chunk_shape_sizes: tuple[tuple[int, int, int], ...] = ()
    prefill_self_chunk_h_tile_shapes: tuple[tuple[int, int, int, int], ...] = ()
    prefill_self_chunk_model_shapes: tuple[tuple[int, int, int, int], ...] = ()
    prefill_self_chunk_model_shapes_only: bool = False
    prefill_scan_block_m: int | None = None
    prefill_scan_block_m_b2: int | None = None
    prefill_scan_block_m_b4: int | None = None
    prefill_scan_block_m_shapes: tuple[tuple[int, int, int], ...] = ()
    # Exact HxBxTxM routes, where H is the model hidden size and M is
    # the recurrent-scan row tile.  Keep model-specific wins out of the
    # generic BxT table so a smaller checkpoint cannot regress a larger one.
    prefill_scan_block_m_model_shapes: tuple[tuple[int, int, int, int], ...] = ()
    prefill_scan_num_warps: int | None = None
    prefill_blas_library: str | None = None
    prefill_blas_large_library: str | None = None
    prefill_blas_large_min_rows: int = 4096
    prefill_graph: bool = False
    prefill_graph_cache_size: int = 2
    prefill_graph_model_shapes: tuple[tuple[int, int, int, int], ...] = ()
    # Auto-policy safety ceiling.  Explicit RWKV7_NATIVE_PREFILL_GRAPH keeps
    # its force-on semantics for experiments that intentionally exceed it.
    prefill_graph_max_layers: int | None = None
    prefill_fp16_recurrent: bool = False
    fused_prefill_shift_mix: bool = False
    prefill_shift_mix_model_shapes: tuple[tuple[int, int, int, int], ...] = ()
    prefill_attn_shift_mix_strict_fp16_model_shapes: tuple[tuple[int, int, int, int], ...] = ()
    prefill_ffn_shift_mix_strict_fp16_model_shapes: tuple[tuple[int, int, int, int], ...] = ()
    # hidden, layers, batch, tokens, block size, warps
    prefill_attn_shift_mix_launch_profiles: tuple[tuple[int, int, int, int, int, int], ...] = ()
    prefill_ffn_shift_mix_launch_profiles: tuple[tuple[int, int, int, int, int, int], ...] = ()
    fused_prefill_state_prep: bool = False
    prefill_state_prep_model_shapes: tuple[tuple[int, int, int, int], ...] = ()
    # hidden, layers, batch, tokens, enabled leading-layer count
    prefill_state_prep_layer_counts: tuple[tuple[int, int, int, int, int], ...] = ()
    fused_prefill_state_scan: bool = False
    fused_prefill_state_scan_max_batch: int | None = None
    fused_prefill_output: bool = False
    prefill_fused_output_model_shapes: tuple[tuple[int, int, int, int], ...] = ()
    fused_prefill_residual_gemm: bool = False
    fused_prefill_clampw_scan: bool = False
    prefill_clampw_scan_model_shapes: tuple[tuple[int, int, int, int], ...] = ()
    fused_prefill_stacked_rkv: bool = False
    prefill_stacked_rkv_min_rows: int = 128
    prefill_stacked_rkv_max_rows: int | None = None
    prefill_stacked_rkv_extra_rows: tuple[int, ...] = ()
    prefill_stacked_rkv_shapes: tuple[tuple[int, int], ...] = ()
    prefill_stacked_rkv_model_shapes: tuple[tuple[int, int, int, int], ...] = ()
    fused_prefill_sequence_ffn: bool = False
    prefill_sequence_ffn_min_rows: int = 128
    prefill_sequence_ffn_max_rows: int | None = None
    prefill_sequence_ffn_extra_rows: tuple[int, ...] = ()
    prefill_sequence_ffn_model_shapes: tuple[tuple[int, int, int, int], ...] = ()
    prefill_sequence_ffn_blocks: tuple[int, int, int, int, int] = (128, 128, 32, 64, 8)
    prefill_sequence_ffn_large_min_rows: int = 1024
    prefill_sequence_ffn_large_blocks: tuple[int, int, int, int, int] = (128, 128, 32, 64, 8)
    prefill_sequence_ffn_num_stages: int = 3
    prefill_sequence_ffn_num_warps: int = 4
    prefill_fp16_accum_ffn_key_model_shapes: tuple[tuple[int, int, int, int], ...] = ()
    prefill_fp16_accum_ffn_key_layer_counts: tuple[tuple[int, int, int, int, int], ...] = ()
    fused_recurrent_output: bool = False
    fused_recurrent_raw: bool = False
    fused_output: bool = False
    fused_norm_mix: bool = False
    norm_mix_num_warps: int = 4
    native_graph_state_dtype: str = "fp32"
    native_graph_fp16_recurrent: bool = False
    native_graph_precompute_embedding: bool = False
    # Auto backend safety ceiling.  An explicitly requested native_graph
    # backend remains available for profiling/experimentation.
    native_graph_max_layers: int | None = None
    sm70_linear: bool = False
    sm70_wagv_lora: bool = False
    ada_linear: bool = False
    ada_linear_rows: str = "2 4"
    ada_linear_roles: str = "auto"
    ada_wagv_lora: bool = False
    ada_wag_lora: bool = False
    ada_sparse_ffn: bool = False
    ada_sparse_ffn_max_rows: int = 19
    ada_sparse_ffn_inplace: bool = False
    ada_sparse_ffn_up: bool = True
    ada_sparse_ffn_low_memory_pack: bool = False
    ada_sparse_ffn_share_pack: bool = False
    ada_sparse_ffn_fp32_accum: bool = False
    ada_sparse_ffn_deterministic_splits: int = 0
    ada_sparse_ffn_official_boundary: bool = False
    blackwell_cmix: bool = False
    rkv_policy: str = "manual"
    fused_output_project: bool = False
    fused_projection: bool = False
    fused_wag_lora: bool = False
    fused_wavg_lora: bool = False
    wavg_lora_bsz1_max_hidden: int | None = None
    output_project_block_m: int = 16
    wag_lora_blocks: tuple[int, int, int] = (64, 64, 64)
    wavg_lora_blocks: tuple[int, int, int] = (64, 64, 64)
    wavg_lora_num_warps: int = 4
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
    "apple_mps": GPUAdaptationRule(
        family="apple_mps",
        cards=("Apple Silicon M-series / MPS", "Apple MLX / Metal", "CoreML / ANE"),
        status="M5 compatibility and MLX rows exist; stateful CoreML 0.1B correctness passes",
        default_stance="native/no-FLA compatibility; CUDA/Triton kernels off; MLX/CoreML are separate explicit backends",
        default_on=("fast_cache", "native_model fallback"),
        default_off=("CUDA native_graph fused kernels", "bnb CUDA quantization"),
        required_functional=(
            "MPS load/generate",
            "PEFT/Trainer/TRL smoke",
            "MLX recurrent/cache/chunked-prefill smoke",
            "CoreML state transfer + chunk split + HF greedy parity when CoreML is claimed",
        ),
        required_benchmarks=(
            "exact M-series chip/memory/macOS rows",
            "MLX fp16 and W8/W4 speed/footprint rows",
            "CoreML runtime placement evidence before ANE claims",
        ),
        quant_rule="native/MLX/CoreML W8/W4 only; require footprint reduction, greedy/quality parity, and exact-device speed rows",
        promotion_rule="do not infer ANE use from CPU_AND_NE eligibility or promote fp16 CoreML while HF greedy parity fails",
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
        status="Tesla T4 0.1B-2.9B fp16 HF/cache/prefill/decode/quant/training integration validated; production performance and RTX 20 remain open",
        default_stance="card-local defaults: T4 uses native fused prefill; unvalidated RTX 20 stays conservative",
        default_on=(
            "fast_cache",
            "fused_recurrent_output",
            "fused_output",
            "Tesla T4 only: fast_prefill and fused_prefill_scan",
        ),
        default_off=(
            "RTX 20: fused_prefill_scan",
            "fused_output_project",
            "projection/LoRA fusions",
        ),
        required_functional=COMMON_FUNCTIONAL_SMOKES
        + ("HF Trainer", "PEFT save/load/merge", "TRL SFT/DPO/GRPO"),
        required_benchmarks=COMMON_PERF_BENCHMARKS
        + ("prompt512 fused-scan bsz=1/2/4/8", "same-card Albatross decode/prefill"),
        quant_rule="T4 head-only native W8/W4 is a measured decode-speed lane; full-model W8/W4 remains a memory/B1-decode lane until every prefill and batch row beats fp16",
        promotion_rule="T4 stays validated, not production-close, until dense Albatross and full-model all-phase quant gates pass; never inherit T4 defaults on RTX 20 without exact-card rows",
    ),
    "ampere": GPUAdaptationRule(
        family="ampere",
        cards=("A100", "A800", "RTX A6000", "A10", "RTX 30-series"),
        status="A100/A800/RTX A6000 rows exist; RTX 3090 native-prefill graph and quant-policy rows exist",
        default_stance="stable family defaults with exact-card RTX 3090 prefill and decode-hot quant routing",
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
        status="RTX 4090 promoted matrices and exact RTX 4080 native/Qwen3.5/training/quant rows exist; unmeasured Ada cards remain card-local validation targets",
        default_stance="exact RTX 4090 and RTX 4080 shape-routed paths with compatible fallbacks elsewhere",
        default_on=(
            "fast_cache", "fused_recurrent_output", "fused_recurrent_raw", "fused_output",
            "fused_norm_mix", "exact-card prefill graph/scan policy",
            "exact-4080 prefill shift/state/output for measured 0.4B/1.5B shapes",
            "exact-4090 ada_linear for rows=1/2/4 hidden projections", "ada_wagv_lora for rows<=4",
            "exact-4090 BnB W8 native bridge", "exact-4090 batched MM4 output head",
        ),
        default_off=(
            "fused_output_project", "generic Triton projection/LoRA fusions",
            "RTX 4080 ada_linear and sparse FFN", "unmeasured Ada-card promotion",
        ),
        required_functional=COMMON_FUNCTIONAL_SMOKES,
        required_benchmarks=COMMON_PERF_BENCHMARKS
        + ("fast-prefill TTFT/TPOT rows when RWKV7_FAST_PREFILL is considered", "exact-card W8/W4 footprint, peak-VRAM and end-to-end speed rows"),
        quant_rule="RTX 4090 routes and RTX 4080 B1/B8 output-head A8W8/TorchAO-W4 routes have exact end-to-end rows; RTX 4080 full-model BNB8/BNB4 remains memory-only",
        promotion_rule="do not generalize one Ada card's shapes or tiles without exact-card correctness and speed rows",
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


def classify_gpu(
    name: str | None,
    capability: tuple[int, int] | None,
    *,
    is_hip: bool = False,
    is_mps: bool = False,
) -> GPUProfile:
    """Classify a GPU without requiring torch/CUDA to be available."""

    gpu_name = (name or "unknown").strip() or "unknown"
    lower = gpu_name.lower()
    if is_mps or any(token in lower for token in ("apple silicon", "apple m1", "apple m2", "apple m3", "apple m4", "apple m5")):
        return GPUProfile(name=gpu_name, vendor="apple", family="apple_mps", is_mps=True)
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
    try:
        cuda_available = bool(callable(is_available) and is_available())
    except Exception:
        cuda_available = False
    if not cuda_available:
        mps = getattr(getattr(torch_module, "backends", None), "mps", None)
        mps_available = getattr(mps, "is_available", None)
        if callable(mps_available):
            try:
                if bool(mps_available()):
                    return GPUProfile(
                        name="Apple Silicon MPS",
                        vendor="apple",
                        family="apple_mps",
                        is_mps=True,
                    )
            except Exception:
                pass
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
    if family == "apple_mps":
        return KernelPolicy(
            profile=profile,
            fast_token_backend="native",
            fast_cache=True,
            fused_recurrent_output=False,
            fused_output=False,
            fused_prefill_scan=False,
            quant_policy="apple_native_mlx_coreml",
            notes="Apple MPS: use native/no-FLA HF compatibility; CUDA/Triton fusions off; MLX/CoreML selected explicitly",
        )
    if family in {"amd_hip", "legacy_cuda", "pascal", "unknown_cuda"}:
        return KernelPolicy(
            profile=profile,
            fused_recurrent_output=False,
            fused_output=False,
            notes="compatibility-first: keep experimental Triton/native_graph fusions off; Pascal uses native/no-FLA fallback unless overridden",
        )
    if family == "volta":
        is_v100 = is_v100_name(profile.name)
        return KernelPolicy(
            profile=profile,
            fast_prefill=True,
            fused_recurrent_output=True,
            fused_recurrent_raw=True,
            fused_output=True,
            fused_prefill_scan=True,
            prefill_graph=True,
            prefill_graph_cache_size=4,
            prefill_graph_max_layers=32 if is_v100 else None,
            fused_prefill_shift_mix=True,
            fused_prefill_state_prep=True,
            fused_prefill_state_scan=True,
            fused_prefill_state_scan_max_batch=1,
            fused_prefill_output=True,
            fused_norm_mix=True,
            native_graph_max_layers=32 if is_v100 else None,
            fused_wavg_lora=True,
            wavg_lora_bsz1_max_hidden=4096,
            wavg_lora_blocks=(32, 64, 256),
            wavg_lora_num_warps=8,
            sm70_linear=True,
            sm70_wagv_lora=True,
            ada_sparse_ffn=True,
            ada_sparse_ffn_max_rows=4,
            ada_sparse_ffn_inplace=True,
            ada_sparse_ffn_up=False,
            # The sparse FFN kernel consumes the down projection transposed.
            # Reuse the original parameter storage instead of retaining one
            # additional full FFN-down copy per layer (about 4 GiB on 7.2B).
            # Exact-card V100 acceptance measured both lower peak VRAM and a
            # non-negative prefill/decode speed change with this layout.
            ada_sparse_ffn_low_memory_pack=is_v100,
            output_project_block_m=16,
            quant_policy="memory_first_decode_hot_optional",
            notes=(
                "V100 production path: graph capture is automatic through 32 layers and falls back to native_jit for larger checkpoints; four-shape prefill graph cache, fused shift mix, tuned WAVG/WAGV, low-memory sparse FFN, shape-routed sm70 linear/RKV, output/recurrent-output, and decode norm/mix are default; full projection/output-project remain opt-in"
                if is_v100
                else "Unmeasured Volta card: preserve the pre-existing sm70 fused defaults, but do not inherit V100-only graph ceilings or destructive low-memory packing"
            ),
        )
    if family == "turing":
        is_tesla_t4 = is_tesla_t4_name(profile.name)
        return KernelPolicy(
            profile=profile,
            fast_prefill=is_tesla_t4,
            fused_recurrent_output=True,
            fused_output=True,
            fused_prefill_scan=is_tesla_t4,
            output_project_block_m=16,
            notes=(
                "Tesla T4: use safe native prefill because the measured FLA 0.5 / Triton 3.3 chunk kernel fails sm_75 lowering; "
                "the exact-card prompt512 matrix promotes fused prefill scan; projection/LoRA kernels stay off"
                if is_tesla_t4
                else "Turing: use stable output fusions; require exact-card rows before native prefill or projection/LoRA defaults"
            ),
        )
    if family == "ampere":
        is_3090 = "3090" in profile.name.lower()
        return KernelPolicy(
            profile=profile,
            fast_prefill=is_3090,
            bnb_skip_policy="memory",
            bnb_int8_threshold=0.0 if is_3090 else None,
            native_external_quant_prefill=is_3090,
            native_external_quant_graph=is_3090,
            # Threshold-zero BnB projection kernels and the fused activation
            # preparation route are graph-safe on the exact RTX 3090 lane.
            native_external_quant_prefill_graph=is_3090,
            native_bnb8_direct=is_3090,
            native_bnb8_relu_quant=is_3090,
            native_bnb8_rkv_mix_quant=is_3090,
            native_bnb8_ffn_mix_quant=is_3090,
            native_bnb8_attn_mix_block=4096 if is_3090 else 1024,
            native_bnb8_ffn_mix_block=2048 if is_3090 else 1024,
            a8w8_gemv_max_rows=8 if is_3090 else 1,
            # Exact 4096x65536 lm-head sweep at fixed 1800 MHz. B1 improves
            # 0.640 -> 0.385 ms; B2 uses the tensor-core batch kernel at
            # 0.238 ms instead of duplicating a GEMV launch per row.
            mm4_fused_max_rows=16 if is_3090 else None,
            mm4_gemv_block_pairs=128 if is_3090 else None,
            mm4_gemv_block_n=128 if is_3090 else None,
            mm4_dot_min_rows=2 if is_3090 else None,
            mm4_dot_block_b=16 if is_3090 else None,
            mm4_dot_block_pairs=64 if is_3090 else None,
            mm4_dot_block_n=64 if is_3090 else None,
            mm4_dot_warps=4 if is_3090 else None,
            fused_recurrent_output=True,
            fused_output=True,
            fused_prefill_scan=is_3090,
            fused_prefill_self_chunk=is_3090,
            prefill_self_chunk_min_tokens=1024,
            # Exact RTX 3090 7.2B sweep: P2048/B2 favors chunk-16 while B4
            # favors chunk-32; the short promoted shapes also retain chunk-16.
            prefill_self_chunk_size=32,
            prefill_self_chunk_shape_sizes=(
                ((2, 512, 16), (2, 2048, 16), (8, 128, 16)) if is_3090 else ()
            ),
            prefill_self_chunk_h_tile_shapes=(
                ((4, 2048, 16, 16),) if is_3090 else ()
            ),
            prefill_self_chunk_model_shapes=(
                (
                    (4096, 32, 1, 512),
                    (4096, 32, 2, 512),
                    (4096, 32, 4, 512),
                    (4096, 32, 8, 512),
                    (4096, 32, 8, 128),
                )
                if is_3090
                else ()
            ),
            prefill_scan_block_m=8 if is_3090 else None,
            prefill_scan_block_m_b2=8 if is_3090 else None,
            prefill_scan_block_m_b4=8 if is_3090 else None,
            prefill_scan_num_warps=4 if is_3090 else None,
            prefill_blas_library="cublaslt" if is_3090 else None,
            prefill_blas_large_library="cublas" if is_3090 else None,
            prefill_blas_large_min_rows=4096,
            prefill_graph=is_3090,
            prefill_graph_cache_size=4 if is_3090 else 2,
            fused_prefill_shift_mix=is_3090,
            fused_prefill_state_prep=is_3090,
            fused_prefill_output=is_3090,
            fused_prefill_residual_gemm=is_3090,
            fused_prefill_stacked_rkv=is_3090,
            prefill_stacked_rkv_min_rows=192 if is_3090 else 128,
            prefill_stacked_rkv_max_rows=384 if is_3090 else None,
            prefill_stacked_rkv_extra_rows=(),
            # Exact RTX 3090 7.2B/Qwen3.5-9B A/B. B8/P512 deliberately uses
            # separate GEMMs: it is faster and avoids the 3 GiB R/K/V pack.
            prefill_stacked_rkv_model_shapes=(
                (
                    (4096, 32, 1, 512),
                    (4096, 32, 2, 512),
                    (4096, 32, 4, 512),
                    (4096, 32, 4, 128),
                )
                if is_3090
                else ()
            ),
            fused_prefill_sequence_ffn=is_3090,
            prefill_sequence_ffn_min_rows=192 if is_3090 else 128,
            prefill_sequence_ffn_max_rows=384 if is_3090 else None,
            prefill_sequence_ffn_extra_rows=(),
            prefill_sequence_ffn_model_shapes=(
                (
                    (4096, 32, 2, 2048),
                    (4096, 32, 8, 512),
                )
                if is_3090
                else ()
            ),
            prefill_sequence_ffn_blocks=(64, 64, 32, 64, 8) if is_3090 else (128, 128, 32, 64, 8),
            prefill_sequence_ffn_large_min_rows=1024,
            prefill_sequence_ffn_large_blocks=(128, 128, 32, 64, 8),
            prefill_sequence_ffn_num_stages=4 if is_3090 else 3,
            prefill_sequence_ffn_num_warps=8 if is_3090 else 4,
            output_project_block_m=16,
            notes=(
                "RTX 3090: measured cublasLt + row-8 scan, sequence shift-mix, state-prep, "
                "output-prep, row-8 scan, shape-routed DPLR/stacked R/K/V/sequence FFN, fused BnB W8 activation preparation, native quant prefill/decode, and memory-first bnb routing; "
                "other CUDA tensor-core cards retain stable output fusions pending a local sweep"
                if is_3090
                else "CUDA tensor-core generation: use stable output fusions; require local sweep before projection/LoRA defaults"
            ),
        )
    if family == "ada":
        is_4090 = "4090" in profile.name.lower()
        is_4080 = "4080" in profile.name.lower()
        rtx4080_prefill_shapes = (
            tuple(
                (hidden, 24, batch, tokens)
                for hidden in (1024, 2048)
                for batch in (1, 2, 4, 8)
                for tokens in (128, 512, 2048)
            )
            + tuple(
                (2560, 32, batch, tokens)
                for batch in (1, 8)
                for tokens in (128, 512, 2048)
            )
            if is_4080
            else ()
        )
        return KernelPolicy(
            profile=profile,
            fast_prefill=is_4090 or is_4080,
            bnb_skip_policy="memory",
            # The exact RTX 4090 W8 lane is graph-safe with threshold zero.
            # It removes the host-synchronizing BnB outlier branch and is a
            # prerequisite for the measured native prefill/decode bridge.
            bnb_int8_threshold=0.0 if is_4090 else None,
            native_external_quant_prefill=is_4090,
            native_external_quant_graph=is_4090,
            native_external_quant_prefill_graph=is_4090,
            native_bnb8_direct=is_4090,
            native_bnb8_relu_quant=is_4090,
            native_bnb8_rkv_mix_quant=is_4090,
            native_bnb8_ffn_mix_quant=is_4090,
            native_bnb8_attn_mix_block=4096 if is_4090 else 1024,
            native_bnb8_ffn_mix_block=2048 if is_4090 else 1024,
            # Exact bsz8 4090 output-head route. One tensor-core batch launch
            # avoids eight independently captured W4 GEMV kernels and their
            # graph-pool pressure.
            mm4_fused_max_rows=16 if is_4090 else None,
            mm4_gemv_block_pairs=128 if is_4090 else None,
            mm4_gemv_block_n=128 if is_4090 else None,
            mm4_dot_min_rows=2 if is_4090 else None,
            mm4_dot_block_b=16 if is_4090 else None,
            mm4_dot_block_pairs=64 if is_4090 else None,
            mm4_dot_block_n=64 if is_4090 else None,
            mm4_dot_warps=4 if is_4090 else None,
            # Exact 4090 sweeps: row-32 wins at B8/P128 across the measured
            # models.  The 1.5B (hidden=2048) also needs row-32 at B8/P512;
            # larger checkpoints retain row-8 for P512/chunk-512 P2048.
            prefill_scan_block_m_shapes=((8, 128, 32),) if is_4090 else (),
            prefill_scan_block_m_model_shapes=(
                ((2048, 8, 512, 32),)
                if is_4090
                else (
                    (2048, 1, 128, 4),
                    (2048, 1, 512, 4),
                    (2048, 1, 2048, 4),
                )
                if is_4080
                else ()
            ),
            fused_recurrent_output=True,
            fused_recurrent_raw=True,
            fused_output=True,
            fused_norm_mix=True,
            norm_mix_num_warps=8 if is_4090 else 4,
            fused_prefill_scan=is_4090 or is_4080,
            fused_prefill_self_chunk=is_4080,
            prefill_self_chunk_min_tokens=1024,
            prefill_self_chunk_size=32,
            prefill_self_chunk_shape_sizes=(
                ((1, 512, 32), (1, 2048, 32)) if is_4080 else ()
            ),
            prefill_self_chunk_h_tile_shapes=(
                ((1, 512, 32, 32), (1, 2048, 32, 32)) if is_4080 else ()
            ),
            prefill_self_chunk_model_shapes=(
                ((2048, 24, 1, 512), (2048, 24, 1, 2048)) if is_4080 else ()
            ),
            prefill_self_chunk_model_shapes_only=is_4080,
            prefill_scan_model_shapes=rtx4080_prefill_shapes,
            prefill_graph=is_4090 or is_4080,
            prefill_graph_cache_size=4 if is_4080 else 2,
            prefill_graph_model_shapes=rtx4080_prefill_shapes,
            fused_prefill_shift_mix=is_4090 or is_4080,
            prefill_shift_mix_model_shapes=rtx4080_prefill_shapes,
            prefill_attn_shift_mix_launch_profiles=(
                (
                    (2048, 24, 1, 512, 512, 1),
                    (2048, 24, 1, 2048, 512, 1),
                )
                if is_4080
                else ()
            ),
            prefill_ffn_shift_mix_launch_profiles=(
                ((2048, 24, 1, 512, 1024, 1),) if is_4080 else ()
            ),
            fused_prefill_stacked_rkv=is_4080,
            prefill_stacked_rkv_min_rows=1 if is_4080 else 128,
            prefill_stacked_rkv_max_rows=1 if is_4080 else None,
            prefill_stacked_rkv_model_shapes=(
                ((2048, 24, 1, 2048),) if is_4080 else ()
            ),
            fused_prefill_state_prep=is_4090 or is_4080,
            prefill_state_prep_model_shapes=rtx4080_prefill_shapes,
            fused_prefill_output=is_4090 or is_4080,
            prefill_fused_output_model_shapes=rtx4080_prefill_shapes,
            ada_linear=not is_4080,
            ada_linear_rows="1 2 4" if is_4090 else "2 4",
            ada_wagv_lora=True,
            ada_sparse_ffn=is_4090,
            ada_sparse_ffn_max_rows=2 if is_4090 else 19,
            ada_sparse_ffn_inplace=is_4090,
            rkv_policy="vkwr_auto" if is_4090 else "manual",
            output_project_block_m=16,
            notes=(
                "RTX 4080: exact 0.4B/1.5B fp16 rows promote B=1/2/4/8 and exact "
                "2.9B rows promote B=1/8 at T=128/512/2048; 1.5B/B1/P512 and P2048 use "
                "exact-card self-chunk routes, with stacked R/K/V at P2048; grouped W/A/G/V remains enabled for "
                "rows<=4 while the regressing Ada linear route stays disabled"
                if is_4080
                else "RTX 40/Ada: exact-4090 rows promote fixed-shape prefill graph plus raw recurrent decode, 8-warp norm/mix, rows=1/2/4 exact linear, stacked-copy-free R/K/V including layer 0, graph-safe one/two-row sparse FFN, threshold-zero BnB W8 native prefill/decode, and bsz8 tensor-core MM4 output-head dispatch; other Ada cards retain the compatible fallback until measured"
            ),
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
        is_5090 = "5090" in profile.name.lower()
        production_prefill_graph_shapes = (
            (2560, 32, 1, 128),
            (2560, 32, 1, 512),
            (2560, 32, 1, 2048),
            (2560, 32, 8, 128),
            (2560, 32, 8, 512),
            (2560, 32, 8, 2048),
            (4096, 61, 1, 128),
            (4096, 61, 1, 512),
            (4096, 61, 1, 2048),
            (4096, 61, 8, 128),
            (4096, 61, 8, 512),
        )
        g1h_13b_prefill_shapes = (
            (4096, 61, 1, 128),
            (4096, 61, 1, 512),
            (4096, 61, 1, 2048),
            (4096, 61, 8, 128),
            (4096, 61, 8, 512),
            (4096, 61, 8, 2048),
        )
        return KernelPolicy(
            profile=profile,
            fused_recurrent_output=True,
            fused_output=True,
            fused_prefill_scan=is_5090,
            prefill_graph=is_5090,
            prefill_graph_model_shapes=production_prefill_graph_shapes if is_5090 else (),
            prefill_fp16_recurrent=is_5090,
            # Exact RTX 5090 B8 sweeps on g1h 1.5B/P512 and 7.2B/P128. The
            # fused prefill route remains opt-in globally; these shape gates
            # only select combinations measured end to end on this card.
            prefill_scan_block_m_model_shapes=((2048, 8, 512, 8),) if is_5090 else (),
            fused_prefill_shift_mix=is_5090,
            prefill_shift_mix_model_shapes=(
                (2048, 24, 8, 128),
                (2048, 24, 8, 512),
                (2048, 24, 8, 2048),
                *g1h_13b_prefill_shapes,
            ) if is_5090 else (),
            prefill_attn_shift_mix_strict_fp16_model_shapes=(
                (4096, 61, 1, 128),
                (4096, 61, 1, 512),
                (4096, 61, 1, 2048),
                (4096, 61, 8, 128),
            ) if is_5090 else (),
            prefill_ffn_shift_mix_strict_fp16_model_shapes=(
                (4096, 61, 1, 128),
            ) if is_5090 else (),
            prefill_attn_shift_mix_launch_profiles=tuple(
                (*shape, 2048, 8) for shape in g1h_13b_prefill_shapes
            ) if is_5090 else (),
            prefill_ffn_shift_mix_launch_profiles=tuple(
                (*shape, 2048, 8) for shape in g1h_13b_prefill_shapes
            ) if is_5090 else (),
            fused_prefill_state_prep=is_5090,
            prefill_state_prep_model_shapes=(
                (2048, 24, 8, 512),
                (2048, 24, 8, 2048),
                *g1h_13b_prefill_shapes,
            ) if is_5090 else (),
            prefill_state_prep_layer_counts=(
                (2048, 24, 8, 512, 24),
                (2048, 24, 8, 2048, 18),
            ) if is_5090 else (),
            fused_prefill_state_scan=is_5090,
            fused_prefill_state_scan_max_batch=1 if is_5090 else None,
            fused_prefill_output=is_5090,
            prefill_fused_output_model_shapes=(
                (4096, 61, 1, 128),
                (4096, 61, 1, 2048),
                (4096, 61, 8, 128),
                (4096, 61, 8, 512),
                (4096, 61, 8, 2048),
            ) if is_5090 else (),
            fused_prefill_residual_gemm=is_5090,
            prefill_clampw_scan_model_shapes=((2048, 24, 8, 512),) if is_5090 else (),
            fused_prefill_stacked_rkv=is_5090,
            prefill_stacked_rkv_min_rows=1,
            prefill_stacked_rkv_max_rows=1,
            prefill_stacked_rkv_model_shapes=(
                (4096, 32, 8, 128),
            ) if is_5090 else (),
            fused_prefill_sequence_ffn=is_5090,
            prefill_sequence_ffn_min_rows=1,
            prefill_sequence_ffn_max_rows=1,
            prefill_sequence_ffn_model_shapes=(
                (2048, 24, 8, 128),
                (2048, 24, 8, 512),
                (2048, 24, 8, 2048),
            ) if is_5090 else (),
            prefill_sequence_ffn_large_blocks=(64, 128, 32, 64, 8),
            prefill_sequence_ffn_num_stages=3,
            prefill_sequence_ffn_num_warps=8 if is_5090 else 4,
            # RTX 5090 / B8 / P128: limiting reduced-precision accumulation to
            # measured FFN-key layers keeps strict official FP16-state tensor
            # gates while closing the final short-prompt prefill gaps.
            prefill_fp16_accum_ffn_key_model_shapes=(
                (2560, 32, 8, 128),
                (4096, 32, 8, 128),
                (4096, 61, 1, 128),
            ) if is_5090 else (),
            prefill_fp16_accum_ffn_key_layer_counts=(
                (2560, 32, 8, 128, 28),
                (4096, 61, 1, 128, 12),
            ) if is_5090 else (),
            fused_norm_mix=is_5090,
            norm_mix_num_warps=8 if is_5090 else 4,
            native_graph_state_dtype="fp16" if is_5090 else "fp32",
            native_graph_fp16_recurrent=is_5090,
            native_graph_precompute_embedding=is_5090,
            ada_linear=is_5090,
            ada_linear_rows="1" if is_5090 else "2 4",
            ada_linear_roles="hidden,ffn_up,ffn_down" if is_5090 else "auto",
            ada_wagv_lora=is_5090,
            ada_wag_lora=is_5090,
            ada_sparse_ffn=is_5090,
            ada_sparse_ffn_max_rows=19,
            ada_sparse_ffn_up=True,
            ada_sparse_ffn_low_memory_pack=is_5090,
            ada_sparse_ffn_share_pack=is_5090,
            ada_sparse_ffn_deterministic_splits=4 if is_5090 else 0,
            ada_sparse_ffn_official_boundary=is_5090,
            blackwell_cmix=is_5090,
            rkv_policy="manual",
            marlin_w4_ffn_shapes=(
                (8192, 2048),
                (2048, 8192),
                (10240, 2560),
                (2560, 10240),
                (16384, 4096),
                (4096, 16384),
            ) if is_5090 else (),
            marlin_w4_model_profiles=(
                (2048, 8192, 24, 128, False, 1),
                (2560, 10240, 32, 128, False, 0),
                (4096, 16384, 32, 128, True, 0),
                (4096, 16384, 61, 128, True, 1),
            ) if is_5090 else (),
            output_project_block_m=32,
            notes="RTX 50/Blackwell: exact RTX 5090 rows promote the official-FP16-state native graph decode profile and allowlisted 2.9B/13.3B B1/B8 prefill shapes. The 13.3B B8/P2048 row intentionally stays outside the graph allowlist because graph-private pools exceed 32 GiB; its measured eager fused route remains active. Existing 1.5B/2.9B/7.2B shape-specific prefill and quant routes remain exact-card gates. Other Blackwell cards retain the compatible fallback; use triton_compat for early sm_120 stacks and keep unvalidated projection/LoRA fusions off",
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
