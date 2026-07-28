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


def single_cuda_device_from_device_map(
    device_map: Any,
) -> tuple[bool, int | str | None]:
    """Resolve an unambiguous CUDA target for load-time hardware policy.

    ``None`` means the caller did not request placement and may use the
    process's current CUDA device.  Automatic, CPU-only, or multi-CUDA maps
    return ``(False, None)`` so exact-card quantization defaults fail closed
    instead of inheriting CUDA device 0 by accident.
    """

    if device_map is None:
        return True, None
    values = list(device_map.values()) if isinstance(device_map, dict) else [device_map]
    cuda_devices: set[str] = set()
    for value in values:
        if isinstance(value, bool):
            return False, None
        if isinstance(value, int):
            cuda_devices.add(f"cuda:{int(value)}")
            continue
        device_type = getattr(value, "type", None)
        device_index = getattr(value, "index", None)
        if device_type is not None:
            if str(device_type).lower() == "cuda":
                cuda_devices.add(
                    "cuda" if device_index is None else f"cuda:{int(device_index)}"
                )
            continue
        text = str(value).strip().lower()
        if text.isdigit():
            cuda_devices.add(f"cuda:{int(text)}")
        elif text == "cuda" or text.startswith("cuda:"):
            cuda_devices.add(text)
        elif text in {"auto", "balanced", "balanced_low_0", "sequential"}:
            return False, None
        # cpu, disk and mps placements are offload targets and do not identify
        # the CUDA card whose exact kernel/quant policy should be selected.
    if len(cuda_devices) != 1:
        return False, None
    return True, next(iter(cuda_devices))


def _gpu_name_tokens(name: str) -> tuple[str, ...]:
    """Return normalized product-name tokens for exact-card policy gates."""

    normalized = "".join(
        character if character.isalnum() else " "
        for character in str(name).lower()
    )
    return tuple(normalized.split())


def is_rtx_model_name(name: str, model: str) -> bool:
    """Match an exact desktop RTX model without accepting adjacent products.

    NVIDIA device strings often add ``GeForce`` and a trailing ``GPU``.  Those
    words are harmless, but Laptop, SUPER, Ti, Max-Q and similar suffixes
    identify different products whose measured launch policy must not leak.
    """

    tokens = _gpu_name_tokens(name)
    model_token = str(model).lower()
    if "rtx" not in tokens or model_token not in tokens:
        return False
    model_index = tokens.index(model_token)
    suffix = tokens[model_index + 1 :]
    return bool(
        not {"laptop", "mobile", "maxq", "max", "q", "super", "ti"}.intersection(tokens)
        and all(token == "gpu" for token in suffix)
    )


def is_rtx_laptop_model_name(name: str, model: str) -> bool:
    """Match one exact RTX Laptop product without accepting desktop/Ti variants."""

    tokens = _gpu_name_tokens(name)
    model_token = str(model).lower()
    if "rtx" not in tokens or model_token not in tokens or "laptop" not in tokens:
        return False
    model_index = tokens.index(model_token)
    suffix = tokens[model_index + 1 :]
    return bool(
        not {"mobile", "maxq", "max", "q", "super", "ti"}.intersection(tokens)
        and suffix in (("laptop",), ("laptop", "gpu"))
    )


def is_tesla_t4_name(name: str) -> bool:
    """Match the exact T4 product token without accepting names like T400."""

    return "t4" in _gpu_name_tokens(name)


def is_tesla_v100_name(name: str) -> bool:
    """Match V100 product strings without promoting other sm_70 cards."""

    return "v100" in _gpu_name_tokens(name)


def is_mtt_s70_name(name: str) -> bool:
    """Match the exact first-generation MTT S70 product tokens."""

    tokens = _gpu_name_tokens(name)
    return "mtt" in tokens and "s70" in tokens


def is_metax_c500_name(name: str) -> bool:
    """Match the exact C500 product without capability-family guessing."""

    return _gpu_name_tokens(name) == ("metax", "c500")


def is_biren_br106m_name(name: str) -> bool:
    """Match the exact BR106M runtime product with harmless separators."""

    return "".join(_gpu_name_tokens(name)) == "biren106m"


def _musa_hardware_metadata(name: str) -> tuple[str, str, str]:
    """Return generation, evidence scope, and compute profile for MUSA.

    The available development card is a legacy first-generation MTT S70. Its
    exact-card evidence must not become a capability ceiling or default for
    later MUSA accelerators such as MTT S4000/S5000, which require their own
    runtime probes and real-device acceptance rows.
    """

    tokens = _gpu_name_tokens(name)
    if is_mtt_s70_name(name):
        return (
            "musa_legacy_s70",
            "exact_card_smoke",
            "fp32_compute_fp16_io",
        )
    if "mtt" in tokens and any(model in tokens for model in ("s4000", "s5000")):
        return (
            "musa_post_s70",
            "unvalidated",
            "device_specific_unvalidated",
        )
    return ("musa_unknown", "unvalidated", "device_specific_unvalidated")


@dataclass(frozen=True)
class GPUProfile:
    """Normalized hardware identity used by the kernel policy."""

    name: str
    vendor: str
    family: str
    capability: tuple[int, int] | None = None
    architecture: str | None = None
    device_index: int | None = None
    is_cuda: bool = False
    is_hip: bool = False
    is_mps: bool = False
    is_musa: bool = False
    is_supa: bool = False
    hardware_generation: str | None = None
    validation_scope: str = "unvalidated"
    compute_profile: str = "unknown"


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
    a8w8_fused_ffn: bool = False
    mm8_fused_max_rows: int | None = None
    mm8_dot_min_rows: int | None = None
    mm8_dot_block_b: int | None = None
    mm8_dot_block_m: int | None = None
    mm8_dot_block_n: int | None = None
    mm8_dot_warps: int | None = None
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
    # Safe dynamic HxL envelopes for recurrent prefill scan.  Each profile is
    # (hidden, layers, max_batch, max_prompt_tokens, max_total_tokens).  These
    # profiles deliberately do not promote CUDA Graph capture, reduced-
    # precision accumulation, or exact-shape launch schedules.
    prefill_scan_model_profiles: tuple[tuple[int, int, int, int, int], ...] = ()
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
    prefill_fp16_recurrent: bool = False
    fused_prefill_shift_mix: bool = False
    prefill_shift_mix_model_shapes: tuple[tuple[int, int, int, int], ...] = ()
    prefill_shift_mix_model_profiles: tuple[tuple[int, int, int, int, int], ...] = ()
    prefill_attn_shift_mix_strict_fp16_model_shapes: tuple[tuple[int, int, int, int], ...] = ()
    prefill_ffn_shift_mix_strict_fp16_model_shapes: tuple[tuple[int, int, int, int], ...] = ()
    # hidden, layers, batch, tokens, block size, warps
    prefill_attn_shift_mix_launch_profiles: tuple[tuple[int, int, int, int, int, int], ...] = ()
    prefill_ffn_shift_mix_launch_profiles: tuple[tuple[int, int, int, int, int, int], ...] = ()
    fused_prefill_state_prep: bool = False
    prefill_state_prep_model_shapes: tuple[tuple[int, int, int, int], ...] = ()
    prefill_state_prep_model_profiles: tuple[tuple[int, int, int, int, int], ...] = ()
    # hidden, layers, batch, tokens, enabled leading-layer count
    prefill_state_prep_layer_counts: tuple[tuple[int, int, int, int, int], ...] = ()
    fused_prefill_state_scan: bool = False
    fused_prefill_state_scan_max_batch: int | None = None
    fused_prefill_output: bool = False
    prefill_fused_output_model_shapes: tuple[tuple[int, int, int, int], ...] = ()
    prefill_fused_output_model_profiles: tuple[tuple[int, int, int, int, int], ...] = ()
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
    # Exact HxLxBxT fp16 prefill routes where every dense GEMM may use fp16
    # accumulation. This is intentionally separate from the narrower FFN-key
    # lane because it changes every cuBLAS projection in the sequence graph.
    prefill_global_fp16_accum_model_shapes: tuple[tuple[int, int, int, int], ...] = ()
    # Exact HxLxBxT routes where transformer-block GEMMs use fp16
    # accumulation, while the final norm and vocabulary head retain the
    # default FP32 accumulation. This narrower boundary can preserve greedy
    # parity for shapes where the full-prefill route crosses a token boundary.
    prefill_block_fp16_accum_model_shapes: tuple[tuple[int, int, int, int], ...] = ()
    fused_recurrent_output: bool = False
    fused_recurrent_raw: bool = False
    # Exact hidden-size x batch routes for raw recurrent preparation. Empty
    # preserves the historical card-wide behavior on existing policies.
    native_graph_fused_recurrent_raw_shapes: tuple[tuple[int, int], ...] = ()
    recurrent_raw_num_warps: int = 8
    fused_output: bool = False
    fused_norm_mix: bool = False
    # Exact hidden-size x batch routes for decode norm/mix fusion. Empty keeps
    # the historical card-wide behavior for already validated policies.
    native_graph_fused_norm_mix_shapes: tuple[tuple[int, int], ...] = ()
    norm_mix_num_warps: int = 4
    native_graph_state_dtype: str = "fp32"
    native_graph_fp16_recurrent: bool = False
    native_graph_triton_fp16_state: bool = False
    # hidden, layers, batch
    native_graph_triton_fp16_state_model_shapes: tuple[tuple[int, int, int], ...] = ()
    native_graph_precompute_embedding: bool = False
    sm70_linear: bool = False
    sm70_wagv_lora: bool = False
    ada_linear: bool = False
    ada_linear_rows: str = "2 4"
    ada_linear_roles: str = "auto"
    ada_wagv_lora: bool = False
    ada_wagv_lora_max_rows: int = 4
    ada_wagv_bmm: bool = False
    # Exact SM120/B8/H1024+H2048 torch.compile dense FFN. This remains
    # explicit-only until full-model horizon and paired-matrix evidence lands.
    sm120_compiled_ffn: bool = False
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
    wavg_lora_b8_blocks: tuple[int, int, int] | None = None
    wavg_lora_b8_num_warps: int | None = None
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
    "musa": GPUAdaptationRule(
        family="musa",
        cards=("legacy first-generation MTT S70", "unvalidated later MUSA devices including MTT S4000/S5000"),
        status="MTT S70 exact-card narrow HF smoke and paired evidence validated; later MUSA generations remain unvalidated",
        default_stance="backend compatibility only; S70-specific routes remain exact-card gated and must not constrain or promote later MUSA hardware",
        default_on=("fast_cache", "S70-only optional fp16-IO/fp32-compute MUSA WKV kernel"),
        default_off=("unvalidated MUSA card-specific kernels", "CUDA native_graph fused kernels", "Triton/FLA", "bnb CUDA-only quantization"),
        required_functional=(
            "MUSA import/load/generate",
            "cache continuation and chunked prefill",
            "MUSA WKV vs pure-PyTorch recurrence",
            "PEFT/Trainer smoke when training is claimed",
        ),
        required_benchmarks=(
            "exact MUSA device/runtime/model/dtype rows",
            "prefill/decode/peak-memory plus selected-route evidence",
        ),
        quant_rule="no torch quantization claim without exact MUSA operator, quality, footprint and speed evidence",
        promotion_rule="do not infer CUDA/ROCm behavior or later-card capability from S70; promote only behavior documented by MUSA sources and proven on the exact device",
    ),
    "metax": GPUAdaptationRule(
        family="metax",
        cards=("MetaX C500 64GB",),
        status="exact-card HF compatibility and 0.4B real-checkpoint smoke validated in the standalone repository",
        default_stance="MXMACA torch.cuda compatibility with native eager/no-FLA routing; never inherit NVIDIA CUDA kernels from reported capability 8.0",
        default_on=("native eager", "recurrent cache", "chunked prefill"),
        default_off=(
            "native JIT/graph",
            "NVIDIA CUDA extensions",
            "Triton/FLA fusions",
            "unvalidated W8/W4 speed routes",
        ),
        required_functional=COMMON_FUNCTIONAL_SMOKES
        + ("CPU-oracle FP16/BF16", "HF Trainer", "PEFT save/load/merge"),
        required_benchmarks=(
            "same-card RWKV-LM/Albatross B1/B2/B4/B8 prefill/decode",
            "W8/W4 footprint, quality and speed before promotion",
        ),
        quant_rule="no production W8/W4 route is promoted from the current compatibility evidence",
        promotion_rule="require a current-main exact-C500 rerun; do not treat CUDA capability 8.0 as NVIDIA hardware evidence",
    ),
    "biren": GPUAdaptationRule(
        family="biren",
        cards=("Biren106M 32GB", "unvalidated other BIRENSUPA devices"),
        status="all released 0.1B-13.3B checkpoints have exact-card standalone HF functional evidence",
        default_stance="BF16 model math, FP32 recurrent state and native eager/no-FLA execution; exact BR106M evidence must not promote other SUPA products",
        default_on=("native eager", "recurrent cache", "chunked prefill"),
        default_off=(
            "FP16 GEMM",
            "native JIT/graph/torch.compile",
            "CUDA/Triton/FLA fusions",
            "unvalidated W8/W4 routes",
        ),
        required_functional=COMMON_FUNCTIONAL_SMOKES
        + (
            "BF16 auto-load/generate for 0.1B-13.3B",
            "FP32 recurrent state",
            "PEFT/Trainer",
        ),
        required_benchmarks=(
            "same-card RWKV-LM/Albatross B1/B2/B4/B8 prefill/decode",
            "W8/W4 footprint, quality and speed before promotion",
        ),
        quant_rule="no production quantized route is promoted from functional-only evidence",
        promotion_rule="require a current-main exact-BR106M rerun; do not infer other SUPA devices from BR10x evidence",
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
        status="RTX 4090 promoted matrices plus 4080-route reproduction rows and exact RTX 4080 native/Qwen3.5/training/quant rows exist; unmeasured Ada cards remain card-local validation targets",
        default_stance="exact RTX 4090 and RTX 4080 shape-routed paths with compatible fallbacks elsewhere",
        default_on=(
            "fast_cache", "fused_recurrent_output", "fused_recurrent_raw", "fused_output",
            "fused_norm_mix", "exact-card prefill graph/scan policy",
            "exact-4080 prefill shift/state/output for measured 0.4B/1.5B shapes",
            "exact-4090 ada_linear for rows=1/2/4 hidden projections",
            "exact-4080/4090 grouped W/A/V BMM for rows=8",
            "exact-4090 block-scoped FP16 accumulation for measured B1/B8 prefill shapes",
            "exact-4090 1.5B/B1/P2048 self-chunk plus stacked R/K/V",
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
        status="not release-validated",
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
        status="gfx1100 / ROCm 7.2.1 native-HF compatibility, fused decode, measured fused-prefill scan rows, and output-head MM8/MM4 speed lanes validated; full-model quantization remains open",
        default_stance="pure PyTorch/native_model first; exact-gfx1100 measured prefill/decode fusions only; every other AMD architecture fails closed",
        default_on=(
            "fast_cache",
            "gfx1100 only: fused recurrent/output/norm-mix decode",
            "gfx1100 only: recurrent-scan prefill on exact model/BxT rows",
            "gfx1100 only: output-head MM8/MM4 speed route",
        ),
        default_off=("unmeasured AMD fused kernels/shapes", "bnb CUDA-only speed paths"),
        required_functional=COMMON_FUNCTIONAL_SMOKES
        + ("ROCm import/generate", "pure PyTorch/native_model forward/backward"),
        required_benchmarks=("ROCm smoke rows", "independent unfused-reference exact-GCN speed rows before promotion"),
        quant_rule="exact-gfx1100 output-head MM8/MM4 is the measured speed lane; full-model W8/W4/A8W8 stays a memory/experimental lane until every required phase is >= fp16",
        promotion_rule="never generalize a gfx architecture's kernels or launch policy to another AMD architecture without real-card rows",
    ),
}


def classify_gpu(
    name: str | None,
    capability: tuple[int, int] | None,
    *,
    is_hip: bool = False,
    is_mps: bool = False,
    is_musa: bool = False,
    is_supa: bool = False,
    architecture: str | None = None,
) -> GPUProfile:
    """Classify a GPU without requiring torch/CUDA to be available."""

    gpu_name = (name or "unknown").strip() or "unknown"
    lower = gpu_name.lower()
    if is_supa or is_biren_br106m_name(gpu_name):
        exact = is_biren_br106m_name(gpu_name)
        return GPUProfile(
            name=gpu_name,
            vendor="biren",
            family="biren",
            hardware_generation="br10x_br106m" if exact else "biren_unknown",
            validation_scope="exact_card_smoke" if exact else "unvalidated",
            compute_profile="bf16_model_fp32_state" if exact else "unvalidated",
            is_supa=True,
        )
    if is_musa or any(token in lower for token in ("moore threads", "mthreads", "mtt s70", "mtt s80", "mtt s4000", "mtt s5000")):
        generation, validation_scope, compute_profile = _musa_hardware_metadata(gpu_name)
        return GPUProfile(
            name=gpu_name,
            vendor="moore_threads",
            family="musa",
            hardware_generation=generation,
            validation_scope=validation_scope,
            compute_profile=compute_profile,
            is_musa=True,
        )
    if is_metax_c500_name(gpu_name):
        return GPUProfile(
            name=gpu_name,
            vendor="metax",
            family="metax",
            capability=capability,
            is_cuda=True,
            hardware_generation="metax_c500",
            validation_scope="exact_card_smoke",
            compute_profile="mxmaca_cuda_compatible",
        )
    if is_mps or any(token in lower for token in ("apple silicon", "apple m1", "apple m2", "apple m3", "apple m4", "apple m5")):
        return GPUProfile(name=gpu_name, vendor="apple", family="apple_mps", is_mps=True)
    if is_hip or any(token in lower for token in ("amd", "radeon", "instinct", "mi250", "mi300")):
        return GPUProfile(
            name=gpu_name,
            vendor="amd",
            family="amd_hip",
            capability=capability,
            architecture=(str(architecture).split(":", 1)[0].lower() if architecture else None),
            is_cuda=False,
            is_hip=True,
        )
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
    musa = getattr(torch_module, "musa", None)
    musa_is_available = getattr(musa, "is_available", None)
    try:
        musa_available = bool(callable(musa_is_available) and musa_is_available())
    except Exception:
        musa_available = False
    if musa_available:
        try:
            if device is None:
                musa_index = int(musa.current_device())
            else:
                resolved_index = torch_module.device(device).index
                musa_index = (
                    int(resolved_index)
                    if resolved_index is not None
                    else int(musa.current_device())
                )
        except Exception:
            musa_index = 0
        try:
            musa_name = str(musa.get_device_name(musa_index))
        except Exception:
            musa_name = "Moore Threads MUSA"
        profile = classify_gpu(musa_name, None, is_musa=True)
        return GPUProfile(
            name=profile.name,
            vendor=profile.vendor,
            family=profile.family,
            device_index=musa_index,
            hardware_generation=profile.hardware_generation,
            validation_scope=profile.validation_scope,
            compute_profile=profile.compute_profile,
            is_musa=True,
        )
    supa = getattr(torch_module, "supa", None)
    supa_is_available = getattr(supa, "is_available", None)
    try:
        supa_available = bool(
            callable(supa_is_available) and supa_is_available()
        )
    except Exception:
        supa_available = False
    if supa_available:
        try:
            if device is None:
                supa_index = int(supa.current_device())
            else:
                text = str(device).strip().lower()
                if text in {"biren", "supa"}:
                    supa_index = 0
                elif text.startswith(("biren:", "supa:")):
                    supa_index = int(text.split(":", 1)[1])
                else:
                    resolved_index = torch_module.device(device).index
                    supa_index = (
                        int(resolved_index)
                        if resolved_index is not None
                        else int(supa.current_device())
                    )
        except Exception:
            supa_index = 0
        try:
            supa_name = str(supa.get_device_name(supa_index))
        except Exception:
            supa_name = "BIRENSUPA device"
        profile = classify_gpu(supa_name, None, is_supa=True)
        return GPUProfile(
            name=profile.name,
            vendor=profile.vendor,
            family=profile.family,
            device_index=supa_index,
            hardware_generation=profile.hardware_generation,
            validation_scope=profile.validation_scope,
            compute_profile=profile.compute_profile,
            is_supa=True,
        )
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
        if device is None:
            index = int(cuda.current_device())
        else:
            index = torch_module.device(device).index
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
    architecture = None
    if is_hip:
        try:
            properties = cuda.get_device_properties(index)
            architecture = getattr(properties, "gcnArchName", None)
        except Exception:
            architecture = None
    profile = classify_gpu(
        name,
        capability,
        is_hip=is_hip,
        architecture=architecture,
    )
    return GPUProfile(
        name=profile.name,
        vendor=profile.vendor,
        family=profile.family,
        capability=profile.capability,
        architecture=profile.architecture,
        device_index=index,
        hardware_generation=profile.hardware_generation,
        validation_scope=profile.validation_scope,
        compute_profile=profile.compute_profile,
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
    if family == "musa":
        s70_validated = profile.hardware_generation == "musa_legacy_s70"
        return KernelPolicy(
            profile=profile,
            fast_token_backend="native",
            fast_cache=True,
            fused_recurrent_output=False,
            fused_output=False,
            fused_prefill_scan=False,
            quant_policy="musa_unvalidated",
            notes=(
                "MUSA exact-card S70: legacy SDK 4.2.0 lane with fp16 storage/IO "
                "and fp32 compute; optional S70-validated kernels remain fail-safe"
                if s70_validated
                else "MUSA unvalidated device: use conservative native/no-FLA compatibility; "
                "do not inherit legacy S70 compute limits or exact-card kernels"
            ),
        )
    if family == "metax":
        return KernelPolicy(
            profile=profile,
            fast_token_backend="native",
            fast_cache=True,
            fast_prefill=False,
            fused_recurrent_output=False,
            fused_recurrent_raw=False,
            fused_output=False,
            fused_norm_mix=False,
            fused_prefill_scan=False,
            prefill_graph=False,
            quant_policy="metax_unvalidated",
            notes=(
                "MetaX C500 uses MXMACA's torch.cuda-compatible API but must "
                "not inherit NVIDIA kernels from capability 8.0; the validated "
                "compatibility route is native eager/no-FLA"
            ),
        )
    if family == "biren":
        return KernelPolicy(
            profile=profile,
            fast_token_backend="native",
            fast_cache=True,
            fast_prefill=False,
            fused_recurrent_output=False,
            fused_recurrent_raw=False,
            fused_output=False,
            fused_norm_mix=False,
            fused_prefill_scan=False,
            prefill_graph=False,
            quant_policy="biren_unvalidated",
            notes=(
                "BIRENSUPA compatibility uses BF16 model math, FP32 recurrent "
                "state, decomposed GroupNorm, and native eager/no-FLA execution"
            ),
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
    if family == "amd_hip":
        is_gfx1100 = profile.architecture == "gfx1100"
        # The gfx1100 recurrent-scan route is promoted only for the exact
        # checkpoints and BxT rows retained in the ROCm 7.2.1 acceptance
        # matrix.  Other AMD architectures and unmeasured shapes continue to
        # use the compatibility-first native recurrence.
        gfx1100_prefill_shapes = (
            tuple(
                (hidden, layers, batch, tokens)
                for hidden, layers in ((1024, 24), (2048, 24), (2560, 32))
                for batch in (1, 2, 4, 8)
                for tokens in (32, 64, 128, 256, 512)
            )
            if is_gfx1100
            else ()
        )
        return KernelPolicy(
            profile=profile,
            # gfx1100's rocBLAS int8 path padded to 17 rows beats the direct
            # W8A16 Triton GEMV even at B1; zero disables that GEMV route.
            a8w8_gemv_max_rows=0 if is_gfx1100 else 1,
            a8w8_fused_ffn=is_gfx1100,
            mm8_fused_max_rows=16 if is_gfx1100 else None,
            mm8_dot_min_rows=2 if is_gfx1100 else None,
            mm8_dot_block_b=16 if is_gfx1100 else None,
            mm8_dot_block_m=256 if is_gfx1100 else None,
            mm8_dot_block_n=16 if is_gfx1100 else None,
            mm8_dot_warps=8 if is_gfx1100 else None,
            mm4_fused_max_rows=16 if is_gfx1100 else None,
            mm4_dot_min_rows=2 if is_gfx1100 else None,
            mm4_dot_block_b=16 if is_gfx1100 else None,
            mm4_dot_block_pairs=64 if is_gfx1100 else None,
            mm4_dot_block_n=32 if is_gfx1100 else None,
            mm4_dot_warps=2 if is_gfx1100 else None,
            fused_recurrent_output=is_gfx1100,
            fused_recurrent_raw=is_gfx1100,
            fused_output=is_gfx1100,
            fused_norm_mix=is_gfx1100,
            norm_mix_num_warps=4,
            fused_prefill_scan=is_gfx1100,
            prefill_scan_model_shapes=gfx1100_prefill_shapes,
            prefill_scan_block_m=64 if is_gfx1100 else None,
            prefill_scan_num_warps=8 if is_gfx1100 else None,
            notes=(
                "Exact gfx1100/ROCm 7.2.1 rows promote recurrent-output, raw recurrent, output-prep, four-warp norm/mix, measured recurrent-scan prefill shapes and output-head MM8/MM4 decode routes; paired A8W8 FFNs are available inside the opt-in memory lane, while unmeasured prefill shapes and full-model quant speed claims remain conservative"
                if is_gfx1100
                else "Unmeasured AMD GCN architecture: compatibility-first native path; fused prefill/decode and quant speed routes stay off"
            ),
        )
    if family in {"legacy_cuda", "pascal", "unknown_cuda"}:
        return KernelPolicy(
            profile=profile,
            fused_recurrent_output=False,
            fused_output=False,
            notes="compatibility-first: keep experimental Triton/native_graph fusions off; Pascal uses native/no-FLA fallback unless overridden",
        )
    if family == "volta":
        is_v100 = is_tesla_v100_name(profile.name)
        return KernelPolicy(
            profile=profile,
            fast_prefill=True,
            fused_recurrent_output=True,
            fused_recurrent_raw=True,
            fused_output=True,
            fused_prefill_scan=True,
            prefill_graph=True,
            prefill_graph_cache_size=4,
            fused_prefill_shift_mix=True,
            fused_prefill_state_prep=True,
            fused_prefill_state_scan=True,
            fused_prefill_state_scan_max_batch=1,
            fused_prefill_output=True,
            fused_norm_mix=True,
            native_graph_triton_fp16_state=is_v100,
            native_graph_triton_fp16_state_model_shapes=(
                ((1024, 24, 8), (2048, 24, 8), (4096, 32, 8)) if is_v100 else ()
            ),
            fused_wavg_lora=True,
            wavg_lora_bsz1_max_hidden=4096,
            wavg_lora_blocks=(32, 64, 256),
            wavg_lora_num_warps=8,
            # Keep the established B1/B2/B4 launch.  On V100, B8 benefits
            # from a smaller rank tile and four warps without changing the
            # fused kernel's numerical contract.
            wavg_lora_b8_blocks=(32, 32, 256) if is_v100 else None,
            wavg_lora_b8_num_warps=4 if is_v100 else None,
            sm70_linear=True,
            sm70_wagv_lora=True,
            ada_sparse_ffn=True,
            ada_sparse_ffn_max_rows=4,
            ada_sparse_ffn_inplace=True,
            ada_sparse_ffn_up=False,
            output_project_block_m=16,
            quant_policy="memory_first_decode_hot_optional",
            notes="V100 production path: four-shape prefill graph cache, fused shift mix, tuned WAVG/WAGV, sparse FFN, shape-routed sm70 linear/RKV, output/recurrent-output, decode norm/mix, and exact 0.4B/1.5B/7.2B B8 FP16 state are default; full projection/output-project remain opt-in",
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
        is_3090 = is_rtx_model_name(profile.name, "3090")
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
            # Exact RTX 3090 g1d/g1i full-model sweeps: row-32 reaches
            # ~78.9k tok/s for 0.4B B8/P512 and improves 1.5B B8/P128/P512.
            prefill_scan_block_m_model_shapes=(
                (
                    (1024, 8, 512, 32),
                    (2048, 8, 128, 32),
                    (2048, 8, 512, 32),
                    (2560, 1, 512, 32),
                    (2560, 8, 128, 64),
                    (2560, 8, 512, 64),
                )
                if is_3090
                else ()
            ),
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
            # PyTorch >=2.7 only. These exact shapes pass prompt and cache-
            # handoff cosine >=0.9999 plus greedy parity while closing their
            # parameter-adjusted Qwen3.5 prefill-PD cells.
            prefill_global_fp16_accum_model_shapes=(
                (
                    (1024, 24, 1, 512),
                    (1024, 24, 1, 2048),
                    (1024, 24, 8, 128),
                    (1024, 24, 8, 512),
                    (2048, 24, 1, 128),
                    (2048, 24, 1, 512),
                    (2048, 24, 1, 2048),
                    (2048, 24, 8, 128),
                    (2048, 24, 8, 512),
                    (2560, 32, 1, 128),
                    (2560, 32, 1, 512),
                    (2560, 32, 8, 128),
                    (2560, 32, 8, 512),
                    (4096, 32, 1, 128),
                    (4096, 32, 1, 512),
                    (4096, 32, 8, 128),
                    (4096, 32, 8, 512),
                )
                if is_3090
                else ()
            ),
            output_project_block_m=16,
            notes=(
                "RTX 3090: measured cublasLt + row-8 scan, sequence shift-mix, state-prep, "
                "output-prep, shape-routed row-32 scan/FP16 accumulation, DPLR/stacked R/K/V/sequence FFN, fused BnB W8 activation preparation, native quant prefill/decode, and memory-first bnb routing; "
                "other CUDA tensor-core cards retain stable output fusions pending a local sweep"
                if is_3090
                else "CUDA tensor-core generation: use stable output fusions; require local sweep before projection/LoRA defaults"
            ),
        )
    if family == "ada":
        is_4090 = is_rtx_model_name(profile.name, "4090")
        is_4080 = is_rtx_model_name(profile.name, "4080")
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
        rtx4080_dynamic_prefill_profiles = (
            (
                # Shape-safe FP32-state/fp32-accumulation fallback envelope.
                # Keep exact Graph, accumulation, self-chunk, and launch-tile
                # promotions in their existing HxLxBxT allowlists below.
                # max_total covers the complete B<=8, T<=4096 rectangle.
                # The full 4080 boundary sweep caught B8/T2049 falling just
                # eight rows beyond the former 16384 cap and reverting to the
                # 9.7x slower unfused reference route.
                (1024, 24, 8, 4096, 32768),
                (2048, 24, 8, 4096, 32768),
            )
            if is_4080
            else ()
        )
        rtx4080_global_fp16_accum_shapes = (
            # Exact parity sweep: these three shapes crossed a greedy-token
            # boundary only when the final vocabulary head also used FP16
            # accumulation. They use the block-only route below instead.
            tuple(
                shape
                for shape in rtx4080_prefill_shapes
                if shape
                not in {
                    (1024, 24, 8, 512),
                    (2048, 24, 8, 512),
                    (2560, 32, 1, 512),
                }
            )
            if is_4080
            else ()
        )
        rtx4080_block_fp16_accum_shapes = (
            (
                (1024, 24, 8, 512),
                (2048, 24, 8, 512),
                (2560, 32, 1, 512),
            )
            if is_4080
            else ()
        )
        rtx4090_block_fp16_accum_shapes = (
            tuple(
                (hidden, layers, batch, tokens)
                for hidden, layers in (
                    (1024, 24),
                    (2048, 24),
                    (2560, 32),
                    (4096, 32),
                )
                for batch in (1, 8)
                for tokens in (128, 512, 2048)
            )
            if is_4090
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
            # Exact RTX 4080 A8W8 validation: sub-tile output-head matrices use
            # the activation-stable W8A16 path. The tiled FP16 kernel is also
            # faster than padding these rows into cuBLASLt dynamic-A8 GEMM.
            a8w8_gemv_max_rows=32 if is_4080 else 1,
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
                    (2560, 1, 512, 8),
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
            fused_prefill_self_chunk=is_4090 or is_4080,
            prefill_self_chunk_min_tokens=1024,
            # Keep the exact 4080 row-32 tile card-local.  The 4090 acceptance
            # matrix explicitly selected row 16 when enabling self-chunk.
            prefill_self_chunk_size=32 if is_4080 else 16,
            prefill_self_chunk_shape_sizes=(
                ((1, 512, 32), (1, 2048, 32))
                if is_4080
                else ((1, 2048, 16),)
                if is_4090
                else ()
            ),
            prefill_self_chunk_h_tile_shapes=(
                ((1, 512, 32, 32), (1, 2048, 32, 32))
                if is_4080
                else ((1, 2048, 16, 16),)
                if is_4090
                else ()
            ),
            prefill_self_chunk_model_shapes=(
                ((2048, 24, 1, 512), (2048, 24, 1, 2048))
                if is_4080
                else ((2048, 24, 1, 2048),)
                if is_4090
                else ()
            ),
            prefill_self_chunk_model_shapes_only=is_4090 or is_4080,
            prefill_scan_model_shapes=rtx4080_prefill_shapes,
            prefill_scan_model_profiles=rtx4080_dynamic_prefill_profiles,
            prefill_graph=is_4090 or is_4080,
            prefill_graph_cache_size=4 if is_4080 else 2,
            prefill_graph_model_shapes=rtx4080_prefill_shapes,
            prefill_global_fp16_accum_model_shapes=rtx4080_global_fp16_accum_shapes,
            # The RTX 4090 same-process forward/reverse A/B selected the
            # block-only boundary on all measured 0.4B/1.5B/2.9B/7.2B B1/B8
            # P128/P512/P2048 shapes.  It retains FP32 accumulation for the
            # final norm and vocabulary head while recovering essentially the
            # same Tensor Core gain as process-global FP16 accumulation.
            prefill_block_fp16_accum_model_shapes=(
                rtx4080_block_fp16_accum_shapes
                + rtx4090_block_fp16_accum_shapes
            ),
            fused_prefill_shift_mix=is_4090 or is_4080,
            prefill_shift_mix_model_shapes=rtx4080_prefill_shapes,
            prefill_shift_mix_model_profiles=rtx4080_dynamic_prefill_profiles,
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
            fused_prefill_stacked_rkv=is_4090 or is_4080,
            prefill_stacked_rkv_min_rows=1 if is_4090 or is_4080 else 128,
            prefill_stacked_rkv_max_rows=1 if is_4090 or is_4080 else None,
            prefill_stacked_rkv_model_shapes=(
                ((2048, 24, 1, 2048),) if is_4090 or is_4080 else ()
            ),
            fused_prefill_state_prep=is_4090 or is_4080,
            prefill_state_prep_model_shapes=rtx4080_prefill_shapes,
            prefill_state_prep_model_profiles=rtx4080_dynamic_prefill_profiles,
            fused_prefill_output=is_4090 or is_4080,
            prefill_fused_output_model_shapes=rtx4080_prefill_shapes,
            prefill_fused_output_model_profiles=rtx4080_dynamic_prefill_profiles,
            ada_linear=not is_4080,
            ada_linear_rows="1 2 4" if is_4090 else "2 4",
            ada_wagv_lora=True,
            # Exact RTX 4090 B8 rows reproduce the 4080 grouped tensor-core
            # projection win across 0.4B/1.5B/2.9B.  Keep every adjacent Ada
            # product and every unmeasured batch on the existing fallback.
            # The BMM dispatch has its own exact B8/hidden-size gate and does
            # not require widening the generic grouped fallback.  Keep the
            # 4090 fallback at rows<=4 so unmeasured 7.2B/hidden=4096 B8 does
            # not silently inherit the 4080 grouped route.
            ada_wagv_lora_max_rows=8 if is_4080 else 4,
            ada_wagv_bmm=is_4090 or is_4080,
            # Exact RTX 4080 g1h-7.2B/B8 decode: keeping the recurrent state
            # in FP16 lets the existing raw Triton kernel avoid the FP32 state
            # traffic.  The route is model- and batch-local because smaller
            # checkpoints and adjacent Ada products have separate evidence.
            native_graph_triton_fp16_state=is_4080,
            native_graph_triton_fp16_state_model_shapes=(
                ((4096, 32, 8),) if is_4080 else ()
            ),
            ada_sparse_ffn=is_4090,
            ada_sparse_ffn_max_rows=2 if is_4090 else 19,
            ada_sparse_ffn_inplace=is_4090,
            rkv_policy="vkwr_auto" if is_4090 else "manual",
            output_project_block_m=16,
            notes=(
                "RTX 4080: shape-safe 0.4B/1.5B dynamic B<=8, T<=4096, B*T<=32768 "
                "prefill uses fused scan/shift/state/output with FP32 state and default accumulation; "
                "exact fp16 rows promote B=1/2/4/8 and exact "
                "2.9B rows promote B=1/8 at T=128/512/2048; 1.5B/B1/P512 and P2048 use "
                "exact-card self-chunk routes, with stacked R/K/V at P2048; grouped W/A/G/V remains enabled for "
                "rows<=4, with a tensor-core grouped BMM on measured B8 model shapes; "
                "parity-approved prefill shapes use scoped full-GEMM FP16 accumulation; "
                "7.2B/B8 decode uses exact-shape Triton FP16 state, while the regressing Ada linear route stays disabled"
                if is_4080
                else "RTX 40/Ada: exact-4090 rows promote fixed-shape prefill graph plus raw recurrent decode, 8-warp norm/mix, rows=1/2/4 exact linear, exact 1.5B/B1/P2048 self-chunk plus stacked-copy-free R/K/V, graph-safe one/two-row sparse FFN, threshold-zero BnB W8 native prefill/decode, bsz8 grouped tensor-core W/A/V projection and MM4 output-head dispatch, plus block-scoped FP16 accumulation on measured 0.4B/1.5B/2.9B/7.2B B1/B8 prompt shapes; other Ada cards retain the compatible fallback until measured"
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
        is_5090 = is_rtx_model_name(profile.name, "5090")
        is_5070_laptop = is_rtx_laptop_model_name(profile.name, "5070")
        rtx5070_laptop_prefill_shapes = (
            (1024, 24, 1, 128),
            (1024, 24, 1, 512),
            (1024, 24, 2, 128),
            (1024, 24, 2, 512),
            (1024, 24, 4, 128),
            (1024, 24, 4, 512),
            (1024, 24, 8, 128),
            (1024, 24, 8, 512),
            (2048, 24, 1, 128),
            (2048, 24, 1, 512),
            (2048, 24, 2, 128),
            (2048, 24, 2, 512),
            (2048, 24, 4, 128),
            (2048, 24, 4, 512),
            (2048, 24, 8, 128),
            (2048, 24, 8, 512),
        )
        production_prefill_graph_shapes = (
            # Exact 5090 graph rows remove enough launch overhead to keep every
            # latest-checkpoint B1/B8 prefill-PD cell above its Qwen3.5 pair.
            (1024, 24, 1, 128),
            (1024, 24, 1, 512),
            (1024, 24, 1, 2048),
            (1024, 24, 8, 128),
            (1024, 24, 8, 512),
            (1024, 24, 8, 2048),
            (2048, 24, 1, 128),
            (2048, 24, 1, 512),
            (2048, 24, 1, 2048),
            (2048, 24, 8, 512),
            (2048, 24, 8, 2048),
            # g1h 1.5B B8/P128: the graph removes Python/custom-op launch
            # overhead from the Marlin W4 FFN route.  An exclusive 5090
            # paired run measured W4 at 1.0633x dense BF16 prefill while
            # preserving the full greedy stream; eager W4 was only 0.9287x.
            (2048, 24, 8, 128),
            (2560, 32, 1, 128),
            (2560, 32, 1, 512),
            (2560, 32, 1, 2048),
            (2560, 32, 8, 128),
            (2560, 32, 8, 512),
            (2560, 32, 8, 2048),
            (4096, 32, 1, 128),
            (4096, 32, 1, 512),
            (4096, 32, 1, 2048),
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
            fused_recurrent_raw=is_5070_laptop,
            native_graph_fused_recurrent_raw_shapes=(
                (1024, 1),
                (1024, 2),
                (1024, 4),
                (1024, 8),
                (2048, 1),
                (2048, 2),
                (2048, 4),
                (2048, 8),
            ) if is_5070_laptop else (),
            fused_output=True,
            fused_prefill_scan=is_5090 or is_5070_laptop,
            prefill_scan_model_shapes=(
                rtx5070_laptop_prefill_shapes if is_5070_laptop else ()
            ),
            prefill_graph=is_5090 or is_5070_laptop,
            prefill_graph_model_shapes=(
                production_prefill_graph_shapes
                if is_5090
                else rtx5070_laptop_prefill_shapes
                if is_5070_laptop
                else ()
            ),
            prefill_fp16_recurrent=is_5090,
            # Exact RTX 5090 B8 sweeps on g1h 1.5B/P512 and 7.2B/P128. The
            # fused prefill route remains opt-in globally; these shape gates
            # only select combinations measured end to end on this card.
            prefill_scan_block_m_model_shapes=((2048, 8, 512, 8),) if is_5090 else (),
            fused_prefill_shift_mix=is_5090,
            prefill_shift_mix_model_shapes=(
                (1024, 24, 8, 512),
                (1024, 24, 8, 2048),
                (2048, 24, 8, 128),
                (2048, 24, 8, 512),
                (2048, 24, 8, 2048),
                (4096, 32, 1, 128),
                (4096, 32, 1, 512),
                (4096, 32, 8, 128),
                (4096, 32, 8, 512),
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
                (1024, 24, 8, 512),
                (1024, 24, 8, 2048),
                (2048, 24, 8, 512),
                (2048, 24, 8, 2048),
                (4096, 32, 1, 128),
                (4096, 32, 1, 512),
                (4096, 32, 8, 128),
                (4096, 32, 8, 512),
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
                (1024, 24, 8, 512),
                (1024, 24, 8, 2048),
                (2048, 24, 8, 512),
                (2048, 24, 8, 2048),
                (4096, 32, 1, 128),
                (4096, 32, 1, 512),
                (4096, 32, 8, 128),
                (4096, 32, 8, 512),
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
                (1024, 24, 8, 512),
                (1024, 24, 8, 2048),
                (2048, 24, 8, 512),
                (2048, 24, 8, 2048),
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
            # Scoped full-prefill accumulation is independently correctness-
            # gated and is restored after each call. Never inherit by family.
            prefill_global_fp16_accum_model_shapes=(
                (1024, 24, 8, 512),
                (1024, 24, 8, 2048),
                (2048, 24, 8, 512),
                (2048, 24, 8, 2048),
                (2560, 32, 8, 128),
                (2560, 32, 8, 512),
                (2560, 32, 8, 2048),
                (4096, 32, 1, 128),
                (4096, 32, 1, 512),
                (4096, 32, 1, 2048),
                (4096, 32, 8, 128),
                (4096, 32, 8, 512),
                (4096, 32, 8, 2048),
            ) if is_5090 else (),
            fused_norm_mix=is_5090 or is_5070_laptop,
            native_graph_fused_norm_mix_shapes=(
                (1024, 1),
                (1024, 2),
                (1024, 4),
                (1024, 8),
                (2048, 1),
                (2048, 2),
                (2048, 8),
            ) if is_5070_laptop else (),
            norm_mix_num_warps=8 if is_5090 else 4,
            native_graph_triton_fp16_state=is_5070_laptop,
            native_graph_triton_fp16_state_model_shapes=(
                (1024, 24, 8),
                (2048, 24, 8),
            ) if is_5070_laptop else (),
            native_graph_state_dtype="fp16" if is_5090 else "fp32",
            native_graph_fp16_recurrent=is_5090,
            native_graph_precompute_embedding=is_5090,
            ada_linear=is_5090,
            ada_linear_rows="1" if is_5090 else "2 4",
            ada_linear_roles="hidden,ffn_up,ffn_down" if is_5090 else "auto",
            ada_wagv_lora=is_5090,
            sm120_compiled_ffn=False,
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
            notes=(
                "RTX 5070 Laptop: exact fp16 0.4B/1.5B B1/B2/B4/B8 P128/P512 rows promote only "
                "the allowlisted fused-scan prefill graph plus B8-only Triton FP16 decode "
                "state for hidden 1024/2048; B1/B2/B4 keep FP32 recurrent state and all "
                "other Blackwell fusions remain conservative"
                if is_5070_laptop
                else "RTX 50/Blackwell: exact RTX 5090 rows promote the official-FP16-state native graph decode profile and allowlisted 0.4B/1.5B/2.9B/7.2B/13.3B B1/B8 prefill shapes. Latest-checkpoint dense FP16 rows use scoped full-prefill accumulation only on the exact shapes that pass the 24/24 parameter-adjusted Qwen3.5 gate; 7.2B keeps B8 graph disabled because eager is faster. The 1.5B B8/P128 graph is shared by dense and Marlin W4. The 13.3B B8/P2048 row intentionally stays outside the graph allowlist because graph-private pools exceed 32 GiB. Other Blackwell cards retain the compatible fallback; use triton_compat for early sm_120 stacks and keep unvalidated projection/LoRA fusions off"
            ),
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
