"""Public package surface for the canonical native RWKV-7 HF adapter."""

from .kernel_package import inspect_kernel_package, kernel_runtime_report

try:
    from .native_model import (
        NativeRWKV7Cache,
        NativeRWKV7Config,
        NativeRWKV7ForCausalLM,
        NativeRWKV7Model,
    )
except Exception:  # Keep lightweight tooling importable without torch/Transformers.
    NativeRWKV7Cache = None
    NativeRWKV7Config = None
    NativeRWKV7ForCausalLM = None
    NativeRWKV7Model = None

RWKV7Config = NativeRWKV7Config
RWKV7Model = NativeRWKV7Model
RWKV7ForCausalLM = NativeRWKV7ForCausalLM
RWKV7StateCache = NativeRWKV7Cache

try:
    from .biren_runtime import (
        BirenDTypeError,
        BirenRuntimeInfo,
        BirenRuntimePolicy,
        biren_available,
        biren_runtime_policy,
        configure_biren_defaults,
        enable_biren,
        memory_stats as biren_memory_stats,
        synchronize as biren_synchronize,
        validate_biren_model_dtype,
    )
except Exception:  # Keep lightweight tooling importable without torch_br.
    BirenDTypeError = None
    BirenRuntimeInfo = None
    BirenRuntimePolicy = None
    biren_available = None
    biren_runtime_policy = None
    configure_biren_defaults = None
    enable_biren = None
    biren_memory_stats = None
    biren_synchronize = None
    validate_biren_model_dtype = None

try:
    from .metax_runtime import (
        MetaXRuntimeInfo,
        configure_metax_defaults,
        enable_metax,
        memory_stats as metax_memory_stats,
        metax_available,
        synchronize as metax_synchronize,
    )
except Exception:  # Keep lightweight tooling importable without a MetaX stack.
    MetaXRuntimeInfo = None
    configure_metax_defaults = None
    enable_metax = None
    metax_available = None
    metax_memory_stats = None
    metax_synchronize = None

try:
    from .ascend_runtime import (
        AscendRuntimeInfo,
        ascend_available,
        configure_ascend_defaults,
        enable_ascend,
        memory_stats as ascend_memory_stats,
        synchronize as ascend_synchronize,
    )
except Exception:  # Keep package importable without the optional torch-npu stack.
    AscendRuntimeInfo = None
    ascend_available = None
    configure_ascend_defaults = None
    enable_ascend = None
    ascend_memory_stats = None
    ascend_synchronize = None

try:
    from .ascend_quant import (
        ASCEND_910B3_W8_SPEED_ROWS,
        AscendQuantDecision,
        AscendW8A16Linear,
        ascend_w8a16_decision,
        quantize_ascend_w8a16,
    )
    from .ascend_quant_w4 import (
        AscendW4A16Linear,
        AscendWeightOnlyLinear,
        quantize_ascend_w4a16_candidate,
    )
except Exception:  # Optional PyTorch-backed Ascend quantization surface.
    ASCEND_910B3_W8_SPEED_ROWS = None
    AscendQuantDecision = None
    AscendW8A16Linear = None
    AscendW4A16Linear = None
    AscendWeightOnlyLinear = None
    ascend_w8a16_decision = None
    quantize_ascend_w8a16 = None
    quantize_ascend_w4a16_candidate = None

try:
    from .tokenization_rwkv7 import RWKV7Tokenizer
except ImportError:
    RWKV7Tokenizer = None

def __getattr__(name):
    """Load the historical FLA wrapper only through explicit reference names."""

    if name == "FLAReferenceRWKV7Config":
        from .configuration_rwkv7 import RWKV7Config as reference_config

        return reference_config
    if name in {"FLAReferenceRWKV7Model", "FLAReferenceRWKV7ForCausalLM"}:
        from .modeling_rwkv7 import RWKV7ForCausalLM as reference_causal_lm
        from .modeling_rwkv7 import RWKV7Model as reference_model

        return reference_model if name == "FLAReferenceRWKV7Model" else reference_causal_lm
    raise AttributeError(name)

try:
    from .mlx_model import (
        MLXGenerateOutput,
        MLXGenerationSession,
        MLXGenerationSessionBatch,
        MLXRWKV7Model,
        MLXRWKV7State,
        MLXSessionStepOutput,
        generate_text_from_hf,
        load_mlx_generation_session,
    )
except Exception:  # Keep imports working when optional MLX/torch deps are absent.
    MLXGenerateOutput = None
    MLXGenerationSession = None
    MLXGenerationSessionBatch = None
    MLXRWKV7Model = None
    MLXRWKV7State = None
    MLXSessionStepOutput = None
    generate_text_from_hf = None
    load_mlx_generation_session = None

try:
    from .mlx_speculative import MLXSpeculativeResult, speculative_decode_greedy
except Exception:  # Optional MLX runtime.
    MLXSpeculativeResult = None
    speculative_decode_greedy = None

try:
    from .mlx_cache import MLXPrefixCacheHit, MLXPrefixStateCache, mlx_model_cache_fingerprint
    from .mlx_scheduler import (
        MLXBackpressureError,
        MLXDynamicBatchScheduler,
        MLXDynamicRequest,
        create_cached_mlx_generation_session,
    )
except Exception:  # Optional MLX serving runtime.
    MLXPrefixCacheHit = None
    MLXPrefixStateCache = None
    mlx_model_cache_fingerprint = None
    MLXBackpressureError = None
    MLXDynamicBatchScheduler = None
    MLXDynamicRequest = None
    create_cached_mlx_generation_session = None

__all__ = [
    "RWKV7Config",
    "RWKV7ForCausalLM",
    "RWKV7Model",
    "RWKV7StateCache",
    "RWKV7Tokenizer",
    "NativeRWKV7Config",
    "NativeRWKV7ForCausalLM",
    "NativeRWKV7Model",
    "NativeRWKV7Cache",
    "inspect_kernel_package",
    "kernel_runtime_report",
    "BirenDTypeError",
    "BirenRuntimeInfo",
    "BirenRuntimePolicy",
    "biren_available",
    "biren_runtime_policy",
    "configure_biren_defaults",
    "enable_biren",
    "biren_memory_stats",
    "biren_synchronize",
    "validate_biren_model_dtype",
    "MetaXRuntimeInfo",
    "configure_metax_defaults",
    "enable_metax",
    "metax_available",
    "metax_memory_stats",
    "metax_synchronize",
    "AscendRuntimeInfo",
    "ascend_available",
    "configure_ascend_defaults",
    "enable_ascend",
    "ascend_memory_stats",
    "ascend_synchronize",
    "ASCEND_910B3_W8_SPEED_ROWS",
    "AscendQuantDecision",
    "AscendW8A16Linear",
    "AscendW4A16Linear",
    "AscendWeightOnlyLinear",
    "ascend_w8a16_decision",
    "quantize_ascend_w8a16",
    "quantize_ascend_w4a16_candidate",
    "MLXGenerateOutput",
    "MLXGenerationSession",
    "MLXGenerationSessionBatch",
    "MLXRWKV7Model",
    "MLXRWKV7State",
    "MLXSessionStepOutput",
    "generate_text_from_hf",
    "load_mlx_generation_session",
    "MLXSpeculativeResult",
    "speculative_decode_greedy",
    "MLXPrefixCacheHit",
    "MLXPrefixStateCache",
    "mlx_model_cache_fingerprint",
    "MLXBackpressureError",
    "MLXDynamicBatchScheduler",
    "MLXDynamicRequest",
    "create_cached_mlx_generation_session",
]
