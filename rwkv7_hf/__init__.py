"""Public package surface for the canonical native RWKV-7 HF adapter."""

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
    from .tokenization_rwkv7 import RWKV7Tokenizer
except ImportError:
    RWKV7Tokenizer = None

from .ascend_runtime import (
    AscendRuntimeInfo,
    ascend_available,
    configure_ascend_defaults,
    enable_ascend,
)
from .ascend_quant_w4 import AscendW4A16Linear, quantize_ascend_w4a16_candidate
from .ascend_w4_cle import (
    W4CLECandidate,
    apply_sqrelu_channel_equalization,
    calibrate_sqrelu_value_w4,
)
from .ascend_quant import (
    AscendQuantDecision,
    AscendW8A16Linear,
    ascend_w8a16_decision,
    quantize_ascend_w8a16,
)

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
    "AscendRuntimeInfo",
    "ascend_available",
    "configure_ascend_defaults",
    "enable_ascend",
    "AscendQuantDecision",
    "AscendW8A16Linear",
    "ascend_w8a16_decision",
    "quantize_ascend_w8a16",
    "AscendW4A16Linear",
    "quantize_ascend_w4a16_candidate",
    "W4CLECandidate",
    "apply_sqrelu_channel_equalization",
    "calibrate_sqrelu_value_w4",
    "NativeRWKV7Config",
    "NativeRWKV7ForCausalLM",
    "NativeRWKV7Model",
    "NativeRWKV7Cache",
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
