try:
    from .configuration_rwkv7 import RWKV7Config
except ImportError:  # Allows importing experimental native_model without FLA.
    RWKV7Config = None

try:
    from .tokenization_rwkv7 import RWKV7Tokenizer
except ImportError:
    RWKV7Tokenizer = None

try:
    from .modeling_rwkv7 import RWKV7ForCausalLM, RWKV7Model
except ImportError:  # Allows importing experimental native_model without FLA.
    RWKV7ForCausalLM = None
    RWKV7Model = None

try:
    from .native_model import NativeRWKV7Config, NativeRWKV7ForCausalLM, NativeRWKV7Model
except Exception:  # Keep lightweight cache/unit tests importable with stubs.
    NativeRWKV7Config = None
    NativeRWKV7ForCausalLM = None
    NativeRWKV7Model = None

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
    "RWKV7Tokenizer",
    "NativeRWKV7Config",
    "NativeRWKV7ForCausalLM",
    "NativeRWKV7Model",
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
