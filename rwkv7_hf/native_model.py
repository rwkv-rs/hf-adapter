"""Compatibility aliases for the retired 0.8 NativeRWKV7 class names."""
from .cache_rwkv7 import NativeRWKV7Cache, RWKV7Cache
from .configuration_rwkv7 import NativeRWKV7Config, RWKV7Config
from .modeling_rwkv7 import (
    NativeRWKV7ForCausalLM,
    NativeRWKV7Model,
    RWKV7ForCausalLM,
    RWKV7Model,
)

__all__ = [
    "RWKV7Config",
    "RWKV7Cache",
    "RWKV7Model",
    "RWKV7ForCausalLM",
    "NativeRWKV7Config",
    "NativeRWKV7Cache",
    "NativeRWKV7Model",
    "NativeRWKV7ForCausalLM",
]
