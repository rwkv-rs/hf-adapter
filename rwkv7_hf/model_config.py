"""Compatibility import for rwkv7-hf 0.8 callers."""
from .configuration_rwkv7 import NativeRWKV7Config, RWKV7Config

__all__ = ["RWKV7Config", "NativeRWKV7Config"]
