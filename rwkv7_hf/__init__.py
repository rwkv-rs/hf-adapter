"""Readable, pure-PyTorch RWKV-7 integration for Hugging Face Transformers."""

from importlib.metadata import PackageNotFoundError, version

from .cache_rwkv7 import RWKV7Cache
from .configuration_rwkv7 import RWKV7Config
from .modeling_rwkv7 import (
    RWKV7Block,
    RWKV7ChannelMix,
    RWKV7ForCausalLM,
    RWKV7Model,
    RWKV7PreTrainedModel,
    RWKV7TimeMix,
)
from .tokenization_rwkv7 import RWKV7Tokenizer

try:
    __version__ = version("rwkv7-hf")
except PackageNotFoundError:
    __version__ = "1.0.0"


__all__ = [
    "__version__",
    "RWKV7Config",
    "RWKV7Cache",
    "RWKV7TimeMix",
    "RWKV7ChannelMix",
    "RWKV7Block",
    "RWKV7PreTrainedModel",
    "RWKV7Model",
    "RWKV7ForCausalLM",
    "RWKV7Tokenizer",
]
