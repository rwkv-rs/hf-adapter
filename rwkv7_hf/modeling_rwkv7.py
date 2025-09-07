# coding=utf-8
"""Remote-code wrapper around FLA RWKV7 HF modules.

Requires flash-linear-attention (`fla`) on PYTHONPATH / installed in the env.
"""
from fla.models.rwkv7.modeling_rwkv7 import RWKV7Model as _RWKV7Model
from fla.models.rwkv7.modeling_rwkv7 import RWKV7ForCausalLM as _RWKV7ForCausalLM


class RWKV7Model(_RWKV7Model):
    pass


class RWKV7ForCausalLM(_RWKV7ForCausalLM):
    # Transformers >=5 expects dict-like _tied_weights_keys in save_pretrained.
    _tied_weights_keys = {}
