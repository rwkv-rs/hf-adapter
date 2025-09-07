# coding=utf-8
"""Remote-code wrapper around FLA RWKV7Config."""
from fla.models.rwkv7.configuration_rwkv7 import RWKV7Config as _RWKV7Config


class RWKV7Config(_RWKV7Config):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
