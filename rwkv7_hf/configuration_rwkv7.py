# coding=utf-8
"""Remote-code wrapper around FLA RWKV7Config."""
from fla.models.rwkv7.configuration_rwkv7 import RWKV7Config as _RWKV7Config


class RWKV7HFAdapterConfig(_RWKV7Config):
    """RWKV-7 adapter config with a unique AutoClass identity.

    FLA registers a local `RWKV7Config` / `rwkv7` AutoModel mapping. If this
    remote-code config has the same class name/model_type, Transformers treats
    the FLA model as explicit local code and bypasses this repository's remote
    wrapper. A unique class name and model_type force `trust_remote_code=True` to
    resolve `AutoModelForCausalLM` to `modeling_rwkv7.RWKV7ForCausalLM`.
    """

    model_type = "rwkv7_hf_adapter"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# Keep the public remote-code symbol stable for config.json auto_map.
RWKV7Config = RWKV7HFAdapterConfig
