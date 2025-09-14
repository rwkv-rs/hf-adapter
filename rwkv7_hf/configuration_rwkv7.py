# coding=utf-8
"""Remote-code wrapper around FLA RWKV7Config.

The normal optimized wrapper path uses FLA's config class.  The opt-in
``RWKV7_NATIVE_MODEL=1`` path must also be importable on machines where FLA is
not installed, so keep a minimal Transformers config fallback here instead of
failing at module import time.
"""

try:
    from fla.models.rwkv7.configuration_rwkv7 import RWKV7Config as _RWKV7Config
except Exception:  # pragma: no cover - exercised by fla-free native backend tests
    from transformers import PretrainedConfig

    class _RWKV7Config(PretrainedConfig):
        model_type = "rwkv7"

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.hidden_size = kwargs.get("hidden_size", 768)
            self.num_hidden_layers = kwargs.get("num_hidden_layers", 12)
            self.head_dim = kwargs.get("head_dim", 64)
            self.num_heads = kwargs.get("num_heads", None) or self.hidden_size // self.head_dim
            self.intermediate_size = kwargs.get("intermediate_size", self.hidden_size * 4)
            self.decay_low_rank_dim = kwargs.get("decay_low_rank_dim", 64)
            self.gate_low_rank_dim = kwargs.get("gate_low_rank_dim", 128)
            self.a_low_rank_dim = kwargs.get("a_low_rank_dim", 64)
            self.v_low_rank_dim = kwargs.get("v_low_rank_dim", 32)
            self.use_cache = kwargs.get("use_cache", True)


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
