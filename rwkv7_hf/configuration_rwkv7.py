# coding=utf-8
"""Remote-code wrapper around FLA RWKV7Config.

The normal optimized wrapper path uses FLA's config class.  The opt-in
``RWKV7_NATIVE_MODEL=1`` path must also be importable on machines where FLA is
not installed, so keep a minimal Transformers config fallback here instead of
failing at module import time.
"""

try:
    from .triton_compat import apply_runtime_compat as _rwkv7_apply_runtime_compat
except ImportError:  # pragma: no cover - direct remote-file execution fallback
    try:
        from triton_compat import apply_runtime_compat as _rwkv7_apply_runtime_compat
    except Exception:  # pragma: no cover - compatibility helper is optional
        _rwkv7_apply_runtime_compat = None
if _rwkv7_apply_runtime_compat is not None:
    _rwkv7_apply_runtime_compat()


def _normalize_head_count_kwargs(kwargs: dict) -> dict:
    """Keep RWKV ``num_heads`` and HF ``num_attention_heads`` synchronized."""

    normalized = dict(kwargs)
    num_heads = normalized.get("num_heads")
    num_attention_heads = normalized.get("num_attention_heads")
    if (
        num_heads is not None
        and num_attention_heads is not None
        and int(num_heads) != int(num_attention_heads)
    ):
        raise ValueError(
            "num_heads and num_attention_heads must match when both are provided"
        )
    resolved = num_heads if num_heads is not None else num_attention_heads
    if resolved is not None:
        normalized["num_heads"] = int(resolved)
        normalized["num_attention_heads"] = int(resolved)
    return normalized


try:
    from fla.models.rwkv7.configuration_rwkv7 import RWKV7Config as _RWKV7Config
except Exception:  # pragma: no cover - exercised by fla-free native backend tests
    from transformers import PretrainedConfig

    class _RWKV7Config(PretrainedConfig):
        model_type = "rwkv7"

        def __init__(self, **kwargs):
            kwargs = _normalize_head_count_kwargs(kwargs)
            super().__init__(**kwargs)
            self.hidden_size = kwargs.get("hidden_size", 768)
            self.num_hidden_layers = kwargs.get("num_hidden_layers", 12)
            self.num_heads = kwargs.get("num_heads")
            requested_attention_width = int(
                kwargs.get("attention_hidden_size", self.hidden_size)
            )
            requested_head_dim = kwargs.get("head_dim", None)
            if self.num_heads is None and requested_head_dim is None:
                requested_head_dim = (
                    64 if requested_attention_width % 64 == 0 else requested_attention_width
                )
            if requested_head_dim is None:
                if requested_attention_width % int(self.num_heads):
                    raise ValueError("attention_hidden_size must be divisible by num_heads")
                requested_head_dim = requested_attention_width // int(self.num_heads)
            self.head_dim = int(requested_head_dim)
            if self.num_heads is None:
                if requested_attention_width % self.head_dim:
                    raise ValueError("attention_hidden_size must be divisible by head_dim")
                self.num_heads = requested_attention_width // self.head_dim
            self.attention_hidden_size = int(
                kwargs.get("attention_hidden_size", self.num_heads * self.head_dim)
            )
            if self.attention_hidden_size != int(self.num_heads) * int(self.head_dim):
                raise ValueError("attention_hidden_size must equal num_heads * head_dim")
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
        kwargs = _normalize_head_count_kwargs(kwargs)
        # Native W8/W4 persistence: when True, from_pretrained re-quantizes
        # eligible linears into MM8Linear / MM4Linear after loading the fp16
        # weights. The packed state is a deterministic function of the dense
        # weights, so this round-trips without serializing the uint8 buffers.
        self.use_native_mm8 = kwargs.pop("use_native_mm8", False)
        self.native_mm8_min_params = kwargs.pop("native_mm8_min_params", 8_000_000)
        self.native_mm8_policy = kwargs.pop("native_mm8_policy", "memory")
        self.use_native_mm4 = kwargs.pop("use_native_mm4", False)
        self.native_mm4_min_params = kwargs.pop("native_mm4_min_params", 8_000_000)
        self.native_mm4_policy = kwargs.pop("native_mm4_policy", "memory")
        self.native_mm4_group_size = kwargs.pop("native_mm4_group_size", 0)
        self.native_mm4_group_policy = kwargs.pop("native_mm4_group_policy", "all")
        # FP8 E4M3 quantization (requires Hopper/Ada/Blackwell for _scaled_mm)
        self.use_native_fp8 = kwargs.pop("use_native_fp8", False)
        self.native_fp8_min_params = kwargs.pop("native_fp8_min_params", 8_000_000)
        self.native_fp8_policy = kwargs.pop("native_fp8_policy", "memory")
        super().__init__(*args, **kwargs)
        requested_attention_width = kwargs.get("attention_hidden_size", None)
        num_heads = int(
            getattr(self, "num_heads", getattr(self, "num_attention_heads", 0))
        )
        requested_num_attention_heads = kwargs.get("num_attention_heads")
        if (
            requested_num_attention_heads is not None
            and int(requested_num_attention_heads) != num_heads
        ):
            raise ValueError(
                "num_heads and num_attention_heads must match when both are provided"
            )
        self.num_heads = num_heads
        self.num_attention_heads = num_heads
        head_dim = int(getattr(self, "head_dim", 0))
        if num_heads <= 0 or head_dim <= 0:
            raise ValueError("num_heads and head_dim must be positive")
        self.attention_hidden_size = int(
            requested_attention_width
            if requested_attention_width is not None
            else num_heads * head_dim
        )
        if self.attention_hidden_size != num_heads * head_dim:
            raise ValueError("attention_hidden_size must equal num_heads * head_dim")


# Keep the public remote-code symbol stable for config.json auto_map.
RWKV7Config = RWKV7HFAdapterConfig
