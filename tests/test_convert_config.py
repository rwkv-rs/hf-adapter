#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
from pathlib import Path


class ShapeTensor:
    def __init__(self, *shape: int):
        self.shape = tuple(shape)


class DummyConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class DummyModel:
    def __init__(self, config):
        self.config = config
        self.dtype = None

    def to(self, dtype=None):
        self.dtype = dtype
        return self


class MissingFlaModel:
    def __init__(self, config):
        raise ImportError("flash-linear-attention (`fla`) is required")


def install_stubs() -> None:
    torch_mod = types.ModuleType("torch")
    torch_mod.Tensor = ShapeTensor
    torch_mod.float16 = "float16"
    torch_mod.bfloat16 = "bfloat16"
    torch_mod.float32 = "float32"
    torch_mod.load = lambda *args, **kwargs: {}
    sys.modules["torch"] = torch_mod

    hf_mod = types.ModuleType("rwkv7_hf")
    hf_mod.__path__ = []
    hf_mod.RWKV7Config = DummyConfig
    hf_mod.RWKV7ForCausalLM = DummyModel
    sys.modules["rwkv7_hf"] = hf_mod

    native_mod = types.ModuleType("rwkv7_hf.native_model")
    native_mod.NativeRWKV7Config = DummyConfig
    native_mod.NativeRWKV7ForCausalLM = DummyModel
    sys.modules["rwkv7_hf.native_model"] = native_mod

    manifest_mod = types.ModuleType("adapter_manifest")
    manifest_mod.ADAPTER_FILES = ()
    manifest_mod.LEGACY_REMOTE_CODE_FILES = ()
    sys.modules["adapter_manifest"] = manifest_mod
    sys.modules["scripts.adapter_manifest"] = manifest_mod


def import_converter():
    install_stubs()
    path = Path(__file__).resolve().parents[1] / "scripts" / "convert_rwkv7_to_hf.py"
    spec = importlib.util.spec_from_file_location("convert_rwkv7_to_hf_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def make_weights(
    hidden: int = 1024,
    attention_hidden: int | None = None,
    layers: int = 3,
    head_dim: int = 128,
):
    attention_hidden = hidden if attention_hidden is None else attention_hidden
    num_heads = attention_hidden // head_dim
    weights = {
        "emb.weight": ShapeTensor(65536, hidden),
        "blocks.0.att.w1": ShapeTensor(hidden, 64),
        "blocks.0.att.g1": ShapeTensor(hidden, 96),
        "blocks.0.att.a1": ShapeTensor(hidden, 32),
    }
    if layers > 1:
        weights["blocks.1.att.v1"] = ShapeTensor(hidden, 16)
    for i in range(layers):
        weights[f"blocks.{i}.ffn.key.weight"] = ShapeTensor(hidden * 4, hidden)
        weights[f"blocks.{i}.att.r_k"] = ShapeTensor(num_heads, head_dim)
        for projection in ("receptance", "key", "value"):
            weights[f"blocks.{i}.att.{projection}.weight"] = ShapeTensor(attention_hidden, hidden)
        weights[f"blocks.{i}.att.output.weight"] = ShapeTensor(hidden, attention_hidden)
        weights[f"blocks.{i}.att.ln_x.weight"] = ShapeTensor(attention_hidden)
        weights[f"blocks.{i}.att.ln_x.bias"] = ShapeTensor(attention_hidden)
    return weights


def main() -> int:
    conv = import_converter()
    weights = make_weights(attention_hidden=896)
    cfg = conv.infer_config(weights, dtype_name="float16", attn_mode="fused_recurrent", fuse_norm=False)
    assert cfg.vocab_size == 65536
    assert cfg.hidden_size == 1024
    assert cfg.intermediate_size == 4096
    assert cfg.num_hidden_layers == 3
    assert cfg.head_dim == 128
    assert cfg.num_heads == 7
    assert cfg.attention_hidden_size == 896
    assert cfg.value_dim == [896, 896, 896]
    assert cfg.decay_low_rank_dim == 64
    assert cfg.gate_low_rank_dim == 96
    assert cfg.a_low_rank_dim == 32
    assert cfg.v_low_rank_dim == 16
    assert cfg.torch_dtype == "float16"
    assert cfg.fuse_norm is False

    template = conv.build_template_model(cfg, "float16")
    assert isinstance(template, DummyModel)
    assert template.dtype == "float16"

    assert conv.NativeRWKV7ForCausalLM is DummyModel

    with tempfile.TemporaryDirectory() as td:
        config_path = Path(td) / "config.json"
        config_path.write_text("{}\n", encoding="utf-8")
        conv.patch_hf_metadata(Path(td))
        metadata = json.loads(config_path.read_text(encoding="utf-8"))
        assert metadata["architectures"] == ["NativeRWKV7ForCausalLM"]
        assert metadata["model_type"] == "rwkv7_native"
        assert metadata["auto_map"] == {
            "AutoConfig": "native_model.NativeRWKV7Config",
            "AutoModel": "native_model.NativeRWKV7Model",
            "AutoModelForCausalLM": "native_model.NativeRWKV7ForCausalLM",
        }

    assert conv.infer_num_layers(weights) == 3
    assert conv.infer_attention_shape(weights) == (7, 128, 896)
    assert conv.infer_head_dim(weights, 1024) == 128
    assert conv.infer_value_dim(weights, 3, 896, 7) == [896, 896, 896]

    singleton_rk = make_weights(attention_hidden=896)
    for i in range(3):
        singleton_rk[f"blocks.{i}.att.r_k"] = ShapeTensor(1, 1, 7, 128)
    singleton_cfg = conv.infer_config(
        singleton_rk,
        dtype_name="float16",
        attn_mode="fused_recurrent",
        fuse_norm=False,
    )
    assert singleton_cfg.attention_hidden_size == 896

    missing_middle = make_weights(layers=3)
    del missing_middle["blocks.1.ffn.key.weight"]
    try:
        conv.infer_num_layers(missing_middle)
    except ValueError:
        pass
    else:
        raise AssertionError("non-contiguous layer indices should raise ValueError")

    bad_heads = make_weights()
    bad_heads["blocks.0.att.r_k"] = ShapeTensor(1024)
    try:
        conv.infer_head_dim(bad_heads, 1024)
    except ValueError:
        pass
    else:
        raise AssertionError("rank-one r_k should raise ValueError")

    bad_value = make_weights()
    bad_value["blocks.1.att.value.weight"] = ShapeTensor(1030, 1024)
    try:
        conv.infer_value_dim(bad_value, 3, 1024, 8)
    except ValueError:
        pass
    else:
        raise AssertionError("value_dim not divisible by num_heads should raise ValueError")

    bad_output = make_weights(attention_hidden=896)
    bad_output["blocks.2.att.output.weight"] = ShapeTensor(896, 1024)
    try:
        conv.infer_config(
            bad_output,
            dtype_name="float16",
            attn_mode="fused_recurrent",
            fuse_norm=False,
        )
    except ValueError as exc:
        assert "blocks.2.att.output.weight" in str(exc)
    else:
        raise AssertionError("transposed output projection should be rejected")

    dst, transposed = conv.translate_name("blocks.2.att.receptance.weight", 3)
    assert dst == "model.layers.2.attn.r_proj.weight" and not transposed
    dst, transposed = conv.translate_name("blocks.2.att.w1", 3)
    assert dst == "model.layers.2.attn.w_lora.lora.0.weight" and transposed
    dst, transposed = conv.translate_name("blocks.2.att.w0", 3)
    assert dst == "model.layers.2.attn.w_lora.lora.2.bias" and not transposed
    dst, transposed = conv.translate_name("blocks.0.att.v1", 3)
    assert dst == "" and not transposed

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
