from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from rwkv7_hf.native_model import NativeRWKV7Config, NativeRWKV7ForCausalLM
from scripts.convert_rwkv7_to_hf import convert


def _official_checkpoint_from_native(model: NativeRWKV7ForCausalLM) -> dict[str, torch.Tensor]:
    official: dict[str, torch.Tensor] = {}
    for name, value in model.state_dict().items():
        source = value.detach().cpu().clone()
        if name == "model.embeddings.weight":
            official["emb.weight"] = source
            continue
        if name.startswith("model.norm."):
            official[name.replace("model.norm.", "ln_out.")] = source
            continue
        if name == "lm_head.weight":
            official["head.weight"] = source
            continue
        if not name.startswith("model.layers."):
            raise AssertionError(f"Unhandled native checkpoint key: {name}")

        parts = name.split(".")
        layer = int(parts[2])
        suffix = ".".join(parts[3:])
        prefix = f"blocks.{layer}."
        direct = {
            "attn.x_r": "att.x_r",
            "attn.x_w": "att.x_w",
            "attn.x_k": "att.x_k",
            "attn.x_v": "att.x_v",
            "attn.x_a": "att.x_a",
            "attn.x_g": "att.x_g",
            "attn.k_k": "att.k_k",
            "attn.k_a": "att.k_a",
            "attn.r_k": "att.r_k",
            "attn.r_proj.weight": "att.receptance.weight",
            "attn.k_proj.weight": "att.key.weight",
            "attn.v_proj.weight": "att.value.weight",
            "attn.o_proj.weight": "att.output.weight",
            "attn.g_norm.weight": "att.ln_x.weight",
            "attn.g_norm.bias": "att.ln_x.bias",
            "ffn.x_k": "ffn.x_k",
            "ffn.key.weight": "ffn.key.weight",
            "ffn.value.weight": "ffn.value.weight",
            "attn_norm.weight": "ln1.weight",
            "attn_norm.bias": "ln1.bias",
            "ffn_norm.weight": "ln2.weight",
            "ffn_norm.bias": "ln2.bias",
            "pre_norm.weight": "ln0.weight",
            "pre_norm.bias": "ln0.bias",
        }
        if suffix in direct:
            official[prefix + direct[suffix]] = source
            continue

        lora_match = suffix.split(".")
        if len(lora_match) == 5 and lora_match[0] == "attn" and lora_match[1].endswith("_lora"):
            kind = lora_match[1][0]
            layer_name = lora_match[3]
            parameter = lora_match[4]
            if layer_name == "0" and parameter == "weight":
                official[f"{prefix}att.{kind}1"] = source.t().contiguous()
                continue
            if layer_name == "2" and parameter == "weight":
                official[f"{prefix}att.{kind}2"] = source.t().contiguous()
                continue
            if layer_name == "2" and parameter == "bias":
                official[f"{prefix}att.{kind}0"] = source
                continue
        raise AssertionError(f"Unhandled native checkpoint key: {name}")
    return official


def test_real_width_split_checkpoint_conversion_and_hf_reload() -> None:
    torch.manual_seed(17)
    config = NativeRWKV7Config(
        vocab_size=31,
        hidden_size=8,
        attention_hidden_size=16,
        num_heads=4,
        head_dim=4,
        num_hidden_layers=2,
        intermediate_size=24,
        decay_low_rank_dim=3,
        a_low_rank_dim=2,
        gate_low_rank_dim=4,
        v_low_rank_dim=5,
        fuse_norm=False,
    )
    source_model = NativeRWKV7ForCausalLM(config).eval()
    source_model.model.layers[0].attn.r_k.data = source_model.model.layers[0].attn.r_k.data.reshape(4, 4)
    official = _official_checkpoint_from_native(source_model)
    for layer in range(config.num_hidden_layers):
        official[f"blocks.{layer}.att.r_k"] = official[f"blocks.{layer}.att.r_k"].reshape(1, 1, 4, 4)

    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    with torch.no_grad():
        expected = source_model(input_ids=input_ids, use_cache=True).logits

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        checkpoint = root / "width-split.pth"
        output = root / "hf"
        torch.save(official, checkpoint)
        convert(
            argparse.Namespace(
                input=str(checkpoint),
                output=str(output),
                vocab_file=None,
                precision="fp32",
                attn_mode="fused_recurrent",
                fuse_norm=False,
                max_shard_size="1GB",
                low_memory=False,
            )
        )

        auto_config = AutoConfig.from_pretrained(output, trust_remote_code=True)
        assert auto_config.hidden_size == 8
        assert auto_config.attention_hidden_size == 16
        loaded = AutoModelForCausalLM.from_pretrained(output, trust_remote_code=True).eval()
        with torch.no_grad():
            actual = loaded(input_ids=input_ids, use_cache=True).logits
            generated = loaded.generate(input_ids=input_ids, max_new_tokens=2, use_cache=True)
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
        assert generated.shape == (1, 5)

        reloaded_dir = root / "reloaded"
        loaded.save_pretrained(reloaded_dir, safe_serialization=True)
        reloaded = AutoModelForCausalLM.from_pretrained(reloaded_dir, trust_remote_code=True).eval()
        with torch.no_grad():
            reloaded_logits = reloaded(input_ids=input_ids, use_cache=True).logits
        assert torch.allclose(reloaded_logits, actual, atol=1e-6, rtol=1e-6)
