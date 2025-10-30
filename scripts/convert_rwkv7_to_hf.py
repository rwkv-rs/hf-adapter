#!/usr/bin/env python3
# coding=utf-8
"""Convert official RWKV-7 .pth checkpoints to a Hugging Face model directory.

This first-stage adapter uses the FLA RWKV7 PreTrainedModel implementation but emits a
normal HF-style directory with config.json, generation_config.json, model.safetensors,
remote-code wrapper files, tokenizer_config.json, and the RWKV trie vocab.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Tuple

import torch

# Importing rwkv7_hf imports FLA-backed PreTrainedModel classes.
from rwkv7_hf import RWKV7Config, RWKV7ForCausalLM


DTYPES = {
    "fp16": ("float16", torch.float16),
    "float16": ("float16", torch.float16),
    "bf16": ("bfloat16", torch.bfloat16),
    "bfloat16": ("bfloat16", torch.bfloat16),
    "fp32": ("float32", torch.float32),
    "float32": ("float32", torch.float32),
}


def tensor_shape(weights: Dict[str, torch.Tensor], name: str) -> tuple[int, ...]:
    """Return a plain int shape with a useful error for missing checkpoint keys."""
    if name not in weights:
        raise KeyError(f"Missing required RWKV-7 weight: {name}")
    return tuple(int(v) for v in weights[name].shape)


def infer_num_layers(weights: Dict[str, torch.Tensor]) -> int:
    """Infer and validate contiguous RWKV block indices from checkpoint keys."""
    layers = sorted(
        int(m.group(1))
        for name in weights
        if (m := re.match(r"blocks\.(\d+)\.ffn\.key\.weight$", name))
    )
    if not layers:
        raise KeyError("No blocks.*.ffn.key.weight tensors found in checkpoint")
    expected = list(range(layers[-1] + 1))
    if layers != expected:
        raise ValueError(f"RWKV block indices must be contiguous from 0: got {layers[:20]} ...")
    return len(layers)


def infer_head_dim(weights: Dict[str, torch.Tensor], hidden_size: int) -> int:
    """Infer attention head dimension instead of hard-coding 64 for every model."""
    rk_shape = tensor_shape(weights, "blocks.0.att.r_k")
    if len(rk_shape) >= 2:
        num_heads, head_dim = int(rk_shape[-2]), int(rk_shape[-1])
        if num_heads * head_dim != hidden_size:
            raise ValueError(
                "blocks.0.att.r_k shape does not match hidden size: "
                f"{rk_shape} -> {num_heads}*{head_dim} != {hidden_size}"
            )
        return head_dim
    if hidden_size % 64 != 0:
        raise ValueError(f"Cannot infer head_dim from r_k={rk_shape}; hidden_size={hidden_size} is not divisible by 64")
    return 64


def infer_value_dim(weights: Dict[str, torch.Tensor], num_layers: int, hidden_size: int, num_heads: int) -> list[int]:
    """Infer per-layer value dimensions from official value projection weights."""
    dims: list[int] = []
    for layer_idx in range(num_layers):
        value_shape = tensor_shape(weights, f"blocks.{layer_idx}.att.value.weight")
        value_dim = int(value_shape[0])
        if value_dim % num_heads != 0:
            raise ValueError(
                f"blocks.{layer_idx}.att.value.weight output dim {value_dim} is not divisible by num_heads={num_heads}"
            )
        dims.append(value_dim)
    if any(v <= 0 for v in dims):
        raise ValueError(f"Invalid value_dim list: {dims}")
    if dims[0] != hidden_size:
        raise ValueError(f"Layer-0 value_dim should equal hidden_size for RWKV-7: {dims[0]} != {hidden_size}")
    return dims


def validate_layer_shapes(weights: Dict[str, torch.Tensor], num_layers: int, hidden_size: int, head_dim: int) -> None:
    """Catch size/shape mismatches before constructing the HF model."""
    num_heads = hidden_size // head_dim
    for layer_idx in range(num_layers):
        ffn_key = tensor_shape(weights, f"blocks.{layer_idx}.ffn.key.weight")
        if len(ffn_key) != 2 or int(ffn_key[1]) != hidden_size:
            raise ValueError(f"blocks.{layer_idx}.ffn.key.weight has inconsistent shape {ffn_key}")
        rk_shape = tensor_shape(weights, f"blocks.{layer_idx}.att.r_k")
        if tuple(rk_shape[-2:]) != (num_heads, head_dim):
            raise ValueError(
                f"blocks.{layer_idx}.att.r_k has inconsistent shape {rk_shape}; "
                f"expected trailing {(num_heads, head_dim)}"
            )


def infer_config(weights: Dict[str, torch.Tensor], dtype_name: str, attn_mode: str, fuse_norm: bool) -> RWKV7Config:
    hidden_size = tensor_shape(weights, "blocks.0.ffn.key.weight")[1]
    intermediate_size = tensor_shape(weights, "blocks.0.ffn.key.weight")[0]
    num_layers = infer_num_layers(weights)
    head_dim = infer_head_dim(weights, hidden_size)
    if hidden_size % head_dim != 0:
        raise ValueError(f"hidden_size={hidden_size} must be divisible by head_dim={head_dim}")
    num_heads = hidden_size // head_dim
    value_dim = infer_value_dim(weights, num_layers, hidden_size, num_heads)
    validate_layer_shapes(weights, num_layers, hidden_size, head_dim)
    try:
        v_low_rank_dim = tensor_shape(weights, "blocks.1.att.v1")[1]
    except KeyError:
        v_low_rank_dim = 32
    cfg = RWKV7Config(
        attn_mode=attn_mode,
        vocab_size=tensor_shape(weights, "emb.weight")[0],
        hidden_size=hidden_size,
        hidden_ratio=intermediate_size / hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        value_dim=value_dim,
        decay_low_rank_dim=tensor_shape(weights, "blocks.0.att.w1")[1],
        gate_low_rank_dim=tensor_shape(weights, "blocks.0.att.g1")[1],
        a_low_rank_dim=tensor_shape(weights, "blocks.0.att.a1")[1],
        v_low_rank_dim=v_low_rank_dim,
        head_dim=head_dim,
        # 0 is unused by the official trie vocab; use it as a HF generation sentinel/pad id.
        pad_token_id=0,
        eos_token_id=0,
        bos_token_id=1,
        tie_word_embeddings=False,
        fuse_norm=fuse_norm,
    )
    cfg.torch_dtype = dtype_name
    return cfg


def build_template_model(config: RWKV7Config, dtype: torch.dtype):
    """Construct the HF-shaped model used as the state_dict template.

    Conversion only needs module names and tensor shapes.  Prefer the optimized
    FLA wrapper when it is installed, but allow offline/no-FLA conversion through
    the native backend, which intentionally uses the same converted key layout.
    """

    try:
        return RWKV7ForCausalLM(config).to(dtype=dtype)
    except ImportError as exc:
        message = str(exc).lower()
        if (
            "flash-linear-attention" not in message
            and "`fla`" not in message
            and "no module named 'fla'" not in message
        ):
            raise
        from rwkv7_hf.native_model import NativeRWKV7ForCausalLM

        print("FLA unavailable; using native RWKV-7 template for conversion.")
        return NativeRWKV7ForCausalLM(config).to(dtype=dtype)


def translate_name(name: str, num_layers: int) -> Tuple[str, bool]:
    unused_names = {"blocks.0.att.v0", "blocks.0.att.v1", "blocks.0.att.v2"}
    emb_head = {
        "emb.weight": "model.embeddings.weight",
        "ln_out.weight": "model.norm.weight",
        "ln_out.bias": "model.norm.bias",
        "head.weight": "lm_head.weight",
    }
    proj = {
        "receptance": "r_proj",
        "key": "k_proj",
        "value": "v_proj",
        "ln_x": "g_norm",
        "output": "o_proj",
    }
    if name in unused_names:
        return "", False
    if name in emb_head:
        return emb_head[name], False

    parts = name.split(".")
    if len(parts) < 4 or parts[0] != "blocks":
        raise KeyError(f"Unexpected RWKV weight name: {name}")
    layer_idx = int(parts[1])
    if layer_idx not in range(num_layers):
        raise KeyError(f"Layer index out of range in {name}")
    parts[0] = "model.layers"
    parts[2] = {"att": "attn", "ffn": "ffn", "ln0": "pre_norm", "ln1": "attn_norm", "ln2": "ffn_norm"}[parts[2]]
    transposed = False
    if re.match(r"[wvag][012]", parts[3]):
        typ, num = parts[3]
        parts[3] = f"{typ}_lora.lora." + {"0": "2.bias", "1": "0.weight", "2": "2.weight"}[num]
        transposed = num in {"1", "2"}
    elif parts[2] == "attn" and parts[3] in proj:
        parts[3] = proj[parts[3]]
    return ".".join(parts), transposed


def copy_adapter_files(output: Path, vocab_file: Path | None) -> None:
    root = Path(__file__).resolve().parents[1]
    for name in [
        "configuration_rwkv7.py",
        "dplr_prefill.py",
        "dplr_prefill_triton.py",
        "fused_attention_projection.py",
        "fused_ffn.py",
        "fused_lora.py",
        "fused_decode_norm_mix.py",
        "fused_norm_mix.py",
        "fused_output.py",
        "fused_prefill.py",
        "fused_projection.py",
        "fused_recurrent_update.py",
        "fused_time_mix.py",
        "kernel_policy.py",
        "mlx_bridge.py",
        "mlx_model.py",
        "mlx_mix.py",
        "mlx_quant.py",
        "mlx_wkv.py",
        "modeling_rwkv7.py",
        "native.py",
        "native_jit.py",
        "native_model.py",
        "native_quant.py",
        "native_quant_mm4.py",
        "native_quant_mm8.py",
        "native_quant_policy.py",
        "sm70_linear.py",
        "triton_compat.py",
        "tokenization_rwkv7.py",
    ]:
        shutil.copyfile(root / "rwkv7_hf" / name, output / name)
    if vocab_file is not None:
        shutil.copyfile(vocab_file, output / "rwkv_vocab_v20230424.txt")


def patch_hf_metadata(output: Path) -> None:
    cfg_path = output / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["architectures"] = ["RWKV7ForCausalLM"]
    cfg["model_type"] = "rwkv7_hf_adapter"
    cfg["auto_map"] = {
        "AutoConfig": "configuration_rwkv7.RWKV7Config",
        "AutoModel": "modeling_rwkv7.RWKV7Model",
        "AutoModelForCausalLM": "modeling_rwkv7.RWKV7ForCausalLM",
    }
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")

    tok_cfg = {
        "tokenizer_class": "RWKV7Tokenizer",
        "auto_map": {"AutoTokenizer": ["tokenization_rwkv7.RWKV7Tokenizer", None]},
        "model_vocab_size": int(cfg.get("vocab_size", 65536)),
        "pad_token": "<|padding|>",
        "eos_token": "<|endoftext|>",
        "errors": "replace",
    }
    (output / "tokenizer_config.json").write_text(json.dumps(tok_cfg, indent=2, ensure_ascii=False) + "\n")
    special = {"pad_token": "<|padding|>", "eos_token": "<|endoftext|>"}
    (output / "special_tokens_map.json").write_text(json.dumps(special, indent=2, ensure_ascii=False) + "\n")


def convert(args: argparse.Namespace) -> None:
    dtype_name, dtype = DTYPES[args.precision]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    weights = torch.load(args.input, weights_only=True, map_location="cpu")
    config = infer_config(weights, dtype_name=dtype_name, attn_mode=args.attn_mode, fuse_norm=args.fuse_norm)
    model = build_template_model(config, dtype)
    model_dict = model.state_dict()
    missing = set(model_dict)

    for src_name, src_weight in weights.items():
        dst_name, transposed = translate_name(src_name, config.num_hidden_layers)
        if not dst_name:
            continue
        if dst_name not in model_dict:
            raise KeyError(f"Translated name not in HF model: {src_name} -> {dst_name}")
        weight = src_weight.detach().clone()
        if transposed:
            weight = weight.t().contiguous()
        if list(weight.shape) == [1, 1, config.hidden_size]:
            weight = weight.squeeze()
        expected = model_dict[dst_name]
        if "attn.x_" in dst_name:
            ok = tuple(expected.shape[2:]) == tuple(weight.shape)
        else:
            ok = tuple(expected.shape) == tuple(weight.shape)
        if not ok:
            raise AssertionError(f"Shape mismatch {src_name} -> {dst_name}: {tuple(weight.shape)} vs {tuple(expected.shape)}")
        expected.copy_(weight.to(dtype=expected.dtype))
        missing.discard(dst_name)

    allowed_missing = {"model.layers.0.pre_norm.weight", "model.layers.0.pre_norm.bias"}
    unexpected_missing = sorted(missing - allowed_missing)
    if unexpected_missing:
        raise KeyError(f"Uninitialized HF parameters: {unexpected_missing[:20]} ... total={len(unexpected_missing)}")

    model.save_pretrained(output, max_shard_size=args.max_shard_size, safe_serialization=True)
    if args.vocab_file:
        vocab = Path(args.vocab_file)
    else:
        vocab = None
    copy_adapter_files(output, vocab)
    patch_hf_metadata(output)
    print(f"Saved HF RWKV-7 model to: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Official RWKV-7 .pth checkpoint")
    parser.add_argument("--output", required=True, help="Output HF model directory")
    parser.add_argument("--vocab-file", default=None, help="rwkv_vocab_v20230424.txt to copy into the model dir")
    parser.add_argument("--precision", choices=sorted(DTYPES), default="fp16")
    parser.add_argument("--attn-mode", choices=["chunk", "fused_recurrent"], default="chunk")
    norm_group = parser.add_mutually_exclusive_group()
    norm_group.add_argument("--fuse-norm", dest="fuse_norm", action="store_true", help="Use FLA fused norm modules in the generated config")
    norm_group.add_argument("--no-fuse-norm", dest="fuse_norm", action="store_false", help="Use native PyTorch norm modules; faster for V100 decode in current tests")
    parser.set_defaults(fuse_norm=False)
    parser.add_argument("--max-shard-size", default="1000GB")
    args = parser.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
