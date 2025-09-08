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


def infer_config(weights: Dict[str, torch.Tensor], dtype_name: str, attn_mode: str, fuse_norm: bool) -> RWKV7Config:
    hidden_size = weights["blocks.0.ffn.key.weight"].shape[1]
    intermediate_size = weights["blocks.0.ffn.key.weight"].shape[0]
    num_layers = 0
    while f"blocks.{num_layers}.ffn.key.weight" in weights:
        num_layers += 1
    try:
        v_low_rank_dim = weights["blocks.1.att.v1"].shape[1]
    except KeyError:
        v_low_rank_dim = 32
    cfg = RWKV7Config(
        attn_mode=attn_mode,
        vocab_size=weights["emb.weight"].shape[0],
        hidden_size=hidden_size,
        hidden_ratio=intermediate_size / hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        value_dim=[hidden_size] * num_layers,
        decay_low_rank_dim=weights["blocks.0.att.w1"].shape[1],
        gate_low_rank_dim=weights["blocks.0.att.g1"].shape[1],
        a_low_rank_dim=weights["blocks.0.att.a1"].shape[1],
        v_low_rank_dim=v_low_rank_dim,
        head_dim=64,
        # 0 is unused by the official trie vocab; use it as a HF generation sentinel/pad id.
        pad_token_id=0,
        eos_token_id=0,
        bos_token_id=1,
        tie_word_embeddings=False,
        fuse_norm=fuse_norm,
    )
    cfg.torch_dtype = dtype_name
    return cfg


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
    for name in ["configuration_rwkv7.py", "modeling_rwkv7.py", "tokenization_rwkv7.py"]:
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
    model = RWKV7ForCausalLM(config).to(dtype=dtype)
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
