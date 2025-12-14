#!/usr/bin/env python3
# coding=utf-8
"""Convert official RWKV-7 .pth checkpoints to a Hugging Face model directory.

The converted directory uses the repository-native RWKV-7 PreTrainedModel by
default and does not require FLA at load time. It contains config.json,
generation_config.json, model.safetensors, remote-code files,
tokenizer_config.json, and the RWKV trie vocab.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Dict, Tuple

import torch

try:
    from scripts.adapter_manifest import ADAPTER_FILES, LEGACY_REMOTE_CODE_FILES
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from adapter_manifest import ADAPTER_FILES, LEGACY_REMOTE_CODE_FILES

from rwkv7_hf.native_model import NativeRWKV7Config, NativeRWKV7ForCausalLM


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


def infer_attention_shape(weights: Dict[str, torch.Tensor]) -> tuple[int, int, int]:
    """Infer H, N, and attention width A directly from ``att.r_k``."""
    rk_shape = tensor_shape(weights, "blocks.0.att.r_k")
    if len(rk_shape) < 2:
        raise ValueError(f"Cannot infer attention shape from blocks.0.att.r_k={rk_shape}")
    num_heads, head_dim = (int(value) for value in rk_shape[-2:])
    if num_heads <= 0 or head_dim <= 0:
        raise ValueError(f"Invalid attention shape from blocks.0.att.r_k={rk_shape}")
    return num_heads, head_dim, num_heads * head_dim


def infer_head_dim(weights: Dict[str, torch.Tensor], hidden_size: int | None = None) -> int:
    """Compatibility helper returning N without requiring attention width A=D."""
    del hidden_size
    return infer_attention_shape(weights)[1]


def infer_value_dim(
    weights: Dict[str, torch.Tensor],
    num_layers: int,
    attention_hidden_size: int,
    num_heads: int,
) -> list[int]:
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
    if any(value_dim != attention_hidden_size for value_dim in dims):
        raise ValueError(
            "Native RWKV-7 conversion requires every value projection output "
            f"to equal attention_hidden_size={attention_hidden_size}; got {dims}"
        )
    return dims


def _expect_shape(
    weights: Dict[str, torch.Tensor],
    name: str,
    expected: tuple[int, ...],
) -> None:
    actual = tensor_shape(weights, name)
    if actual != expected:
        raise ValueError(f"{name} has shape {actual}; expected {expected}")


def _expect_recurrent_head_shape(
    weights: Dict[str, torch.Tensor],
    name: str,
    num_heads: int,
    head_dim: int,
) -> None:
    """Validate an official r_k tensor while accepting leading singleton axes."""
    actual = tensor_shape(weights, name)
    expected = (num_heads, head_dim)
    if len(actual) < 2 or actual[-2:] != expected:
        raise ValueError(f"{name} has shape {actual}; expected trailing dimensions {expected}")
    if any(dim != 1 for dim in actual[:-2]):
        raise ValueError(f"{name} has shape {actual}; leading dimensions must be singleton")


def validate_layer_shapes(
    weights: Dict[str, torch.Tensor],
    num_layers: int,
    hidden_size: int,
    num_heads: int,
    head_dim: int,
    attention_hidden_size: int,
) -> None:
    """Catch size/shape mismatches before constructing the HF model."""
    for layer_idx in range(num_layers):
        ffn_key = tensor_shape(weights, f"blocks.{layer_idx}.ffn.key.weight")
        if len(ffn_key) != 2 or int(ffn_key[1]) != hidden_size:
            raise ValueError(f"blocks.{layer_idx}.ffn.key.weight has inconsistent shape {ffn_key}")
        _expect_recurrent_head_shape(
            weights,
            f"blocks.{layer_idx}.att.r_k",
            num_heads,
            head_dim,
        )
        for projection in ("receptance", "key", "value"):
            _expect_shape(
                weights,
                f"blocks.{layer_idx}.att.{projection}.weight",
                (attention_hidden_size, hidden_size),
            )
        _expect_shape(
            weights,
            f"blocks.{layer_idx}.att.output.weight",
            (hidden_size, attention_hidden_size),
        )
        for affine in ("weight", "bias"):
            _expect_shape(
                weights,
                f"blocks.{layer_idx}.att.ln_x.{affine}",
                (attention_hidden_size,),
            )


def infer_config(
    weights: Dict[str, torch.Tensor],
    dtype_name: str,
    attn_mode: str,
    fuse_norm: bool,
) -> NativeRWKV7Config:
    hidden_size = tensor_shape(weights, "blocks.0.ffn.key.weight")[1]
    intermediate_size = tensor_shape(weights, "blocks.0.ffn.key.weight")[0]
    num_layers = infer_num_layers(weights)
    num_heads, head_dim, attention_hidden_size = infer_attention_shape(weights)
    value_dim = infer_value_dim(
        weights,
        num_layers,
        attention_hidden_size,
        num_heads,
    )
    validate_layer_shapes(
        weights,
        num_layers,
        hidden_size,
        num_heads,
        head_dim,
        attention_hidden_size,
    )
    try:
        v_low_rank_dim = tensor_shape(weights, "blocks.1.att.v1")[1]
    except KeyError:
        v_low_rank_dim = 32
    cfg = NativeRWKV7Config(
        attn_mode=attn_mode,
        vocab_size=tensor_shape(weights, "emb.weight")[0],
        hidden_size=hidden_size,
        attention_hidden_size=attention_hidden_size,
        hidden_ratio=intermediate_size / hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        value_dim=value_dim,
        decay_low_rank_dim=tensor_shape(weights, "blocks.0.att.w1")[1],
        gate_low_rank_dim=tensor_shape(weights, "blocks.0.att.g1")[1],
        a_low_rank_dim=tensor_shape(weights, "blocks.0.att.a1")[1],
        v_low_rank_dim=v_low_rank_dim,
        head_dim=head_dim,
        num_heads=num_heads,
        # 0 is unused by the official trie vocab; use it as a HF generation sentinel/pad id.
        pad_token_id=0,
        eos_token_id=0,
        bos_token_id=1,
        tie_word_embeddings=False,
        fuse_norm=fuse_norm,
    )
    cfg.torch_dtype = dtype_name
    return cfg


def build_template_model(config: NativeRWKV7Config, dtype: torch.dtype):
    """Construct the HF-shaped model used as the state_dict template.

    Conversion only needs module names and tensor shapes. The canonical native
    model intentionally uses the same converted key layout as the historical
    wrapper, so conversion has no FLA dependency.
    """

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
    for name in LEGACY_REMOTE_CODE_FILES:
        (output / name).unlink(missing_ok=True)
    for name in ADAPTER_FILES:
        shutil.copyfile(root / "rwkv7_hf" / name, output / name)
    if vocab_file is not None:
        shutil.copyfile(vocab_file, output / "rwkv_vocab_v20230424.txt")


def patch_hf_metadata(output: Path) -> None:
    cfg_path = output / "config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["architectures"] = ["NativeRWKV7ForCausalLM"]
    cfg["model_type"] = "rwkv7_native"
    cfg["auto_map"] = {
        "AutoConfig": "native_model.NativeRWKV7Config",
        "AutoModel": "native_model.NativeRWKV7Model",
        "AutoModelForCausalLM": "native_model.NativeRWKV7ForCausalLM",
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


def prepare_translated_weight(
    src_weight: torch.Tensor,
    *,
    src_name: str,
    dst_name: str,
    transposed: bool,
    expected: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Translate one tensor without cloning when its dtype/layout already fit."""

    weight = src_weight.detach()
    if transposed:
        weight = weight.t().contiguous()
    if tuple(weight.shape) != tuple(expected.shape):
        # Official time-mix vectors are sometimes stored as [H] while the HF
        # module keeps [1, 1, H] (or vice versa). They are the same parameter;
        # materialize the exact template shape so a directly saved state dict
        # loads without relying on copy_ broadcasting.
        if int(weight.numel()) == int(expected.numel()):
            weight = weight.reshape(expected.shape)
        else:
            raise AssertionError(
                f"Shape mismatch {src_name} -> {dst_name}: "
                f"{tuple(weight.shape)} vs {tuple(expected.shape)}"
            )
    if weight.dtype != dtype:
        weight = weight.to(dtype=dtype)
    return weight


def save_low_memory_model(
    *,
    weights: Dict[str, torch.Tensor],
    config: NativeRWKV7Config,
    dtype: torch.dtype,
    output: Path,
    max_shard_size: str,
) -> None:
    """Translate and shard a checkpoint without allocating a dense template.

    A 13.3B fp16 template is another ~26GB allocation on top of the official
    checkpoint. Building the exact module structure on the ``meta`` device
    preserves key/shape validation while keeping resident memory close to one
    checkpoint. Source entries are popped as translated tensors are retained,
    so dtype conversion also stays near that bound.
    """

    with torch.device("meta"):
        template = build_template_model(config, dtype)
    expected_state = template.state_dict()
    missing = set(expected_state)
    translated: Dict[str, torch.Tensor] = {}

    for src_name in list(weights):
        src_weight = weights.pop(src_name)
        dst_name, transposed = translate_name(src_name, config.num_hidden_layers)
        if not dst_name:
            continue
        if dst_name not in expected_state:
            raise KeyError(f"Translated name not in HF model: {src_name} -> {dst_name}")
        translated[dst_name] = prepare_translated_weight(
            src_weight,
            src_name=src_name,
            dst_name=dst_name,
            transposed=transposed,
            expected=expected_state[dst_name],
            dtype=dtype,
        )
        missing.discard(dst_name)

    allowed_missing = {"model.layers.0.pre_norm.weight", "model.layers.0.pre_norm.bias"}
    unexpected_missing = sorted(missing - allowed_missing)
    if unexpected_missing:
        raise KeyError(f"Uninitialized HF parameters: {unexpected_missing[:20]} ... total={len(unexpected_missing)}")
    for name in sorted(missing & allowed_missing):
        expected = expected_state[name]
        fill = torch.ones if name.endswith("weight") else torch.zeros
        translated[name] = fill(tuple(expected.shape), dtype=dtype, device="cpu")

    del template, expected_state
    from huggingface_hub import save_torch_state_dict
    from transformers import GenerationConfig

    config.save_pretrained(output)
    GenerationConfig.from_model_config(config).save_pretrained(output)
    save_torch_state_dict(
        translated,
        output,
        max_shard_size=max_shard_size,
        safe_serialization=True,
        metadata={"format": "pt"},
    )


def convert(args: argparse.Namespace) -> None:
    dtype_name, dtype = DTYPES[args.precision]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    weights = torch.load(
        args.input,
        weights_only=True,
        map_location="cpu",
        mmap=bool(args.low_memory),
    )
    config = infer_config(weights, dtype_name=dtype_name, attn_mode=args.attn_mode, fuse_norm=args.fuse_norm)
    if args.low_memory:
        save_low_memory_model(
            weights=weights,
            config=config,
            dtype=dtype,
            output=output,
            max_shard_size=args.max_shard_size,
        )
        vocab = Path(args.vocab_file) if args.vocab_file else None
        copy_adapter_files(output, vocab)
        patch_hf_metadata(output)
        print(f"Saved HF RWKV-7 model to: {output} (low-memory path)")
        return

    model = build_template_model(config, dtype)
    model_dict = model.state_dict()
    missing = set(model_dict)

    for src_name, src_weight in weights.items():
        dst_name, transposed = translate_name(src_name, config.num_hidden_layers)
        if not dst_name:
            continue
        if dst_name not in model_dict:
            raise KeyError(f"Translated name not in HF model: {src_name} -> {dst_name}")
        expected = model_dict[dst_name]
        weight = prepare_translated_weight(
            src_weight,
            src_name=src_name,
            dst_name=dst_name,
            transposed=transposed,
            expected=expected,
            dtype=expected.dtype,
        )
        expected.copy_(weight)
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
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Build the template on meta and stream translated tensors into safetensors shards; required for 13B on ~48GB RAM hosts",
    )
    args = parser.parse_args()
    convert(args)


if __name__ == "__main__":
    main()
