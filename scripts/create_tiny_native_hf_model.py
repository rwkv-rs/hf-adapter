#!/usr/bin/env python3
"""Create a tiny random native RWKV-7 remote-code fixture for API smokes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from rwkv7_hf.native_model import NativeRWKV7Config, NativeRWKV7ForCausalLM


def write_byte_vocab(path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for byte in range(256):
            f.write(f"{byte + 1} {bytes([byte])!r} 1\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=20260724)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = NativeRWKV7Config(
        vocab_size=320, hidden_size=16, num_hidden_layers=2, head_dim=4,
        intermediate_size=32, decay_low_rank_dim=4, gate_low_rank_dim=4,
        a_low_rank_dim=4, v_low_rank_dim=4, use_cache=True,
        pad_token_id=0, eos_token_id=0, bos_token_id=1,
        tie_word_embeddings=False,
    )
    NativeRWKV7ForCausalLM(cfg).eval().save_pretrained(output, safe_serialization=True)
    (output / "native_model.py").write_text(
        '"""Remote-code bridge to the installed RWKV-7 adapter."""\n'
        "from rwkv7_hf.native_model import NativeRWKV7Config, NativeRWKV7Model, NativeRWKV7ForCausalLM\n"
        '__all__ = ["NativeRWKV7Config", "NativeRWKV7Model", "NativeRWKV7ForCausalLM"]\n',
        encoding="utf-8",
    )
    root = Path(__file__).resolve().parents[1]
    (output / "tokenization_rwkv7.py").write_bytes((root / "rwkv7_hf/tokenization_rwkv7.py").read_bytes())
    write_byte_vocab(output / "rwkv_vocab_v20230424.txt")
    config_path = output / "config.json"
    data = json.loads(config_path.read_text(encoding="utf-8"))
    data.update({
        "architectures": ["NativeRWKV7ForCausalLM"],
        "model_type": "rwkv7_native",
        "auto_map": {
            "AutoConfig": "native_model.NativeRWKV7Config",
            "AutoModel": "native_model.NativeRWKV7Model",
            "AutoModelForCausalLM": "native_model.NativeRWKV7ForCausalLM",
        },
    })
    config_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    (output / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "RWKV7Tokenizer",
        "auto_map": {"AutoTokenizer": ["tokenization_rwkv7.RWKV7Tokenizer", None]},
        "model_vocab_size": 320, "pad_token": "<|padding|>",
        "eos_token": "<|endoftext|>", "errors": "replace",
    }, indent=2) + "\n", encoding="utf-8")
    (output / "special_tokens_map.json").write_text(json.dumps({
        "pad_token": "<|padding|>", "eos_token": "<|endoftext|>"
    }, indent=2) + "\n", encoding="utf-8")
    print(output)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
