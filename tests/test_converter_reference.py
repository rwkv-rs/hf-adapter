from __future__ import annotations

import json

from rwkv7_hf.converter import (
    ADAPTER_LAYOUTS,
    REFERENCE_ADAPTER_FILES,
    build_parser,
    install_adapter_layout,
    patch_hf_metadata,
)


def test_reference_is_default_layout():
    parser = build_parser()
    args = parser.parse_args(["--input", "model.pth", "--output", "out"])
    assert args.adapter_layout == "reference"
    assert ADAPTER_LAYOUTS == ("reference", "thin")


def test_reference_layout_contains_complete_model_code(tmp_path):
    vocab = tmp_path / "source_vocab.txt"
    vocab.write_text("test vocab\n")
    install_adapter_layout(
        tmp_path,
        vocab,
        adapter_layout="reference",
        runtime_version="0.9.0",
    )
    for name in REFERENCE_ADAPTER_FILES:
        assert (tmp_path / name).is_file()
    assert (tmp_path / "rwkv_vocab_v20230424.txt").is_file()

    (tmp_path / "config.json").write_text(
        json.dumps({"vocab_size": 64, "model_type": "old"})
    )
    patch_hf_metadata(tmp_path, adapter_layout="reference")
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["model_type"] == "rwkv7"
    assert config["architectures"] == ["RWKV7ForCausalLM"]
    assert "rwkv7_hf_runtime_version" not in config
    tokenizer_config = json.loads((tmp_path / "tokenizer_config.json").read_text())
    assert "Assistant:" in tokenizer_config["chat_template"]
