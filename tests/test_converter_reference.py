from __future__ import annotations

import json

from rwkv7_hf_tools.converter import (
    MODEL_CODE_FILES,
    build_parser,
    copy_model_files,
    patch_hf_metadata,
)


def test_converter_has_one_self_contained_layout():
    parser = build_parser()
    args = parser.parse_args(["--input", "model.pth", "--output", "out"])
    assert not hasattr(args, "adapter_layout")
    assert not hasattr(args, "runtime_package_version")


def test_self_contained_output_contains_complete_model_code(tmp_path):
    vocab = tmp_path / "source_vocab.txt"
    vocab.write_text("test vocab\n")
    copy_model_files(tmp_path, vocab)
    for name in MODEL_CODE_FILES:
        assert (tmp_path / name).is_file()
    assert (tmp_path / "rwkv_vocab_v20230424.txt").is_file()

    (tmp_path / "config.json").write_text(
        json.dumps({"vocab_size": 64, "model_type": "old"})
    )
    patch_hf_metadata(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["model_type"] == "rwkv7"
    assert config["architectures"] == ["RWKV7ForCausalLM"]
    tokenizer_config = json.loads((tmp_path / "tokenizer_config.json").read_text())
    assert "Assistant:" in tokenizer_config["chat_template"]
