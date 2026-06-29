from __future__ import annotations

import json

import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from rwkv7_hf.modeling_rwkv7 import RWKV7ForCausalLM


def test_save_reload_and_package_free_auto_model(tmp_path, tiny_config):
    torch.manual_seed(19)
    model = RWKV7ForCausalLM(tiny_config).eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        expected = model(input_ids=ids, use_cache=False).logits
    model.save_pretrained(tmp_path, safe_serialization=True)

    required = {
        "config.json",
        "model.safetensors",
        "configuration_rwkv7.py",
        "cache_rwkv7.py",
        "ops_rwkv7.py",
        "modeling_rwkv7.py",
    }
    assert required.issubset({path.name for path in tmp_path.iterdir()})
    config_json = json.loads((tmp_path / "config.json").read_text())
    assert config_json["model_type"] == "rwkv7"
    assert config_json["auto_map"]["AutoModelForCausalLM"] == (
        "modeling_rwkv7.RWKV7ForCausalLM"
    )

    loaded_config = AutoConfig.from_pretrained(tmp_path, trust_remote_code=True)
    assert loaded_config.model_type == "rwkv7"
    backbone = AutoModel.from_pretrained(tmp_path, trust_remote_code=True).eval()
    loaded = AutoModelForCausalLM.from_pretrained(
        tmp_path, trust_remote_code=True
    ).eval()
    with torch.inference_mode():
        actual = loaded(input_ids=ids, use_cache=False).logits
        hidden = backbone(input_ids=ids, use_cache=False).last_hidden_state
    torch.testing.assert_close(actual, expected)
    assert hidden.shape == (1, 4, tiny_config.hidden_size)


def test_greedy_and_beam_generation(tiny_config):
    model = RWKV7ForCausalLM(tiny_config).eval()
    ids = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        greedy = model.generate(
            ids,
            max_new_tokens=3,
            do_sample=False,
            eos_token_id=None,
            pad_token_id=0,
        )
        beam = model.generate(
            ids,
            max_new_tokens=2,
            num_beams=2,
            do_sample=False,
            eos_token_id=None,
            pad_token_id=0,
        )
    assert greedy.shape == (1, 6)
    assert beam.shape == (1, 5)
