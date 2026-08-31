from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_hf_release", ROOT / "scripts" / "prepare_hf_release.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_release_config_removes_backend_policy():
    config = MODULE.canonical_config(
        {
            "hidden_size": 256,
            "attention_hidden_size": 256,
            "attn_mode": "chunk",
            "fuse_norm": True,
            "kernel_impl": "triton",
            "model_kernel_impl": "native",
            "rwkv7_backend": "optimized",
        }
    )
    assert config["model_type"] == "rwkv7"
    assert config["architectures"] == ["RWKV7ForCausalLM"]
    assert config["auto_map"] == MODULE.AUTO_MAP
    assert config["head_dim"] == 64
    assert config["num_heads"] == config["num_attention_heads"] == 4
    assert (
        not {
            "attn_mode",
            "fuse_norm",
            "kernel_impl",
            "model_kernel_impl",
            "rwkv7_backend",
        }
        & config.keys()
    )


def test_v1_model_card_keeps_reference_self_contained_and_kernel_optional():
    card = MODULE.model_card(
        "owner/rwkv7-g1d-0.1b-hf",
        {"num_hidden_layers": 4, "hidden_size": 256, "vocab_size": 65536},
        {
            "source": {"filename": "model.pth", "sha256": "source-sha"},
            "weights": {"parameter_count": 100, "dtype": "float16"},
        },
        "code-sha",
        "v1.0.0",
    )
    normalized = " ".join(card.split())
    assert 'revision = "v1.0.0"' in card
    assert '"rwkv7-kernels==1.0.0"' in card
    assert '"rwkv7-hf[kernels]==1.0.0"' in card
    assert "does **not** require `rwkv7-hf`" in card
    assert "does not replace the model/config/cache classes" in card
    assert "https://github.com/rwkv-rs/hf-adapter" in card
    assert "123123213weqw/hf-adapter" not in card
    assert "one complete readable reference program" in normalized
    assert "isolated diagnostics only" in normalized
    assert "not certified HF training routes" in normalized
    assert "may accelerate" not in card


def test_version_specific_release_script_names_are_removed():
    scripts = ROOT / "scripts"
    assert (scripts / "prepare_hf_release.py").is_file()
    assert (scripts / "publish_hf_release.py").is_file()
    assert not (scripts / "prepare_hf_v090_release.py").exists()
    assert not (scripts / "publish_hf_v090_release.py").exists()


def test_release_stage_requires_source_sha_to_match_checkout():
    with pytest.raises(ValueError, match="does not equal checkout HEAD"):
        MODULE.verify_reference_checkout(ROOT, "0" * 40)


def test_weight_inventory_rejects_missing_lfs_metadata():
    class Sibling:
        rfilename = "model.safetensors"
        lfs = {"size": 123, "sha256": None}

    with pytest.raises(ValueError, match="LFS SHA256"):
        MODULE.weight_rows([Sibling()])
