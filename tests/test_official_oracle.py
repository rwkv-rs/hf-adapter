from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))

from official_rwkv7_oracle import (  # noqa: E402
    OFFICIAL_NUMPY_SHA256,
    OFFICIAL_RNN_SHA256,
    OfficialRWKV7,
    sha256_file,
    strict_target_passed,
    tensor_metrics,
    tensor_passed,
)
from rwkv7_hf.configuration_rwkv7 import RWKV7Config  # noqa: E402
from rwkv7_hf.modeling_rwkv7 import RWKV7ForCausalLM  # noqa: E402


def _official_checkpoint(model: RWKV7ForCausalLM) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    config = model.config
    result = {
        "emb.weight": state["model.embeddings.weight"],
        "head.weight": state["lm_head.weight"],
        "ln_out.weight": state["model.norm.weight"],
        "ln_out.bias": state["model.norm.bias"],
    }
    for layer_idx in range(config.num_hidden_layers):
        model_prefix = f"model.layers.{layer_idx}."
        official_prefix = f"blocks.{layer_idx}."
        attention = model_prefix + "attn."
        official_attention = official_prefix + "att."
        ffn = model_prefix + "ffn."
        official_ffn = official_prefix + "ffn."
        for official_name, model_name in (
            ("ln0", "pre_norm"),
            ("ln1", "attn_norm"),
            ("ln2", "ffn_norm"),
        ):
            if official_name == "ln0" and layer_idx:
                continue
            result[official_prefix + official_name + ".weight"] = state[
                model_prefix + model_name + ".weight"
            ]
            result[official_prefix + official_name + ".bias"] = state[
                model_prefix + model_name + ".bias"
            ]
        for name in ("x_r", "x_w", "x_k", "x_v", "x_a", "x_g", "k_k", "k_a", "r_k"):
            result[official_attention + name] = state[attention + name]
        for official_name, model_name in (
            ("receptance", "r_proj"),
            ("key", "k_proj"),
            ("value", "v_proj"),
            ("output", "o_proj"),
        ):
            result[official_attention + official_name + ".weight"] = state[
                attention + model_name + ".weight"
            ]
        result[official_attention + "ln_x.weight"] = state[attention + "g_norm.weight"]
        result[official_attention + "ln_x.bias"] = state[attention + "g_norm.bias"]
        for kind in ("w", "a", "g"):
            low_rank = attention + kind + "_lora.lora."
            result[official_attention + kind + "1"] = state[low_rank + "0.weight"].T
            result[official_attention + kind + "2"] = state[low_rank + "2.weight"].T
            if kind != "g":
                result[official_attention + kind + "0"] = state[low_rank + "2.bias"]
        if layer_idx:
            low_rank = attention + "v_lora.lora."
            result[official_attention + "v1"] = state[low_rank + "0.weight"].T
            result[official_attention + "v2"] = state[low_rank + "2.weight"].T
            result[official_attention + "v0"] = state[low_rank + "2.bias"]
        else:
            # Present in official checkpoints but ignored by layer zero.
            result[official_attention + "v0"] = result[official_attention + "a0"].clone()
            result[official_attention + "v1"] = result[official_attention + "a1"].clone()
            result[official_attention + "v2"] = result[official_attention + "a2"].clone()
        result[official_ffn + "x_k"] = state[ffn + "x_k"]
        result[official_ffn + "key.weight"] = state[ffn + "key.weight"]
        result[official_ffn + "value.weight"] = state[ffn + "value.weight"]
    return result


def test_vendored_official_sources_are_pinned():
    vendor = ROOT / "evaluation" / "vendor" / "rwkv_lm"
    assert sha256_file(vendor / "rwkv_v7_numpy.py") == OFFICIAL_NUMPY_SHA256
    assert sha256_file(vendor / "rwkv_v7_demo_rnn.py") == OFFICIAL_RNN_SHA256


def test_official_equations_match_tiny_hf_model(tmp_path):
    config = RWKV7Config(
        vocab_size=32,
        hidden_size=8,
        attention_hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_heads=2,
        head_dim=4,
        decay_low_rank_dim=3,
        a_low_rank_dim=3,
        gate_low_rank_dim=3,
        v_low_rank_dim=3,
    )
    torch.manual_seed(5)
    model = RWKV7ForCausalLM(config).eval()
    checkpoint = tmp_path / "official.pth"
    torch.save(_official_checkpoint(model), checkpoint)
    oracle = OfficialRWKV7(
        checkpoint, device=torch.device("cpu"), dtype=torch.float32
    )
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    official_logits, official_state, blocks, final_hidden = oracle.forward(
        input_ids, collect_blocks=True
    )
    output = model(input_ids, use_cache=True, output_hidden_states=True)
    # Batched and token-wise BLAS calls can differ by one FP32 rounding unit;
    # the synthetic fixture must still agree far inside the release tolerance.
    assert tensor_metrics(official_logits, output.logits)["max_abs"] < 1e-7
    assert tensor_metrics(blocks[0], output.hidden_states[1])["max_abs"] < 1e-7
    assert tensor_metrics(final_hidden, output.hidden_states[-1])["max_abs"] < 1e-7
    for layer_idx in range(config.num_hidden_layers):
        row = tensor_metrics(
            official_state.recurrent_vk[layer_idx].transpose(-1, -2),
            output.past_key_values.recurrent_state[layer_idx],
        )
        assert row["max_abs"] < 1e-7


def test_official_fp32_logit_gate_uses_upstream_normalized_metric():
    reference = torch.tensor([[-5.0, 0.0, 5.0]])
    candidate = reference + torch.tensor([[2e-4, 2e-4, 2e-4]])
    row = tensor_metrics(reference, candidate)
    assert not row["fp32_allclose"]
    assert tensor_passed("fp32", row, logits=True)
    assert not tensor_passed("fp32", row, logits=False)


def test_calibrated_bf16_gate_keeps_original_target_as_diagnostic():
    reference = torch.arange(1, 17, dtype=torch.float32)
    candidate = reference.clone()
    candidate[-1] += 0.7
    row = tensor_metrics(reference, candidate)
    assert row["cosine"] >= 0.9995
    assert row["cosine"] < 0.9999
    assert tensor_passed("bf16", row, logits=True)
    assert not strict_target_passed("bf16", row, logits=True)
