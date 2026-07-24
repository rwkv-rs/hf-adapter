import json
from pathlib import Path

import torch

from rwkv7_hf.ascend_reference_oracle import (
    FLA_COMMIT,
    NaiveRWKV7Oracle,
    SafetensorStore,
    evaluate_capture,
    state_tensor_map,
    tensor_map_sha256,
)


def _tiny_checkpoint(root: Path, *, layers: int = 2) -> Path:
    from safetensors.torch import save_file

    torch.manual_seed(19)
    hidden, heads, head_dim, intermediate, vocab = 8, 2, 4, 16, 32
    config = {
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_hidden_layers": layers,
        "num_heads": 1,  # deliberately stale; r_k is authoritative
        "head_dim": head_dim,
        "norm_eps": 1e-5,
        "vocab_size": vocab,
    }
    tensors = {
        "model.embeddings.weight": torch.randn(vocab, hidden) * 0.1,
        "model.norm.weight": torch.randn(hidden) * 0.05 + 1,
        "model.norm.bias": torch.randn(hidden) * 0.01,
        "lm_head.weight": torch.randn(vocab, hidden) * 0.1,
    }
    for layer in range(layers):
        base = f"model.layers.{layer}"
        for norm in ("attn_norm", "ffn_norm"):
            tensors[f"{base}.{norm}.weight"] = torch.randn(hidden) * 0.05 + 1
            tensors[f"{base}.{norm}.bias"] = torch.randn(hidden) * 0.01
        if layer == 0:
            tensors[f"{base}.pre_norm.weight"] = torch.randn(hidden) * 0.05 + 1
            tensors[f"{base}.pre_norm.bias"] = torch.randn(hidden) * 0.01
        attn = base + ".attn"
        for role in ("r", "w", "k", "v", "a", "g"):
            tensors[f"{attn}.x_{role}"] = torch.rand(1, 1, hidden)
        tensors[f"{attn}.k_k"] = torch.randn(hidden)
        tensors[f"{attn}.k_a"] = torch.randn(hidden) * 0.05 + 1
        tensors[f"{attn}.r_k"] = torch.randn(heads, head_dim) * 0.05
        for proj in ("r_proj", "k_proj", "v_proj", "o_proj"):
            tensors[f"{attn}.{proj}.weight"] = torch.randn(hidden, hidden) * 0.1
        ranks = {"w_lora": 3, "a_lora": 3, "g_lora": 5}
        if layer:
            ranks["v_lora"] = 3
        for name, rank in ranks.items():
            tensors[f"{attn}.{name}.lora.0.weight"] = torch.randn(rank, hidden) * 0.1
            tensors[f"{attn}.{name}.lora.2.weight"] = torch.randn(hidden, rank) * 0.1
            if name != "g_lora":
                tensors[f"{attn}.{name}.lora.2.bias"] = torch.randn(hidden) * 0.01
        tensors[f"{attn}.g_norm.weight"] = torch.randn(hidden) * 0.05 + 1
        tensors[f"{attn}.g_norm.bias"] = torch.randn(hidden) * 0.01
        ffn = base + ".ffn"
        tensors[f"{ffn}.x_k"] = torch.rand(hidden)
        tensors[f"{ffn}.key.weight"] = torch.randn(intermediate, hidden) * 0.1
        tensors[f"{ffn}.value.weight"] = torch.randn(hidden, intermediate) * 0.1
    root.mkdir()
    save_file(tensors, root / "model.safetensors")
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "model.safetensors" for name in tensors}}),
        encoding="utf-8",
    )
    return root


def test_pinned_source_and_tensor_hash_are_stable():
    assert FLA_COMMIT == "d1ce07369d581813553f30a750af3b6b5f9af6a9"
    tensors = {"a": torch.tensor([1, 2], dtype=torch.int64), "b": torch.tensor([3.0])}
    assert tensor_map_sha256(tensors) == tensor_map_sha256(dict(reversed(list(tensors.items()))))
    changed = dict(tensors)
    changed["a"] = torch.tensor([1, 3], dtype=torch.int64)
    assert tensor_map_sha256(tensors) != tensor_map_sha256(changed)


def test_tiny_incremental_and_b2_ragged_match_compacted_sequences(tmp_path):
    model_dir = _tiny_checkpoint(tmp_path / "tiny")
    with SafetensorStore(model_dir) as store:
        oracle = NaiveRWKV7Oracle(store, dtype=torch.float32)
        assert oracle.config_repair == {
            "declared_num_heads": 1,
            "inferred_num_heads": 2,
            "inference_source": "model.layers.0.attn.r_k.shape",
            "changed": True,
        }
        ids = torch.tensor([[3, 4, 5]])
        full = oracle.forward(ids)
        state = None
        pieces = []
        for token in ids[0]:
            step = oracle.forward(token.view(1, 1), state=state)
            state = step.state
            pieces.append(step.logits)
        torch.testing.assert_close(torch.cat(pieces, dim=1), full.logits, rtol=1e-5, atol=1e-6)
        for left, right in zip(state.recurrent, full.state.recurrent):
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)

        ragged_ids = torch.tensor([[3, 4, 5, 6], [0, 0, 7, 8]])
        ragged_mask = torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]])
        batched = oracle.forward(ragged_ids, attention_mask=ragged_mask)
        compact = oracle.forward(torch.tensor([[7, 8]]))
        torch.testing.assert_close(batched.logits[1, -1], compact.logits[0, -1], rtol=1e-5, atol=1e-6)
        for left, right in zip(batched.state.recurrent, compact.state.recurrent):
            torch.testing.assert_close(left[1], right[0], rtol=1e-5, atol=1e-6)
        for left, right in zip(batched.state.attn_shift, compact.state.attn_shift):
            torch.testing.assert_close(left[1], right[0], rtol=1e-5, atol=1e-6)
        assert batched.state.valid_tokens.tolist() == [4, 2]


def test_capture_gate_is_fail_closed_on_missing_or_changed_tensor(tmp_path):
    model_dir = _tiny_checkpoint(tmp_path / "tiny")
    with SafetensorStore(model_dir) as store:
        output = NaiveRWKV7Oracle(store, dtype=torch.float32).forward(torch.tensor([[3, 4]]))
    reference = {
        "b1.prefill.logits": output.logits,
        "b1.greedy.token_ids": torch.tensor([4]),
        "b1.input.attention_mask": torch.tensor([[1, 1]]),
    }
    reference.update(state_tensor_map(output.state, "b1.prefill.state"))
    exact = {name: value.clone() for name, value in reference.items()}
    assert evaluate_capture(reference, exact)["status"] == "pass"
    missing = dict(exact)
    missing.pop("b1.prefill.logits")
    report = evaluate_capture(reference, missing)
    assert report["status"] == "fail" and not report["gates"]["complete_capture"]
    changed = dict(exact)
    changed["b1.greedy.token_ids"] = torch.tensor([5])
    report = evaluate_capture(reference, changed)
    assert report["status"] == "fail"
    assert not report["gates"]["greedy_token_ids_exact"]
    changed_mask = dict(exact)
    changed_mask["b1.input.attention_mask"] = torch.tensor([[1, 0]])
    assert evaluate_capture(reference, changed_mask)["status"] == "fail"
