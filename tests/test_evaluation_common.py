from __future__ import annotations

from evaluation.common import model_fingerprint


def test_model_fingerprint_covers_code_tokenizer_vocab_and_weights(tmp_path):
    rows = {
        "config.json": b"{}",
        "modeling_rwkv7.py": b"class RWKV7Model: pass\n",
        "rwkv_vocab_v20230424.txt": b"vocab",
        "tokenizer_config.json": b"{}",
        "model.safetensors": b"weights",
    }
    for name, payload in rows.items():
        (tmp_path / name).write_bytes(payload)
    fingerprint = model_fingerprint(tmp_path)
    assert set(fingerprint["payloads"]) == set(rows)
    assert fingerprint["weights"][0]["name"] == "model.safetensors"
    assert len(fingerprint["resolved_revision"]) == 64
