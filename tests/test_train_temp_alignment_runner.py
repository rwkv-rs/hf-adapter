from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from bench.bench_train_temp_alignment import (
    compare_artifacts,
    make_deterministic_batch,
    normalize_official_tensors,
    write_json_atomic,
)


def _artifact(path: Path, snapshot: Path, *, loss: float, phase: str = "backward") -> None:
    write_json_atomic(
        path,
        {
            "schema_version": 1,
            "axis": "train_temp_alignment_capture",
            "backend": "official" if "official" in path.name else "hf",
            "phase": phase,
            "precision": "bf16",
            "checkpoint_sha256": "checkpoint-sha",
            "batch_sha256": "batch-sha",
            "loss": loss,
            "snapshot_file": str(snapshot),
        },
    )


def test_make_deterministic_batch_is_shifted_and_repeatable(tmp_path: Path) -> None:
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"
    first_meta = make_deterministic_batch(first, vocab_size=32, batch_size=2, seq_len=16, seed=42)
    second_meta = make_deterministic_batch(second, vocab_size=32, batch_size=2, seq_len=16, seed=42)

    a = load_file(first)
    b = load_file(second)
    assert tuple(a["input_ids"].shape) == (2, 16)
    assert tuple(a["targets"].shape) == (2, 16)
    assert torch.equal(a["input_ids"][:, 1:], a["targets"][:, :-1])
    assert torch.equal(a["input_ids"], b["input_ids"])
    assert a["input_ids"].untyped_storage().data_ptr() != a["targets"].untyped_storage().data_ptr()
    assert first_meta["content_sha256"] == second_meta["content_sha256"]


def test_compare_artifacts_accepts_matching_snapshots(tmp_path: Path) -> None:
    official_snapshot = tmp_path / "official.safetensors"
    hf_snapshot = tmp_path / "hf.safetensors"
    tensors = {
        "logits": torch.tensor([1.0, 2.0, 3.0]),
        "grad::model.layers.0.attn.k_proj.weight": torch.tensor([0.5, -0.5]),
    }
    save_file(tensors, official_snapshot)
    save_file({key: value.clone() for key, value in tensors.items()}, hf_snapshot)
    official_json = tmp_path / "official.json"
    hf_json = tmp_path / "hf.json"
    _artifact(official_json, official_snapshot, loss=4.25)
    _artifact(hf_json, hf_snapshot, loss=4.25)

    report = compare_artifacts(official_json, hf_json)

    assert report["status"] == "pass"
    assert report["tensor_count"] == 2
    assert report["missing_in_reference"] == []
    assert report["missing_in_candidate"] == []
    assert report["worst_cosine"] == 1.0
    assert report["max_relative_l2"] == 0.0


def test_compare_artifacts_rejects_provenance_and_tensor_failures(tmp_path: Path) -> None:
    official_snapshot = tmp_path / "official.safetensors"
    hf_snapshot = tmp_path / "hf.safetensors"
    save_file({"grad::x": torch.tensor([1.0, 0.0])}, official_snapshot)
    save_file({"grad::x": torch.tensor([-1.0, 0.0]), "grad::extra": torch.ones(1)}, hf_snapshot)
    official_json = tmp_path / "official.json"
    hf_json = tmp_path / "hf.json"
    _artifact(official_json, official_snapshot, loss=1.0)
    _artifact(hf_json, hf_snapshot, loss=2.0)
    candidate = json.loads(hf_json.read_text(encoding="utf-8"))
    candidate["batch_sha256"] = "different-batch"
    write_json_atomic(hf_json, candidate)

    report = compare_artifacts(official_json, hf_json)

    assert report["status"] == "fail"
    assert "batch_sha256" in report["provenance_mismatches"]
    assert report["missing_in_reference"] == ["grad::extra"]
    assert report["loss_abs_diff"] == 1.0
    assert report["worst_cosine"] == -1.0
    assert report["failures"]


def test_write_json_atomic_leaves_valid_json(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "result.json"
    write_json_atomic(output, {"status": "pass", "value": 3})
    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "pass", "value": 3}
    assert not output.with_suffix(output.suffix + ".tmp").exists()


def test_normalize_official_tensors_applies_converter_transpose() -> None:
    tensors = {
        "blocks.0.att.w1": torch.arange(6, dtype=torch.float32).view(2, 3),
        "blocks.0.att.w0": torch.ones(3),
        "blocks.0.att.v1": torch.full((2, 2), 7.0),
    }

    def translate(name: str, num_layers: int) -> tuple[str, bool]:
        assert num_layers == 1
        mapping = {
            "blocks.0.att.w1": ("model.layers.0.attn.w_lora.lora.0.weight", True),
            "blocks.0.att.w0": ("model.layers.0.attn.w_lora.lora.2.bias", False),
            "blocks.0.att.v1": ("", False),
        }
        return mapping[name]

    normalized = normalize_official_tensors(
        tensors,
        num_layers=1,
        prefix="grad::",
        translate=translate,
    )
    assert set(normalized) == {
        "grad::model.layers.0.attn.w_lora.lora.0.weight",
        "grad::model.layers.0.attn.w_lora.lora.2.bias",
    }
    torch.testing.assert_close(
        normalized["grad::model.layers.0.attn.w_lora.lora.0.weight"],
        tensors["blocks.0.att.w1"].t(),
    )


def test_checked_in_official_config_is_production_shaped() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs" / "train_temp_x070_12x768.json").read_text(encoding="utf-8")
    )
    assert config["my_testing"] == "x070"
    assert config["head_size"] == 64
    assert config["n_layer"] == 12
    assert config["n_embd"] == 768
    assert config["dim_att"] == config["n_embd"]
    assert config["dim_ffn"] == 2688
    assert config["ctx_len"] >= 512
    assert config["vocab_size"] == 65536
