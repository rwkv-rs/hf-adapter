from __future__ import annotations

from pathlib import Path

import pytest
import torch

from bench.bench_train_temp_alignment import build_parser
from rwkv7_hf import train_temp_cuda


def test_vendored_train_temp_sources_pin_provenance_and_license() -> None:
    root = Path(train_temp_cuda.__file__).resolve().parent / "csrc" / "train_temp"
    readme = (root / "README.md").read_text(encoding="utf-8")
    assert train_temp_cuda.TRAIN_TEMP_SOURCE_COMMIT in readme
    assert "Apache-2.0" in readme
    assert (root / "LICENSE").is_file()
    for filenames in train_temp_cuda._OP_SOURCES.values():
        assert all((root / filename).is_file() for filename in filenames)
    assert (root / "rwkv7_clampw_v3.cpp").is_file()
    assert (root / "rwkv7_clampw_v3_for_h100.cu").is_file()
    assert (root / "rwkv7_l2wrap_ce_bf16_v2.cpp").is_file()
    assert (root / "rwkv7_l2wrap_ce_bf16_v2.cu").is_file()


def test_train_temp_dense_mask_contract() -> None:
    train_temp_cuda._dense_mask_only(None)
    train_temp_cuda._dense_mask_only(torch.ones(2, 16, dtype=torch.long))
    with pytest.raises(ValueError, match="does not support padded"):
        train_temp_cuda._dense_mask_only(torch.tensor([[1, 1, 0]]))


def test_train_temp_backend_reports_unavailable_without_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert train_temp_cuda.train_temp_cuda_available() is False
    with pytest.raises(RuntimeError, match="requires Linux with an available CUDA GPU"):
        train_temp_cuda.load_train_temp_cuda_extension()


@pytest.mark.parametrize("command", ["capture-hf", "converge-hf"])
def test_train_temp_cli_rejects_native_combination(command: str, tmp_path: Path) -> None:
    parser = build_parser()
    common = [
        command,
        "--output-json",
        str(tmp_path / "result.json"),
        "--device",
        "cpu",
        "--seed",
        "1",
        "--model",
        "model",
        "--checkpoint-sha256",
        "sha",
        "--native",
        "--train-temp-cuda",
    ]
    if command == "capture-hf":
        common += [
            "--batch",
            str(tmp_path / "batch.safetensors"),
            "--snapshot",
            str(tmp_path / "snapshot.safetensors"),
        ]
    else:
        common += [
            "--sequence",
            str(tmp_path / "sequence.safetensors"),
            "--validation-batch",
            str(tmp_path / "validation.safetensors"),
        ]
    with pytest.raises(SystemExit):
        parser.parse_args(common)
