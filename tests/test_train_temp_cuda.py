from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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


def test_train_temp_causal_cross_entropy_shifts_dense_labels(monkeypatch) -> None:
    logits = torch.randn(2, 5, 11)
    labels = torch.randint(0, 11, (2, 5))
    sentinel = torch.tensor(1.25)

    def fake_fused(shifted_logits: torch.Tensor, shifted_labels: torch.Tensor) -> torch.Tensor:
        assert shifted_logits.is_contiguous()
        assert shifted_labels.is_contiguous()
        torch.testing.assert_close(shifted_logits, logits[:, :-1])
        torch.testing.assert_close(shifted_labels, labels[:, 1:])
        return sentinel

    monkeypatch.setattr(train_temp_cuda, "train_temp_fused_cross_entropy", fake_fused)
    assert train_temp_cuda.train_temp_causal_cross_entropy(logits, labels) is sentinel


@pytest.mark.parametrize(
    ("logits", "labels", "error"),
    [
        (torch.randn(2, 5), torch.zeros(2, 5, dtype=torch.long), "logits must have shape"),
        (torch.randn(2, 5, 11), torch.zeros(10, dtype=torch.long), "labels must have shape"),
        (
            torch.randn(2, 5, 11),
            torch.zeros(2, 4, dtype=torch.long),
            "share batch/token",
        ),
        (torch.randn(2, 1, 11), torch.zeros(2, 1, dtype=torch.long), "at least two tokens"),
        (torch.randn(2, 5, 11), torch.zeros(2, 5, dtype=torch.int32), "torch.int64"),
        (torch.randn(2, 5, 11), torch.full((2, 5), -100, dtype=torch.long), "-100 is unsupported"),
    ],
)
def test_train_temp_causal_cross_entropy_rejects_unsupported_batches(
    logits: torch.Tensor,
    labels: torch.Tensor,
    error: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        train_temp_cuda.train_temp_causal_cross_entropy(logits, labels)


def test_train_temp_backend_enable_disable_restores_model(monkeypatch) -> None:
    import fla.layers.rwkv7 as layer_module
    import fla.models.rwkv7.modeling_rwkv7 as model_module

    class FakeAttention(torch.nn.Module):
        def forward(self, value):
            return ("attention", value)

    class FakeFeedForward(torch.nn.Module):
        def forward(self, value):
            return ("ffn", value)

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention = FakeAttention()
            self.ffn = FakeFeedForward()
            self.config = SimpleNamespace(use_cache=True)

    monkeypatch.setattr(train_temp_cuda, "load_train_temp_cuda_extension", lambda: None)
    monkeypatch.setattr(layer_module, "RWKV7Attention", FakeAttention)
    monkeypatch.setattr(model_module, "RWKV7FeedForward", FakeFeedForward)

    model = FakeModel()
    metadata = train_temp_cuda.enable_train_temp_cuda_backend(model)
    assert metadata["attention_modules"] == metadata["ffn_modules"] == 1
    assert model.config.use_cache is False
    assert model._rwkv7_train_temp_cuda_enabled is True

    train_temp_cuda.disable_train_temp_cuda_backend(model)
    assert model.config.use_cache is True
    assert model._rwkv7_train_temp_cuda_enabled is False
    assert model.attention.forward(3) == ("attention", 3)
    assert model.ffn.forward(4) == ("ffn", 4)


def test_train_temp_backend_rejects_unbalanced_model_before_patching(
    monkeypatch,
) -> None:
    import fla.layers.rwkv7 as layer_module
    import fla.models.rwkv7.modeling_rwkv7 as model_module

    class FakeAttention(torch.nn.Module):
        def forward(self, value):
            return value

    class FakeFeedForward(torch.nn.Module):
        pass

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention = FakeAttention()
            self.config = SimpleNamespace(use_cache=True)

    monkeypatch.setattr(train_temp_cuda, "load_train_temp_cuda_extension", lambda: None)
    monkeypatch.setattr(layer_module, "RWKV7Attention", FakeAttention)
    monkeypatch.setattr(model_module, "RWKV7FeedForward", FakeFeedForward)

    model = FakeModel()
    with pytest.raises(TypeError, match="balanced FLA RWKV-7 model"):
        train_temp_cuda.enable_train_temp_cuda_backend(model)
    assert not hasattr(model.attention, "_rwkv7_train_temp_original_forward")
    assert model.config.use_cache is True


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
