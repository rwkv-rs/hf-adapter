from __future__ import annotations

import builtins
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from bench.bench_train_temp_alignment import build_parser
from rwkv7_hf import train_temp_cuda
from rwkv7_hf.native_model import NativeRWKV7Config, NativeRWKV7ForCausalLM


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


def test_cuda_include_paths_support_pip_split_toolkit(tmp_path, monkeypatch) -> None:
    cuda_home = tmp_path / "cuda"
    (cuda_home / "include").mkdir(parents=True)
    target_include = cuda_home / "targets" / "x86_64-linux" / "include"
    target_include.mkdir(parents=True)
    site_packages = tmp_path / "site-packages"
    fake_torch = site_packages / "torch" / "__init__.py"
    fake_torch.parent.mkdir(parents=True)
    fake_torch.write_text("", encoding="utf-8")
    cusparse = site_packages / "nvidia" / "cusparse" / "include"
    cublas = site_packages / "nvidia" / "cublas" / "include"
    cusparse.mkdir(parents=True)
    cublas.mkdir(parents=True)
    monkeypatch.setattr(train_temp_cuda.torch, "__file__", str(fake_torch))

    assert train_temp_cuda._cuda_include_paths(cuda_home) == [
        str(cuda_home / "include"),
        str(cublas),
        str(cusparse),
    ]
    assert train_temp_cuda._cuda_include_paths(
        cuda_home, include_target=True
    ) == [
        str(cuda_home / "include"),
        str(target_include),
        str(cublas),
        str(cusparse),
    ]


def test_resolve_cuda_home_refreshes_cpp_extension_cache(tmp_path, monkeypatch) -> None:
    cuda_home = tmp_path / "cuda"
    nvcc = cuda_home / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("", encoding="utf-8")
    cpp_extension = SimpleNamespace(CUDA_HOME=None)
    monkeypatch.setenv("CUDA_HOME", str(cuda_home))

    assert train_temp_cuda._resolve_cuda_home(cpp_extension) == cuda_home.resolve()
    assert cpp_extension.CUDA_HOME == str(cuda_home.resolve())


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
    layer_module = pytest.importorskip("fla.layers.rwkv7")
    model_module = pytest.importorskip("fla.models.rwkv7.modeling_rwkv7")

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
    layer_module = pytest.importorskip("fla.layers.rwkv7")
    model_module = pytest.importorskip("fla.models.rwkv7.modeling_rwkv7")

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


def test_train_temp_backend_enables_native_model_without_fla_patching(monkeypatch) -> None:
    config = NativeRWKV7Config(
        vocab_size=17,
        hidden_size=64,
        num_hidden_layers=2,
        head_dim=64,
        intermediate_size=128,
        decay_low_rank_dim=8,
        gate_low_rank_dim=8,
        a_low_rank_dim=8,
        v_low_rank_dim=8,
        use_cache=True,
    )
    model = NativeRWKV7ForCausalLM(config)
    monkeypatch.setattr(train_temp_cuda, "load_train_temp_cuda_extension", lambda: None)
    original_import = builtins.__import__

    def import_without_fla(name, *args, **kwargs):
        if name == "fla" or name.startswith("fla."):
            raise AssertionError("Native train_temp enablement must not import FLA")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_fla)

    metadata = train_temp_cuda.enable_train_temp_cuda_backend(model)
    assert metadata["backend"] == "native_train_temp_cuda"
    assert metadata["attention_modules"] == metadata["ffn_modules"] == 2
    assert model.config.use_cache is False
    assert model._rwkv7_train_temp_cuda_enabled is True
    assert all(layer.attn._rwkv7_train_temp_cuda_enabled for layer in model.model.layers)
    assert all(layer.ffn._rwkv7_train_temp_cuda_enabled for layer in model.model.layers)

    train_temp_cuda.disable_train_temp_cuda_backend(model)
    assert model.config.use_cache is True
    assert model._rwkv7_train_temp_cuda_enabled is False
    assert all(not hasattr(layer.attn, "_rwkv7_train_temp_cuda_enabled") for layer in model.model.layers)
    assert all(not hasattr(layer.ffn, "_rwkv7_train_temp_cuda_enabled") for layer in model.model.layers)


def test_native_causal_lm_dispatches_injected_train_temp_forward() -> None:
    model = NativeRWKV7ForCausalLM(
        NativeRWKV7Config(
            vocab_size=17,
            hidden_size=64,
            num_hidden_layers=1,
            head_dim=64,
            intermediate_size=128,
            decay_low_rank_dim=8,
            gate_low_rank_dim=8,
            a_low_rank_dim=8,
            v_low_rank_dim=8,
            use_cache=False,
        )
    )
    sentinel = object()
    captured = {}

    def injected(**kwargs):
        captured.update(kwargs)
        return sentinel

    model._rwkv7_train_temp_forward = injected
    input_ids = torch.tensor([[1, 2, 3]])
    assert model(input_ids=input_ids, use_cache=False) is sentinel
    assert captured["input_ids"] is input_ids
    assert captured["use_cache"] is False


@pytest.mark.parametrize("command", ["capture-hf", "converge-hf"])
def test_train_temp_cli_accepts_native_combination(command: str, tmp_path: Path) -> None:
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
            "--phase",
            "backward",
        ]
    else:
        common += [
            "--sequence",
            str(tmp_path / "sequence.safetensors"),
            "--validation-batch",
            str(tmp_path / "validation.safetensors"),
        ]
    args = parser.parse_args(common)
    assert args.native is True
    assert args.train_temp_cuda is True
