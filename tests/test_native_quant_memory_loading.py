from argparse import Namespace
import json
from pathlib import Path

import pytest
import torch

from bench import bench_native_quant_e2e_decode as decode_bench


ROOT = Path(__file__).resolve().parents[1]


def _args(**overrides) -> Namespace:
    values = {
        "quantize_before_device": True,
        "device": "cuda",
        "single_quantization": "mm4",
        "policy": "memory",
        "paired_baseline": False,
        "allow_missing_baseline": True,
    }
    values.update(overrides)
    return Namespace(**values)


def test_quantize_before_device_accepts_explicit_memory_quant_only_mode() -> None:
    decode_bench.validate_quantize_before_device(_args())
    decode_bench.validate_quantize_before_device(
        _args(single_quantization="mm8")
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"device": "cpu"}, "CUDA target device"),
        ({"single_quantization": None}, "single-quantization"),
        ({"single_quantization": "torchao_w4"}, "single-quantization"),
        ({"policy": "speed"}, "policy memory"),
        ({"paired_baseline": True}, "in-process fp16 baseline"),
        ({"allow_missing_baseline": False}, "allow-missing-baseline"),
    ),
)
def test_quantize_before_device_rejects_ambiguous_modes(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        decode_bench.validate_quantize_before_device(_args(**overrides))


def test_load_model_keeps_dense_weights_on_cpu_before_packing(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class DummyConfig:
        fuse_norm = False
        attn_mode = "fused_recurrent"

    class DummyModel:
        config = DummyConfig()
        model = Namespace(layers=[])

        def eval(self):
            return self

    class DummyLoader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            captured["path"] = path
            captured.update(kwargs)
            return DummyModel()

    monkeypatch.setattr(decode_bench, "AutoModelForCausalLM", DummyLoader)
    args = Namespace(
        hf_dir="model",
        fast_token_backend="native_graph",
        fast_cache="auto",
        fuse_norm="auto",
        attn_mode="fused_recurrent",
        device="cuda:0",
    )

    decode_bench.load_model(args, torch.float16, load_on_cpu=True)

    assert captured["path"] == "model"
    assert captured["device_map"] is None
    assert captured["low_cpu_mem_usage"] is True
    assert captured["torch_dtype"] is torch.float16

    captured.clear()
    decode_bench.load_model(args, torch.float16, load_on_cpu=False)
    assert captured["device_map"] == "cuda:0"
    assert "low_cpu_mem_usage" not in captured


def test_5070_memory_smoke_keeps_unmeasured_gates_null() -> None:
    path = (
        ROOT
        / "bench"
        / "5070_native_memory_loading_20260716"
        / "results.jsonl"
    )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert {row["quantization"] for row in rows} == {"mm8", "mm4"}
    for row in rows:
        assert row["status"] == "pass"
        assert row["quantize_before_device"] is True
        assert row["native_mm_policy"] == "memory"
        assert row["replaced_modules"] == 49
        assert row["footprint_ratio_vs_fp16"] < 1.0
        assert row["decode_speed_ratio_vs_fp16"] is None
        assert row["prompt_logits_cos_vs_fp16"] is None
        assert row["final_logits_cos_vs_fp16"] is None
        assert row["same_next_token_as_fp16"] is None
