from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

import bench.bench_native_quant_mm4_tuning as tuning


def test_t4_default_grids_are_bounded_cross_products() -> None:
    gemv = tuning.gemv_launch_grid(
        tuning.DEFAULT_GEMV_BLOCK_PAIRS,
        tuning.DEFAULT_GEMV_BLOCK_N,
    )
    dot = tuning.dot_launch_grid(
        tuning.DEFAULT_DOT_BLOCK_B,
        tuning.DEFAULT_DOT_BLOCK_PAIRS,
        tuning.DEFAULT_DOT_BLOCK_N,
        tuning.DEFAULT_DOT_WARPS,
    )
    assert len(gemv) == 9
    assert len(dot) == 12
    assert gemv[0] == tuning.GemvLaunch(block_pairs=32, block_n=32)
    assert dot[-1] == tuning.DotLaunch(block_b=16, block_pairs=128, block_n=64, num_warps=4)


def test_launch_grids_reject_values_not_accepted_by_native_apis() -> None:
    with pytest.raises(ValueError, match="allowed values"):
        tuning.gemv_launch_grid([8], [32])
    with pytest.raises(ValueError, match="allowed values"):
        tuning.dot_launch_grid([8], [32], [32], [4])
    with pytest.raises(ValueError, match="positive"):
        tuning.dot_launch_grid([16], [32], [32], [0])


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.left = torch.nn.Module()
        self.left.proj = torch.nn.Linear(4, 3, bias=False)
        self.right = torch.nn.Module()
        self.right.proj = torch.nn.Linear(4, 2, bias=False)
        self.lm_head = torch.nn.Linear(4, 5, bias=False)


def test_resolve_linear_modules_supports_exact_suffix_and_glob() -> None:
    model = _TinyModel()
    assert [name for name, _ in tuning.resolve_linear_modules(model, ["lm_head"])] == ["lm_head"]
    assert [name for name, _ in tuning.resolve_linear_modules(model, ["left.proj"])] == ["left.proj"]
    assert [name for name, _ in tuning.resolve_linear_modules(model, ["*.proj"])] == [
        "left.proj",
        "right.proj",
    ]
    with pytest.raises(ValueError, match="ambiguous"):
        tuning.resolve_linear_modules(model, ["proj"])


def test_correctness_metrics_are_cpu_safe_and_row_reduced() -> None:
    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    actual = expected + torch.tensor([[0.0, 0.25], [-0.5, 0.0]])
    metrics = tuning.correctness_metrics(actual, expected)
    assert metrics["finite"] is True
    assert metrics["max_abs"] == pytest.approx(0.5)
    assert metrics["mean_abs"] == pytest.approx(0.1875)
    assert 0.99 < metrics["min_cosine"] <= 1.0

    nonfinite = tuning.correctness_metrics(torch.tensor([float("nan")]), torch.ones(1))
    assert nonfinite == {
        "finite": False,
        "max_abs": None,
        "mean_abs": None,
        "min_cosine": None,
    }


def test_append_jsonl_writes_independent_records(tmp_path) -> None:
    output = tmp_path / "nested" / "t4.jsonl"
    tuning.append_jsonl(output, [{"candidate": 1}, {"candidate": 2, "policy_promoted": False}])
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert rows == [{"candidate": 1}, {"candidate": 2, "policy_promoted": False}]


def test_cli_defaults_cover_t4_batches_and_disable_promotion() -> None:
    args = tuning.build_parser().parse_args(["--hf-dir", "/tmp/model"])
    assert tuple(args.batches) == (1, 2, 4, 8)
    assert args.modules == ["lm_head"]
    assert args.quantize_on == "cpu"
    assert args.strict is False
    assert not any(action.dest.startswith("promote") for action in tuning.build_parser()._actions)


def test_main_loads_the_hf_model_once(monkeypatch) -> None:
    calls = {"loads": 0, "benchmarks": 0}
    sentinel = object()

    monkeypatch.setattr(tuning, "_validate_runtime_args", lambda args: torch.device("cpu"))

    def fake_load(hf_dir, *, device):
        calls["loads"] += 1
        assert hf_dir == "model-id"
        assert device.type == "cpu"
        return sentinel

    def fake_benchmark(model, args, *, device):
        calls["benchmarks"] += 1
        assert model is sentinel
        return [{"status": "pass", "policy_promotion": "disabled", "policy_promoted": False}]

    monkeypatch.setattr(tuning, "load_hf_model", fake_load)
    monkeypatch.setattr(tuning, "benchmark_model", fake_benchmark)
    assert tuning.main(["--hf-dir", "model-id", "--results", ""]) == 0
    assert calls == {"loads": 1, "benchmarks": 1}
