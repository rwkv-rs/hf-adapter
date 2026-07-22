from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


EVALUATION = Path(__file__).resolve().parents[1] / "evaluation"
sys.path.insert(0, str(EVALUATION))

import profile_training_hotspots as hotspots  # noqa: E402


class FakeEvent:
    def __init__(
        self,
        key: str,
        *,
        count: int,
        cpu: float = 0.0,
        device: float = 0.0,
    ) -> None:
        self.key = key
        self.count = count
        self.self_cpu_time_total = cpu
        self.cpu_time_total = cpu + 1.0
        self.self_device_time_total = device
        self.device_time_total = device + 1.0
        self.cpu_memory_usage = 11
        self.device_memory_usage = 22


class FakeProfiler:
    def __init__(self, events: list[FakeEvent]) -> None:
        self.events = events
        self.steps = 0
        self.kwargs = None

    def factory(self, **kwargs):
        self.kwargs = kwargs
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def step(self) -> None:
        self.steps += 1

    def key_averages(self) -> list[FakeEvent]:
        return self.events


def test_profile_case_uses_only_model_loss_and_records_route_and_shape():
    backward_calls = []

    class Output:
        def __init__(self, loss):
            self._loss = loss
            self.loss_reads = 0

        @property
        def loss(self):
            self.loss_reads += 1
            assert self.loss_reads == 1
            return self._loss

    class Model:
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.tensor(2.0))
            self.calls = []

        def zero_grad(self, *, set_to_none):
            assert set_to_none
            self.weight.grad = None

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            loss = self.weight.square()
            loss.register_hook(lambda gradient: backward_calls.append(gradient))
            return Output(loss)

    profiler = FakeProfiler(
        [
            FakeEvent("aten::mm", count=4, cpu=7.0, device=9.0),
            FakeEvent("rwkv7_clampw_v3::forward", count=2, device=3.0),
            FakeEvent("rwkv7_clampw_v3::backward", count=2, device=5.0),
        ]
    )
    route = {
        "selected": "optimized",
        "implementation": "native-nvidia-train-temp-autograd-v2",
    }
    ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    labels = ids.clone()
    model = Model()

    row = hotspots.profile_training_case(
        model,
        ids,
        labels,
        lane="optimized",
        warmup=1,
        active=2,
        route_getter=lambda lane: route,
        profiler_factory=profiler.factory,
    )

    assert len(model.calls) == 3
    assert len(backward_calls) == 3
    assert all(call["labels"] is labels for call in model.calls)
    assert all(call["use_cache"] is False for call in model.calls)
    assert all(call["logits_to_keep"] == 0 for call in model.calls)
    assert profiler.steps == 2
    assert profiler.kwargs["record_shapes"] is True
    assert profiler.kwargs["profile_memory"] is True
    assert row["loss_mode"] == "model-output-loss"
    assert row["shape"] == {"batch": 2, "tokens": 4}
    assert row["route"] == route
    assert row["route_passed"]
    assert row["loss"] == {"samples": [4.0, 4.0], "finite": True, "last": 4.0}
    assert row["hotspots"]["selected_operators"]["aten::mm"]["count"] == 4
    assert row["hotspots"]["selected_operators"]["aten::cat"]["count"] == 0
    assert row["hotspots"]["recurrent"]["forward"]["aggregate"]["count"] == 2
    assert row["hotspots"]["recurrent"]["backward"]["aggregate"]["count"] == 2


def test_event_summary_has_stable_operator_and_recurrent_schema():
    report = hotspots.summarize_events(
        [
            FakeEvent("aten::addmm", count=3, cpu=2.0, device=4.0),
            FakeEvent("aten::copy_", count=5, cpu=1.0, device=6.0),
            FakeEvent("ChunkDPLRFunction", count=7, device=8.0),
            FakeEvent("autograd::engine::evaluate_function: ChunkDPLRBackward", count=7),
        ]
    )

    assert set(report["selected_operators"]) == set(hotspots.SELECTED_OPERATORS)
    assert report["selected_operators"]["aten::addmm"]["device_time_us"] == 5.0
    assert report["selected_operators"]["aten::copy_"]["count"] == 5
    assert report["recurrent"]["forward"]["aggregate"]["count"] == 7
    assert report["recurrent"]["backward"]["aggregate"]["count"] == 7


def test_build_report_writes_json_with_provenance(monkeypatch, tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}\n")
    wheel = tmp_path / "rwkv7_kernels.whl"
    wheel.write_bytes(b"wheel")
    args = SimpleNamespace(
        model=model_dir,
        lane=["optimized"],
        batch=[4],
        tokens=[128],
        warmup=1,
        active=2,
        dtype="bf16",
        seed=42,
        code_sha="abc123",
        hf_wheel=None,
        kernel_wheel=wheel,
    )
    case = {
        "route_passed": True,
        "loss": {"finite": True},
        "shape": {"batch": 4, "tokens": 128},
    }
    monkeypatch.setattr(hotspots, "environment", lambda: {"gpu": "mock"})

    report = hotspots.build_report(
        args,
        cases={"optimized-b4-t128": case},
        fla=None,
    )
    output = tmp_path / "profile.json"
    hotspots.write_json(output, report)
    loaded = json.loads(output.read_text())

    assert loaded["schema"] == hotspots.SCHEMA
    assert loaded["status"] == "passed"
    assert loaded["code_sha"] == "abc123"
    assert loaded["settings"]["loss_mode"] == "model-output-loss"
    assert loaded["cases"]["optimized-b4-t128"]["shape"] == {
        "batch": 4,
        "tokens": 128,
    }
    assert loaded["wheels"]["rwkv7_kernels"]["path"] == str(wheel)
    assert len(loaded["wheels"]["rwkv7_kernels"]["sha256"]) == 64
    assert loaded["environment"] == {"gpu": "mock"}


def test_arguments_validate_fla_and_native_shapes(tmp_path):
    base = {
        "lane": ["fla"],
        "batch": [1],
        "tokens": [128],
        "warmup": 0,
        "active": 1,
        "fla_source": None,
    }
    with pytest.raises(ValueError, match="fla-source"):
        hotspots.validate_arguments(SimpleNamespace(**base))

    base.update(lane=["optimized"], tokens=[17], fla_source=tmp_path)
    with pytest.raises(ValueError, match="divisible by 16"):
        hotspots.validate_arguments(SimpleNamespace(**base))
