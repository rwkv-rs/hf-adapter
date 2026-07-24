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
        input_shapes=None,
    ) -> None:
        self.key = key
        self.count = count
        self.self_cpu_time_total = cpu
        self.cpu_time_total = cpu + 1.0
        self.self_device_time_total = device
        self.device_time_total = device + 1.0
        self.cpu_memory_usage = 11
        self.device_memory_usage = 22
        self.input_shapes = input_shapes


class FakeProfiler:
    def __init__(self, events: list[FakeEvent]) -> None:
        self.events = events
        self.steps = 0
        self.kwargs = None
        self.key_average_kwargs = None

    def factory(self, **kwargs):
        self.kwargs = kwargs
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def step(self) -> None:
        self.steps += 1

    def key_averages(self, **kwargs) -> list[FakeEvent]:
        self.key_average_kwargs = kwargs
        return self.events


def optimized_route(*, fast_domain: bool) -> dict:
    return {
        "model": {
            "selected": "reference",
            "implementation": "torch-reference-model-v1",
        },
        "recurrent": {
            "selected": "optimized",
            "implementation": (
                "native-nvidia-rwkv7-factorized-recurrent-training-v1"
                if fast_domain
                else "torch-cuda-rwkv7-batched-matrix-recurrent-training-v1"
            ),
        },
        "linear": {
            "selected": "optimized" if fast_domain else "reference",
            "implementation": (
                "torch-cuda-rwkv7-flattened-linear-training-v1"
                if fast_domain
                else "torch-reference-linear-v1"
            ),
        },
        "mix6": {
            "selected": "optimized",
            "implementation": "native-nvidia-rwkv7-mix6-training-v1",
        },
        "program": {
            "selected": "optimized" if fast_domain else "reference",
            "implementation": "native-nvidia-rwkv7-adaptive-training-program-v1",
        },
    }


def test_profile_route_oracle_matches_certified_adaptive_domain():
    assert hotspots.expected_route(
        "optimized", optimized_route(fast_domain=True), batch=4, tokens=128
    )
    assert hotspots.expected_route(
        "optimized", optimized_route(fast_domain=False), batch=1, tokens=128
    )
    assert not hotspots.expected_route(
        "optimized", optimized_route(fast_domain=True), batch=1, tokens=128
    )


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
        "model": {
            "selected": "reference",
            "implementation": "torch-reference-model-v1",
        },
        "recurrent": {
            "selected": "optimized",
            "implementation": "torch-cuda-rwkv7-batched-matrix-recurrent-training-v1",
        },
        "linear": {
            "selected": "reference",
            "implementation": "torch-reference-linear-v1",
        },
        "mix6": {
            "selected": "optimized",
            "implementation": "native-nvidia-rwkv7-mix6-training-v1",
        },
        "program": {
            "selected": "reference",
            "implementation": "native-nvidia-rwkv7-adaptive-training-program-v1",
        },
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
    assert profiler.kwargs["record_shapes"] is False
    assert profiler.kwargs["profile_memory"] is False
    assert profiler.key_average_kwargs == {"group_by_input_shape": False}
    assert row["loss_mode"] == "model-output-loss"
    assert row["shape"] == {"batch": 2, "tokens": 4}
    assert row["route"] == route
    assert row["route_passed"]
    assert row["host_synchronization_passed"]
    assert row["profiler_settings"] == {
        "record_shapes": False,
        "group_by_input_shape": False,
        "profile_memory": False,
        "profile_memory_events": False,
        "with_stack": False,
    }
    assert row["process_peak_rss_bytes"] > 0
    assert row["loss"] == {"samples": [4.0, 4.0], "finite": True, "last": 4.0}
    assert row["hotspots"]["selected_operators"]["aten::mm"]["count"] == 4
    assert row["hotspots"]["selected_operators"]["aten::cat"]["count"] == 0
    assert row["hotspots"]["recurrent"]["forward"]["aggregate"]["count"] == 2
    assert row["hotspots"]["recurrent"]["backward"]["aggregate"]["count"] == 2
    assert row["profiled_wall_time_includes_profiler_overhead"] is True
    assert "wall_time_ms" not in row
    assert row["hotspots"]["top_self_device_time"][0]["name"] == "aten::mm"


def test_event_summary_has_stable_operator_and_recurrent_schema():
    report = hotspots.summarize_events(
        [
            FakeEvent("aten::addmm", count=3, cpu=2.0, device=4.0),
            FakeEvent("aten::copy_", count=5, cpu=1.0, device=6.0),
            FakeEvent("ChunkDPLRFunction", count=7, device=8.0),
            FakeEvent(
                "autograd::engine::evaluate_function: ChunkDPLRBackward", count=7
            ),
            FakeEvent("rwkv7_tmix_mix6_bf16_v5::backward", count=2, device=9.0),
            FakeEvent("aten::cross_entropy_loss", count=1, device=3.0),
            FakeEvent("aten::native_group_norm", count=4, device=2.0),
            FakeEvent("aten::zero_", count=13, device=1.0),
            FakeEvent("aten::item", count=11, cpu=0.5),
            FakeEvent("aten::_local_scalar_dense", count=9, cpu=0.5),
        ]
    )

    assert set(report["selected_operators"]) == set(hotspots.SELECTED_OPERATORS)
    assert report["selected_operators"]["aten::addmm"]["device_time_us"] == 5.0
    assert report["selected_operators"]["aten::copy_"]["count"] == 5
    assert report["recurrent"]["forward"]["aggregate"]["count"] == 7
    assert report["recurrent"]["backward"]["aggregate"]["count"] == 7
    assert report["categories"]["mix6"]["aggregate"]["count"] == 2
    assert report["categories"]["causal_loss"]["aggregate"]["count"] == 1
    assert report["categories"]["normalization"]["aggregate"]["count"] == 4
    assert report["categories"]["allocation_or_zeroing"]["aggregate"]["count"] == 13
    assert report["categories"]["host_synchronization"]["aggregate"]["count"] == 20
    assert not report["host_synchronization_gate"]["passed"]
    assert report["host_synchronization_gate"]["exact_operators"] == list(
        hotspots.EXACT_HOST_SYNC_OPERATORS
    )
    assert report["top_launch_count"][0]["name"] == "aten::zero_"
    assert report["total_operator_calls"] == 62


def test_event_summary_aggregates_duplicate_exact_names_and_keeps_shape_groups():
    report = hotspots.summarize_events(
        [
            FakeEvent(
                "aten::mm",
                count=3,
                cpu=2.0,
                device=4.0,
                input_shapes=[[4, 8], [8, 16]],
            ),
            FakeEvent(
                "aten::mm",
                count=5,
                cpu=7.0,
                device=11.0,
                input_shapes=[[32, 8], [8, 16]],
            ),
            # These names contain sync-like terms but are deliberately not
            # part of the exact scalar-host-sync gate.
            FakeEvent("cudaMemcpy", count=17, cpu=3.0),
        ],
        include_input_shapes=True,
    )

    mm = report["selected_operators"]["aten::mm"]
    assert mm["count"] == 8
    assert mm["self_cpu_time_us"] == 9.0
    assert mm["self_device_time_us"] == 15.0
    assert [group["input_shapes"] for group in mm["input_shape_groups"]] == [
        [[4, 8], [8, 16]],
        [[32, 8], [8, 16]],
    ]
    assert report["event_count"] == 2
    assert report["key_average_row_count"] == 3
    assert report["total_operator_calls"] == 25
    assert report["host_synchronization_gate"]["passed"]


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
        group_by_input_shape=False,
        profile_memory_events=False,
        dtype="bf16",
        seed=42,
        code_sha="abc123",
        hf_wheel=None,
        kernel_wheel=wheel,
    )
    case = {
        "route_passed": True,
        "loss": {"finite": True},
        "host_synchronization_passed": True,
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
    assert loaded["gates"] == {
        "routes": True,
        "finite_loss": True,
        "no_profiled_scalar_host_sync": True,
    }
    assert loaded["code_sha"] == "abc123"
    assert loaded["settings"]["loss_mode"] == "model-output-loss"
    assert loaded["settings"]["record_shapes"] is False
    assert loaded["settings"]["profile_memory"] is False
    assert loaded["settings"]["profile_memory_events"] is False
    assert loaded["cases"]["optimized-b4-t128"]["shape"] == {
        "batch": 4,
        "tokens": 128,
    }
    assert loaded["wheels"]["rwkv7_kernels"]["path"] == str(wheel)
    assert len(loaded["wheels"]["rwkv7_kernels"]["sha256"]) == 64
    assert loaded["environment"] == {"gpu": "mock"}

    case["host_synchronization_passed"] = False
    failed = hotspots.build_report(
        args,
        cases={"optimized-b4-t128": case},
        fla=None,
    )
    assert failed["status"] == "failed"
    assert failed["gates"]["routes"]
    assert failed["gates"]["finite_loss"]
    assert not failed["gates"]["no_profiled_scalar_host_sync"]


def test_arguments_validate_fla_and_clean_leaf_dtype(tmp_path):
    base = {
        "lane": ["fla"],
        "batch": [1],
        "tokens": [128],
        "warmup": 0,
        "active": 1,
        "dtype": "bf16",
        "fla_source": None,
    }
    with pytest.raises(ValueError, match="fla-source"):
        hotspots.validate_arguments(SimpleNamespace(**base))

    base.update(lane=["optimized"], tokens=[17], fla_source=tmp_path)
    hotspots.validate_arguments(SimpleNamespace(**base))

    base.update(dtype="fp16")
    with pytest.raises(ValueError, match="requires --dtype bf16"):
        hotspots.validate_arguments(SimpleNamespace(**base))


def test_arguments_default_to_one_low_memory_active_step(tmp_path):
    args = hotspots.arguments(
        ["--model", str(tmp_path), "--output", str(tmp_path / "out.json")]
    )

    assert args.active == 1
    assert args.profile_memory_events is False
    assert args.group_by_input_shape is False


def test_process_peak_rss_prefers_linux_vmhwm(monkeypatch):
    monkeypatch.setattr(
        hotspots.Path,
        "read_text",
        lambda _self: "Name:\tpython\nVmHWM:\t123 kB\n",
    )
    assert hotspots._process_peak_rss_bytes() == 123 * 1024


@pytest.mark.parametrize(
    ("platform", "expected"),
    (("darwin", 123), ("linux", 123 * 1024)),
)
def test_process_peak_rss_fallback_uses_platform_units(monkeypatch, platform, expected):
    def unavailable(_self):
        raise OSError("no procfs")

    monkeypatch.setattr(hotspots.Path, "read_text", unavailable)
    monkeypatch.setattr(
        hotspots.resource,
        "getrusage",
        lambda _who: SimpleNamespace(ru_maxrss=123),
    )
    monkeypatch.setattr(hotspots.sys, "platform", platform)

    assert hotspots._process_peak_rss_bytes() == expected
