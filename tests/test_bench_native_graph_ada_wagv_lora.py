from __future__ import annotations

import sys
import types
from types import SimpleNamespace

from bench.bench_native_graph_ada_wagv_lora import model_metadata, wagv_extension_status


def test_model_metadata_identifies_checkpoint_shape() -> None:
    args = SimpleNamespace(hf_dir="../models/rwkv7-g1g-1.5b-hf")
    model = SimpleNamespace(
        config=SimpleNamespace(
            hidden_size=2048,
            intermediate_size=8192,
            num_hidden_layers=24,
            head_dim=64,
            num_heads=32,
        )
    )

    assert model_metadata(args, model) == {
        "model_name": "rwkv7-g1g-1.5b-hf",
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_hidden_layers": 24,
        "head_dim": 64,
        "num_heads": 32,
    }


def test_extension_status_reports_build_and_error(monkeypatch) -> None:
    package = "fake_wagv_bench_package"
    model_type = type("FakeModel", (), {"__module__": package + ".native_model"})
    module = types.ModuleType(package + ".ada_lora")
    module.ada_wagv_lora_available = lambda device, build: device == "cuda" and build
    module.ada_wagv_lora_build_error = lambda device: None
    monkeypatch.setitem(sys.modules, package, types.ModuleType(package))
    monkeypatch.setitem(sys.modules, package + ".ada_lora", module)

    assert wagv_extension_status(model_type(), "cuda") == {
        "wagv_extension_available": True,
        "wagv_extension_error": None,
    }
