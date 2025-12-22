from __future__ import annotations

import pytest

from bench.bench_native_prefill_scan import native_jit_packs


@pytest.mark.parametrize(
    "name",
    ["_native_jit_packs", "_native_graph_packs", "_rwkv7_native_jit_packs"],
)
def test_native_jit_packs_supports_all_model_interfaces(name: str) -> None:
    sentinel = object()
    model = type("Model", (), {name: lambda self: sentinel})()
    assert native_jit_packs(model) is sentinel


def test_native_jit_packs_prefers_native_model_interface() -> None:
    model = type(
        "Model",
        (),
        {
            "_native_jit_packs": lambda self: "native",
            "_rwkv7_native_jit_packs": lambda self: "legacy",
        },
    )()
    assert native_jit_packs(model) == "native"


def test_native_jit_packs_rejects_unknown_model() -> None:
    with pytest.raises(AttributeError, match="projection packs"):
        native_jit_packs(object())
