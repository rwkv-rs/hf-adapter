from __future__ import annotations

import sys
import types

import pytest
import torch

from rwkv7_hf import ops_rwkv7
from rwkv7_hf.ops_rwkv7 import (
    get_last_model_route,
    get_last_recurrent_route,
    maybe_model_forward,
    rwkv7_recurrent,
    rwkv7_recurrent_reference,
)


def recurrent_inputs(*, requires_grad: bool = False):
    torch.manual_seed(7)
    shape = (2, 3, 2, 4)
    values = [
        torch.randn(shape, dtype=torch.float64, requires_grad=requires_grad)
        for _ in range(6)
    ]
    state = torch.randn(
        2, 2, 4, 4, dtype=torch.float64, requires_grad=requires_grad
    )
    mask = torch.tensor([[True, True, True], [False, True, True]])
    return (*values, state, mask)


@pytest.fixture(autouse=True)
def reset_optional_kernel(monkeypatch):
    monkeypatch.delenv("RWKV7_BACKEND", raising=False)
    monkeypatch.delitem(sys.modules, "rwkv7_kernels", raising=False)
    ops_rwkv7._reset_kernel_discovery_for_tests()
    yield
    ops_rwkv7._reset_kernel_discovery_for_tests()


def install_fake_kernel(
    monkeypatch,
    *,
    api_version: int = 2,
    supported: bool = True,
    probe_error: Exception | None = None,
    run_error: Exception | None = None,
):
    calls = {"probe": 0, "run": 0}
    module = types.ModuleType("rwkv7_kernels")
    module.RWKV7_KERNEL_API_VERSION = api_version

    def probe(*_args):
        calls["probe"] += 1
        if probe_error is not None:
            raise probe_error
        return {
            "supported": supported,
            "implementation": "fake-recurrent-v1",
            "reason": "supported by fake" if supported else "unsupported by fake",
        }

    def run(*args):
        calls["run"] += 1
        if run_error is not None:
            raise run_error
        return rwkv7_recurrent_reference(*args)

    module.probe_recurrent_v1 = probe
    module.recurrent_v1 = run
    monkeypatch.setitem(sys.modules, "rwkv7_kernels", module)
    ops_rwkv7._reset_kernel_discovery_for_tests()
    return calls


def assert_reference_equal(actual, expected):
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_auto_without_kernel_package_uses_reference(monkeypatch):
    real_import = ops_rwkv7.importlib.import_module

    def missing(name, *args, **kwargs):
        if name == "rwkv7_kernels":
            raise ModuleNotFoundError("test package is absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(ops_rwkv7.importlib, "import_module", missing)
    inputs = recurrent_inputs()
    assert_reference_equal(
        rwkv7_recurrent(*inputs), rwkv7_recurrent_reference(*inputs)
    )
    route = get_last_recurrent_route()
    assert route is not None
    assert route["requested"] == "auto"
    assert route["selected"] == "reference"


def test_forced_optimized_without_package_fails_clearly(monkeypatch):
    monkeypatch.setattr(
        ops_rwkv7.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ModuleNotFoundError("absent")),
    )
    with pytest.raises(RuntimeError, match="optimized RWKV7 backend is unavailable"):
        rwkv7_recurrent(*recurrent_inputs(), backend="optimized")


def test_supported_kernel_is_selected_and_records_actual_route(monkeypatch):
    calls = install_fake_kernel(monkeypatch)
    inputs = recurrent_inputs()
    assert_reference_equal(
        rwkv7_recurrent(*inputs), rwkv7_recurrent_reference(*inputs)
    )
    assert calls == {"probe": 1, "run": 1}
    assert get_last_recurrent_route() == {
        "requested": "auto",
        "selected": "optimized",
        "implementation": "fake-recurrent-v1",
        "reason": "supported by fake",
    }


def test_auto_falls_back_on_unsupported_and_optimized_surfaces(monkeypatch):
    calls = install_fake_kernel(monkeypatch, supported=False)
    inputs = recurrent_inputs()
    assert_reference_equal(
        rwkv7_recurrent(*inputs), rwkv7_recurrent_reference(*inputs)
    )
    assert calls == {"probe": 1, "run": 0}
    assert get_last_recurrent_route()["reason"] == "unsupported by fake"

    with pytest.raises(RuntimeError, match="unsupported by fake"):
        rwkv7_recurrent(*inputs, backend="optimized")


@pytest.mark.parametrize("failure_stage", ["probe", "run"])
def test_broken_kernel_is_contained_in_auto_and_surfaced_in_optimized(
    monkeypatch, failure_stage
):
    error = RuntimeError(f"broken {failure_stage}")
    kwargs = {f"{failure_stage}_error": error}
    install_fake_kernel(monkeypatch, **kwargs)
    inputs = recurrent_inputs()
    assert_reference_equal(
        rwkv7_recurrent(*inputs), rwkv7_recurrent_reference(*inputs)
    )
    assert "optional kernel failure" in get_last_recurrent_route()["reason"]

    with pytest.raises(RuntimeError, match=f"broken {failure_stage}"):
        rwkv7_recurrent(*inputs, backend="optimized")


def test_api_version_mismatch_falls_back_or_fails_by_mode(monkeypatch):
    install_fake_kernel(monkeypatch, api_version=999)
    inputs = recurrent_inputs()
    assert_reference_equal(
        rwkv7_recurrent(*inputs), rwkv7_recurrent_reference(*inputs)
    )
    assert "kernel API mismatch" in get_last_recurrent_route()["reason"]

    ops_rwkv7._reset_kernel_discovery_for_tests()
    with pytest.raises(RuntimeError, match="kernel API mismatch"):
        rwkv7_recurrent(*inputs, backend="optimized")


def test_training_semantics_never_enter_inference_only_kernel(monkeypatch):
    calls = install_fake_kernel(monkeypatch)
    inputs = recurrent_inputs(requires_grad=True)
    assert_reference_equal(
        rwkv7_recurrent(*inputs, training=True),
        rwkv7_recurrent_reference(*inputs),
    )
    assert calls == {"probe": 0, "run": 0}
    assert get_last_recurrent_route()["selected"] == "reference"

    with pytest.raises(RuntimeError, match="inference-only"):
        rwkv7_recurrent(*inputs, training=True, backend="optimized")


def test_invalid_backend_mode_is_rejected():
    with pytest.raises(ValueError, match="auto, reference, optimized"):
        rwkv7_recurrent(*recurrent_inputs(), backend="fastest")


def install_fake_model_kernel(
    monkeypatch,
    *,
    supported: bool = True,
    malformed: bool = False,
):
    module = types.ModuleType("rwkv7_kernels")
    module.RWKV7_KERNEL_API_VERSION = 2
    module.probe_model_forward_v1 = lambda _owner, _request: {
        "supported": supported,
        "implementation": "fake-model-v1",
        "reason": "fake model route" if supported else "fake model unsupported",
        "phase": "decode",
    }

    def run(_owner, request):
        if malformed:
            return {"output_kind": request["model_kind"]}
        return {
            "output_kind": request["model_kind"],
            "last_hidden_state": torch.ones(1, 1, 4),
        }

    module.model_forward_v1 = run
    monkeypatch.setitem(sys.modules, "rwkv7_kernels", module)
    ops_rwkv7._reset_kernel_discovery_for_tests()


def model_request():
    return {"model_kind": "base", "training": False, "use_cache": True}


def test_model_protocol_selects_supported_result_and_records_route(monkeypatch):
    install_fake_model_kernel(monkeypatch)
    result = maybe_model_forward(object(), model_request())
    assert result is not None
    assert tuple(result["last_hidden_state"].shape) == (1, 1, 4)
    assert get_last_model_route() == {
        "requested": "auto",
        "selected": "optimized",
        "implementation": "fake-model-v1",
        "reason": "fake model route",
        "phase": "decode",
    }


def test_model_protocol_auto_falls_back_but_optimized_is_strict(monkeypatch):
    install_fake_model_kernel(monkeypatch, supported=False)
    assert maybe_model_forward(object(), model_request()) is None
    assert get_last_model_route()["selected"] == "reference"
    with pytest.raises(RuntimeError, match="fake model unsupported"):
        maybe_model_forward(object(), model_request(), backend="optimized")

    install_fake_model_kernel(monkeypatch, malformed=True)
    assert maybe_model_forward(object(), model_request()) is None
    assert "kernel model result is missing" in get_last_model_route()["reason"]
