from __future__ import annotations

import sys
import types

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from rwkv7_hf import ops_rwkv7
from rwkv7_hf.ops_rwkv7 import (
    get_last_linear_route,
    get_last_mix6_route,
    get_last_model_route,
    get_last_recurrent_route,
    maybe_linear_training,
    maybe_mix6_training,
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
    state = torch.randn(2, 2, 4, 4, dtype=torch.float64, requires_grad=requires_grad)
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
    api_version: int = 3,
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
    assert_reference_equal(rwkv7_recurrent(*inputs), rwkv7_recurrent_reference(*inputs))
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
    assert_reference_equal(rwkv7_recurrent(*inputs), rwkv7_recurrent_reference(*inputs))
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
    assert_reference_equal(rwkv7_recurrent(*inputs), rwkv7_recurrent_reference(*inputs))
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
    assert_reference_equal(rwkv7_recurrent(*inputs), rwkv7_recurrent_reference(*inputs))
    assert "optional kernel failure" in get_last_recurrent_route()["reason"]

    with pytest.raises(RuntimeError, match=f"broken {failure_stage}"):
        rwkv7_recurrent(*inputs, backend="optimized")


def test_api_version_mismatch_falls_back_or_fails_by_mode(monkeypatch):
    install_fake_kernel(monkeypatch, api_version=999)
    inputs = recurrent_inputs()
    assert_reference_equal(rwkv7_recurrent(*inputs), rwkv7_recurrent_reference(*inputs))
    assert "kernel API mismatch" in get_last_recurrent_route()["reason"]

    ops_rwkv7._reset_kernel_discovery_for_tests()
    with pytest.raises(RuntimeError, match="kernel API mismatch"):
        rwkv7_recurrent(*inputs, backend="optimized")


def test_training_falls_back_when_optional_package_has_no_training_protocol(
    monkeypatch,
):
    calls = install_fake_kernel(monkeypatch)
    inputs = recurrent_inputs(requires_grad=True)
    assert_reference_equal(
        rwkv7_recurrent(*inputs, training=True),
        rwkv7_recurrent_reference(*inputs),
    )
    assert calls == {"probe": 0, "run": 0}
    assert get_last_recurrent_route()["selected"] == "reference"

    assert "recurrent-training-v1" in get_last_recurrent_route()["reason"]

    with pytest.raises(RuntimeError, match="recurrent-training-v1"):
        rwkv7_recurrent(*inputs, training=True, backend="optimized")


def test_training_uses_separate_leaf_autograd_protocol(monkeypatch):
    module = types.ModuleType("rwkv7_kernels")
    module.RWKV7_KERNEL_API_VERSION = 3
    calls = {"probe": 0, "run": 0}

    def probe(*_args):
        calls["probe"] += 1
        return {
            "supported": True,
            "implementation": "fake-cuda-training-v1",
            "reason": "fake training leaf",
        }

    def run(*args):
        calls["run"] += 1
        return rwkv7_recurrent_reference(*args)

    module.probe_recurrent_training_v1 = probe
    module.recurrent_training_v1 = run
    monkeypatch.setitem(sys.modules, "rwkv7_kernels", module)
    ops_rwkv7._reset_kernel_discovery_for_tests()

    inputs = recurrent_inputs(requires_grad=True)
    output, state = rwkv7_recurrent(*inputs, training=True)
    loss = output.square().mean() + state.square().mean()
    loss.backward()

    assert calls == {"probe": 1, "run": 1}
    assert all(value.grad is not None for value in inputs[:-1])
    assert get_last_recurrent_route() == {
        "requested": "auto",
        "selected": "optimized",
        "implementation": "fake-cuda-training-v1",
        "reason": "fake training leaf",
    }


def test_training_recurrent_protocol_receives_model_context_hints(monkeypatch):
    module = types.ModuleType("rwkv7_kernels")
    module.RWKV7_KERNEL_API_VERSION = 3
    received = []

    def probe(*_args, **kwargs):
        received.append(("probe", kwargs))
        return {
            "supported": True,
            "implementation": "fake-cuda-training-v1",
            "reason": "fake training leaf",
        }

    def run(*args, **kwargs):
        received.append(("run", kwargs))
        return rwkv7_recurrent_reference(*args)

    module.probe_recurrent_training_v1 = probe
    module.recurrent_training_v1 = run
    monkeypatch.setitem(sys.modules, "rwkv7_kernels", module)
    ops_rwkv7._reset_kernel_discovery_for_tests()

    inputs = recurrent_inputs(requires_grad=True)
    ops_rwkv7.set_training_batch_context(inputs[-1], training=True, fully_active=False)
    rwkv7_recurrent(*inputs, training=True, initial_state_zero=True)

    expected = {
        "adaptive_fast_program": False,
        "fully_active": False,
        "initial_state_zero": True,
        "token_aligned": False,
    }
    assert received == [("probe", expected), ("run", expected)]


def test_training_recurrent_prefers_atomic_execute_and_is_strict(monkeypatch):
    module = types.ModuleType("rwkv7_kernels")
    module.RWKV7_KERNEL_API_VERSION = 3
    calls = {"execute": 0, "legacy": 0}

    def execute(*args, **kwargs):
        calls["execute"] += 1
        assert kwargs["initial_state_zero"] is True
        return {
            "supported": True,
            "implementation": "fake-atomic-recurrent-training-v1",
            "reason": "atomic recurrent",
            "result": rwkv7_recurrent_reference(*args),
        }

    def legacy(*_args, **_kwargs):
        calls["legacy"] += 1
        raise AssertionError("atomic recurrent must bypass legacy probe/run")

    module.execute_recurrent_training_v1 = execute
    module.probe_recurrent_training_v1 = legacy
    module.recurrent_training_v1 = legacy
    monkeypatch.setitem(sys.modules, "rwkv7_kernels", module)
    ops_rwkv7._reset_kernel_discovery_for_tests()
    inputs = recurrent_inputs(requires_grad=True)

    actual = rwkv7_recurrent(*inputs, training=True, initial_state_zero=True)
    assert_reference_equal(actual, rwkv7_recurrent_reference(*inputs))
    assert calls == {"execute": 1, "legacy": 0}

    def unsupported(*_args, **_kwargs):
        calls["execute"] += 1
        return {
            "supported": False,
            "implementation": "fake-atomic-recurrent-training-v1",
            "reason": "atomic recurrent unsupported",
            "result": None,
        }

    module.execute_recurrent_training_v1 = unsupported
    fallback = rwkv7_recurrent(*inputs, training=True, initial_state_zero=True)
    assert_reference_equal(fallback, rwkv7_recurrent_reference(*inputs))
    with pytest.raises(RuntimeError, match="atomic recurrent unsupported"):
        rwkv7_recurrent(
            *inputs,
            training=True,
            initial_state_zero=True,
            backend="optimized",
        )
    assert calls == {"execute": 3, "legacy": 0}


def test_training_linear_uses_stateless_optional_protocol(monkeypatch):
    module = types.ModuleType("rwkv7_kernels")
    module.RWKV7_KERNEL_API_VERSION = 3
    calls = {"probe": 0, "run": 0}

    def probe(
        value,
        weight,
        bias,
        *,
        adaptive_fast_program,
        fully_active,
        initial_state_zero,
        token_aligned,
    ):
        calls["probe"] += 1
        assert bias is None
        assert fully_active is None
        assert adaptive_fast_program is None
        assert initial_state_zero is None
        assert token_aligned is None
        return {
            "supported": True,
            "implementation": "fake-cuda-linear-training-v1",
            "reason": "fake stateless linear leaf",
        }

    def run(
        value,
        weight,
        bias,
        *,
        adaptive_fast_program,
        fully_active,
        initial_state_zero,
        token_aligned,
    ):
        calls["run"] += 1
        assert fully_active is None
        assert adaptive_fast_program is None
        assert initial_state_zero is None
        assert token_aligned is None
        return torch.nn.functional.linear(value, weight, bias)

    module.probe_linear_training_v1 = probe
    module.linear_training_v1 = run
    monkeypatch.setitem(sys.modules, "rwkv7_kernels", module)
    ops_rwkv7._reset_kernel_discovery_for_tests()

    value = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(5, 4, dtype=torch.float64, requires_grad=True)
    output = maybe_linear_training(value, weight, None, training=True)
    assert output is not None
    output.square().mean().backward()

    assert calls == {"probe": 1, "run": 1}
    assert value.grad is not None and weight.grad is not None
    assert get_last_linear_route() == {
        "requested": "auto",
        "selected": "optimized",
        "implementation": "fake-cuda-linear-training-v1",
        "reason": "fake stateless linear leaf",
    }


def test_training_linear_prefers_atomic_execute_without_legacy_probe(monkeypatch):
    module = types.ModuleType("rwkv7_kernels")
    module.RWKV7_KERNEL_API_VERSION = 3
    calls = {"execute": 0, "legacy_probe": 0, "legacy_run": 0}

    def execute(
        value,
        weight,
        bias,
        *,
        adaptive_fast_program,
        fully_active,
        initial_state_zero,
        token_aligned,
    ):
        calls["execute"] += 1
        assert fully_active is None
        assert adaptive_fast_program is None
        assert initial_state_zero is None
        assert token_aligned is None
        return {
            "supported": True,
            "implementation": "fake-atomic-linear-training-v1",
            "reason": "one call-local validation",
            "output": torch.nn.functional.linear(value, weight, bias),
        }

    def legacy_probe(*_args, **_kwargs):
        calls["legacy_probe"] += 1
        raise AssertionError("atomic protocol must bypass the legacy probe/run pair")

    def legacy_run(*_args, **_kwargs):
        calls["legacy_run"] += 1
        raise AssertionError("atomic protocol must bypass the legacy probe/run pair")

    module.execute_linear_training_v1 = execute
    module.probe_linear_training_v1 = legacy_probe
    module.linear_training_v1 = legacy_run
    monkeypatch.setitem(sys.modules, "rwkv7_kernels", module)
    ops_rwkv7._reset_kernel_discovery_for_tests()

    value = torch.randn(2, 3, 4, requires_grad=True)
    weight = torch.randn(5, 4, requires_grad=True)
    output = maybe_linear_training(value, weight, None, training=True)

    assert output is not None and tuple(output.shape) == (2, 3, 5)
    assert calls == {"execute": 1, "legacy_probe": 0, "legacy_run": 0}


def test_training_linear_preserves_non_reentrant_checkpoint_control_flow(monkeypatch):
    """Fail-closed dispatch must not swallow checkpoint's replay sentinel."""

    module = types.ModuleType("rwkv7_kernels")
    module.RWKV7_KERNEL_API_VERSION = 3
    calls = {"execute": 0}

    def execute(value, weight, bias, **_kwargs):
        calls["execute"] += 1
        return {
            "supported": True,
            "implementation": "fake-atomic-linear-training-v1",
            "reason": "checkpoint replay",
            "output": torch.nn.functional.linear(value, weight, bias),
        }

    module.execute_linear_training_v1 = execute
    monkeypatch.setitem(sys.modules, "rwkv7_kernels", module)
    ops_rwkv7._reset_kernel_discovery_for_tests()

    value = torch.randn(4, 8, requires_grad=True)
    weight = torch.randn(8, 8, requires_grad=True)

    def projection(hidden):
        output = maybe_linear_training(
            hidden,
            weight,
            None,
            training=True,
        )
        assert output is not None
        return output

    output = checkpoint(projection, value, use_reentrant=False)
    output.sum().backward()

    assert calls == {"execute": 2}
    assert value.grad is not None
    assert weight.grad is not None
    assert torch.isfinite(value.grad).all()
    assert torch.isfinite(weight.grad).all()
    assert torch.count_nonzero(value.grad)
    assert torch.count_nonzero(weight.grad)


def test_training_linear_atomic_fallback_and_error_are_fail_closed(monkeypatch):
    module = types.ModuleType("rwkv7_kernels")
    module.RWKV7_KERNEL_API_VERSION = 3
    calls = {"execute": 0}

    def unsupported(*_args, **_kwargs):
        calls["execute"] += 1
        return {
            "supported": False,
            "implementation": "fake-atomic-linear-training-v1",
            "reason": "unsupported atomic request",
            "output": None,
        }

    module.execute_linear_training_v1 = unsupported
    monkeypatch.setitem(sys.modules, "rwkv7_kernels", module)
    ops_rwkv7._reset_kernel_discovery_for_tests()
    value = torch.randn(2, 3, 4, requires_grad=True)
    weight = torch.randn(5, 4, requires_grad=True)

    assert maybe_linear_training(value, weight, None, training=True) is None
    assert calls == {"execute": 1}
    assert get_last_linear_route()["reason"] == "unsupported atomic request"

    def broken(*_args, **_kwargs):
        calls["execute"] += 1
        raise RuntimeError("broken atomic execution")

    module.execute_linear_training_v1 = broken
    assert maybe_linear_training(value, weight, None, training=True) is None
    assert calls == {"execute": 2}
    assert "broken atomic execution" in get_last_linear_route()["reason"]


def test_training_linear_auto_falls_back_and_optimized_is_strict(monkeypatch):
    install_fake_kernel(monkeypatch)
    value = torch.randn(2, 3, 4, requires_grad=True)
    weight = torch.randn(5, 4, requires_grad=True)

    assert maybe_linear_training(value, weight, None, training=True) is None
    assert get_last_linear_route()["selected"] == "reference"
    assert "linear-training-v1" in get_last_linear_route()["reason"]
    with pytest.raises(RuntimeError, match="linear-training-v1"):
        maybe_linear_training(
            value,
            weight,
            None,
            training=True,
            backend="optimized",
        )


def test_training_mix6_uses_stateless_explicit_shift_protocol(monkeypatch):
    module = types.ModuleType("rwkv7_kernels")
    module.RWKV7_KERNEL_API_VERSION = 3
    calls = {"probe": 0, "run": 0}

    def probe(value, shifted, *mixes, fully_active, token_aligned):
        calls["probe"] += 1
        assert value.shape == shifted.shape
        assert len(mixes) == 6
        assert fully_active is None
        assert token_aligned is None
        return {
            "supported": True,
            "implementation": "fake-cuda-mix6-training-v1",
            "reason": "fake explicit-shift Mix6 leaf",
        }

    def run(value, shifted, *mixes, fully_active, token_aligned):
        calls["run"] += 1
        assert fully_active is None
        assert token_aligned is None
        delta = shifted - value
        return tuple(value + delta * mix.view(1, 1, -1) for mix in mixes)

    module.probe_mix6_training_v1 = probe
    module.mix6_training_v1 = run
    monkeypatch.setitem(sys.modules, "rwkv7_kernels", module)
    ops_rwkv7._reset_kernel_discovery_for_tests()

    value = torch.randn(2, 3, 4, requires_grad=True)
    shifted = torch.randn_like(value, requires_grad=True)
    mixes = tuple(torch.randn(4, requires_grad=True) for _ in range(6))
    outputs = maybe_mix6_training(
        value,
        shifted,
        mixes,
        training=True,
    )
    assert outputs is not None
    sum(item.square().mean() for item in outputs).backward()

    assert calls == {"probe": 1, "run": 1}
    assert value.grad is not None and shifted.grad is not None
    assert all(mix.grad is not None for mix in mixes)
    assert get_last_mix6_route() == {
        "requested": "auto",
        "selected": "optimized",
        "implementation": "fake-cuda-mix6-training-v1",
        "reason": "fake explicit-shift Mix6 leaf",
    }


def test_training_mix6_prefers_atomic_execute_and_is_strict(monkeypatch):
    module = types.ModuleType("rwkv7_kernels")
    module.RWKV7_KERNEL_API_VERSION = 3
    calls = {"execute": 0, "legacy": 0}

    def execute(value, shifted, *mixes, **_kwargs):
        calls["execute"] += 1
        delta = shifted - value
        return {
            "supported": True,
            "implementation": "fake-atomic-mix6-training-v1",
            "reason": "atomic Mix6",
            "result": tuple(value + delta * mix.view(1, 1, -1) for mix in mixes),
        }

    def legacy(*_args, **_kwargs):
        calls["legacy"] += 1
        raise AssertionError("atomic Mix6 must bypass legacy probe/run")

    module.execute_mix6_training_v1 = execute
    module.probe_mix6_training_v1 = legacy
    module.mix6_training_v1 = legacy
    monkeypatch.setitem(sys.modules, "rwkv7_kernels", module)
    ops_rwkv7._reset_kernel_discovery_for_tests()
    value = torch.randn(2, 3, 4, requires_grad=True)
    shifted = torch.randn_like(value, requires_grad=True)
    mixes = tuple(torch.randn(4, requires_grad=True) for _ in range(6))

    result = maybe_mix6_training(value, shifted, mixes, training=True)
    assert result is not None and len(result) == 6
    assert calls == {"execute": 1, "legacy": 0}

    def unsupported(*_args, **_kwargs):
        calls["execute"] += 1
        return {
            "supported": False,
            "implementation": "fake-atomic-mix6-training-v1",
            "reason": "atomic Mix6 unsupported",
            "result": None,
        }

    module.execute_mix6_training_v1 = unsupported
    assert maybe_mix6_training(value, shifted, mixes, training=True) is None
    with pytest.raises(RuntimeError, match="atomic Mix6 unsupported"):
        maybe_mix6_training(
            value,
            shifted,
            mixes,
            training=True,
            backend="optimized",
        )
    assert calls == {"execute": 3, "legacy": 0}


def test_certified_training_program_never_silently_falls_back_mix6(monkeypatch):
    module = types.ModuleType("rwkv7_kernels")
    module.RWKV7_KERNEL_API_VERSION = 3
    module.execute_mix6_training_v1 = lambda *_args, **_kwargs: {
        "supported": False,
        "implementation": "fake-atomic-mix6-training-v1",
        "reason": "late certified Mix6 decline",
        "result": None,
    }
    monkeypatch.setitem(sys.modules, "rwkv7_kernels", module)
    ops_rwkv7._reset_kernel_discovery_for_tests()
    context = ops_rwkv7.RWKV7TrainingBatchContext(
        fully_active=True,
        token_aligned=True,
        initial_state_zero=True,
        autograd_leaf_eligible=True,
        adaptive_fast_program=True,
        force_reference_recurrent=False,
        program_implementation="fake-adaptive-training-program-v1",
        program_reason="certified test program",
    )
    value = torch.randn(4, 128, 4, requires_grad=True)
    shifted = torch.randn_like(value, requires_grad=True)
    mixes = tuple(torch.randn(4, requires_grad=True) for _ in range(6))

    with ops_rwkv7.training_batch_context(context):
        with pytest.raises(RuntimeError, match="late certified Mix6 decline"):
            maybe_mix6_training(value, shifted, mixes, training=True)


def test_training_mix6_auto_falls_back_and_optimized_is_strict(monkeypatch):
    install_fake_kernel(monkeypatch)
    value = torch.randn(2, 3, 4, requires_grad=True)
    shifted = torch.randn_like(value, requires_grad=True)
    mixes = tuple(torch.randn(4, requires_grad=True) for _ in range(6))

    assert maybe_mix6_training(value, shifted, mixes, training=True) is None
    assert get_last_mix6_route()["selected"] == "reference"
    assert "mix6-training-v1" in get_last_mix6_route()["reason"]
    with pytest.raises(RuntimeError, match="mix6-training-v1"):
        maybe_mix6_training(
            value,
            shifted,
            mixes,
            training=True,
            backend="optimized",
        )


def test_training_model_keeps_readable_loop_and_selects_both_leafs(
    monkeypatch, tiny_config
):
    from rwkv7_hf.modeling_rwkv7 import RWKV7ForCausalLM

    module = types.ModuleType("rwkv7_kernels")
    module.RWKV7_KERNEL_API_VERSION = 3

    def forbidden_model_boundary(*_args, **_kwargs):
        raise AssertionError("HF training must not enter whole-model dispatch")

    module.probe_model_forward_v1 = forbidden_model_boundary
    module.model_forward_v1 = forbidden_model_boundary
    recurrent_hints = []
    linear_hints = []

    def probe_recurrent(*_args, **kwargs):
        recurrent_hints.append(kwargs)
        return {
            "supported": True,
            "implementation": "fake-cuda-recurrent-training-v1",
            "reason": "fake recurrent leaf",
        }

    module.probe_recurrent_training_v1 = probe_recurrent
    module.recurrent_training_v1 = lambda *args, **_kwargs: rwkv7_recurrent_reference(
        *args
    )

    def probe_linear(*_args, **kwargs):
        linear_hints.append(kwargs)
        return {
            "supported": True,
            "implementation": "fake-cuda-linear-training-v1",
            "reason": "fake linear leaf",
        }

    module.probe_linear_training_v1 = probe_linear
    module.linear_training_v1 = lambda value, weight, bias, **_kwargs: (
        torch.nn.functional.linear(value, weight, bias)
    )
    module.probe_mix6_training_v1 = lambda *_args, **_kwargs: {
        "supported": True,
        "implementation": "fake-cuda-mix6-training-v1",
        "reason": "fake Mix6 leaf",
    }
    module.mix6_training_v1 = lambda value, shifted, *mixes, **_kwargs: tuple(
        value + (shifted - value) * mix.view(1, 1, -1) for mix in mixes
    )
    monkeypatch.setitem(sys.modules, "rwkv7_kernels", module)
    ops_rwkv7._reset_kernel_discovery_for_tests()

    model = RWKV7ForCausalLM(tiny_config).train()
    output = model(
        input_ids=torch.tensor([[1, 2, 3]]),
        labels=torch.tensor([[1, 2, 3]]),
        use_cache=False,
    )
    output.loss.backward()

    assert get_last_model_route()["selected"] == "reference"
    assert get_last_recurrent_route()["implementation"] == (
        "fake-cuda-recurrent-training-v1"
    )
    assert get_last_linear_route()["implementation"] == ("fake-cuda-linear-training-v1")
    assert get_last_mix6_route()["implementation"] == ("fake-cuda-mix6-training-v1")
    assert recurrent_hints
    assert linear_hints
    assert all(
        hints
        == {
            "adaptive_fast_program": False,
            "fully_active": True,
            "initial_state_zero": True,
            "token_aligned": False,
        }
        for hints in recurrent_hints
    )
    assert all(
        hints
        == {
            "adaptive_fast_program": False,
            "fully_active": True,
            "initial_state_zero": True,
            "token_aligned": False,
        }
        for hints in linear_hints
    )


def test_training_cache_provenance_disables_zero_state_leaf(monkeypatch, tiny_config):
    from rwkv7_hf.cache_rwkv7 import RWKV7Cache
    from rwkv7_hf.modeling_rwkv7 import RWKV7ForCausalLM

    module = types.ModuleType("rwkv7_kernels")
    module.RWKV7_KERNEL_API_VERSION = 3
    recurrent_hints = []
    linear_hints = []

    def probe(*_args, **kwargs):
        recurrent_hints.append(kwargs)
        return {
            "supported": False,
            "implementation": "fake-zero-state-training-v1",
            "reason": "capture cache provenance",
        }

    module.probe_recurrent_training_v1 = probe
    module.recurrent_training_v1 = lambda *_args, **_kwargs: None

    def probe_linear(*_args, **kwargs):
        linear_hints.append(kwargs)
        return {
            "supported": False,
            "implementation": "fake-stateful-linear-training-v1",
            "reason": "capture cache provenance",
        }

    module.probe_linear_training_v1 = probe_linear
    module.linear_training_v1 = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "rwkv7_kernels", module)
    ops_rwkv7._reset_kernel_discovery_for_tests()

    batch = 1
    recurrent = torch.ones(
        batch,
        tiny_config.num_heads,
        tiny_config.head_dim,
        tiny_config.head_dim,
    )
    shift = torch.zeros(batch, tiny_config.hidden_size)
    cache = RWKV7Cache(
        [recurrent.clone() for _ in range(tiny_config.num_hidden_layers)],
        [shift.clone() for _ in range(tiny_config.num_hidden_layers)],
        [shift.clone() for _ in range(tiny_config.num_hidden_layers)],
    )
    model = RWKV7ForCausalLM(tiny_config).train()
    model(
        input_ids=torch.tensor([[1, 2, 3]]),
        past_key_values=cache,
        use_cache=False,
    )

    assert recurrent_hints
    assert all(hints["initial_state_zero"] is False for hints in recurrent_hints)
    assert linear_hints
    assert all(hints["initial_state_zero"] is False for hints in linear_hints)


def test_checkpoint_replay_preserves_atomic_recurrent_hints(monkeypatch, tiny_config):
    from rwkv7_hf.modeling_rwkv7 import RWKV7ForCausalLM

    module = types.ModuleType("rwkv7_kernels")
    module.RWKV7_KERNEL_API_VERSION = 3
    recurrent_hints = []
    linear_hints = []
    program_probes = []

    def probe_program(*_args, **kwargs):
        program_probes.append(dict(kwargs))
        return {
            "supported": True,
            "implementation": "fake-coupled-adaptive-training-v1",
            "reason": "certified test program",
        }

    module.probe_training_program_v1 = probe_program

    def execute(*args, **kwargs):
        recurrent_hints.append(dict(kwargs))
        return {
            "supported": True,
            "implementation": "fake-checkpoint-recurrent-training-v1",
            "reason": "checkpoint hint capture",
            "result": rwkv7_recurrent_reference(*args),
        }

    module.execute_recurrent_training_v1 = execute

    def execute_linear(value, weight, bias, **kwargs):
        linear_hints.append(dict(kwargs))
        return {
            "supported": True,
            "implementation": "fake-checkpoint-linear-training-v1",
            "reason": "checkpoint linear hint capture",
            "output": torch.nn.functional.linear(value, weight, bias),
        }

    module.execute_linear_training_v1 = execute_linear

    def execute_mix6(value, shifted, *mixes, **_kwargs):
        delta = shifted - value
        return {
            "supported": True,
            "implementation": "fake-checkpoint-mix6-training-v1",
            "reason": "checkpoint Mix6",
            "result": tuple(value + delta * mix.view(1, 1, -1) for mix in mixes),
        }

    module.execute_mix6_training_v1 = execute_mix6
    monkeypatch.setitem(sys.modules, "rwkv7_kernels", module)
    ops_rwkv7._reset_kernel_discovery_for_tests()
    model = RWKV7ForCausalLM(tiny_config).train()
    model.gradient_checkpointing_enable()
    ids = torch.arange(4 * 128).reshape(4, 128) % tiny_config.vocab_size

    model(input_ids=ids, labels=ids, use_cache=False).loss.backward()

    assert len(recurrent_hints) >= 2 * tiny_config.num_hidden_layers
    assert all(
        hints
        == {
            "adaptive_fast_program": True,
            "fully_active": True,
            "initial_state_zero": True,
            "token_aligned": True,
        }
        for hints in recurrent_hints
    )
    assert linear_hints
    assert all(
        hints
        == {
            "adaptive_fast_program": True,
            "fully_active": True,
            "initial_state_zero": True,
            "token_aligned": True,
        }
        for hints in linear_hints
    )
    assert len(program_probes) == 1
    assert program_probes[0]["autograd_leaf_eligible"] is True


def _install_autograd_sensitive_training_kernel(monkeypatch):
    module = types.ModuleType("rwkv7_kernels")
    module.RWKV7_KERNEL_API_VERSION = 3
    program_probes = []
    recurrent_calls = []

    def probe_program(*_args, **kwargs):
        program_probes.append(dict(kwargs))
        supported = kwargs["autograd_leaf_eligible"] is True
        return {
            "supported": supported,
            "implementation": "fake-coupled-autograd-program-v1",
            "reason": (
                "autograd-bearing program"
                if supported
                else "frozen or reentrant input retains exact training"
            ),
        }

    def execute_linear(value, weight, bias, **kwargs):
        supported = kwargs["adaptive_fast_program"] is True and any(
            tensor.requires_grad
            for tensor in (value, weight, bias)
            if isinstance(tensor, torch.Tensor)
        )
        return {
            "supported": supported,
            "implementation": "fake-autograd-linear-v1",
            "reason": "linear autograd eligibility",
            "output": (
                torch.nn.functional.linear(value, weight, bias) if supported else None
            ),
        }

    def execute_recurrent(*args, **kwargs):
        # Match the exact matrix leaf: eligibility changes between a reentrant
        # checkpoint's no-grad forward and its grad-enabled replay.
        supported = any(tensor.requires_grad for tensor in args[:7])
        recurrent_calls.append(supported)
        return {
            "supported": supported,
            "implementation": "fake-autograd-recurrent-v1",
            "reason": "recurrent autograd eligibility",
            "result": rwkv7_recurrent_reference(*args) if supported else None,
        }

    module.probe_training_program_v1 = probe_program
    module.execute_linear_training_v1 = execute_linear
    module.execute_recurrent_training_v1 = execute_recurrent
    monkeypatch.setitem(sys.modules, "rwkv7_kernels", module)
    ops_rwkv7._reset_kernel_discovery_for_tests()
    return program_probes, recurrent_calls


def test_frozen_base_declines_atomic_program_before_first_linear(
    monkeypatch, tiny_config
):
    from rwkv7_hf.modeling_rwkv7 import RWKV7ForCausalLM

    program_probes, _recurrent_calls = _install_autograd_sensitive_training_kernel(
        monkeypatch
    )
    model = RWKV7ForCausalLM(tiny_config).train()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.lm_head.parameters():
        parameter.requires_grad_(True)
    ids = torch.arange(4 * 128).reshape(4, 128) % tiny_config.vocab_size

    model(input_ids=ids, labels=ids, use_cache=False).loss.backward()

    assert len(program_probes) == 1
    assert program_probes[0]["autograd_leaf_eligible"] is False
    assert ops_rwkv7.get_last_training_program_route()["selected"] == "reference"
    assert model.lm_head.weight.grad is not None


def test_reentrant_checkpoint_declines_atomic_program_before_replay(
    monkeypatch, tiny_config
):
    from rwkv7_hf.modeling_rwkv7 import RWKV7ForCausalLM

    program_probes, recurrent_calls = _install_autograd_sensitive_training_kernel(
        monkeypatch
    )
    model = RWKV7ForCausalLM(tiny_config).train()
    model.gradient_checkpointing_enable({"use_reentrant": True})
    ids = torch.arange(4 * 128).reshape(4, 128) % tiny_config.vocab_size

    model(input_ids=ids, labels=ids, use_cache=False).loss.backward()

    assert len(program_probes) == 1
    assert program_probes[0]["autograd_leaf_eligible"] is False
    program_route = ops_rwkv7.get_last_training_program_route()
    assert program_route["selected"] == "reference"
    assert program_route["facts"]["force_reference_recurrent"] is True
    # The immutable model context bypasses the optional recurrent dispatcher in
    # both phases, so its requires-grad-sensitive matrix route cannot change.
    assert recurrent_calls == []
    assert (
        ops_rwkv7.get_last_recurrent_route()["implementation"] == "torch-reference-v1"
    )


def test_atomic_fast_program_never_mixes_flattened_linear_with_fallback_recurrence(
    monkeypatch, tiny_config
):
    from rwkv7_hf.modeling_rwkv7 import RWKV7ForCausalLM

    module = types.ModuleType("rwkv7_kernels")
    module.RWKV7_KERNEL_API_VERSION = 3
    module.probe_training_program_v1 = lambda *_args, **_kwargs: {
        "supported": True,
        "implementation": "fake-coupled-adaptive-training-v1",
        "reason": "certified test program",
    }
    linear_calls = []

    def execute_linear(value, weight, bias, **kwargs):
        linear_calls.append(dict(kwargs))
        return {
            "supported": True,
            "implementation": "fake-atomic-linear-training-v1",
            "reason": "flattened projection executed",
            "output": torch.nn.functional.linear(value, weight, bias),
        }

    module.execute_linear_training_v1 = execute_linear
    module.execute_mix6_training_v1 = lambda value, shifted, *mixes, **_kwargs: {
        "supported": True,
        "implementation": "fake-atomic-mix6-training-v1",
        "reason": "atomic Mix6 executed",
        "result": tuple(
            value + (shifted - value) * mix.view(1, 1, -1) for mix in mixes
        ),
    }
    module.execute_recurrent_training_v1 = lambda *_args, **_kwargs: {
        "supported": False,
        "implementation": "fake-atomic-recurrent-training-v1",
        "reason": "simulated late recurrent decline",
        "result": None,
    }
    monkeypatch.setitem(sys.modules, "rwkv7_kernels", module)
    ops_rwkv7._reset_kernel_discovery_for_tests()
    model = RWKV7ForCausalLM(tiny_config).train()
    ids = torch.arange(4 * 128).reshape(4, 128) % tiny_config.vocab_size

    with pytest.raises(RuntimeError, match="atomic adaptive RWKV7 recurrent"):
        model(input_ids=ids, labels=ids, use_cache=False)

    assert linear_calls
    assert all(row["adaptive_fast_program"] is True for row in linear_calls)
    # Lexical publication must restore the caller even on a failed layer.
    assert ops_rwkv7._training_batch_adaptive_fast_program.get() is None
    assert ops_rwkv7._training_batch_initial_state_zero.get() is None


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
    module.RWKV7_KERNEL_API_VERSION = 3
    module.probe_model_forward_v1 = lambda _owner, _request: {
        "supported": supported,
        "implementation": "fake-model-v1",
        "reason": "fake model route" if supported else "fake model unsupported",
        "phase": "decode",
    }

    def run(_owner, request):
        if malformed:
            return {"output_kind": request["model_kind"]}
        if request["model_kind"] == "causal_lm":
            return {
                "output_kind": "causal_lm",
                "logits": torch.full((1, 2, 7), 3.0),
                "past_key_values": request["past_key_values"],
                "implementation": "fake-causal-prefill-v1",
                "phase": "prefill",
            }
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


def test_causal_lm_uses_single_early_model_boundary(monkeypatch, tiny_config):
    from rwkv7_hf.modeling_rwkv7 import RWKV7ForCausalLM

    install_fake_model_kernel(monkeypatch)
    model = RWKV7ForCausalLM(tiny_config).eval()
    with torch.inference_mode():
        output = model(input_ids=torch.tensor([[1, 2]]), use_cache=True)
    assert tuple(output.logits.shape) == (1, 2, 7)
    assert bool((output.logits == 3).all())
    assert get_last_model_route()["implementation"] == "fake-causal-prefill-v1"
    assert get_last_model_route()["phase"] == "prefill"
