from __future__ import annotations

import sys
import types

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from rwkv7_hf import ops_rwkv7
from rwkv7_hf.cache_rwkv7 import RWKV7Cache
from rwkv7_hf.ops_rwkv7 import (
    RWKV7ExecutionContext,
    get_last_linear_route,
    get_last_mix6_route,
    get_last_model_route,
    get_last_recurrent_route,
    maybe_linear_training,
    maybe_mix6_training,
    maybe_model_forward,
    resolve_execution_context,
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


def envelope(kind, *, supported, implementation, reason, result=None, phase=None):
    return {
        "api_version": 4,
        "kind": kind,
        "supported": supported,
        "implementation": implementation,
        "reason": reason,
        "result": result,
        "phase": phase or ("training" if "training" in kind else "prefill"),
    }


@pytest.fixture(autouse=True)
def reset_optional_kernel(monkeypatch):
    monkeypatch.delenv("RWKV7_BACKEND", raising=False)
    monkeypatch.delitem(sys.modules, "rwkv7_kernels", raising=False)
    ops_rwkv7._reset_kernel_discovery_for_tests()
    yield
    ops_rwkv7._reset_kernel_discovery_for_tests()


def install_fake_kernel(monkeypatch, execute):
    module = types.ModuleType("rwkv7_kernels")
    module.RWKV7_KERNEL_API_VERSION = 4
    module.execute_optional_v4 = execute
    monkeypatch.setitem(sys.modules, "rwkv7_kernels", module)
    ops_rwkv7._reset_kernel_discovery_for_tests()
    return module


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
    assert get_last_recurrent_route()["selected"] == "reference"


def test_forced_optimized_without_package_fails_clearly(monkeypatch):
    monkeypatch.setattr(
        ops_rwkv7.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ModuleNotFoundError("absent")),
    )
    with pytest.raises(RuntimeError, match="optimized RWKV7 backend is unavailable"):
        rwkv7_recurrent(*recurrent_inputs(), backend="optimized")


def test_recurrent_v4_supported_records_route_and_gradients(monkeypatch):
    calls = []

    def execute(kind, *args, **kwargs):
        calls.append((kind, kwargs))
        assert kind == "recurrent"
        return envelope(
            kind,
            supported=True,
            implementation="fake-recurrent-v4",
            reason="fake accepted",
            result=rwkv7_recurrent_reference(*args),
        )

    install_fake_kernel(monkeypatch, execute)
    inputs = recurrent_inputs(requires_grad=True)
    output, state = rwkv7_recurrent(*inputs)
    (output.square().mean() + state.square().mean()).backward()
    assert calls == [("recurrent", {"training": False})]
    assert all(tensor.grad is not None for tensor in inputs[:-1])
    assert get_last_recurrent_route() == {
        "requested": "auto",
        "selected": "optimized",
        "implementation": "fake-recurrent-v4",
        "reason": "fake accepted",
        "phase": "prefill",
    }


@pytest.mark.parametrize("failure", ["unsupported", "exception", "malformed"])
def test_auto_contains_optional_failures_and_strict_surfaces(monkeypatch, failure):
    def execute(kind, *args, **_kwargs):
        if failure == "exception":
            raise RuntimeError("broken v4 execution")
        if failure == "malformed":
            return {"api_version": 4, "kind": kind}
        return envelope(
            kind,
            supported=False,
            implementation="fake-recurrent-v4",
            reason="unsupported v4 request",
        )

    install_fake_kernel(monkeypatch, execute)
    inputs = recurrent_inputs()
    assert_reference_equal(rwkv7_recurrent(*inputs), rwkv7_recurrent_reference(*inputs))
    assert get_last_recurrent_route()["selected"] == "reference"
    with pytest.raises(RuntimeError):
        rwkv7_recurrent(*inputs, backend="optimized")


@pytest.mark.parametrize("api_version", (3, True, 4.0))
def test_api_version_mismatch_is_not_silently_selected(monkeypatch, api_version):
    module = types.ModuleType("rwkv7_kernels")
    module.RWKV7_KERNEL_API_VERSION = api_version
    module.execute_optional_v4 = lambda *_a, **_k: None
    monkeypatch.setitem(sys.modules, "rwkv7_kernels", module)
    ops_rwkv7._reset_kernel_discovery_for_tests()
    inputs = recurrent_inputs()
    assert_reference_equal(rwkv7_recurrent(*inputs), rwkv7_recurrent_reference(*inputs))
    assert "kernel API mismatch" in get_last_recurrent_route()["reason"]


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("api_version", True),
        ("api_version", 4.0),
        ("kind", b"recurrent"),
        ("supported", "false"),
        ("supported", 1),
        ("implementation", 7),
        ("reason", None),
        ("phase", False),
    ),
)
def test_hf_boundary_rejects_coercible_envelope_fields(monkeypatch, field, value):
    def execute(kind, *_args, **_kwargs):
        result = envelope(
            kind,
            supported=False,
            implementation="fake-recurrent-v4",
            reason="unsupported",
        )
        result[field] = value
        return result

    install_fake_kernel(monkeypatch, execute)
    with pytest.raises(TypeError, match=field):
        ops_rwkv7._kernel_envelope(
            "recurrent", *recurrent_inputs(), training=False
        )


def test_hf_boundary_rejects_unknown_envelope_fields(monkeypatch):
    def execute(kind, *_args, **_kwargs):
        result = envelope(
            kind,
            supported=False,
            implementation="fake-recurrent-v4",
            reason="unsupported",
        )
        result["private_payload"] = object()
        return result

    install_fake_kernel(monkeypatch, execute)
    with pytest.raises(TypeError, match="unknown fields"):
        ops_rwkv7._kernel_envelope(
            "recurrent", *recurrent_inputs(), training=False
        )


def certified_context() -> RWKV7ExecutionContext:
    return RWKV7ExecutionContext(
        training=True,
        fully_active=True,
        initial_state_zero=True,
        autograd_leaf_eligible=True,
        force_reference_program=False,
        optimized_program=True,
        program_id="fake-program-v4",
        token_aligned=True,
        program_implementation="fake-program-v4",
        program_reason="certified",
    )


def test_training_preflight_is_one_v4_operation(monkeypatch):
    calls = []

    def execute(kind, *args, **kwargs):
        calls.append((kind, args, kwargs))
        return envelope(
            kind,
            supported=True,
            implementation="fake-program-v4",
            reason="certified",
            result={"program_id": "fake-program-v4", "token_aligned": True},
            phase="training",
        )

    install_fake_kernel(monkeypatch, execute)
    hidden = torch.randn(4, 128, 64, requires_grad=True)
    mask = torch.ones(4, 128, dtype=torch.bool)
    context = resolve_execution_context(
        mask,
        training=True,
        fully_active=True,
        initial_state_zero=True,
        autograd_leaf_eligible=True,
        hidden_states=hidden,
        head_dim=64,
    )
    assert context.optimized_program is True
    assert context.program_id == "fake-program-v4"
    assert len(calls) == 1 and calls[0][0] == "training_program"
    assert "token_aligned" not in calls[0][2]


def test_strict_optimized_training_decline_fails_at_model_boundary(monkeypatch):
    calls = []

    def execute(kind, *_args, **_kwargs):
        calls.append(kind)
        return envelope(
            kind,
            supported=False,
            implementation="native-training-program-v4",
            reason="complete three-leaf plan is unavailable",
            phase="training",
        )

    install_fake_kernel(monkeypatch, execute)
    monkeypatch.setenv("RWKV7_BACKEND", "optimized")
    hidden = torch.randn(4, 128, 64, requires_grad=True)
    mask = torch.ones(4, 128, dtype=torch.bool)
    with pytest.raises(RuntimeError, match="unavailable at the model boundary"):
        resolve_execution_context(
            mask,
            training=True,
            fully_active=True,
            initial_state_zero=True,
            autograd_leaf_eligible=True,
            hidden_states=hidden,
            head_dim=64,
        )
    assert calls == ["training_program"]


def test_explicit_context_flows_to_all_training_leaves(monkeypatch):
    received = []
    context = certified_context()

    def execute(kind, *args, **kwargs):
        received.append((kind, kwargs))
        if kind == "linear_training":
            result = torch.nn.functional.linear(*args)
        elif kind == "mix6_training":
            value, shifted, *mixes = args
            result = tuple(
                value + (shifted - value) * mix.view(1, 1, -1) for mix in mixes
            )
        else:
            result = rwkv7_recurrent_reference(*args)
        return envelope(
            kind,
            supported=True,
            implementation=f"fake-{kind}-v4",
            reason="executed",
            result=result,
            phase="training",
        )

    install_fake_kernel(monkeypatch, execute)
    value = torch.randn(2, 3, 4, requires_grad=True)
    shifted = torch.randn_like(value)
    mixes = tuple(torch.randn(4) for _ in range(6))
    weight = torch.randn(5, 4, requires_grad=True)
    assert maybe_linear_training(
        value, weight, None, training=True, execution_context=context
    ) is not None
    assert maybe_mix6_training(
        value, shifted, mixes, training=True, execution_context=context
    ) is not None
    rwkv7_recurrent(
        *recurrent_inputs(requires_grad=True),
        training=True,
        initial_state_zero=True,
        execution_context=context,
    )
    assert [row[0] for row in received] == [
        "linear_training",
        "mix6_training",
        "recurrent",
    ]
    assert all(row[1]["program_id"] == "fake-program-v4" for row in received)
    assert all(row[1]["facts"]["fully_active"] is True for row in received)


@pytest.mark.parametrize("force_reference_program", [False, True])
def test_uncertified_program_keeps_every_leaf_on_reference(
    monkeypatch, force_reference_program
):
    calls = []
    install_fake_kernel(monkeypatch, lambda *a, **k: calls.append((a, k)))
    context = RWKV7ExecutionContext(
        training=True,
        fully_active=True,
        initial_state_zero=True,
        autograd_leaf_eligible=False,
        force_reference_program=force_reference_program,
        optimized_program=False,
        program_id=None,
        token_aligned=None,
        program_implementation="torch-reference-training-program-v1",
        program_reason="no certificate",
    )
    value = torch.randn(2, 3, 4, requires_grad=True)
    shifted = torch.randn_like(value)
    mixes = tuple(torch.randn(4) for _ in range(6))
    weight = torch.randn(5, 4, requires_grad=True)
    assert maybe_linear_training(
        value, weight, None, training=True, execution_context=context
    ) is None
    assert maybe_mix6_training(
        value, shifted, mixes, training=True, execution_context=context
    ) is None
    rwkv7_recurrent(
        *recurrent_inputs(requires_grad=True),
        training=True,
        execution_context=context,
    )
    assert calls == []
    assert get_last_linear_route()["selected"] == "reference"
    assert get_last_mix6_route()["selected"] == "reference"
    assert get_last_recurrent_route()["selected"] == "reference"


@pytest.mark.parametrize("leaf", ["linear", "mix6", "recurrent"])
@pytest.mark.parametrize("force_reference_program", [False, True])
def test_uncertified_strict_program_fails_before_kernel(
    monkeypatch, leaf, force_reference_program
):
    install_fake_kernel(
        monkeypatch,
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("must not execute")),
    )
    context = RWKV7ExecutionContext(
        training=True,
        fully_active=True,
        initial_state_zero=True,
        autograd_leaf_eligible=False,
        force_reference_program=force_reference_program,
        optimized_program=False,
        program_id=None,
        token_aligned=None,
        program_implementation="torch-reference-training-program-v1",
        program_reason="no certificate",
    )
    with pytest.raises(RuntimeError, match="atomic fast-program certificate"):
        if leaf == "linear":
            maybe_linear_training(
                torch.randn(2, 3, 4),
                torch.randn(5, 4),
                None,
                training=True,
                backend="optimized",
                execution_context=context,
            )
        elif leaf == "mix6":
            value = torch.randn(2, 3, 4)
            maybe_mix6_training(
                value,
                value,
                tuple(torch.randn(4) for _ in range(6)),
                training=True,
                backend="optimized",
                execution_context=context,
            )
        else:
            rwkv7_recurrent(
                *recurrent_inputs(),
                training=True,
                backend="optimized",
                execution_context=context,
            )


def test_atomic_program_late_decline_is_fail_closed(monkeypatch):
    def execute(kind, *_args, **_kwargs):
        return envelope(
            kind,
            supported=False,
            implementation="fake-recurrent-v4",
            reason="late decline",
            phase="training",
        )

    install_fake_kernel(monkeypatch, execute)
    with pytest.raises(RuntimeError, match="late decline"):
        rwkv7_recurrent(
            *recurrent_inputs(),
            training=True,
            execution_context=certified_context(),
        )


def test_checkpoint_control_flow_is_not_swallowed(monkeypatch):
    calls = 0

    def execute(kind, value, weight, bias, **_kwargs):
        nonlocal calls
        calls += 1
        return envelope(
            kind,
            supported=True,
            implementation="fake-linear-v4",
            reason="checkpoint",
            result=torch.nn.functional.linear(value, weight, bias),
            phase="training",
        )

    install_fake_kernel(monkeypatch, execute)
    value = torch.randn(4, 8, requires_grad=True)
    weight = torch.randn(8, 8, requires_grad=True)

    def projection(hidden):
        output = maybe_linear_training(hidden, weight, None, training=True)
        assert output is not None
        return output

    checkpoint(projection, value, use_reentrant=False).sum().backward()
    assert calls == 2
    assert value.grad is not None and weight.grad is not None


def model_request(kind="base", *, training=False):
    return {
        "model_kind": kind,
        "training": training,
        "use_cache": True,
        "hidden_states": torch.zeros(1, 1, 4),
        "past_key_values": RWKV7Cache(num_layers=0),
    }


def transactional_model_request(*, use_cache=True, cache_type=RWKV7Cache):
    recurrent = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    attention_shift = torch.arange(4, dtype=torch.float32).reshape(1, 4)
    ffn_shift = -attention_shift
    cache = cache_type(
        [recurrent],
        [attention_shift],
        [ffn_shift],
        seen_tokens=5,
    )
    return {
        "model_kind": "base",
        "training": False,
        "use_cache": use_cache,
        "hidden_states": torch.zeros(1, 2, 4),
        "past_key_values": cache,
    }


def snapshot_cache(cache):
    snapshot = {"seen_tokens": cache.seen_tokens}
    for name in ("recurrent_state", "attention_shift", "ffn_shift"):
        values = getattr(cache, name)
        snapshot[name] = (
            values,
            tuple(
                (value, None if value is None else value.clone()) for value in values
            ),
        )
    return snapshot


def assert_cache_unchanged(cache, snapshot):
    assert cache.seen_tokens == snapshot["seen_tokens"]
    for name in ("recurrent_state", "attention_shift", "ffn_shift"):
        values = getattr(cache, name)
        original_list, original_values = snapshot[name]
        assert values is original_list
        assert len(values) == len(original_values)
        for value, (original_value, expected) in zip(values, original_values):
            assert value is original_value
            if expected is None:
                assert value is None
            else:
                torch.testing.assert_close(value, expected)


def mutate_attempt_cache(cache, *, token_count=2):
    cache.recurrent_state[0].add_(100)
    cache.attention_shift[0].add_(200)
    cache.ffn_shift[0].sub_(300)
    cache.seen_tokens += token_count


class SeenTokenWriteCountingCache(RWKV7Cache):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen_token_writes = 0

    @RWKV7Cache.seen_tokens.setter
    def seen_tokens(self, value):
        self.seen_token_writes += 1
        self._seen_tokens = int(value)


def test_model_v4_supported_and_malformed_result_is_fail_closed(monkeypatch):
    def execute(kind, _owner, request):
        request["past_key_values"].seen_tokens += 1
        result = {
            "output_kind": request["model_kind"],
            "last_hidden_state": torch.ones(1, 1, 4),
            "past_key_values": request["past_key_values"],
        }
        return envelope(
            kind,
            supported=True,
            implementation="fake-model-v4",
            reason="model accepted",
            result=result,
            phase="decode",
        )

    install_fake_kernel(monkeypatch, execute)
    result = maybe_model_forward(object(), model_request())
    assert tuple(result["last_hidden_state"].shape) == (1, 1, 4)
    assert get_last_model_route()["implementation"] == "fake-model-v4"

    def malformed(kind, *_args):
        return envelope(
            kind,
            supported=True,
            implementation="fake-model-v4",
            reason="bad output",
            result={"output_kind": "base"},
        )

    install_fake_kernel(monkeypatch, malformed)
    with pytest.raises(RuntimeError, match="fail-closed"):
        maybe_model_forward(object(), model_request())


@pytest.mark.parametrize("backend", [None, "optimized"], ids=["auto", "strict"])
def test_negative_model_probe_is_side_effect_free(monkeypatch, backend):
    request = transactional_model_request()
    original_cache = request["past_key_values"]
    original_snapshot = snapshot_cache(original_cache)

    def execute(kind, _owner, attempt_request):
        assert attempt_request["past_key_values"] is original_cache
        return envelope(
            kind,
            supported=False,
            implementation="fake-model-v4",
            reason="side-effect-free negative probe",
        )

    install_fake_kernel(monkeypatch, execute)
    if backend is None:
        assert maybe_model_forward(object(), request) is None
        assert get_last_model_route()["selected"] == "reference"
    else:
        with pytest.raises(RuntimeError):
            maybe_model_forward(object(), request, backend=backend)
    assert_cache_unchanged(original_cache, original_snapshot)


@pytest.mark.parametrize("backend", [None, "optimized"], ids=["auto", "strict"])
def test_model_execution_exception_is_fail_closed(monkeypatch, backend):
    request = transactional_model_request()
    original_cache = request["past_key_values"]
    original_snapshot = snapshot_cache(original_cache)

    def execute(_kind, _owner, attempt_request):
        assert attempt_request["past_key_values"] is original_cache
        raise RuntimeError("native execution failed before cache update")

    install_fake_kernel(monkeypatch, execute)
    with pytest.raises(RuntimeError, match="fail-closed"):
        maybe_model_forward(object(), request, backend=backend)
    assert_cache_unchanged(original_cache, original_snapshot)


def test_successful_model_cache_execution_preserves_identity(monkeypatch):
    request = transactional_model_request(cache_type=SeenTokenWriteCountingCache)
    original_cache = request["past_key_values"]
    original_snapshot = snapshot_cache(original_cache)

    def execute(kind, _owner, attempt_request):
        attempt_cache = attempt_request["past_key_values"]
        assert attempt_cache is original_cache
        mutate_attempt_cache(attempt_cache)
        return envelope(
            kind,
            supported=True,
            implementation="fake-model-v4",
            reason="valid zero-copy result",
            result={
                "output_kind": "base",
                "last_hidden_state": torch.ones(1, 2, 4),
                "past_key_values": attempt_cache,
            },
        )

    install_fake_kernel(monkeypatch, execute)
    result = maybe_model_forward(object(), request)
    assert result is not None
    assert result["past_key_values"] is original_cache
    assert original_cache.seen_tokens == original_snapshot["seen_tokens"] + 2
    assert original_cache.seen_token_writes == 1
    torch.testing.assert_close(
        original_cache.recurrent_state[0],
        original_snapshot["recurrent_state"][1][0][1] + 100,
    )
    torch.testing.assert_close(
        original_cache.attention_shift[0],
        original_snapshot["attention_shift"][1][0][1] + 200,
    )
    torch.testing.assert_close(
        original_cache.ffn_shift[0],
        original_snapshot["ffn_shift"][1][0][1] - 300,
    )


def test_use_cache_false_backend_contract_preserves_callers_cache(monkeypatch):
    request = transactional_model_request(use_cache=False)
    original_cache = request["past_key_values"]
    original_snapshot = snapshot_cache(original_cache)

    def execute(kind, _owner, attempt_request):
        attempt_cache = attempt_request["past_key_values"]
        assert attempt_cache is original_cache
        return envelope(
            kind,
            supported=True,
            implementation="fake-model-v4",
            reason="valid cache-free result",
            result={
                "output_kind": "base",
                "last_hidden_state": torch.ones(1, 2, 4),
                "past_key_values": None,
            },
        )

    install_fake_kernel(monkeypatch, execute)
    result = maybe_model_forward(object(), request)
    assert result is not None and result["past_key_values"] is None
    assert_cache_unchanged(original_cache, original_snapshot)


@pytest.mark.parametrize("backend", [None, "optimized"], ids=["auto", "strict"])
@pytest.mark.parametrize(
    "defect", ["shape", "dtype", "cache_identity", "cache_advance"]
)
def test_model_result_payload_validation_is_fail_closed(monkeypatch, backend, defect):
    request = transactional_model_request()

    def execute(kind, _owner, attempt_request):
        attempt_cache = attempt_request["past_key_values"]
        mutate_attempt_cache(
            attempt_cache,
            token_count=1 if defect == "cache_advance" else 2,
        )
        output = torch.ones(1, 2, 4)
        if defect == "shape":
            output = output[:, :1]
        elif defect == "dtype":
            output = output.to(dtype=torch.float64)
        returned_cache = attempt_cache
        if defect == "cache_identity":
            returned_cache = attempt_cache.clone()
        return envelope(
            kind,
            supported=True,
            implementation="fake-model-v4",
            reason=f"invalid {defect}",
            result={
                "output_kind": "base",
                "last_hidden_state": output,
                "past_key_values": returned_cache,
            },
        )

    install_fake_kernel(monkeypatch, execute)
    with pytest.raises(RuntimeError, match="fail-closed"):
        maybe_model_forward(object(), request, backend=backend)


@pytest.mark.parametrize("backend", [None, "optimized"], ids=["auto", "strict"])
@pytest.mark.parametrize("defect", ["shape", "dtype"])
def test_linear_result_shape_and_dtype_are_fail_closed(monkeypatch, backend, defect):
    value = torch.randn(2, 3, 4, dtype=torch.float64)
    weight = torch.randn(5, 4, dtype=torch.float64)

    def execute(kind, *args, **_kwargs):
        result = torch.nn.functional.linear(*args)
        if defect == "shape":
            result = result[..., :-1]
        else:
            result = result.to(dtype=torch.float32)
        return envelope(
            kind,
            supported=True,
            implementation="fake-linear-v4",
            reason=f"invalid {defect}",
            result=result,
            phase="training",
        )

    install_fake_kernel(monkeypatch, execute)
    if backend is None:
        assert maybe_linear_training(value, weight, None, training=True) is None
        assert get_last_linear_route()["selected"] == "reference"
    else:
        with pytest.raises(RuntimeError):
            maybe_linear_training(
                value,
                weight,
                None,
                training=True,
                backend=backend,
            )


@pytest.mark.parametrize("backend", [None, "optimized"], ids=["auto", "strict"])
@pytest.mark.parametrize(
    "defect", ["output_shape", "output_dtype", "state_shape", "state_dtype"]
)
def test_recurrent_result_shape_and_dtype_are_fail_closed(
    monkeypatch, backend, defect
):
    inputs = recurrent_inputs()
    reference = rwkv7_recurrent_reference(*inputs)

    def execute(kind, *args, **_kwargs):
        output, state = rwkv7_recurrent_reference(*args)
        if defect == "output_shape":
            output = output[..., :-1]
        elif defect == "output_dtype":
            output = output.to(dtype=torch.float32)
        elif defect == "state_shape":
            state = state[..., :-1]
        else:
            state = state.to(dtype=torch.float32)
        return envelope(
            kind,
            supported=True,
            implementation="fake-recurrent-v4",
            reason=f"invalid {defect}",
            result=(output, state),
        )

    install_fake_kernel(monkeypatch, execute)
    if backend is None:
        assert_reference_equal(rwkv7_recurrent(*inputs), reference)
        assert get_last_recurrent_route()["selected"] == "reference"
    else:
        with pytest.raises(RuntimeError):
            rwkv7_recurrent(*inputs, backend=backend)


def test_training_never_enters_whole_model_backend(monkeypatch):
    calls = []
    install_fake_kernel(monkeypatch, lambda *args, **kwargs: calls.append((args, kwargs)))
    assert maybe_model_forward(object(), model_request(training=True)) is None
    assert calls == []
    assert get_last_model_route()["selected"] == "reference"


def test_invalid_backend_mode_is_rejected():
    with pytest.raises(ValueError, match="auto, reference, optimized"):
        rwkv7_recurrent(*recurrent_inputs(), backend="fastest")
