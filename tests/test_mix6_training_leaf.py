from __future__ import annotations

import importlib
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def source_kernel_package(monkeypatch):
    """Keep this test module independent of collection/execution order."""

    monkeypatch.syspath_prepend(str(ROOT / "kernels"))
    for name in tuple(sys.modules):
        if name == "rwkv7_kernels" or name.startswith("rwkv7_kernels."):
            sys.modules.pop(name)
    yield
    for name in tuple(sys.modules):
        if name == "rwkv7_kernels" or name.startswith("rwkv7_kernels."):
            sys.modules.pop(name)


def test_mix6_leaf_probe_rejects_cpu_without_building():
    leaf = importlib.import_module("rwkv7_kernels.time_mix.training_mix6")
    value = torch.randn(2, 16, 8, dtype=torch.bfloat16, requires_grad=True)
    shifted = torch.randn_like(value, requires_grad=True)
    mixes = tuple(torch.randn(8, dtype=torch.bfloat16) for _ in range(6))

    support = leaf.probe_mix6_training_v1(value, shifted, *mixes)

    assert not support["supported"]
    assert support["implementation"] == leaf.IMPLEMENTATION
    assert "CUDA" in support["reason"]


def test_mix6_dispatcher_is_policy_gated_but_shape_semantics_are_explicit(
    monkeypatch,
):
    dispatcher = importlib.import_module("rwkv7_kernels.training_dispatcher")
    accepted = {
        "supported": True,
        "implementation": "native-nvidia-rwkv7-mix6-training-v1",
        "reason": "test leaf",
    }
    calls = []
    monkeypatch.setattr(dispatcher, "_probe_mix6", lambda *_args: dict(accepted))
    monkeypatch.setattr(
        dispatcher,
        "_run_mix6",
        lambda value, shifted, *mixes: (
            calls.append((value, shifted, mixes)) or tuple(value for _ in mixes)
        ),
    )
    value = torch.randn(2, 16, 8)
    shifted = torch.randn_like(value)
    mixes = tuple(torch.randn(8) for _ in range(6))

    monkeypatch.setenv("RWKV7_TRAINING_KERNEL_IMPL", "auto")
    assert not dispatcher.probe_mix6_training_v1(
        value,
        shifted,
        *mixes,
        fully_active=True,
        token_aligned=True,
    )["supported"]

    monkeypatch.setenv("RWKV7_TRAINING_KERNEL_IMPL", "adaptive")
    masked_unaligned = dispatcher.probe_mix6_training_v1(
        value,
        shifted,
        *mixes,
        fully_active=False,
        token_aligned=False,
    )
    assert masked_unaligned["supported"]
    outputs = dispatcher.mix6_training_v1(
        value,
        shifted,
        *mixes,
        fully_active=False,
        token_aligned=False,
    )
    assert len(outputs) == 6
    assert len(calls) == 1


def test_mix6_atomic_execute_probes_once_for_success_fallback_and_error(monkeypatch):
    dispatcher = importlib.import_module("rwkv7_kernels.training_dispatcher")
    monkeypatch.setenv("RWKV7_TRAINING_KERNEL_IMPL", "adaptive")
    value = torch.randn(2, 16, 8)
    shifted = torch.randn_like(value)
    mixes = tuple(torch.randn(8) for _ in range(6))
    calls = {"probe": 0, "run": 0}

    def probe(*_args):
        calls["probe"] += 1
        return {
            "supported": True,
            "implementation": "native-nvidia-rwkv7-mix6-training-v1",
            "reason": "atomic Mix6 test",
        }

    def run(value, _shifted, *mixes):
        calls["run"] += 1
        return tuple(value for _ in mixes)

    monkeypatch.setattr(dispatcher, "_probe_mix6", probe)
    monkeypatch.setattr(dispatcher, "_run_mix6", run)
    execution = dispatcher.execute_mix6_training_v1(value, shifted, *mixes)
    assert execution["supported"]
    assert calls == {"probe": 1, "run": 1}

    def unsupported(*_args):
        calls["probe"] += 1
        return {
            "supported": False,
            "implementation": "native-nvidia-rwkv7-mix6-training-v1",
            "reason": "unsupported Mix6",
        }

    monkeypatch.setattr(dispatcher, "_probe_mix6", unsupported)
    execution = dispatcher.execute_mix6_training_v1(value, shifted, *mixes)
    assert not execution["supported"] and execution["result"] is None
    assert calls == {"probe": 2, "run": 1}

    monkeypatch.setattr(dispatcher, "_probe_mix6", probe)

    def broken(*_args):
        calls["run"] += 1
        raise RuntimeError("Mix6 execution failed")

    monkeypatch.setattr(dispatcher, "_run_mix6", broken)
    with pytest.raises(RuntimeError, match="Mix6 execution failed"):
        dispatcher.mix6_training_v1(value, shifted, *mixes)
    assert calls == {"probe": 3, "run": 2}


def test_mix6_probe_does_not_require_trainable_inputs():
    leaf = importlib.import_module("rwkv7_kernels.time_mix.training_mix6")
    source = leaf.probe_mix6_training_v1.__code__

    # A frozen/PEFT first layer can legitimately need only the native forward.
    # Keep this protocol independent of ``requires_grad`` so strict optimized
    # mode does not reject that request before a later trainable projection.
    assert "requires_grad" not in source.co_names


def test_mix6_loader_is_independent_of_legacy_train_temp_module():
    leaf = importlib.import_module("rwkv7_kernels.time_mix.training_mix6")
    build = importlib.import_module("rwkv7_kernels.nvidia.extension_build")
    source = Path(leaf.__file__).read_text(encoding="utf-8")

    assert "train_temp_cuda" not in source
    assert "include_target=True" in source
    assert "--use_fast_math" not in build.CUDA_EXTENSION_OPTIMIZATION_FLAGS
    assert "-O3" in build.CUDA_EXTENSION_OPTIMIZATION_FLAGS


def test_mix6_packed_input_helper_repairs_odd_bf16_storage_offset():
    leaf = importlib.import_module("rwkv7_kernels.time_mix.training_mix6")
    storage = torch.randn(9, dtype=torch.bfloat16)
    odd_offset = storage[1:]

    assert odd_offset.is_contiguous()
    assert odd_offset.data_ptr() % 4 == 2
    packed = leaf._bf16x2_contiguous(odd_offset)

    assert packed.is_contiguous()
    assert packed.data_ptr() % 4 == 0
    torch.testing.assert_close(packed, odd_offset)


def test_mix6_small_backward_replays_explicit_shift_math(monkeypatch):
    leaf = importlib.import_module("rwkv7_kernels.time_mix.training_mix6")
    torch.manual_seed(211)
    value = torch.randn(1, 16, 8, requires_grad=True)
    shifted = torch.randn_like(value, requires_grad=True)
    mixes = tuple(torch.randn(8, requires_grad=True) for _ in range(6))
    output_grads = tuple(torch.randn_like(value) for _ in range(6))

    namespace = getattr(torch.ops, leaf.OPERATOR_NAMESPACE)

    def reject_native(*_args):
        raise AssertionError("small Mix6 backward must replay PyTorch math")

    monkeypatch.setattr(namespace, "backward", reject_native, raising=False)
    expected_outputs = leaf._canonical_mix6(value, shifted, mixes)
    expected = torch.autograd.grad(
        expected_outputs,
        (value, shifted, *mixes),
        output_grads,
    )

    class Context:
        saved_tensors = (
            value.detach(),
            shifted.detach(),
            *(mix.detach() for mix in mixes),
        )

    with torch.no_grad():
        actual = leaf._Mix6Shifted.backward(Context(), *output_grads)
    for candidate, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(candidate, reference)


def test_mix6_small_custom_autograd_preserves_shared_parent_graph(monkeypatch):
    leaf = importlib.import_module("rwkv7_kernels.time_mix.training_mix6")
    torch.manual_seed(223)
    parent = torch.randn(1, 16, 8, requires_grad=True)
    mixes = tuple(torch.randn(8, requires_grad=True) for _ in range(6))
    namespace = getattr(torch.ops, leaf.OPERATOR_NAMESPACE)
    monkeypatch.setattr(
        namespace,
        "forward",
        lambda value, shifted, *parameters: leaf._canonical_mix6(
            value,
            shifted,
            tuple(parameters),
        ),
        raising=False,
    )

    value = parent.square()
    shifted = parent.sin()
    outputs = leaf._Mix6Shifted.apply(value, shifted, *mixes)
    loss = sum(output.float().square().mean() for output in outputs)
    actual = torch.autograd.grad(loss, (parent, *mixes))

    reference_parent = parent.detach().clone().requires_grad_(True)
    reference_mixes = tuple(mix.detach().clone().requires_grad_(True) for mix in mixes)
    reference_outputs = leaf._canonical_mix6(
        reference_parent.square(),
        reference_parent.sin(),
        reference_mixes,
    )
    reference_loss = sum(output.float().square().mean() for output in reference_outputs)
    expected = torch.autograd.grad(
        reference_loss,
        (reference_parent, *reference_mixes),
    )
    for candidate, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(candidate, reference)


def test_mix6_custom_autograd_higher_order_uses_local_vjp(monkeypatch):
    leaf = importlib.import_module("rwkv7_kernels.time_mix.training_mix6")
    torch.manual_seed(227)
    parent = torch.randn(1, 32, 8, requires_grad=True)
    mixes = tuple(torch.randn(8, requires_grad=True) for _ in range(6))
    namespace = getattr(torch.ops, leaf.OPERATOR_NAMESPACE)
    monkeypatch.setattr(
        namespace,
        "forward",
        lambda value, shifted, *parameters: leaf._canonical_mix6(
            value,
            shifted,
            tuple(parameters),
        ),
        raising=False,
    )

    outputs = leaf._Mix6Shifted.apply(parent.square(), parent.sin(), *mixes)
    output_gradients = tuple(torch.randn_like(output) for output in outputs)
    first = torch.autograd.grad(
        outputs,
        (parent, *mixes),
        grad_outputs=output_gradients,
        create_graph=True,
    )
    second = torch.autograd.grad(
        sum(gradient.square().mean() for gradient in first),
        (parent, *mixes),
        allow_unused=True,
    )

    assert all(gradient.requires_grad for gradient in first)
    assert all(gradient is not None for gradient in second)
    assert all(torch.isfinite(gradient).all() for gradient in (*first, *second))


def test_mix6_native_backward_starts_at_exact_row_threshold(monkeypatch):
    leaf = importlib.import_module("rwkv7_kernels.time_mix.training_mix6")
    rows = leaf.NATIVE_BACKWARD_MIN_ROWS
    value = torch.randn(1, rows, 8, dtype=torch.bfloat16)
    shifted = torch.randn_like(value)
    mixes = tuple(torch.randn(8, dtype=torch.bfloat16) for _ in range(6))
    output_grads = tuple(torch.randn_like(value) for _ in range(6))
    calls = []

    namespace = getattr(torch.ops, leaf.OPERATOR_NAMESPACE)

    def record_native(*args):
        calls.append(args)
        return (
            torch.empty_like(value),
            torch.empty_like(shifted),
            *(torch.empty_like(mix) for mix in mixes),
        )

    monkeypatch.setattr(namespace, "backward", record_native, raising=False)

    class Context:
        saved_tensors = (value, shifted, *mixes)

    with torch.no_grad():
        actual = leaf._Mix6Shifted.backward(Context(), *output_grads)

    assert len(calls) == 1
    assert len(actual) == 8
    assert tuple(item.shape for item in actual) == (
        value.shape,
        shifted.shape,
        *(mix.shape for mix in mixes),
    )


def test_mix6_v1_cuda_sources_are_stream_and_device_safe():
    source = ROOT / "kernels" / "rwkv7_kernels" / "nvidia" / "csrc" / "training"
    cpp = (source / "rwkv7_tmix_mix6_shifted_bf16_v1.cpp").read_text()
    cuda = (source / "rwkv7_tmix_mix6_shifted_bf16_v1.cu").read_text()

    assert "c10::cuda::CUDAGuard" in cpp
    assert "check_same_device" in cpp
    assert "four-byte aligned for packed BF16x2 access" in cpp
    assert cuda.count("getCurrentCUDAStream(x.get_device())") == 2
    assert cuda.count("C10_CUDA_KERNEL_LAUNCH_CHECK()") == 3
    assert "shifted" in cpp and "shifted" in cuda


def test_mix6_v1_backward_uses_parallel_deterministic_partial_reduction():
    source = (
        ROOT
        / "kernels"
        / "rwkv7_kernels"
        / "nvidia"
        / "csrc"
        / "training"
        / "rwkv7_tmix_mix6_shifted_bf16_v1.cu"
    ).read_text()

    assert "MIX6_ROWS_PER_PARTIAL = 64" in source
    assert "MIX6_CHANNEL_PAIRS_PER_TILE = 16" in source
    assert "mix6_shifted_backward_partials_kernel" in source
    assert "mix6_shifted_backward_finalize_kernel" in source
    assert "parameter_partials" in source
    assert "parameter-partial workspace size overflow" in source
    assert "partial_index * 6 * channels" in source
    assert "row += MIX6_ROW_WORKERS_PER_BLOCK" in source
    assert "for (int64_t partial = 0; partial < partial_count; ++partial)" in source
    assert "value + divisor - 1" not in source
    assert "atomicAdd" not in source
    assert "blockIdx.y" not in source
    assert "for (int64_t row = 0; row < rows; ++row)" not in source


def test_clean_model_passes_masked_shift_tensor_to_mix6_boundary():
    source = (ROOT / "rwkv7_hf" / "modeling_rwkv7.py").read_text()
    assert "shifted, final_shift = _masked_token_shift(" in source
    assert (
        "maybe_mix6_training(\n            hidden_states,\n            shifted,"
        in source
    )
    assert "hidden_states + delta * self.x_r" in source
    dispatch = source.index("mixed_inputs = maybe_mix6_training(")
    fallback = source.index("if mixed_inputs is None:", dispatch)
    delta = source.index("delta = shifted - hidden_states", dispatch)
    assert dispatch < fallback < delta
