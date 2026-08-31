"""Opt-in official RWKV-LM CUDA training operators for HF RWKV-7 models.

The kernels under ``csrc/training/rwkv_lm`` are vendored from RWKV-LM at the exact
commit recorded in that directory. This module keeps them lazy and isolated:
reference inference, reference training and production ``auto`` do not compile
or route through these ops. Only an explicit factorized/adaptive training leaf
may request the recurrent extension.

Private symbols containing ``train_temp`` intentionally retain the upstream
RWKV-LM recipe/extension namespace for source and operator provenance. The
stable first-party module name and plug-in boundary use ``official_training``.
"""

from __future__ import annotations

import importlib.util
import os
import threading
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .extension_build import cuda_extension_build_environment
from .training_math import low_rank_projection, module_linear


TRAIN_TEMP_SOURCE_COMMIT = "e6f74b63a06e08606d130043599d218209628bad"
TRAIN_TEMP_HEAD_SIZE = 64
TRAIN_TEMP_CHUNK_LEN = 16

_LOAD_LOCK = threading.Lock()
_LOADED = False
_LOAD_ERROR: BaseException | None = None
_MIX6_LOADED = False
_MIX6_LOAD_ERROR: BaseException | None = None
_RECURRENT_LOADED = False
_RECURRENT_LOAD_ERROR: BaseException | None = None
_L2WRAP_EXTENSION: Any | None = None

_COMMON_CUDA_FLAGS = [
    "-res-usage",
    "--use_fast_math",
    "-O3",
    "-Xptxas",
    "-O3",
    "--extra-device-vectorization",
]
_RECURRENT_CUDA_FLAGS = [
    flag for flag in _COMMON_CUDA_FLAGS if flag != "--use_fast_math"
] + ["--fmad=false"]
_OP_SOURCES = {
    "rwkv7_cmix_bf16_v5": (
        "rwkv7_cmix_bf16_v5.cpp",
        "rwkv7_cmix_bf16_v5.cu",
    ),
    "rwkv7_tmix_mix6_bf16_v5": (
        "rwkv7_tmix_mix6_bf16_v5.cpp",
        "rwkv7_tmix_mix6_bf16_v5.cu",
    ),
    "rwkv7_tmix_kk_pre_bf16_v5": (
        "rwkv7_tmix_kk_pre_bf16_v5.cpp",
        "rwkv7_tmix_kk_pre_bf16_v5.cu",
    ),
    "rwkv7_tmix_lnx_rkvres_xg_bf16_v1": (
        "rwkv7_tmix_lnx_rkvres_xg_bf16_v1.cpp",
        "rwkv7_tmix_lnx_rkvres_xg_bf16_v1.cu",
    ),
    "rwkv7_tmix_a_gate_bf16": (
        "rwkv7_tmix_a_gate_bf16.cpp",
        "rwkv7_tmix_a_gate_bf16.cu",
    ),
    "rwkv7_tmix_vres_gate_bf16_v3": (
        "rwkv7_tmix_vres_gate_bf16_v3.cpp",
        "rwkv7_tmix_vres_gate_bf16_v3.cu",
    ),
}


def _train_temp_checkpoint_backend() -> str:
    requested = (
        os.environ.get("RWKV7_TRAIN_TEMP_CHECKPOINT_BACKEND", "auto").strip().lower()
    )
    if requested not in {"auto", "deepspeed", "torch"}:
        raise ValueError(
            "RWKV7_TRAIN_TEMP_CHECKPOINT_BACKEND must be auto, deepspeed or torch"
        )
    if requested == "torch":
        return "torch_non_reentrant"
    if importlib.util.find_spec("deepspeed") is not None:
        return "deepspeed"
    if requested == "deepspeed":
        raise RuntimeError("DeepSpeed checkpointing was requested but is unavailable")
    return "torch_non_reentrant"


def _train_temp_checkpoint(function, *args):
    """Checkpoint one layer with the official train_temp backend when present."""

    backend = _train_temp_checkpoint_backend()
    if backend == "deepspeed":
        import deepspeed

        return deepspeed.checkpointing.checkpoint(function, *args)

    from torch.utils.checkpoint import checkpoint

    return checkpoint(function, *args, use_reentrant=False)


def _source_root() -> Path:
    return Path(__file__).resolve().parent / "csrc" / "training" / "rwkv_lm"


def _cuda_include_paths(
    cuda_home: str | os.PathLike[str], *, include_target: bool = False
) -> list[str]:
    """Resolve conventional and pip-split CUDA development headers.

    A reproducible compiler overlay may intentionally contain only the CUDA
    compiler and core runtime headers.  PyTorch's pip CUDA packages then own
    library-specific headers such as ``cusparse.h``.  Keep the overlay first,
    but always append the installed ``nvidia/*/include`` directories so a
    partial overlay cannot hide the complementary development packages.
    """

    home = Path(cuda_home)
    candidates = [home / "include"]
    if include_target:
        candidates.append(home / "targets" / "x86_64-linux" / "include")
    site_packages = Path(torch.__file__).resolve().parent.parent
    nvidia_packages = site_packages / "nvidia"
    if nvidia_packages.is_dir():
        candidates.extend(sorted(nvidia_packages.glob("*/include")))
    resolved: list[str] = []
    for candidate in candidates:
        value = str(candidate)
        if candidate.is_dir() and value not in resolved:
            resolved.append(value)
    return resolved


def _op_registered(namespace: str) -> bool:
    try:
        getattr(getattr(torch.ops, namespace), "forward")
    except (AttributeError, RuntimeError):
        return False
    return True


def _validate_runtime() -> None:
    if os.name == "nt" or not torch.cuda.is_available():
        raise RuntimeError(
            "train_temp CUDA backend requires Linux with an available CUDA GPU"
        )
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) < (8, 0):
        raise RuntimeError(
            "train_temp BF16 CUDA backend requires compute capability sm_80 or newer; "
            f"found sm_{major}{minor}"
        )


def _cuda_arch_list() -> str:
    """Return the active device architecture for one isolated JIT build."""

    major, minor = torch.cuda.get_device_capability()
    return f"{major}.{minor}"


def _resolve_cuda_home(cpp_extension: Any) -> Path | None:
    candidates = [
        os.environ.get("CUDA_HOME"),
        f"/usr/local/cuda-{torch.version.cuda}" if torch.version.cuda else None,
        getattr(cpp_extension, "CUDA_HOME", None),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if (path / "bin" / "nvcc").is_file():
            cpp_extension.CUDA_HOME = str(path)
            return path
    return None


def _build_verbose(verbose: bool | None) -> bool:
    if verbose is not None:
        return bool(verbose)
    return os.environ.get("RWKV7_TRAIN_TEMP_VERBOSE", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _build_recurrent_operator(
    cpp_extension: Any, cuda_home: Path, *, verbose: bool
) -> None:
    if _op_registered("rwkv7_clampw_v3"):
        return
    root = _source_root()
    cpp_extension.load(
        name="rwkv7_kernels_clampw_v3",
        sources=[
            str(root / "rwkv7_clampw_v3_for_h100.cu"),
            str(root / "rwkv7_clampw_v3.cpp"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            *_RECURRENT_CUDA_FLAGS,
            f"-D_N_={TRAIN_TEMP_HEAD_SIZE}",
            f"-D_CHUNK_LEN_={TRAIN_TEMP_CHUNK_LEN}",
        ],
        extra_include_paths=_cuda_include_paths(cuda_home, include_target=True),
        is_python_module=False,
        verbose=verbose,
    )


def _build_mix6_operator(cpp_extension: Any, cuda_home: Path, *, verbose: bool) -> None:
    """Build only the Mix6 leaf used by the accepted whole-model runtime."""

    namespace = "rwkv7_tmix_mix6_bf16_v5"
    if _op_registered(namespace):
        return
    root = _source_root()
    cpp_extension.load(
        name=f"rwkv7_kernels_{namespace}",
        sources=[str(root / filename) for filename in _OP_SOURCES[namespace]],
        extra_cflags=["-O3"],
        extra_cuda_cflags=list(_COMMON_CUDA_FLAGS),
        extra_include_paths=_cuda_include_paths(cuda_home),
        is_python_module=False,
        verbose=verbose,
    )


def load_mix6_training_cuda_extension(*, verbose: bool | None = None) -> None:
    """Build only the Mix6 training leaf once.

    The accepted whole-model path uses Mix6 plus the recurrent ClampW leaf.
    Keeping this loader separate avoids compiling four experimental fused
    gates and the L2Wrap loss before the first ordinary HF training step.
    """

    global _MIX6_LOADED, _MIX6_LOAD_ERROR
    _validate_runtime()
    namespace = "rwkv7_tmix_mix6_bf16_v5"
    if _MIX6_LOADED or _op_registered(namespace):
        _MIX6_LOADED = True
        return
    if _MIX6_LOAD_ERROR is not None:
        raise RuntimeError(
            "Mix6 training CUDA extension previously failed to load"
        ) from _MIX6_LOAD_ERROR
    with _LOAD_LOCK:
        if _MIX6_LOADED or _op_registered(namespace):
            _MIX6_LOADED = True
            return
        try:
            with cuda_extension_build_environment(arch_list=_cuda_arch_list()):
                from torch.utils import cpp_extension

                cuda_home = _resolve_cuda_home(cpp_extension)
                if cuda_home is None:
                    raise RuntimeError(
                        "Mix6 training CUDA JIT requires a local CUDA toolkit; "
                        "set CUDA_HOME to the toolkit matching the PyTorch CUDA build"
                    )
                _build_mix6_operator(
                    cpp_extension,
                    cuda_home,
                    verbose=_build_verbose(verbose),
                )
            if not _op_registered(namespace):
                raise RuntimeError(
                    "Mix6 training extension did not register rwkv7_tmix_mix6_bf16_v5"
                )
            _MIX6_LOADED = True
        except BaseException as exc:
            _MIX6_LOAD_ERROR = exc
            raise RuntimeError(
                f"Mix6 training CUDA extension failed to load: {exc}"
            ) from exc


def load_recurrent_training_cuda_extension(*, verbose: bool | None = None) -> None:
    """Build only the canonical recurrent training operator once."""

    global _RECURRENT_LOADED, _RECURRENT_LOAD_ERROR
    _validate_runtime()
    if _RECURRENT_LOADED or _op_registered("rwkv7_clampw_v3"):
        _RECURRENT_LOADED = True
        return
    if _RECURRENT_LOAD_ERROR is not None:
        raise RuntimeError(
            "recurrent training CUDA extension previously failed to load"
        ) from _RECURRENT_LOAD_ERROR
    with _LOAD_LOCK:
        if _RECURRENT_LOADED or _op_registered("rwkv7_clampw_v3"):
            _RECURRENT_LOADED = True
            return
        try:
            with cuda_extension_build_environment(arch_list=_cuda_arch_list()):
                from torch.utils import cpp_extension

                cuda_home = _resolve_cuda_home(cpp_extension)
                if cuda_home is None:
                    raise RuntimeError(
                        "recurrent training CUDA JIT requires a local CUDA toolkit; "
                        "set CUDA_HOME to the toolkit matching the PyTorch CUDA build"
                    )
                _build_recurrent_operator(
                    cpp_extension,
                    cuda_home,
                    verbose=_build_verbose(verbose),
                )
            if not _op_registered("rwkv7_clampw_v3"):
                raise RuntimeError(
                    "recurrent training extension did not register rwkv7_clampw_v3"
                )
            _RECURRENT_LOADED = True
        except BaseException as exc:
            _RECURRENT_LOAD_ERROR = exc
            raise RuntimeError(
                f"recurrent training CUDA extension failed to load: {exc}"
            ) from exc


def load_train_temp_cuda_extension(*, verbose: bool | None = None) -> None:
    """Build and load the vendored train_temp operators once."""

    global _L2WRAP_EXTENSION, _LOADED, _LOAD_ERROR, _MIX6_LOADED
    global _RECURRENT_LOADED
    _validate_runtime()
    if _LOADED:
        return
    if _LOAD_ERROR is not None:
        raise RuntimeError(
            "train_temp CUDA extension previously failed to load"
        ) from _LOAD_ERROR
    with _LOAD_LOCK:
        if _LOADED:
            return
        try:
            with cuda_extension_build_environment(arch_list=_cuda_arch_list()):
                from torch.utils import cpp_extension

                cuda_home = _resolve_cuda_home(cpp_extension)
                if cuda_home is None:
                    raise RuntimeError(
                        "train_temp CUDA JIT requires a local CUDA toolkit; set "
                        "CUDA_HOME to the toolkit matching the PyTorch CUDA build"
                    )
                verbose = _build_verbose(verbose)
                root = _source_root()
                include_paths = _cuda_include_paths(cuda_home)
                for namespace, filenames in _OP_SOURCES.items():
                    if _op_registered(namespace):
                        continue
                    cpp_extension.load(
                        name=f"rwkv7_kernels_{namespace}",
                        sources=[str(root / filename) for filename in filenames],
                        extra_cflags=["-O3"],
                        extra_cuda_cflags=list(_COMMON_CUDA_FLAGS),
                        extra_include_paths=include_paths,
                        is_python_module=False,
                        verbose=verbose,
                    )
                _build_recurrent_operator(cpp_extension, cuda_home, verbose=verbose)
                _RECURRENT_LOADED = True
                _L2WRAP_EXTENSION = cpp_extension.load(
                    name="rwkv7_kernels_l2wrap_ce_bf16_v2",
                    sources=[
                        str(root / "rwkv7_l2wrap_ce_bf16_v2.cpp"),
                        str(root / "rwkv7_l2wrap_ce_bf16_v2.cu"),
                    ],
                    extra_cflags=["-O3"],
                    extra_cuda_cflags=list(_COMMON_CUDA_FLAGS),
                    extra_include_paths=_cuda_include_paths(
                        cuda_home,
                        include_target=True,
                    ),
                    verbose=verbose,
                )
            missing = [
                namespace for namespace in _OP_SOURCES if not _op_registered(namespace)
            ]
            if not _op_registered("rwkv7_clampw_v3"):
                missing.append("rwkv7_clampw_v3")
            if missing:
                raise RuntimeError(
                    f"train_temp extension did not register required ops: {missing}"
                )
            _MIX6_LOADED = True
            _LOADED = True
        except BaseException as exc:
            _LOAD_ERROR = exc
            raise RuntimeError(
                f"train_temp CUDA extension failed to load: {exc}"
            ) from exc


def train_temp_cuda_available(*, build: bool = False) -> bool:
    """Return whether the backend is supported, optionally compiling it."""

    try:
        _validate_runtime()
        if build:
            load_train_temp_cuda_extension()
    except Exception:
        return False
    return True


def recurrent_training_cuda_available(*, build: bool = False) -> bool:
    """Return support, optionally compiling only the recurrent operator."""

    try:
        _validate_runtime()
        if build:
            load_recurrent_training_cuda_extension()
    except Exception:
        return False
    return True


def load_training_runtime_cuda_extensions(*, verbose: bool | None = None) -> None:
    """Load the minimal operator set used by whole-model HF training."""

    load_mix6_training_cuda_extension(verbose=verbose)
    load_recurrent_training_cuda_extension(verbose=verbose)


class _Mix6(torch.autograd.Function):
    # The fused CUDA backward is profitable once there are enough rows to
    # amortize its fixed launch cost. At the smallest accepted whole-model
    # shape (B1/T16), replaying the canonical PyTorch expression is both cheap
    # and measurably closer to the complete reference optimizer update.
    _NATIVE_BACKWARD_MIN_ROWS = 32

    @staticmethod
    def forward(ctx, x, x_r, x_w, x_k, x_v, x_a, x_g):
        saved = (x, x_r, x_w, x_k, x_v, x_a, x_g)
        inputs = tuple(value.contiguous() for value in saved)
        # Save the original inputs rather than the temporary contiguous views.
        # The CUDA backward makes its own contiguous views below, while retaining
        # the originals here keeps higher-order derivatives connected to callers.
        ctx.save_for_backward(*saved)
        return tuple(torch.ops.rwkv7_tmix_mix6_bf16_v5.forward(*inputs))

    @staticmethod
    def backward(ctx, grad_r, grad_w, grad_k, grad_v, grad_a, grad_g):
        x, *mixes = ctx.saved_tensors
        grads = tuple(
            (value if value is not None else torch.zeros_like(x)).contiguous()
            for value in (grad_r, grad_w, grad_k, grad_v, grad_a, grad_g)
        )
        create_graph = torch.is_grad_enabled()
        flattened_rows = int(x.shape[0]) * int(x.shape[1])
        if create_graph or flattened_rows < _Mix6._NATIVE_BACKWARD_MIN_ROWS:
            # CUDA custom operators do not define a double backward. The
            # smallest row count also stays on this exact expression: RTX 4080
            # full-model validation showed that it closes the last global
            # gradient tolerance miss while leaving the large-batch hot path on
            # the deterministic fused reduction below.
            with torch.enable_grad():
                references = tuple(
                    value
                    if value.requires_grad
                    else value.detach().requires_grad_(True)
                    for value in (x, *mixes)
                )
                x_ref, *mix_refs = references
                shifted = torch.cat(
                    (torch.zeros_like(x_ref[:, :1]), x_ref[:, :-1]), dim=1
                )
                delta = shifted - x_ref
                outputs = [x_ref + delta * mix.view(1, 1, -1) for mix in mix_refs]
                return tuple(
                    torch.autograd.grad(
                        outputs,
                        references,
                        grads,
                        create_graph=create_graph,
                    )
                )

        inputs = tuple(value.contiguous() for value in (x, *mixes))
        return tuple(
            torch.ops.rwkv7_tmix_mix6_bf16_v5.backward(
                *grads,
                *inputs,
            )
        )


class _KkPre(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, k_k, a, k_a):
        inputs = tuple(value.contiguous() for value in (k, k_k, a, k_a))
        outputs = torch.ops.rwkv7_tmix_kk_pre_bf16_v5.forward(
            *inputs, TRAIN_TEMP_HEAD_SIZE
        )
        ctx.save_for_backward(*inputs, outputs[3])
        return outputs[0], outputs[1], outputs[2]

    @staticmethod
    def backward(ctx, grad_k, grad_neg_kk, grad_kka):
        k, k_k, a, k_a, inv_d = ctx.saved_tensors
        return tuple(
            torch.ops.rwkv7_tmix_kk_pre_bf16_v5.backward(
                grad_k.contiguous(),
                grad_neg_kk.contiguous(),
                grad_kka.contiguous(),
                k,
                k_k,
                a,
                k_a,
                inv_d,
                TRAIN_TEMP_HEAD_SIZE,
            )
        )


class _LnxOutput(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, r, k, v, r_k, weight, bias, g):
        inputs = tuple(
            value.contiguous() for value in (x, r, k, v, r_k, weight, bias, g)
        )
        outputs = torch.ops.rwkv7_tmix_lnx_rkvres_xg_bf16_v1.forward(*inputs)
        ctx.save_for_backward(*inputs, outputs[1], outputs[2])
        return outputs[0]

    @staticmethod
    def backward(ctx, grad_output):
        x, r, k, v, r_k, weight, bias, g, mean, rstd = ctx.saved_tensors
        return tuple(
            torch.ops.rwkv7_tmix_lnx_rkvres_xg_bf16_v1.backward(
                grad_output.contiguous(), x, r, k, v, r_k, weight, bias, g, mean, rstd
            )
        )


class _AGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a0, a12):
        inputs = a0.contiguous(), a12.contiguous()
        ctx.save_for_backward(*inputs)
        return torch.ops.rwkv7_tmix_a_gate_bf16.forward(*inputs)

    @staticmethod
    def backward(ctx, grad_output):
        return tuple(
            torch.ops.rwkv7_tmix_a_gate_bf16.backward(
                grad_output.contiguous(), *ctx.saved_tensors
            )
        )


class _VResGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, v_first, v0, v12):
        inputs = tuple(value.contiguous() for value in (v, v_first, v0, v12))
        ctx.save_for_backward(*inputs)
        return torch.ops.rwkv7_tmix_vres_gate_bf16_v3.forward(*inputs)

    @staticmethod
    def backward(ctx, grad_output):
        return tuple(
            torch.ops.rwkv7_tmix_vres_gate_bf16_v3.backward(
                grad_output.contiguous(), *ctx.saved_tensors
            )
        )


def _recurrent_decay_reference(r, decay, k, v, a, b, initial_state):
    """Replay the canonical rank-one recurrence for autograd."""

    batch, tokens, heads, head_size = r.shape
    sample_outputs = []
    final_states = []
    for batch_idx in range(batch):
        state = initial_state[batch_idx : batch_idx + 1]
        token_outputs = []
        for token_idx in range(tokens):
            r_t = r[batch_idx : batch_idx + 1, token_idx]
            decay_t = decay[batch_idx : batch_idx + 1, token_idx]
            k_t = k[batch_idx : batch_idx + 1, token_idx]
            v_t = v[batch_idx : batch_idx + 1, token_idx]
            a_t = a[batch_idx : batch_idx + 1, token_idx]
            b_t = b[batch_idx : batch_idx + 1, token_idx]
            state_vk = state.transpose(-1, -2)
            vk = v_t.unsqueeze(-1) @ k_t.unsqueeze(-2)
            state_a = torch.zeros_like(state_vk[..., 0])
            for key_idx in range(head_size):
                state_a = state_a + (
                    state_vk[..., key_idx] * a_t[..., key_idx].unsqueeze(-1).float()
                )
            state_vk = state_vk * decay_t.unsqueeze(-2) + (
                state_a.unsqueeze(-1) * b_t.unsqueeze(-2).float() + vk.float()
            )
            state = state_vk.transpose(-1, -2)
            state_output = state_vk.to(dtype=r_t.dtype)
            output_fp32 = torch.zeros(
                state_output.shape[:-1],
                dtype=torch.float32,
                device=r.device,
            )
            for key_idx in range(head_size):
                output_fp32 = output_fp32 + (
                    state_output[..., key_idx].float()
                    * r_t[..., key_idx].unsqueeze(-1).float()
                )
            token_outputs.append(output_fp32.to(dtype=v.dtype))
        sample_outputs.append(torch.stack(token_outputs, dim=1))
        final_states.append(state)
    return torch.cat(sample_outputs, dim=0), torch.cat(final_states, dim=0)


class _ClampW(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, decay, k, v, a, b, initial_state):
        batch, tokens, heads, head_size = r.shape
        if head_size != TRAIN_TEMP_HEAD_SIZE or tokens % TRAIN_TEMP_CHUNK_LEN:
            raise ValueError(
                f"train_temp clampw requires head_size={TRAIN_TEMP_HEAD_SIZE} and "
                f"tokens divisible by {TRAIN_TEMP_CHUNK_LEN}; got {head_size=} {tokens=}"
            )
        if decay.dtype != torch.float32:
            raise TypeError(f"train_temp decay must be FP32, got {decay.dtype}")
        recurrent_inputs = tuple(value.contiguous() for value in (r, decay, k, v, a, b))
        output = torch.empty_like(v)
        state = torch.empty(
            batch,
            heads,
            tokens // TRAIN_TEMP_CHUNK_LEN,
            head_size,
            head_size,
            dtype=torch.float32,
            device=decay.device,
        )
        state_aux = torch.empty(
            batch, tokens, heads, head_size, dtype=torch.float32, device=decay.device
        )
        torch.ops.rwkv7_clampw_v3.forward(*recurrent_inputs, output, state, state_aux)
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(*recurrent_inputs, state, state_aux, initial_state)
        # The last CUDA checkpoint is the canonical [B,H,K,V] final state.
        # Returning it preserves the public recurrent operator contract.
        return output, state[:, :, -1]

    @staticmethod
    def backward(ctx, grad_output, grad_final_state):
        create_graph = torch.is_grad_enabled()
        r, decay, k, v, a, b, state, state_aux, initial_state = ctx.saved_tensors
        if grad_output is None:
            grad_output = torch.zeros_like(v)
        recurrent_inputs = (r, decay, k, v, a, b)
        recurrent_grads = [torch.empty_like(value) for value in recurrent_inputs]
        torch.ops.rwkv7_clampw_v3.backward(
            *recurrent_inputs,
            grad_output.contiguous(),
            state,
            state_aux,
            *recurrent_grads,
        )

        initial_state_grad = None
        if grad_final_state is not None:
            # The CUDA kernel computes the standard output-loss gradient.  A
            # caller that consumes the returned state adds only that state's
            # contribution through the canonical recurrence.
            with torch.enable_grad():
                replay_inputs = [
                    value.detach().requires_grad_(True) for value in recurrent_inputs
                ]
                replay_state = initial_state.detach().requires_grad_(True)
                _, replay_final_state = _recurrent_decay_reference(
                    *replay_inputs, replay_state
                )
                state_grads = torch.autograd.grad(
                    replay_final_state,
                    (*replay_inputs, replay_state),
                    grad_final_state.contiguous(),
                    create_graph=create_graph,
                    allow_unused=True,
                )
                state_grads = tuple(
                    torch.zeros_like(value) if gradient is None else gradient
                    for value, gradient in zip(
                        (*replay_inputs, replay_state), state_grads, strict=True
                    )
                )
            recurrent_grads = [
                native + state_contribution
                for native, state_contribution in zip(
                    recurrent_grads, state_grads[:-1], strict=True
                )
            ]
            initial_state_grad = state_grads[-1]

        if ctx.needs_input_grad[-1]:
            # Training normally starts from a detached zero state.  Stateful
            # callers still receive the exact output-to-state derivative.
            with torch.enable_grad():
                replay_state = initial_state.detach().requires_grad_(True)
                replay_output, _ = _recurrent_decay_reference(
                    *(value.detach() for value in recurrent_inputs), replay_state
                )
                output_state_grad = torch.autograd.grad(
                    replay_output,
                    replay_state,
                    grad_output.contiguous(),
                    create_graph=create_graph,
                )[0]
            initial_state_grad = (
                output_state_grad
                if initial_state_grad is None
                else initial_state_grad + output_state_grad
            )

        return (*recurrent_grads, initial_state_grad)


def rwkv7_training_recurrent(
    r: torch.Tensor,
    decay: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute the native training recurrence with zero initial state.

    The public HF adapter owns decay translation, cache layout and masking.
    This function owns only the dense BF16 recurrence and its autograd edge.
    Capability checks live in ``recurrent.training_factorized`` so a direct call
    never silently falls back to another implementation.
    """

    load_recurrent_training_cuda_extension()
    return _ClampW.apply(r, decay, k, v, a, b, initial_state)


class _CMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x_k, key_weight, value_weight):
        inputs = tuple(
            value.contiguous() for value in (x, x_k, key_weight, value_weight)
        )
        output, mixed, activation = torch.ops.rwkv7_cmix_bf16_v5.forward(*inputs)
        ctx.save_for_backward(*inputs, mixed, activation)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, x_k, key_weight, value_weight, mixed, activation = ctx.saved_tensors
        return tuple(
            torch.ops.rwkv7_cmix_bf16_v5.backward(
                grad_output.contiguous(),
                x,
                x_k,
                key_weight,
                value_weight,
                mixed,
                activation,
            )
        )


class _L2WrapCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets):
        assert _L2WRAP_EXTENSION is not None
        logits = logits.contiguous()
        targets = targets.contiguous()
        loss, lse, max_values, argmax = _L2WRAP_EXTENSION.forward(logits, targets)
        ctx.save_for_backward(logits, targets.reshape(-1), lse, max_values, argmax)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        assert _L2WRAP_EXTENSION is not None
        logits, targets, lse, max_values, argmax = ctx.saved_tensors
        grad_logits = _L2WRAP_EXTENSION.backward(
            grad_output.contiguous().float(), logits, targets, lse, max_values, argmax
        )
        return grad_logits, None


def train_temp_fused_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """Run the exact train_temp fused FP32 CE plus L2Wrap gradient."""

    load_train_temp_cuda_extension()
    return _L2WrapCrossEntropy.apply(logits, targets)


def train_temp_causal_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Apply the fused train_temp loss to standard causal-LM logits and labels.

    Unlike :func:`train_temp_fused_cross_entropy`, this helper performs the
    next-token shift expected by Hugging Face causal-language-model batches.
    The current CUDA kernel accepts dense int64 labels only; padding and the
    usual ``-100`` ignore index are intentionally rejected by its contract.
    """

    if logits.ndim != 3:
        raise ValueError(
            f"logits must have shape [batch, tokens, vocab], got {tuple(logits.shape)}"
        )
    if labels.ndim != 2:
        raise ValueError(
            f"labels must have shape [batch, tokens], got {tuple(labels.shape)}"
        )
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            "logits and labels must share batch/token dimensions; got "
            f"{tuple(logits.shape[:2])} and {tuple(labels.shape)}"
        )
    if logits.shape[1] < 2:
        raise ValueError("causal train_temp loss requires at least two tokens")
    if labels.dtype != torch.long:
        raise TypeError(f"labels must be torch.int64, got {labels.dtype}")
    if labels.device != logits.device:
        raise ValueError(
            "logits and labels must share a device, got "
            f"{logits.device} and {labels.device}"
        )
    if bool(torch.any((labels < 0) | (labels >= logits.shape[-1])).item()):
        raise ValueError(
            "train_temp CUDA loss requires dense labels in [0, vocab_size); "
            "-100 is unsupported"
        )
    return train_temp_fused_cross_entropy(
        logits[:, :-1].contiguous(), labels[:, 1:].contiguous()
    )


def _train_temp_attention_forward(
    self, hidden_states, v_first, *, native_lora_math: bool
):
    """Run fused TMix while preserving each backend's LoRA activation ownership."""

    if (
        hidden_states.dtype != torch.bfloat16
        or hidden_states.shape[1] % TRAIN_TEMP_CHUNK_LEN
    ):
        raise ValueError(
            "train_temp CUDA backend requires BF16 and sequence length divisible by "
            f"{TRAIN_TEMP_CHUNK_LEN}; got {hidden_states.dtype} and T={hidden_states.shape[1]}"
        )
    xr, xw, xk, xv, xa, xg = _Mix6.apply(
        hidden_states,
        self.x_r.reshape(-1),
        self.x_w.reshape(-1),
        self.x_k.reshape(-1),
        self.x_v.reshape(-1),
        self.x_a.reshape(-1),
        self.x_g.reshape(-1),
    )
    # This private whole-model runtime owns the complete TMix calculation.
    # Calling RWKV7Linear.forward here would recursively enter the optional
    # training-leaf dispatcher, where small LoRA projections are intentionally
    # unsupported even though the enclosing native request is valid.
    r = module_linear(self.r_proj, xr)
    if native_lora_math:
        decay_projection = self.w_lora.lora[2]
        # The clean HF model deliberately keeps the official w0 bias and decay
        # transform in FP32. The adapted private kernel accepts that FP32 decay
        # directly, preserving the public parameter and its autograd edge.
        raw_decay = low_rank_projection(
            self.w_lora,
            xw,
            activation=torch.tanh,
            include_bias=False,
        )
        decay_bias = decay_projection.bias
        decay_logits = (
            raw_decay.float()
            if decay_bias is None
            else raw_decay.float() + decay_bias.float()
        )
        decay = torch.exp(-0.6065306597 * torch.sigmoid(decay_logits))
    else:
        decay_logits = low_rank_projection(self.w_lora, xw).float()
        decay = torch.exp(-0.6065306597 * torch.sigmoid(decay_logits))
    k = module_linear(self.k_proj, xk)
    v = module_linear(self.v_proj, xv)
    if self.layer_idx == 0:
        v_first = v
    else:
        # Keep the small value-residual gate in canonical HF math. The
        # historical fused gate changes the BF16 rounding point before the
        # sigmoid, and the difference compounds across layers during training.
        value_mix = torch.sigmoid(low_rank_projection(self.v_lora, xv))
        v = v + (v_first - v) * value_mix
    # The preparation gates are a tiny fraction of layer time but sit on every
    # recurrent update. Preserve their canonical BF16 rounding points instead
    # of compounding the historical fused-gate approximation over all layers.
    a = torch.sigmoid(low_rank_projection(self.a_lora, xa))
    g = (
        low_rank_projection(self.g_lora, xg, activation=torch.sigmoid)
        if native_lora_math
        else low_rank_projection(self.g_lora, xg)
    )
    batch, tokens, _ = r.shape
    heads = int(self.num_heads)
    head_dim = int(self.head_dim)
    head_v_dim = int(getattr(self, "head_v_dim", head_dim))
    if head_dim != TRAIN_TEMP_HEAD_SIZE or head_v_dim != TRAIN_TEMP_HEAD_SIZE:
        raise ValueError(
            "train_temp CUDA backend currently requires K/V head dimensions of 64"
        )
    weighted_key = k * self.k_k.view(1, 1, -1)
    normalized_key = F.normalize(
        weighted_key.view(batch, tokens, heads, head_dim), p=2, dim=-1
    ).view(batch, tokens, -1)
    k = k * (1 + (a - 1) * self.k_a.view(1, 1, -1))
    neg_kk = -normalized_key
    kka = normalized_key * a
    initial_state = torch.zeros(
        batch,
        heads,
        TRAIN_TEMP_HEAD_SIZE,
        TRAIN_TEMP_HEAD_SIZE,
        dtype=torch.float32,
        device=r.device,
    )
    values, _final_state = _ClampW.apply(
        r.reshape(batch, tokens, heads, TRAIN_TEMP_HEAD_SIZE),
        decay.reshape(batch, tokens, heads, TRAIN_TEMP_HEAD_SIZE),
        k.reshape(batch, tokens, heads, TRAIN_TEMP_HEAD_SIZE),
        v.reshape(batch, tokens, heads, TRAIN_TEMP_HEAD_SIZE),
        neg_kk.reshape(batch, tokens, heads, TRAIN_TEMP_HEAD_SIZE),
        kka.reshape(batch, tokens, heads, TRAIN_TEMP_HEAD_SIZE),
        initial_state,
    )
    values = values.reshape(batch, tokens, -1)
    # The fused LNX leaf remains available for throughput experiments, but its
    # BF16 reduction order compounds across deep training graphs. Keep the
    # accepted training route on the exact HF GroupNorm/direct/gate expression
    # while the recurrent scan—the expensive sequential part—stays native.
    normalized = F.group_norm(
        values.reshape(batch * tokens, -1),
        num_groups=heads,
        weight=self.g_norm.weight,
        bias=self.g_norm.bias,
        eps=self.g_norm.eps,
    ).reshape(batch, tokens, heads, head_dim)
    direct = (
        r.reshape(batch, tokens, heads, head_dim)
        * k.reshape(batch, tokens, heads, head_dim)
        * self.r_k.reshape(1, 1, heads, head_dim)
    ).sum(dim=-1, keepdim=True) * v.reshape(batch, tokens, heads, head_dim)
    values = (normalized + direct).reshape(batch, tokens, -1) * g
    return module_linear(self.o_proj, values), v_first


# Whole-model/layer dispatch intentionally lives in training_runtime.py.  This
# module exposes only leaf autograd operators and their lazy extension loader;
# it never replaces a Hugging Face or FLA module's forward method.


__all__ = [
    "TRAIN_TEMP_CHUNK_LEN",
    "TRAIN_TEMP_HEAD_SIZE",
    "TRAIN_TEMP_SOURCE_COMMIT",
    "load_mix6_training_cuda_extension",
    "load_recurrent_training_cuda_extension",
    "load_train_temp_cuda_extension",
    "load_training_runtime_cuda_extensions",
    "recurrent_training_cuda_available",
    "rwkv7_training_recurrent",
    "train_temp_causal_cross_entropy",
    "train_temp_cuda_available",
    "train_temp_fused_cross_entropy",
]
