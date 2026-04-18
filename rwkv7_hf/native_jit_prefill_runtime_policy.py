# coding=utf-8
"""Native prefill runtime feature gates and launch policy.

The implementation is isolated from execution math. The stable native_jit
facade refreshes the small dependency set before each compatibility call, so
existing policy and optional-kernel monkeypatch points remain effective.
"""
from __future__ import annotations

import os

import torch


_POLICY_NAMES = {'_native_prefill_state_prep_layers', '_native_prefill_shift_mix_num_warps', '_native_prefill_dplr_scan_enabled', '_native_prefill_fused_sequence_ffn_enabled', '_native_prefill_fused_wavg_lora_requested', '_native_prefill_self_chunk_size', '_native_prefill_shift_mix_launch_profile', '_native_prefill_policy_model_shape_selected', '_native_prefill_sequence_ffn_blocks', '_native_prefill_fused_state_scan_enabled', '_native_prefill_fp16_accum_ffn_key_layers', '_native_prefill_fp16_recurrent_requested', '_native_prefill_fused_residual_gemm_enabled', '_native_prefill_attn_shift_mix_block_size', '_native_prefill_fused_wavg_lora_max_m', '_native_prefill_default_scan_block_m', '_native_prefill_stacked_rkv_enabled', '_native_prefill_ffn_shift_mix_block_size', '_native_prefill_fused_clampw_scan_enabled', '_native_prefill_self_chunk_enabled', '_native_prefill_fused_wavg_lora_enabled', '_native_prefill_scan_num_warps', '_native_prefill_sequence_ffn_launch', '_native_prefill_scan_block_m', '_native_prefill_fused_shift_mix_enabled', '_native_prefill_fused_wavg_lora_blocks', '_native_prefill_fused_output_project_block_m', '_native_prefill_model_shape_selected', '_native_prefill_fused_output_enabled', '_native_prefill_dplr_chunk_size', '_native_prefill_fused_state_scan_max_batch', '_native_prefill_fp16_recurrent_enabled', '_native_prefill_self_chunk_safe_gate', '_native_prefill_fused_state_prep_enabled', '_native_prefill_state_prep_w_dtype', '_native_prefill_shift_mix_layers', '_native_prefill_fused_scan_enabled', '_native_prefill_fp16_accum_ffn_key_enabled', '_native_prefill_global_fp16_accum_enabled', '_native_prefill_block_fp16_accum_enabled', '_native_prefill_fused_output_project_enabled', '_native_prefill_fused_scan_output_enabled', '_native_prefill_self_chunk_h_tiles'}
_OWNED_NAMES = _POLICY_NAMES | {"bind_runtime"}
_RUNTIME_NAMES = ('_is_rtx_model_name', '_kernel_policy', '_prefill_model_shape_selected_impl', '_prefill_policy_model_shape_selected_impl', '_prefill_self_chunk_h_tiles_impl', '_prefill_self_chunk_shape_eligible', '_prefill_self_chunk_size_impl', 'dplr_chunk_scan', 'env_flag', 'env_int', 'fused_attn_output_prepare', 'fused_attn_output_prepare_available', 'fused_attn_output_project', 'fused_attn_output_project_available', 'fused_attn_shift_mix', 'fused_attn_shift_mix_available', 'fused_prefill_state_prep', 'fused_prefill_state_prep_available', 'fused_recurrent_scan', 'fused_recurrent_scan_available', 'fused_recurrent_scan_clampw', 'fused_recurrent_scan_clampw_available', 'fused_recurrent_scan_output_prepare', 'fused_recurrent_scan_output_prepare_available', 'fused_recurrent_scan_state_prep', 'fused_recurrent_scan_state_prep_available', 'fused_sequence_ffn', 'fused_sequence_ffn_available', 'fused_wavg_lora', 'fused_wavg_lora_available', 'native_fp16_sequence', 'self_chunk_rwkv7', 'self_chunk_rwkv7_available')


def bind_runtime(runtime: dict[str, object]) -> None:
    for name in _RUNTIME_NAMES:
        if name in runtime and name not in _OWNED_NAMES:
            globals()[name] = runtime[name]
    implementations = globals().get("_IMPLEMENTATIONS", {})
    for name in _POLICY_NAMES:
        implementation = implementations.get(name)
        facade_value = runtime.get(name)
        if implementation is None or facade_value is None:
            continue
        if getattr(facade_value, "__wrapped__", None) is implementation:
            globals()[name] = implementation
        else:
            # Preserve overrides patched on the historical facade so sibling
            # policy gates observe them exactly as before the split.
            globals()[name] = facade_value


def _native_prefill_fused_scan_enabled(
    batch_size: int | None = None,
    prompt_tokens: int | None = None,
    hidden_size: int | None = None,
    num_layers: int | None = None,
) -> bool:
    """Runtime switch for the experimental native prefill recurrent scan."""

    policy = _kernel_policy()
    flag_name = "RWKV7_NATIVE_PREFILL_FUSED_SCAN"
    shape_name = "RWKV7_NATIVE_PREFILL_SCAN_MODEL_SHAPES"
    explicit_flag = os.environ.get(flag_name)
    if not env_flag(flag_name, bool(getattr(policy, "fused_prefill_scan", False))):
        return False
    if (explicit_flag is None or os.environ.get(shape_name) is not None) and not _native_prefill_model_shape_selected(
        shape_name,
        "prefill_scan_model_shapes",
        batch_size,
        prompt_tokens,
        hidden_size,
        num_layers,
    ):
        return False
    if fused_recurrent_scan is None or fused_recurrent_scan_available is None:
        return False
    try:
        return bool(fused_recurrent_scan_available())
    except Exception:
        return False

def _native_prefill_fp16_recurrent_requested() -> bool:
    """Select official-precision sequence recurrence without changing defaults."""

    policy = _kernel_policy()
    return env_flag(
        "RWKV7_NATIVE_PREFILL_FP16_RECURRENT",
        bool(getattr(policy, "prefill_fp16_recurrent", False)),
    )

def _native_prefill_fp16_recurrent_enabled(state: torch.Tensor) -> bool:
    return bool(
        _native_prefill_fp16_recurrent_requested()
        and native_fp16_sequence is not None
        and state.dtype == torch.float16
        and int(state.shape[-1]) == 64
    )


def _native_prefill_global_fp16_accum_enabled(
    batch_size: int,
    prompt_tokens: int,
    hidden_size: int,
    num_layers: int,
    dtype: torch.dtype,
) -> bool:
    """Select exact-card full-prefill fp16 GEMM accumulation routes."""

    if dtype != torch.float16:
        return False
    matmul = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
    if matmul is None or not hasattr(matmul, "allow_fp16_accumulation"):
        return False
    try:
        visible_cuda_devices = int(torch.cuda.device_count())
    except Exception:
        visible_cuda_devices = 1
    if visible_cuda_devices > 1 and not env_flag(
        "RWKV7_NATIVE_PREFILL_FP16_ACCUM_MULTI_GPU",
        False,
    ):
        return False

    policy = _kernel_policy()
    raw_shapes = os.environ.get(
        "RWKV7_NATIVE_PREFILL_GLOBAL_FP16_ACCUM_MODEL_SHAPES"
    )
    if raw_shapes is None:
        model_shapes = {
            tuple(int(value) for value in shape)
            for shape in getattr(
                policy,
                "prefill_global_fp16_accum_model_shapes",
                (),
            )
            if len(shape) == 4
        }
    else:
        model_shapes = set()
        try:
            for item in raw_shapes.replace(",", " ").split():
                values = tuple(int(value) for value in item.lower().split("x"))
                if len(values) != 4 or any(value <= 0 for value in values):
                    raise ValueError
                model_shapes.add(values)
        except ValueError as exc:
            raise ValueError(
                "RWKV7_NATIVE_PREFILL_GLOBAL_FP16_ACCUM_MODEL_SHAPES must "
                "contain HxLxBxT tuples"
            ) from exc
    target = (
        int(hidden_size),
        int(num_layers),
        int(batch_size),
        int(prompt_tokens),
    )
    selected = target in model_shapes
    return bool(
        selected
        and env_flag(
            "RWKV7_NATIVE_PREFILL_GLOBAL_FP16_ACCUM",
            selected,
        )
    )


def _native_prefill_block_fp16_accum_enabled(
    batch_size: int,
    prompt_tokens: int,
    hidden_size: int,
    num_layers: int,
    dtype: torch.dtype,
) -> bool:
    """Select block-only FP16 accumulation with an FP32-accumulation head."""

    if dtype != torch.float16:
        return False
    matmul = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
    if matmul is None or not hasattr(matmul, "allow_fp16_accumulation"):
        return False
    try:
        visible_cuda_devices = int(torch.cuda.device_count())
    except Exception:
        visible_cuda_devices = 1
    if visible_cuda_devices > 1 and not env_flag(
        "RWKV7_NATIVE_PREFILL_FP16_ACCUM_MULTI_GPU",
        False,
    ):
        return False

    policy = _kernel_policy()
    raw_shapes = os.environ.get(
        "RWKV7_NATIVE_PREFILL_BLOCK_FP16_ACCUM_MODEL_SHAPES"
    )
    if raw_shapes is None:
        model_shapes = {
            tuple(int(value) for value in shape)
            for shape in getattr(
                policy,
                "prefill_block_fp16_accum_model_shapes",
                (),
            )
            if len(shape) == 4
        }
    else:
        model_shapes = set()
        try:
            for item in raw_shapes.replace(",", " ").split():
                values = tuple(int(value) for value in item.lower().split("x"))
                if len(values) != 4 or any(value <= 0 for value in values):
                    raise ValueError
                model_shapes.add(values)
        except ValueError as exc:
            raise ValueError(
                "RWKV7_NATIVE_PREFILL_BLOCK_FP16_ACCUM_MODEL_SHAPES must "
                "contain HxLxBxT tuples"
            ) from exc
    target = (
        int(hidden_size),
        int(num_layers),
        int(batch_size),
        int(prompt_tokens),
    )
    selected = target in model_shapes
    return bool(
        selected
        and env_flag(
            "RWKV7_NATIVE_PREFILL_BLOCK_FP16_ACCUM",
            selected,
        )
    )


def _native_prefill_self_chunk_enabled(
    tokens: int,
    head_dim: int,
    batch_size: int | None = None,
    hidden_size: int | None = None,
    num_layers: int | None = None,
) -> bool:
    """Select the vendored sequence-parallel DPLR forward for long prompts."""

    policy = _kernel_policy()
    if not env_flag(
        "RWKV7_NATIVE_PREFILL_SELF_CHUNK",
        bool(getattr(policy, "fused_prefill_self_chunk", False)),
    ):
        return False
    min_tokens = env_int(
        "RWKV7_NATIVE_PREFILL_SELF_CHUNK_MIN_TOKENS",
        int(getattr(policy, "prefill_self_chunk_min_tokens", 1024)),
        lower=16,
    )
    if not _prefill_self_chunk_shape_eligible(
        policy=policy,
        tokens=tokens,
        head_dim=head_dim,
        batch_size=batch_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        min_tokens=min_tokens,
        raw_model_shapes=os.environ.get(
            "RWKV7_NATIVE_PREFILL_SELF_CHUNK_MODEL_SHAPES"
        ),
    ):
        return False
    if self_chunk_rwkv7 is None or self_chunk_rwkv7_available is None:
        return False
    try:
        return bool(self_chunk_rwkv7_available())
    except Exception:
        return False

def _native_prefill_self_chunk_size(batch_size: int, tokens: int | None = None) -> int:
    """Return the exact-card sequence chunk size."""

    return _prefill_self_chunk_size_impl(
        policy=_kernel_policy(),
        batch_size=batch_size,
        tokens=tokens,
        env_int_fn=env_int,
    )

def _native_prefill_self_chunk_h_tiles(
    batch_size: int,
    tokens: int,
) -> tuple[int, int] | None:
    """Return an exact-shape tile override from the centralized card policy."""

    return _prefill_self_chunk_h_tiles_impl(
        policy=_kernel_policy(),
        batch_size=batch_size,
        tokens=tokens,
    )

def _native_prefill_self_chunk_safe_gate() -> bool:
    """Select the numerically conservative tensor-core intra-chunk kernel."""

    return env_flag("RWKV7_NATIVE_PREFILL_SELF_CHUNK_SAFE_GATE", True)

def _native_prefill_dplr_scan_enabled() -> bool:
    """Runtime switch for the correctness-first DPLR/chunked prefill scan."""

    if not env_flag("RWKV7_NATIVE_PREFILL_DPLR_SCAN", False):
        return False
    return dplr_chunk_scan is not None

def _native_prefill_fused_residual_gemm_enabled() -> bool:
    """Use GEMM beta=1 epilogues for the two residual projections."""

    policy = _kernel_policy()
    return env_flag(
        "RWKV7_NATIVE_PREFILL_FUSED_RESIDUAL_GEMM",
        bool(getattr(policy, "fused_prefill_residual_gemm", False)),
    )

def _native_prefill_dplr_chunk_size() -> int:
    """Chunk length for the pure-torch DPLR/chunked prefill reference path."""

    return env_int("RWKV7_NATIVE_PREFILL_DPLR_CHUNK_SIZE", 64, lower=1, upper=4096)

def _native_prefill_fused_clampw_scan_enabled(
    batch_size: int | None = None,
    prompt_tokens: int | None = None,
    hidden_size: int | None = None,
    num_layers: int | None = None,
) -> bool:
    """Runtime switch for raw-W clampw native prefill recurrent scan."""

    env_name = "RWKV7_NATIVE_PREFILL_FUSED_CLAMPW_SCAN"
    raw_enabled = os.environ.get(env_name)
    if raw_enabled is not None:
        if not env_flag(env_name, False):
            return False
    else:
        policy = _kernel_policy()
        exact_model_shape = (
            (int(hidden_size), int(num_layers), int(batch_size), int(prompt_tokens))
            if None not in (hidden_size, num_layers, batch_size, prompt_tokens)
            else None
        )
        model_shapes = {
            tuple(int(v) for v in shape)
            for shape in getattr(policy, "prefill_clampw_scan_model_shapes", ())
            if len(shape) == 4
        }
        if not (
            bool(getattr(policy, "fused_prefill_clampw_scan", False))
            or exact_model_shape in model_shapes
        ):
            return False
    if not _native_prefill_fused_scan_enabled(
        batch_size,
        prompt_tokens,
        hidden_size,
        num_layers,
    ):
        return False
    if fused_recurrent_scan_clampw is None or fused_recurrent_scan_clampw_available is None:
        return False
    try:
        return bool(fused_recurrent_scan_clampw_available())
    except Exception:
        return False

def _native_prefill_fused_scan_output_enabled() -> bool:
    """Runtime switch for fused prefill scan plus attention output prep."""

    if not env_flag("RWKV7_NATIVE_PREFILL_FUSED_SCAN_OUTPUT", False):
        return False
    if fused_recurrent_scan_output_prepare is None or fused_recurrent_scan_output_prepare_available is None:
        return False
    try:
        return bool(fused_recurrent_scan_output_prepare_available())
    except Exception:
        return False

def _native_prefill_default_scan_block_m(
    head_dim: int,
    batch_size: int | None = None,
    tokens: int | None = None,
    hidden_size: int | None = None,
) -> int:
    """Architecture-aware default row tile for optional prefill scans."""

    head_dim = int(head_dim)
    policy = _kernel_policy()
    if hidden_size is not None and batch_size is not None and tokens is not None:
        for policy_hidden, policy_batch, policy_tokens, policy_block_m in getattr(
            policy,
            "prefill_scan_block_m_model_shapes",
            (),
        ):
            if (int(hidden_size), int(batch_size), int(tokens)) == (
                int(policy_hidden),
                int(policy_batch),
                int(policy_tokens),
            ):
                return int(policy_block_m)
    if batch_size is not None and tokens is not None:
        for policy_batch, policy_tokens, policy_block_m in getattr(
            policy,
            "prefill_scan_block_m_shapes",
            (),
        ):
            if (int(batch_size), int(tokens)) == (int(policy_batch), int(policy_tokens)):
                return int(policy_block_m)
    policy_value = getattr(policy, "prefill_scan_block_m", None)
    if policy_value is not None:
        if batch_size is not None and int(batch_size) >= 4:
            batch_value = getattr(policy, "prefill_scan_block_m_b4", None)
            if batch_value is not None:
                return int(batch_value)
        if batch_size is not None and int(batch_size) >= 2:
            batch_value = getattr(policy, "prefill_scan_block_m_b2", None)
            if batch_value is not None:
                return int(batch_value)
        return int(policy_value)
    if head_dim == 64 and torch.cuda.is_available():
        try:
            major, minor = torch.cuda.get_device_capability()
        except Exception:
            major, minor = 0, 0
        if (int(major), int(minor)) == (7, 0):
            return 32 if batch_size is not None and int(batch_size) >= 4 else 16
        if (int(major), int(minor)) == (8, 9):
            try:
                name = str(torch.cuda.get_device_name()).lower()
            except Exception:
                name = ""
            if _is_rtx_model_name(name, "4090"):
                if (
                    batch_size is not None
                    and int(batch_size) >= 8
                    and tokens is not None
                    and int(tokens) == 128
                ):
                    return 32
                if (
                    batch_size is not None
                    and int(batch_size) >= 8
                    and tokens is not None
                    and int(tokens) >= 512
                    and hidden_size is not None
                    and int(hidden_size) == 2048
                ):
                    return 32
                return 8 if batch_size is not None and int(batch_size) >= 2 else 4
        if int(major) >= 12:
            batch_size = 1 if batch_size is None else int(batch_size)
            if batch_size <= 1:
                return 8
            if batch_size <= 2:
                return 16
            if batch_size <= 4:
                return 32
            return 64
    return head_dim

def _native_prefill_scan_block_m(
    head_dim: int,
    batch_size: int | None = None,
    tokens: int | None = None,
    hidden_size: int | None = None,
) -> int:
    """Row tile for optional recurrent scans; explicit env always wins."""

    return env_int(
        "RWKV7_NATIVE_PREFILL_SCAN_BLOCK_M",
        _native_prefill_default_scan_block_m(head_dim, batch_size, tokens, hidden_size),
        lower=1,
        upper=int(head_dim),
    )

def _native_prefill_scan_num_warps(head_dim: int, block_m: int | None = None) -> int:
    """Triton warp count for the optional native prefill recurrent scan."""

    if block_m is None:
        block_m = _native_prefill_scan_block_m(head_dim)
    policy = _kernel_policy()
    policy_value = getattr(policy, "prefill_scan_num_warps", None)
    if policy_value is not None:
        default = int(policy_value)
    else:
        is_blackwell = False
        if torch.cuda.is_available():
            try:
                major, _minor = torch.cuda.get_device_capability()
                is_blackwell = int(major) >= 12
            except Exception:
                pass
        if is_blackwell and int(head_dim) == 64:
            default = 4 if int(block_m) >= 64 else 1
        else:
            default = 4 if int(block_m) < int(head_dim) else 8
    value = env_int("RWKV7_NATIVE_PREFILL_SCAN_NUM_WARPS", default, lower=1, upper=8)
    if value not in {1, 2, 4, 8}:
        raise ValueError(f"RWKV7_NATIVE_PREFILL_SCAN_NUM_WARPS must be one of 1, 2, 4, or 8; got {value}")
    return value

def _native_prefill_model_shape_selected(
    env_name: str,
    policy_name: str,
    batch_size: int | None,
    prompt_tokens: int | None,
    hidden_size: int | None,
    num_layers: int | None,
) -> bool:
    """Apply an optional exact model-shape restriction to a prefill route."""

    return _prefill_model_shape_selected_impl(
        policy=_kernel_policy(),
        env_name=env_name,
        policy_name=policy_name,
        raw=os.environ.get(env_name),
        batch_size=batch_size,
        prompt_tokens=prompt_tokens,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )

def _native_prefill_policy_model_shape_selected(
    policy_name: str,
    batch_size: int | None,
    prompt_tokens: int | None,
    hidden_size: int | None,
    num_layers: int | None,
) -> bool:
    """Return whether an exact shape is explicitly promoted by policy."""

    return _prefill_policy_model_shape_selected_impl(
        policy=_kernel_policy(),
        policy_name=policy_name,
        batch_size=batch_size,
        prompt_tokens=prompt_tokens,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )

def _native_prefill_fused_shift_mix_enabled(
    batch_size: int | None = None,
    prompt_tokens: int | None = None,
    hidden_size: int | None = None,
    num_layers: int | None = None,
) -> bool:
    """Runtime switch for prefill attention shift-mix fusion telemetry."""

    policy = _kernel_policy()
    if not env_flag(
        "RWKV7_NATIVE_PREFILL_FUSED_SHIFT_MIX",
        bool(getattr(policy, "fused_prefill_shift_mix", False)),
    ):
        return False
    if not _native_prefill_model_shape_selected(
        "RWKV7_NATIVE_PREFILL_SHIFT_MIX_MODEL_SHAPES",
        "prefill_shift_mix_model_shapes",
        batch_size,
        prompt_tokens,
        hidden_size,
        num_layers,
    ):
        return False
    if fused_attn_shift_mix is None or fused_attn_shift_mix_available is None:
        return False
    try:
        return bool(fused_attn_shift_mix_available())
    except Exception:
        return False

def _native_prefill_shift_mix_layers(
    batch_size: int,
    prompt_tokens: int,
    num_layers: int,
) -> set[int] | None:
    """Return selected shift-mix layers, or ``None`` for every layer."""

    specific = f"RWKV7_NATIVE_PREFILL_SHIFT_MIX_LAYERS_B{batch_size}_T{prompt_tokens}"
    raw = os.environ.get(
        specific,
        os.environ.get("RWKV7_NATIVE_PREFILL_SHIFT_MIX_LAYERS"),
    )
    if raw is None:
        return None
    selected: set[int] = set()
    try:
        for item in raw.replace(" ", ",").split(","):
            item = item.strip()
            if not item:
                continue
            if "-" in item:
                start, end = (int(value) for value in item.split("-", 1))
                if start < 0 or end < start:
                    raise ValueError
                selected.update(range(start, end + 1))
            else:
                value = int(item)
                if value < 0:
                    raise ValueError
                selected.add(value)
    except ValueError as exc:
        raise ValueError(
            f"{specific} must contain non-negative layers or inclusive ranges"
        ) from exc
    return {value for value in selected if value < int(num_layers)}

def _native_prefill_shift_mix_launch_profile(
    role: str,
    batch_size: int | None,
    prompt_tokens: int | None,
    hidden_size: int | None,
    num_layers: int | None,
) -> tuple[int, int] | None:
    if None in (batch_size, prompt_tokens, hidden_size, num_layers):
        return None
    policy_name = f"prefill_{role.lower()}_shift_mix_launch_profiles"
    target = (int(hidden_size), int(num_layers), int(batch_size), int(prompt_tokens))
    for profile in getattr(_kernel_policy(), policy_name, ()):
        if len(profile) == 6 and tuple(int(value) for value in profile[:4]) == target:
            return int(profile[4]), int(profile[5])
    return None

def _native_prefill_attn_shift_mix_block_size(
    strict_fp16_rounding: bool,
    batch_size: int | None = None,
    prompt_tokens: int | None = None,
    hidden_size: int | None = None,
    num_layers: int | None = None,
) -> int:
    """Choose a validated elementwise tile for sequence attention shift-mix."""

    profile = _native_prefill_shift_mix_launch_profile(
        "attn", batch_size, prompt_tokens, hidden_size, num_layers
    )
    default = profile[0] if profile is not None else (2048 if strict_fp16_rounding else 256)
    value = env_int(
        "RWKV7_NATIVE_PREFILL_ATTN_SHIFT_MIX_BLOCK_SIZE",
        default,
        lower=64,
        upper=2048,
    )
    if value not in (64, 128, 256, 512, 1024, 2048):
        raise ValueError(
            "RWKV7_NATIVE_PREFILL_ATTN_SHIFT_MIX_BLOCK_SIZE must be a power "
            "of two between 64 and 2048"
        )
    return value

def _native_prefill_shift_mix_num_warps(
    role: str,
    batch_size: int | None = None,
    prompt_tokens: int | None = None,
    hidden_size: int | None = None,
    num_layers: int | None = None,
) -> int:
    """Return a validated launch width for attention or FFN sequence mix."""

    role = role.strip().upper()
    if role not in ("ATTN", "FFN"):
        raise ValueError("shift-mix role must be ATTN or FFN")
    profile = _native_prefill_shift_mix_launch_profile(
        role.lower(), batch_size, prompt_tokens, hidden_size, num_layers
    )
    value = env_int(
        f"RWKV7_NATIVE_PREFILL_{role}_SHIFT_MIX_NUM_WARPS",
        profile[1] if profile is not None else 4,
        lower=1,
        upper=8,
    )
    if value not in (1, 2, 4, 8):
        raise ValueError(
            f"RWKV7_NATIVE_PREFILL_{role}_SHIFT_MIX_NUM_WARPS must be 1, 2, 4, or 8"
        )
    return value

def _native_prefill_ffn_shift_mix_block_size(
    batch_size: int | None = None,
    prompt_tokens: int | None = None,
    hidden_size: int | None = None,
    num_layers: int | None = None,
) -> int:
    """Return a validated elementwise tile for FFN sequence shift-mix."""

    profile = _native_prefill_shift_mix_launch_profile(
        "ffn", batch_size, prompt_tokens, hidden_size, num_layers
    )
    value = env_int(
        "RWKV7_NATIVE_PREFILL_FFN_SHIFT_MIX_BLOCK_SIZE",
        profile[0] if profile is not None else 256,
        lower=64,
        upper=2048,
    )
    if value not in (64, 128, 256, 512, 1024, 2048):
        raise ValueError(
            "RWKV7_NATIVE_PREFILL_FFN_SHIFT_MIX_BLOCK_SIZE must be a power "
            "of two between 64 and 2048"
        )
    return value

def _native_prefill_fused_state_prep_enabled(
    batch_size: int | None = None,
    prompt_tokens: int | None = None,
    hidden_size: int | None = None,
    num_layers: int | None = None,
) -> bool:
    """Runtime switch for the native prefill state-prep fusion probe."""

    policy = _kernel_policy()
    if not env_flag(
        "RWKV7_NATIVE_PREFILL_FUSED_STATE_PREP",
        bool(getattr(policy, "fused_prefill_state_prep", False)),
    ):
        return False
    if not _native_prefill_model_shape_selected(
        "RWKV7_NATIVE_PREFILL_STATE_PREP_MODEL_SHAPES",
        "prefill_state_prep_model_shapes",
        batch_size,
        prompt_tokens,
        hidden_size,
        num_layers,
    ):
        return False
    if fused_prefill_state_prep is None or fused_prefill_state_prep_available is None:
        return False
    try:
        return bool(fused_prefill_state_prep_available())
    except Exception:
        return False

def _native_prefill_state_prep_layers(
    batch_size: int,
    prompt_tokens: int,
    hidden_size: int,
    num_layers: int,
) -> set[int] | None:
    """Return selected state-prep layers, or ``None`` for every layer."""

    specific = f"RWKV7_NATIVE_PREFILL_STATE_PREP_LAYERS_B{batch_size}_T{prompt_tokens}"
    raw = os.environ.get(
        specific,
        os.environ.get("RWKV7_NATIVE_PREFILL_STATE_PREP_LAYERS"),
    )
    if raw is not None:
        selected: set[int] = set()
        try:
            for item in raw.replace(" ", ",").split(","):
                item = item.strip()
                if not item:
                    continue
                if "-" in item:
                    start, end = (int(value) for value in item.split("-", 1))
                    if start < 0 or end < start:
                        raise ValueError
                    selected.update(range(start, end + 1))
                else:
                    value = int(item)
                    if value < 0:
                        raise ValueError
                    selected.add(value)
        except ValueError as exc:
            raise ValueError(
                f"{specific} must contain non-negative layers or inclusive ranges"
            ) from exc
        return {value for value in selected if value < int(num_layers)}

    target = (
        int(hidden_size),
        int(num_layers),
        int(batch_size),
        int(prompt_tokens),
    )
    for profile in getattr(
        _kernel_policy(), "prefill_state_prep_layer_counts", ()
    ):
        if len(profile) == 5 and tuple(int(value) for value in profile[:4]) == target:
            count = min(max(int(profile[4]), 0), int(num_layers))
            return set(range(count))
    return None

def _native_prefill_fused_state_scan_max_batch() -> int | None:
    """Optional batch ceiling for the fused state-prep scan route."""

    raw = os.environ.get("RWKV7_NATIVE_PREFILL_FUSED_STATE_SCAN_MAX_BATCH")
    if raw is None or not raw.strip():
        policy = _kernel_policy()
        value = getattr(policy, "fused_prefill_state_scan_max_batch", None)
        return None if value is None else int(value)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("RWKV7_NATIVE_PREFILL_FUSED_STATE_SCAN_MAX_BATCH must be an integer") from exc
    if value < 1:
        raise ValueError("RWKV7_NATIVE_PREFILL_FUSED_STATE_SCAN_MAX_BATCH must be >= 1")
    return value

def _native_prefill_fused_state_scan_enabled(batch_size: int | None = None) -> bool:
    """Runtime switch for the fused state-prep plus scan probe."""

    policy = _kernel_policy()
    if not env_flag(
        "RWKV7_NATIVE_PREFILL_FUSED_STATE_SCAN",
        bool(getattr(policy, "fused_prefill_state_scan", False)),
    ):
        return False
    max_batch = _native_prefill_fused_state_scan_max_batch()
    if max_batch is not None and batch_size is not None and int(batch_size) > max_batch:
        return False
    if fused_recurrent_scan_state_prep is None or fused_recurrent_scan_state_prep_available is None:
        return False
    try:
        return bool(fused_recurrent_scan_state_prep_available())
    except Exception:
        return False

def _native_prefill_state_prep_w_dtype() -> str:
    """Output dtype policy for fused native-prefill W decay.

    ``fp32`` preserves the historical torch expression
    ``exp(... w.float())``. ``input`` stores the decay in the model dtype to
    reduce bandwidth into the split-row scan; it is opt-in until end-to-end
    rows prove correctness and speed for a card/model.
    """

    raw = os.environ.get("RWKV7_NATIVE_PREFILL_STATE_PREP_W_DTYPE", "fp32").strip().lower()
    aliases = {
        "fp32": "fp32",
        "float32": "fp32",
        "f32": "fp32",
        "input": "input",
        "model": "input",
        "same": "input",
        "fp16": "input",
        "bf16": "input",
    }
    if raw not in aliases:
        raise ValueError(
            "RWKV7_NATIVE_PREFILL_STATE_PREP_W_DTYPE must be 'fp32' or 'input' "
            f"(aliases: same/model/fp16/bf16); got {raw!r}"
        )
    return aliases[raw]

def _native_prefill_fused_output_enabled(
    batch_size: int | None = None,
    prompt_tokens: int | None = None,
    hidden_size: int | None = None,
    num_layers: int | None = None,
) -> bool:
    """Runtime switch for native prefill output-prep fusion.

    This reuses the profitable decode fused-output-prep kernel, but keeps the
    prefill path explicit until end-to-end prompt rows prove it helps each
    card/model shape.
    """

    policy = _kernel_policy()
    if not env_flag(
        "RWKV7_NATIVE_PREFILL_FUSED_OUTPUT",
        bool(getattr(policy, "fused_prefill_output", False)),
    ):
        return False
    if not _native_prefill_model_shape_selected(
        "RWKV7_NATIVE_PREFILL_FUSED_OUTPUT_MODEL_SHAPES",
        "prefill_fused_output_model_shapes",
        batch_size,
        prompt_tokens,
        hidden_size,
        num_layers,
    ):
        return False
    if fused_attn_output_prepare is None or fused_attn_output_prepare_available is None:
        return False
    try:
        return bool(fused_attn_output_prepare_available())
    except Exception:
        return False

def _native_prefill_fused_output_project_enabled() -> bool:
    """Runtime switch for native prefill output-prep plus ``o_proj`` fusion.

    This is an opt-in experiment for the bsz=1 prompt-prefill gap.  The kernel
    is intentionally disabled by default because it recomputes the prepared
    attention output inside the projection tile; exact-card benchmark rows must
    prove it beats the cuBLAS ``o_proj`` path before it becomes a default.
    """

    if not env_flag("RWKV7_NATIVE_PREFILL_FUSED_OUTPUT_PROJECT", False):
        return False
    if fused_attn_output_project is None or fused_attn_output_project_available is None:
        return False
    try:
        return bool(fused_attn_output_project_available())
    except Exception:
        return False

def _native_prefill_fused_output_project_block_m() -> int:
    """Output tile for the native prefill fused output-project experiment."""

    default = env_int("RWKV7_NATIVE_GRAPH_FUSED_OUTPUT_PROJECT_BLOCK_M", 16, lower=1, upper=128)
    return env_int("RWKV7_NATIVE_PREFILL_FUSED_OUTPUT_PROJECT_BLOCK_M", default, lower=1, upper=128)

def _native_prefill_fused_wavg_lora_requested() -> bool:
    """Return whether the prefill W/A/G/V-gate LoRA fusion probe is requested."""

    return env_flag("RWKV7_NATIVE_PREFILL_FUSED_WAVG_LORA", False)

def _native_prefill_fused_wavg_lora_max_m() -> int:
    """Maximum flattened rows for prefill WAVG LoRA before falling back.

    Initial card-local probes were profitable for `B*T=512` but slower for
    `B*T=2048`, so the opt-in path defaults to the small-prefill shape until an
    exact-card sweep proves a larger tile.
    """

    return env_int("RWKV7_NATIVE_PREFILL_FUSED_WAVG_LORA_MAX_M", 1024, lower=1, upper=1 << 30)

def _native_prefill_fused_wavg_lora_enabled(total_rows: int) -> bool:
    """Runtime switch for the native prefill W/A/G/V-gate LoRA fusion probe."""

    if not _native_prefill_fused_wavg_lora_requested():
        return False
    if int(total_rows) > _native_prefill_fused_wavg_lora_max_m():
        return False
    if fused_wavg_lora is None or fused_wavg_lora_available is None:
        return False
    try:
        return bool(fused_wavg_lora_available())
    except Exception:
        return False

def _native_prefill_fused_wavg_lora_blocks() -> tuple[int, int, int]:
    """Return ``(block_m, block_r, block_k)`` for prefill WAVG LoRA."""

    vals = []
    for name, fallback, default, upper in (
        ("RWKV7_NATIVE_PREFILL_FUSED_WAVG_LORA_BLOCK_M", "RWKV7_NATIVE_GRAPH_FUSED_WAVG_LORA_BLOCK_M", 64, 128),
        ("RWKV7_NATIVE_PREFILL_FUSED_WAVG_LORA_BLOCK_R", "RWKV7_NATIVE_GRAPH_FUSED_WAVG_LORA_BLOCK_R", 64, 128),
        ("RWKV7_NATIVE_PREFILL_FUSED_WAVG_LORA_BLOCK_K", "RWKV7_NATIVE_GRAPH_FUSED_WAVG_LORA_BLOCK_K", 64, 256),
    ):
        raw = os.environ.get(name, os.environ.get(fallback))
        if raw is None:
            vals.append(env_int(name, int(default), lower=1, upper=upper))
        else:
            try:
                val = int(str(raw).strip())
            except ValueError:
                val = int(default)
            vals.append(min(max(1, val), upper))
    return vals[0], vals[1], vals[2]

def _native_prefill_fused_sequence_ffn_enabled(
    total_rows: int,
    batch_size: int | None = None,
    prompt_tokens: int | None = None,
    hidden_size: int | None = None,
    num_layers: int | None = None,
    dtype: torch.dtype | None = None,
) -> bool:
    """Enable the tensor-core sequence FFN only for measured prefill shapes."""

    if (
        dtype is not None
        and None not in (batch_size, prompt_tokens, hidden_size, num_layers)
        and _native_prefill_global_fp16_accum_enabled(
            int(batch_size),
            int(prompt_tokens),
            int(hidden_size),
            int(num_layers),
            dtype,
        )
    ):
        # Exact 5090 B8 rows show cuBLAS with global fp16 accumulation beating
        # the custom sequence-FFN path. Keep BF16/FP32 and all other shapes on
        # their existing policy.
        return False

    policy = _kernel_policy()
    if not env_flag(
        "RWKV7_NATIVE_PREFILL_FUSED_SEQUENCE_FFN",
        bool(getattr(policy, "fused_prefill_sequence_ffn", False)),
    ):
        return False
    min_rows = env_int(
        "RWKV7_NATIVE_PREFILL_SEQUENCE_FFN_MIN_ROWS",
        int(getattr(policy, "prefill_sequence_ffn_min_rows", 128)),
        lower=1,
    )
    policy_max = getattr(policy, "prefill_sequence_ffn_max_rows", None)
    max_rows = env_int(
        "RWKV7_NATIVE_PREFILL_SEQUENCE_FFN_MAX_ROWS",
        (1 << 30) if policy_max is None else int(policy_max),
        lower=1,
    )
    raw_extra = os.environ.get("RWKV7_NATIVE_PREFILL_SEQUENCE_FFN_EXTRA_ROWS")
    if raw_extra is None:
        extra_rows = {int(v) for v in getattr(policy, "prefill_sequence_ffn_extra_rows", ())}
    else:
        try:
            extra_rows = {int(v) for v in raw_extra.replace(",", " ").split() if int(v) > 0}
        except ValueError as exc:
            raise ValueError("RWKV7_NATIVE_PREFILL_SEQUENCE_FFN_EXTRA_ROWS must contain integers") from exc
    raw_model_shapes = os.environ.get("RWKV7_NATIVE_PREFILL_SEQUENCE_FFN_MODEL_SHAPES")
    if raw_model_shapes is None:
        model_shapes = {
            tuple(int(value) for value in shape)
            for shape in getattr(policy, "prefill_sequence_ffn_model_shapes", ())
            if len(shape) == 4
        }
    else:
        model_shapes = set()
        try:
            for item in raw_model_shapes.replace(",", " ").split():
                values = tuple(int(value) for value in item.lower().split("x"))
                if len(values) != 4 or any(value <= 0 for value in values):
                    raise ValueError
                model_shapes.add(values)
        except ValueError as exc:
            raise ValueError(
                "RWKV7_NATIVE_PREFILL_SEQUENCE_FFN_MODEL_SHAPES must contain HxLxBxT tuples"
            ) from exc
    exact_model_shape = (
        (int(hidden_size), int(num_layers), int(batch_size), int(prompt_tokens))
        if None not in (hidden_size, num_layers, batch_size, prompt_tokens)
        else None
    )
    if not (
        min_rows <= int(total_rows) <= max_rows
        or int(total_rows) in extra_rows
        or exact_model_shape in model_shapes
    ):
        return False
    if fused_sequence_ffn is None or fused_sequence_ffn_available is None:
        return False
    try:
        return bool(fused_sequence_ffn_available())
    except Exception:
        return False

def _native_prefill_fp16_accum_ffn_key_enabled(
    batch_size: int,
    prompt_tokens: int,
    hidden_size: int,
    num_layers: int,
    dtype: torch.dtype,
) -> bool:
    """Select reduced-precision accumulation only for measured FFN-key shapes."""

    if dtype != torch.float16:
        return False
    matmul = getattr(getattr(torch.backends, "cuda", None), "matmul", None)
    if matmul is None or not hasattr(matmul, "allow_fp16_accumulation"):
        return False
    try:
        visible_cuda_devices = int(torch.cuda.device_count())
    except Exception:
        visible_cuda_devices = 1
    if visible_cuda_devices > 1 and not env_flag(
        "RWKV7_NATIVE_PREFILL_FP16_ACCUM_MULTI_GPU",
        False,
    ):
        # This PyTorch switch is process-global. Keep the exact-5090 default
        # off in multi-GPU processes so a concurrent 4080/4090 request cannot
        # observe reduced accumulation during its own GEMM. Isolated workers
        # retain the measured route; explicit multi-GPU opt-in stays possible.
        return False
    policy = _kernel_policy()
    raw_shapes = os.environ.get(
        "RWKV7_NATIVE_PREFILL_FP16_ACCUM_FFN_KEY_MODEL_SHAPES"
    )
    if raw_shapes is None:
        model_shapes = {
            tuple(int(value) for value in shape)
            for shape in getattr(
                policy,
                "prefill_fp16_accum_ffn_key_model_shapes",
                (),
            )
            if len(shape) == 4
        }
    else:
        model_shapes = set()
        try:
            for item in raw_shapes.replace(",", " ").split():
                values = tuple(int(value) for value in item.lower().split("x"))
                if len(values) != 4 or any(value <= 0 for value in values):
                    raise ValueError
                model_shapes.add(values)
        except ValueError as exc:
            raise ValueError(
                "RWKV7_NATIVE_PREFILL_FP16_ACCUM_FFN_KEY_MODEL_SHAPES must "
                "contain HxLxBxT tuples"
            ) from exc
    exact_shape = (
        int(hidden_size),
        int(num_layers),
        int(batch_size),
        int(prompt_tokens),
    )
    selected = exact_shape in model_shapes
    return bool(
        selected
        and env_flag(
            "RWKV7_NATIVE_PREFILL_FP16_ACCUM_FFN_KEY",
            selected,
        )
    )

def _native_prefill_fp16_accum_ffn_key_layers(
    batch_size: int,
    prompt_tokens: int,
    hidden_size: int,
    num_layers: int,
) -> set[int] | None:
    """Return selected FFN-key accumulation layers, or ``None`` for all."""

    specific = (
        "RWKV7_NATIVE_PREFILL_FP16_ACCUM_FFN_KEY_LAYERS_"
        f"B{batch_size}_T{prompt_tokens}"
    )
    raw = os.environ.get(
        specific,
        os.environ.get("RWKV7_NATIVE_PREFILL_FP16_ACCUM_FFN_KEY_LAYERS"),
    )
    if raw is not None:
        selected: set[int] = set()
        try:
            for item in raw.replace(" ", ",").split(","):
                item = item.strip()
                if not item:
                    continue
                if "-" in item:
                    start, end = (int(value) for value in item.split("-", 1))
                    if start < 0 or end < start:
                        raise ValueError
                    selected.update(range(start, end + 1))
                else:
                    value = int(item)
                    if value < 0:
                        raise ValueError
                    selected.add(value)
        except ValueError as exc:
            raise ValueError(
                f"{specific} must contain non-negative layers or inclusive ranges"
            ) from exc
        return {value for value in selected if value < int(num_layers)}

    target = (
        int(hidden_size),
        int(num_layers),
        int(batch_size),
        int(prompt_tokens),
    )
    for profile in getattr(
        _kernel_policy(), "prefill_fp16_accum_ffn_key_layer_counts", ()
    ):
        if len(profile) == 5 and tuple(int(value) for value in profile[:4]) == target:
            count = min(max(int(profile[4]), 0), int(num_layers))
            return set(range(count))
    return None

def _native_prefill_stacked_rkv_enabled(
    total_rows: int,
    batch_size: int | None = None,
    prompt_tokens: int | None = None,
    hidden_size: int | None = None,
    num_layers: int | None = None,
) -> bool:
    """Shape gate for lazy packed strided-batched R/K/V GEMM."""

    policy = _kernel_policy()
    if not env_flag(
        "RWKV7_NATIVE_PREFILL_STACKED_RKV",
        bool(getattr(policy, "fused_prefill_stacked_rkv", False)),
    ):
        return False
    min_rows = env_int(
        "RWKV7_NATIVE_PREFILL_STACKED_RKV_MIN_ROWS",
        int(getattr(policy, "prefill_stacked_rkv_min_rows", 128)),
        lower=1,
    )
    policy_max = getattr(policy, "prefill_stacked_rkv_max_rows", None)
    max_rows = env_int(
        "RWKV7_NATIVE_PREFILL_STACKED_RKV_MAX_ROWS",
        (1 << 30) if policy_max is None else int(policy_max),
        lower=1,
    )
    raw_extra = os.environ.get("RWKV7_NATIVE_PREFILL_STACKED_RKV_EXTRA_ROWS")
    if raw_extra is None:
        extra_rows = {int(v) for v in getattr(policy, "prefill_stacked_rkv_extra_rows", ())}
    else:
        try:
            extra_rows = {int(v) for v in raw_extra.replace(",", " ").split() if int(v) > 0}
        except ValueError as exc:
            raise ValueError("RWKV7_NATIVE_PREFILL_STACKED_RKV_EXTRA_ROWS must contain integers") from exc
    raw_shapes = os.environ.get("RWKV7_NATIVE_PREFILL_STACKED_RKV_SHAPES")
    if raw_shapes is None:
        shapes = {
            (int(shape[0]), int(shape[1]))
            for shape in getattr(policy, "prefill_stacked_rkv_shapes", ())
            if len(shape) == 2
        }
    else:
        shapes = set()
        try:
            for item in raw_shapes.replace(",", " ").split():
                left, right = item.lower().split("x", 1)
                shape = (int(left), int(right))
                if shape[0] <= 0 or shape[1] <= 0:
                    raise ValueError
                shapes.add(shape)
        except ValueError as exc:
            raise ValueError(
                "RWKV7_NATIVE_PREFILL_STACKED_RKV_SHAPES must contain BxT pairs"
            ) from exc
    exact_shape = (
        (int(batch_size), int(prompt_tokens))
        if batch_size is not None and prompt_tokens is not None
        else None
    )
    raw_model_shapes = os.environ.get("RWKV7_NATIVE_PREFILL_STACKED_RKV_MODEL_SHAPES")
    if raw_model_shapes is None:
        model_shapes = {
            tuple(int(v) for v in shape)
            for shape in getattr(policy, "prefill_stacked_rkv_model_shapes", ())
            if len(shape) == 4
        }
    else:
        model_shapes = set()
        try:
            for item in raw_model_shapes.replace(",", " ").split():
                values = tuple(int(v) for v in item.lower().split("x"))
                if len(values) != 4 or any(v <= 0 for v in values):
                    raise ValueError
                model_shapes.add(values)
        except ValueError as exc:
            raise ValueError(
                "RWKV7_NATIVE_PREFILL_STACKED_RKV_MODEL_SHAPES must contain HxLxBxT tuples"
            ) from exc
    exact_model_shape = (
        (int(hidden_size), int(num_layers), int(batch_size), int(prompt_tokens))
        if None not in (hidden_size, num_layers, batch_size, prompt_tokens)
        else None
    )
    return bool(
        min_rows <= int(total_rows) <= max_rows
        or int(total_rows) in extra_rows
        or exact_shape in shapes
        or exact_model_shape in model_shapes
    )

def _native_prefill_sequence_ffn_blocks(total_rows: int | None = None) -> tuple[int, int, int, int, int]:
    """Return measured ``(BM, BN, key-BK, value-BK, group-M)`` tiles."""

    policy = _kernel_policy()
    large_min = int(getattr(policy, "prefill_sequence_ffn_large_min_rows", 1024))
    use_large = total_rows is not None and int(total_rows) >= large_min
    defaults = tuple(
        getattr(
            policy,
            "prefill_sequence_ffn_large_blocks" if use_large else "prefill_sequence_ffn_blocks",
            (128, 128, 32, 64, 8),
        )
    )
    names = (
        f"RWKV7_NATIVE_PREFILL_SEQUENCE_FFN_{'LARGE_' if use_large else ''}BLOCK_M",
        f"RWKV7_NATIVE_PREFILL_SEQUENCE_FFN_{'LARGE_' if use_large else ''}BLOCK_N",
        f"RWKV7_NATIVE_PREFILL_SEQUENCE_FFN_{'LARGE_' if use_large else ''}KEY_BLOCK_K",
        f"RWKV7_NATIVE_PREFILL_SEQUENCE_FFN_{'LARGE_' if use_large else ''}VALUE_BLOCK_K",
        f"RWKV7_NATIVE_PREFILL_SEQUENCE_FFN_{'LARGE_' if use_large else ''}GROUP_M",
    )
    return tuple(env_int(name, int(default), lower=1, upper=256) for name, default in zip(names, defaults))  # type: ignore[return-value]

def _native_prefill_sequence_ffn_launch() -> tuple[int, int]:
    """Return measured ``(num_stages, num_warps)`` launch settings."""

    policy = _kernel_policy()
    stages = env_int(
        "RWKV7_NATIVE_PREFILL_SEQUENCE_FFN_NUM_STAGES",
        int(getattr(policy, "prefill_sequence_ffn_num_stages", 3)),
        lower=1,
        upper=5,
    )
    warps = env_int(
        "RWKV7_NATIVE_PREFILL_SEQUENCE_FFN_NUM_WARPS",
        int(getattr(policy, "prefill_sequence_ffn_num_warps", 4)),
        lower=1,
        upper=8,
    )
    if warps not in {1, 2, 4, 8}:
        raise ValueError("RWKV7_NATIVE_PREFILL_SEQUENCE_FFN_NUM_WARPS must be 1, 2, 4, or 8")
    return stages, warps

__all__ = ['_native_prefill_fused_scan_enabled', '_native_prefill_fp16_recurrent_requested', '_native_prefill_fp16_recurrent_enabled', '_native_prefill_global_fp16_accum_enabled', '_native_prefill_self_chunk_enabled', '_native_prefill_self_chunk_size', '_native_prefill_self_chunk_h_tiles', '_native_prefill_self_chunk_safe_gate', '_native_prefill_dplr_scan_enabled', '_native_prefill_fused_residual_gemm_enabled', '_native_prefill_dplr_chunk_size', '_native_prefill_fused_clampw_scan_enabled', '_native_prefill_fused_scan_output_enabled', '_native_prefill_default_scan_block_m', '_native_prefill_scan_block_m', '_native_prefill_scan_num_warps', '_native_prefill_model_shape_selected', '_native_prefill_policy_model_shape_selected', '_native_prefill_fused_shift_mix_enabled', '_native_prefill_shift_mix_layers', '_native_prefill_shift_mix_launch_profile', '_native_prefill_attn_shift_mix_block_size', '_native_prefill_shift_mix_num_warps', '_native_prefill_ffn_shift_mix_block_size', '_native_prefill_fused_state_prep_enabled', '_native_prefill_state_prep_layers', '_native_prefill_fused_state_scan_max_batch', '_native_prefill_fused_state_scan_enabled', '_native_prefill_state_prep_w_dtype', '_native_prefill_fused_output_enabled', '_native_prefill_fused_output_project_enabled', '_native_prefill_fused_output_project_block_m', '_native_prefill_fused_wavg_lora_requested', '_native_prefill_fused_wavg_lora_max_m', '_native_prefill_fused_wavg_lora_enabled', '_native_prefill_fused_wavg_lora_blocks', '_native_prefill_fused_sequence_ffn_enabled', '_native_prefill_fp16_accum_ffn_key_enabled', '_native_prefill_fp16_accum_ffn_key_layers', '_native_prefill_stacked_rkv_enabled', '_native_prefill_sequence_ffn_blocks', '_native_prefill_sequence_ffn_launch']

_IMPLEMENTATIONS = {name: globals()[name] for name in __all__}
