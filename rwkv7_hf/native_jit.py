# coding=utf-8
"""TorchScript-native RWKV-7 decode. The ENTIRE per-layer block (LayerNorms +
TMix_one + CMix_one) is fused into one torch.jit.script function, so per token
there is only ~1 C++ call per layer + embedding/head. Math ports the official
RWKV_x070 TMix_one/CMix_one (bit-exact vs FLA, see native.py).

Run: python -m rwkv7_hf.native_jit <hf_dir>
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F

try:  # pragma: no cover - optional in older converted model dirs
    from .kernel_policy import current_kernel_policy, env_blocks, env_flag, env_int
except Exception:  # pragma: no cover - direct remote-file execution fallback
    try:
        from kernel_policy import current_kernel_policy, env_blocks, env_flag, env_int
    except Exception:
        current_kernel_policy = None  # type: ignore[assignment]

        def env_flag(name: str, default: bool) -> bool:
            raw = os.environ.get(name)
            if raw is None:
                return bool(default)
            return raw.strip().lower() not in {"0", "false", "no", "off"}

        def env_int(name: str, default: int, *, lower: int = 1, upper: int | None = None) -> int:
            try:
                value = int(os.environ.get(name, str(default)).strip())
            except Exception:
                value = default
            value = max(lower, value)
            return min(value, upper) if upper is not None else value

        def env_blocks(names: tuple[str, str, str], defaults: tuple[int, int, int], uppers: tuple[int, int, int]) -> tuple[int, int, int]:
            return (
                env_int(names[0], defaults[0], lower=1, upper=uppers[0]),
                env_int(names[1], defaults[1], lower=1, upper=uppers[1]),
                env_int(names[2], defaults[2], lower=1, upper=uppers[2]),
            )

try:  # pragma: no cover - optional Triton fast path on CUDA hosts
    from .fused_recurrent_update import (
        fused_recurrent_output_prepare,
        fused_recurrent_output_prepare_available,
        fused_recurrent_scan,
        fused_recurrent_scan_available,
        fused_recurrent_scan_clampw,
        fused_recurrent_scan_clampw_available,
        fused_recurrent_scan_state_prep,
        fused_recurrent_scan_state_prep_available,
        fused_recurrent_scan_state_prep_nokv,
        fused_recurrent_scan_state_prep_nokv_available,
        fused_recurrent_scan_state_prep_correction,
        fused_recurrent_scan_state_prep_correction_available,
        fused_recurrent_scan_state_prep_sk,
        fused_recurrent_scan_state_prep_sk_available,
        fused_recurrent_scan_state_prep_output_prepare,
        fused_recurrent_scan_state_prep_output_prepare_available,
        fused_recurrent_scan_output_prepare,
        fused_recurrent_scan_output_prepare_available,
        fused_recurrent_update,
        fused_recurrent_update_available,
    )
except Exception:  # pragma: no cover - direct remote-file execution fallback
    try:
        from fused_recurrent_update import (
            fused_recurrent_output_prepare,
            fused_recurrent_output_prepare_available,
            fused_recurrent_scan,
            fused_recurrent_scan_available,
            fused_recurrent_scan_clampw,
            fused_recurrent_scan_clampw_available,
            fused_recurrent_scan_state_prep,
            fused_recurrent_scan_state_prep_available,
            fused_recurrent_scan_state_prep_nokv,
            fused_recurrent_scan_state_prep_nokv_available,
            fused_recurrent_scan_state_prep_correction,
            fused_recurrent_scan_state_prep_correction_available,
            fused_recurrent_scan_state_prep_sk,
            fused_recurrent_scan_state_prep_sk_available,
            fused_recurrent_scan_state_prep_output_prepare,
            fused_recurrent_scan_state_prep_output_prepare_available,
            fused_recurrent_scan_output_prepare,
            fused_recurrent_scan_output_prepare_available,
            fused_recurrent_update,
            fused_recurrent_update_available,
        )
    except Exception:
        fused_recurrent_output_prepare = None  # type: ignore[assignment]
        fused_recurrent_output_prepare_available = None  # type: ignore[assignment]
        fused_recurrent_scan = None  # type: ignore[assignment]
        fused_recurrent_scan_available = None  # type: ignore[assignment]
        fused_recurrent_scan_clampw = None  # type: ignore[assignment]
        fused_recurrent_scan_clampw_available = None  # type: ignore[assignment]
        fused_recurrent_scan_state_prep = None  # type: ignore[assignment]
        fused_recurrent_scan_state_prep_available = None  # type: ignore[assignment]
        fused_recurrent_scan_state_prep_nokv = None  # type: ignore[assignment]
        fused_recurrent_scan_state_prep_nokv_available = None  # type: ignore[assignment]
        fused_recurrent_scan_state_prep_correction = None  # type: ignore[assignment]
        fused_recurrent_scan_state_prep_correction_available = None  # type: ignore[assignment]
        fused_recurrent_scan_state_prep_sk = None  # type: ignore[assignment]
        fused_recurrent_scan_state_prep_sk_available = None  # type: ignore[assignment]
        fused_recurrent_scan_state_prep_output_prepare = None  # type: ignore[assignment]
        fused_recurrent_scan_state_prep_output_prepare_available = None  # type: ignore[assignment]
        fused_recurrent_scan_output_prepare = None  # type: ignore[assignment]
        fused_recurrent_scan_output_prepare_available = None  # type: ignore[assignment]
        fused_recurrent_update = None  # type: ignore[assignment]
        fused_recurrent_update_available = None  # type: ignore[assignment]

try:  # pragma: no cover - optional pure-torch DPLR/chunked prefill prototype
    from .dplr_prefill import dplr_chunk_scan
except Exception:  # pragma: no cover - direct remote-file execution fallback
    try:
        from dplr_prefill import dplr_chunk_scan
    except Exception:
        dplr_chunk_scan = None  # type: ignore[assignment]

try:  # pragma: no cover - optional CUDA extension prototype
    from .cuda_state_scan import cuda_state_scan_prep, cuda_state_scan_prep_available
except Exception:  # pragma: no cover - direct remote-file execution fallback
    try:
        from cuda_state_scan import cuda_state_scan_prep, cuda_state_scan_prep_available
    except Exception:
        cuda_state_scan_prep = None  # type: ignore[assignment]
        cuda_state_scan_prep_available = None  # type: ignore[assignment]

try:  # pragma: no cover - optional Triton fast path on CUDA hosts
    from .fused_output import (
        fused_attn_output_prepare,
        fused_attn_output_prepare_available,
        fused_attn_output_prepare_from_correction,
        fused_attn_output_prepare_from_correction_available,
        fused_attn_output_prepare_from_sk_raw_v,
        fused_attn_output_prepare_from_sk_raw_v_available,
        fused_attn_output_prepare_raw_kv,
        fused_attn_output_prepare_raw_kv_available,
        fused_attn_output_project,
        fused_attn_output_project_available,
    )
except Exception:  # pragma: no cover - direct remote-file execution fallback
    try:
        from fused_output import (
            fused_attn_output_prepare,
            fused_attn_output_prepare_available,
            fused_attn_output_prepare_from_correction,
            fused_attn_output_prepare_from_correction_available,
            fused_attn_output_prepare_from_sk_raw_v,
            fused_attn_output_prepare_from_sk_raw_v_available,
            fused_attn_output_prepare_raw_kv,
            fused_attn_output_prepare_raw_kv_available,
            fused_attn_output_project,
            fused_attn_output_project_available,
        )
    except Exception:
        fused_attn_output_prepare = None  # type: ignore[assignment]
        fused_attn_output_prepare_available = None  # type: ignore[assignment]
        fused_attn_output_prepare_from_correction = None  # type: ignore[assignment]
        fused_attn_output_prepare_from_correction_available = None  # type: ignore[assignment]
        fused_attn_output_prepare_from_sk_raw_v = None  # type: ignore[assignment]
        fused_attn_output_prepare_from_sk_raw_v_available = None  # type: ignore[assignment]
        fused_attn_output_prepare_raw_kv = None  # type: ignore[assignment]
        fused_attn_output_prepare_raw_kv_available = None  # type: ignore[assignment]
        fused_attn_output_project = None  # type: ignore[assignment]
        fused_attn_output_project_available = None  # type: ignore[assignment]

try:  # pragma: no cover - optional Triton fast path on CUDA hosts
    from .fused_attention_projection import (
        fused_rkv_wag_projection,
        fused_rkv_wag_projection_available,
        fused_rkv_wavg_projection,
        fused_rkv_wavg_projection_available,
    )
except Exception:  # pragma: no cover - direct remote-file execution fallback
    try:
        from fused_attention_projection import (
            fused_rkv_wag_projection,
            fused_rkv_wag_projection_available,
            fused_rkv_wavg_projection,
            fused_rkv_wavg_projection_available,
        )
    except Exception:
        fused_rkv_wag_projection = None  # type: ignore[assignment]
        fused_rkv_wag_projection_available = None  # type: ignore[assignment]
        fused_rkv_wavg_projection = None  # type: ignore[assignment]
        fused_rkv_wavg_projection_available = None  # type: ignore[assignment]

try:  # pragma: no cover - optional Triton fast path on CUDA hosts
    from .fused_lora import fused_wag_lora, fused_wag_lora_available, fused_wavg_lora, fused_wavg_lora_available
except Exception:  # pragma: no cover - direct remote-file execution fallback
    try:
        from fused_lora import fused_wag_lora, fused_wag_lora_available, fused_wavg_lora, fused_wavg_lora_available
    except Exception:
        fused_wag_lora = None  # type: ignore[assignment]
        fused_wag_lora_available = None  # type: ignore[assignment]
        fused_wavg_lora = None  # type: ignore[assignment]
        fused_wavg_lora_available = None  # type: ignore[assignment]

try:  # pragma: no cover - optional Triton fast path on CUDA hosts
    from .fused_prefill import (
        fused_prefill_kv_kk_prep,
        fused_prefill_kv_kk_prep_available,
        fused_prefill_state_prep,
        fused_prefill_state_prep_available,
    )
except Exception:  # pragma: no cover - direct remote-file execution fallback
    try:
        from fused_prefill import (
            fused_prefill_kv_kk_prep,
            fused_prefill_kv_kk_prep_available,
            fused_prefill_state_prep,
            fused_prefill_state_prep_available,
        )
    except Exception:
        fused_prefill_kv_kk_prep = None  # type: ignore[assignment]
        fused_prefill_kv_kk_prep_available = None  # type: ignore[assignment]
        fused_prefill_state_prep = None  # type: ignore[assignment]
        fused_prefill_state_prep_available = None  # type: ignore[assignment]

try:  # pragma: no cover - optional Triton fast path on CUDA hosts
    from .fused_time_mix import fused_attn_shift_mix, fused_attn_shift_mix_available
except Exception:  # pragma: no cover - direct remote-file execution fallback
    try:
        from fused_time_mix import fused_attn_shift_mix, fused_attn_shift_mix_available
    except Exception:
        fused_attn_shift_mix = None  # type: ignore[assignment]
        fused_attn_shift_mix_available = None  # type: ignore[assignment]


_FALSE_VALUES = {"0", "false", "False", "no", "off"}


def _kernel_policy():
    if current_kernel_policy is None:
        return None
    try:
        return current_kernel_policy(torch_module=torch)
    except Exception:
        return None


def _native_graph_fused_recurrent_enabled() -> bool:
    """Runtime switch for the experimental native-graph recurrent Triton path."""

    policy = _kernel_policy()
    if not env_flag("RWKV7_NATIVE_GRAPH_FUSED_RECURRENT", bool(getattr(policy, "fused_recurrent", False))):
        return False
    if fused_recurrent_update is None or fused_recurrent_update_available is None:
        return False
    try:
        return bool(fused_recurrent_update_available())
    except Exception:
        return False


def _native_prefill_fused_scan_enabled() -> bool:
    """Runtime switch for the experimental native prefill recurrent scan."""

    policy = _kernel_policy()
    if not env_flag("RWKV7_NATIVE_PREFILL_FUSED_SCAN", bool(getattr(policy, "fused_prefill_scan", False))):
        return False
    if fused_recurrent_scan is None or fused_recurrent_scan_available is None:
        return False
    try:
        return bool(fused_recurrent_scan_available())
    except Exception:
        return False


def _native_prefill_dplr_scan_enabled() -> bool:
    """Runtime switch for the correctness-first DPLR/chunked prefill scan."""

    if not env_flag("RWKV7_NATIVE_PREFILL_DPLR_SCAN", False):
        return False
    return dplr_chunk_scan is not None


def _native_prefill_dplr_chunk_size() -> int:
    """Chunk length for the pure-torch DPLR/chunked prefill reference path."""

    return env_int("RWKV7_NATIVE_PREFILL_DPLR_CHUNK_SIZE", 64, lower=1, upper=4096)


def _native_prefill_fused_clampw_scan_enabled() -> bool:
    """Runtime switch for raw-W clampw native prefill recurrent scan."""

    if not env_flag("RWKV7_NATIVE_PREFILL_FUSED_CLAMPW_SCAN", False):
        return False
    if not _native_prefill_fused_scan_enabled():
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


def _native_prefill_default_scan_block_m(head_dim: int) -> int:
    """Default recurrent-scan row tile for the current CUDA architecture.

    Ada/4090 validation prefers the full-head tile (`head_dim=64`), while the
    sm70/V100 sweep prefers a narrower split-row tile.  Keep this as a default
    only: explicit `RWKV7_NATIVE_PREFILL_SCAN_BLOCK_M` still wins so benchmark
    rows remain reproducible.
    """

    head_dim = int(head_dim)
    if head_dim == 64 and torch.cuda.is_available():
        try:
            major, _minor = torch.cuda.get_device_capability()
        except Exception:
            major = 0
        if int(major) == 7:
            return 16
    return head_dim


def _native_prefill_scan_block_m(head_dim: int) -> int:
    """Row tile for the optional split-row recurrent scan kernel."""

    return env_int(
        "RWKV7_NATIVE_PREFILL_SCAN_BLOCK_M",
        _native_prefill_default_scan_block_m(head_dim),
        lower=1,
        upper=int(head_dim),
    )


def _native_prefill_scan_num_warps(head_dim: int, block_m: int | None = None) -> int:
    """Triton warp count for the optional native prefill recurrent scan."""

    if block_m is None:
        block_m = _native_prefill_scan_block_m(head_dim)
    default = 4 if int(block_m) < int(head_dim) else 8
    value = env_int("RWKV7_NATIVE_PREFILL_SCAN_NUM_WARPS", default, lower=1, upper=8)
    if value not in {1, 2, 4, 8}:
        raise ValueError(f"RWKV7_NATIVE_PREFILL_SCAN_NUM_WARPS must be one of 1, 2, 4, or 8; got {value}")
    return value


def _native_prefill_scan_num_stages() -> int:
    """Triton pipeline stage count for optional native prefill scan kernels."""

    return env_int("RWKV7_NATIVE_PREFILL_SCAN_NUM_STAGES", 3, lower=1, upper=8)


def _native_prefill_scan_algebraic_output_enabled() -> bool:
    """Use an algebraically expanded recurrent output inside state-scan."""

    return env_flag("RWKV7_NATIVE_PREFILL_SCAN_ALGEBRAIC_OUTPUT", False)


def _native_prefill_scan_nomask64_enabled() -> bool:
    """Use the specialized unmasked full-head scan for head_dim=64."""

    return env_flag("RWKV7_NATIVE_PREFILL_SCAN_NOMASK64", False)


def _native_prefill_cuda_state_scan_enabled() -> bool:
    """Runtime switch for the experimental CUDA N=64 state-scan prototype."""

    if not env_flag("RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN", False):
        return False
    if cuda_state_scan_prep is None or cuda_state_scan_prep_available is None:
        return False
    try:
        return bool(cuda_state_scan_prep_available())
    except Exception:
        return False


def _native_prefill_cuda_state_scan_lanes_per_row() -> int:
    """Per-row CUDA parallelism for the experimental N=64 state-scan."""

    value = env_int("RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_LANES", 1, lower=1, upper=64)
    if value not in {1, 2, 4, 8, 16, 64}:
        raise ValueError("RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_LANES must be one of 1, 2, 4, 8, 16, or 64")
    return value


def _native_prefill_cuda_state_scan_precompute_enabled() -> bool:
    """Use two-stage vector precompute before the CUDA row-block scan."""

    return env_flag("RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_PRECOMPUTE", False)


def _native_prefill_cuda_state_scan_precompute_mode() -> str:
    """Vector precompute variant for the experimental CUDA row-block scan."""

    if not _native_prefill_cuda_state_scan_precompute_enabled():
        return "none"
    value = os.environ.get("RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_PRECOMPUTE_MODE", "full")
    mode = str(value).strip().lower().replace("-", "_")
    if mode in {"1", "true", "yes", "on", "full"}:
        return "full"
    if mode in {"2", "wk", "wkk", "w_kk", "reduced", "reduced_temp", "wk_fp16kv", "fp16kv"}:
        return "wk"
    if mode in {"0", "false", "no", "off", "none"}:
        return "none"
    raise ValueError(
        "RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_PRECOMPUTE_MODE must be one of full, wk/reduced_temp, or none"
    )


def _native_prefill_cuda_state_scan_rows_per_block() -> int:
    """Rows handled by one CUDA row-block in the cooperative N=64 scan."""

    value = env_int("RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_ROWS_PER_BLOCK", 1, lower=1, upper=8)
    if value not in {1, 2, 4, 8}:
        raise ValueError("RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_ROWS_PER_BLOCK must be one of 1, 2, 4, or 8")
    return value


def _native_prefill_cuda_state_scan_schedule() -> str:
    """CUDA row-block schedule variant for the experimental N=64 scan."""

    value = os.environ.get("RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_SCHEDULE", "default")
    schedule = str(value).strip().lower().replace("-", "_")
    if schedule in {"", "0", "default", "normal", "rowblock", "none"}:
        return "default"
    if schedule in {"1", "warp", "warp_specialized", "warp_specialised", "producer_worker", "producer"}:
        return "warp_specialized"
    raise ValueError("RWKV7_NATIVE_PREFILL_CUDA_STATE_SCAN_SCHEDULE must be default or warp_specialized")


def _native_prefill_fused_shift_mix_enabled() -> bool:
    """Runtime switch for prefill attention shift-mix fusion telemetry."""

    if not env_flag("RWKV7_NATIVE_PREFILL_FUSED_SHIFT_MIX", False):
        return False
    if fused_attn_shift_mix is None or fused_attn_shift_mix_available is None:
        return False
    try:
        return bool(fused_attn_shift_mix_available())
    except Exception:
        return False


def _native_prefill_fused_state_prep_enabled() -> bool:
    """Runtime switch for the native prefill state-prep fusion probe."""

    if not env_flag("RWKV7_NATIVE_PREFILL_FUSED_STATE_PREP", False):
        return False
    if fused_prefill_state_prep is None or fused_prefill_state_prep_available is None:
        return False
    try:
        return bool(fused_prefill_state_prep_available())
    except Exception:
        return False


def _native_prefill_fused_state_scan_enabled() -> bool:
    """Runtime switch for the fused state-prep plus scan probe."""

    if not env_flag("RWKV7_NATIVE_PREFILL_FUSED_STATE_SCAN", False):
        return False
    if fused_recurrent_scan_state_prep is None or fused_recurrent_scan_state_prep_available is None:
        return False
    try:
        return bool(fused_recurrent_scan_state_prep_available())
    except Exception:
        return False


def _native_prefill_scan_precompute_w_enabled() -> bool:
    """Opt-in precompute of W decay before the fused state-scan kernel.

    The default fused state-scan computes ``exp(-0.606531 * sigmoid(w_raw))``
    inside the dominant recurrent scan loop.  This experiment materializes the
    decay once before the scan and lets the scan load it directly.  It is kept
    behind an env flag because it trades extra pointwise launches and memory
    traffic for lower special-function pressure inside the scan kernel.
    """

    return env_flag("RWKV7_NATIVE_PREFILL_SCAN_PRECOMPUTE_W", False)


def _native_prefill_scan_precompute_w_dtype() -> str:
    """Dtype for opt-in precomputed W decay passed to fused state-scan."""

    raw = os.environ.get("RWKV7_NATIVE_PREFILL_SCAN_PRECOMPUTE_W_DTYPE", "fp32").strip().lower()
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
            "RWKV7_NATIVE_PREFILL_SCAN_PRECOMPUTE_W_DTYPE must be 'fp32' or 'input' "
            f"(aliases: same/model/fp16/bf16); got {raw!r}"
        )
    return aliases[raw]


def _native_prefill_fused_state_scan_correction_enabled() -> bool:
    """Runtime switch for state-scan that emits correction instead of K/V."""

    if not env_flag("RWKV7_NATIVE_PREFILL_FUSED_STATE_SCAN_CORRECTION", False):
        return False
    if (
        fused_recurrent_scan_state_prep_correction is None
        or fused_recurrent_scan_state_prep_correction_available is None
        or fused_attn_output_prepare_from_correction is None
        or fused_attn_output_prepare_from_correction_available is None
    ):
        return False
    try:
        return bool(fused_recurrent_scan_state_prep_correction_available()) and bool(
            fused_attn_output_prepare_from_correction_available()
        )
    except Exception:
        return False


def _native_prefill_fused_state_scan_raw_output_enabled() -> bool:
    """Runtime switch for no-K/V scan plus raw-K/V output-prep recompute."""

    if not env_flag("RWKV7_NATIVE_PREFILL_FUSED_STATE_SCAN_RAW_OUTPUT", False):
        return False
    if (
        fused_recurrent_scan_state_prep_nokv is None
        or fused_recurrent_scan_state_prep_nokv_available is None
        or fused_attn_output_prepare_raw_kv is None
        or fused_attn_output_prepare_raw_kv_available is None
    ):
        return False
    try:
        return bool(fused_recurrent_scan_state_prep_nokv_available()) and bool(
            fused_attn_output_prepare_raw_kv_available()
        )
    except Exception:
        return False


def _native_prefill_fused_state_scan_sk_output_enabled() -> bool:
    """Runtime switch for no-K/V scan that emits sk plus raw-V output prep."""

    if not env_flag("RWKV7_NATIVE_PREFILL_FUSED_STATE_SCAN_SK_OUTPUT", False):
        return False
    if (
        fused_recurrent_scan_state_prep_sk is None
        or fused_recurrent_scan_state_prep_sk_available is None
        or fused_attn_output_prepare_from_sk_raw_v is None
        or fused_attn_output_prepare_from_sk_raw_v_available is None
    ):
        return False
    try:
        return bool(fused_recurrent_scan_state_prep_sk_available()) and bool(
            fused_attn_output_prepare_from_sk_raw_v_available()
        )
    except Exception:
        return False


def _native_prefill_fused_state_scan_output_enabled() -> bool:
    """Runtime switch for fused state-prep plus scan plus output-prep."""

    if not env_flag("RWKV7_NATIVE_PREFILL_FUSED_STATE_SCAN_OUTPUT", False):
        return False
    if (
        fused_recurrent_scan_state_prep_output_prepare is None
        or fused_recurrent_scan_state_prep_output_prepare_available is None
    ):
        return False
    try:
        return bool(fused_recurrent_scan_state_prep_output_prepare_available())
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


def _native_prefill_fused_output_enabled() -> bool:
    """Runtime switch for native prefill output-prep fusion.

    This reuses the profitable decode fused-output-prep kernel, but keeps the
    prefill path explicit until end-to-end prompt rows prove it helps each
    card/model shape.
    """

    if not env_flag("RWKV7_NATIVE_PREFILL_FUSED_OUTPUT", False):
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

    The first RTX 4090 probe is profitable for `B*T=512` but slower for
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


def _native_prefill_fused_projection_requested() -> bool:
    """Return whether the prefill R/K/V + LoRA projection fusion probe is requested."""

    return env_flag("RWKV7_NATIVE_PREFILL_FUSED_PROJECTION", False)


def _native_prefill_fused_projection_max_m() -> int:
    """Maximum flattened rows for the prefill fused projection experiment."""

    return env_int("RWKV7_NATIVE_PREFILL_FUSED_PROJECTION_MAX_M", 1024, lower=1, upper=1 << 30)


def _native_prefill_fused_projection_enabled(total_rows: int) -> bool:
    """Runtime switch for prefill R/K/V + W/A/G(/V-gate) projection fusion."""

    if not _native_prefill_fused_projection_requested():
        return False
    if int(total_rows) > _native_prefill_fused_projection_max_m():
        return False
    if (
        fused_rkv_wag_projection is None
        or fused_rkv_wag_projection_available is None
        or fused_rkv_wavg_projection is None
        or fused_rkv_wavg_projection_available is None
    ):
        return False
    try:
        return bool(fused_rkv_wag_projection_available()) and bool(fused_rkv_wavg_projection_available())
    except Exception:
        return False


def _native_prefill_fused_projection_blocks() -> tuple[int, int, int]:
    """Return ``(block_m, block_r, block_k)`` for prefill fused projection."""

    vals = []
    for name, fallback, default, upper in (
        ("RWKV7_NATIVE_PREFILL_FUSED_PROJECTION_BLOCK_M", "RWKV7_NATIVE_GRAPH_FUSED_PROJECTION_BLOCK_M", 64, 128),
        ("RWKV7_NATIVE_PREFILL_FUSED_PROJECTION_BLOCK_R", "RWKV7_NATIVE_GRAPH_FUSED_PROJECTION_BLOCK_R", 64, 128),
        ("RWKV7_NATIVE_PREFILL_FUSED_PROJECTION_BLOCK_K", "RWKV7_NATIVE_GRAPH_FUSED_PROJECTION_BLOCK_K", 64, 256),
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


def _native_graph_fused_recurrent_output_enabled() -> bool:
    """Runtime switch for fused recurrent update plus output-prep."""

    policy = _kernel_policy()
    if not env_flag("RWKV7_NATIVE_GRAPH_FUSED_RECURRENT_OUTPUT", bool(getattr(policy, "fused_recurrent_output", True))):
        return False
    if fused_recurrent_output_prepare is None or fused_recurrent_output_prepare_available is None:
        return False
    try:
        return bool(fused_recurrent_output_prepare_available())
    except Exception:
        return False


def _native_graph_fused_output_enabled() -> bool:
    """Runtime switch for the experimental native-graph output-prep Triton path."""

    policy = _kernel_policy()
    if not env_flag("RWKV7_NATIVE_GRAPH_FUSED_OUTPUT", bool(getattr(policy, "fused_output", True))):
        return False
    if fused_attn_output_prepare is None or fused_attn_output_prepare_available is None:
        return False
    try:
        return bool(fused_attn_output_prepare_available())
    except Exception:
        return False


def _native_graph_fused_output_project_enabled() -> bool:
    """Runtime switch for fused output-prep plus ``o_proj`` in native_graph."""

    policy = _kernel_policy()
    if not env_flag("RWKV7_NATIVE_GRAPH_FUSED_OUTPUT_PROJECT", bool(getattr(policy, "fused_output_project", False))):
        return False
    if fused_attn_output_project is None or fused_attn_output_project_available is None:
        return False
    try:
        return bool(fused_attn_output_project_available())
    except Exception:
        return False


def _native_graph_fused_output_project_block_m() -> int:
    """Output-projection row tile used by the prototype fused output-project kernel."""

    policy = _kernel_policy()
    default = int(getattr(policy, "output_project_block_m", 16))
    return env_int("RWKV7_NATIVE_GRAPH_FUSED_OUTPUT_PROJECT_BLOCK_M", default, lower=1, upper=128)


def _native_graph_fused_projection_enabled() -> bool:
    """Runtime switch for the experimental native-graph R/K/V + W/A/G projection path."""

    policy = _kernel_policy()
    if not env_flag("RWKV7_NATIVE_GRAPH_FUSED_PROJECTION", bool(getattr(policy, "fused_projection", False))):
        return False
    if fused_rkv_wavg_projection is None or fused_rkv_wavg_projection_available is None:
        return False
    try:
        return bool(fused_rkv_wavg_projection_available())
    except Exception:
        return False


def _native_graph_fused_wag_lora_enabled() -> bool:
    """Runtime switch for the native-graph W/A/G LoRA-only fusion probe."""

    policy = _kernel_policy()
    if not env_flag("RWKV7_NATIVE_GRAPH_FUSED_WAG_LORA", bool(getattr(policy, "fused_wag_lora", False))):
        return False
    if fused_wag_lora is None or fused_wag_lora_available is None:
        return False
    try:
        return bool(fused_wag_lora_available())
    except Exception:
        return False


def _native_graph_fused_wavg_lora_enabled() -> bool:
    """Runtime switch for the native-graph W/A/G/V-gate LoRA fusion probe."""

    policy = _kernel_policy()
    if not env_flag("RWKV7_NATIVE_GRAPH_FUSED_WAVG_LORA", bool(getattr(policy, "fused_wavg_lora", False))):
        return False
    if fused_wavg_lora is None or fused_wavg_lora_available is None:
        return False
    try:
        return bool(fused_wavg_lora_available())
    except Exception:
        return False


def _native_graph_rkv_policy() -> str:
    """Return the optional VKWR-inspired R/K/V projection dispatch policy.

    VKWR stacks the receptance/key/value matrices and uses a grouped batched
    projection for selected small-row decode cases.  Keep the HF adapter's
    historical three-``F.linear`` path by default and enable the stacked path
    only through ``RWKV7_NATIVE_GRAPH_RKV_POLICY=vkwr_auto`` while collecting
    telemetry.
    """

    raw = os.environ.get("RWKV7_NATIVE_GRAPH_RKV_POLICY", "manual").strip().lower()
    if raw in {"", "manual", "explicit", "env"}:
        return "manual"
    if raw in {"0", "false", "no", "off", "disabled"}:
        return "off"
    if raw in {"vkwr", "vkwr_auto", "auto", "stacked", "bmm"}:
        return "vkwr_auto"
    return "manual"


def _native_graph_int_env(name: str, default: int, *, lo: int = 1, hi: int | None = None) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        value = default
    value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    return value


def _native_graph_vkwr_rkv_dispatch(rows: int, hidden_size: int) -> bool:
    """VKWR-style row gate for stacked R/K/V native-graph decode.

    VKWR's automatic RKV path is used for one-row decode and medium tiny-row
    batches (roughly 4..64 rows), but not for rows 2/3.  Mirroring that rule
    avoids forcing a grouped path into shapes where three cuBLAS calls can be
    competitive or faster.
    """

    if _native_graph_rkv_policy() != "vkwr_auto":
        return False
    if rows <= 0 or hidden_size <= 0:
        return False
    min_hidden = _native_graph_int_env("RWKV7_NATIVE_GRAPH_RKV_MIN_HIDDEN", 1, lo=1)
    max_rows = _native_graph_int_env("RWKV7_NATIVE_GRAPH_RKV_MAX_ROWS", 64, lo=4, hi=4096)
    if hidden_size < min_hidden:
        return False
    return rows == 1 or (4 <= rows <= max_rows)


def _native_graph_rkv_project(
    xr: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    Rw: torch.Tensor,
    Kw: torch.Tensor,
    Vw: torch.Tensor,
    RKVw: torch.Tensor,
    rows: int,
    hidden_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project R/K/V with either separate linears or VKWR-style stacked bmm."""

    if not _native_graph_vkwr_rkv_dispatch(int(rows), int(hidden_size)) or RKVw.numel() == 0:
        return F.linear(xr, Rw), F.linear(xk, Kw), F.linear(xv, Vw)
    if xr.dim() == 1:
        flat = torch.stack(
            (
                xr.reshape(1, hidden_size),
                xk.reshape(1, hidden_size),
                xv.reshape(1, hidden_size),
            ),
            dim=0,
        )
        rkv = torch.bmm(flat, RKVw)
        return rkv[0, 0], rkv[1, 0], rkv[2, 0]
    flat = torch.stack(
        (
            xr.reshape(rows, hidden_size),
            xk.reshape(rows, hidden_size),
            xv.reshape(rows, hidden_size),
        ),
        dim=0,
    )
    rkv = torch.bmm(flat, RKVw)
    return rkv[0], rkv[1], rkv[2]


def _native_graph_fused_wag_lora_blocks() -> tuple[int, int, int]:
    """Return ``(block_m, block_r, block_k)`` for the W/A/G LoRA probe."""

    policy = _kernel_policy()
    defaults = tuple(getattr(policy, "wag_lora_blocks", (64, 64, 64)))
    return env_blocks(
        ("RWKV7_NATIVE_GRAPH_FUSED_WAG_LORA_BLOCK_M", "RWKV7_NATIVE_GRAPH_FUSED_WAG_LORA_BLOCK_R", "RWKV7_NATIVE_GRAPH_FUSED_WAG_LORA_BLOCK_K"),
        defaults,  # type: ignore[arg-type]
        (128, 128, 256),
    )


def _native_graph_fused_wavg_lora_blocks() -> tuple[int, int, int]:
    """Return ``(block_m, block_r, block_k)`` for the W/A/G/V-gate probe."""

    policy = _kernel_policy()
    defaults = tuple(getattr(policy, "wavg_lora_blocks", (64, 64, 64)))
    vals = []
    for name, fallback, default, upper in (
        ("RWKV7_NATIVE_GRAPH_FUSED_WAVG_LORA_BLOCK_M", "RWKV7_NATIVE_GRAPH_FUSED_WAG_LORA_BLOCK_M", defaults[0], 128),
        ("RWKV7_NATIVE_GRAPH_FUSED_WAVG_LORA_BLOCK_R", "RWKV7_NATIVE_GRAPH_FUSED_WAG_LORA_BLOCK_R", defaults[1], 128),
        ("RWKV7_NATIVE_GRAPH_FUSED_WAVG_LORA_BLOCK_K", "RWKV7_NATIVE_GRAPH_FUSED_WAG_LORA_BLOCK_K", defaults[2], 256),
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


def _recurrent_update_unbatched(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    state: torch.Tensor,
    H: int,
    N: int,
):
    if _native_graph_fused_recurrent_enabled():
        out, new_state = fused_recurrent_update(
            r.view(1, H, N),
            w.view(1, H, N),
            k.view(1, H, N),
            v.view(1, H, N),
            kk.view(1, H, N),
            a.view(1, H, N),
            state.view(1, H, N, N),
            block_n=N,
        )
        return out.reshape(H * N), new_state.reshape(H, N, N)
    vk = v.view(H, N, 1) @ k.view(H, 1, N)
    ab = (-kk).view(H, N, 1) @ (kk * a).view(H, 1, N)
    new_state = state * w.view(H, 1, N) + state @ ab.float() + vk.float()
    out = new_state.to(r.dtype) @ r.view(H, N, 1)
    return out.view(H * N), new_state


def _recurrent_update_batched(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    state: torch.Tensor,
    B: int,
    H: int,
    N: int,
):
    if _native_graph_fused_recurrent_enabled():
        out, new_state = fused_recurrent_update(
            r.view(B, H, N),
            w.view(B, H, N),
            k.view(B, H, N),
            v.view(B, H, N),
            kk.view(B, H, N),
            a.view(B, H, N),
            state,
            block_n=N,
        )
        return out.reshape(B, H * N), new_state
    vk = v.view(B, H, N, 1) @ k.view(B, H, 1, N)
    ab = (-kk).view(B, H, N, 1) @ (kk * a).view(B, H, 1, N)
    new_state = state * w.view(B, H, 1, N) + state @ ab.float() + vk.float()
    out = new_state.to(r.dtype) @ r.view(B, H, N, 1)
    return out.view(B, H * N), new_state


@torch.jit.script
def block_step(x: torch.Tensor, xpa: torch.Tensor, xpf: torch.Tensor,
               v_first: torch.Tensor, state: torch.Tensor,
               layer_id: int, H: int, N: int, eps: float, has_pre: int,
               pre_w: torch.Tensor, pre_b: torch.Tensor,
               an_w: torch.Tensor, an_b: torch.Tensor,
               fn_w: torch.Tensor, fn_b: torch.Tensor,
               x_r: torch.Tensor, x_w: torch.Tensor, x_k: torch.Tensor,
               x_v: torch.Tensor, x_a: torch.Tensor, x_g: torch.Tensor,
               k_k: torch.Tensor, k_a: torch.Tensor, r_k: torch.Tensor,
               Rw: torch.Tensor, Kw: torch.Tensor, Vw: torch.Tensor, Ow: torch.Tensor,
               w1: torch.Tensor, w2: torch.Tensor, w0: torch.Tensor,
               a1: torch.Tensor, a2: torch.Tensor, a0: torch.Tensor,
               v1: torch.Tensor, v2: torch.Tensor, v0: torch.Tensor,
               g1: torch.Tensor, g2: torch.Tensor,
               gn_w: torch.Tensor, gn_b: torch.Tensor,
               fx_k: torch.Tensor, fK: torch.Tensor, fV: torch.Tensor,
               RKVw: torch.Tensor):
    # --- block wiring (fuse_norm=False) ---
    if has_pre == 1:
        residual = F.layer_norm(x, [H * N], pre_w, pre_b, 1e-5)
    else:
        residual = x
    h = F.layer_norm(residual, [H * N], an_w, an_b, 1e-5)

    # --- TMix_one ---
    xx = xpa - h
    xpa = h
    xr = h + xx * x_r; xw = h + xx * x_w; xk = h + xx * x_k
    xv = h + xx * x_v; xa = h + xx * x_a; xg = h + xx * x_g
    r = F.linear(xr, Rw)
    w = F.linear(torch.tanh(F.linear(xw, w1)), w2, w0)
    k = F.linear(xk, Kw)
    v = F.linear(xv, Vw)
    a = torch.sigmoid(a0 + F.linear(F.linear(xa, a1), a2))
    g = F.linear(torch.sigmoid(F.linear(xg, g1)), g2)
    kk = F.normalize((k * k_k).view(H, N), dim=-1, p=2.0).view(H * N)
    k = k * (1 + (a - 1) * k_a)
    if layer_id == 0:
        v_first = v
    else:
        v = v + (v_first - v) * torch.sigmoid(v0 + F.linear(F.linear(xv, v1), v2))
    w = torch.exp(-0.606531 * torch.sigmoid(w.float()))
    vk = v.view(H, N, 1) @ k.view(H, 1, N)
    ab = (-kk).view(H, N, 1) @ (kk * a).view(H, 1, N)
    state = state * w.view(H, 1, N) + state @ ab.float() + vk.float()
    out = state.to(h.dtype) @ r.view(H, N, 1)
    out = out.view(H * N)
    out = F.group_norm(out.view(1, H * N), H, gn_w, gn_b, eps).view(H * N)
    sk = (r.view(H, N) * k.view(H, N) * r_k).sum(dim=-1, keepdim=True)
    out = out + (sk * v.view(H, N)).view(H * N)
    out = F.linear(out * g, Ow)
    x = residual + out

    # --- CMix_one ---
    residual = x
    h2 = F.layer_norm(x, [H * N], fn_w, fn_b, 1e-5)
    fxx = xpf - h2
    xpf = h2
    fk = h2 + fxx * fx_k
    fk = torch.relu(F.linear(fk, fK)) ** 2
    x = residual + F.linear(fk, fV)
    return x, xpa, xpf, v_first, state


@torch.jit.script
def block_step_batched(x: torch.Tensor, xpa: torch.Tensor, xpf: torch.Tensor,
                       v_first: torch.Tensor, state: torch.Tensor,
                       layer_id: int, H: int, N: int, eps: float, has_pre: int,
                       pre_w: torch.Tensor, pre_b: torch.Tensor,
                       an_w: torch.Tensor, an_b: torch.Tensor,
                       fn_w: torch.Tensor, fn_b: torch.Tensor,
                       x_r: torch.Tensor, x_w: torch.Tensor, x_k: torch.Tensor,
                       x_v: torch.Tensor, x_a: torch.Tensor, x_g: torch.Tensor,
                       k_k: torch.Tensor, k_a: torch.Tensor, r_k: torch.Tensor,
                       Rw: torch.Tensor, Kw: torch.Tensor, Vw: torch.Tensor, Ow: torch.Tensor,
                       w1: torch.Tensor, w2: torch.Tensor, w0: torch.Tensor,
                       a1: torch.Tensor, a2: torch.Tensor, a0: torch.Tensor,
                       v1: torch.Tensor, v2: torch.Tensor, v0: torch.Tensor,
                       g1: torch.Tensor, g2: torch.Tensor,
                       gn_w: torch.Tensor, gn_b: torch.Tensor,
                       fx_k: torch.Tensor, fK: torch.Tensor, fV: torch.Tensor,
                       RKVw: torch.Tensor):
    # Batched variant of block_step. Shapes:
    # x/xpa/xpf/v_first:[B,H*N], state:[B,H,N,N].
    B = x.shape[0]
    if has_pre == 1:
        residual = F.layer_norm(x, [H * N], pre_w, pre_b, 1e-5)
    else:
        residual = x
    h = F.layer_norm(residual, [H * N], an_w, an_b, 1e-5)

    xx = xpa - h
    xpa = h
    xr = h + xx * x_r; xw = h + xx * x_w; xk = h + xx * x_k
    xv = h + xx * x_v; xa = h + xx * x_a; xg = h + xx * x_g
    r = F.linear(xr, Rw)
    w = F.linear(torch.tanh(F.linear(xw, w1)), w2, w0)
    k = F.linear(xk, Kw)
    v = F.linear(xv, Vw)
    a = torch.sigmoid(a0 + F.linear(F.linear(xa, a1), a2))
    g = F.linear(torch.sigmoid(F.linear(xg, g1)), g2)
    kk = F.normalize((k * k_k).view(B, H, N), dim=-1, p=2.0).view(B, H * N)
    k = k * (1 + (a - 1) * k_a)
    if layer_id == 0:
        v_first = v
    else:
        v = v + (v_first - v) * torch.sigmoid(v0 + F.linear(F.linear(xv, v1), v2))
    w = torch.exp(-0.606531 * torch.sigmoid(w.float()))
    vk = v.view(B, H, N, 1) @ k.view(B, H, 1, N)
    ab = (-kk).view(B, H, N, 1) @ (kk * a).view(B, H, 1, N)
    state = state * w.view(B, H, 1, N) + state @ ab.float() + vk.float()
    out = state.to(h.dtype) @ r.view(B, H, N, 1)
    out = out.view(B, H * N)
    out = F.group_norm(out, H, gn_w, gn_b, eps).view(B, H * N)
    sk = (r.view(B, H, N) * k.view(B, H, N) * r_k).sum(dim=-1, keepdim=True)
    out = out + (sk * v.view(B, H, N)).view(B, H * N)
    out = F.linear(out * g, Ow)
    x = residual + out

    residual = x
    h2 = F.layer_norm(x, [H * N], fn_w, fn_b, 1e-5)
    fxx = xpf - h2
    xpf = h2
    fk = h2 + fxx * fx_k
    fk = torch.relu(F.linear(fk, fK)) ** 2
    x = residual + F.linear(fk, fV)
    return x, xpa, xpf, v_first, state


def extract(model):
    layers = model.model.layers
    H = layers[0].attn.num_heads
    N = layers[0].attn.head_dim
    eps = float(N * 1e-5)
    packs = []
    hidden = int(layers[0].attn.hidden_size)
    stack_rkv = _native_graph_rkv_policy() == "vkwr_auto"
    for i, layer in enumerate(layers):
        a = layer.attn
        ref = a.w_lora.lora[0].weight
        vl = getattr(a, "v_lora", None)
        v1 = vl.lora[0].weight if vl is not None else torch.zeros(1, ref.shape[1], device=ref.device, dtype=ref.dtype)
        v2 = vl.lora[2].weight if vl is not None else torch.zeros(hidden, 1, device=ref.device, dtype=ref.dtype)
        v0 = vl.lora[2].bias if vl is not None else torch.zeros(hidden, device=ref.device, dtype=ref.dtype)
        if hasattr(layer, "pre_norm"):
            pre_w, pre_b, has_pre = layer.pre_norm.weight, layer.pre_norm.bias, 1
        else:
            pre_w = torch.zeros(hidden, device=ref.device, dtype=ref.dtype)
            pre_b = torch.zeros(hidden, device=ref.device, dtype=ref.dtype)
            has_pre = 0
        packs.append((
            i, H, N, eps, has_pre,
            pre_w, pre_b, layer.attn_norm.weight, layer.attn_norm.bias,
            layer.ffn_norm.weight, layer.ffn_norm.bias,
            a.x_r.reshape(-1), a.x_w.reshape(-1), a.x_k.reshape(-1),
            a.x_v.reshape(-1), a.x_a.reshape(-1), a.x_g.reshape(-1),
            a.k_k, a.k_a, a.r_k,
            a.r_proj.weight, a.k_proj.weight, a.v_proj.weight, a.o_proj.weight,
            a.w_lora.lora[0].weight, a.w_lora.lora[2].weight, a.w_lora.lora[2].bias,
            a.a_lora.lora[0].weight, a.a_lora.lora[2].weight, a.a_lora.lora[2].bias,
            v1, v2, v0,
            a.g_lora.lora[0].weight, a.g_lora.lora[2].weight,
            a.g_norm.weight, a.g_norm.bias,
            layer.ffn.x_k, layer.ffn.key.weight, layer.ffn.value.weight,
            torch.stack((a.r_proj.weight.t(), a.k_proj.weight.t(), a.v_proj.weight.t())).contiguous()
            if stack_rkv
            else ref.new_empty((0,)),
        ))
    return packs, H, N, eps


def _ensure_rkv_pack(p):
    """Accept legacy 40-field packs and current packs with optional RKVw.

    The native-graph VKWR/RKV policy appended ``RKVw`` to layer packs.  Some
    synthetic tests and older remote-code checkpoints still build the previous
    40-field pack shape.  Keep those callers working by appending an empty
    tensor with the same device/dtype as the dense projection weights.
    """

    if len(p) == 41:
        return p
    if len(p) == 40:
        return (*p, p[20].new_empty((0,)))
    raise ValueError(f"unexpected native_jit layer pack length {len(p)}")


def _init(model, device, dtype):
    layers = model.model.layers
    n = len(layers)
    H = layers[0].attn.num_heads
    N = layers[0].attn.head_dim
    hid = layers[0].attn.hidden_size
    state = [torch.zeros(H, N, N, device=device, dtype=torch.float32) for _ in range(n)]
    xpa = [torch.zeros(hid, device=device, dtype=dtype) for _ in range(n)]
    xpf = [torch.zeros(hid, device=device, dtype=dtype) for _ in range(n)]
    v_first = torch.zeros(hid, device=device, dtype=dtype)
    return state, xpa, xpf, v_first


def _init_batched_from_packs(packs, batch_size: int, device, dtype):
    n = len(packs)
    H = int(packs[0][1])
    N = int(packs[0][2])
    hid = H * N
    state = [torch.zeros(batch_size, H, N, N, device=device, dtype=torch.float32) for _ in range(n)]
    xpa = [torch.zeros(batch_size, hid, device=device, dtype=dtype) for _ in range(n)]
    xpf = [torch.zeros(batch_size, hid, device=device, dtype=dtype) for _ in range(n)]
    return state, xpa, xpf


def step(model, x, state, xpa, xpf, v_first, packs):
    for p in packs:
        p = _ensure_rkv_pack(p)
        x, xpa[p[0]], xpf[p[0]], v_first, state[p[0]] = block_step(x, xpa[p[0]], xpf[p[0]], v_first, state[p[0]], *p)
    return x, state, xpa, xpf, v_first


def step_batched(model, x, state, xpa, xpf, v_first, packs):
    """Batched TorchScript block-step decode for native_model caches.

    Shapes mirror ``rwkv7_hf.native._step_token_batched``: x/xpa/xpf/v_first
    are ``[B, hidden]`` and recurrent state is ``[B, H, N, N]`` per layer.
    Keeping this helper in native_jit lets the experimental FLA-free model use
    the same reduced-dispatch H2 decode idea without importing the wrapper.
    """
    for p in packs:
        p = _ensure_rkv_pack(p)
        x, xpa[p[0]], xpf[p[0]], v_first, state[p[0]] = block_step_batched(
            x, xpa[p[0]], xpf[p[0]], v_first, state[p[0]], *p
        )
    return x, state, xpa, xpf, v_first


def _native_prefill_scan(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kk: torch.Tensor,
    a: torch.Tensor,
    state: torch.Tensor,
    B: int,
    T: int,
    H: int,
    N: int,
    *,
    w_is_raw: bool = False,
):
    """Run the recurrent prefill scan, using Triton only when explicitly enabled."""

    if w_is_raw and _native_prefill_fused_clampw_scan_enabled():
        scan_block_m = _native_prefill_scan_block_m(N)
        out, new_state = fused_recurrent_scan_clampw(
            r.view(B, T, H, N),
            w.view(B, T, H, N),
            k.view(B, T, H, N),
            v.view(B, T, H, N),
            kk.view(B, T, H, N),
            a.view(B, T, H, N),
            state,
            block_n=N,
            block_m=scan_block_m,
            num_warps=_native_prefill_scan_num_warps(N, scan_block_m),
        )
        return out.reshape(B, T, H * N), new_state

    if w_is_raw:
        w = torch.exp(-0.606531 * torch.sigmoid(w.float()))

    if _native_prefill_fused_scan_enabled():
        scan_block_m = _native_prefill_scan_block_m(N)
        out, new_state = fused_recurrent_scan(
            r.view(B, T, H, N),
            w.view(B, T, H, N),
            k.view(B, T, H, N),
            v.view(B, T, H, N),
            kk.view(B, T, H, N),
            a.view(B, T, H, N),
            state,
            block_n=N,
            block_m=scan_block_m,
            num_warps=_native_prefill_scan_num_warps(N, scan_block_m),
        )
        return out.reshape(B, T, H * N), new_state

    if _native_prefill_dplr_scan_enabled() and T > 1:
        out, new_state = dplr_chunk_scan(
            r.view(B, T, H, N),
            w.view(B, T, H, N),
            k.view(B, T, H, N),
            v.view(B, T, H, N),
            kk.view(B, T, H, N),
            a.view(B, T, H, N),
            state,
            chunk_size=_native_prefill_dplr_chunk_size(),
        )
        return out.reshape(B, T, H * N), new_state

    cur_state = state
    outs = []
    for t in range(T):
        out, cur_state = _recurrent_update_batched(
            r[:, t],
            w[:, t],
            k[:, t],
            v[:, t],
            kk[:, t],
            a[:, t],
            cur_state,
            B,
            H,
            N,
        )
        outs.append(out)
    return torch.stack(outs, dim=1), cur_state


def prefill(
    model,
    ids,
    packs,
    *,
    state=None,
    xpa=None,
    xpf=None,
    logits_to_keep: int | None = 1,
):
    """Layer-wise native RWKV-7 prefill over a full prompt.

    This is the first production-facing bridge for the fused recurrent scan
    prototype: it computes every layer over `[batch, tokens]` using vectorized
    projections and an optional fused recurrent scan instead of repeatedly
    calling the one-token decode path.  Returned state uses the native layout
    `[B,H,N,N]`; callers that expose HF/FLA cache state should transpose the
    final two dimensions, matching the native-graph decode runner.
    """

    base = model.model
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    if ids.dim() != 2:
        raise ValueError("native_jit.prefill expects ids shaped [batch, tokens]")
    B = int(ids.shape[0])
    T = int(ids.shape[1])
    if T <= 0:
        raise ValueError("native_jit.prefill requires at least one token")
    H = int(packs[0][1])
    N = int(packs[0][2])
    hidden = H * N
    dtype = base.embeddings.weight.dtype
    if state is None or xpa is None or xpf is None:
        state, xpa, xpf = _init_batched_from_packs(packs, B, ids.device, dtype)
    else:
        state = [s.to(device=ids.device, dtype=torch.float32).contiguous() for s in state]
        xpa = [s.to(device=ids.device, dtype=dtype).contiguous() for s in xpa]
        xpf = [s.to(device=ids.device, dtype=dtype).contiguous() for s in xpf]

    x = F.embedding(ids, base.embeddings.weight).reshape(B, T, hidden)
    v_first_seq = torch.zeros(B, T, hidden, device=ids.device, dtype=dtype)

    for p in packs:
        p = _ensure_rkv_pack(p)
        (i, H, N, eps, has_pre,
         pre_w, pre_b, an_w, an_b, fn_w, fn_b,
         x_r, x_w, x_k, x_v, x_a, x_g, k_k, k_a, r_k,
         Rw, Kw, Vw, Ow, w1, w2, w0, a1, a2, a0, v1, v2, v0, g1, g2,
         gn_w, gn_b, fx_k, fK, fV, RKVw) = p
        layer_idx = int(i)
        H = int(H)
        N = int(N)
        hidden = H * N

        residual = F.layer_norm(x, [hidden], pre_w, pre_b, 1e-5) if int(has_pre) == 1 else x
        h = F.layer_norm(residual, [hidden], an_w, an_b, 1e-5)
        prev_h = torch.cat([xpa[layer_idx].view(B, 1, hidden), h[:, :-1, :]], dim=1)
        if _native_prefill_fused_shift_mix_enabled():
            xr, xw, xk, xv, xa, xg = fused_attn_shift_mix(h, prev_h, x_r, x_w, x_k, x_v, x_a, x_g)
        else:
            xx = prev_h - h
            xr = h + xx * x_r.view(1, 1, hidden)
            xw = h + xx * x_w.view(1, 1, hidden)
            xk = h + xx * x_k.view(1, 1, hidden)
            xv = h + xx * x_v.view(1, 1, hidden)
            xa = h + xx * x_a.view(1, 1, hidden)
            xg = h + xx * x_g.view(1, 1, hidden)

        v_gate = None
        use_prefill_projection = _native_prefill_fused_projection_enabled(B * T)
        if use_prefill_projection:
            block_m, block_r, block_k = _native_prefill_fused_projection_blocks()
            if layer_idx == 0:
                r, k, v, w, a, g = fused_rkv_wag_projection(
                    xr,
                    xk,
                    xv,
                    xw,
                    xa,
                    xg,
                    Rw,
                    Kw,
                    Vw,
                    w1,
                    a1,
                    g1,
                    w2,
                    a2,
                    g2,
                    w0,
                    a0,
                    None,
                    block_m=block_m,
                    block_r=block_r,
                    block_k=block_k,
                )
                a = torch.sigmoid(a)
            else:
                r, k, v, w, a, g, v_gate = fused_rkv_wavg_projection(
                    xr,
                    xk,
                    xv,
                    xw,
                    xa,
                    xg,
                    Rw,
                    Kw,
                    Vw,
                    w1,
                    a1,
                    g1,
                    v1,
                    w2,
                    a2,
                    g2,
                    v2,
                    w0,
                    a0,
                    None,
                    v0,
                    block_m=block_m,
                    block_r=block_r,
                    block_k=block_k,
                )
                a = torch.sigmoid(a)
                v_gate = torch.sigmoid(v_gate)
        else:
            r = F.linear(xr, Rw)
            k = F.linear(xk, Kw)
            v = F.linear(xv, Vw)
        use_prefill_wavg_lora = (
            not use_prefill_projection and layer_idx > 0 and _native_prefill_fused_wavg_lora_enabled(B * T)
        )
        if use_prefill_wavg_lora:
            block_m, block_r, block_k = _native_prefill_fused_wavg_lora_blocks()
            w, a, g, v_gate = fused_wavg_lora(
                xw.reshape(B * T, hidden),
                xa.reshape(B * T, hidden),
                xg.reshape(B * T, hidden),
                xv.reshape(B * T, hidden),
                w1,
                a1,
                g1,
                v1,
                w2,
                a2,
                g2,
                v2,
                w0,
                a0,
                None,
                v0,
                block_m=block_m,
                block_r=block_r,
                block_k=block_k,
            )
            w = w.view(B, T, hidden)
            a = torch.sigmoid(a.view(B, T, hidden))
            g = g.view(B, T, hidden)
            v_gate = v_gate.view(B, T, hidden)
        elif not use_prefill_projection:
            w = F.linear(torch.tanh(F.linear(xw, w1)), w2, w0)
            a = torch.sigmoid(a0 + F.linear(F.linear(xa, a1), a2))
            g = F.linear(torch.sigmoid(F.linear(xg, g1)), g2)
            if layer_idx != 0:
                v_gate = torch.sigmoid(v0 + F.linear(F.linear(xv, v1), v2))
        use_fused_state_scan_output = _native_prefill_fused_state_scan_output_enabled()
        use_fused_state_scan_correction = (
            _native_prefill_fused_state_scan_correction_enabled() and not use_fused_state_scan_output
        )
        use_fused_state_scan_raw_output = (
            _native_prefill_fused_state_scan_raw_output_enabled()
            and not use_fused_state_scan_output
            and not use_fused_state_scan_correction
        )
        use_fused_state_scan_sk_output = (
            _native_prefill_fused_state_scan_sk_output_enabled()
            and not use_fused_state_scan_output
            and not use_fused_state_scan_correction
            and not use_fused_state_scan_raw_output
        )
        use_fused_scan_output = (
            _native_prefill_fused_scan_output_enabled()
            and not use_fused_state_scan_output
            and not use_fused_state_scan_correction
            and not use_fused_state_scan_raw_output
            and not use_fused_state_scan_sk_output
        )
        use_clampw_scan = (
            _native_prefill_fused_clampw_scan_enabled()
            and not use_fused_scan_output
            and not use_fused_state_scan_output
            and not use_fused_state_scan_correction
            and not use_fused_state_scan_raw_output
            and not use_fused_state_scan_sk_output
        )
        use_fused_state_scan = (
            _native_prefill_fused_state_scan_enabled()
            and not use_fused_scan_output
            and not use_fused_state_scan_output
            and not use_fused_state_scan_correction
            and not use_fused_state_scan_raw_output
            and not use_fused_state_scan_sk_output
        )
        if use_clampw_scan and _native_prefill_fused_state_prep_enabled() and fused_prefill_kv_kk_prep is None:
            use_clampw_scan = False
        state_scan_done = False
        state_scan_output_done = False
        state_scan_correction_done = False
        state_scan_raw_output_done = False
        state_scan_sk_output_done = False
        correction = None
        sk_scale = None
        if use_fused_state_scan_output:
            state_scan_num_warps = _native_prefill_scan_num_warps(N, N)
            if layer_idx == 0:
                out, new_state = fused_recurrent_scan_state_prep_output_prepare(
                    r.view(B, T, H, N),
                    w.view(B, T, H, N),
                    k.view(B, T, H, N),
                    v.view(B, T, H, N),
                    a.view(B, T, H, N),
                    state[layer_idx],
                    k_k,
                    k_a,
                    g.view(B, T, H, N),
                    r_k,
                    gn_w,
                    gn_b,
                    eps=eps,
                    block_n=N,
                    num_warps=state_scan_num_warps,
                )
                v_first_seq = v
            else:
                out, new_state = fused_recurrent_scan_state_prep_output_prepare(
                    r.view(B, T, H, N),
                    w.view(B, T, H, N),
                    k.view(B, T, H, N),
                    v.view(B, T, H, N),
                    a.view(B, T, H, N),
                    state[layer_idx],
                    k_k,
                    k_a,
                    g.view(B, T, H, N),
                    r_k,
                    gn_w,
                    gn_b,
                    eps=eps,
                    v_first=v_first_seq.view(B, T, H, N),
                    v_gate=v_gate.view(B, T, H, N),
                    block_n=N,
                    num_warps=state_scan_num_warps,
                )
            out = out.reshape(B, T, hidden)
            state_scan_done = True
            state_scan_output_done = True
        elif use_fused_state_scan_correction:
            state_scan_block_m = _native_prefill_scan_block_m(N)
            state_scan_num_warps = _native_prefill_scan_num_warps(N, state_scan_block_m)
            state_scan_num_stages = _native_prefill_scan_num_stages()
            if layer_idx == 0:
                out, new_state, correction = fused_recurrent_scan_state_prep_correction(
                    r.view(B, T, H, N),
                    w.view(B, T, H, N),
                    k.view(B, T, H, N),
                    v.view(B, T, H, N),
                    a.view(B, T, H, N),
                    state[layer_idx],
                    k_k,
                    k_a,
                    r_k,
                    block_n=N,
                    block_m=state_scan_block_m,
                    num_warps=state_scan_num_warps,
                    num_stages=state_scan_num_stages,
                )
                # Layer 0 adjusted V is the raw V projection, so keep it for
                # later layer V interpolation without materializing a scan V_out.
                v_first_seq = v
            else:
                out, new_state, correction = fused_recurrent_scan_state_prep_correction(
                    r.view(B, T, H, N),
                    w.view(B, T, H, N),
                    k.view(B, T, H, N),
                    v.view(B, T, H, N),
                    a.view(B, T, H, N),
                    state[layer_idx],
                    k_k,
                    k_a,
                    r_k,
                    v_first=v_first_seq.view(B, T, H, N),
                    v_gate=v_gate.view(B, T, H, N),
                    block_n=N,
                    block_m=state_scan_block_m,
                    num_warps=state_scan_num_warps,
                    num_stages=state_scan_num_stages,
                )
            out = out.reshape(B, T, hidden)
            correction = correction.reshape(B, T, hidden)
            state_scan_done = True
            state_scan_correction_done = True
        elif use_fused_state_scan_raw_output:
            state_scan_block_m = _native_prefill_scan_block_m(N)
            state_scan_num_warps = _native_prefill_scan_num_warps(N, state_scan_block_m)
            state_scan_num_stages = _native_prefill_scan_num_stages()
            if layer_idx == 0:
                out, new_state = fused_recurrent_scan_state_prep_nokv(
                    r.view(B, T, H, N),
                    w.view(B, T, H, N),
                    k.view(B, T, H, N),
                    v.view(B, T, H, N),
                    a.view(B, T, H, N),
                    state[layer_idx],
                    k_k,
                    k_a,
                    block_n=N,
                    block_m=state_scan_block_m,
                    num_warps=state_scan_num_warps,
                    num_stages=state_scan_num_stages,
                )
                # Raw V is also the adjusted V for layer 0.
                v_first_seq = v
            else:
                out, new_state = fused_recurrent_scan_state_prep_nokv(
                    r.view(B, T, H, N),
                    w.view(B, T, H, N),
                    k.view(B, T, H, N),
                    v.view(B, T, H, N),
                    a.view(B, T, H, N),
                    state[layer_idx],
                    k_k,
                    k_a,
                    v_first=v_first_seq.view(B, T, H, N),
                    v_gate=v_gate.view(B, T, H, N),
                    block_n=N,
                    block_m=state_scan_block_m,
                    num_warps=state_scan_num_warps,
                    num_stages=state_scan_num_stages,
                )
            out = out.reshape(B, T, hidden)
            state_scan_done = True
            state_scan_raw_output_done = True
        elif use_fused_state_scan_sk_output:
            state_scan_block_m = _native_prefill_scan_block_m(N)
            state_scan_num_warps = _native_prefill_scan_num_warps(N, state_scan_block_m)
            state_scan_num_stages = _native_prefill_scan_num_stages()
            if layer_idx == 0:
                out, new_state, sk_scale = fused_recurrent_scan_state_prep_sk(
                    r.view(B, T, H, N),
                    w.view(B, T, H, N),
                    k.view(B, T, H, N),
                    v.view(B, T, H, N),
                    a.view(B, T, H, N),
                    state[layer_idx],
                    k_k,
                    k_a,
                    r_k,
                    block_n=N,
                    num_warps=state_scan_num_warps,
                    num_stages=state_scan_num_stages,
                )
                # Layer 0 adjusted V is the raw V projection.
                v_first_seq = v
            else:
                out, new_state, sk_scale = fused_recurrent_scan_state_prep_sk(
                    r.view(B, T, H, N),
                    w.view(B, T, H, N),
                    k.view(B, T, H, N),
                    v.view(B, T, H, N),
                    a.view(B, T, H, N),
                    state[layer_idx],
                    k_k,
                    k_a,
                    r_k,
                    v_first=v_first_seq.view(B, T, H, N),
                    v_gate=v_gate.view(B, T, H, N),
                    block_n=N,
                    num_warps=state_scan_num_warps,
                    num_stages=state_scan_num_stages,
                )
            out = out.reshape(B, T, hidden)
            sk_scale = sk_scale.reshape(B * T, H)
            state_scan_done = True
            state_scan_sk_output_done = True
        elif use_fused_state_scan:
            state_scan_block_m = _native_prefill_scan_block_m(N)
            state_scan_num_warps = _native_prefill_scan_num_warps(N, state_scan_block_m)
            state_scan_num_stages = _native_prefill_scan_num_stages()
            state_scan_algebraic_output = _native_prefill_scan_algebraic_output_enabled()
            state_scan_nomask64 = _native_prefill_scan_nomask64_enabled()
            use_cuda_state_scan = (
                _native_prefill_cuda_state_scan_enabled()
                and N == 64
                and state_scan_block_m == 64
                and x.dtype == torch.float16
            )
            cuda_state_scan_lanes = _native_prefill_cuda_state_scan_lanes_per_row() if use_cuda_state_scan else 1
            cuda_state_scan_precompute = (
                _native_prefill_cuda_state_scan_precompute_enabled() if use_cuda_state_scan else False
            )
            cuda_state_scan_precompute_mode = (
                _native_prefill_cuda_state_scan_precompute_mode() if use_cuda_state_scan else "none"
            )
            cuda_state_scan_rows_per_block = (
                _native_prefill_cuda_state_scan_rows_per_block() if use_cuda_state_scan else 1
            )
            cuda_state_scan_schedule = _native_prefill_cuda_state_scan_schedule() if use_cuda_state_scan else "default"
            state_scan_precompute_w = _native_prefill_scan_precompute_w_enabled() and not use_cuda_state_scan
            state_scan_precompute_w_dtype = _native_prefill_scan_precompute_w_dtype()
            w_for_state_scan = w
            if state_scan_precompute_w:
                w_for_state_scan = torch.sigmoid(w.float()).mul_(-0.606531).exp_()
                if state_scan_precompute_w_dtype == "input":
                    w_for_state_scan = w_for_state_scan.to(dtype=w.dtype)
            if use_cuda_state_scan and layer_idx == 0:
                out, new_state, k, v = cuda_state_scan_prep(
                    r.view(B, T, H, N),
                    w.view(B, T, H, N),
                    k.view(B, T, H, N),
                    v.view(B, T, H, N),
                    a.view(B, T, H, N),
                    state[layer_idx],
                    k_k,
                    k_a,
                    lanes_per_row=cuda_state_scan_lanes,
                    precompute_vector=cuda_state_scan_precompute,
                    precompute_mode=cuda_state_scan_precompute_mode,
                    rows_per_block=cuda_state_scan_rows_per_block,
                    schedule=cuda_state_scan_schedule,
                )
                v_first_seq = v.reshape(B, T, hidden)
            elif use_cuda_state_scan:
                out, new_state, k, v = cuda_state_scan_prep(
                    r.view(B, T, H, N),
                    w.view(B, T, H, N),
                    k.view(B, T, H, N),
                    v.view(B, T, H, N),
                    a.view(B, T, H, N),
                    state[layer_idx],
                    k_k,
                    k_a,
                    v_first=v_first_seq.view(B, T, H, N),
                    v_gate=v_gate.view(B, T, H, N),
                    lanes_per_row=cuda_state_scan_lanes,
                    precompute_vector=cuda_state_scan_precompute,
                    precompute_mode=cuda_state_scan_precompute_mode,
                    rows_per_block=cuda_state_scan_rows_per_block,
                    schedule=cuda_state_scan_schedule,
                )
            elif layer_idx == 0:
                out, new_state, k, v = fused_recurrent_scan_state_prep(
                    r.view(B, T, H, N),
                    w_for_state_scan.view(B, T, H, N),
                    k.view(B, T, H, N),
                    v.view(B, T, H, N),
                    a.view(B, T, H, N),
                    state[layer_idx],
                    k_k,
                    k_a,
                    block_n=N,
                    block_m=state_scan_block_m,
                    num_warps=state_scan_num_warps,
                    num_stages=state_scan_num_stages,
                    algebraic_output=state_scan_algebraic_output,
                    nomask64=state_scan_nomask64,
                    precomputed_w=state_scan_precompute_w,
                )
                v_first_seq = v.reshape(B, T, hidden)
            else:
                out, new_state, k, v = fused_recurrent_scan_state_prep(
                    r.view(B, T, H, N),
                    w_for_state_scan.view(B, T, H, N),
                    k.view(B, T, H, N),
                    v.view(B, T, H, N),
                    a.view(B, T, H, N),
                    state[layer_idx],
                    k_k,
                    k_a,
                    v_first=v_first_seq.view(B, T, H, N),
                    v_gate=v_gate.view(B, T, H, N),
                    block_n=N,
                    block_m=state_scan_block_m,
                    num_warps=state_scan_num_warps,
                    num_stages=state_scan_num_stages,
                    algebraic_output=state_scan_algebraic_output,
                    nomask64=state_scan_nomask64,
                    precomputed_w=state_scan_precompute_w,
                )
            out = out.reshape(B, T, hidden)
            k = k.reshape(B, T, hidden)
            v = v.reshape(B, T, hidden)
            state_scan_done = True
        elif _native_prefill_fused_state_prep_enabled():
            if use_clampw_scan:
                if layer_idx == 0:
                    k, v, kk = fused_prefill_kv_kk_prep(
                        k,
                        v,
                        a,
                        k_k,
                        k_a,
                        num_heads=H,
                        head_dim=N,
                    )
                    v_first_seq = v
                else:
                    k, v, kk = fused_prefill_kv_kk_prep(
                        k,
                        v,
                        a,
                        k_k,
                        k_a,
                        v_first=v_first_seq,
                        v_gate=v_gate,
                        num_heads=H,
                        head_dim=N,
                    )
            elif layer_idx == 0:
                w, k, v, kk = fused_prefill_state_prep(
                    w,
                    k,
                    v,
                    a,
                    k_k,
                    k_a,
                    num_heads=H,
                    head_dim=N,
                    w_out_dtype=_native_prefill_state_prep_w_dtype(),
                )
                v_first_seq = v
            else:
                w, k, v, kk = fused_prefill_state_prep(
                    w,
                    k,
                    v,
                    a,
                    k_k,
                    k_a,
                    v_first=v_first_seq,
                    v_gate=v_gate,
                    num_heads=H,
                    head_dim=N,
                    w_out_dtype=_native_prefill_state_prep_w_dtype(),
                )
        else:
            kk = F.normalize((k * k_k.view(1, 1, hidden)).view(B, T, H, N), dim=-1, p=2.0).view(B, T, hidden)
            k = k * (1 + (a - 1) * k_a.view(1, 1, hidden))
            if layer_idx == 0:
                v_first_seq = v
            else:
                v = v + (v_first_seq - v) * v_gate
            if not use_clampw_scan:
                w = torch.exp(-0.606531 * torch.sigmoid(w.float()))

        if use_fused_state_scan_output:
            pass
        elif use_fused_scan_output:
            out, new_state = fused_recurrent_scan_output_prepare(
                r.view(B, T, H, N),
                w.view(B, T, H, N),
                k.view(B, T, H, N),
                v.view(B, T, H, N),
                kk.view(B, T, H, N),
                a.view(B, T, H, N),
                state[layer_idx],
                g.view(B, T, H, N),
                r_k,
                gn_w,
                gn_b,
                eps=eps,
                block_n=N,
            )
            out = out.reshape(B, T, hidden)
        elif not state_scan_done:
            out, new_state = _native_prefill_scan(r, w, k, v, kk, a, state[layer_idx], B, T, H, N, w_is_raw=use_clampw_scan)
        out_projected = False
        if state_scan_output_done or use_fused_scan_output:
            pass
        elif state_scan_correction_done:
            out = fused_attn_output_prepare_from_correction(
                out.reshape(B * T, hidden),
                correction.reshape(B * T, hidden),
                g.reshape(B * T, hidden),
                gn_w,
                gn_b,
                num_heads=H,
                head_v_dim=N,
                eps=eps,
            ).view(B, T, hidden)
        elif state_scan_raw_output_done:
            if layer_idx == 0:
                out = fused_attn_output_prepare_raw_kv(
                    out.reshape(B * T, hidden),
                    r.reshape(B * T, H, N),
                    k.reshape(B * T, H, N),
                    v.reshape(B * T, H, N),
                    a.reshape(B * T, H, N),
                    g.reshape(B * T, hidden),
                    k_a.view(H, N),
                    r_k,
                    gn_w,
                    gn_b,
                    num_heads=H,
                    head_dim=N,
                    head_v_dim=N,
                    eps=eps,
                ).view(B, T, hidden)
            else:
                out = fused_attn_output_prepare_raw_kv(
                    out.reshape(B * T, hidden),
                    r.reshape(B * T, H, N),
                    k.reshape(B * T, H, N),
                    v.reshape(B * T, H, N),
                    a.reshape(B * T, H, N),
                    g.reshape(B * T, hidden),
                    k_a.view(H, N),
                    r_k,
                    gn_w,
                    gn_b,
                    v_first=v_first_seq.reshape(B * T, H, N),
                    v_gate=v_gate.reshape(B * T, H, N),
                    num_heads=H,
                    head_dim=N,
                    head_v_dim=N,
                    eps=eps,
                ).view(B, T, hidden)
        elif state_scan_sk_output_done:
            if layer_idx == 0:
                out = fused_attn_output_prepare_from_sk_raw_v(
                    out.reshape(B * T, hidden),
                    sk_scale,
                    v.reshape(B * T, H, N),
                    g.reshape(B * T, hidden),
                    gn_w,
                    gn_b,
                    num_heads=H,
                    head_v_dim=N,
                    eps=eps,
                ).view(B, T, hidden)
            else:
                out = fused_attn_output_prepare_from_sk_raw_v(
                    out.reshape(B * T, hidden),
                    sk_scale,
                    v.reshape(B * T, H, N),
                    g.reshape(B * T, hidden),
                    gn_w,
                    gn_b,
                    v_first=v_first_seq.reshape(B * T, H, N),
                    v_gate=v_gate.reshape(B * T, H, N),
                    num_heads=H,
                    head_v_dim=N,
                    eps=eps,
                ).view(B, T, hidden)
        elif _native_prefill_fused_output_project_enabled():
            out = fused_attn_output_project(
                out.reshape(B * T, hidden),
                r.reshape(B * T, H, N),
                k.reshape(B * T, H, N),
                v.reshape(B * T, H, N),
                g.reshape(B * T, hidden),
                r_k,
                gn_w,
                gn_b,
                Ow,
                None,
                num_heads=H,
                head_dim=N,
                head_v_dim=N,
                eps=eps,
                block_m=_native_prefill_fused_output_project_block_m(),
            ).view(B, T, hidden)
            out_projected = True
        elif _native_prefill_fused_output_enabled():
            out = fused_attn_output_prepare(
                out.reshape(B * T, hidden),
                r.reshape(B * T, H, N),
                k.reshape(B * T, H, N),
                v.reshape(B * T, H, N),
                g.reshape(B * T, hidden),
                r_k,
                gn_w,
                gn_b,
                num_heads=H,
                head_dim=N,
                head_v_dim=N,
                eps=eps,
            ).view(B, T, hidden)
        else:
            out = F.group_norm(out.reshape(B * T, hidden), H, gn_w, gn_b, eps).view(B, T, hidden)
            sk = (r.view(B, T, H, N) * k.view(B, T, H, N) * r_k.view(1, 1, H, N)).sum(dim=-1, keepdim=True)
            out = (out + (sk * v.view(B, T, H, N)).view(B, T, hidden)) * g
        if not out_projected:
            out = F.linear(out, Ow)
        x = residual + out
        xpa[layer_idx] = h[:, -1, :].contiguous()
        state[layer_idx] = new_state.contiguous()

        residual = x
        h2 = F.layer_norm(x, [hidden], fn_w, fn_b, 1e-5)
        prev_h2 = torch.cat([xpf[layer_idx].view(B, 1, hidden), h2[:, :-1, :]], dim=1)
        fxx = prev_h2 - h2
        fk = h2 + fxx * fx_k.view(1, 1, hidden)
        fk = torch.relu(F.linear(fk, fK)) ** 2
        x = residual + F.linear(fk, fV)
        xpf[layer_idx] = h2[:, -1, :].contiguous()

    x = F.layer_norm(x, [hidden], base.norm.weight, base.norm.bias, 1e-5)
    keep = T if logits_to_keep is None or int(logits_to_keep) <= 0 else min(int(logits_to_keep), T)
    logits = F.linear(x[:, -keep:, :], model.lm_head.weight, model.lm_head.bias)
    return logits, state, xpa, xpf


def forward(model, ids, packs):
    base = model.model
    H, N = packs[0][1], packs[0][2]
    state, xpa, xpf, v_first = _init(model, ids.device, base.embeddings.weight.dtype)
    x = None
    for t in range(ids.shape[1]):
        x = F.embedding(ids[0, t:t + 1], base.embeddings.weight).reshape(-1)
        x, state, xpa, xpf, v_first = step(model, x, state, xpa, xpf, v_first, packs)
    x = F.layer_norm(x, [H * N], base.norm.weight, base.norm.bias, 1e-5)
    return F.linear(x, model.lm_head.weight)


def decode_speed(model, ids, packs, n=128):
    import time
    base = model.model
    H, N = packs[0][1], packs[0][2]
    state, xpa, xpf, v_first = _init(model, ids.device, base.embeddings.weight.dtype)
    emb = base.embeddings.weight
    head = model.lm_head.weight
    norm_w = base.norm.weight
    norm_b = base.norm.bias
    x = None
    for t in range(ids.shape[1]):
        x = F.embedding(ids[0, t:t + 1], emb).reshape(-1)
        x, state, xpa, xpf, v_first = step(model, x, state, xpa, xpf, v_first, packs)
    nx = F.linear(F.layer_norm(x, [H * N], norm_w, norm_b, 1e-5), head).argmax()
    with torch.no_grad():
        for _ in range(5):
            x = F.embedding(nx.reshape(1, 1), emb).reshape(-1)
            x, state, xpa, xpf, v_first = step(model, x, state, xpa, xpf, v_first, packs)
            nx = F.linear(F.layer_norm(x, [H * N], norm_w, norm_b, 1e-5), head).argmax()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(n):
            x = F.embedding(nx.reshape(1, 1), emb).reshape(-1)
            x, state, xpa, xpf, v_first = step(model, x, state, xpa, xpf, v_first, packs)
            nx = F.linear(F.layer_norm(x, [H * N], norm_w, norm_b, 1e-5), head).argmax()
        torch.cuda.synchronize(); dt = time.time() - t0
    return n / dt


def _block_ip(x, state, xpa, xpf, v_first, p):
    """In-place (eager) block step for CUDA-graph capture: state/xpa/xpf/v_first
    are fixed buffers updated in place. Same math as block_step."""
    (i, H, N, eps, has_pre,
     pre_w, pre_b, an_w, an_b, fn_w, fn_b,
     x_r, x_w, x_k, x_v, x_a, x_g, k_k, k_a, r_k,
     Rw, Kw, Vw, Ow, w1, w2, w0, a1, a2, a0, v1, v2, v0, g1, g2,
     gn_w, gn_b, fx_k, fK, fV, RKVw) = p
    residual = F.layer_norm(x, [H * N], pre_w, pre_b, 1e-5) if has_pre else x
    h = F.layer_norm(residual, [H * N], an_w, an_b, 1e-5)
    xx = xpa - h
    xr = h + xx * x_r; xw = h + xx * x_w; xk = h + xx * x_k
    xv = h + xx * x_v; xa = h + xx * x_a; xg = h + xx * x_g
    v_gate = None
    if _native_graph_fused_projection_enabled():
        r, k, v, w, a, g, v_gate = fused_rkv_wavg_projection(
            xr.view(1, H * N),
            xk.view(1, H * N),
            xv.view(1, H * N),
            xw.view(1, H * N),
            xa.view(1, H * N),
            xg.view(1, H * N),
            Rw,
            Kw,
            Vw,
            w1,
            a1,
            g1,
            v1,
            w2,
            a2,
            g2,
            v2,
            w0,
            a0,
            None,
            v0,
        )
        r = r.view(H * N)
        k = k.view(H * N)
        v = v.view(H * N)
        w = w.view(H * N)
        a = torch.sigmoid(a.view(H * N))
        g = g.view(H * N)
        v_gate = torch.sigmoid(v_gate.view(H * N))
    elif _native_graph_fused_wavg_lora_enabled():
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, 1, H * N)
        if i == 0:
            w = F.linear(torch.tanh(F.linear(xw, w1)), w2, w0)
            a = a0 + F.linear(F.linear(xa, a1), a2)
            g = F.linear(torch.sigmoid(F.linear(xg, g1)), g2)
        else:
            block_m, block_r, block_k = _native_graph_fused_wavg_lora_blocks()
            w, a, g, v_gate = fused_wavg_lora(
                xw.view(1, H * N),
                xa.view(1, H * N),
                xg.view(1, H * N),
                xv.view(1, H * N),
                w1,
                a1,
                g1,
                v1,
                w2,
                a2,
                g2,
                v2,
                w0,
                a0,
                None,
                v0,
                block_m=block_m,
                block_r=block_r,
                block_k=block_k,
            )
            w = w.view(H * N)
            a = a.view(H * N)
            g = g.view(H * N)
            v_gate = v_gate.view(H * N)
        a = torch.sigmoid(a)
    elif _native_graph_fused_wag_lora_enabled():
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, 1, H * N)
        block_m, block_r, block_k = _native_graph_fused_wag_lora_blocks()
        w, a, g = fused_wag_lora(
            xw.view(1, H * N),
            xa.view(1, H * N),
            xg.view(1, H * N),
            w1,
            a1,
            g1,
            w2,
            a2,
            g2,
            w0,
            a0,
            None,
            block_m=block_m,
            block_r=block_r,
            block_k=block_k,
        )
        w = w.view(H * N)
        a = torch.sigmoid(a.view(H * N))
        g = g.view(H * N)
    else:
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, 1, H * N)
        w = F.linear(torch.tanh(F.linear(xw, w1)), w2, w0)
        a = torch.sigmoid(a0 + F.linear(F.linear(xa, a1), a2))
        g = F.linear(torch.sigmoid(F.linear(xg, g1)), g2)
    kk = F.normalize((k * k_k).view(H, N), dim=-1, p=2.0).view(H * N)
    k = k * (1 + (a - 1) * k_a)
    if i == 0:
        v_first.copy_(v)
    else:
        if v_gate is None:
            v_gate = torch.sigmoid(v0 + F.linear(F.linear(xv, v1), v2))
        v = v + (v_first - v) * v_gate
    w = torch.exp(-0.606531 * torch.sigmoid(w.float()))
    if _native_graph_fused_recurrent_output_enabled():
        out, new_state = fused_recurrent_output_prepare(
            r.view(1, H, N),
            w.view(1, H, N),
            k.view(1, H, N),
            v.view(1, H, N),
            kk.view(1, H, N),
            a.view(1, H, N),
            state.view(1, H, N, N),
            g.view(1, H, N),
            r_k,
            gn_w,
            gn_b,
            eps=eps,
            block_n=N,
        )
        out = out.view(H * N)
        new_state = new_state.view(H, N, N)
    else:
        out, new_state = _recurrent_update_unbatched(r, w, k, v, kk, a, state, H, N)
    if _native_graph_fused_recurrent_output_enabled():
        out = F.linear(out, Ow)
    elif _native_graph_fused_output_project_enabled():
        out = fused_attn_output_project(
            out.view(1, H * N),
            r.view(1, H, N),
            k.view(1, H, N),
            v.view(1, H, N),
            g.view(1, H * N),
            r_k,
            gn_w,
            gn_b,
            Ow,
            None,
            num_heads=H,
            head_dim=N,
            head_v_dim=N,
            eps=eps,
            block_m=_native_graph_fused_output_project_block_m(),
        ).view(H * N)
    elif _native_graph_fused_output_enabled():
        out = fused_attn_output_prepare(
            out.view(1, H * N),
            r.view(1, H, N),
            k.view(1, H, N),
            v.view(1, H, N),
            g.view(1, H * N),
            r_k,
            gn_w,
            gn_b,
            num_heads=H,
            head_dim=N,
            head_v_dim=N,
            eps=eps,
        ).view(H * N)
        out = F.linear(out, Ow)
    else:
        out = F.group_norm(out.view(1, H * N), H, gn_w, gn_b, eps).view(H * N)
        sk = (r.view(H, N) * k.view(H, N) * r_k).sum(dim=-1, keepdim=True)
        out = (out + (sk * v.view(H, N)).view(H * N)) * g
        out = F.linear(out, Ow)
    xpa.copy_(h)
    state.copy_(new_state)
    x = residual + out
    residual = x
    h2 = F.layer_norm(x, [H * N], fn_w, fn_b, 1e-5)
    fxx = xpf - h2
    fk = h2 + fxx * fx_k
    fk = torch.relu(F.linear(fk, fK)) ** 2
    xpf.copy_(h2)
    return residual + F.linear(fk, fV)


def _block_ip_batched(x, state, xpa, xpf, v_first, p):
    """In-place batched block step for CUDA-graph capture.

    Shapes:
      x/xpa/xpf/v_first: [B, H*N]
      state: [B, H, N, N]

    This mirrors `block_step_batched` but writes recurrent/cache buffers in
    place so a captured CUDA graph can replay across decode tokens.
    """
    (i, H, N, eps, has_pre,
     pre_w, pre_b, an_w, an_b, fn_w, fn_b,
     x_r, x_w, x_k, x_v, x_a, x_g, k_k, k_a, r_k,
     Rw, Kw, Vw, Ow, w1, w2, w0, a1, a2, a0, v1, v2, v0, g1, g2,
     gn_w, gn_b, fx_k, fK, fV, RKVw) = p
    B = x.shape[0]
    residual = F.layer_norm(x, [H * N], pre_w, pre_b, 1e-5) if has_pre else x
    h = F.layer_norm(residual, [H * N], an_w, an_b, 1e-5)
    xx = xpa - h
    xr = h + xx * x_r; xw = h + xx * x_w; xk = h + xx * x_k
    xv = h + xx * x_v; xa = h + xx * x_a; xg = h + xx * x_g
    v_gate = None
    if _native_graph_fused_projection_enabled():
        r, k, v, w, a, g, v_gate = fused_rkv_wavg_projection(
            xr,
            xk,
            xv,
            xw,
            xa,
            xg,
            Rw,
            Kw,
            Vw,
            w1,
            a1,
            g1,
            v1,
            w2,
            a2,
            g2,
            v2,
            w0,
            a0,
            None,
            v0,
        )
        a = torch.sigmoid(a)
        v_gate = torch.sigmoid(v_gate)
    elif _native_graph_fused_wavg_lora_enabled():
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, B, H * N)
        if i == 0:
            w = F.linear(torch.tanh(F.linear(xw, w1)), w2, w0)
            a = a0 + F.linear(F.linear(xa, a1), a2)
            g = F.linear(torch.sigmoid(F.linear(xg, g1)), g2)
        else:
            block_m, block_r, block_k = _native_graph_fused_wavg_lora_blocks()
            w, a, g, v_gate = fused_wavg_lora(
                xw,
                xa,
                xg,
                xv,
                w1,
                a1,
                g1,
                v1,
                w2,
                a2,
                g2,
                v2,
                w0,
                a0,
                None,
                v0,
                block_m=block_m,
                block_r=block_r,
                block_k=block_k,
            )
        a = torch.sigmoid(a)
    elif _native_graph_fused_wag_lora_enabled():
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, B, H * N)
        block_m, block_r, block_k = _native_graph_fused_wag_lora_blocks()
        w, a, g = fused_wag_lora(
            xw,
            xa,
            xg,
            w1,
            a1,
            g1,
            w2,
            a2,
            g2,
            w0,
            a0,
            None,
            block_m=block_m,
            block_r=block_r,
            block_k=block_k,
        )
        a = torch.sigmoid(a)
    else:
        r, k, v = _native_graph_rkv_project(xr, xk, xv, Rw, Kw, Vw, RKVw, B, H * N)
        w = F.linear(torch.tanh(F.linear(xw, w1)), w2, w0)
        a = torch.sigmoid(a0 + F.linear(F.linear(xa, a1), a2))
        g = F.linear(torch.sigmoid(F.linear(xg, g1)), g2)
    kk = F.normalize((k * k_k).view(B, H, N), dim=-1, p=2.0).view(B, H * N)
    k = k * (1 + (a - 1) * k_a)
    if i == 0:
        v_first.copy_(v)
    else:
        if v_gate is None:
            v_gate = torch.sigmoid(v0 + F.linear(F.linear(xv, v1), v2))
        v = v + (v_first - v) * v_gate
    w = torch.exp(-0.606531 * torch.sigmoid(w.float()))
    if _native_graph_fused_recurrent_output_enabled():
        out, new_state = fused_recurrent_output_prepare(
            r.view(B, H, N),
            w.view(B, H, N),
            k.view(B, H, N),
            v.view(B, H, N),
            kk.view(B, H, N),
            a.view(B, H, N),
            state,
            g.view(B, H, N),
            r_k,
            gn_w,
            gn_b,
            eps=eps,
            block_n=N,
        )
        out = out.reshape(B, H * N)
    else:
        out, new_state = _recurrent_update_batched(r, w, k, v, kk, a, state, B, H, N)
    if _native_graph_fused_recurrent_output_enabled():
        out = F.linear(out, Ow)
    elif _native_graph_fused_output_project_enabled():
        out = fused_attn_output_project(
            out,
            r.view(B, H, N),
            k.view(B, H, N),
            v.view(B, H, N),
            g,
            r_k,
            gn_w,
            gn_b,
            Ow,
            None,
            num_heads=H,
            head_dim=N,
            head_v_dim=N,
            eps=eps,
            block_m=_native_graph_fused_output_project_block_m(),
        )
    elif _native_graph_fused_output_enabled():
        out = fused_attn_output_prepare(
            out,
            r.view(B, H, N),
            k.view(B, H, N),
            v.view(B, H, N),
            g,
            r_k,
            gn_w,
            gn_b,
            num_heads=H,
            head_dim=N,
            head_v_dim=N,
            eps=eps,
        )
        out = F.linear(out, Ow)
    else:
        out = F.group_norm(out, H, gn_w, gn_b, eps).view(B, H * N)
        sk = (r.view(B, H, N) * k.view(B, H, N) * r_k).sum(dim=-1, keepdim=True)
        out = (out + (sk * v.view(B, H, N)).view(B, H * N)) * g
        out = F.linear(out, Ow)
    xpa.copy_(h)
    state.copy_(new_state)
    x = residual + out

    residual = x
    h2 = F.layer_norm(x, [H * N], fn_w, fn_b, 1e-5)
    fxx = xpf - h2
    fk = h2 + fxx * fx_k
    fk = torch.relu(F.linear(fk, fK)) ** 2
    xpf.copy_(h2)
    return residual + F.linear(fk, fV)


def cuda_graph_decode(model, ids, packs, n=128):
    import time
    base = model.model
    device = ids.device
    dtype = base.embeddings.weight.dtype
    nL = len(packs)
    H, N = packs[0][1], packs[0][2]
    hid = base.layers[0].attn.hidden_size
    state = [torch.zeros(H, N, N, device=device, dtype=torch.float32) for _ in range(nL)]
    xpa = [torch.zeros(hid, device=device, dtype=dtype) for _ in range(nL)]
    xpf = [torch.zeros(hid, device=device, dtype=dtype) for _ in range(nL)]
    v_first = torch.zeros(hid, device=device, dtype=dtype)
    tok_id = torch.zeros(1, dtype=torch.long, device=device)
    logits = torch.zeros(base.embeddings.weight.shape[0], device=device, dtype=dtype)
    emb = base.embeddings.weight
    head = model.lm_head.weight
    nw, nb = base.norm.weight, base.norm.bias

    x = None
    for t in range(ids.shape[1]):
        x = F.embedding(ids[0, t:t + 1], emb).reshape(-1)
        for li, p in enumerate(packs):
            x = _block_ip(x, state[li], xpa[li], xpf[li], v_first, p)
    tok_id.copy_(F.linear(F.layer_norm(x, [H * N], nw, nb, 1e-5), head).argmax())

    def one_step():
        x = F.embedding(tok_id, emb).reshape(-1)
        for li, p in enumerate(packs):
            x = _block_ip(x, state[li], xpa[li], xpf[li], v_first, p)
        logits.copy_(F.linear(F.layer_norm(x, [H * N], nw, nb, 1e-5), head))

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            one_step()
            tok_id.copy_(logits.argmax())
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        one_step()

    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(n):
        g.replay()
        tok_id.copy_(logits.argmax())
    torch.cuda.synchronize(); dt = time.time() - t0
    return n / dt


def greedy_jit(model, ids, packs, n=40):
    base = model.model
    H, N = packs[0][1], packs[0][2]
    nw, nb = base.norm.weight, base.norm.bias
    state, xpa, xpf, v_first = _init(model, ids.device, base.embeddings.weight.dtype)
    x = None
    for t in range(ids.shape[1]):
        x = F.embedding(ids[0, t:t + 1], base.embeddings.weight).reshape(-1)
        x, state, xpa, xpf, v_first = step(model, x, state, xpa, xpf, v_first, packs)
    nx = F.linear(F.layer_norm(x, [H * N], nw, nb, 1e-5), model.lm_head.weight).argmax().clone()
    toks = [int(nx)]
    with torch.no_grad():
        for _ in range(n - 1):
            x = F.embedding(nx.reshape(1, 1), base.embeddings.weight).reshape(-1)
            x, state, xpa, xpf, v_first = step(model, x, state, xpa, xpf, v_first, packs)
            nx = F.linear(F.layer_norm(x, [H * N], nw, nb, 1e-5), model.lm_head.weight).argmax()
            toks.append(int(nx))
    return toks


def greedy_graph(model, ids, packs, n=40):
    base = model.model
    device = ids.device
    dtype = base.embeddings.weight.dtype
    nL = len(packs)
    H, N = packs[0][1], packs[0][2]
    hid = base.layers[0].attn.hidden_size
    state = [torch.zeros(H, N, N, device=device, dtype=torch.float32) for _ in range(nL)]
    xpa = [torch.zeros(hid, device=device, dtype=dtype) for _ in range(nL)]
    xpf = [torch.zeros(hid, device=device, dtype=dtype) for _ in range(nL)]
    v_first = torch.zeros(hid, device=device, dtype=dtype)
    tok_id = torch.zeros(1, dtype=torch.long, device=device)
    logits = torch.zeros(base.embeddings.weight.shape[0], device=device, dtype=dtype)
    emb, head = base.embeddings.weight, model.lm_head.weight
    nw, nb = base.norm.weight, base.norm.bias
    x = None
    for t in range(ids.shape[1]):
        x = F.embedding(ids[0, t:t + 1], emb).reshape(-1)
        for li, p in enumerate(packs):
            x = _block_ip(x, state[li], xpa[li], xpf[li], v_first, p)
    tok_id.copy_(F.linear(F.layer_norm(x, [H * N], nw, nb, 1e-5), head).argmax())
    # snapshot post-prefill state so we can realign after warmup advances it
    st_s = [s.clone() for s in state]
    xpa_s = [s.clone() for s in xpa]
    xpf_s = [s.clone() for s in xpf]
    vf_s = v_first.clone()
    tok_s = tok_id.clone()

    def one_step():
        x = F.embedding(tok_id, emb).reshape(-1)
        for li, p in enumerate(packs):
            x = _block_ip(x, state[li], xpa[li], xpf[li], v_first, p)
        logits.copy_(F.linear(F.layer_norm(x, [H * N], nw, nb, 1e-5), head))

    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            one_step(); tok_id.copy_(logits.argmax())
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        one_step()
    # restore post-prefill state so the captured graph replays from the right point
    for i in range(len(state)):
        state[i].copy_(st_s[i]); xpa[i].copy_(xpa_s[i]); xpf[i].copy_(xpf_s[i])
    v_first.copy_(vf_s)
    tok_id.copy_(tok_s)
    toks = [int(tok_id)]
    for _ in range(n - 1):
        g.replay()
        nt = logits.argmax()
        tok_id.copy_(nt)
        toks.append(int(nt))
    return toks


def fast_generate(model, tokenizer, prompt, max_new_tokens=48, use_graph=True):
    """End-to-end greedy generation via the native (CUDA-graph) decode path.
    Returns the full decoded text (prompt + new tokens). Same result as the
    FLA model's greedy generate(), but ~10x faster on the 5070."""
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    packs, _, _, _ = extract(model)
    fn = greedy_graph if use_graph else greedy_jit
    new_tokens = fn(model, ids, packs, n=max_new_tokens)
    full = ids[0].tolist() + new_tokens
    return tokenizer.decode(full, skip_special_tokens=True)


if __name__ == "__main__":
    import os, sys
    os.environ.setdefault("RWKV_V7_ON", "1")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    d = sys.argv[1] if len(sys.argv) > 1 else "D:/rwkv7-models/rwkv7-g1d-0.1b-hf"
    tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
    # correctness at fp32 vs fla
    model = AutoModelForCausalLM.from_pretrained(d, trust_remote_code=True, torch_dtype=torch.float32, device_map="cuda").eval()
    packs, H, N, eps = extract(model)
    for prompt in ["The quick brown fox jumps over the lazy dog.",
                   "Once upon a time, in a faraway land,"]:
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        with torch.no_grad():
            fla = model(ids).logits[0, -1].float().cpu()
            nat = forward(model, ids, packs).float().cpu()
        cos = F.cosine_similarity(fla.unsqueeze(0), nat.unsqueeze(0)).item()
        maxabs = (fla - nat).abs().max().item()
        print(f"[correctness] cos={cos:.6f} maxabs={maxabs:.4f} "
              f"argmax={int(fla.argmax() == nat.argmax())}  {prompt[:36]!r}")
    del model; torch.cuda.empty_cache()
    # speed
    for dt_name, dt in [("fp16", torch.float16), ("fp32", torch.float32)]:
        model = AutoModelForCausalLM.from_pretrained(d, trust_remote_code=True, torch_dtype=dt, device_map="cuda").eval()
        packs, H, N, eps = extract(model)
        ids = tok("The quick brown fox.", return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        with torch.no_grad():
            tps_jit = decode_speed(model, ids, packs)
            tps_cg = cuda_graph_decode(model, ids, packs)
            tj = greedy_jit(model, ids, packs)
            tg = greedy_graph(model, ids, packs)
        match = sum(int(a == b) for a, b in zip(tj, tg))
        print(f"[decode {dt_name}] jit-fused {tps_jit:.1f} | cuda-graph {tps_cg:.1f} tok/s | "
              f"graph-correct {match}/{len(tj)} tokens == jit")
        del model; torch.cuda.empty_cache()

    # end-to-end: native greedy token ids vs fla model.generate (must match)
    model = AutoModelForCausalLM.from_pretrained(d, trust_remote_code=True, torch_dtype=torch.float16, device_map="cuda").eval()
    packs, _, _, _ = extract(model)
    prompt = "User: Hello!\n\nAssistant:"
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
    with torch.no_grad():
        fla_out = model.generate(ids, max_new_tokens=32, do_sample=False, use_cache=True, pad_token_id=0)
    fla_ids = fla_out[0, ids.shape[1]:].tolist()
    nat_ids = greedy_graph(model, ids, packs, n=32)
    print(f"[e2e] fla   : {tok.decode(fla_ids)!r}")
    print(f"[e2e] native: {tok.decode(nat_ids)!r}")
    print(f"[e2e] token-identical: {fla_ids == nat_ids} ({sum(int(a==b) for a,b in zip(fla_ids,nat_ids))}/{len(fla_ids)})")
