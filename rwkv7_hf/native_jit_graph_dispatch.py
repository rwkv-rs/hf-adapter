# coding=utf-8
"""Native CUDA-graph policy gates and projection/FFN dispatch.

The stable native_jit facade binds its optional kernels, policy helpers and
linear adapters once at import time. All exported hot-path functions are then
direct aliases, avoiding another Python frame per projection.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F


_OWNED_NAMES = {'_native_graph_fused_recurrent_raw_enabled', '_native_graph_linear_dispatch', '_native_graph_fused_output_project_block_m', '_native_graph_ada_sparse_ffn_enabled', '_native_graph_ada_linear_should_route', '_native_graph_fused_wavg_lora_blocks', '_native_graph_blackwell_norm_mix_enabled', '_native_graph_rkv_project', '_native_graph_ada_wagv_lora_enabled', '_native_graph_fused_recurrent_output_enabled', '_native_graph_fp16_recurrent_enabled', '_native_graph_fused_wag_lora_blocks', '_native_graph_vkwr_rkv_dispatch', '_native_graph_ffn_dispatch', '_native_graph_fused_projection_enabled', '_native_graph_ada_wag_lora_enabled', '_native_graph_int_env', '_native_graph_fused_output_project_enabled', '_native_graph_ffn_down_add_dispatch', 'prewarm_ada_sparse_ffn', '_native_graph_fused_wavg_lora_enabled', '_native_graph_ffn_up_relu2_dispatch', '_native_graph_sm70_linear_enabled', '_native_graph_fused_wavg_lora_num_warps', '_native_graph_fused_output_enabled', '_native_graph_fused_norm_mix_num_warps', '_native_graph_fused_norm_mix_enabled', '_native_graph_fused_wag_lora_enabled', '_native_graph_rkv_policy', '_native_graph_ada_linear_enabled', '_native_graph_sm70_wagv_lora_enabled'} | {"bind_runtime", "_facade_value"}


def bind_runtime(runtime: dict[str, object]) -> None:
    """Bind facade dependencies without replacing functions owned here."""

    for name, value in runtime.items():
        if name not in _OWNED_NAMES and not name.startswith("__"):
            globals()[name] = value


def _facade_value(name: str, fallback):
    facade = sys.modules.get(f"{__package__}.native_jit")
    return getattr(facade, name, fallback) if facade is not None else fallback


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

def _native_graph_fused_recurrent_raw_enabled() -> bool:
    """Fold W decay and K/KK preparation into recurrent output fusion."""

    policy = _kernel_policy()
    if not env_flag("RWKV7_NATIVE_GRAPH_FUSED_RECURRENT_RAW", bool(getattr(policy, "fused_recurrent_raw", False))):
        return False
    return bool(fused_recurrent_output_prepare_raw is not None and _native_graph_fused_recurrent_output_enabled())

def _native_graph_fp16_recurrent_enabled(state: torch.Tensor, elapsed) -> bool:
    policy = _kernel_policy()
    return bool(
        env_flag(
            "RWKV7_NATIVE_GRAPH_FP16_RECURRENT",
            bool(getattr(policy, "native_graph_fp16_recurrent", False)),
        )
        and native_fp16_recurrent_output_prepare_raw is not None
        and elapsed is not None
        and state.dtype == torch.float16
        and int(state.shape[-1]) == 64
    )

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

def _native_graph_sm70_wagv_lora_enabled(rows: int, hidden_size: int) -> bool:
    policy = _kernel_policy()
    if not env_flag(
        "RWKV7_NATIVE_GRAPH_SM70_WAGV_LORA",
        bool(getattr(policy, "sm70_wagv_lora", False)),
    ):
        return False
    if int(rows) < 1 or int(rows) > 4 or int(hidden_size) < 1024:
        return False
    if sm70_wagv_lora is None or sm70_wagv_lora_available is None:
        return False
    try:
        return bool(sm70_wagv_lora_available())
    except Exception:
        return False

def _native_graph_fused_wavg_lora_enabled(rows: int, hidden_size: int) -> bool:
    """Runtime switch for the native-graph W/A/G/V-gate LoRA fusion probe."""

    policy = _kernel_policy()
    if not env_flag("RWKV7_NATIVE_GRAPH_FUSED_WAVG_LORA", bool(getattr(policy, "fused_wavg_lora", False))):
        return False
    default_max = getattr(policy, "wavg_lora_bsz1_max_hidden", None)
    bsz1_max_hidden = env_int(
        "RWKV7_NATIVE_GRAPH_FUSED_WAVG_LORA_BSZ1_MAX_HIDDEN",
        0 if default_max is None else int(default_max),
        lower=0,
    )
    if int(rows) == 1 and bsz1_max_hidden > 0 and int(hidden_size) > bsz1_max_hidden:
        return False
    if fused_wavg_lora is None or fused_wavg_lora_available is None:
        return False
    try:
        return bool(fused_wavg_lora_available())
    except Exception:
        return False

def _native_graph_fused_norm_mix_enabled() -> bool:
    """Runtime switch for decode layer-norm/residual/time-mix fusion."""

    policy = _kernel_policy()
    if not env_flag(
        "RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX",
        bool(getattr(policy, "fused_norm_mix", False)),
    ):
        return False
    if (
        fused_attn_norm_mix6_decode is None
        or fused_ffn_add_norm_mix_decode is None
        or fused_decode_norm_mix_available is None
    ):
        return False
    try:
        return bool(fused_decode_norm_mix_available())
    except Exception:
        return False

def _native_graph_fused_norm_mix_num_warps() -> int:
    policy = _kernel_policy()
    default = int(getattr(policy, "norm_mix_num_warps", 4))
    value = env_int("RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX_NUM_WARPS", default, lower=1, upper=8)
    if value not in {1, 2, 4, 8}:
        raise ValueError(f"RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX_NUM_WARPS must be one of 1, 2, 4, or 8; got {value}")
    return value

def _native_graph_blackwell_norm_mix_enabled(
    residual, attention, previous, *, layer_index: int
) -> bool:
    if not env_flag("RWKV7_NATIVE_GRAPH_BLACKWELL_NORM_MIX", False):
        return False
    batch_size = int(residual.shape[0]) if residual.ndim > 1 else 1
    selected = os.environ.get(
        f"RWKV7_NATIVE_GRAPH_BLACKWELL_NORM_MIX_LAYERS_B{batch_size}",
        os.environ.get("RWKV7_NATIVE_GRAPH_BLACKWELL_NORM_MIX_LAYERS", ""),
    ).strip()
    if selected:
        try:
            layers = {int(value.strip()) for value in selected.split(",") if value.strip()}
        except ValueError as exc:
            raise ValueError(
                "RWKV7_NATIVE_GRAPH_BLACKWELL_NORM_MIX_LAYERS must be comma-separated integers"
            ) from exc
        if int(layer_index) not in layers:
            return False
    add_norm_mix = _facade_value(
        "blackwell_ffn_add_norm_mix", blackwell_ffn_add_norm_mix
    )
    should_use = _facade_value(
        "blackwell_norm_mix_should_use", blackwell_norm_mix_should_use
    )
    if add_norm_mix is None or should_use is None:
        return False
    try:
        return bool(should_use(residual, attention, previous))
    except Exception:
        return False

def _native_graph_sm70_linear_enabled() -> bool:
    """Whether measured sm_70 small-row linear routes may be captured."""

    policy = _kernel_policy()
    return bool(
        env_flag("RWKV7_NATIVE_GRAPH_SM70_LINEAR", bool(getattr(policy, "sm70_linear", False)))
        and sm70_linear is not None
        and sm70_linear_should_use is not None
        and sm70_linear_threads is not None
    )

def _native_graph_ada_sparse_ffn_enabled() -> bool:
    """Whether the measured sm_89 sparse FFN route may be captured."""

    policy = _kernel_policy()
    return bool(
        env_flag(
            "RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN",
            bool(getattr(policy, "ada_sparse_ffn", False)),
        )
        and ada_sparse_ffn_down_add is not None
        and ada_ffn_up is not None
        and ada_sparse_ffn_should_use is not None
    )

def _native_graph_ada_linear_enabled() -> bool:
    """Whether measured no-copy sm_89 exact-row linears may be captured."""

    policy = _kernel_policy()
    return bool(
        env_flag(
            "RWKV7_NATIVE_GRAPH_ADA_LINEAR",
            bool(getattr(policy, "ada_linear", False)),
        )
        and ada_linear is not None
        and ada_linear_should_use is not None
    )

def _native_graph_ada_linear_should_route(rows: int, role: str) -> bool:
    """Shape/role gate; row 1 remains a probe while measured row 2 is default."""

    if not _native_graph_ada_linear_enabled():
        return False
    policy = _kernel_policy()
    raw_rows = os.environ.get(
        "RWKV7_NATIVE_GRAPH_ADA_LINEAR_ROWS",
        str(getattr(policy, "ada_linear_rows", "2 4")),
    )
    try:
        enabled_rows = {int(item) for item in raw_rows.replace(",", " ").split()}
    except ValueError:
        enabled_rows = {2}
    raw_roles = os.environ.get("RWKV7_NATIVE_GRAPH_ADA_LINEAR_ROLES")
    if raw_roles is None:
        raw_roles = str(getattr(policy, "ada_linear_roles", "auto"))
        if raw_roles.strip().lower() == "auto":
            raw_roles = "hidden" if int(rows) == 4 else "hidden,ffn_up,ffn_down"
    enabled_roles = {item.strip().lower() for item in raw_roles.replace(",", " ").split() if item.strip()}
    return int(rows) in enabled_rows and role.lower() in enabled_roles

def _native_graph_ada_wagv_lora_enabled(rows: int, hidden_size: int, max_rank: int) -> bool:
    """Whether the no-copy sm_89 grouped low-rank route may be captured."""

    policy = _kernel_policy()
    return bool(
        env_flag(
            "RWKV7_NATIVE_GRAPH_ADA_WAGV_LORA",
            bool(getattr(policy, "ada_wagv_lora", False)),
        )
        and ada_wagv_lora is not None
        and ada_wagv_lora_should_use is not None
        and ada_wagv_lora_should_use(int(rows), int(hidden_size), int(max_rank))
    )

def _native_graph_ada_wag_lora_enabled() -> bool:
    """Whether the exact-card W/A/G-only low-rank route may be captured."""

    policy = _kernel_policy()
    if not env_flag(
        "RWKV7_NATIVE_GRAPH_ADA_WAG_LORA",
        bool(getattr(policy, "ada_wag_lora", False)),
    ):
        return False
    if ada_wag_lora is None or ada_wagv_lora_available is None:
        return False
    try:
        return bool(ada_wagv_lora_available())
    except Exception:
        return False

def _native_graph_linear_dispatch(x: torch.Tensor, weight, *, role: str) -> torch.Tensor:
    """Dispatch dense or native-quantized linears during graph capture."""

    rows = 1 if x.dim() == 1 else int(x.shape[0])
    outputs, inputs = _graph_linear_shape(weight)
    if (
        _graph_linear_is_dense(weight)
        and role != "head"
        and _native_graph_ada_linear_should_route(rows, role)
        and ada_linear_should_use(rows, outputs, inputs)
    ):
        return ada_linear(x, weight)
    if not _graph_linear_is_dense(weight):
        return _graph_linear_call(x, weight)
    if (
        sm70_orig_linear is not None
        and role == "hidden"
        and rows in {2, 4}
        and outputs == inputs
        and inputs >= 2048
    ):
        return sm70_orig_linear(x, weight)
    if not _native_graph_sm70_linear_enabled():
        return F.linear(x, weight)
    if not sm70_linear_should_use(rows, outputs, inputs, role=role):
        return F.linear(x, weight)
    threads = sm70_linear_threads(rows, outputs, inputs, role=role)
    return sm70_linear(x, weight, threads=threads)

def _native_graph_ffn_up_relu2_dispatch(x: torch.Tensor, weight) -> torch.Tensor:
    rows = 1 if x.dim() == 1 else int(x.shape[0])
    outputs, inputs = _graph_linear_shape(weight)
    if (
        _graph_linear_is_dense(weight)
        and _native_graph_ada_linear_should_route(rows, "ffn_up")
        and ada_linear_should_use(rows, outputs, inputs)
    ):
        return torch.relu(ada_linear(x, weight)) ** 2
    fused_quant = getattr(weight, "rwkv7_forward_relu2", None)
    if not _graph_linear_is_dense(weight) and callable(fused_quant):
        return fused_quant(x)
    if not _graph_linear_is_dense(weight):
        fused = getattr(weight, "rwkv7_forward_relu2", None)
        if bool(getattr(weight, "fused_relu2", False)) and callable(fused):
            return fused(x)
        return torch.relu(_graph_linear_call(x, weight)) ** 2
    if (
        not _native_graph_sm70_linear_enabled()
        or sm70_ffn_up_relu2 is None
        or sm70_ffn_up_relu2_should_use is None
    ):
        return torch.relu(F.linear(x, weight)) ** 2
    if not sm70_ffn_up_relu2_should_use(rows, outputs, inputs):
        return torch.relu(F.linear(x, weight)) ** 2
    threads = sm70_linear_threads(rows, outputs, inputs, role="ffn_up")
    return sm70_ffn_up_relu2(x, weight, threads=threads)

def _native_graph_ffn_down_add_dispatch(
    x: torch.Tensor,
    weight,
    residual: torch.Tensor,
) -> torch.Tensor:
    rows = 1 if x.dim() == 1 else int(x.shape[0])
    outputs, inputs = _graph_linear_shape(weight)
    if (
        _graph_linear_is_dense(weight)
        and _native_graph_ada_linear_should_route(rows, "ffn_down")
        and ada_linear_should_use(rows, outputs, inputs)
    ):
        return residual + ada_linear(x, weight)
    fused_quant = getattr(weight, "rwkv7_forward_add", None)
    if not _graph_linear_is_dense(weight) and callable(fused_quant):
        return fused_quant(x, residual)
    if not _graph_linear_is_dense(weight):
        return residual + _graph_linear_call(x, weight)
    if (
        not _native_graph_sm70_linear_enabled()
        or sm70_ffn_down_add is None
        or sm70_ffn_down_add_should_use is None
    ):
        return residual + F.linear(x, weight)
    if not sm70_ffn_down_add_should_use(rows, outputs, inputs):
        return residual + F.linear(x, weight)
    threads = sm70_linear_threads(rows, outputs, inputs, role="ffn_down")
    return sm70_ffn_down_add(x, weight, residual, threads=threads)

def _native_graph_ffn_dispatch(
    x: torch.Tensor,
    up_weight,
    down_weight,
    residual: torch.Tensor,
    *,
    sparse_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Route the complete FFN boundary so sparse kernels can avoid ReLU² IO."""

    rows = 1 if x.dim() == 1 else int(x.shape[0])
    outputs, inputs = _graph_linear_shape(down_weight)
    if (
        _graph_linear_is_dense(up_weight)
        and _graph_linear_is_dense(down_weight)
        and _native_graph_ada_sparse_ffn_enabled()
        and rows <= env_int(
            "RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_MAX_ROWS",
            int(getattr(_kernel_policy(), "ada_sparse_ffn_max_rows", 19)),
            lower=1,
            upper=19,
        )
        and ada_sparse_ffn_should_use(rows, outputs, inputs)
    ):
        if env_flag(
            "RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_UP",
            bool(getattr(_kernel_policy(), "ada_sparse_ffn_up", True)),
        ):
            preact = ada_ffn_up(x, up_weight)
        else:
            preact = F.linear(x, up_weight)
        target = residual if env_flag(
            "RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_INPLACE",
            bool(getattr(_kernel_policy(), "ada_sparse_ffn_inplace", False)),
        ) else sparse_out
        return ada_sparse_ffn_down_add(preact, down_weight, residual, out=target)
    fused_quant_ffn = getattr(up_weight, "rwkv7_forward_ffn", None)
    if callable(fused_quant_ffn):
        fused = fused_quant_ffn(x, down_weight, residual)
        if fused is not None:
            return fused
    hidden = _native_graph_ffn_up_relu2_dispatch(x, up_weight)
    return _native_graph_ffn_down_add_dispatch(hidden, down_weight, residual)

def prewarm_ada_sparse_ffn(packs, rows: int = 1) -> int:
    """Pack sparse FFN down weights before CUDA graph capture.

    Creating the transposed weights during capture places them in a graph
    private pool. Prepacking on the normal stream gives each enabled batch
    shape a stable read-only operand before its independent graph is captured.
    """

    max_rows = env_int(
        "RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_MAX_ROWS",
        int(getattr(_kernel_policy(), "ada_sparse_ffn_max_rows", 19)),
        lower=1,
        upper=19,
    )
    if (
        not _native_graph_ada_sparse_ffn_enabled()
        or ada_sparse_ffn_pack_weight is None
        or int(rows) > max_rows
    ):
        return 0
    packed = 0
    for operands in packs:
        down_weight = operands[-2]
        if not _graph_linear_is_dense(down_weight):
            continue
        outputs, inputs = _graph_linear_shape(down_weight)
        if not ada_sparse_ffn_should_use(1, outputs, inputs):
            continue
        ada_sparse_ffn_pack_weight(down_weight, cache_tag=int(rows))
        if (
            env_flag(
                "RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_FP32_ACCUM",
                bool(getattr(_kernel_policy(), "ada_sparse_ffn_fp32_accum", False)),
            )
            and ada_sparse_ffn_prepare_fp32_scratch is not None
        ):
            ada_sparse_ffn_prepare_fp32_scratch(down_weight, int(rows))
        elif (
            os.environ.get(
                "RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_DETERMINISTIC_SPLITS",
                str(getattr(_kernel_policy(), "ada_sparse_ffn_deterministic_splits", 0)),
            ).strip()
            == "4"
            and ada_sparse_ffn_deterministic4_should_use is not None
            and ada_sparse_ffn_deterministic4_should_use(
                int(rows), outputs, inputs
            )
            and ada_sparse_ffn_prepare_deterministic_scratch is not None
        ):
            ada_sparse_ffn_prepare_deterministic_scratch(down_weight, int(rows))
        packed += 1
    return packed

def _native_graph_rkv_policy() -> str:
    """Return the optional VKWR-inspired R/K/V projection dispatch policy.

    VKWR stacks the receptance/key/value matrices and uses a grouped batched
    projection for selected small-row decode cases.  Keep the HF adapter's
    historical three-``F.linear`` path by default and enable the stacked path
    only through ``RWKV7_NATIVE_GRAPH_RKV_POLICY=vkwr_auto`` while collecting
    telemetry.
    """

    policy = _kernel_policy()
    default = str(getattr(policy, "rkv_policy", "manual"))
    raw = os.environ.get("RWKV7_NATIVE_GRAPH_RKV_POLICY", default).strip().lower()
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
    max_rows = _native_graph_int_env("RWKV7_NATIVE_GRAPH_RKV_MAX_ROWS", 64, lo=1, hi=4096)
    if hidden_size < min_hidden:
        return False
    return rows == 1 or (4 <= rows <= max_rows)

def _native_graph_rkv_project(
    xr: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    Rw,
    Kw,
    Vw,
    RKVw: torch.Tensor,
    rows: int,
    hidden_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project R/K/V with either separate linears or VKWR-style stacked bmm."""

    dense_rkv = all(_graph_linear_is_dense(item) for item in (Rw, Kw, Vw))
    output_size = int(_graph_linear_shape(Rw)[0])
    if dense_rkv and _native_graph_vkwr_rkv_dispatch(int(rows), int(hidden_size)) and RKVw.numel() != 0:
        shared_storage = False
        try:
            row_values = int(rows) * int(hidden_size)
            shared_storage = bool(
                xr.is_contiguous()
                and xk.is_contiguous()
                and xv.is_contiguous()
                and xr.untyped_storage().data_ptr() == xk.untyped_storage().data_ptr()
                and xr.untyped_storage().data_ptr() == xv.untyped_storage().data_ptr()
                and int(xk.storage_offset()) == int(xr.storage_offset()) + row_values
                and int(xv.storage_offset()) == int(xr.storage_offset()) + 2 * row_values
            )
        except Exception:
            shared_storage = False
        if shared_storage:
            flat = xr.as_strided(
                (3, int(rows), int(hidden_size)),
                (int(rows) * int(hidden_size), int(hidden_size), 1),
            )
        elif xr.dim() == 1:
            flat = torch.stack(
                (
                    xr.reshape(1, hidden_size),
                    xk.reshape(1, hidden_size),
                    xv.reshape(1, hidden_size),
                ),
                dim=0,
            )
        else:
            flat = torch.stack(
                (
                    xr.reshape(rows, hidden_size),
                    xk.reshape(rows, hidden_size),
                    xv.reshape(rows, hidden_size),
                ),
                dim=0,
            )
        rkv = torch.bmm(flat, RKVw)
        if xr.dim() == 1:
            return rkv[0, 0], rkv[1, 0], rkv[2, 0]
        return rkv[0], rkv[1], rkv[2]
    if (
        dense_rkv
        and output_size == int(hidden_size)
        and sm70_orig_rkv is not None
        and int(rows) in {2, 4}
        and int(hidden_size) >= 2048
    ):
        return sm70_orig_rkv(xr, xk, xv, Rw, Kw, Vw)
    if (
        dense_rkv
        and output_size == int(hidden_size)
        and _native_graph_sm70_linear_enabled()
        and sm70_rkv is not None
        and sm70_rkv_should_use is not None
        and sm70_rkv_threads is not None
        and sm70_rkv_should_use(int(rows), int(hidden_size))
    ):
        threads = sm70_rkv_threads(int(rows), int(hidden_size))
        return sm70_rkv(xr, xk, xv, Rw, Kw, Vw, threads=threads)
    return (
        _native_graph_linear_dispatch(xr, Rw, role="hidden"),
        _native_graph_linear_dispatch(xk, Kw, role="hidden"),
        _native_graph_linear_dispatch(xv, Vw, role="hidden"),
    )

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

def _native_graph_fused_wavg_lora_num_warps() -> int:
    policy = _kernel_policy()
    default = int(getattr(policy, "wavg_lora_num_warps", 4))
    value = env_int("RWKV7_NATIVE_GRAPH_FUSED_WAVG_LORA_NUM_WARPS", default, lower=1, upper=8)
    if value not in {1, 2, 4, 8}:
        raise ValueError(
            "RWKV7_NATIVE_GRAPH_FUSED_WAVG_LORA_NUM_WARPS must be one of 1, 2, 4, or 8; "
            f"got {value}"
        )
    return value

__all__ = ['_native_graph_fused_recurrent_output_enabled', '_native_graph_fused_recurrent_raw_enabled', '_native_graph_fp16_recurrent_enabled', '_native_graph_fused_output_enabled', '_native_graph_fused_output_project_enabled', '_native_graph_fused_output_project_block_m', '_native_graph_fused_projection_enabled', '_native_graph_fused_wag_lora_enabled', '_native_graph_sm70_wagv_lora_enabled', '_native_graph_fused_wavg_lora_enabled', '_native_graph_fused_norm_mix_enabled', '_native_graph_fused_norm_mix_num_warps', '_native_graph_blackwell_norm_mix_enabled', '_native_graph_sm70_linear_enabled', '_native_graph_ada_sparse_ffn_enabled', '_native_graph_ada_linear_enabled', '_native_graph_ada_linear_should_route', '_native_graph_ada_wagv_lora_enabled', '_native_graph_ada_wag_lora_enabled', '_native_graph_linear_dispatch', '_native_graph_ffn_up_relu2_dispatch', '_native_graph_ffn_down_add_dispatch', '_native_graph_ffn_dispatch', 'prewarm_ada_sparse_ffn', '_native_graph_rkv_policy', '_native_graph_int_env', '_native_graph_vkwr_rkv_dispatch', '_native_graph_rkv_project', '_native_graph_fused_wag_lora_blocks', '_native_graph_fused_wavg_lora_blocks', '_native_graph_fused_wavg_lora_num_warps']
