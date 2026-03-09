# coding=utf-8
# Ported from rwkv-rs/rwkv7-ascend-npu@b6391271f into the canonical HF adapter.
"""Shape-gated W8A16 inference for Huawei Ascend 910B3.

Only the exact large FFN projections with paired device evidence are selected
by the ``speed`` policy.  torch_npu is imported lazily.  The packed module owns
only int8 weights and fp16 per-output scales, so replacing a dense fp16 Linear
reduces its resident payload to about one half.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ascend_runtime import (
    allow_unvalidated_ascend,
    detect_cann_version,
    import_torch_npu,
    validate_ascend_stack,
)


_NPU_W8_OP = None


def _npu_w8_op():
    global _NPU_W8_OP
    if _NPU_W8_OP is None:
        _NPU_W8_OP = import_torch_npu(required=True).npu_weight_quant_batchmatmul
    return _NPU_W8_OP


_PROMOTED_910B3_W8_SHAPES = {
    # (in_features, out_features): validated logical role
    (4096, 16384): "ffn.key",
    (16384, 4096): "ffn.value",
}
ASCEND_910B3_W8_SPEED_ROWS = (1, 4, 8)
_ASCEND_910B3_W8_SPEED_CONFIG = {
    "hidden_size": 4096,
    "intermediate_size": 16384,
    "num_hidden_layers": 32,
}


@dataclass(frozen=True)
class AscendQuantDecision:
    enabled: bool
    reason: str
    policy: str
    scheme: str = "w8a16_per_channel"
    speed_validated: bool = False
    stack_validated: bool = False


def ascend_w8a16_decision(
    module_name: str,
    in_features: int,
    out_features: int,
    *,
    device_name: str = "",
    torch_version: str | None = None,
    torch_npu_version: str | None = None,
    cann_version: str | None = None,
    rows: int | None = None,
    dtype: torch.dtype | str | None = None,
    policy: str = "speed",
) -> AscendQuantDecision:
    """Return a fail-closed W8 decision for one logical Linear.

    Speed promotion is exact to Ascend 910B3, both 7.2B FFN projections, FP16
    activation, and measured logical row counts B1/B4/B8. ``rows=None`` is the
    load-time preflight: it permits packing only because the replacement module
    enforces the measured rows again at runtime. Other cards, stacks, dtypes,
    shapes and row counts remain fail-closed.
    """

    policy = str(policy).strip().lower()
    if policy not in {"speed", "memory", "candidate"}:
        return AscendQuantDecision(False, f"unsupported policy {policy!r}", policy)
    dtype_text = None
    if dtype is not None:
        dtype_text = str(dtype).lower().replace("torch.", "")
        if dtype_text not in {"float16", "fp16", "half", "bfloat16", "bf16"}:
            return AscendQuantDecision(
                False,
                f"W8A16 requires fp16 or bf16 activations/weights, got {dtype}",
                policy,
            )
    shape = (int(in_features), int(out_features))
    expected_role = _PROMOTED_910B3_W8_SHAPES.get(shape)
    role = str(module_name).lower()
    if expected_role is None:
        return AscendQuantDecision(
            False, f"shape {shape} has no promoted 910B3 W8 row", policy
        )
    if not role.endswith(expected_role):
        return AscendQuantDecision(
            False, f"shape {shape} is not named as {expected_role}", policy
        )
    stack_validated, stack_reason = validate_ascend_stack(
        device_name=device_name,
        torch_version=torch_version,
        torch_npu_version=torch_npu_version,
        cann_version=cann_version,
    )
    override = allow_unvalidated_ascend()
    if policy == "speed":
        if not stack_validated:
            return AscendQuantDecision(
                False,
                f"unvalidated Ascend production stack: {stack_reason}",
                policy,
            )
        if dtype_text not in {"float16", "fp16", "half"}:
            return AscendQuantDecision(
                False,
                "production W8 speed route requires FP16 activations",
                policy,
                stack_validated=True,
            )
        if rows is not None and int(rows) not in ASCEND_910B3_W8_SPEED_ROWS:
            return AscendQuantDecision(
                False,
                (
                    f"logical rows {int(rows)} are not in measured W8 rows "
                    f"{ASCEND_910B3_W8_SPEED_ROWS}"
                ),
                policy,
                stack_validated=True,
            )
        return AscendQuantDecision(
            True,
            (
                "real 7.2B HF NPUGraph W8 route passed quality, HBM and "
                "paired no-slower gates"
            ),
            policy,
            speed_validated=True,
            stack_validated=True,
        )
    if not stack_validated and not override:
        return AscendQuantDecision(
            False,
            f"unvalidated Ascend stack: {stack_reason}",
            policy,
        )
    if policy == "candidate":
        if expected_role != "ffn.value":
            return AscendQuantDecision(
                False, "candidate is ffn.value contraction only", policy
            )
        return AscendQuantDecision(
            True,
            (
                "explicit rejected-candidate reproduction on exact validated stack"
                if stack_validated
                else "explicit unvalidated-stack candidate override; never a production default"
            ),
            policy,
            speed_validated=False,
            stack_validated=stack_validated,
        )
    return AscendQuantDecision(
        True,
        (
            "large-FFN W8 memory route on exact validated stack"
            if stack_validated
            else "unvalidated-stack W8 memory override"
        ),
        policy,
        speed_validated=False,
        stack_validated=stack_validated,
    )


class AscendW8A16Linear(nn.Module):
    """Per-output-channel int8 Linear backed by npu_weight_quant_batchmatmul."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        dtype: torch.dtype = torch.float16,
        admitted_rows: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        if bias:
            raise ValueError(
                "AscendW8A16Linear currently supports bias-free RWKV projections only"
            )
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.bit = 8
        self.group_size = 0
        self.admitted_rows = tuple(sorted({int(row) for row in admitted_rows}))
        if any(row <= 0 for row in self.admitted_rows):
            raise ValueError("admitted W8 row counts must be positive")
        # Operator ABI is transposed [K, N], unlike nn.Linear [N, K].
        self.register_buffer(
            "q_weight",
            torch.empty(self.in_features, self.out_features, dtype=torch.int8),
        )
        self.register_buffer("scale", torch.empty(self.out_features, dtype=dtype))

    @classmethod
    @torch.no_grad()
    def from_float(
        cls,
        linear: nn.Linear,
        *,
        chunk_rows: int = 1024,
        admitted_rows: tuple[int, ...] = (),
    ) -> "AscendW8A16Linear":
        if linear.bias is not None:
            raise ValueError("RWKV Ascend W8 conversion requires a bias-free Linear")
        if linear.weight.dtype not in {torch.float16, torch.bfloat16}:
            raise ValueError(
                f"Ascend W8A16 requires fp16/bf16 source weight; got {linear.weight.dtype}"
            )
        module = cls(
            linear.in_features,
            linear.out_features,
            bias=False,
            dtype=linear.weight.dtype,
            admitted_rows=admitted_rows,
        ).to(linear.weight.device)
        chunk_rows = max(1, int(chunk_rows))
        weight = linear.weight.detach()
        for start in range(0, linear.out_features, chunk_rows):
            end = min(linear.out_features, start + chunk_rows)
            block = weight[start:end]
            scale = (
                (block.float().abs().amax(dim=1) / 127.0)
                .clamp_min(1e-8)
                .to(linear.weight.dtype)
            )
            quant = (
                torch.round(block.float() / scale[:, None].float())
                .clamp(-127, 127)
                .to(torch.int8)
            )
            module.scale[start:end].copy_(scale)
            module.q_weight[:, start:end].copy_(quant.transpose(0, 1))
        return module

    @property
    def packed_bytes(self) -> int:
        return (
            self.q_weight.numel() * self.q_weight.element_size()
            + self.scale.numel() * self.scale.element_size()
        )

    @property
    def dense_fp16_bytes(self) -> int:
        return self.in_features * self.out_features * 2

    def _validate_runtime_rows(self, logical_rows: int) -> None:
        logical_rows = int(logical_rows)
        if self.admitted_rows and logical_rows not in self.admitted_rows:
            raise RuntimeError(
                f"Ascend W8 logical rows {logical_rows} were not "
                f"production-validated; admitted rows={self.admitted_rows}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if int(x.shape[-1]) != self.in_features:
            raise ValueError(
                f"input last dim {x.shape[-1]} != in_features {self.in_features}"
            )
        original_shape = x.shape[:-1]
        if x.device.type == "npu":
            logical_rows = int(x.numel()) // self.in_features
            self._validate_runtime_rows(logical_rows)
            if x.dtype != self.scale.dtype:
                raise TypeError(
                    f"Ascend W8A16 activation dtype {x.dtype} must match scale dtype {self.scale.dtype}"
                )
            # Preserve the 2-D decode/prefill fast path without extra view calls.
            if x.dim() == 2:
                return _npu_w8_op()(x, self.q_weight, self.scale)
            flat = x.reshape(-1, self.in_features)
            out = _npu_w8_op()(flat, self.q_weight, self.scale)
            return out.reshape(*original_shape, self.out_features)
        flat = x.reshape(-1, self.in_features)
        # Portable correctness oracle; not a CPU performance route.
        dense = (
            self.q_weight.transpose(0, 1).to(flat.dtype)
            * self.scale.to(flat.dtype)[:, None]
        )
        out = F.linear(flat, dense)
        return out.reshape(*original_shape, self.out_features)


def _set_submodule(
    root: nn.Module, qualified_name: str, replacement: nn.Module
) -> None:
    parent_name, _, leaf = qualified_name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, leaf, replacement)


def _speed_model_selection_reason(
    model: nn.Module,
    selected_names: tuple[str, ...],
) -> str | None:
    """Reject partial or lookalike models before the production path mutates."""

    config = getattr(model, "config", None)
    for field, expected in _ASCEND_910B3_W8_SPEED_CONFIG.items():
        actual = getattr(config, field, None)
        if actual is None or int(actual) != expected:
            return f"production W8 requires {field}={expected}; got {actual!r}"
    expected_names = {
        f"model.layers.{layer}.{role}"
        for layer in range(_ASCEND_910B3_W8_SPEED_CONFIG["num_hidden_layers"])
        for role in ("ffn.key", "ffn.value")
    }
    actual_names = set(selected_names)
    if actual_names != expected_names or len(selected_names) != len(expected_names):
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        return (
            "production W8 requires exactly all 64 canonical 7.2B FFN "
            f"projections; selected={len(selected_names)}, "
            f"missing={missing[:4]}, unexpected={unexpected[:4]}"
        )
    return None


@torch.no_grad()
def quantize_ascend_w8a16(
    model: nn.Module,
    *,
    policy: str = "speed",
    strict: bool = False,
    chunk_rows: int = 1024,
) -> list[str]:
    """Replace exact promoted large FFN projections with packed W8 modules.

    The model must already be fp16 on NPU. The default speed policy refuses
    unrecognized cards/shapes/roles. ``memory`` still stays limited to the same
    known operator-compatible large FFN shapes but does not make a speed claim.
    """

    policy = str(policy).strip().lower()
    parameter = next(model.parameters(), None)
    if parameter is None:
        return []
    if parameter.device.type != "npu":
        if strict:
            raise RuntimeError("Ascend W8 conversion requires a model resident on NPU")
        return []
    torch_npu_module = import_torch_npu(required=True)
    import torch

    device_name = str(torch.npu.get_device_name(parameter.device.index or 0))
    torch_version = str(torch.__version__)
    torch_npu_version = str(getattr(torch_npu_module, "__version__", ""))
    cann_version = detect_cann_version(torch_npu_module)
    selected: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        decision = ascend_w8a16_decision(
            name,
            module.in_features,
            module.out_features,
            device_name=device_name,
            torch_version=torch_version,
            torch_npu_version=torch_npu_version,
            cann_version=cann_version,
            rows=None,
            dtype=module.weight.dtype,
            policy=policy,
        )
        if decision.enabled:
            selected.append((name, module))
    rejection_reason = None
    if policy == "speed":
        rejection_reason = _speed_model_selection_reason(
            model,
            tuple(name for name, _ in selected),
        )
    elif not selected:
        rejection_reason = (
            "No exact Ascend W8A16 projection matched this model/card policy"
        )
    if rejection_reason is not None:
        if strict:
            raise RuntimeError(rejection_reason)
        return []
    replaced: list[str] = []
    for name, module in selected:
        replacement = AscendW8A16Linear.from_float(
            module,
            chunk_rows=chunk_rows,
            admitted_rows=(ASCEND_910B3_W8_SPEED_ROWS if policy == "speed" else ()),
        )
        _set_submodule(model, name, replacement)
        replaced.append(name)
    clear_graphs = getattr(model, "rwkv7_clear_native_graph_cache", None)
    if replaced and callable(clear_graphs):
        clear_graphs()
    return replaced


__all__ = [
    "ASCEND_910B3_W8_SPEED_ROWS",
    "AscendQuantDecision",
    "AscendW8A16Linear",
    "ascend_w8a16_decision",
    "quantize_ascend_w8a16",
]
