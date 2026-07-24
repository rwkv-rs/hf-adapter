"""Portable, fail-closed weight-only quantization for RWKV-7 on Ascend.

This module is intentionally import-safe on hosts without ``torch_npu``.  The
packed checkpoint format is ordinary PyTorch state_dict data.  Raw operator
measurements are retained as candidate evidence only: no W4/W8 tuple in this
module is production-promoted until an end-to-end model gate passes.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import functools
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .ascend_runtime import validate_ascend_stack

FORMAT_VERSION = 2
RAW_CANDIDATE_STACK = {
    "device_name": "Ascend910B3",
    "cann_version": "8.5.0",
    "torch_version": "2.9.0+cpu",
    "torch_npu_version": "2.9.0",
    "activation_dtype": "float16",
}
# Raw-op-only rows where a kernel win was measured.  These rows do *not*
# authorize model conversion or serving dispatch: the real 7.2B paired gate
# failed quality and/or latency.  Keep the name explicit to prevent raw kernel
# timings from being reported as a production verification.
RAW_CANDIDATE_BATCHES = {
    4: tuple(range(1, 9)),
    8: tuple(range(17, 29)),
}
RAW_CANDIDATE_FFN_SHAPES = ((4096, 16384), (16384, 4096))
PRODUCTION_PROMOTED_BATCHES: dict[int, tuple[int, ...]] = {}


class UnverifiedQuantShapeError(RuntimeError):
    """Raised when a quantized kernel has no accepted no-regression result."""


@functools.lru_cache(maxsize=1)
def _torch_npu():
    try:
        import torch_npu  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised on non-Ascend CI
        raise RuntimeError("torch_npu is required for Ascend execution") from exc
    return torch_npu


@functools.lru_cache(maxsize=1)
def _runtime_device_name() -> str | None:
    try:
        mod = _torch_npu()
        if not torch.npu.is_available():
            return None
        return str(mod.npu.get_device_name(0))
    except (AttributeError, RuntimeError):
        return None


@functools.lru_cache(maxsize=1)
def _runtime_torch_npu_version() -> str | None:
    try:
        return str(_torch_npu().__version__)
    except RuntimeError:
        return None


def _runtime_cann_version() -> str | None:
    """Best-effort CANN version lookup used only by the raw-candidate audit."""
    try:
        mod = _torch_npu()
    except RuntimeError:
        return None
    accessors = (
        getattr(getattr(mod, "npu", None), "get_cann_version", None),
        getattr(getattr(mod, "_C", None), "_npu_getCANNVersion", None),
    )
    for accessor in accessors:
        if callable(accessor):
            try:
                value = accessor()
            except (AttributeError, RuntimeError, TypeError):
                continue
            if value is not None:
                return str(value)
    return None


def raw_candidate_supported(
    in_features: int,
    out_features: int,
    batch: int,
    bit: int,
    *,
    dtype: torch.dtype = torch.float16,
    device_name: str | None = None,
    torch_version: str | None = None,
    torch_npu_version: str | None = None,
    cann_version: str | None = None,
) -> bool:
    """Return whether a tuple exactly matches the recorded raw-op candidate.

    This is diagnostic evidence, not permission to quantize a production
    model.  Callers must use :func:`should_quantize` for production policy.
    """
    if bit not in RAW_CANDIDATE_BATCHES or dtype is not torch.float16:
        return False
    if (int(in_features), int(out_features)) not in RAW_CANDIDATE_FFN_SHAPES:
        return False
    if int(batch) not in RAW_CANDIDATE_BATCHES[bit]:
        return False
    validated, _ = validate_ascend_stack(
        device_name=device_name if device_name is not None else _runtime_device_name(),
        cann_version=cann_version if cann_version is not None else _runtime_cann_version(),
        torch_version=torch_version if torch_version is not None else str(torch.__version__),
        torch_npu_version=(
            torch_npu_version
            if torch_npu_version is not None
            else _runtime_torch_npu_version()
        ),
    )
    return validated


def should_quantize(
    in_features: int,
    out_features: int,
    batch: int,
    bit: int,
    *,
    dtype: torch.dtype = torch.float16,
    device_name: str | None = None,
    torch_version: str | None = None,
    torch_npu_version: str | None = None,
    cann_version: str | None = None,
) -> bool:
    """Fail-closed production policy.

    All tuples currently return ``False``.  The parameters intentionally match
    :func:`raw_candidate_supported` so a caller cannot accidentally substitute
    raw operator support for an end-to-end promotion decision.
    """
    del (
        in_features,
        out_features,
        batch,
        bit,
        dtype,
        device_name,
        torch_version,
        torch_npu_version,
        cann_version,
    )
    return False


def _pack_int4_cpu(q_kn: Tensor) -> Tensor:
    """Pack signed int4 ``[K,N]`` into eight nibbles per int32."""
    if q_kn.ndim != 2 or q_kn.shape[1] % 8:
        raise ValueError("int4 output dimension must be divisible by 8")
    q = q_kn.to(device="cpu", dtype=torch.int64).reshape(q_kn.shape[0], -1, 8)
    shifts = torch.arange(8, dtype=torch.int64).reshape(1, 1, 8) * 4
    return torch.sum((q & 0xF) << shifts, dim=-1).to(torch.int32).contiguous()


def _unpack_int4_cpu(packed: Tensor, out_features: int) -> Tensor:
    """Unpack the checkpoint's int32 format into signed int8 ``[K,N]``."""
    p = packed.detach().to(device="cpu", dtype=torch.int64)
    shifts = torch.arange(8, dtype=torch.int64).reshape(1, 1, 8) * 4
    q = ((p.unsqueeze(-1) >> shifts) & 0xF)
    q = torch.where(q >= 8, q - 16, q).to(torch.int8)
    return q.reshape(p.shape[0], -1)[:, :out_features].contiguous()


@dataclass(frozen=True)
class QuantManifest:
    format_version: int
    implementation: str
    bit: int
    group_size: int
    in_features: int
    out_features: int
    bias: bool
    weight_layout: str
    scale_layout: str
    raw_candidate_stack: dict[str, str]
    raw_candidate_batches: tuple[int, ...]
    production_promoted: bool


class AscendWeightOnlyLinear(nn.Module):
    """Inference-only Ascend W8A16 or groupwise W4A16 linear layer.

    ``load_fp_weight`` accepts a normal ``[out_features, in_features]`` FP16
    weight and drops it after packing.  Consequently the module delivers a real
    weight-memory reduction; it does not hide a second FP16 fallback copy.
    Production policy rejects every tuple.  Run with
    ``enforce_verified_shape=False`` only inside an explicit acceptance
    harness; :func:`raw_candidate_supported` is not a production gate.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        *,
        bit: int = 8,
        group_size: int = 128,
        enforce_verified_shape: bool = True,
    ) -> None:
        super().__init__()
        if bit not in (4, 8):
            raise ValueError("bit must be 4 or 8")
        if in_features <= 0 or out_features <= 0:
            raise ValueError("linear dimensions must be positive")
        if bit == 4 and (in_features % group_size or out_features % 8):
            raise ValueError("W4 requires in_features % group_size == 0 and out_features % 8 == 0")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.bit = int(bit)
        self.group_size = int(group_size if bit == 4 else 0)
        self.enforce_verified_shape = bool(enforce_verified_shape)
        self.register_buffer("qweight", torch.empty(0, dtype=torch.int32 if bit == 4 else torch.int8))
        self.register_buffer("scales", torch.empty(0, dtype=torch.float16))
        self.register_buffer("offsets", torch.empty(0, dtype=torch.float16), persistent=bit == 4)
        self.register_buffer("bias", torch.empty(0, dtype=torch.float16), persistent=bias)
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))
        self.has_bias = bool(bias)
        # Per-M closures capture packed buffers and the raw op after one policy
        # check. They are deliberately non-persistent and cleared by .to().
        self._npu_fastpaths: dict[int, Any] = {}

    def _apply(self, fn, recurse: bool = True):
        self._npu_fastpaths.clear()
        return super()._apply(fn, recurse=recurse)

    @property
    def manifest(self) -> QuantManifest:
        return QuantManifest(
            format_version=FORMAT_VERSION,
            implementation="rwkv7_hf.ascend_quant_w4.AscendWeightOnlyLinear",
            bit=self.bit,
            group_size=self.group_size,
            in_features=self.in_features,
            out_features=self.out_features,
            bias=self.has_bias,
            weight_layout="[K,N] int8" if self.bit == 8 else "[K,N/8] int32, eight signed nibbles",
            scale_layout="[N]" if self.bit == 8 else "[K/group_size,N]",
            raw_candidate_stack=dict(RAW_CANDIDATE_STACK),
            raw_candidate_batches=RAW_CANDIDATE_BATCHES[self.bit],
            production_promoted=False,
        )

    @torch.no_grad()
    def load_fp_weight(self, weight: Tensor, bias: Tensor | None = None) -> "AscendWeightOnlyLinear":
        if tuple(weight.shape) != (self.out_features, self.in_features):
            raise ValueError(
                f"weight shape {tuple(weight.shape)} != {(self.out_features, self.in_features)}"
            )
        if not weight.dtype.is_floating_point:
            raise TypeError("source weight must be floating point")
        if self.bit == 4 and weight.dtype is torch.bfloat16:
            raise TypeError(
                "Ascend W4 conversion requires an FP16 source weight; BF16 is "
                "rejected before packing because the measured raw candidate is FP16-only"
            )
        original_device = weight.device
        if self.bit == 8:
            wf = weight.float()
            scale = (wf.abs().amax(dim=1) / 127.0).clamp_min(1e-8)
            q = torch.round(wf / scale[:, None]).clamp(-127, 127).to(torch.int8)
            self.qweight = q.t().contiguous().to(original_device)
            self.scales = scale.to(device=original_device, dtype=torch.float16).contiguous()
            self.offsets = torch.empty(0, device=original_device, dtype=torch.float16)
        else:
            wf = weight.float()
            groups = self.in_features // self.group_size
            grouped = wf.reshape(self.out_features, groups, self.group_size)
            scale = (grouped.abs().amax(dim=2) / 7.0).clamp_min(1e-8)
            q_nk = torch.round(grouped / scale[:, :, None]).clamp(-8, 7).to(torch.int32)
            q_kn = q_nk.reshape(self.out_features, self.in_features).t().contiguous()
            if original_device.type == "npu":
                packed = _torch_npu().npu_convert_weight_to_int4pack(q_kn)
            else:
                packed = _pack_int4_cpu(q_kn)
            self.qweight = packed.to(original_device)
            self.scales = scale.t().to(device=original_device, dtype=torch.float16).contiguous()
            self.offsets = torch.zeros_like(self.scales)
        if self.has_bias:
            if bias is None or tuple(bias.shape) != (self.out_features,):
                raise ValueError(f"bias must have shape {(self.out_features,)}")
            self.bias = bias.detach().to(device=original_device, dtype=torch.float16).contiguous()
        elif bias is not None:
            raise ValueError("module was constructed without bias")
        self.initialized = torch.tensor(True, device=original_device, dtype=torch.bool)
        self._npu_fastpaths.clear()
        return self

    @classmethod
    def from_float(
        cls,
        linear: nn.Linear,
        *,
        bit: int = 8,
        group_size: int = 128,
        enforce_verified_shape: bool = True,
    ) -> "AscendWeightOnlyLinear":
        result = cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            bit=bit,
            group_size=group_size,
            enforce_verified_shape=enforce_verified_shape,
        )
        return result.load_fp_weight(linear.weight.detach(), None if linear.bias is None else linear.bias.detach())

    def packed_weight_bytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (self.qweight, self.scales, self.offsets, self.bias)
        )

    def make_npu_fastpath(self, batch: int, *, dtype: torch.dtype = torch.float16):
        """Return a cached raw-op closure after a single acceptance check.

        Serving schedulers should create this once per accepted batch row count,
        then reuse it across layers/tokens or graph capture. This avoids repeating
        Python policy and buffer lookup on every projection.
        """
        if self.qweight.device.type != "npu":
            raise RuntimeError("packed buffers must be on an NPU")
        if self.enforce_verified_shape and not should_quantize(
            self.in_features, self.out_features, batch, self.bit, dtype=dtype
        ):
            raise UnverifiedQuantShapeError(
                "no production-promoted Ascend W4/W8 tuple: "
                f"bit={self.bit}, M={batch}, K={self.in_features}, "
                f"N={self.out_features}; raw candidate stack={RAW_CANDIDATE_STACK}"
            )
        op = _torch_npu().npu_weight_quant_batchmatmul
        qweight, scales = self.qweight, self.scales
        bias = self.bias if self.has_bias else None
        if self.bit == 8:
            if bias is None:
                def kernel(x):
                    return op(x, qweight, scales)
            else:
                def kernel(x):
                    return op(x, qweight, scales) + bias
        else:
            offsets, group_size = self.offsets, self.group_size
            if bias is None:
                def kernel(x):
                    return op(x, qweight, scales, offsets, None, None, None, group_size, 1)
            else:
                def kernel(x):
                    return op(x, qweight, scales, offsets, None, None, None, group_size, 1) + bias
        self._npu_fastpaths[int(batch)] = kernel
        return kernel

    def _cpu_forward(self, x2: Tensor) -> Tensor:
        if self.bit == 8:
            w = self.qweight.t().float() * self.scales.float()[:, None]
        else:
            q_kn = _unpack_int4_cpu(self.qweight, self.out_features).float()
            scale_kn = self.scales.float().repeat_interleave(self.group_size, dim=0)
            w = (q_kn * scale_kn).t().contiguous()
        out = F.linear(x2.float(), w, None)
        if self.has_bias:
            out = out + self.bias.float()
        return out.to(x2.dtype)

    def forward(self, x: Tensor) -> Tensor:
        if self.qweight.numel() == 0 or self.scales.numel() == 0:
            raise RuntimeError("load_fp_weight or load_state_dict must initialize the module")
        if x.shape[-1] != self.in_features:
            raise ValueError(f"last input dimension must be {self.in_features}")
        if x.dtype is not torch.float16:
            raise TypeError("accepted Ascend kernels are FP16-activation only")
        is_matrix = x.ndim == 2
        shape = x.shape[:-1] + (self.out_features,)
        x2 = x if is_matrix else x.reshape(-1, self.in_features)
        if x.device.type != "npu":
            result = self._cpu_forward(x2)
            return result if is_matrix else result.reshape(shape)
        batch = int(x2.shape[0])
        kernel = self._npu_fastpaths.get(batch)
        if kernel is None:
            kernel = self.make_npu_fastpath(batch, dtype=x.dtype)
        out = kernel(x2)
        return out if is_matrix else out.reshape(shape)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bit={self.bit}, group_size={self.group_size}, bias={self.has_bias}, "
            f"enforce_verified_shape={self.enforce_verified_shape}"
        )


def save_quantized_linear(module: AscendWeightOnlyLinear, directory: str | Path) -> Path:
    """Save a standard state_dict plus a human-readable format manifest."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(module.state_dict(), path / "quantized_linear.pt")
    (path / "quant_manifest.json").write_text(
        json.dumps(asdict(module.manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def load_quantized_linear(
    directory: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    enforce_verified_shape: bool = True,
) -> AscendWeightOnlyLinear:
    path = Path(directory)
    manifest: dict[str, Any] = json.loads((path / "quant_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"unsupported quant format version {manifest.get('format_version')}")
    module = AscendWeightOnlyLinear(
        manifest["in_features"],
        manifest["out_features"],
        manifest["bias"],
        bit=manifest["bit"],
        group_size=manifest["group_size"] or 128,
        enforce_verified_shape=enforce_verified_shape,
    )
    state = torch.load(path / "quantized_linear.pt", map_location=map_location, weights_only=True)
    # Empty constructor buffers intentionally resize to the packed checkpoint.
    for key in ("qweight", "scales", "offsets", "bias", "initialized"):
        if key in state:
            setattr(module, key, state[key])
    module.load_state_dict(state, strict=True)
    return module


__all__ = [
    "AscendWeightOnlyLinear",
    "PRODUCTION_PROMOTED_BATCHES",
    "QuantManifest",
    "RAW_CANDIDATE_BATCHES",
    "RAW_CANDIDATE_FFN_SHAPES",
    "RAW_CANDIDATE_STACK",
    "UnverifiedQuantShapeError",
    "load_quantized_linear",
    "raw_candidate_supported",
    "save_quantized_linear",
    "should_quantize",
]


def _set_quant_submodule(root: nn.Module, qualified_name: str, replacement: nn.Module) -> None:
    parent_name, _, leaf = qualified_name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, leaf, replacement)


@torch.no_grad()
def quantize_ascend_w4a16_candidate(
    model: nn.Module,
    *,
    group_size: int = 128,
    roles: tuple[str, ...] = ("ffn.key", "ffn.value"),
    require_explicit_candidate: bool = True,
) -> list[str]:
    """Pack exact 7.2B FFN projections for an explicit W4 gate only."""
    if require_explicit_candidate:
        raise RuntimeError(
            "Ascend W4 is not production-promoted; pass "
            "require_explicit_candidate=False only in an acceptance harness"
        )
    roles = tuple(str(role).lower() for role in roles)
    selected = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        role_ok = name.lower().endswith(roles)
        shape_ok = (module.in_features, module.out_features) in RAW_CANDIDATE_FFN_SHAPES
        if role_ok and shape_ok:
            selected.append((name, module))
    bf16 = [name for name, module in selected if module.weight.dtype is torch.bfloat16]
    if bf16:
        raise TypeError(
            "Ascend W4 candidate conversion is FP16-only and made no changes; "
            "BF16 source layers: " + ", ".join(bf16[:8])
        )
    replaced = []
    for name, module in selected:
        replacement = AscendWeightOnlyLinear.from_float(
            module, bit=4, group_size=group_size, enforce_verified_shape=False
        )
        _set_quant_submodule(model, name, replacement)
        replaced.append(name)
    return replaced


AscendW4A16Linear = AscendWeightOnlyLinear
__all__.extend(["AscendW4A16Linear", "quantize_ascend_w4a16_candidate"])
