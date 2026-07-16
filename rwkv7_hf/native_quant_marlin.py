# coding=utf-8
"""Inference-only BF16/W4 Marlin linear used by measured CUDA dispatches.

The CUDA implementation under :mod:`rwkv7_hf.csrc.marlin` is an Apache-2.0
derivative of GPTQModel/vLLM Marlin.  It is compiled lazily so importing the HF
adapter never requires CUDA or a compiler.  The first use needs a local CUDA
toolkit compatible with the installed PyTorch build; subsequent processes use
PyTorch's extension cache.

This module deliberately exposes a small contract: symmetric GPTQ-style W4,
group size 128, BF16 activations, and inference only.  Hardware selection stays
in ``native_quant_torchao`` so an unmeasured card cannot enter this path merely
because it can compile the kernel.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

try:  # pragma: no cover - optional dependency / platform
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


_MARLIN_LOCK = threading.Lock()
_MARLIN_LOADED = False
_MARLIN_LOAD_ERROR: BaseException | None = None
_MARLIN_NAMESPACE = "rwkv7_marlin_bf16"

# Mirrors vllm::kU4B8 / GPTQModel scalar_types.uint4b8.  The ID layout is
# exponent:u8, mantissa:u8, signed:bool, bias:i32, finite:bool, nan_repr:u8.
_MARLIN_U4B8_TYPE_ID = (4 << 8) + (8 << 17) + (1 << 50)


def _normalize_marlin_schedule(schedule):
    """Return an explicit Marlin K/CTA-N/thread/SM schedule or auto sentinels.

    Marlin historically calls its CTA output tile ``thread_n``.  It is *not*
    the per-output-writer TN used by :mod:`native_quant_bn_tn`; the production
    BN/TN layer maps this second value to BN and separately fixes the physical
    coalesced epilogue TN at eight BF16 columns.
    """

    if schedule is None:
        return (-1, -1, -1, -1, -1)
    if len(schedule) not in (3, 4, 5):
        raise ValueError(
            "Marlin schedule must be (tile_k, block_n, num_threads[, sms[, stages]])"
        )
    thread_k, thread_n, num_threads = (int(value) for value in schedule[:3])
    sms = -1 if len(schedule) == 3 else int(schedule[3])
    stages = -1 if len(schedule) < 5 else int(schedule[4])
    auto_tiles = (thread_k, thread_n, num_threads) == (-1, -1, -1)
    if not auto_tiles:
        if thread_k not in (64, 128) or thread_n not in (64, 128, 256):
            raise ValueError("unsupported Marlin tile_k/block_n schedule")
        if num_threads not in (128, 256) or num_threads % 32:
            raise ValueError("Marlin num_threads must be 128 or 256")
    if sms == 0 or sms < -1:
        raise ValueError("Marlin sms must be -1 or positive")
    if stages not in (-1, 2, 4):
        raise ValueError("Marlin stages must be -1, 2, or 4")
    return (thread_k, thread_n, num_threads, sms, stages)


def _marlin_source_root() -> Path:
    root = Path(__file__).resolve().parent / "csrc" / "marlin"
    if (root / "marlin_torch_bf16.cpp").is_file():
        return root
    # Transformers' dynamic-module cache follows Python imports but does not
    # preserve arbitrary package data. Converted checkpoints therefore ship a
    # deterministic Python source bundle that materializes into a user cache.
    from .native_quant_marlin_sources import materialize_marlin_sources

    return materialize_marlin_sources()


def _marlin_sources() -> list[str]:
    root = _marlin_source_root()
    sources = [
        root / "marlin_torch_bf16.cpp",
        root / "gptq_marlin_bf16.cu",
        root / "gptq_marlin_repack.cu",
        root / "awq_marlin_repack.cu",
    ]
    sources.extend(sorted(root.glob("kernel_bf16_*.cu")))
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise RuntimeError(f"RWKV7 Marlin sources are incomplete: {missing[0]}")
    return [str(path) for path in sources]


def _marlin_ops_registered() -> bool:
    if torch is None:
        return False
    try:
        namespace = getattr(torch.ops, _MARLIN_NAMESPACE)
        getattr(namespace, "gptq_marlin_gemm_bf16")
        getattr(namespace, "gptq_marlin_repack")
    except (AttributeError, RuntimeError):
        return False
    return True


def load_marlin_bf16_extension(*, verbose: bool | None = None) -> None:
    """Build/load the vendored BF16 Marlin torch.ops extension once."""

    global _MARLIN_LOADED, _MARLIN_LOAD_ERROR
    if torch is None or not torch.cuda.is_available():
        raise RuntimeError("RWKV7 Marlin requires CUDA")
    if _MARLIN_LOADED or _marlin_ops_registered():
        _MARLIN_LOADED = True
        return
    if _MARLIN_LOAD_ERROR is not None:
        raise RuntimeError("RWKV7 Marlin extension previously failed to load") from _MARLIN_LOAD_ERROR

    with _MARLIN_LOCK:
        if _MARLIN_LOADED or _marlin_ops_registered():
            _MARLIN_LOADED = True
            return
        try:
            from torch.utils.cpp_extension import CUDA_HOME, load

            if CUDA_HOME is None:
                raise RuntimeError(
                    "RWKV7 Marlin JIT needs a local CUDA toolkit; set CUDA_HOME "
                    "to the toolkit matching the PyTorch CUDA build"
                )
            root = _marlin_source_root()
            if verbose is None:
                verbose = str(os.environ.get("RWKV7_MARLIN_VERBOSE", "0")).lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
            load(
                name="rwkv7_marlin_bf16_ops",
                sources=_marlin_sources(),
                extra_include_paths=[str(root), str(Path(CUDA_HOME) / "include")],
                extra_cflags=["-O3", "-std=c++17", "-DENABLE_BF16"],
                extra_cuda_cflags=[
                    "-O3",
                    "-std=c++17",
                    "-DENABLE_BF16",
                    "-static-global-template-stub=false",
                    "--threads",
                    str(os.environ.get("NVCC_THREADS", "4")),
                    "-lineinfo",
                    "-Xptxas=-O3,-dlcm=ca",
                    "-diag-suppress=179,39,177",
                ],
                verbose=bool(verbose),
                is_python_module=False,
            )
            if not _marlin_ops_registered():
                raise RuntimeError("RWKV7 Marlin loaded without registering required torch.ops")
            _MARLIN_LOADED = True
        except BaseException as exc:  # retain the original compiler diagnostic
            _MARLIN_LOAD_ERROR = exc
            raise RuntimeError(f"RWKV7 Marlin BF16 extension failed to load: {exc}") from exc


def marlin_bf16_available() -> bool:
    """Return whether the runtime can be loaded, compiling it if necessary."""

    try:
        load_marlin_bf16_extension()
    except Exception:
        return False
    return True


def _permute_marlin_scales(scales, *, out_features: int):
    scale_perm: list[int] = []
    for i in range(8):
        scale_perm.extend(i + 8 * j for j in range(8))
    return scales.reshape(-1, 64)[:, scale_perm].reshape(-1, out_features).contiguous()


@torch.no_grad()
def _pack_symmetric_w4(weight, *, group_size: int):
    """Return Marlin-repacked U4B8 weights and permuted BF16 scales.

    The quantizer matches the canonical symmetric GPTQ representation: each
    group spans ``[-amax, +amax]`` with 15 intervals and stores signed values
    with an unsigned bias of eight.
    """

    if weight.ndim != 2:
        raise ValueError(f"Marlin W4 expects a 2D weight, got {tuple(weight.shape)}")
    out_features, in_features = (int(weight.shape[0]), int(weight.shape[1]))
    if in_features % int(group_size):
        raise ValueError("Marlin W4 input width must be divisible by group_size")
    if in_features % 8 or out_features % 64:
        raise ValueError("Marlin W4 requires input width divisible by 8 and output width by 64")

    groups = in_features // int(group_size)
    values = weight.float().reshape(out_features, groups, int(group_size))
    max_abs = values.abs().amax(dim=-1).clamp_min_(1.0e-8)
    scales_f32 = max_abs.mul_(2.0 / 15.0)
    quant = (
        torch.round(values / scales_f32.unsqueeze(-1) + 8.0)
        .clamp_(0, 15)
        .to(torch.int32)
        .reshape(out_features, in_features)
    )

    packed = torch.zeros(
        (in_features // 8, out_features),
        dtype=torch.int32,
        device=weight.device,
    )
    for nibble in range(8):
        packed.bitwise_or_(
            quant[:, nibble::8].t().contiguous() << (4 * nibble)
        )

    empty = torch.empty(0, dtype=torch.int32, device=weight.device)
    repack = getattr(torch.ops, _MARLIN_NAMESPACE).gptq_marlin_repack
    qweight = repack(packed, empty, in_features, out_features, 4)
    scales = _permute_marlin_scales(
        scales_f32.to(weight.dtype).t().contiguous(),
        out_features=out_features,
    )
    return qweight, scales


class MarlinW4Linear(torch.nn.Module):
    """BF16 activation / symmetric W4 weight Marlin linear."""

    def __init__(
        self,
        linear,
        *,
        group_size: int = 128,
        fp32_reduce: bool = True,
        schedule=None,
        production_bn_tn: bool = False,
        fuse_relu2: bool = False,
    ):
        super().__init__()
        if linear.weight.device.type != "cuda":
            raise ValueError("Marlin W4 requires a CUDA-resident Linear")
        if linear.weight.dtype != torch.bfloat16:
            raise ValueError("Marlin W4 currently requires BF16 weights and activations")
        if int(group_size) != 128:
            raise ValueError("the measured RWKV7 Marlin route currently requires group_size=128")
        load_marlin_bf16_extension()

        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        self.group_size = int(group_size)
        self.fp32_reduce = bool(fp32_reduce)
        self.schedule = _normalize_marlin_schedule(schedule)
        self.production_bn_tn = bool(production_bn_tn)
        self.fused_relu2 = bool(fuse_relu2)
        if self.fused_relu2 and linear.bias is not None:
            raise ValueError("fused ReLU2 requires a bias-free Linear")
        qweight, scales = _pack_symmetric_w4(linear.weight.detach(), group_size=self.group_size)
        self.register_buffer("qweight", qweight)
        self.register_buffer("scales", scales)
        self.register_buffer(
            "workspace",
            torch.zeros(
                max(torch.cuda.get_device_properties(linear.weight.device).multi_processor_count, 128),
                dtype=torch.int32,
                device=linear.weight.device,
            ),
        )
        self.register_buffer(
            "empty",
            torch.empty(0, dtype=torch.int32, device=linear.weight.device),
        )
        if linear.bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", linear.bias.detach().clone())

    def _apply_marlin(
        self,
        x2,
        out=None,
        *,
        schedule=None,
        expected_bn_tn=None,
        fuse_relu2: bool = False,
    ):
        op = getattr(torch.ops, _MARLIN_NAMESPACE).gptq_marlin_gemm_bf16
        thread_k, thread_n, num_threads, sms, stages = (
            self.schedule
            if schedule is None
            else _normalize_marlin_schedule(schedule)
        )
        if expected_bn_tn is None:
            if self.production_bn_tn:
                # -2 asks CUDA to validate BN after each internal row segment
                # has been formed. A logical GEMM can contain both a BN=256
                # bulk launch and a BN=128 low-row tail launch.
                expected_bn, expected_tn = -2, 8
            else:
                expected_bn, expected_tn = -1, -1
        else:
            expected_bn, expected_tn = (int(value) for value in expected_bn_tn)
        return op(
            x2,
            out,
            self.qweight,
            None,
            self.scales,
            None,
            self.empty,
            self.empty,
            self.empty,
            self.workspace,
            _MARLIN_U4B8_TYPE_ID,
            int(x2.shape[0]),
            self.out_features,
            self.in_features,
            thread_k,
            thread_n,
            num_threads,
            sms,
            stages,
            expected_bn,
            expected_tn,
            bool(fuse_relu2),
            True,
            False,
            self.fp32_reduce,
            False,
        )

    def forward(self, x):
        if x.dtype != torch.bfloat16 or x.device != self.qweight.device:
            raise ValueError("Marlin W4 input must be BF16 on the packed weight device")
        leading = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features).contiguous()
        # Preserve the nn.Linear contract for generic HF/FLA callers.  The
        # fused RWKV epilogue is entered only through rwkv7_forward_relu2();
        # otherwise an upstream FFN that still applies ReLU-square would do it
        # twice and silently corrupt logits.
        result = self._apply_marlin(x2, fuse_relu2=False).reshape(
            *leading, self.out_features
        )
        if self.bias is not None:
            result = result + self.bias
        return result

    def rwkv7_forward_into(self, x, out):
        x2 = x.reshape(-1, self.in_features).contiguous()
        out2 = out.reshape(-1, self.out_features)
        self._apply_marlin(x2, out=out2, fuse_relu2=False)
        if self.bias is not None:
            out.add_(self.bias)
        return out

    def rwkv7_forward_relu2(self, x):
        """Apply this Linear and the RWKV FFN ReLU-square in one epilogue."""

        if not self.fused_relu2:
            raise RuntimeError("this MarlinW4Linear was not enabled for fused ReLU2")
        if x.dtype != torch.bfloat16 or x.device != self.qweight.device:
            raise ValueError("Marlin W4 input must be BF16 on the packed weight device")
        leading = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features).contiguous()
        return self._apply_marlin(x2, fuse_relu2=True).reshape(
            *leading, self.out_features
        )

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"group_size={self.group_size}, schedule={self.schedule}, "
            f"production_bn_tn={self.production_bn_tn}, fused_relu2={self.fused_relu2}, "
            "bf16_w4_marlin"
        )

    def effective_bn_tn_plan(self, rows: int):
        if not self.production_bn_tn:
            return None
        from .native_quant_bn_tn import rtx5090_w4_launch_plan

        return rtx5090_w4_launch_plan(rows, self.out_features)


__all__ = [
    "MarlinW4Linear",
    "load_marlin_bf16_extension",
    "marlin_bf16_available",
]
