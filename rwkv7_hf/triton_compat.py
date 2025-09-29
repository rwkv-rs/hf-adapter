# coding=utf-8
"""Runtime compatibility helpers for newer CUDA/Triton remote-code loads.

Some early newer-GPU images pair PyTorch 2.6-era Inductor/FLA code with newer
Triton 3.3 wheels. That combination removed the legacy
``triton.compiler.compiler.AttrsDescriptor`` import path while FLA and PyTorch
still reference it, and the FLA ``sqrelu`` torch.compile path can fail during
Triton code generation on recent architectures. Keep the workaround local,
conservative, and opt-out so converted HF model directories can run without
requiring users to patch site-packages by hand.
"""
from __future__ import annotations

import os
from typing import Any


def patch_legacy_attrs_descriptor() -> bool:
    """Restore Triton's legacy ``AttrsDescriptor`` import path when missing.

    Returns ``True`` when a shim was installed.  Triton <=3.2 already exposes
    the legacy class, and future PyTorch/FLA stacks may not need this at all.
    """

    try:
        import triton.compiler.compiler as compiler  # type: ignore
    except Exception:
        return False
    if hasattr(compiler, "AttrsDescriptor"):
        return False

    class AttrsDescriptor:
        def __init__(self, divisible_by_16=None, equal_to_1=None, **kwargs: Any):
            self.divisible_by_16 = tuple(divisible_by_16 or ())
            self.equal_to_1 = tuple(equal_to_1 or ())
            # Torch's fallback wrapper only checks these keys on newer backend
            # descriptors.  Keep them present for callers that inspect them.
            self.property_values = {"tt.divisibility": 16, "tt.equal_to": 1}
            for key, value in kwargs.items():
                setattr(self, key, value)

        @classmethod
        def from_dict(cls, data):
            data = data or {}
            props = data.get("arg_properties", {}) if isinstance(data, dict) else {}
            div = data.get("divisible_by_16", ()) if isinstance(data, dict) else ()
            eq = data.get("equal_to_1", ()) if isinstance(data, dict) else ()
            if isinstance(props.get("tt.divisibility"), (tuple, list)):
                div = props.get("tt.divisibility")
            if isinstance(props.get("tt.equal_to"), (tuple, list)):
                eq = props.get("tt.equal_to")
            return cls(divisible_by_16=div, equal_to_1=eq)

        def to_dict(self):
            return {"divisible_by_16": self.divisible_by_16, "equal_to_1": self.equal_to_1}

        def _attr_dict(self):
            out: dict[Any, list[tuple[str, int]]] = {}
            for idx in self.divisible_by_16:
                path = idx if isinstance(idx, tuple) else (int(idx),)
                out.setdefault(path, []).append(("tt.divisibility", 16))
            return out

        # Triton 3.3's AST path treats attrs as a mapping.  This is best-effort;
        # the adapter disables fragile torch.compile paths by default, but
        # exposing mapping methods keeps simple imports robust.
        def items(self):
            return self._attr_dict().items()

        def __iter__(self):
            return iter(self._attr_dict())

        def __len__(self):
            return len(self._attr_dict())

        def __getitem__(self, key):
            return self._attr_dict()[key]

        def __repr__(self):
            return f"AttrsDescriptor(divisible_by_16={self.divisible_by_16!r}, equal_to_1={self.equal_to_1!r})"

    compiler.AttrsDescriptor = AttrsDescriptor
    return True


def maybe_disable_blackwell_torch_compile() -> bool:
    """Disable the known-broken FLA torch.compile activation path on sm_120.

    Set ``RWKV7_BLACKWELL_TORCH_COMPILE=1`` to opt out once a newer PyTorch /
    Triton / FLA stack proves Inductor works on the target card.  The helper
    sets the import-time env flag *and* patches ``torch.compile`` to identity
    for already-imported torch processes, which is common in HF scripts.
    """

    if os.environ.get("RWKV7_BLACKWELL_TORCH_COMPILE", "").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        major, _minor = torch.cuda.get_device_capability(0)
    except Exception:
        return False
    if int(major) < 12:
        return False

    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    try:
        import torch._dynamo  # type: ignore

        torch._dynamo.config.suppress_errors = True
    except Exception:
        pass

    try:
        original = getattr(torch, "compile", None)
        if original is not None and not getattr(torch, "_rwkv7_blackwell_compile_patched", False):
            setattr(torch, "_rwkv7_original_compile", original)

            def _rwkv7_identity_compile(model=None, *args, **kwargs):
                if model is None:
                    return lambda fn: fn
                return model

            torch.compile = _rwkv7_identity_compile  # type: ignore[assignment]
            setattr(torch, "_rwkv7_blackwell_compile_patched", True)
    except Exception:
        pass
    return True


def apply_runtime_compat() -> dict[str, bool]:
    return {
        "legacy_attrs_descriptor": patch_legacy_attrs_descriptor(),
        "blackwell_torch_compile_disabled": maybe_disable_blackwell_torch_compile(),
    }
