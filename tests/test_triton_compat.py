from __future__ import annotations

from dataclasses import fields, is_dataclass
import sys
import types


def test_legacy_attrs_descriptor_is_dataclass_compatible() -> None:
    from rwkv7_hf.triton_compat import apply_runtime_compat

    apply_runtime_compat()
    try:
        import triton.compiler.compiler as compiler
    except Exception:
        return
    descriptor = getattr(compiler, "AttrsDescriptor", None)
    if descriptor is None:
        return
    # torch._inductor calls dataclasses.fields() during DeepSpeed import.
    # The compatibility shim must satisfy that contract.
    fields(descriptor)


def test_triton32_metadata_patch_preserves_native_behavior(monkeypatch) -> None:
    from rwkv7_hf.triton_compat import patch_legacy_attrs_descriptor

    class NativeDescriptor:
        def __init__(self, marker=None):
            self.marker = marker

        def native_method(self):
            return self.marker

    original_init = NativeDescriptor.__init__
    original_repr = NativeDescriptor.__repr__
    original_eq = NativeDescriptor.__eq__
    compiler = types.ModuleType("triton.compiler.compiler")
    compiler.AttrsDescriptor = NativeDescriptor
    compiler_package = types.ModuleType("triton.compiler")
    compiler_package.compiler = compiler
    triton_package = types.ModuleType("triton")
    triton_package.compiler = compiler_package
    monkeypatch.setitem(sys.modules, "triton", triton_package)
    monkeypatch.setitem(sys.modules, "triton.compiler", compiler_package)
    monkeypatch.setitem(sys.modules, "triton.compiler.compiler", compiler)

    assert patch_legacy_attrs_descriptor() is False
    assert is_dataclass(NativeDescriptor)
    assert NativeDescriptor.__init__ is original_init
    assert NativeDescriptor.__repr__ is original_repr
    assert NativeDescriptor.__eq__ is original_eq
    assert NativeDescriptor("native").native_method() == "native"


def test_torch_compile_compat_is_software_and_card_scoped() -> None:
    from rwkv7_hf.triton_compat import torch_compile_compat_required

    assert torch_compile_compat_required(
        capability=(7, 5),
        torch_version="2.7.1+cu126",
        triton_version="3.3.1",
        legacy_attrs_missing=True,
    )
    assert not torch_compile_compat_required(
        capability=(7, 5),
        torch_version="2.8.0",
        triton_version="3.3.1",
        legacy_attrs_missing=True,
    )
    assert not torch_compile_compat_required(
        capability=(7, 5),
        torch_version="2.7.1",
        triton_version="3.2.0",
        legacy_attrs_missing=False,
    )
    assert not torch_compile_compat_required(
        capability=(8, 9),
        torch_version="2.7.1",
        triton_version="3.3.1",
        legacy_attrs_missing=True,
    )
    assert torch_compile_compat_required(
        capability=(12, 0),
        torch_version="2.6.0",
        triton_version="3.3.0",
        legacy_attrs_missing=True,
    )
