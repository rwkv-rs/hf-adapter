from __future__ import annotations

from dataclasses import fields


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
