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
