# RWKV-7 prebuilt kernel wheel

This directory builds an optional, environment-specific binary companion for
`rwkv7-hf`. The ordinary adapter wheel remains portable and continues to work
without this package.

Builds must use the target Python, PyTorch, CUDA toolkit, C++ ABI, and GPU
architecture. The generated wheel embeds a machine-readable compatibility
manifest and is rejected by `rwkv7-hf` when any runtime field differs.

```bash
RWKV7_KERNEL_ARCH_LIST=8.9 \
python -m build --wheel --no-isolation kernel_wheel
```

Use `scripts/build_kernel_wheel.sh` for the validated build and inspection
workflow. Generated source, metadata, and binaries are intentionally ignored.


## Licenses

The wheel includes the repository MIT license and the Apache License 2.0 text
for CUDA code adapted from the attributed RWKV-LM/Albatross sources. Source
attributions are retained in the canonical adapter modules used by the builder.
