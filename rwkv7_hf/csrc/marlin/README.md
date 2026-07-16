# Vendored Marlin BF16 runtime

This directory contains the minimal BF16 Marlin torch.ops runtime used by the
RWKV7 exact-card W4 dispatch. The implementation is derived from GPTQModel's
Apache-2.0 Marlin runtime, itself adapted from vLLM and the original Marlin
project. See the adjacent `LICENSE` and the SPDX/copyright headers in each
source file.

RWKV7 changes are intentionally limited to packaging the BF16 sources and
renaming the registered torch.ops namespace to `rwkv7_marlin_bf16`, preventing
a collision when GPTQModel is loaded in the same Python process.
