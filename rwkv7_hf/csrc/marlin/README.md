# Vendored Marlin BF16 runtime

This directory contains the minimal BF16 Marlin torch.ops runtime used by the
RWKV7 exact-card W4 dispatch. The implementation is derived from GPTQModel's
Apache-2.0 Marlin runtime, itself adapted from vLLM and the original Marlin
project. See the adjacent `LICENSE` and the SPDX/copyright headers in each
source file.

RWKV7-specific modifications include:

- isolated source packaging and the `rwkv7_marlin_bf16` torch.ops namespace;
- explicit K/CTA-N/thread/SM/stage controls used by offline tuning;
- per-internal-launch BN/TN assertions, including mixed bulk/tail segments;
- BF16 U4B8 stage specializations used by the tuner;
- an optional final-reduction fused ReLU-square epilogue entered only through
  the explicit RWKV ABI.

The Tensor Core MMA/dequantization architecture remains Marlin-derived. This
directory is not a from-scratch GEMM implementation.
