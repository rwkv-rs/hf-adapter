# RTX 3090 exact-shape FP16 accumulation max-performance design

> Lifecycle: implemented and validated on 2026-08-12; retained as a historical
> decision record for the exact-card policy and evidence review.

## Context and decision

The first RTX 3090 latest-checkpoint matrix closed all 24 strict
parameter-adjusted Prefill cells, but it intentionally promoted only the
FP16-accumulation shapes already covered by the initial correctness sweep. A
second exact-card profile shows that the weakest 2.9B B8/P128 cell still spends
52.2% of instrumented Prefill time in the FFN and another 22% in dense R/K/V
and output projections. Fused output preparation is only 1.3%, so deeper
output-only fusion is not the next useful target.

The selected approach is to extend the existing scoped full-prefill FP16 GEMM
accumulation policy. This uses the RTX 3090 Tensor Core path for the dominant
dense GEMMs while retaining FP32 recurrent-state accumulation and the current
fused scan/output kernels. It changes no HF API, cache layout, checkpoint, or
other GPU-family default. The policy remains an exact tuple of hidden size,
layer count, batch size, and prompt block length; environment variables still
override it.

The screening gate requires at least a 5% paired same-process improvement.
Eight previously disabled shapes reach 1.1068x-1.4570x and are candidates. The
0.4B B1/P128 result is only 1.0079x and stays disabled as noise-sized telemetry.

## Alternatives rejected

An output-prep/application-only kernel is rejected for this phase because the
measured component is too small to move end-to-end throughput materially. A
new FFN-only Triton GEMM is also rejected: prior repository evidence shows that
standalone FFN kernels lose to cuBLAS, while this card-local accumulation mode
accelerates the existing cuBLAS-backed FFN without replacing it. Compact
WY/DPLR remains a higher-risk future path for recurrent-scan-bound B1 prompts,
not the fastest way to close the currently measured 3090 gap.

## Correctness, fallback, and tests

Every newly selected tuple must compare the scoped FP16 candidate with the
FP32-accumulation native Prefill oracle. Direct and chunk-carried Prefill must
preserve prompt and first-decode cosine at or above 0.9999 plus exact greedy
tokens. Tests must also prove that adjacent Ampere names, unlisted shapes,
non-FP16 dtypes, unsupported PyTorch runtimes, and explicit disable flags do
not enter the route.

Promotion requires an alternating-order A/B confirmation, unchanged peak VRAM,
the complete B1/B8 P128/P512/P2048 Qwen3.5 matrix, and repository CPU/document
tests. The full Qwen reference remains fail-closed on FLA plus Triton causal
convolution. This design concerns inference execution speed only and makes no
model-quality claim.
