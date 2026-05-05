# V100 Qwen3.5-9B SDPA math policy

## Problem

On Tesla V100, Qwen3.5-9B with batch-eight distinct prompts passes the
StaticCache eager-to-raw-CUDA-Graph oracle, but the default DynamicCache and
StaticCache paths can select different SDPA implementations. A small early
logit difference changes a greedy token and then expands autoregressively. The
strict full-horizon DynamicCache-to-StaticCache gate therefore fails even
though raw graph replay is identical to StaticCache eager execution.

An isolated diagnostic that disables flash, memory-efficient, and cuDNN SDPA
while retaining PyTorch math SDPA restores the complete 128-step greedy match.
It also preserves the same FLA linear-attention kernels, repository Triton
causal convolution, StaticCache, and raw CUDA Graph scope.

## Design

Add an explicit Qwen-only `qwen_sdpa_policy` with two values:

- `auto`: enable all PyTorch SDPA backends and let PyTorch select the kernel.
- `math_only`: disable flash, memory-efficient, and cuDNN SDPA while enabling
  the math backend.

The policy is applied before model loading. Each result row records the
requested/effective policy plus the four effective backend switches. This is a
backend contract, not an untracked environment tweak.

For the strict V100 matrix, Qwen3.5-0.8B/2B/4B require `auto`; Qwen3.5-9B
requires `math_only` for all twelve cells. Per-cell policy switching is
forbidden. The per-model route manifest records the same policy and the paired
validator checks both row and manifest evidence.

## Rejected alternatives

Using eager attention is more invasive and expected to be slower. Relaxing the
DynamicCache-to-StaticCache full-greedy gate would hide a real semantic change
and is not acceptable. Repeating the same prompt across the correctness batch
would also hide batch-specific failures and remains forbidden.

## Verification

Unit tests cover argument validation, exact backend switch telemetry, per-pair
V100 policy enforcement, and route-manifest binding. GPU promotion additionally
requires B1/B8, prompt 128/512/2048, decode 128/512, seven timing samples,
finite logits, complete greedy agreement, same-cache cosine at least 0.9999,
one raw CUDA Graph launch, and stable cache pointers.
