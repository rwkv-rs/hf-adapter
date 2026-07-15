# Native MM4 groupwise quality prototype

## Problem

The current native MM4 format uses one factorized affine approximation across
each complete matrix. It is fast on V100 and RTX 5070, but strict large-model
greedy checks fail: V100 loses 1/7, 1/7, and 3/7 cells across 1.5B/2.9B/7.2B,
while RTX 5070 2.9B loses 7/7. Fusion-off and fusion-on rows fail identically,
so the epilogue is not the source of the quality loss.

## Options

1. K-grouped affine W4: quantize each output column independently over groups
   of 32, 64, or 128 input features. This preserves W4 for every selected
   matrix and directly reduces local dynamic range. It adds group scale/bias
   traffic and needs a new fused kernel after quality is proven.
2. Q4_K_M-style mixed precision: keep sensitive FFN value/down and lm_head
   matrices at W8 while using current MM4 elsewhere. This is easy and likely to
   recover quality, but part of the gain comes from moving weights back to W8.
3. Activation calibration/AWQ-style scaling: optimize weight error against a
   calibration set. This can preserve W4 and speed, but adds dataset and
   checkpoint calibration complexity before the simpler local-range hypothesis
   has been tested.

Start with option 1 and compare group sizes 32, 64, and 128. This directly tests
whether local weight range is the root cause. Option 2 is the fallback if pure
groupwise W4 still misses the greedy gate.

## Prototype contract

- New code is isolated in `native_quant_mm4_groupwise.py` and is never selected
  by defaults or kernel policy.
- The first implementation is a torch quantize/dequantize oracle. It may
  materialize dequantized weights and is not a performance claim.
- Packed nibbles remain `[K, N/2]`; fp16 scale and bias are stored per
  `[K/group_size, N]` group.
- Unit tests cover packing, dequantization, module replacement, footprint, and
  linear output against the dense oracle.
- A safetensors weight probe compares current MM4, MM8, and groupwise W4 on
  exact 2.9B FFN key/value and lm_head shapes.
- Only after the linear oracle improves should one 2.9B bsz1 prompt128/decode1
  paired-fp16 greedy row run. A complete matrix is forbidden until that row
  passes quality.

## Acceptance

The quality prototype advances to a fused kernel only when it improves the
exact-weight and random-activation error over current MM4 and restores the
minimal 2.9B greedy token. Performance remains unaccepted until a later
card-local fused implementation also beats fp16 with lower footprint.

## Outcome

The exact-weight oracle passed for all three group sizes. Group32 had the best
local accuracy, and its first paired 2.9B row restored greedy with prompt/final
cosine `0.99981511/0.99975199`, but the torch oracle reached only `0.5937x`
fp16. A fused groupwise GEMV raised that FLA probe to `3.1406x` its paired fp16
baseline, while native_graph exposed the stricter cost at `0.6045x` fp16.

Group64 preserved greedy but reached only `0.9049x`. Group128 preserved greedy,
reached `1.2019x` on the initial bsz1 probe, and reduced footprint to `0.5402x`.
The first bsz2 implementation used independent GEMVs and reached `0.9748x`;
the groupwise tensor-core batched dot replacement raised the same probe to
`1.2185x`.

The final seven-cell 2.9B matrix passes speed, footprint, and greedy in 7/7
cells. Decode is `1.0895x-1.1656x` fp16, minimum final cosine is `0.99966836`,
and evidence is in `bench/5070_native_mm4_groupwise_20260713/`. The profile
remains explicit and default-off. V100 and other model/card combinations are
separate acceptance gates.
