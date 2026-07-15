# RTX 5070 Laptop 2.9B groupwise MM4 evidence

Date: 2026-07-13

Hardware: one `NVIDIA GeForce RTX 5070 Laptop GPU`, exact `sm_120`, with
`8151 MiB` visible VRAM. Model: official `BlinkDL/rwkv7-g1` 2.9B converted to
the repository HF format and loaded as fp16 with `fused_recurrent`, no fused
norm, and repo code.

## Result

The previous full-matrix affine MM4 path was faster than fp16 in 7/7 cells but
failed greedy parity in 0/7. The new default-off K-grouped affine W4 format
stores fp16 scale and bias per 128 input features and uses fused paired-nibble
GEMV for bsz1 plus a tensor-core batched dot path from bsz2.

All seven fresh-process, same-shape, same-process paired-fp16 cells pass:

| Batch | Prompt | Decode | Decode / fp16 | Footprint / fp16 | Final cosine | Greedy |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 128 | `1.1562x` | `0.5402x` | `0.99969435` | pass |
| 1 | 128 | 512 | `1.0895x` | `0.5402x` | `0.99974322` | pass |
| 1 | 512 | 128 | `1.1188x` | `0.5402x` | `0.99974263` | pass |
| 1 | 2048 | 128 | `1.1085x` | `0.5402x` | `0.99971509` | pass |
| 2 | 128 | 128 | `1.1397x` | `0.5402x` | `0.99969685` | pass |
| 4 | 128 | 128 | `1.1593x` | `0.5402x` | `0.99969268` | pass |
| 8 | 128 | 128 | `1.1656x` | `0.5402x` | `0.99966836` | pass |

The speed range is `1.0895x-1.1656x` with median `1.1397x`. Greedy parity is
7/7, minimum prompt/final cosine is `0.99965918/0.99966836`, and maximum peak
VRAM is `3604.4 MiB`.

## Development gates

`weight-oracle.jsonl` compares the old affine MM4, MM8, and groupwise W4 at
group sizes 32/64/128 on exact layer-0 FFN key/value and `lm_head` tensors.
Every groupwise format reduces weight and random-activation error relative to
the old MM4. Group32 has the lowest error but is too expensive end to end.

The preserved probes show why group128 and batched dot were selected:

| Probe | Decode / fp16 | Greedy | Conclusion |
|---|---:|---:|---|
| group32 bsz1 | `0.6045x` | pass | quality-first, too many scale groups |
| group64 bsz1 | `0.9049x` | pass | closer, still below fp16 |
| Q4_K_M-inspired W4/W8 bsz1 | `0.3955x` | pass | quality passes, mixed kernels regress |
| group128 bsz2 before batched dot | `0.9748x` | pass | independent GEMV misses speed gate |

After adding the groupwise tensor-core batched dot path, the same bsz2 probe
reached `1.2185x` before the final exact 128-token matrix.

## Scope

`mm4_groupwise` is reachable only through explicit benchmark/API selection and
all fused FFN flags remain default-off. This closes the exact RTX 5070 Laptop,
2.9B, fp16, group128 matrix. It does not change kernel policy defaults, close
V100 MM4 quality, close V100 MM8 speed, or provide a same-card 7.2B fp16 gate.

Raw strict rows are in `results.jsonl`; machine-readable acceptance is in
`summary.json`. Development probes are retained under `probes/`.
