# RWKV-7 HF follow-up work

Last audited: **2026-08-20**.

## Scope and current boundary

The current HF milestone is complete; there are **no remaining blocking items**
for the released adapter scope. The list below contains post-release expansion
projects, not release blockers.

## Post-release expansion projects

- Extend current paired Qwen3.5 Prefill/Decode coverage to hardware without a
  retained Native paired matrix.
- Continue DPLR/WY compact-prefill kernel work and exact-card validation.
- Build native fused full-model W8/W4 paths that beat FP16 end to end, rather
  than relying on memory-only quantization claims.
- Expand AMD and non-NVIDIA exact-card performance evidence.
- Expand prebuilt kernel-wheel coverage beyond the initial CPython 3.11,
  CUDA 12.4, Torch 2.5/SM70 and Torch 2.6/SM89 lanes when matching exact-card
  builders are available.
- Add broader multi-GPU and long-context stress testing where hardware is
  available.

New evidence must be added to
[`bench/CURRENT_ARTIFACTS.json`](bench/CURRENT_ARTIFACTS.json) and must replace,
not accumulate beside, a superseded bundle on the same line.
