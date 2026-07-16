# Tesla V100 production packed-MM4 BN/TN decode — 2026-07-16

Status: **production gate passed for three exact V100 FP16/MM4 cached-decode
profiles**.

This artifact promotes three explicit load-time configurations. It does not
promote a universal W4 policy, cross-GPU defaults, or full-memory W4 prefill.
The exact software and hardware environment is recorded in
[`environment.txt`](environment.txt).

## Physical grid

The exact-SM70 extension has two execution modes:

| Logical rows `M` | Activations | Packed operation | CUDA row grid |
|---:|---|---|---|
| `M=1` | FP16 (A16) | packed-W4 multiply-accumulate | one logical row |
| `M>1` | dynamic rowwise INT8 (A8) | DP4A against packed W4 | `ceil(M / 8)` row tiles |

For rowwise W4, `BN` is the number of output columns owned by one CTA and
`TN` is the number of output columns accumulated by one 32-lane warp. CUDA
threads are `(BN / TN) * 32`. For groupwise W4, `TN` is owned by one 16-lane
subwarp and CUDA threads are `(BN / TN) * 16`. Batched kernels use
`grid.y = ceil(M / 8)`, so one launch supports every measured B1/B2/B4/B8
shape without host-side row loops.

The extension instantiates these 13 legal physical pairs:

```text
(1,1), (2,1), (4,1), (4,2), (4,4), (8,1), (8,2), (8,4),
(16,1), (16,2), (16,4), (32,1), (32,2)
```

Rowwise and groupwise routes have independent exact `(M,K,N)` tables. The
promoted groupwise `lm_head` routes for cached decode are:

| Profile | Head `(K,N)` | Group | B1 BN/TN | B2 BN/TN | B4 BN/TN | B8 BN/TN |
|---|---|---:|---:|---:|---:|---:|
| 1.5B memory | `(2048,65536)` | 128 | `8/1` | `8/1` | `8/1` | `8/1` |
| 2.9B speed | `(2560,65536)` | 256 | `32/1` | `8/1` | `8/1` | `32/1` |
| 7.2B memory | `(4096,65536)` | 128 | `16/1` | `4/1` | `8/1` | `8/1` |

The tuning corpus contains 432 rowwise rows, 108 group-128 head rows and 52
group-256 head rows. Best correct candidates beat the same-shape FP16
operation in 35/48, 11/12 and 4/4 shape groups respectively. These isolated
rows select physical grids only; end-to-end promotion is decided by the
paired matrices below.

## End-to-end acceptance

The matrices use official `rwkv7-g1g-{1.5b,2.9b,7.2b}-hf` checkpoints, FP16,
`fused_recurrent`, `native_graph`, and a paired FP16 baseline in the same fresh
process. Every cell uses two warmups and three timing repeats. The tightest
cell in each profile uses five repeats; the tables report those five-repeat
rows directly rather than substituting a faster short-run probe.

A cell passes only if:

- decode throughput is at least the paired FP16 throughput;
- `model_footprint_mb` is lower than FP16;
- final-logit cosine is at least `0.998`;
- the complete timed greedy sequence equals FP16; and
- every repeat produces the same greedy-sequence SHA256.

`model_footprint_mb` is static loaded-model storage. It is not peak VRAM, and
this artifact does not use it as a substitute for a peak-VRAM claim.

### 1.5B memory / group128 / fused epilogue

This profile replaces 49 modules. Model footprint is 1571.8 MB versus
2913.3 MB for FP16 (`0.5395x`).

| B | Prompt | Decode | Prefill / FP16 | Decode / FP16 | Quant prefill | Quant decode | Final cosine | Complete greedy | Repeat SHA | Repeats |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| 1 | 128 | 128 | `0.3187x` | `1.1837x` | 1892.0 tok/s | 271.9 tok/s | `0.99850047` | yes | stable | 3 |
| 2 | 128 | 128 | `0.2056x` | `1.0651x` | 2101.8 tok/s | 402.4 tok/s | `0.99834460` | yes | stable | 3 |
| 4 | 128 | 128 | `0.1531x` | `1.0255x` | 2187.0 tok/s | 620.4 tok/s | `0.99835515` | yes | stable | 5 |
| 8 | 128 | 128 | `0.1276x` | `1.0470x` | 2250.5 tok/s | 942.1 tok/s | `0.99828702` | yes | stable | 3 |
| 1 | 512 | 128 | `0.1952x` | `1.1836x` | 2076.8 tok/s | 272.0 tok/s | `0.99850303` | yes | stable | 3 |
| 1 | 2048 | 128 | `0.1612x` | `1.1776x` | 2157.4 tok/s | 270.5 tok/s | `0.99835551` | yes | stable | 3 |
| 1 | 128 | 512 | `0.3192x` | `1.1783x` | 1887.6 tok/s | 270.9 tok/s | `0.99852455` | yes | stable | 3 |

### 2.9B speed / group256 / unfused

This head-only profile replaces one module. Model footprint is 5382.5 MB
versus 5622.4 MB for FP16 (`0.9573x`).

| B | Prompt | Decode | Prefill / FP16 | Decode / FP16 | Quant prefill | Quant decode | Final cosine | Complete greedy | Repeat SHA | Repeats |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| 1 | 128 | 128 | `1.0125x` | `1.0337x` | 3323.8 tok/s | 128.8 tok/s | `0.99970627` | yes | stable | 3 |
| 2 | 128 | 128 | `1.0046x` | `1.0330x` | 5930.2 tok/s | 228.3 tok/s | `0.99968970` | yes | stable | 3 |
| 4 | 128 | 128 | `1.0050x` | `1.0224x` | 8760.3 tok/s | 365.3 tok/s | `0.99968600` | yes | stable | 3 |
| 8 | 128 | 128 | `1.0006x` | `1.0111x` | 10453.4 tok/s | 519.1 tok/s | `0.99965668` | yes | stable | 5 |
| 1 | 512 | 128 | `1.0050x` | `1.0337x` | 6860.2 tok/s | 128.9 tok/s | `0.99974191` | yes | stable | 3 |
| 1 | 2048 | 128 | `1.0014x` | `1.0346x` | 8295.9 tok/s | 128.7 tok/s | `0.99974895` | yes | stable | 3 |
| 1 | 128 | 512 | `1.0603x` | `1.0337x` | 3271.2 tok/s | 129.0 tok/s | `0.99974155` | yes | stable | 3 |

### 7.2B memory / group128 / unfused

This profile replaces 193 modules. Model footprint is 4137.5 MB versus
13731.3 MB for FP16 (`0.3013x`).

| B | Prompt | Decode | Prefill / FP16 | Decode / FP16 | Quant prefill | Quant decode | Final cosine | Complete greedy | Repeat SHA | Repeats |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| 1 | 128 | 128 | `0.1514x` | `1.8416x` | 316.4 tok/s | 103.5 tok/s | `0.99915242` | yes | stable | 3 |
| 2 | 128 | 128 | `0.0947x` | `1.1883x` | 322.3 tok/s | 121.8 tok/s | `0.99916017` | yes | stable | 3 |
| 4 | 128 | 128 | `0.0787x` | `1.1368x` | 326.2 tok/s | 189.5 tok/s | `0.99903870` | yes | stable | 3 |
| 8 | 128 | 128 | `0.0716x` | `1.0810x` | 327.5 tok/s | 262.8 tok/s | `0.99906474` | yes | stable | 5 |
| 1 | 512 | 128 | `0.0865x` | `1.8422x` | 323.2 tok/s | 103.9 tok/s | `0.99925268` | yes | stable | 3 |
| 1 | 2048 | 128 | `0.0786x` | `1.8369x` | 325.2 tok/s | 103.6 tok/s | `0.99911547` | yes | stable | 3 |
| 1 | 128 | 512 | `0.1516x` | `1.8336x` | 316.8 tok/s | 103.6 tok/s | `0.99924892` | yes | stable | 3 |

## Fused FFN epilogue

The generic `MM4Linear.forward()` ABI remains a plain Linear. Only
`rwkv7_forward_relu2()` and `rwkv7_forward_add()` select the fused
ReLU-squared and residual-add epilogues. The residual input is not modified in
place, and native FFN dispatch calls these explicit methods so generic HF/FLA
callers cannot apply an epilogue twice.

The 1.5B B4 cell is the promotion boundary:

| Profile / cell | Fused | Prefill / FP16 | Decode / FP16 | Final cosine | Greedy / repeat |
|---|---|---:|---:|---:|---|
| 1.5B memory, B4/P128/D128 | no | `0.1532x` | `0.9997x` | `0.99835533` | yes / stable |
| 1.5B memory, B4/P128/D128 | yes | `0.1531x` | `1.0255x` | `0.99835515` | yes / stable |
| 2.9B memory, B4/P128/D128 | yes | `0.1266x` | `0.9997x` | `0.99850547` | yes / stable |

Fused epilogues therefore remain globally default-off. The exact 1.5B
deployment profile opts in; the promoted 2.9B and 7.2B profiles remain
unfused.

## Rejected schedules

- The 2.9B group-128 speed head reached only `0.9984x` FP16 decode at B8 over
  five repeats. Group 256 with physical `BN/TN=32/1` reaches `1.0111x` and is
  the promoted route.
- The 2.9B group-128 memory route reached `0.9888x` at B4 despite a lower
  `0.5310x` model footprint. It is retained as a memory result, not promoted
  as a speed route.
- Enabling fused epilogues did not close that 2.9B memory B4 boundary
  (`0.9997x`).
- The tiled WMMA prefill prototype reached only `0.054305x-0.115137x` FP16 on
  the measured shapes. Its runtime prototype was removed.

Production therefore keeps separate rowwise/groupwise tables, uses group 256
only for the exact 2.9B speed head, and does not infer end-to-end promotion
from a microbenchmark winner.

## Scope and fallback

The production claim is limited to Tesla V100 (`sm_70`), FP16, the three
checkpoint/configuration pairs above, and the seven measured cells per pair.
Group size and fused state are explicit load-time configuration and are part
of the fail-closed runner contract. The repository defaults remain rowwise
group size 0 with fused epilogues disabled.

The exact-SM70 CUDA extension is not dispatched on another compute capability.
Other NVIDIA families, ROCm and CPU retain their existing quantized or dense
paths and require independent evidence. The 1.5B and 7.2B memory profiles do
not pass paired FP16 prefill speed; only the head-only 2.9B speed profile passes
prefill in all seven measured cells (`1.0006x-1.0603x`).

## Reproduce

```bash
export CUDA_HOME=/usr/local/cuda-12.4
export TORCH_CUDA_ARCH_LIST=7.0
export PYTHONPATH=$PWD
export RWKV_V7_ON=1
export RWKV7_FAST_TOKEN_BACKEND=native_graph

python bench/run_v100_sm70_mm4_production_matrix.py \
  --model 1.5b=/path/to/rwkv7-g1g-1.5b-hf \
  --policy memory --group-size 128 --group-policy lm_head \
  --fused-epilogue true \
  --output-dir /tmp/v100-mm4-1p5b

python bench/run_v100_sm70_mm4_production_matrix.py \
  --model 2.9b=/path/to/rwkv7-g1g-2.9b-hf \
  --policy speed --group-size 256 --group-policy lm_head \
  --fused-epilogue false \
  --output-dir /tmp/v100-mm4-2p9b

python bench/run_v100_sm70_mm4_production_matrix.py \
  --model 7.2b=/path/to/rwkv7-g1g-7.2b-hf \
  --policy memory --group-size 128 --group-policy lm_head \
  --fused-epilogue false \
  --output-dir /tmp/v100-mm4-7p2b
```

Each command exits nonzero if a cell is missing, a stored row has the wrong
policy/group/fused configuration, or any production gate fails. A successful
`summary.json` reports `completed=7`, `failures=0`, and seven passes for
decode, footprint, logits, complete greedy equality and repeat determinism.

Raw files in this directory retain the three promoted matrices and summaries,
the rowwise/group-128/group-256 physical-grid sweeps, independent weakest-cell
confirmations, and every rejected route described above.
