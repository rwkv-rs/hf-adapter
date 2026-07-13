# RTX 5070 Laptop tuned deep-MM4 close

Date: 2026-07-13

Hardware: one `NVIDIA GeForce RTX 5070 Laptop GPU`, observed `sm_120`, 8 GB.

This artifact closes the 1.5B full-memory MM4 speed/footprint gate for the
seven-cell expanded matrix. The path combines:

- paired-nibble MM4 kernels that reuse each packed byte;
- fused FFN-up ReLU-square and FFN-down residual-add epilogues;
- exact-card GEMV tiles `BLOCK_PAIRS=64`, `BLOCK_N=256` for bsz1;
- an exact-card tensor-core dot route starting at bsz2 with `BLOCK_B=16`,
  `BLOCK_N=128`, and output-aware `BLOCK_PAIRS=64` for small projections or
  `128` for the 8192-wide FFN-up projection.

The measured kernel tiles are automatic only for device names containing
`5070`. `RWKV7_NATIVE_MM4_BLOCK_*` and `RWKV7_NATIVE_MM4_DOT_BLOCK_*` remain
authoritative overrides. The two FFN fusion flags remain default-off:

```text
RWKV7_NATIVE_GRAPH_FUSED_QUANT_FFN=1
RWKV7_NATIVE_GRAPH_FUSED_QUANT_FFN_DOWN_ADD=1
```

| Bsz | Prompt | Decode | MM4 tok/s | fp16 tok/s | MM4 / fp16 |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 128 | 117.8 | 103.7 | `1.1360x` |
| 2 | 128 | 128 | 231.8 | 185.1 | `1.2523x` |
| 4 | 128 | 128 | 457.3 | 365.1 | `1.2525x` |
| 8 | 128 | 128 | 847.8 | 702.8 | `1.2063x` |
| 1 | 512 | 128 | 109.2 | 97.1 | `1.1246x` |
| 1 | 2048 | 128 | 107.4 | 97.2 | `1.1049x` |
| 1 | 128 | 512 | 98.5 | 93.1 | `1.0580x` |

All `7/7` rows beat their same-process fp16 baseline. The minimum/median/
maximum ratios are `1.0580x/1.1360x/1.2525x`. Model footprint is `0.5394x`
fp16, minimum prompt/final cosine is `0.99767560/0.99809039`, and all `7/7`
greedy tokens match. Peak VRAM across the matrix is `1970.9 MiB`.

Each cell is a fresh process with a paired dense baseline, one warmup, and five
decode timing repeats. `results.jsonl` records the effective exact-card tiles
even though no tile override was set in the environment.

This closes MM4 for this exact model/card/matrix. It does not promote the
default-off fusion flags, close larger model sizes, or establish a tile policy
for RTX 5090, V100, or other cards.
