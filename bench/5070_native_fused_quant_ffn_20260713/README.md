# RTX 5070 Laptop native fused quant FFN matrix

Date: 2026-07-13

Hardware: one `NVIDIA GeForce RTX 5070 Laptop GPU`, observed `sm_120`, 8 GB.

This artifact is the exact-card 1.5B expanded end-to-end matrix for the
default-off native MM8/MM4 FFN experiments. It covers:

- batch sweep: bsz 1/2/4/8 at prompt128/decode128;
- context sweep: prompt512/2048 at bsz1/decode128;
- sustained decode: prompt128/decode512 at bsz1;
- fp16 baseline, MM8 off/up/deep, and MM4 off/up;
- three timing repeats per row, logits cosine, greedy token, model footprint,
  and peak VRAM.

All `42/42` valid rows pass and all `35/35` quantized rows preserve the fp16
greedy token. The original local run attempted 42 rows before Windows repo-code
staging was fixed; those infrastructure failures were caused by unprivileged
symbolic-link creation and are retained only in the D-drive audit directory.
The committed artifact contains the clean 42-row rerun after same-volume
hardlink staging was added.

| Quant / fusion | Cells | Median decode / fp16 | Range | Footprint / fp16 | Min final cosine | Greedy |
|---|---:|---:|---:|---:|---:|---:|
| MM8 off | 7 | `0.9551x` | `0.9413x-1.0820x` | `0.6932x` | `0.99995482` | 7/7 |
| MM8 up | 7 | `0.9620x` | `0.9472x-1.0852x` | `0.6932x` | `0.99995518` | 7/7 |
| MM8 deep | 7 | `0.9671x` | `0.9471x-1.0893x` | `0.6932x` | `0.99995530` | 7/7 |
| MM4 off | 7 | `0.8171x` | `0.8025x-0.9868x` | `0.5394x` | `0.99809140` | 7/7 |
| MM4 up | 7 | `0.8171x` | `0.7870x-0.9911x` | `0.5394x` | `0.99808919` | 7/7 |

MM8 up beats unfused MM8 in 7/7 cells with a median `1.0062x` paired gain.
The deeper MM8 down+residual epilogue beats up-only in 5/7 cells with a median
`1.0059x` gain, but its minimum paired ratio is `0.9888x`. MM4 up wins 4/7
cells and is effectively neutral at the median, while both MM4 modes remain
slower than fp16.

This is a useful Blackwell architecture result, not a promotion. Both
`RWKV7_NATIVE_GRAPH_FUSED_QUANT_FFN` and
`RWKV7_NATIVE_GRAPH_FUSED_QUANT_FFN_DOWN_ADD` remain disabled by default. The
deep flag is independent because the matching V100 micro and end-to-end rows
regress on common bsz1/2/4 shapes.

Files:

- `results.jsonl`: 42 valid end-to-end rows.
- `summary.json` / `summary.md`: generated completeness and paired analysis.
- `environment.txt`: local software and exact-card snapshot.
