# V100 acceptance summary

Evidence validation: **PASS**.

| Lane | Coverage | Result | Boundary |
|---|---:|---|---|
| Production close | 12 dense decode + 12 dense prefill + 48 selected-module quant rows | Albatross minima `0.908x` decode / `0.9298x` prefill | Selected-module W8/W4, not full-memory quant |
| Full-FLA Qwen | 2/2 cells | raw prefill/decode min `2.815921x/5.270432x`; active-work min `2.285574x/4.277804x` | Only 1.5B/2B, P512/D64, B1/B8, dense fp16 |
| Torch-fallback Qwen diagnostic | 216/216 cells | prefill/decode min `1.246447x/1.002879x` | Not an optimized-Qwen comparison |

## Disclosed boundaries

- Full-FLA B1 peak VRAM ratio is `1.024885x`; only 1/2 cells use no more peak VRAM.
- The historical 216-cell matrix is pinned to `--required-reference-backend torch`.
- The production W8/W4 speed lane quantizes selected modules. Full-memory native MM8/MM4 remains open.
- Inference speed does not establish instruction, reasoning, math, code or multilingual quality.

## GPU-required follow-ups

- `qwen_full_fla_expansion`: Expand beyond 1.5B/2B, prompt512/decode64 and B1/B8.
- `full_memory_native_quant`: Draft PR #21 records MM4 speed with greedy mismatches and MM8 at 0/21 speed cells per model; no full-memory path is promoted.
- `large_training_and_zero`: Longer training and ZeRO resume beyond the promoted smoke/model sizes remain open.
