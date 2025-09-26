# Albatross v3a vs v4 4090 tune smoke — 2026-07-03

GPU: `NVIDIA GeForce RTX 4090, 8.9, 570.124.06, 24564 MiB`

| Case | v3a tok/s | v4 tok/s | v4/v3a | Note |
|---|---:|---:|---:|---|
| B1T1 | 837.53 | 855.73 | 1.022x | v4 faster |
| B1T512 | 48311.51 | 58933.80 | 1.220x | v4 faster |
| B64T1 | 25130.68 | 25183.30 | 1.002x | v4 faster |
| B4T128 | 81847.50 | 89226.80 | 1.090x | v4 faster |
| B8T64 | 94940.28 | 96756.70 | 1.019x | v4 faster |

Interpretation: v4 improves all tested 4090 cases in this smoke, with the largest win on the prompt-prefill B1T512 case. Treat tuned Albatross reference as per-case/per-GPU, not a single v3a number.
