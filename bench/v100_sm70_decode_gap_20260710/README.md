# V100 fused decode evidence (2026-07-10)

This directory records same-host FP16 decode measurements on an idle Tesla V100 PCIe 32GB (GPU1). The HF figures are the public `rwkv7_forward_token` API backed by `native_graph`; Albatross uses `faster3a` with `wkv=fp32io16`. Each throughput row uses 100 measured decode steps after warmup. GPU0 contention runs were excluded.

Acceptance labels used by this project: P1 >= 0.55x, P2 >= 0.75x, P3 >= 0.90x Albatross.

| Model | bsz | HF API tok/s | Albatross tok/s | Ratio | Gate | Peak VRAM MiB |
|---|---:|---:|---:|---:|:---:|---:|
| 0.1b | 1 | 637.9 | 788.19 | 0.809x | P2 | 405.5 |
| 0.1b | 2 | 1114.0 | 1504.72 | 0.740x | P1 | 428.4 |
| 0.1b | 4 | 1852.8 | 2612.88 | 0.709x | P1 | 470.4 |
| 0.1b | 8 | 3531.7 | 3611.88 | 0.978x | P3 | 549.6 |
| 0.4b | 1 | 331.5 | 469.16 | 0.707x | P1 | 927.6 |
| 0.4b | 2 | 573.4 | 810.76 | 0.707x | P1 | 973.5 |
| 0.4b | 4 | 970.6 | 1281.16 | 0.758x | P2 | 1068.9 |
| 0.4b | 8 | 1855.7 | 1565.95 | 1.185x | P3 | 1251.7 |
| 1.5b | 1 | 162.0 | 239.12 | 0.677x | P1 | 3023.9 |
| 1.5b | 2 | 261.1 | 415.21 | 0.629x | P1 | 3106.8 |
| 1.5b | 4 | 459.1 | 594.33 | 0.772x | P2 | 3288.1 |
| 1.5b | 8 | 874.0 | 860.64 | 1.016x | P3 | 3642.5 |

## Result

- All 12 model/batch rows pass P1.
- 0.1B bsz1 and 0.4B/1.5B bsz4 pass P2.
- All three bsz8 rows pass P3; 0.4B and 1.5B bsz8 exceed Albatross.
- The raw recurrent A/B passes 32-step greedy equality at 0.4B and 1.5B bsz2; see `raw_recurrent_ab/e2e.jsonl`.

## Scope

This is decode evidence only. It does not claim that prefill, W8/W4, training, or untested GPU families have reached Albatross parity. Raw JSONL files are retained for independent analysis.
