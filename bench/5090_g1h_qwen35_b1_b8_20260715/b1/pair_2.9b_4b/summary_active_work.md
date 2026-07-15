# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `6/6` cells.

Required Qwen backend: `fla`; verified: `6/6` cells.

Required Qwen full fusion: `true`; verified: `6/6` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 2.368x | 2.715x | 5.174x | 6/6 |
| Decode RWKV/Qwen | 5.006x | 5.236x | 5.301x | 6/6 |
| Model footprint RWKV/Qwen | 0.701x | 0.701x | 0.701x | 6/6 |
| Peak VRAM RWKV/Qwen | 0.710x | 0.713x | 0.717x | 6/6 |
| Runtime working set RWKV/Qwen | 1.063x | 1.489x | 1.626x | 0/6 |
| Active parameters RWKV/Qwen | 0.701x | 0.701x | 0.701x | 6/6 |
| Prefill tok/s per active-B | 3.379x | 3.874x | 7.381x | 6/6 |
| Decode tok/s per active-B | 7.142x | 7.471x | 7.563x | 6/6 |
| Prefill active-param work rate | 1.660x | 1.903x | - | 6/6 |
| Decode active-param work rate | 3.508x | 3.670x | - | 6/6 |

Strict speed cells: `6/6`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 6 | 2.368x / 2.715x | 5.006x / 5.236x | - | - | - | - | - |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
