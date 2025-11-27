# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `18/18` cells.

Required Qwen backend: `fla`; verified: `18/18` cells.

Required Qwen full fusion: `true`; verified: `18/18` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 2.761x | 4.215x | 8.589x | 18/18 |
| Decode RWKV/Qwen | 6.533x | 7.208x | 8.502x | 18/18 |
| Model footprint RWKV/Qwen | 0.330x | 0.455x | 0.812x | 18/18 |
| Peak VRAM RWKV/Qwen | 0.387x | 0.508x | 0.849x | 18/18 |
| Runtime working set RWKV/Qwen | 1.890x | 2.734x | 8.755x | 0/18 |
| Active parameters RWKV/Qwen | 0.740x | 0.812x | 0.812x | 18/18 |
| Prefill tok/s per active-B | 3.402x | 5.693x | 11.601x | 18/18 |
| Decode tok/s per active-B | 8.049x | 9.736x | 10.474x | 18/18 |
| Prefill active-param work rate | 2.241x | 3.121x | - | reported 18/18 |
| Decode active-param work rate | 5.302x | 5.360x | - | reported 18/18 |

Strict speed cells: `18/18`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 6 | 2.772x / 3.212x | 6.533x / 6.574x | - | - | - | - | - |
| w4 | 6 | 2.761x / 3.537x | 8.386x / 8.438x | 0.794x | 1.283x | 1.262x | 0.407x | 0.516x |
| w8 | 6 | 4.209x / 5.586x | 7.164x / 7.208x | 1.147x | 1.096x | 1.096x | 0.561x | 0.641x |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
