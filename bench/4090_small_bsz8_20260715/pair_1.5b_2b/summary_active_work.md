# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `6/6` cells.

Required Qwen backend: `fla`; verified: `6/6` cells.

Required Qwen full fusion: `true`; verified: `6/6` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 1.042x | 1.115x | 1.965x | 6/6 |
| Decode RWKV/Qwen | 5.637x | 5.658x | 5.662x | 6/6 |
| Model footprint RWKV/Qwen | 0.812x | 0.812x | 0.812x | 6/6 |
| Peak VRAM RWKV/Qwen | 1.105x | 1.158x | 1.208x | 0/6 |
| Runtime working set RWKV/Qwen | 2.458x | 3.259x | 5.721x | 0/6 |
| Active parameters RWKV/Qwen | 0.812x | 0.812x | 0.812x | 6/6 |
| Prefill tok/s per active-B | 1.284x | 1.373x | 2.421x | 6/6 |
| Decode tok/s per active-B | 6.945x | 6.971x | 6.976x | 6/6 |
| Prefill active-param work rate | 0.846x | 0.905x | - | 6/6 |
| Decode active-param work rate | 4.575x | 4.593x | - | 6/6 |

Strict speed cells: `6/6`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 6 | 1.042x / 1.115x | 5.637x / 5.658x | - | - | - | - | - |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
