# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `6/6` cells.

Required Qwen backend: `fla`; verified: `6/6` cells.

Required Qwen full fusion: `true`; verified: `6/6` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 2.811x | 3.258x | 7.526x | 6/6 |
| Decode RWKV/Qwen | 6.687x | 6.737x | 7.169x | 6/6 |
| Model footprint RWKV/Qwen | 0.812x | 0.812x | 0.812x | 6/6 |
| Peak VRAM RWKV/Qwen | 0.829x | 0.845x | 0.849x | 6/6 |
| Runtime working set RWKV/Qwen | 1.890x | 2.338x | 2.600x | 0/6 |
| Active parameters RWKV/Qwen | 0.812x | 0.812x | 0.812x | 6/6 |
| Prefill tok/s per active-B | 3.464x | 4.014x | 9.273x | 6/6 |
| Decode tok/s per active-B | 8.238x | 8.301x | 8.833x | 6/6 |
| Prefill active-param work rate | 2.282x | 2.644x | - | 6/6 |
| Decode active-param work rate | 5.427x | 5.469x | - | 6/6 |

Strict speed cells: `6/6`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 6 | 2.811x / 3.258x | 6.687x / 6.737x | - | - | - | - | - |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
