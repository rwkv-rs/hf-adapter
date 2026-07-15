# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `6/6` cells.

Required Qwen backend: `fla`; verified: `6/6` cells.

Required Qwen full fusion: `true`; verified: `6/6` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 1.302x | 1.482x | 2.166x | 6/6 |
| Decode RWKV/Qwen | 3.987x | 4.042x | 4.046x | 6/6 |
| Model footprint RWKV/Qwen | 0.701x | 0.701x | 0.701x | 6/6 |
| Peak VRAM RWKV/Qwen | 0.705x | 0.751x | 0.766x | 6/6 |
| Runtime working set RWKV/Qwen | 0.722x | 1.159x | 1.330x | 2/6 |
| Active parameters RWKV/Qwen | 0.701x | 0.701x | 0.701x | 6/6 |
| Prefill tok/s per active-B | 1.858x | 2.115x | 3.090x | 6/6 |
| Decode tok/s per active-B | 5.688x | 5.766x | 5.772x | 6/6 |
| Prefill active-param work rate | 0.913x | 1.039x | - | 6/6 |
| Decode active-param work rate | 2.794x | 2.833x | - | 6/6 |

Strict speed cells: `6/6`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 6 | 1.302x / 1.482x | 3.987x / 4.042x | - | - | - | - | - |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
