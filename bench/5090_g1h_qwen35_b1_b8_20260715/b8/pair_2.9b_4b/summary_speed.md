# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `18/18` cells.

Required Qwen backend: `fla`; verified: `18/18` cells.

Required Qwen full fusion: `true`; verified: `18/18` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 1.291x | 1.749x | 3.437x | 18/18 |
| Decode RWKV/Qwen | 3.987x | 4.089x | 4.902x | 18/18 |
| Model footprint RWKV/Qwen | 0.402x | 0.674x | 0.701x | 18/18 |
| Peak VRAM RWKV/Qwen | 0.471x | 0.708x | 0.766x | 18/18 |
| Runtime working set RWKV/Qwen | 0.722x | 1.195x | 1.426x | 6/18 |
| Active parameters RWKV/Qwen | 0.701x | 0.701x | 0.701x | 18/18 |
| Prefill tok/s per active-B | 1.842x | 2.495x | 4.904x | 18/18 |
| Decode tok/s per active-B | 5.688x | 5.834x | 6.994x | 18/18 |
| Prefill active-param work rate | 0.905x | 1.226x | - | reported 18/18 |
| Decode active-param work rate | 2.794x | 2.866x | - | reported 18/18 |

Strict speed cells: `18/18`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 6 | 1.302x / 1.482x | 3.987x / 4.042x | - | - | - | - | - |
| w4 | 6 | 1.291x / 1.469x | 4.037x / 4.089x | 0.991x | 1.011x | 1.005x | 0.961x | 0.969x |
| w8 | 6 | 1.860x / 2.166x | 4.827x / 4.895x | 1.428x | 1.210x | 1.213x | 0.573x | 0.703x |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
