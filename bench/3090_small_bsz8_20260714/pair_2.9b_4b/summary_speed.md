# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `18/18` cells.

Required Qwen backend: `fla`; verified: `18/18` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 1.337x | 1.430x | 2.394x | 18/18 |
| Decode RWKV/Qwen | 2.921x | 3.029x | 3.511x | 18/18 |
| Model footprint RWKV/Qwen | 0.402x | 0.671x | 0.701x | 18/18 |
| Peak VRAM RWKV/Qwen | 0.471x | 0.738x | 0.865x | 18/18 |

Strict speed cells: `18/18`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 6 | 1.356x / 1.364x | 2.921x / 2.957x | - | - | - | - | - |
| w4 | 6 | 1.337x / 1.351x | 2.998x / 3.029x | 0.986x | 1.020x | 1.007x | 0.961x | 0.976x |
| w8 | 6 | 1.947x / 2.041x | 3.428x / 3.486x | 1.411x | 1.171x | 1.181x | 0.573x | 0.760x |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
