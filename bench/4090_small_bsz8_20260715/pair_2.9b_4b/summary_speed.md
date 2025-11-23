# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `18/18` cells.

Required Qwen backend: `fla`; verified: `18/18` cells.

Required Qwen full fusion: `false`; verified: `18/18` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 1.301x | 1.462x | 1.928x | 18/18 |
| Decode RWKV/Qwen | 4.214x | 4.349x | 4.998x | 18/18 |
| Model footprint RWKV/Qwen | 0.382x | 0.672x | 0.701x | 18/18 |
| Peak VRAM RWKV/Qwen | 0.453x | 0.945x | 1.028x | 12/18 |
| Runtime working set RWKV/Qwen | 0.851x | 1.939x | 5.021x | 2/18 |
| Active parameters RWKV/Qwen | 0.661x | 0.681x | 0.701x | 18/18 |
| Prefill tok/s per active-B | 1.857x | 2.109x | 2.917x | 18/18 |
| Decode tok/s per active-B | 6.013x | 6.382x | 7.562x | 18/18 |
| Prefill active-param work rate | 0.912x | 1.013x | - | reported 18/18 |
| Decode active-param work rate | 2.859x | 3.015x | - | reported 18/18 |

Strict speed cells: `18/18`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 6 | 1.305x / 1.424x | 4.214x / 4.242x | - | - | - | - | - |
| w4 | 6 | 1.301x / 1.409x | 4.325x / 4.349x | 0.986x | 1.024x | 1.015x | 0.961x | 0.977x |
| w8 | 6 | 1.659x / 1.888x | 4.951x / 4.982x | 1.271x | 1.173x | 1.176x | 0.545x | 0.509x |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
