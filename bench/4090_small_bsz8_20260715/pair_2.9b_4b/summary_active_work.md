# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `6/6` cells.

Required Qwen backend: `fla`; verified: `6/6` cells.

Required Qwen full fusion: `true`; verified: `6/6` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 1.305x | 1.424x | 1.464x | 6/6 |
| Decode RWKV/Qwen | 4.214x | 4.242x | 4.257x | 6/6 |
| Model footprint RWKV/Qwen | 0.701x | 0.701x | 0.701x | 6/6 |
| Peak VRAM RWKV/Qwen | 0.941x | 1.020x | 1.028x | 2/6 |
| Runtime working set RWKV/Qwen | 1.858x | 3.042x | 5.021x | 0/6 |
| Active parameters RWKV/Qwen | 0.701x | 0.701x | 0.701x | 6/6 |
| Prefill tok/s per active-B | 1.862x | 2.031x | 2.089x | 6/6 |
| Decode tok/s per active-B | 6.013x | 6.053x | 6.073x | 6/6 |
| Prefill active-param work rate | 0.915x | 0.998x | - | 6/6 |
| Decode active-param work rate | 2.954x | 2.973x | - | 6/6 |

Strict speed cells: `6/6`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 6 | 1.305x / 1.424x | 4.214x / 4.242x | - | - | - | - | - |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
