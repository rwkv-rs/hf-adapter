# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `18/18` cells.

Required Qwen backend: `fla`; verified: `18/18` cells.

Required Qwen full fusion: `true`; verified: `18/18` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 1.293x | 1.665x | 3.325x | 18/18 |
| Decode RWKV/Qwen | 3.783x | 3.870x | 4.717x | 18/18 |
| Model footprint RWKV/Qwen | 0.402x | 0.671x | 0.701x | 18/18 |
| Peak VRAM RWKV/Qwen | 0.471x | 0.708x | 0.766x | 18/18 |
| Runtime working set RWKV/Qwen | 0.722x | 1.195x | 1.426x | 6/18 |
| Active parameters RWKV/Qwen | 0.661x | 0.701x | 0.701x | 18/18 |
| Prefill tok/s per active-B | 1.855x | 2.446x | 4.744x | 18/18 |
| Decode tok/s per active-B | 5.398x | 5.845x | 6.731x | 18/18 |
| Prefill active-param work rate | 0.855x | 1.134x | - | reported 18/18 |
| Decode active-param work rate | 2.533x | 2.674x | - | reported 18/18 |

Strict speed cells: `18/18`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 6 | 1.300x / 1.481x | 3.783x / 3.815x | - | - | - | - | - |
| w4 | 6 | 1.293x / 1.470x | 3.832x / 3.863x | - | - | - | 0.957x | 0.966x |
| w8 | 6 | 1.857x / 2.163x | 4.597x / 4.627x | - | - | - | 0.573x | 0.703x |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
