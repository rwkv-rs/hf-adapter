# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `18/18` cells.

Required Qwen backend: `fla`; verified: `18/18` cells.

Required Qwen full fusion: `true`; verified: `18/18` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 4.098x | 5.398x | 9.750x | 18/18 |
| Decode RWKV/Qwen | 10.582x | 10.873x | 11.159x | 18/18 |
| Model footprint RWKV/Qwen | 0.298x | 0.555x | 0.599x | 18/18 |
| Peak VRAM RWKV/Qwen | 0.400x | 0.613x | 0.668x | 18/18 |
| Runtime working set RWKV/Qwen | 1.410x | 1.637x | 1.867x | 0/18 |
| Active parameters RWKV/Qwen | 0.510x | 0.599x | 0.599x | 18/18 |
| Prefill tok/s per active-B | 6.865x | 9.810x | 19.120x | 18/18 |
| Decode tok/s per active-B | 17.662x | 18.498x | 21.713x | 18/18 |
| Prefill active-param work rate | 2.090x | 3.220x | - | reported 18/18 |
| Decode active-param work rate | 5.542x | 6.436x | - | reported 18/18 |

Strict speed cells: `18/18`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 6 | 4.113x / 5.374x | 10.734x / 10.763x | - | - | - | - | - |
| w4 | 6 | 4.098x / 6.510x | 10.582x / 11.073x | 0.996x | 0.984x | 1.016x | 0.891x | 0.909x |
| w8 | 6 | 4.133x / 5.398x | 10.868x / 10.896x | 0.999x | 1.010x | 1.010x | 0.926x | 0.939x |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
