# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `18/18` cells.

Required Qwen backend: `fla`; verified: `18/18` cells.

Required Qwen full fusion: `true`; verified: `18/18` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 4.181x | 5.492x | 10.429x | 18/18 |
| Decode RWKV/Qwen | 10.738x | 11.003x | 11.214x | 18/18 |
| Model footprint RWKV/Qwen | 0.298x | 0.555x | 0.599x | 18/18 |
| Peak VRAM RWKV/Qwen | 0.400x | 0.613x | 0.668x | 18/18 |
| Runtime working set RWKV/Qwen | 1.410x | 1.637x | 1.867x | 0/18 |
| Active parameters RWKV/Qwen | 0.510x | 0.599x | 0.599x | 18/18 |
| Prefill tok/s per active-B | 6.978x | 9.979x | 20.452x | 18/18 |
| Decode tok/s per active-B | 17.923x | 18.717x | 21.931x | 18/18 |
| Prefill active-param work rate | 2.158x | 3.277x | - | reported 18/18 |
| Decode active-param work rate | 5.596x | 6.470x | - | reported 18/18 |

Strict speed cells: `18/18`.

## Precision families

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 6 | 4.181x / 5.470x | 10.861x / 10.901x | - | - | - | - | - |
| w4 | 6 | 4.224x / 6.640x | 10.738x / 11.191x | 0.881x | 0.986x | 1.018x | 0.891x | 0.909x |
| w8 | 6 | 4.233x / 5.492x | 10.973x / 11.008x | 0.862x | 1.008x | 1.008x | 0.926x | 0.939x |

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
