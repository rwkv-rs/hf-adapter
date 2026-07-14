# RWKV-7 vs Qwen3.5 HF speed matrix

Overall: PASS

Coverage: `72/72` cells.

Required Qwen backend: `fla`; verified: `72/72` cells.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | 1.110x | 1.473x | 4.298x | 72/72 |
| Decode RWKV/Qwen | 1.466x | 2.721x | 6.883x | 72/72 |
| Model footprint RWKV/Qwen | 0.729x | 0.812x | 0.857x | 72/72 |
| Peak VRAM RWKV/Qwen | 0.649x | 0.823x | 0.967x | 72/72 |

Strict speed cells: `72/72`.

## Red cells

None.

Missing candidate rows: `0`.
Missing reference rows: `0`.
