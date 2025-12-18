# RTX 4080 RWKV-7 / Qwen3.5 acceptance

Status: **pass**

| Axis | Measured range |
|---|---:|
| Dense prefill / full-FLA Qwen | 1.0242x - 1.1227x |
| Dense decode / full-FLA Qwen | 1.4353x - 1.4729x |
| Active-work decode ratio | 1.7683x - 1.8147x |
| a8w8 paired prefill/decode/total | 0.9988x / 1.0076x / 1.0051x minimum |
| torchao_w4 paired prefill/decode/total | 0.9996x / 1.0458x / 1.0261x minimum |
