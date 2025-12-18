# RTX 4080 rwkv-0.4b__qwen3.5-0.8b B8 acceptance

Status: **pass**

| Axis | Measured range |
|---|---:|
| Dense prefill / full-FLA Qwen | 1.3762x - 1.6539x |
| Dense decode / full-FLA Qwen | 3.5508x - 3.5874x |
| Active-work decode ratio | 5.9268x - 5.9878x |
| a8w8 paired prefill/decode/total | 0.9986x / 1.0080x / 1.0072x minimum |
| torchao_w4 paired prefill/decode/total | 1.0012x / 1.0489x / 1.0331x minimum |
