# RTX 4080 rwkv-2.9b__qwen3.5-4b B8 acceptance

Status: **pass**

| Axis | Measured range |
|---|---:|
| Dense prefill / full-FLA Qwen | 1.2439x - 1.4790x |
| Dense decode / full-FLA Qwen | 1.5372x - 1.7756x |
| Active-work decode ratio | 2.1933x - 2.5334x |
| a8w8 paired prefill/decode/total | 0.9959x / 1.0045x / 1.0031x minimum |
| torchao_w4 paired prefill/decode/total | 0.9994x / 1.0281x / 1.0160x minimum |
