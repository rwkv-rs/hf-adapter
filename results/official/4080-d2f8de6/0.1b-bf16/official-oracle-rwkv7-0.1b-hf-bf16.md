# official-oracle-rwkv7-0.1b-hf-bf16

- status: **passed**
- model: /home/wzu/codex-run/rwkv7-reference-v090/rwkv7-0.1b-hf
- dtype: bf16
- device: NVIDIA GeForce RTX 4080
- code: d2f8de695826af9ddd0dbf1054f73637b6797953

| case | cosine | max abs | mean abs | argmax |
|---|---:|---:|---:|---|
| b1_t1 | 0.99996663 | 0.37500000 | 0.06372294 | True |
| b1_t17 | 0.99997958 | 0.37500000 | 0.04840058 | False |
| b1_t128 | 0.99996683 | 0.62500000 | 0.05502368 | False |
| b4_t1 | 0.99994609 | 0.62500000 | 0.09534124 | True |
| b4_t17 | 0.99995862 | 0.81250000 | 0.06447601 | False |
| b4_t128 | 0.99996922 | 1.00000000 | 0.05250251 | False |
| cached_teacher | 0.99996624 | 0.75000000 | 0.06371344 | False |
