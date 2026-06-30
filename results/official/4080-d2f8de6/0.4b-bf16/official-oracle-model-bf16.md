# official-oracle-model-bf16

- status: **passed**
- model: /home/wzu/codex-run/rwkv7-reference-convert-4080/model
- dtype: bf16
- device: NVIDIA GeForce RTX 4080
- code: d2f8de695826af9ddd0dbf1054f73637b6797953

| case | cosine | max abs | mean abs | argmax |
|---|---:|---:|---:|---|
| b1_t1 | 0.99996409 | 0.56250000 | 0.10735163 | True |
| b1_t17 | 0.99990028 | 1.12500000 | 0.07351333 | False |
| b1_t128 | 0.99995936 | 0.59375000 | 0.04527956 | False |
| b4_t1 | 0.99933069 | 1.12500000 | 0.39696050 | True |
| b4_t17 | 0.99992461 | 0.94921875 | 0.06604693 | False |
| b4_t128 | 0.99995170 | 1.25000000 | 0.04690526 | False |
| cached_teacher | 0.99993245 | 0.68750000 | 0.05590728 | True |
