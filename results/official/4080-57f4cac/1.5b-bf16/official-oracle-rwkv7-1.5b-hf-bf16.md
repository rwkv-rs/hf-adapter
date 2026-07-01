# official-oracle-rwkv7-1.5b-hf-bf16

- status: **passed**
- model: /home/wzu/codex-run/rwkv7-reference-v090/rwkv7-1.5b-hf
- dtype: bf16
- device: NVIDIA GeForce RTX 4080
- code: 57f4cacfff4fe8a1c75a24acae8c1b097a342a34

| case | cosine | max abs | mean abs | argmax |
|---|---:|---:|---:|---|
| b1_t1 | 0.99997638 | 0.75000000 | 0.19678873 | True |
| b1_t17 | 0.99986190 | 1.43750000 | 0.10962194 | True |
| b1_t128 | 0.99995055 | 1.50000000 | 0.05941711 | False |
| b4_t1 | 0.99985904 | 0.87500000 | 0.17209765 | True |
| b4_t17 | 0.99987786 | 1.37500000 | 0.10280864 | False |
| b4_t128 | 0.99994926 | 1.00000000 | 0.06303837 | False |
| cached_teacher | 0.99995351 | 1.37500000 | 0.07132725 | True |
