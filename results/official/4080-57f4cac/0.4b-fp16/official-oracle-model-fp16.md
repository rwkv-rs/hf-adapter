# official-oracle-model-fp16

- status: **passed**
- model: /home/wzu/codex-run/rwkv7-reference-convert-4080/model
- dtype: fp16
- device: NVIDIA GeForce RTX 4080
- code: 57f4cacfff4fe8a1c75a24acae8c1b097a342a34

| case | cosine | max abs | mean abs | argmax |
|---|---:|---:|---:|---|
| b1_t1 | 0.99999927 | 0.12500000 | 0.02785204 | True |
| b1_t17 | 0.99999885 | 0.09375000 | 0.00906390 | True |
| b1_t128 | 0.99999917 | 0.14062500 | 0.00583005 | False |
| b4_t1 | 0.99999239 | 0.18750000 | 0.04330529 | True |
| b4_t17 | 0.99999897 | 0.13281250 | 0.00794717 | True |
| b4_t128 | 0.99999924 | 0.18750000 | 0.00588960 | False |
| cached_teacher | 0.99999903 | 0.09375000 | 0.00709452 | True |
