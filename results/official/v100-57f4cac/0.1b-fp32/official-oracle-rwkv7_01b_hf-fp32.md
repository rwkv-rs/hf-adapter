# official-oracle-rwkv7_01b_hf-fp32

- status: **passed**
- model: /home/data/wangyue/models/rwkv7/reference-v090-6795a75/rwkv7-0.1b-hf
- dtype: fp32
- device: Tesla V100-PCIE-32GB
- code: 57f4cacfff4fe8a1c75a24acae8c1b097a342a34

| case | cosine | max abs | mean abs | argmax |
|---|---:|---:|---:|---|
| b1_t1 | 1.00000000 | 0.00004005 | 0.00000628 | True |
| b1_t17 | 1.00000000 | 0.00009155 | 0.00000613 | True |
| b1_t128 | 1.00000000 | 0.00009155 | 0.00000466 | True |
| b4_t1 | 1.00000000 | 0.00006104 | 0.00000966 | True |
| b4_t17 | 1.00000000 | 0.00007248 | 0.00000581 | True |
| b4_t128 | 1.00000000 | 0.00009537 | 0.00000534 | True |
| cached_teacher | 1.00000000 | 0.00007629 | 0.00000576 | True |
