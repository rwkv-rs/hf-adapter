# official-oracle-rwkv7_04b_hf-fp16

- status: **passed**
- model: /home/data/wangyue/models/rwkv7/reference-v090-6795a75/rwkv7-0.4b-hf
- dtype: fp16
- device: Tesla V100-PCIE-32GB
- code: 57f4cacfff4fe8a1c75a24acae8c1b097a342a34

| case | cosine | max abs | mean abs | argmax |
|---|---:|---:|---:|---|
| b1_t1 | 0.99999862 | 0.12500000 | 0.04380433 | True |
| b1_t17 | 0.99999885 | 0.09375000 | 0.00867449 | True |
| b1_t128 | 0.99999938 | 0.07812500 | 0.00562283 | False |
| b4_t1 | 0.99999286 | 0.18750000 | 0.03505491 | True |
| b4_t17 | 0.99999920 | 0.08593750 | 0.00703631 | True |
| b4_t128 | 0.99999912 | 0.20312500 | 0.00594310 | False |
| cached_teacher | 0.99999921 | 0.08593750 | 0.00669436 | True |
