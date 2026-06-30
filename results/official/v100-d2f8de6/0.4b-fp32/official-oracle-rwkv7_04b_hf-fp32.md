# official-oracle-rwkv7_04b_hf-fp32

- status: **passed**
- model: /home/data/wangyue/models/rwkv7/reference-v090-6795a75/rwkv7-0.4b-hf
- dtype: fp32
- device: Tesla V100-PCIE-32GB
- code: d2f8de695826af9ddd0dbf1054f73637b6797953

| case | cosine | max abs | mean abs | argmax |
|---|---:|---:|---:|---|
| b1_t1 | 1.00000000 | 0.00001717 | 0.00000275 | True |
| b1_t17 | 1.00000000 | 0.00003433 | 0.00000215 | True |
| b1_t128 | 1.00000000 | 0.00003433 | 0.00000182 | True |
| b4_t1 | 1.00000000 | 0.00004196 | 0.00001024 | True |
| b4_t17 | 1.00000000 | 0.00005341 | 0.00000258 | True |
| b4_t128 | 1.00000000 | 0.00005341 | 0.00000184 | True |
| cached_teacher | 1.00000000 | 0.00001812 | 0.00000189 | True |
