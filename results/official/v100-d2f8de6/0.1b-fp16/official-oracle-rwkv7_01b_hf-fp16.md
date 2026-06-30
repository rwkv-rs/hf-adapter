# official-oracle-rwkv7_01b_hf-fp16

- status: **passed**
- model: /home/data/wangyue/models/rwkv7/reference-v090-6795a75/rwkv7-0.1b-hf
- dtype: fp16
- device: Tesla V100-PCIE-32GB
- code: d2f8de695826af9ddd0dbf1054f73637b6797953

| case | cosine | max abs | mean abs | argmax |
|---|---:|---:|---:|---|
| b1_t1 | 0.99999923 | 0.07031250 | 0.01259807 | True |
| b1_t17 | 0.99999922 | 0.06250000 | 0.01002314 | True |
| b1_t128 | 0.99999948 | 0.06250000 | 0.00686052 | False |
| b4_t1 | 0.99999824 | 0.06250000 | 0.01780465 | True |
| b4_t17 | 0.99999956 | 0.06250000 | 0.00632995 | True |
| b4_t128 | 0.99999950 | 0.09375000 | 0.00676042 | False |
| cached_teacher | 0.99999938 | 0.09375000 | 0.00860169 | True |
