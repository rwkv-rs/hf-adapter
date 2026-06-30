# official-oracle-rwkv7_15b_hf-fp32

- status: **passed**
- model: /home/data/wangyue/models/rwkv7/reference-v090-6795a75/rwkv7-1.5b-hf
- dtype: fp32
- device: Tesla V100-PCIE-32GB
- code: d2f8de695826af9ddd0dbf1054f73637b6797953

| case | cosine | max abs | mean abs | argmax |
|---|---:|---:|---:|---|
| b1_t1 | 1.00000000 | 0.00002098 | 0.00000401 | True |
| b1_t17 | 1.00000000 | 0.00010395 | 0.00001392 | True |
| b1_t128 | 1.00000000 | 0.00023174 | 0.00000757 | True |
| b4_t1 | 1.00000000 | 0.00004578 | 0.00001234 | True |
| b4_t17 | 1.00000000 | 0.00027084 | 0.00001301 | True |
| b4_t128 | 1.00000000 | 0.00054359 | 0.00001676 | True |
| cached_teacher | 1.00000000 | 0.00015259 | 0.00001537 | True |
