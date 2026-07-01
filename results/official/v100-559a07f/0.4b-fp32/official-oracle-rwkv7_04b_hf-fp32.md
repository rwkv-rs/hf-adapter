# official-oracle-rwkv7_04b_hf-fp32

- status: **passed**
- model: /home/data/wangyue/models/rwkv7/reference-v090-6795a75/rwkv7-0.4b-hf
- dtype: fp32
- device: Tesla V100-PCIE-32GB
- code: 559a07f2faa458be11fa67a8b36f636ec5b626ba

| case | cosine | max abs | mean abs | argmax |
|---|---:|---:|---:|---|
| b1_t1 | 1.00000000 | 0.00004959 | 0.00000600 | True |
| b1_t17 | 1.00000000 | 0.00005913 | 0.00000454 | True |
| b1_t128 | 1.00000000 | 0.00006485 | 0.00000373 | True |
| b4_t1 | 1.00000000 | 0.00008011 | 0.00001859 | True |
| b4_t17 | 1.00000000 | 0.00016403 | 0.00000611 | True |
| b4_t128 | 1.00000000 | 0.00006866 | 0.00000383 | True |
| cached_teacher | 1.00000000 | 0.00004387 | 0.00000399 | True |
