# official-oracle-rwkv7_15b_hf-fp32

- status: **passed**
- model: /home/data/wangyue/models/rwkv7/reference-v090-6795a75/rwkv7-1.5b-hf
- dtype: fp32
- device: Tesla V100-PCIE-32GB
- code: 559a07f2faa458be11fa67a8b36f636ec5b626ba

| case | cosine | max abs | mean abs | argmax |
|---|---:|---:|---:|---|
| b1_t1 | 1.00000000 | 0.00008392 | 0.00000840 | True |
| b1_t17 | 1.00000000 | 0.00011444 | 0.00001674 | True |
| b1_t128 | 1.00000000 | 0.00018978 | 0.00000893 | True |
| b4_t1 | 1.00000000 | 0.00009537 | 0.00001046 | True |
| b4_t17 | 1.00000000 | 0.00022507 | 0.00001540 | True |
| b4_t128 | 1.00000000 | 0.00052166 | 0.00001836 | True |
| cached_teacher | 1.00000000 | 0.00012970 | 0.00001447 | True |
