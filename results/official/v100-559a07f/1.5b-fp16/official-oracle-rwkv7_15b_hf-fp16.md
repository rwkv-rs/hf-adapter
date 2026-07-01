# official-oracle-rwkv7_15b_hf-fp16

- status: **passed**
- model: /home/data/wangyue/models/rwkv7/reference-v090-6795a75/rwkv7-1.5b-hf
- dtype: fp16
- device: Tesla V100-PCIE-32GB
- code: 559a07f2faa458be11fa67a8b36f636ec5b626ba

| case | cosine | max abs | mean abs | argmax |
|---|---:|---:|---:|---|
| b1_t1 | 0.99999963 | 0.07812500 | 0.02992823 | True |
| b1_t17 | 0.99999866 | 0.10937500 | 0.01263985 | True |
| b1_t128 | 0.99999901 | 0.21875000 | 0.00795685 | True |
| b4_t1 | 0.99999767 | 0.12500000 | 0.02223260 | False |
| b4_t17 | 0.99999816 | 0.12500000 | 0.01278838 | False |
| b4_t128 | 0.99999887 | 0.25000000 | 0.00889715 | False |
| cached_teacher | 0.99999927 | 0.12500000 | 0.00935089 | True |
