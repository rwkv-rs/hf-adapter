# official-oracle-rwkv7_01b_hf-fp32

- status: **passed**
- model: /home/data/wangyue/models/rwkv7/reference-v090-6795a75/rwkv7-0.1b-hf
- dtype: fp32
- device: Tesla V100-PCIE-32GB
- code: d2f8de695826af9ddd0dbf1054f73637b6797953

| case | cosine | max abs | mean abs | argmax |
|---|---:|---:|---:|---|
| b1_t1 | 1.00000000 | 0.00002766 | 0.00000501 | True |
| b1_t17 | 1.00000000 | 0.00004005 | 0.00000368 | True |
| b1_t128 | 1.00000000 | 0.00003052 | 0.00000273 | True |
| b4_t1 | 1.00000000 | 0.00004005 | 0.00000673 | True |
| b4_t17 | 1.00000000 | 0.00005436 | 0.00000400 | True |
| b4_t128 | 1.00000000 | 0.00009727 | 0.00000298 | True |
| cached_teacher | 1.00000000 | 0.00004005 | 0.00000275 | True |
