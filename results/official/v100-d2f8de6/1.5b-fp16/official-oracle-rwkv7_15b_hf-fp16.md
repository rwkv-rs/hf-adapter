# official-oracle-rwkv7_15b_hf-fp16

- status: **passed**
- model: /home/data/wangyue/models/rwkv7/reference-v090-6795a75/rwkv7-1.5b-hf
- dtype: fp16
- device: Tesla V100-PCIE-32GB
- code: d2f8de695826af9ddd0dbf1054f73637b6797953

| case | cosine | max abs | mean abs | argmax |
|---|---:|---:|---:|---|
| b1_t1 | 0.99999952 | 0.09375000 | 0.03676021 | True |
| b1_t17 | 0.99999660 | 0.18750000 | 0.01418135 | True |
| b1_t128 | 0.99999942 | 0.16406250 | 0.00608382 | True |
| b4_t1 | 0.99999931 | 0.10937500 | 0.01451214 | False |
| b4_t17 | 0.99999683 | 0.22656250 | 0.01507093 | True |
| b4_t128 | 0.99999886 | 0.23437500 | 0.00912295 | False |
| cached_teacher | 0.99999936 | 0.12500000 | 0.00958008 | True |
