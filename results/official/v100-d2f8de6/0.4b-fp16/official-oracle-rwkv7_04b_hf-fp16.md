# official-oracle-rwkv7_04b_hf-fp16

- status: **passed**
- model: /home/data/wangyue/models/rwkv7/reference-v090-6795a75/rwkv7-0.4b-hf
- dtype: fp16
- device: Tesla V100-PCIE-32GB
- code: d2f8de695826af9ddd0dbf1054f73637b6797953

| case | cosine | max abs | mean abs | argmax |
|---|---:|---:|---:|---|
| b1_t1 | 0.99999916 | 0.07812500 | 0.01834951 | True |
| b1_t17 | 0.99999827 | 0.12500000 | 0.00985110 | True |
| b1_t128 | 0.99999943 | 0.07812500 | 0.00517234 | False |
| b4_t1 | 0.99999630 | 0.14062500 | 0.02409020 | True |
| b4_t17 | 0.99999881 | 0.12500000 | 0.00773892 | True |
| b4_t128 | 0.99999918 | 0.21875000 | 0.00580952 | False |
| cached_teacher | 0.99999936 | 0.06250000 | 0.00613818 | True |
