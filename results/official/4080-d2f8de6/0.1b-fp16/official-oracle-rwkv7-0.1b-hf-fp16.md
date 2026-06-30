# official-oracle-rwkv7-0.1b-hf-fp16

- status: **passed**
- model: /home/wzu/codex-run/rwkv7-reference-v090/rwkv7-0.1b-hf
- dtype: fp16
- device: NVIDIA GeForce RTX 4080
- code: d2f8de695826af9ddd0dbf1054f73637b6797953

| case | cosine | max abs | mean abs | argmax |
|---|---:|---:|---:|---|
| b1_t1 | 0.99999928 | 0.05468750 | 0.00919824 | True |
| b1_t17 | 0.99999957 | 0.05468750 | 0.00724966 | True |
| b1_t128 | 0.99999954 | 0.09375000 | 0.00609501 | True |
| b4_t1 | 0.99999900 | 0.10156250 | 0.01269905 | True |
| b4_t17 | 0.99999946 | 0.06250000 | 0.00705196 | True |
| b4_t128 | 0.99999950 | 0.07812500 | 0.00675060 | False |
| cached_teacher | 0.99999958 | 0.06250000 | 0.00715009 | False |
