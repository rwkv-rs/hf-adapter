# official-oracle-rwkv7-1.5b-hf-fp16

- status: **passed**
- model: /home/wzu/codex-run/rwkv7-reference-v090/rwkv7-1.5b-hf
- dtype: fp16
- device: NVIDIA GeForce RTX 4080
- code: d2f8de695826af9ddd0dbf1054f73637b6797953

| case | cosine | max abs | mean abs | argmax |
|---|---:|---:|---:|---|
| b1_t1 | 0.99999977 | 0.03906250 | 0.00566946 | True |
| b1_t17 | 0.99999741 | 0.18750000 | 0.01700231 | True |
| b1_t128 | 0.99999921 | 0.12500000 | 0.00723628 | True |
| b4_t1 | 0.99999914 | 0.09375000 | 0.01303034 | True |
| b4_t17 | 0.99999865 | 0.10937500 | 0.01082849 | True |
| b4_t128 | 0.99999923 | 0.14062500 | 0.00761145 | False |
| cached_teacher | 0.99999858 | 0.15625000 | 0.01261816 | True |
