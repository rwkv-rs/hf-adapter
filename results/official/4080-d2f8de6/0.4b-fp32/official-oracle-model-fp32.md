# official-oracle-model-fp32

- status: **passed**
- model: /home/wzu/codex-run/rwkv7-reference-convert-4080/model
- dtype: fp32
- device: NVIDIA GeForce RTX 4080
- code: d2f8de695826af9ddd0dbf1054f73637b6797953

| case | cosine | max abs | mean abs | argmax |
|---|---:|---:|---:|---|
| b1_t1 | 1.00000000 | 0.00002098 | 0.00000473 | True |
| b1_t17 | 1.00000000 | 0.00003433 | 0.00000268 | True |
| b1_t128 | 1.00000000 | 0.00003052 | 0.00000149 | True |
| b4_t1 | 1.00000000 | 0.00002098 | 0.00000437 | True |
| b4_t17 | 1.00000000 | 0.00002861 | 0.00000233 | True |
| b4_t128 | 1.00000000 | 0.00003815 | 0.00000170 | True |
| cached_teacher | 1.00000000 | 0.00002289 | 0.00000208 | True |
