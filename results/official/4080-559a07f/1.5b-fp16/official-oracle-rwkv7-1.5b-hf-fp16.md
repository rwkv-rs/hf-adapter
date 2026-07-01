# official-oracle-rwkv7-1.5b-hf-fp16

- status: **passed**
- model: /home/wzu/codex-run/rwkv7-reference-v090/rwkv7-1.5b-hf
- dtype: fp16
- device: NVIDIA GeForce RTX 4080
- code: 559a07f2faa458be11fa67a8b36f636ec5b626ba

| case | cosine | max abs | mean abs | argmax |
|---|---:|---:|---:|---|
| b1_t1 | 0.99999973 | 0.04687500 | 0.00863584 | True |
| b1_t17 | 0.99999659 | 0.25000000 | 0.01826981 | True |
| b1_t128 | 0.99999927 | 0.12109375 | 0.00745470 | False |
| b4_t1 | 0.99999562 | 0.15625000 | 0.03121589 | True |
| b4_t17 | 0.99999863 | 0.09375000 | 0.01121897 | False |
| b4_t128 | 0.99999916 | 0.28125000 | 0.00792335 | False |
| cached_teacher | 0.99999925 | 0.10937500 | 0.01002292 | True |
