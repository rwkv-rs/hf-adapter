# official-oracle-rwkv7-0.1b-hf-fp32

- status: **passed**
- model: /home/wzu/codex-run/rwkv7-reference-v090/rwkv7-0.1b-hf
- dtype: fp32
- device: NVIDIA GeForce RTX 4080
- code: 559a07f2faa458be11fa67a8b36f636ec5b626ba

| case | cosine | max abs | mean abs | argmax |
|---|---:|---:|---:|---|
| b1_t1 | 1.00000000 | 0.00006104 | 0.00001248 | True |
| b1_t17 | 1.00000000 | 0.00009537 | 0.00000669 | True |
| b1_t128 | 1.00000000 | 0.00009155 | 0.00000496 | True |
| b4_t1 | 1.00000000 | 0.00006294 | 0.00001240 | True |
| b4_t17 | 1.00000000 | 0.00006390 | 0.00000543 | True |
| b4_t128 | 1.00000000 | 0.00009918 | 0.00000532 | True |
| cached_teacher | 1.00000000 | 0.00006866 | 0.00000577 | True |
