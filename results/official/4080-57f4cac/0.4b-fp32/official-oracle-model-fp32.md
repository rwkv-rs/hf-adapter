# official-oracle-model-fp32

- status: **passed**
- model: /home/wzu/codex-run/rwkv7-reference-convert-4080/model
- dtype: fp32
- device: NVIDIA GeForce RTX 4080
- code: 57f4cacfff4fe8a1c75a24acae8c1b097a342a34

| case | cosine | max abs | mean abs | argmax |
|---|---:|---:|---:|---|
| b1_t1 | 1.00000000 | 0.00004196 | 0.00000571 | True |
| b1_t17 | 1.00000000 | 0.00006866 | 0.00000456 | True |
| b1_t128 | 1.00000000 | 0.00006104 | 0.00000327 | True |
| b4_t1 | 1.00000000 | 0.00006866 | 0.00001200 | True |
| b4_t17 | 1.00000000 | 0.00009537 | 0.00000462 | True |
| b4_t128 | 1.00000000 | 0.00007248 | 0.00000336 | True |
| cached_teacher | 1.00000000 | 0.00005341 | 0.00000389 | True |
