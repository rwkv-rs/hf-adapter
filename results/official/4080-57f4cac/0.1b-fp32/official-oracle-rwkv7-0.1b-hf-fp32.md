# official-oracle-rwkv7-0.1b-hf-fp32

- status: **passed**
- model: /home/wzu/codex-run/rwkv7-reference-v090/rwkv7-0.1b-hf
- dtype: fp32
- device: NVIDIA GeForce RTX 4080
- code: 57f4cacfff4fe8a1c75a24acae8c1b097a342a34

| case | cosine | max abs | mean abs | argmax |
|---|---:|---:|---:|---|
| b1_t1 | 1.00000000 | 0.00006104 | 0.00001248 | True |
| b1_t17 | 1.00000000 | 0.00009537 | 0.00000669 | True |
| b1_t128 | 1.00000000 | 0.00009155 | 0.00000496 | True |
| b4_t1 | 1.00000000 | 0.00005054 | 0.00000936 | True |
| b4_t17 | 1.00000000 | 0.00006866 | 0.00000543 | True |
| b4_t128 | 1.00000000 | 0.00010300 | 0.00000533 | True |
| cached_teacher | 1.00000000 | 0.00006866 | 0.00000577 | True |
