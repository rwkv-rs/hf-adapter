# official-oracle-rwkv7-1.5b-hf-bf16

- status: **passed**
- model: /home/wzu/codex-run/rwkv7-reference-v090/rwkv7-1.5b-hf
- dtype: bf16
- device: NVIDIA GeForce RTX 4080
- code: d2f8de695826af9ddd0dbf1054f73637b6797953

| case | cosine | max abs | mean abs | argmax |
|---|---:|---:|---:|---|
| b1_t1 | 0.99998214 | 0.37500000 | 0.07263972 | True |
| b1_t17 | 0.99994042 | 0.75000000 | 0.08514519 | True |
| b1_t128 | 0.99994609 | 1.00000000 | 0.06149691 | False |
| b4_t1 | 0.99991134 | 0.87500000 | 0.13011520 | True |
| b4_t17 | 0.99987023 | 1.75000000 | 0.09318408 | False |
| b4_t128 | 0.99995205 | 0.87500000 | 0.05930820 | False |
| cached_teacher | 0.99993282 | 1.75000000 | 0.07784492 | False |
