# official-oracle-rwkv7-1.5b-hf-fp32

- status: **passed**
- model: /home/wzu/codex-run/rwkv7-reference-v090/rwkv7-1.5b-hf
- dtype: fp32
- device: NVIDIA GeForce RTX 4080
- code: d2f8de695826af9ddd0dbf1054f73637b6797953

| case | cosine | max abs | mean abs | argmax |
|---|---:|---:|---:|---|
| b1_t1 | 1.00000000 | 0.00004292 | 0.00001579 | True |
| b1_t17 | 1.00000000 | 0.00009632 | 0.00001445 | True |
| b1_t128 | 1.00000000 | 0.00017452 | 0.00000759 | True |
| b4_t1 | 1.00000000 | 0.00003052 | 0.00000405 | True |
| b4_t17 | 1.00000000 | 0.00027657 | 0.00001324 | True |
| b4_t128 | 1.00000000 | 0.00056362 | 0.00001668 | True |
| cached_teacher | 1.00000000 | 0.00016594 | 0.00001457 | True |
