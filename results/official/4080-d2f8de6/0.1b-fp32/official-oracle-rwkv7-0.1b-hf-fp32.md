# official-oracle-rwkv7-0.1b-hf-fp32

- status: **passed**
- model: /home/wzu/codex-run/rwkv7-reference-v090/rwkv7-0.1b-hf
- dtype: fp32
- device: NVIDIA GeForce RTX 4080
- code: d2f8de695826af9ddd0dbf1054f73637b6797953

| case | cosine | max abs | mean abs | argmax |
|---|---:|---:|---:|---|
| b1_t1 | 1.00000000 | 0.00002575 | 0.00000661 | True |
| b1_t17 | 1.00000000 | 0.00003624 | 0.00000396 | True |
| b1_t128 | 1.00000000 | 0.00003624 | 0.00000271 | True |
| b4_t1 | 1.00000000 | 0.00003624 | 0.00000826 | True |
| b4_t17 | 1.00000000 | 0.00005054 | 0.00000379 | True |
| b4_t128 | 1.00000000 | 0.00010300 | 0.00000292 | True |
| cached_teacher | 1.00000000 | 0.00004101 | 0.00000324 | True |
