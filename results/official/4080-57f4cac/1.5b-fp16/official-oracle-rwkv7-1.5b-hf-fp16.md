# official-oracle-rwkv7-1.5b-hf-fp16

- status: **passed**
- model: /home/wzu/codex-run/rwkv7-reference-v090/rwkv7-1.5b-hf
- dtype: fp16
- device: NVIDIA GeForce RTX 4080
- code: 57f4cacfff4fe8a1c75a24acae8c1b097a342a34

| case | cosine | max abs | mean abs | argmax |
|---|---:|---:|---:|---|
| b1_t1 | 0.99999973 | 0.04687500 | 0.00863584 | True |
| b1_t17 | 0.99999659 | 0.25000000 | 0.01826981 | True |
| b1_t128 | 0.99999927 | 0.12109375 | 0.00745470 | False |
| b4_t1 | 0.99999801 | 0.12500000 | 0.02317216 | True |
| b4_t17 | 0.99999884 | 0.12500000 | 0.01052263 | True |
| b4_t128 | 0.99999922 | 0.15625000 | 0.00799929 | False |
| cached_teacher | 0.99999925 | 0.10937500 | 0.01002292 | True |
