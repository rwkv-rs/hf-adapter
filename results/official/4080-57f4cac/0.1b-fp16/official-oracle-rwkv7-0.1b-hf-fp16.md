# official-oracle-rwkv7-0.1b-hf-fp16

- status: **passed**
- model: /home/wzu/codex-run/rwkv7-reference-v090/rwkv7-0.1b-hf
- dtype: fp16
- device: NVIDIA GeForce RTX 4080
- code: 57f4cacfff4fe8a1c75a24acae8c1b097a342a34

| case | cosine | max abs | mean abs | argmax |
|---|---:|---:|---:|---|
| b1_t1 | 0.99999931 | 0.05468750 | 0.00816347 | True |
| b1_t17 | 0.99999945 | 0.04687500 | 0.00783115 | True |
| b1_t128 | 0.99999942 | 0.12500000 | 0.00682066 | False |
| b4_t1 | 0.99999915 | 0.08593750 | 0.01173127 | True |
| b4_t17 | 0.99999944 | 0.06250000 | 0.00729435 | True |
| b4_t128 | 0.99999950 | 0.07812500 | 0.00682914 | False |
| cached_teacher | 0.99999945 | 0.07812500 | 0.00796019 | False |
