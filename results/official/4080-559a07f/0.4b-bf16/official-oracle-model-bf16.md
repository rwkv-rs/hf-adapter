# official-oracle-model-bf16

- status: **passed**
- model: /home/wzu/codex-run/rwkv7-reference-convert-4080/model
- dtype: bf16
- device: NVIDIA GeForce RTX 4080
- code: 559a07f2faa458be11fa67a8b36f636ec5b626ba

| case | cosine | max abs | mean abs | argmax |
|---|---:|---:|---:|---|
| b1_t1 | 0.99996471 | 0.56250000 | 0.08414359 | True |
| b1_t17 | 0.99992531 | 1.00000000 | 0.06962442 | False |
| b1_t128 | 0.99996243 | 0.81250000 | 0.04295018 | False |
| b4_t1 | 0.99944647 | 1.37500000 | 0.34172153 | True |
| b4_t17 | 0.99989348 | 1.50000000 | 0.06774695 | False |
| b4_t128 | 0.99995416 | 0.93750000 | 0.04596649 | False |
| cached_teacher | 0.99995290 | 0.56250000 | 0.05048651 | True |
