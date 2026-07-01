# official-oracle-rwkv7-0.1b-hf-bf16

- status: **passed**
- model: /home/wzu/codex-run/rwkv7-reference-v090/rwkv7-0.1b-hf
- dtype: bf16
- device: NVIDIA GeForce RTX 4080
- code: 559a07f2faa458be11fa67a8b36f636ec5b626ba

| case | cosine | max abs | mean abs | argmax |
|---|---:|---:|---:|---|
| b1_t1 | 0.99996605 | 0.38281250 | 0.09248447 | True |
| b1_t17 | 0.99995896 | 0.50000000 | 0.06890647 | True |
| b1_t128 | 0.99996836 | 0.50000000 | 0.05362102 | False |
| b4_t1 | 0.99993373 | 0.56250000 | 0.10569821 | True |
| b4_t17 | 0.99996291 | 0.50000000 | 0.06206237 | False |
| b4_t128 | 0.99997007 | 0.62500000 | 0.05226152 | False |
| cached_teacher | 0.99997848 | 0.56250000 | 0.04722009 | False |
