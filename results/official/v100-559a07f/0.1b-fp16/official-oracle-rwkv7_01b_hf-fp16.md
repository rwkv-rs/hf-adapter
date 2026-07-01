# official-oracle-rwkv7_01b_hf-fp16

- status: **passed**
- model: /home/data/wangyue/models/rwkv7/reference-v090-6795a75/rwkv7-0.1b-hf
- dtype: fp16
- device: Tesla V100-PCIE-32GB
- code: 559a07f2faa458be11fa67a8b36f636ec5b626ba

| case | cosine | max abs | mean abs | argmax |
|---|---:|---:|---:|---|
| b1_t1 | 0.99999916 | 0.05078125 | 0.00895663 | True |
| b1_t17 | 0.99999944 | 0.07031250 | 0.00862856 | True |
| b1_t128 | 0.99999952 | 0.07812500 | 0.00670608 | True |
| b4_t1 | 0.99999901 | 0.07031250 | 0.01411079 | True |
| b4_t17 | 0.99999930 | 0.08593750 | 0.00804486 | True |
| b4_t128 | 0.99999943 | 0.12500000 | 0.00722398 | True |
| cached_teacher | 0.99999929 | 0.10937500 | 0.00802291 | False |
