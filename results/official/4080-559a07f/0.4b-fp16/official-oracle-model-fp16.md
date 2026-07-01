# official-oracle-model-fp16

- status: **passed**
- model: /home/wzu/codex-run/rwkv7-reference-convert-4080/model
- dtype: fp16
- device: NVIDIA GeForce RTX 4080
- code: 559a07f2faa458be11fa67a8b36f636ec5b626ba

| case | cosine | max abs | mean abs | argmax |
|---|---:|---:|---:|---|
| b1_t1 | 0.99999927 | 0.12500000 | 0.02785204 | True |
| b1_t17 | 0.99999885 | 0.09375000 | 0.00906390 | True |
| b1_t128 | 0.99999917 | 0.14062500 | 0.00583005 | False |
| b4_t1 | 0.99999672 | 0.18750000 | 0.04476084 | False |
| b4_t17 | 0.99999864 | 0.15625000 | 0.00844089 | True |
| b4_t128 | 0.99999931 | 0.12500000 | 0.00553372 | False |
| cached_teacher | 0.99999903 | 0.09375000 | 0.00709452 | True |
