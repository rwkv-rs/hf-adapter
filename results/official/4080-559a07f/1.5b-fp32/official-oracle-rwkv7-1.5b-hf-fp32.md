# official-oracle-rwkv7-1.5b-hf-fp32

- status: **passed**
- model: /home/wzu/codex-run/rwkv7-reference-v090/rwkv7-1.5b-hf
- dtype: fp32
- device: NVIDIA GeForce RTX 4080
- code: 559a07f2faa458be11fa67a8b36f636ec5b626ba

| case | cosine | max abs | mean abs | argmax |
|---|---:|---:|---:|---|
| b1_t1 | 1.00000000 | 0.00012589 | 0.00001680 | True |
| b1_t17 | 1.00000000 | 0.00012970 | 0.00001786 | True |
| b1_t128 | 1.00000000 | 0.00032806 | 0.00001032 | True |
| b4_t1 | 1.00000000 | 0.00011063 | 0.00001612 | True |
| b4_t17 | 1.00000000 | 0.00024414 | 0.00001657 | True |
| b4_t128 | 1.00000000 | 0.00056314 | 0.00001782 | True |
| cached_teacher | 1.00000000 | 0.00014877 | 0.00001564 | True |
