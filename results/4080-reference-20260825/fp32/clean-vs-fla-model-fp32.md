# clean-vs-fla-model-fp32

- status: **failed**
- model: /home/wzu/codex-run/rwkv7-reference-convert-4080/model
- dtype: fp32
- device: NVIDIA GeForce RTX 4080
- code: f657c67
- FLA: 80e494f6c588e091fc8316b612870df29375c5b8

| case | cosine | max abs | mean abs | argmax |
|---|---:|---:|---:|---|
| b1_t1 | 1.00000000 | 0.00003052 | 0.00001015 | True |
| b1_t17 | 1.00000000 | 0.00006294 | 0.00000324 | True |
| b1_t128 | 0.99999978 | 0.03845215 | 0.00361549 | True |
| b4_t1 | 1.00000000 | 0.00003815 | 0.00000842 | True |
| b4_t17 | 1.00000000 | 0.00004959 | 0.00000386 | True |
| b4_t128 | 0.99999976 | 0.08531189 | 0.00328223 | False |
| cached_teacher_clean_vs_fla | 1.00000000 | 0.00003529 | 0.00000235 | True |
| clean_cached_vs_clean_full | 1.00000000 | 0.00006485 | 0.00000381 | True |
| fla_cached_vs_fla_full | 1.00000000 | 0.00003624 | 0.00000337 | True |
