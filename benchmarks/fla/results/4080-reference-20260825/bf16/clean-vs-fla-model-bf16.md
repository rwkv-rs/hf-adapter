# clean-vs-fla-model-bf16

- status: **failed**
- model: /home/wzu/codex-run/rwkv7-reference-convert-4080/model
- dtype: bf16
- device: NVIDIA GeForce RTX 4080
- code: f657c67
- FLA: 80e494f6c588e091fc8316b612870df29375c5b8

| case | cosine | max abs | mean abs | argmax |
|---|---:|---:|---:|---|
| b1_t1 | 0.99994114 | 1.25000000 | 0.40262452 | True |
| b1_t17 | 0.99988306 | 1.00000000 | 0.07850362 | True |
| b1_t128 | 0.99995860 | 1.06250000 | 0.04423372 | False |
| b4_t1 | 0.99968572 | 1.25000000 | 0.25150618 | False |
| b4_t17 | 0.99989194 | 1.12500000 | 0.07437638 | False |
| b4_t128 | 0.99994537 | 1.18750000 | 0.04885489 | False |
| cached_teacher_clean_vs_fla | 0.99995225 | 0.68750000 | 0.05283704 | True |
| clean_cached_vs_clean_full | 0.99993892 | 0.87500000 | 0.05572769 | True |
| fla_cached_vs_fla_full | 0.99993336 | 0.62500000 | 0.06236416 | True |
