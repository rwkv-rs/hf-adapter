# clean-vs-fla-model-fp16

- status: **failed**
- model: /home/wzu/codex-run/rwkv7-reference-convert-4080/model
- dtype: fp16
- device: NVIDIA GeForce RTX 4080
- code: f657c67
- FLA: 80e494f6c588e091fc8316b612870df29375c5b8

| case | cosine | max abs | mean abs | argmax |
|---|---:|---:|---:|---|
| b1_t1 | 0.99999937 | 0.09375000 | 0.01535093 | True |
| b1_t17 | 0.99999781 | 0.11718750 | 0.01109659 | True |
| b1_t128 | 0.99999932 | 0.08007812 | 0.00551010 | False |
| b4_t1 | 0.99999654 | 0.15625000 | 0.02391614 | True |
| b4_t17 | 0.99999906 | 0.09375000 | 0.00783405 | True |
| b4_t128 | 0.99999870 | 0.28125000 | 0.00681759 | False |
| cached_teacher_clean_vs_fla | 0.99999925 | 0.09375000 | 0.00647949 | True |
| clean_cached_vs_clean_full | 0.99999917 | 0.10156250 | 0.00691598 | True |
| fla_cached_vs_fla_full | 0.99999930 | 0.07812500 | 0.00616372 | True |
