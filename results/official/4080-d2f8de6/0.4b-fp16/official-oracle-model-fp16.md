# official-oracle-model-fp16

- status: **passed**
- model: /home/wzu/codex-run/rwkv7-reference-convert-4080/model
- dtype: fp16
- device: NVIDIA GeForce RTX 4080
- code: d2f8de695826af9ddd0dbf1054f73637b6797953

| case | cosine | max abs | mean abs | argmax |
|---|---:|---:|---:|---|
| b1_t1 | 0.99999935 | 0.12500000 | 0.04496785 | True |
| b1_t17 | 0.99999866 | 0.09375000 | 0.00905303 | True |
| b1_t128 | 0.99999900 | 0.17968750 | 0.00584165 | False |
| b4_t1 | 0.99999159 | 0.18750000 | 0.04366957 | True |
| b4_t17 | 0.99999915 | 0.09375000 | 0.00733106 | True |
| b4_t128 | 0.99999935 | 0.12500000 | 0.00551726 | False |
| cached_teacher | 0.99999821 | 0.14062500 | 0.00880430 | True |
