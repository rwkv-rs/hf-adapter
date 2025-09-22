# Contributors

This file records who contributed what to the RWKV-7 HF/Transformers adapter,
grouped by work-type, so contribution scoring reflects the kind of work done
rather than raw line counts (benchmark evidence rows can outnumber source
code).

## Work-types

- `algorithm` — architecture and algorithm design
- `engineering` — implementation (modeling, kernels, tests, CI, scripts)
- `validation` — running the project's benchmark/smoke scripts on a GPU and recording the results
- `data` — benchmark result rows
- `docs` — documentation
- `coordination` — issues, review, releases

## Contributors

### @dsadsasdaddas (wangyue) — lead

Designed and implemented the adapter, including: the HF wrapper
(`modeling_rwkv7`), the `native_jit` / `native_graph` fast-token backends, the
FLA-free `native_model`, the `fused_*.py` operators, the `mm8` / `mm4`
quantization ports, the speculative-decoding draft-training recipe and
`rwkv7_speculative_generate`, the DeepSpeed ZeRO checkpoint-resume fix, and the
per-GPU `kernel_policy` rules. Also wrote the benchmark scripts, CI, tests, and
documentation, and handles issue triage (#66–#93) and PR review.

work-types: `algorithm` `engineering` `docs` `coordination` `validation`

### @MosRat

Ran the project's benchmark scripts on A100 (Ampere) and contributed the
large-model validation result rows (#82, #84).

work-types: `validation` `data`

### @aierwiki

Ran the project's benchmark scripts on A800 and contributed the result rows,
and extended the converter / `sync_hf_adapter_code` file list so converted
model directories include all runtime modules (#97).

work-types: `validation` `engineering`

### @yuyi2439

Contributed RTX 3060 test-data rows (#87).

work-types: `data`
