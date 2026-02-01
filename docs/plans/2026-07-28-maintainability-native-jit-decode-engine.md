<!--
provenance:
  canonical_repository: https://github.com/rwkv-rs/hf-adapter
  primary_maintainer: Wang Yue
  github_identity: 123123213weqw
  license: MIT
-->

# Native JIT decode engine split

Status: tenth stacked structural change.

## Scope

Move dense-JIT stepping, eager in-place graph blocks, CUDA Graph runners,
decode benchmarks and greedy generation into `native_jit_decode.py`.

## Invariants

- Every token/layer hot function is a direct alias from the facade.
- Tensor pack ABI, in-place cache mutation and official operation order remain
  unchanged.
- Dense, native-quantized and optional card-kernel dispatch stay unchanged.
- External graph runners keep the same import names and call depth.
- The module is included in the remote-code adapter closure.

## Regression gates

```bash
python -m pytest -q \
  tests/test_native_jit_decode_split.py \
  tests/test_native_attention_width.py \
  tests/test_native_graph_cache.py \
  tests/test_native_graph_runtime_unit.py \
  tests/test_native_model.py \
  tests/test_native_prefill_scan.py \
  tests/test_sync_hf_adapter_code.py
python -m pytest -q -m 'cpu and not slow and not model_required'
PYTHONPATH=. python tests/test_native_transformers_contract_unit.py
python tests/test_markdown_links.py
git diff --check
```

The stacked branch must run B1/B8 dense native-JIT and native-graph decode
correctness, throughput and VRAM on the RTX 4080 before final integration.
