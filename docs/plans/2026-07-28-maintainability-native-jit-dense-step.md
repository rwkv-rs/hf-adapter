<!--
provenance:
  canonical_repository: https://github.com/rwkv-rs/hf-adapter
  primary_maintainer: Wang Yue
  github_identity: 123123213weqw
  license: MIT
-->

# Native JIT dense TorchScript step split

Status: fifth stacked structural change.

## Scope

Move the tensor-only `block_step` and `block_step_batched` ScriptFunctions from
`native_jit.py` into `native_jit_dense_step.py`.

## Invariants

- `native_jit.block_step*` are direct aliases to the same ScriptFunction
  objects; no hot-path wrapper is introduced.
- The positional tensor pack ABI and official RWKV-7 operation order are
  unchanged.
- No policy, quantization, prefill, CUDA Graph or model-container behavior
  changes.
- The implementation is included in the remote-code adapter closure.

## Regression gates

```bash
python -m pytest -q \
  tests/test_native_jit_dense_step_split.py \
  tests/test_native_jit_module_split.py \
  tests/test_sync_hf_adapter_code.py
python -m pytest -q -m 'cpu and not slow and not model_required'
PYTHONPATH=. python tests/test_native_transformers_contract_unit.py
python tests/test_markdown_links.py
git diff --check
```

Before stacking the next structural change, run dense native JIT B1/B8 decode
correctness, throughput and VRAM regression on the RTX 4080.
