<!--
provenance:
  canonical_repository: https://github.com/rwkv-rs/hf-adapter
  primary_maintainer: Wang Yue
  github_identity: 123123213weqw
  license: MIT
-->

# Native JIT BnB W8 split

Status: fourth stacked structural change.

## Scope

Move BnB W8 module detection, policy lookup, threshold-zero direct operators,
prequantized linear dispatch and fused activation-quantization eligibility from
`native_jit.py` into `native_jit_bnb8.py`.

## Invariants

- Existing private names remain importable from `rwkv7_hf.native_jit`.
- Hot helpers are direct aliases, not compatibility wrappers.
- Training, gradient, outlier-threshold and non-BnB fallbacks stay unchanged.
- Existing BnB operators, activation fusion and block-size policy stay intact.
- No dense, MM8, MM4, recurrent, prefill or decode math changes.
- The module is present in the adapter manifest and direct remote-code closure.

## Regression gates

```bash
python -m pytest -q \
  tests/test_native_jit_bnb8_split.py \
  tests/test_native_quant_bnb8.py \
  tests/test_native_jit_module_split.py \
  tests/test_sync_hf_adapter_code.py
python -m pytest -q -m 'cpu and not slow and not model_required'
PYTHONPATH=. python tests/test_native_transformers_contract_unit.py
python tests/test_markdown_links.py
git diff --check
```

Before merging into a production branch, run the existing BnB W8 correctness,
VRAM and B1/B8 performance lane on an Ada or Blackwell GPU.
