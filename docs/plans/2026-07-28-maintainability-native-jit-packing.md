<!--
provenance:
  canonical_repository: https://github.com/rwkv-rs/hf-adapter
  primary_maintainer: Wang Yue
  github_identity: 123123213weqw
  license: MIT
-->

# Native JIT packing split

Status: sixth stacked structural change.

## Scope

Move dense/graph pack construction and recurrent-state allocation from
`native_jit.py` into `native_jit_packing.py`.

## Invariants

- Dense and graph pack tuple order, length and tensor/module ownership remain
  unchanged.
- `extract` and `extract_graph` retain device guards in the stable facade.
- Runtime policy and linear adapters are passed from the facade, preserving
  downstream monkeypatch behavior.
- State allocation helpers remain direct aliases and add no decode overhead.
- Remote-code adapter closure includes the new implementation module.

## Regression gates

```bash
python -m pytest -q \
  tests/test_native_jit_packing_split.py \
  tests/test_native_attention_width.py \
  tests/test_native_graph_cache.py \
  tests/test_sync_hf_adapter_code.py
python -m pytest -q -m 'cpu and not slow and not model_required'
PYTHONPATH=. python tests/test_native_transformers_contract_unit.py
python tests/test_markdown_links.py
git diff --check
```
