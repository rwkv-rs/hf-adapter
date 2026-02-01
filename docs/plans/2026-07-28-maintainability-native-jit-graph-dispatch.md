<!--
provenance:
  canonical_repository: https://github.com/rwkv-rs/hf-adapter
  primary_maintainer: Wang Yue
  github_identity: 123123213weqw
  license: MIT
-->

# Native JIT graph dispatch split

Status: seventh stacked structural change.

## Scope

Move native CUDA-graph feature gates, R/K/V routing, projection/FFN dispatch,
and sparse-FFN prewarm from `native_jit.py` into
`native_jit_graph_dispatch.py`.

## Invariants

- Hot dispatch functions remain direct facade aliases.
- Existing card policy, environment overrides, optional-kernel fallbacks and
  quantized linear behavior remain unchanged.
- Blackwell norm/mix keeps the historical facade monkeypatch surface.
- Recurrent, prefill and decode mathematical order is unchanged.
- The new module is included in the remote-code adapter closure.

## Regression gates

```bash
python -m pytest -q \
  tests/test_native_jit_graph_dispatch_split.py \
  tests/test_blackwell_norm_mix.py \
  tests/test_native_quant_marlin.py \
  tests/test_native_graph_cache.py \
  tests/test_kernel_policy.py \
  tests/test_sync_hf_adapter_code.py
python -m pytest -q -m 'cpu and not slow and not model_required'
PYTHONPATH=. python tests/test_native_transformers_contract_unit.py
python tests/test_markdown_links.py
git diff --check
```
