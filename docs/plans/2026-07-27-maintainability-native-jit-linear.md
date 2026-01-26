<!--
provenance:
  canonical_repository: https://github.com/rwkv-rs/hf-adapter
  primary_maintainer: Wang Yue
  github_identity: 123123213weqw
  license: MIT
-->

# Native JIT linear-boundary split

Status: second stacked structural change, based on the runtime-policy split.

## Scope

Extract the backend-independent linear operand and low-memory FFN relayout
helpers from `native_jit.py` into `native_jit_linear.py`.

The extracted layer owns:

- dense versus callable quantized linear dispatch;
- CUDA-graph operand representation and shape inspection;
- sparse FFN down-weight storage relayout;
- eligibility checks for applying that relayout.

## Compatibility rules

- Keep every historical underscore-prefixed symbol importable from
  `rwkv7_hf.native_jit`.
- Do not add a wrapper call around token-loop linear/shape helpers.
- Keep relayout tensor shape, values, parameter name and state-dict key stable.
- Keep MM8, MM4, BnB and ordinary `nn.Linear` operand behavior unchanged.
- Add the helper to `ADAPTER_FILES` and to `native_model.py` direct dependency
  discovery for older Transformers releases.
- Do not alter kernel policy, model math, graph capture, or hardware defaults.

## Acceptance

```bash
python -m pytest -q \
  tests/test_native_jit_module_split.py \
  tests/test_ada_sparse_ffn.py \
  tests/test_native_model_module_split.py \
  tests/test_sync_hf_adapter_code.py
python -m pytest -q -m 'cpu and not slow and not model_required'
PYTHONPATH=. python tests/test_native_transformers_contract_unit.py
python tests/test_markdown_links.py
git diff --check
```

GPU performance is intentionally unchanged: hot helpers remain direct aliases
and no CUDA/Triton implementation moves in this change.
