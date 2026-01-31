<!--
provenance:
  canonical_repository: https://github.com/rwkv-rs/hf-adapter
  primary_maintainer: Wang Yue
  github_identity: 123123213weqw
  license: MIT
-->

# Native JIT prefill runtime policy split

Status: eighth stacked structural change.

## Scope

Move prefill kernel eligibility, exact-card feature gates, tile selection and
launch policy from `native_jit.py` into
`native_jit_prefill_runtime_policy.py`.

## Invariants

- Mathematical prefill execution remains in the runtime engine.
- Existing private names remain callable from `native_jit`.
- Facade policy and optional-kernel monkeypatches remain observable, including
  overrides of one policy helper used by another.
- Environment and exact-card precedence remain unchanged.
- The module is included in the remote-code adapter closure.

## Regression gates

```bash
python -m pytest -q \
  tests/test_native_jit_prefill_runtime_policy_split.py \
  tests/test_native_jit_prefill_policy.py \
  tests/test_native_prefill_scan.py \
  tests/test_native_prefill_graph.py \
  tests/test_fast_prefill_forward.py \
  tests/test_kernel_policy.py \
  tests/test_sync_hf_adapter_code.py
python -m pytest -q -m 'cpu and not slow and not model_required'
PYTHONPATH=. python tests/test_native_transformers_contract_unit.py
python tests/test_markdown_links.py
git diff --check
```
