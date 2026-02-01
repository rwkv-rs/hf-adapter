<!--
provenance:
  canonical_repository: https://github.com/rwkv-rs/hf-adapter
  primary_maintainer: Wang Yue
  github_identity: 123123213weqw
  license: MIT
-->

# Native JIT prefill-policy split

Status: third stacked structural change.

## Scope

Move exact model-shape parsing, self-chunk eligibility, chunk-size selection
and tile lookup into a Torch-independent `native_jit_prefill_policy.py` module.

`native_jit.py` remains responsible for:

- resolving the current card policy;
- reading route enable/disable flags;
- checking CUDA/Triton implementation availability;
- exposing every historical helper name.

## Invariants

- Exact-card allowlists and environment precedence remain unchanged.
- Invalid `HxLxBxT` values raise the same errors.
- Existing `native_jit._kernel_policy` monkeypatches still control wrappers.
- No token-loop function, recurrence math, kernel launch or quant path moves.
- The new file is shipped and directly discoverable by remote-code loading.

## Acceptance

```bash
python -m pytest -q \
  tests/test_native_jit_prefill_policy.py \
  tests/test_native_prefill_scan.py \
  tests/test_sync_hf_adapter_code.py
python -m pytest -q -m 'cpu and not slow and not model_required'
PYTHONPATH=. python tests/test_native_transformers_contract_unit.py
python tests/test_markdown_links.py
git diff --check
```
