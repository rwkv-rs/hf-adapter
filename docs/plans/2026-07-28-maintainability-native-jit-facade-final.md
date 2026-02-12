<!--
provenance:
  canonical_repository: https://github.com/rwkv-rs/hf-adapter
  primary_maintainer: Wang Yue
  github_identity: 123123213weqw
  license: MIT
-->

# Native JIT facade completion

Status: complete. Local, RTX 4080 and Tesla V100 regression gates pass.

## Scope

Move recurrent kernel selection and fallback math into
`native_jit_recurrent.py`, and reduce generated runtime dependency lists to
the names actually supplied by the facade.

## Result

The original 4,824-line `native_jit.py` is now a 794-line stable facade. Its
remaining responsibilities are deliberate:

- remote-code-compatible optional-kernel imports and fallbacks;
- historical private API aliases and compatibility wrappers;
- input-device guards around pack extraction and prefill;
- dependency binding for the split execution modules;
- the standalone diagnostic CLI.

Execution math, packing, graph dispatch, quantized BnB dispatch, prefill
policy, prefill execution, recurrent updates and decode execution have distinct
owners. No further native-JIT split is required before integration.

## Regression gates

```bash
python -m pytest -q \
  tests/test_native_jit_recurrent_split.py \
  tests/test_native_jit_decode_split.py \
  tests/test_native_jit_prefill_split.py \
  tests/test_native_jit_prefill_runtime_policy_split.py \
  tests/test_native_jit_graph_dispatch_split.py \
  tests/test_native_jit_packing_split.py \
  tests/test_native_jit_dense_step_split.py \
  tests/test_sync_hf_adapter_code.py
python -m pytest -q -m 'cpu and not slow and not model_required'
PYTHONPATH=. python tests/test_native_transformers_contract_unit.py
python tests/test_markdown_links.py
git diff --check
```

## Exact-card integration

- RTX 4080, RWKV-7 1.5B, fp16, B1/B8: native-JIT decode,
  native-graph decode and native-graph prefill pass the `-2%` refactor floor;
  greedy traces, numerical outputs, effective backends and peak VRAM are
  unchanged. Evidence: [`../../bench/4080_native_jit_split_20260728/`](../../bench/4080_native_jit_split_20260728/README.md).
- Tesla V100, RWKV-7 0.1B, fp16, B1/B8: CUDA tests, W8/W4 remote-code smoke,
  native-JIT decode, native-graph decode and native-graph prefill pass. The
  noisy short graph-B1 row was replaced by a bracketed 1,024-token rerun.
  Evidence: [`../../bench/v100_native_jit_split_20260728/`](../../bench/v100_native_jit_split_20260728/README.md).

No further split of `native_jit.py` is required. Its remaining 794 lines are
the intentional remote-code compatibility facade and optional-kernel binding
boundary.
