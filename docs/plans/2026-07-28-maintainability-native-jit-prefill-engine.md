<!--
provenance:
  canonical_repository: https://github.com/rwkv-rs/hf-adapter
  primary_maintainer: Wang Yue
  github_identity: 123123213weqw
  license: MIT
-->

# Native JIT prefill engine split

> Historical implementation record. Current status is tracked in `HF_STATUS.md` and `HF_TODO.md`.

Status: ninth stacked structural change.

## Scope

Move sequence linear helpers, stacked-RKV packing cache, recurrent scan
routing, full layer-wise prefill execution and cache handoff into
`native_jit_prefill.py`.

## Invariants

- Operation order, recurrent state orientation and logits selection are
  unchanged.
- Public `prefill` retains its input-device guard in the stable facade.
- Runtime policy and optional-kernel overrides are refreshed once per public
  call; inner loops do not add compatibility wrappers.
- Dense and quantized projection fallbacks remain unchanged.
- The implementation is included in the remote-code adapter closure.

## Regression gates

```bash
python -m pytest -q \
  tests/test_native_jit_prefill_split.py \
  tests/test_native_prefill_scan.py \
  tests/test_native_attention_width.py \
  tests/test_fast_prefill_forward.py \
  tests/test_native_prefill_graph.py \
  tests/test_sync_hf_adapter_code.py
python -m pytest -q -m 'cpu and not slow and not model_required'
PYTHONPATH=. python tests/test_native_transformers_contract_unit.py
python tests/test_markdown_links.py
git diff --check
```

The stacked branch must run B1/B8 prefill correctness, throughput and VRAM on
the RTX 4080 before final integration.
