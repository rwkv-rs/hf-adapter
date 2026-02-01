<!--
provenance:
  canonical_repository: https://github.com/rwkv-rs/hf-adapter
  primary_maintainer: Wang Yue
  github_identity: 123123213weqw
  license: MIT
-->

# Maintainability runtime-layout plan

Status: active structural migration plan, started 2026-07-27.

## Goal

Make the native Transformers adapter easier to maintain and reuse without
changing validated RWKV-7 math, tensor names, checkpoint compatibility,
hardware dispatch, quantization defaults, or performance routes.

The repository should converge toward the ownership boundaries documented in
`docs/architecture/REPOSITORY_LAYOUT.md`. Each pull request moves one coherent
responsibility and leaves the historical import surface as a compatibility
facade.

## Non-negotiable compatibility contract

- Keep `native_model.NativeRWKV7Config`, `NativeRWKV7Model`, and
  `NativeRWKV7ForCausalLM` as the canonical `auto_map` targets.
- Keep public class module identities and state-dict keys unchanged.
- Keep old converted checkpoints loadable after adapter sync.
- Keep `save_pretrained()` and offline reload working.
- Do not change default kernels, GPU policy tables, quant formats, logits,
  recurrent-state semantics, cache behavior, or generation behavior.
- Do not combine a structural move with performance tuning.
- Ship every extracted remote-code dependency through `ADAPTER_FILES`.

## Ordered migration

1. Extract native model environment/hardware policy behind stable facade
   wrappers. This is the first PR.
2. Separate native JIT orchestration from kernel implementations while keeping
   `native_jit.py` as a compatibility facade.
3. Group quantization formats, policy, and backends behind the existing native
   quantization entrypoints.
4. Group CUDA kernels by common, SM70, Ada, and Blackwell ownership only after
   remote-code packaging tests cover the nested layout.
5. Move MLX runtime ownership under an Apple backend namespace while retaining
   old import shims.

## First PR scope

- Add `model_runtime_policy.py` as the single owner of environment parsing and
  hardware-policy selection used by the native HF model.
- Keep wrappers in `native_model.py` so existing monkeypatches and integrations
  continue to work.
- Add the module to the converted-checkpoint adapter manifest.
- Add ownership, patch-surface, manifest-closure, and policy-parity tests.
- Document why direct remote-code dependency sentinels remain in the facade.

The first PR does **not** move kernels, alter `auto_map`, alter model math, or
claim a speed improvement.

## Acceptance

Run at minimum:

```bash
python -m pytest -q \
  tests/test_model_runtime_policy.py \
  tests/test_native_model_module_split.py \
  tests/test_native_prefill_graph.py \
  tests/test_native_graph_runtime_unit.py \
  tests/test_native_fla_free_import.py \
  tests/test_sync_hf_adapter_code.py \
  tests/test_clean_install_packaging.py
PYTHONPATH=. python tests/test_native_transformers_contract_unit.py
python tests/test_markdown_links.py
git diff --check
```

Before merge, compare public imports, state-dict keys, tiny-model logits/cache,
generation, save/reload, and a synced old converted model against `origin/main`.
GPU throughput reruns are not required for this policy-only move, but existing
card-specific dispatch tests must remain green.
