<!--
provenance:
  canonical_repository: https://github.com/rwkv-rs/hf-adapter
  primary_maintainer: Wang Yue
  github_identity: 123123213weqw
  metadata: ../reference/provenance.yaml
  license: MIT
-->

# Repository layout and refactor contract

## Purpose

This document defines where new work belongs and how the repository can be
reorganized without breaking converted RWKV-7 model directories, remote-code
loading, or validated hardware routes.

The current repository intentionally retains a flat `rwkv7_hf/` runtime because
converted checkpoints copy a manifest of Python files beside `config.json`.
Physical source moves therefore require packaging work before they are cosmetic
cleanup.

## Current top-level ownership

| Path | Purpose | Lifecycle |
|---|---|---|
| `rwkv7_hf/` | Installable adapter, converter, CLI, runtime, kernels, quantization, MLX, training helpers | Active product code |
| `examples/` | Small user-facing examples | Stable public entry points |
| `scripts/` | Compatibility wrappers, sync, acceptance and specialized runners | Active tooling |
| `tests/` | Unit, API, integration, hardware-policy and artifact tests | Active verification |
| `docs/` | Canonical guides, architecture, hardware, validation and history | Lifecycle-classified |
| `bench/` | Benchmark tools plus immutable dated evidence | Append-only evidence |
| `configs/` | Reproducible training/runtime configurations | Active configuration |

Root documents are entry points, not storage for long experiment logs:

```text
README.md / README_ZH.md
AGENTS.md
CONTRIBUTING.md
HF_STATUS.md
HF_TODO.md
BENCHMARK.md
CONTRIBUTORS.md / CONTRIBUTIONS.md
LICENSE / CITATION.cff
```

## Stable converted-model surface

These paths and exported classes must remain available:

```text
rwkv7_hf/native_model.py
rwkv7_hf/tokenization_rwkv7.py
rwkv7_hf/__init__.py
rwkv7_hf/cli.py
rwkv7_hf/converter.py
rwkv7_hf/adapter_manifest.py
scripts/adapter_manifest.py
scripts/convert_rwkv7_to_hf.py
scripts/sync_hf_adapter_code.py
```

`rwkv7_hf/converter.py` is the installed and canonical single-checkpoint
converter. `scripts/convert_rwkv7_to_hf.py` remains a source-checkout wrapper
for existing automation. The package and script manifest lists are kept in
sync by tests so both installed conversion and repository tooling copy the
same bundled runtime surface.

Canonical converted `config.json` contains:

```json
{
  "auto_map": {
    "AutoConfig": "native_model.NativeRWKV7Config",
    "AutoModel": "native_model.NativeRWKV7Model",
    "AutoModelForCausalLM": "native_model.NativeRWKV7ForCausalLM"
  }
}
```

Consequently, `native_model.py` must remain a usable top-level remote-code
entry even after its implementation is split. The source package may gain
nested ownership directories before the converted-model import graph does.

The first remote-safe model split keeps sibling modules flat:

```text
native_model.py      # stable Auto* entrypoint and public re-exports
model_config.py      # NativeRWKV7Config implementation
model_fast_api.py    # native prefill, chunked-prefill and fast-token serving mixin
model_generation.py  # HF generation input preparation and cache reordering mixin
model_cache.py       # NativeRWKV7Cache and recurrent-cache helpers
model_layers.py      # attention, FFN and block module definitions
model_backbone.py    # NativeRWKV7Model and recurrent forward helpers
model_prefill_graph.py # fixed-shape CUDA graph runner for native prefill
model_quantization.py # BNB loading policy and native W8/W4 replacement mixin
model_runtime_policy.py # environment/hardware selection with facade wrappers
model_speculative.py # speculative-generation mixin and acceptance loop
native_jit.py        # stable native JIT facade and execution orchestration
native_jit_bnb8.py   # BnB W8 detection, direct operators and fused eligibility
native_jit_dense_step.py # pure tensor-only TorchScript layer steps
native_jit_decode.py # dense-JIT and CUDA-graph decode execution
native_jit_graph_dispatch.py # graph policy, projection and FFN dispatch
native_jit_linear.py # dense/quant linear operands and low-memory relayout
native_jit_packing.py # model pack extraction and recurrent-state allocation
native_jit_prefill.py # sequence prefill execution and cache handoff
native_jit_prefill_policy.py # pure shape allowlists and tiling selection
native_jit_prefill_runtime_policy.py # kernel eligibility and launch policy
native_jit_recurrent.py # recurrent kernel selection and eager fallback math
```

`native_model.py` preserves the historical public module identity for the
extracted classes so `save_pretrained()` continues to emit
`native_model.NativeRWKV7*` metadata. The extracted implementation files are
part of the adapter manifest and are copied into converted model directories.

Runtime-policy implementation lives in `model_runtime_policy.py`, while the
historical underscore-prefixed helpers remain as wrappers in `native_model.py`.
This preserves the existing hardware-test and integration patch surface: a
caller that replaces `native_model.current_kernel_policy` or
`native_model._native_jit_prefill` still controls the same decision points.

The non-executed direct dependency imports in `native_model.py` are deliberate.
They cannot be hidden behind one dependency-registry module: supported older
Transformers releases may copy only modules directly visible from the Auto*
entrypoint. Moving those imports would make source layout cleaner while making
some converted checkpoints unloadable. Keep that compatibility edge until the
supported Transformers range proves recursive remote-module discovery.

The first native-JIT split follows the same rule. `native_jit_linear.py` owns
linear operand normalization and sparse FFN storage relayout, while
`native_jit.py` keeps the historical underscore-prefixed symbols. Hot-path
helpers are imported as direct aliases, avoiding an additional Python wrapper
call; the relayout compatibility wrapper retains its existing monkeypatch
surface outside the token loop.

Prefill exact-shape parsing and self-chunk tiling live in
`native_jit_prefill_policy.py`. The module is intentionally independent from
Torch, CUDA and optional kernels. `native_jit.py` supplies the current card
policy, environment helpers and availability checks through stable wrappers,
so hardware tests and downstream policy overrides retain their old target.

BnB W8 inference dispatch lives in `native_jit_bnb8.py`. Its historical
underscore-prefixed names are direct aliases from `native_jit.py`, avoiding an
extra Python call for every quantized projection. The module owns only BnB
eligibility and operator dispatch; mixed-sequence orchestration remains in the
prefill runtime until that whole execution boundary is moved.

The tensor-only TorchScript layer functions live in
`native_jit_dense_step.py`. `native_jit.py` re-exports the exact ScriptFunction
objects, so the dense JIT token loop gains no wrapper call and the historical
pack ABI remains unchanged.

Model-container traversal and recurrent-state allocation live in
`native_jit_packing.py`. Policy and operand adapters are passed explicitly by
the facade, preserving the existing `native_jit` monkeypatch points while
keeping checkpoint-specific packing outside execution math.

Native CUDA-graph feature gates and projection/FFN dispatch live in
`native_jit_graph_dispatch.py`. The facade binds optional kernels and policy
helpers once, then re-exports direct function aliases. This preserves the hot
path while separating hardware routing from recurrent orchestration.

Prefill kernel eligibility and launch selection live in
`native_jit_prefill_runtime_policy.py`. The facade uses compatibility wrappers
that refresh only referenced dependencies, preserving existing policy and
optional-kernel overrides without moving sequence execution into policy code.

Sequence projections, recurrent scan routing, layer-wise prefill math and
cache handoff live in `native_jit_prefill.py`. The facade binds the current
policy and optional kernels once per public prefill call; inner layer and scan
calls remain inside the implementation module without compatibility frames.

Dense-JIT stepping, eager graph blocks, CUDA Graph runners and greedy decode
live in `native_jit_decode.py`. Their facade names are direct aliases because
`native_model`, `modeling_rwkv7` and `native_graph_runtime` call the block
functions inside the token/layer loop.

Recurrent kernel eligibility and the tensor fallback update live in
`native_jit_recurrent.py`. Both facade names are direct aliases. After this
split, `native_jit.py` is the 794-line compatibility facade and optional-kernel
binding registry rather than an execution monolith.

## Intended package boundaries

The long-term package may be organized as:

```text
rwkv7_hf/
  __init__.py
  native_model.py          # stable facade
  tokenization_rwkv7.py    # stable facade
  model/
    config.py
    cache.py
    attention.py
    ffn.py
    model.py
    generation.py
  runtime/
    eager.py
    graph.py
    dispatch.py
    policy.py
  kernels/
    common/
    triton/
    sm70/
    ada/
    blackwell/
  quantization/
    formats.py
    mm8.py
    mm4.py
    a8w8.py
    marlin.py
    policy.py
  training/
    train_temp.py
    alignment.py
    resume.py
  backends/
    mlx/
  compat/
    fla/
  csrc/
```

This is a migration target, not permission for a mass move.

## Required migration order

### 1. Documentation and ownership

- Keep root entry documents short.
- Move command catalogs to `docs/contributing/`.
- Move dated milestone narratives to `docs/archive/`.
- Preserve canonical status and raw evidence paths.

### 2. Nested remote-code packaging

Before moving runtime modules:

- represent manifest paths with platform-independent `/` separators;
- reject absolute paths, `..`, duplicate destinations and symlink escapes;
- create nested destination directories during conversion and sync;
- retain the current flat `auto_map` and runtime dependency graph until a
  separate compatibility PR proves another layout;
- load an old flat converted model after sync;
- verify save/reload and tokenizer loading.

Nested manifest copying and nested Python imports are different contracts.
Current Transformers dynamic-module discovery reliably follows sibling
relative imports, but supported releases do not all resolve imports such as
`from .model.config import ...` from a remote entrypoint. Therefore the initial
packaging change ships only a nested package marker and must not move runtime
dependencies behind a nested import. A later model split must either retain a
flat remote-code dependency namespace or first prove a changed nested
entrypoint through offline `AutoConfig`, `AutoModel`, save/reload and old-model
sync tests across the supported Transformers range.

### 3. Model split

Split `native_model.py` into config, cache, layers, model, and generation while
retaining top-level re-exports. Do not change tensor names or state-dict keys,
and do not assume the installed-package directory layout can be copied directly
into a Transformers dynamic-module cache.

### 4. Runtime and kernel split

Move implementations by capability family. Keep dispatch centralized and retain
old module paths as compatibility shims for at least one migration cycle.

### 5. Tests and scripts

The centrally enforced pytest markers are now in place and must remain valid
before and after directory moves:

```text
cpu
cuda
sm70
ada
blackwell
apple
slow
model_required
```

Move benchmark-specific executables only after documentation and CI stop
depending on their old paths, or leave wrapper scripts at the old paths.

## Documentation placement

| Content | Directory |
|---|---|
| Installation and usage | `docs/` canonical user guides |
| Architecture/contracts | `docs/architecture/` |
| Backend boundaries | `docs/BACKENDS.md`, `docs/backends/` when introduced |
| Hardware-specific behavior | `docs/hardware/` |
| Contributor command catalogs | `docs/contributing/` |
| Performance methodology | `docs/performance/` |
| Exact validation summaries | `docs/validation/` |
| Machine-readable contracts | `docs/reference/` |
| Dated plans | `docs/plans/` with historical banner |
| Superseded milestones | `docs/archive/` |

## Benchmark evidence rule

Existing `bench/<artifact>/` paths are immutable because many canonical
documents link to them. Do not reorganize historical artifacts merely for a
cleaner tree.

New artifacts should continue to include:

```text
README
exact command
environment
raw JSONL/logs
correctness
speed
memory
commit and checkpoint provenance
```

`bench/INDEX.md` is the discovery layer; physical movement is unnecessary.

## Compatibility-shim rule

A moved public or remote-code module leaves a small old-path facade:

```python
"""Compatibility import; implementation moved without changing behavior."""

from .new_location import PublicClass, public_function

__all__ = ["PublicClass", "public_function"]
```

Shims must not duplicate state, register a second kernel namespace, or silently
change fallback policy.

## Refactor acceptance

Every structural PR must pass:

```text
Markdown links and document lifecycle
clean package import without optional dependencies
adapter manifest closure
official checkpoint conversion
old converted-model sync/load
new converted-model load
save_pretrained/reload
public import identity
state-dict key equality
no GPU policy or benchmark default changes
```

Source moves and performance changes belong in separate PRs. A structural PR
must not claim speed improvement.
