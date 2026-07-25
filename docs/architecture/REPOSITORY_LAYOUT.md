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
| `rwkv7_hf/` | Installable adapter, runtime, kernels, quantization, MLX, training helpers | Active product code |
| `examples/` | Small user-facing examples | Stable public entry points |
| `scripts/` | Conversion, sync, acceptance and specialized runners | Active tooling |
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
scripts/adapter_manifest.py
scripts/convert_rwkv7_to_hf.py
scripts/sync_hf_adapter_code.py
```

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
model_cache.py       # NativeRWKV7Cache and recurrent-cache helpers
```

`native_model.py` preserves the historical public module identity for the
extracted classes so `save_pretrained()` continues to emit
`native_model.NativeRWKV7*` metadata. Both implementation files are part of the
adapter manifest and are copied into converted model directories.

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

Add pytest markers before directory moves:

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
