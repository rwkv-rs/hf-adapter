# Repository instructions

## Scope

`main` is the readable, compatibility-first RWKV-7 Hugging Face reference
implementation. It contains pure PyTorch model mathematics, a canonical
recurrent cache, checkpoint conversion, ecosystem tests, evaluation tooling,
and reproducible LoRA examples.

CUDA/Triton kernels, JIT, CUDA Graph, quantization, device routing, serving
benchmarks, and hardware-specific policy belong only on
`perf/native-kernels-v0.8`. Do not reintroduce them on `main`. A future
optimized implementation may replace only the explicit
`rwkv7_hf/ops_rwkv7.py` boundary.

## Public contract

Keep these interfaces stable:

- `RWKV7Config`, `RWKV7Cache`, `RWKV7Model`, and `RWKV7ForCausalLM`;
- the `NativeRWKV7*` 0.9 compatibility aliases;
- `AutoConfig`, `AutoModel`, `AutoModelForCausalLM`, and tokenizer `auto_map`;
- converted checkpoint parameter names;
- cache state `[B,H,K,V]`, attention shift, FFN shift, and `seen_tokens`.

`v_first` crosses layers only during one forward and must not enter the cache.
Padding positions must not update recurrent or shift state. Training and
gradient checkpointing disable cache. Do not synthesize attention matrices.

## Validation

Before merging a source change, run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
git diff --check
python -m build
python -m twine check --strict dist/*
```

Also keep the Transformers 4.48 compatibility lane green. Model-math or cache
changes require clean-vs-FLA and official-checkpoint parity on V100 and RTX
4080 before release. Do not relax published tolerances to turn a failed result
into a pass; retain the failure bundle and fix or document the oracle/runtime
problem.

Training changes require a finite loss, non-zero gradient, changed trainable
parameters, adapter save/reload parity, and resume-step advancement. Local
JSON/JSONL files are the source of truth; W&B is optional.

## Repository hygiene

- Do not commit weights, checkpoints, generated distributions, caches, tokens,
  credentials, machine-specific paths, or benchmark scratch data.
- A reference-converted model directory must be self-contained and load with
  Torch + Transformers without installing `rwkv7-hf` or FLA.
- Preserve historical performance work and attribution on the performance
  branch and old tags rather than copying it back to `main`.
- Keep documentation links valid and status claims tied to immutable result
  bundles.

## Rust compilation

Do not compile Rust locally. If a future change introduces Rust, synchronize it
to the configured remote build host and run all Cargo/rustc compilation there;
`cargo fmt` remains allowed locally.
