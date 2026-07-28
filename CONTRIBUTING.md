# Contributing

The main branch is the correctness-first Hugging Face reference line. Changes
should improve readability, mathematical correctness, framework compatibility,
conversion, evaluation or reproducibility.

Performance kernels, hardware routes, JIT, CUDA Graph and quantization belong
in the separately built `rwkv7-kernels` distribution under `kernels/`. They
may replace only the frozen API-v4 operations documented in
[`docs/KERNEL_PLUGIN_API.md`](docs/KERNEL_PLUGIN_API.md); they must not add a
second model, configuration, tokenizer, or public cache implementation.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
python -m pytest -q
```

## Pull-request checks

- keep `modeling_rwkv7.py` understandable without private runtime code;
- preserve the API-v4 operation names, envelope fields, `[B,H,K,V]` public
  cache layout, and fail-closed semantics for the complete 1.0 release line;
- add CPU tests for model, cache, padding, loss and save/reload changes;
- compare mathematical changes with the official RWKV oracle;
- treat FLA comparisons as non-blocking optimized-backend diagnostics;
- do not weaken published tolerances to make a regression pass;
- record commands, code/model/dataset revisions and raw GPU output;
- never commit access tokens or W&B credentials;
- update English and Chinese docs when the public workflow changes.

Release work must also pass the official GPU matrices, formal lm_eval, three fine-tuning
examples, clean-wheel installation and Hub download smoke.
