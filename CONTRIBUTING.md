# Contributing

The main branch is the correctness-first Hugging Face reference line. Changes
should improve readability, mathematical correctness, framework compatibility,
conversion, evaluation or reproducibility.

Performance kernels, hardware routes, JIT, CUDA Graph and quantization belong
on `perf/native-kernels-v0.8`. A future optimized backend must replace only
the explicit operator boundary in `rwkv7_hf/ops_rwkv7.py`; it must not
obscure the reference modeling code.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
python -m pytest -q
```

## Pull-request checks

- keep `modeling_rwkv7.py` understandable without private runtime code;
- add CPU tests for model, cache, padding, loss and save/reload changes;
- compare mathematical changes with the official RWKV oracle;
- treat FLA comparisons as non-blocking optimized-backend diagnostics;
- do not weaken published tolerances to make a regression pass;
- record commands, code/model/dataset revisions and raw GPU output;
- never commit access tokens or W&B credentials;
- update English and Chinese docs when the public workflow changes.

Release work must also pass the official GPU matrices, formal lm_eval, three fine-tuning
examples, clean-wheel installation and Hub download smoke.
