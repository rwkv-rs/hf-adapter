# Evaluation

## Clean reference versus FLA

Release comparisons require FLA commit
`80e494f6c588e091fc8316b612870df29375c5b8`.

```bash
python evaluation/compare_fla.py \
  --model /models/rwkv7-g1d-0.4b-hf \
  --dtype fp16 \
  --device cuda \
  --output-dir results/reference/4080
```

The default matrix covers batch 1/4 and lengths 1/17/128, cached teacher
forcing, final recurrent state and 64-token greedy generation. Formal runs
refuse an unverified FLA installation. `--allow-unverified-fla` exists only
for local smoke tests.

FP32 gates use rtol 1e-4 / atol 1e-5 for model tensors and the dedicated
operator/gradient suite uses rtol 5e-4 / atol 5e-5. FP16/BF16 require finite
values, cosine at least 0.9999, FP16 max logit error at most 0.15, and identical
64-token greedy output. FLA itself warns that official RWKV must remain the
final oracle; official checkpoint logits and greedy bundles are therefore a
separate release gate.

V100 runs FP32 and FP16. RTX 4080 runs FP32, FP16 and BF16. Every output bundle
records command, code SHA, FLA revision, model file hashes, environment and GPU.

The first provenance-complete RTX 4080 FP32/FP16/BF16 runs are archived at
[`results/4080-reference-20260825`](../results/4080-reference-20260825/README.md).
They are deliberately marked failed. FP16 B=4/T=1 measured 0.15625 and
B=4/T=128 measured 0.28125 against the fixed 0.15 max-absolute-logit limit;
BF16 missed cosine and greedy parity; FLA's FP32 path warned that it is not
supported on some platforms and missed both operator and model tolerances. Do
not treat that bundle as a passing release result or loosen the gates to
accommodate it.

## lm_eval

Install the fixed harness:

```bash
python -m pip install "lm_eval==0.4.9.1"
```

Run the 48 units (3 models x 2 batch sizes x 8 tasks):

```bash
python evaluation/run_lm_eval_matrix.py \
  --output-dir results/lm_eval/v0.9.0 \
  --device cuda
python evaluation/validate_lm_eval_matrix.py \
  --result-dir results/lm_eval/v0.9.0
```

Formal execution never uses `--limit`. Pull requests may set
`--smoke-limit`. Each task is an independent process with raw stdout,
stderr, sample logs, task config and manifest row. Batch 1/8 absolute metric
difference must be at most 0.001; Wikitext perplexity relative difference must
be at most 0.1%.
