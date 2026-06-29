# Evaluation

## Official RWKV checkpoint oracle

Official RWKV checkpoint behavior is the model-correctness oracle. The oracle
matrix covers no-cache logits, prefill state, teacher-forced cached decode,
causal loss, padding, and 64-token greedy generation. Every bundle records the
official source revision, code revision, checkpoint hashes, environment, and
exact command. RTX 4080 runs FP32/FP16/BF16; V100 runs FP32/FP16.

The official oracle and the reference implementation use separate source
paths and independent state containers. A mismatch is localized at projection,
decay, normalized key, WKV output/state, normalization, block output, and final
logits before changing model mathematics.

For FP32 logits the harness follows the official NumPy verification metric,
`max(abs(reference-candidate)) / std(reference) <= 1e-4`; recurrent and shift
states use `rtol=1e-4, atol=1e-5`. Low-precision tensors must be finite with
cosine at least 0.9999, FP16 logits additionally use max-absolute error 0.15,
and 64-token greedy output must match exactly. Per-layer traces are diagnostic
and identify where accumulation begins; they are not an additional gate once
the official final outputs, state, loss, padding, and greedy checks pass.

## Optional FLA backend diagnostic

FLA is an optimized training/inference backend reference, not the correctness
oracle and not a runtime dependency. Its comparison lives under
[`benchmarks/fla`](../benchmarks/fla/README.md) and returns success by default
even when diagnostic thresholds are missed. Use `--require-thresholds` only in
an explicitly performance-backend-focused job.

The first RTX 4080 diagnostic bundle is archived at
[`benchmarks/fla/results/4080-reference-20260825`](../benchmarks/fla/results/4080-reference-20260825/README.md).

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
