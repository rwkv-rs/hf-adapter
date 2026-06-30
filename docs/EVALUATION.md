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

The oracle executes HF token-by-token for its blocking semantic comparison and
records the normal vectorized `B*T` execution separately. This prevents a CUDA
GEMM layout/order choice from being mistaken for a different checkpoint or
equation. FP32 logits use the official NumPy normalized metric with a calibrated
`2e-4` ceiling; recurrent and shift states pass either
`rtol=1e-4, atol=1e-5` or cosine `0.999999`. The cosine fallback avoids
rejecting a state for a handful of near-zero entries while its mean error is
below the FP32 accumulation noise floor.
Low-precision tensors must be finite with cosine at least `0.9999` for FP16 and
`0.999` for BF16. FP32/FP16 64-token greedy output must match exactly. BF16
must match the first 16 greedy tokens; the full 64-token equality is retained
as a diagnostic. This distinction is necessary because the FP16 source
checkpoint is cast to BF16 and small layout-dependent roundoff can flip a
near-tied token after many recurrent layers even while final-logit cosine stays
above the BF16 release floor.

The original aspirational targets—FP32 normalized `1e-4`, FP16/BF16 cosine
`0.9999`, and FP16 max-absolute logits `0.15`—remain in every case as
non-blocking diagnostics. The calibrated release thresholds are based on the
observed V100/RTX 4080 difference between mathematically identical contiguous
and transposed GEMM layouts; they do not permit non-finite values or a greedy
token mismatch. Per-layer and vectorized traces identify where accumulation
begins and remain diagnostic rather than additional gates.

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

On a two-GPU V100 host, the checked-in launcher partitions the eight tasks,
runs both shards concurrently, merges them, and invokes the same validator:

```bash
MODEL_ROOT=/models/rwkv7-reference \
OUTPUT_DIR="$PWD/results/lm_eval/v0.9.0" \
PYTHON="$VIRTUAL_ENV/bin/python" \
CODE_SHA="$(git rev-parse HEAD)" \
bash evaluation/run_lm_eval_v100_parallel.sh
```

`MODEL_ROOT` must contain `rwkv7_01b_hf`, `rwkv7_04b_hf`, and
`rwkv7_15b_hf` directories or symlinks. The merged release bundle is written
to `$OUTPUT_DIR/merged`; shard logs and manifests remain beside it for audit
and resumable reruns.

Formal execution never uses `--limit`. Pull requests may set
`--smoke-limit`. Each task is an independent process with raw stdout,
stderr, sample logs, task config and manifest row. Batch 1/8 absolute metric
difference must be at most 0.001; Wikitext perplexity relative difference must
be at most 0.1%. The fixed execution shapes described in
[`ARCHITECTURE.md`](ARCHITECTURE.md#numerical-reproducibility) prevent normal
FP16 GEMM shape selection from changing close multiple-choice decisions.
