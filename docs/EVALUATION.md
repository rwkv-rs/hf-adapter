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

The backend-v2 three-way report applies this consistently to operator,
inference, and training validation. Candidate-vs-readable-reference numerical
checks and observed candidate routes are blocking release gates. Results from
the pinned FLA revision are retained in full as `diagnostic-non-blocking` and
never turn a conforming candidate into a release failure. A failed FLA strict
envelope remains a visible failed diagnostic; it is not rewritten as a pass.

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
  --output-dir results/lm_eval/v1.0.0 \
  --device cuda
python evaluation/validate_lm_eval_matrix.py \
  --result-dir results/lm_eval/v1.0.0
```

On a two-GPU V100 host, the checked-in launcher partitions the eight tasks,
runs both shards concurrently, merges them, and invokes the same validator:

```bash
MODEL_ROOT=/models/rwkv7-reference \
OUTPUT_DIR="$PWD/results/lm_eval/v1.0.0" \
PYTHON="$VIRTUAL_ENV/bin/python" \
CODE_SHA="$(git rev-parse HEAD)" \
bash evaluation/run_lm_eval_v100_parallel.sh
```

`MODEL_ROOT` must contain `rwkv7_01b_hf`, `rwkv7_04b_hf`, and
`rwkv7_15b_hf` directories or symlinks. The merged release bundle is written
to `$OUTPUT_DIR/merged`; shard logs and manifests remain beside it for audit
and resumable reruns.

For the full eager reference run, independent units can occupy otherwise idle
V100 capacity without changing any lm_eval command or batch size:

```bash
python evaluation/run_lm_eval_v100_pool.py \
  --model-root /models/rwkv7-reference \
  --output-dir results/lm_eval/v1.0.0 \
  --python "$VIRTUAL_ENV/bin/python" \
  --code-sha "$(git rev-parse HEAD)"
```

The pool runs all 24 batch-one units first with six processes per V100, then
the higher-memory batch-eight units with two per V100. Every unit retains its
own raw command, logs, manifest and result directory before the normal merge
and validation scripts run.

Formal execution never uses `--limit`. Pull requests may set
`--smoke-limit`. Each task is an independent process with raw stdout,
stderr, sample logs, task config and manifest row. Batch 1/8 absolute metric
difference must be at most 0.001; Wikitext perplexity relative difference must
be at most 0.1%. The fixed execution shapes described in
[`ARCHITECTURE.md`](ARCHITECTURE.md#numerical-reproducibility) prevent normal
FP16 GEMM shape selection from changing close multiple-choice decisions.

## Three-way optional-backend lm_eval

The optional-backend release matrix is distinct from the single reference
lane above. It runs `reference`, strict backend-v2 `optimized`, and the pinned
FLA wrapper for 0.1B/0.4B/1.5B, batch 1/8, and all eight tasks: 144 formal
commands in total. All lanes use one immutable HF wheel, one immutable kernel
wheel and FLA commit
`80e494f6c588e091fc8316b612870df29375c5b8`.

Each lane uses the same command shape (shown for the optimized lane):

```bash
python evaluation/run_lm_eval_matrix.py \
  --output-dir results/backend-v2/lm_eval/optimized \
  --lane optimized \
  --optimized-model-impl native \
  --model 0.1b=/models/rwkv7-0.1b-hf \
  --model 0.4b=/models/rwkv7-0.4b-hf \
  --model 1.5b=/models/rwkv7-1.5b-hf \
  --hf-wheel /artifacts/rwkv7_hf-1.0.0-py3-none-any.whl \
  --kernel-wheel /artifacts/rwkv7_kernels-1.0.0-py3-none-any.whl \
  --fla-source /sources/fla-80e494f6 \
  --code-sha "$(git rev-parse HEAD)"
```

Use the clean model directories for `reference`/`optimized` and directories
created by `prepare_fla_lm_eval_model.py` for `fla`. Then run:

```bash
python evaluation/validate_lm_eval_three_way.py \
  --reference-dir results/backend-v2/lm_eval/reference \
  --optimized-dir results/backend-v2/lm_eval/optimized \
  --fla-dir results/backend-v2/lm_eval/fla \
  --output results/backend-v2/lm_eval/validation.json \
  --require-model-routes
```

The validator rejects requested-route labels without actual backend-v2 trace
counts. It also rejects different wheel hashes, FLA revisions, weights,
tokenizer/vocabulary/template payloads, datasets, or harness SHAs. For
classification it reconstructs each raw and normalized selected answer;
LAMBADA compares greedy-continuation outcomes; Wikitext compares per-document
rolling NLL/word/byte counts and aggregate NLL/PPL at the 0.1% relative gate.
Raw samples remain outside Git; only compact manifests, hashes, commands and
validation summaries are committed.

## Compact release evidence

After a device result root passes, build its reviewable Git artifact without
copying samples, lm_eval result payloads, stdout/stderr logs, checkpoints,
weights, wheels, W&B runtime files, or model directories:

```bash
python evaluation/build_backend_v2_compact_bundle.py \
  --input-dir /results/backend-v2/4080-final \
  --output-dir results/backend-v2/4080-final-compact \
  --device rtx-4080 \
  --harness-sha "$(git rev-parse HEAD)"
```

The builder retains JSON/JSONL summaries and manifests, resolved configs,
environment reports, command lines, exit codes and small text evidence. It
rejects symlinks, oversized evidence, known token forms and output directories
inside the raw result tree. `BUNDLE.json` records the filtering policy and
builder SHA; `MANIFEST.sha256` covers every other bundled file and is verified
before the directory is published.

## Optional training comparison

Training comparison never substitutes a second model class. The reference,
adaptive, and FLA lanes load their documented model paths while retaining the
same readable HF layer loop. `adaptive` asks API v4 for one atomic certificate;
the current full-model fast domain is dense B4/T128 with zero initial state,
fully active aligned tokens, gradient-bearing inputs, and head size 64. The
certificate covers the factorized recurrent, bounded FFN-linear, and Mix6
leaves. `matrix` and `factorized` remain isolated recurrent diagnostics.

Formal training evidence must record the model route, all three actual leaf
implementations, the program identity, complete optimizer-gradient vector,
checkpoint replay, and process-wide route counts. Outside the atomic fast
domain, explicitly adaptive leaves must name their exact matrix/reference
fallback; an unprovable autograd boundary must name the complete reference
program. A certified leaf decline or strict optimized request outside the
domain fails closed.

The factorized rows require a CUDA development toolkit matching
`torch.version.cuda`. The kernel wheel installs Ninja, but it deliberately does
not install or guess a system toolkit. Point `CUDA_HOME` at a prefix containing
`bin/nvcc`, and record both values before running the immutable-wheel gate:

```bash
export CUDA_HOME=/opt/cuda
export PATH="$CUDA_HOME/bin:$PATH"
"$CUDA_HOME/bin/nvcc" --version
python -c 'import torch; print(torch.__version__, torch.version.cuda)'
```

If that toolchain is absent, `adaptive` records the reason and uses the exact
matrix leaf; such a row is valid fallback evidence but cannot be counted as a
factorized speed result. `matrix` never requires a compiler.

Leaf-level output, state and gradient comparison:

```bash
RWKV7_BACKEND=auto \
RWKV7_TRAINING_KERNEL_IMPL=adaptive \
python evaluation/validate_recurrent_training.py \
  --candidate adaptive \
  --fla-source /sources/fla-80e494f6 \
  --output results/training/recurrent.json \
  --batch 1 --batch 4 \
  --tokens 16 --tokens 17 --tokens 128 \
  --padding none --padding left --padding right \
  --hf-wheel /artifacts/rwkv7_hf-1.0.0-py3-none-any.whl \
  --kernel-wheel /artifacts/rwkv7_kernels-1.0.0-py3-none-any.whl
```

Full-model logits, causal loss, complete gradient vector, checkpointing and
forward/backward throughput comparison:

```bash
python evaluation/validate_model_training.py \
  --candidate reference \
  --model /models/rwkv7-0.1b-hf \
  --fla-source /sources/fla-80e494f6 \
  --output results/training/model.json \
  --batch 1 --batch 4 \
  --tokens 16 --tokens 17 --tokens 128 \
  --padding none --padding left --padding right \
  --checkpointing off --checkpointing on \
  --hf-wheel /artifacts/rwkv7_hf-1.0.0-py3-none-any.whl \
  --kernel-wheel /artifacts/rwkv7_kernels-1.0.0-py3-none-any.whl
```

For formal full-model HF rows the gate requires the readable
`torch-reference-model-v1` route and no optional training leaf execution.
Explicit `adaptive`, `matrix`, and `factorized` rows remain isolated operator
experiments, not model-route claims. The gate rejects
non-finite tensors, missing gradients, loss/logit/optimizer-vector threshold
failures. The pinned FLA lane remains a complete non-blocking diagnostic. Named
per-parameter diagnostics remain in JSON even when the aggregate
optimizer-vector gate passes. Speed is reported only after correctness is
evaluated and never changes the correctness result.
