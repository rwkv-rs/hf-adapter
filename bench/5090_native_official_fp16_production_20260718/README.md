# RTX 5090 Native versus official fp16-state production evidence

This artifact records the exact-card Native/no-FLA inference lane that uses the
same FP16 weights, recurrent state, inputs, and outputs as pinned official
RWKV-Gradio-3 v3a. It covers cached decode on official g1h 7.2B and true
sequence prefill on official g1h 2.9B and 13.3B. It does not compare against a
Torch fallback.

## Result

| Axis | Exact scope | Native / official | Correctness | Status |
|---|---|---:|---|---|
| Cached decode | g1h 7.2B, B1, prompt 8, decode 512 | `146.42 / 146.277 tok/s` (`1.00098x`) | one trace across 3 Native runs; tensor oracle passes | PASS |
| Cached decode | g1h 7.2B, B8, prompt 8, decode 512 | `899.51 / 890.21 tok/s` (`1.01045x`) | one trace across 3 Native runs; tensor oracle passes | PASS |
| Sequence prefill | g1h 2.9B, B1/B8, P128/512/2048 | `1.01475x–1.56896x` | 6/6 tensor/state/token gates | PASS |
| Sequence prefill | g1h 13.3B, B1/B8, P128/512/2048 | `1.00285x–1.07347x` | 6/6 tensor/state/token gates | PASS |

The 7.2B 16-step decode oracle reports logits cosine
`0.9999996674/0.9999997850` at B1/B8, exact top-1 `17/17` and `136/136`,
exact greedy `16/16` and `128/128`, and passing prefill/final recurrent-state,
xpa, xpf, and elapsed-position checks.

Every prefill cell checks prompt logits, every layer's final residual output,
FP16 recurrent state, xpa/xpf, the first generated token, and the first cached
decode token. The root [`summary.json`](summary.json) is the compact 12-cell
result. Per-shape reports retain capture SHA256 values and the complete pinned
official source-file manifest.

## Environment and provenance

- GPU: NVIDIA GeForce RTX 5090, SM 12.0, 32,607 MiB.
- Driver: `595.58.03`; OS: Ubuntu 22.04.5 LTS.
- PyTorch: `2.11.0+cu128`; CUDA runtime: `12.8`; Transformers: `5.12.1`;
  Triton: `3.6.0`.
- Native promoted policy commit: `dcc53fc20a1ec3e302e281d6cb6e294ae2549d72`.
- Official source: RWKV-Gradio-3 commit
  `cc57df475465c6cacd42ecd4f2f05a588ee5473b`, verified by the source hashes
  embedded in each report.
- Precision: `fp16_state_fp16_io` on both engines.
- Checkpoints:
  `rwkv7-g1h-2.9b-20260710-ctx10240`,
  `rwkv7-g1h-7.2b-20260710-ctx10240`, and
  `rwkv7-g1h-13.3b-20260710-ctx10240`.

The selected 2.9B/13.3B reports were collected while tuning the policy and
retain their exact Native revisions and explicit runtime environment. The
`g1h-13.3b/default_policy/` replay runs commit `dcc53fc` without any
`RWKV7_*` override and proves that the promoted policy selects the accepted
route: quality and performance both pass 6/6, with throughput
`1.00270x–1.07336x` official. The 13.3B B8/P2048 route is graph-off because
graph capture exceeds the 32 GiB card. Its pre-envelope failure and comparison
log are retained. Four pinned-official processes show first-decode max absolute
variation `0.203125`, while prompt logits, layer outputs, recurrent state,
xpa, and xpf are bit-identical across repeats; the final shape-bound comparison
passes with multiplier `1.25`.

## Pass gates

Decode passes only when all of the following hold:

- three Native repeats have one complete greedy-trace hash;
- Native median throughput is no lower than the pinned official value at both
  B1 and B8;
- logits cosine is at least `0.9999`, logits max absolute error is at most
  `0.125`, and every compared top-1/greedy token matches;
- prefill and final recurrent state, xpa/xpf, elapsed state, precision, and
  requested CUDA-extension activity pass.

Prefill passes only when Native throughput is no lower than official and every
fixed tensor/state/token gate passes. A pinned-official self-repeat envelope
may be used only for first-decode logits when official v3a itself varies across
fresh processes. It is shape-bound, recorded in the report, and multiplied by
only `1.25`. Prompt logits, layer outputs, recurrent state, xpa/xpf, and token
equality always retain fixed thresholds. The pre-envelope failed report is
preserved when this rule is exercised.

## Reproduce

Install the CUDA extra so Ninja is present and extension failures are fatal:

```bash
python -m pip install -e '.[cuda]'
export PYTHONPATH="$PWD"
export NATIVE_EXT=/path/to/native-extension-cache
export OFFICIAL_EXT=/path/to/official-extension-cache
export OFFICIAL_DIR=/path/to/RWKV-Gradio-3
export OFFICIAL_MANIFEST=/path/to/official_source_manifest.json
```

Run the Native decode row with the repository default policy:

```bash
TORCH_EXTENSIONS_DIR="$NATIVE_EXT" \
python bench/bench_native_model_decode.py \
  --hf-dir /models/rwkv7-g1h-7.2b-hf \
  --dtype fp16 --batch-sizes 1 8 --prompt-tokens 8 --decode-steps 512 \
  --warmup 8 --repetitions 3 --fast-token-api \
  --backends native_graph --timing-scope end_to_end \
  --require-active-extensions --results native_decode.jsonl
```

Run a same-precision tensor oracle in separate processes so only one 7.2B
model is resident at a time:

```bash
python scripts/compare_official_native_inference.py capture-native \
  --hf-dir /models/rwkv7-g1h-7.2b-hf --output native.pt \
  --prompt-tokens 8 --decode-steps 16 --batch-sizes 1 8 \
  --native-source-revision dcc53fc

TORCH_EXTENSIONS_DIR="$OFFICIAL_EXT" \
python scripts/compare_official_native_inference.py capture-official \
  --hf-dir /models/rwkv7-g1h-7.2b-hf --output official.pt \
  --prompt-tokens 8 --decode-steps 16 --batch-sizes 1 8 \
  --official-dir "$OFFICIAL_DIR" \
  --official-model /models/rwkv7-g1h-7.2b-20260710-ctx10240.pth \
  --official-commit cc57df475465c6cacd42ecd4f2f05a588ee5473b \
  --official-source-manifest "$OFFICIAL_MANIFEST" \
  --official-emb cpu --official-batched-rkv off \
  --official-cmix-sparse no-fc --official-lowrank-weight transpose \
  --official-orig-linear-groups none

python scripts/compare_official_native_inference.py compare \
  --native native.pt --official official.pt \
  --official-commit cc57df475465c6cacd42ecd4f2f05a588ee5473b \
  --output alignment.json
```

Run the six-cell prefill matrix for one checkpoint. Change both model paths to
repeat the other checkpoint:

```bash
python bench/run_official_native_prefill_matrix.py \
  --hf-dir /models/rwkv7-g1h-13.3b-hf \
  --official-dir "$OFFICIAL_DIR" \
  --official-model /models/rwkv7-g1h-13.3b-20260710-ctx10240.pth \
  --official-source-manifest "$OFFICIAL_MANIFEST" \
  --official-emb cpu --official-batched-rkv off \
  --official-cmix-sparse no-fc --official-lowrank-weight transpose \
  --official-orig-linear-groups none \
  --cases 1x128,1x512,1x2048,8x128,8x512,8x2048 \
  --warmup 10 --repeats 21 --native-source-revision dcc53fc \
  --native-torch-extensions-dir "$NATIVE_EXT" \
  --official-torch-extensions-dir "$OFFICIAL_EXT" \
  --output-dir prefill-default-policy --skip-existing
```

The observable PASS is `status: "pass"`, `quality_pass_cases: 6`, and
`performance_pass_cases: 6` in the matrix summary. Do not infer PASS from a
zero process exit alone.

## Recovery and limits

- If interrupted, rerun the same output directory with `--skip-existing`.
  Never merge captures from a different source revision, checkpoint, shape,
  precision, or official manifest.
- If Ninja or a CUDA extension is missing, install `.[cuda]`, preserve the
  failure log, and rerun. A Torch/eager fallback is not accepted.
- If 13.3B B8/P2048 graph capture runs out of memory, clear the failed process
  and rerun with the exact graph-off policy. Do not present an OOM row as a
  speed result.
- These results validate only the listed RTX 5090 models and shapes. They do
  not prove other Blackwell cards, other batch/context shapes, cross-harness
  memory parity, model quality, training convergence, or universal Native
  superiority.

AI-assisted execution has one canonical entry point: start with task
`inference` in
[`docs/AI_ASSISTED_SETUP.md`](../../docs/AI_ASSISTED_SETUP.md). Production
benchmark promotion still requires an engineer to review the exact-card
artifact and is not part of the ordinary-user AI workflow.
