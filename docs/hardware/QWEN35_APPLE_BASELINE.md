# Qwen3.5 Apple / mobile baseline

This document defines the first reproducible gate for the goal: **RWKV-7 HF /
Apple MLX / CoreML should beat Qwen3.5 on Apple/mobile deployment metrics**.

The current repository does **not** claim that this gate is complete.  The point
of this lane is to make "beat Qwen3.5" measurable before deeper MLX fused and
CoreML/ANE optimization work starts.

## Public comparison targets

Use same-device, same-prompt-text runs against the public Qwen3.5 MLX/mobile
packages.  The initial public size classes are:

| Baseline | Runtime | Public package size | Use in gate |
|---|---|---:|---|
| `qwen3.5:0.8b-mlx` | Ollama / MLX | 1.2GB | tiny/mobile floor |
| `qwen3.5:2b-mlx` | Ollama / MLX | 3.1GB | 1.5B-ish speed/memory comparison |
| `qwen3.5:4b-mlx` | Ollama / MLX | 4.0GB | 2.9B-ish comparison |
| `qwen3.5:9b-mlx` | Ollama / MLX | 8.9GB | upper mobile/local comparison |

Reference pages:

- [Ollama qwen3.5](https://ollama.com/library/qwen3.5)
- [MollySophia/rwkv-mobile](https://github.com/MollySophia/rwkv-mobile)

## Metrics that count

A row only supports the goal when it records all relevant fields in JSONL:

| Area | Required fields |
|---|---|
| Device | platform, machine, macOS version, memory when available |
| Prompt | prompt case name, prompt character target, actual tokenizer prompt token count when available |
| Generation | requested generated tokens, actual generated tokens, response preview |
| Speed | TTFT if available, prefill tok/s, decode tok/s, wall time |
| Memory | MLX active/peak/cache memory or runtime-native memory telemetry |
| Quant | W8/W4/LUT/INT4 mode, backend, quant min params, fallback/Metal counts when available |
| State cache | chunked prefill max diff, seen-token checks, batch/session backend where applicable |
| Evidence | append-only JSONL path plus command line used to produce it |

## Harness

The shared baseline runner is:

```bash
PYTHONPATH=. python bench/run_qwen35_apple_baseline.py --help
```

For end-to-end local collection, use the one-command wrapper:

```bash
# Dry-run the full Qwen/RWKV/CoreML plan without contacting runtimes.
DRY_RUN=1 \
RWKV_MLX_MODELS=/path/to/rwkv7-g1d-0.4b-hf,/path/to/rwkv7-g1g-1.5b-hf \
COREML_EXPORT_MODELS=/path/to/rwkv7-g1g-1.5b-hf \
scripts/run_qwen35_apple_acceptance.sh

# Live same-device acceptance. Set PULL_QWEN=1 only when you want the wrapper
# to pull Qwen through the local Ollama HTTP API before collecting rows. Pulls
# are bounded by OLLAMA_PULL_TIMEOUT_S and OLLAMA_PULL_IDLE_TIMEOUT_S so a stuck
# `pulling manifest` / no-byte-progress registry request records a structured
# failure instead of hanging the whole acceptance run forever.
PULL_QWEN=1 \
RWKV_MLX_MODELS=/path/to/rwkv7-g1d-0.4b-hf,/path/to/rwkv7-g1g-1.5b-hf \
COREML_EXPORT_MODELS=/path/to/rwkv7-g1g-1.5b-hf \
RESULTS=bench/results_qwen35_apple_baseline.jsonl \
scripts/run_qwen35_apple_acceptance.sh
```

The wrapper runs `bench/run_qwen35_apple_baseline.py`, optionally runs
`scripts/export_rwkv7_coreml.py`, optionally emits CoreML runtime/plan rows via
`bench/run_coreml_apple_baseline.py`, optionally scores response quality via
`bench/score_qwen35_quality.py`, then appends
`bench/compare_qwen35_apple_baseline.py` gate rows.  The default comparison
pairs cover the currently available 0.4B/1.5B RWKV classes; override `PAIRS`,
`QWEN_MODELS`, and `RWKV_MLX_MODELS` for 4B/9B or distilled-mobile gates.

It emits rows with `axis=qwen35_apple_baseline` and can run:

1. Qwen3.5 through a local Ollama server using the streaming `/api/generate`
   endpoint.
2. Qwen3.5 through Hugging Face `mlx-community/*-MLX-4bit` models using the
   optional `mlx-vlm` runtime.  This is the fallback path when Ollama model
   pulls are unavailable or stuck.
3. RWKV-7 through this repository's optional MLX recurrent backend.
4. CoreML/ANE rows in the same schema once the CoreML runtime runner lands.

The companion export entry point is `scripts/export_rwkv7_coreml.py`; the companion runtime row generator is `bench/run_coreml_apple_baseline.py`.  It writes
a reproducible CoreML export manifest in `--dry-run` mode on any machine, and
attempts a first `full-logits` `.mlpackage` export when `coremltools`, `torch`,
and `transformers` are installed.  Stateful `decode`/`prefill` CoreML functions
remain follow-up work and are tracked in the manifest rather than claimed as
complete.

Dry-run the matrix without contacting runtimes:

```bash
PYTHONPATH=. python bench/run_qwen35_apple_baseline.py \
  --dry-run \
  --prompt-target-chars 1024,4096 \
  --decode-lengths 128,512 \
  --qwen-models qwen3.5:0.8b-mlx,qwen3.5:2b-mlx,qwen3.5:4b-mlx \
  --rwkv-mlx-models /path/to/rwkv7-g1d-0.4b-hf,/path/to/rwkv7-g1g-1.5b-hf \
  --results bench/results_qwen35_apple_baseline.jsonl
```

Run Qwen3.5 baselines after pulling models into Ollama.  For unattended runs,
prefer `scripts/ollama_pull_with_timeout.py` over the raw CLI spinner:

```bash
PYTHONPATH=. python scripts/ollama_pull_with_timeout.py \
  qwen3.5:0.8b-mlx \
  --host http://127.0.0.1:11434 \
  --timeout-s 7200 \
  --idle-timeout-s 120 \
  --results bench/results_qwen35_apple_baseline.jsonl

PYTHONPATH=. python scripts/ollama_pull_with_timeout.py \
  qwen3.5:2b-mlx \
  --host http://127.0.0.1:11434 \
  --timeout-s 7200 \
  --idle-timeout-s 120 \
  --results bench/results_qwen35_apple_baseline.jsonl

PYTHONPATH=. python bench/run_qwen35_apple_baseline.py \
  --prompt-target-chars 1024,4096,8192 \
  --decode-lengths 128,512 \
  --qwen-models qwen3.5:0.8b-mlx,qwen3.5:2b-mlx,qwen3.5:4b-mlx \
  --rwkv-mlx-models '' \
  --results bench/results_qwen35_apple_baseline.jsonl
```

Run Qwen3.5 directly from Hugging Face MLX/VLM weights when Ollama is blocked:

```bash
# Install the optional runtime in the Apple environment first.
python -m pip install mlx-vlm

# If your machine needs a local proxy for HF large files, export it before the
# run; the runner leaves network policy to the caller.
export HTTP_PROXY=http://127.0.0.1:7897
export HTTPS_PROXY=http://127.0.0.1:7897
export ALL_PROXY=http://127.0.0.1:7897

PYTHONPATH=. python bench/run_qwen35_apple_baseline.py \
  --prompt-target-chars 1024,4096 \
  --decode-lengths 128,512 \
  --qwen-models '' \
  --qwen-mlx-vlm-models mlx-community/Qwen3.5-0.8B-MLX-4bit \
  --rwkv-mlx-models '' \
  --results bench/results_qwen35_apple_baseline.jsonl
```

If `mlx-vlm` text streaming hits a tokenizer/detokenizer
`UnicodeDecodeError`, or when the gate only needs generated-token speed/memory,
use the token-only lane:

```bash
PYTHONPATH=. python bench/run_qwen35_apple_baseline.py \
  --prompt-target-chars 1024,4096 \
  --decode-lengths 128,512 \
  --qwen-models '' \
  --qwen-mlx-vlm-models mlx-community/Qwen3.5-0.8B-MLX-4bit \
  --qwen-mlx-vlm-token-only \
  --rwkv-mlx-models '' \
  --results bench/results_qwen35_apple_baseline.jsonl
```

The one-command wrapper exposes the same path through
`QWEN_MLX_VLM_TOKEN_ONLY=1`.

The MLX/VLM rows use `engine=mlx_vlm`, `runtime=mlx_vlm`, and the same
`prefill_tok_s`, `decode_tok_s`, `ttft_s`, response, and MLX peak-memory fields
as the rest of the `qwen35_apple_baseline` matrix.  Known public model ids are
`mlx-community/Qwen3.5-0.8B-MLX-4bit`,
`mlx-community/Qwen3.5-4B-MLX-4bit`, and
`mlx-community/Qwen3.5-9B-MLX-4bit`.
Token-only rows use `runtime=mlx_vlm_token_only`, keep the same speed/memory
fields, and intentionally leave response text empty.

The first local MLX/VLM smoke is recorded in
[`../../bench/apple_qwen35_mlx_vlm_m5_20260707/`](../../bench/apple_qwen35_mlx_vlm_m5_20260707/).
On the Apple M5 smoke row, RWKV-7 0.4B/mm4 beats the Qwen3.5 0.8B MLX-4bit row
on TTFT, prefill, and peak memory, while decode is still below the configured
1.0x gate (`decode_ratio_rwkv_over_qwen=0.721342`), so the next engineering
action remains decode-kernel/batching optimization.

The follow-up group-quant projection smoke is recorded in
[`../../bench/apple_qwen35_mlx_vlm_group_m5_20260707/`](../../bench/apple_qwen35_mlx_vlm_group_m5_20260707/).
With `RWKV7_MLX_GROUP_RKV_QUANT_PROJECTION=1`, RWKV-7 0.4B/mm4 passes this
short 0.8B comparison gate (`decode_ratio_rwkv_over_qwen=1.052232`,
`prefill_ratio_rwkv_over_qwen=3.899691`, `memory_ratio_rwkv_over_qwen=0.671598`).

The expanded `512 chars / 64 tokens` token-only smoke is recorded in
[`../../bench/apple_qwen35_08b_tokenonly_m5_20260707/`](../../bench/apple_qwen35_08b_tokenonly_m5_20260707/).
This stronger baseline shows RWKV-7 0.4B/mm4 still passing memory
(`memory_ratio_rwkv_over_qwen=0.576421`) but missing speed/latency gates
(`decode_ratio_rwkv_over_qwen=0.256284`,
`prefill_ratio_rwkv_over_qwen=0.039915`, `ttft_ratio_rwkv_over_qwen=26.245268`).
The next production-performance work therefore remains fused decode, faster
prefill/chunked prefill, and TTFT reduction.

Run RWKV-7 MLX rows against the same prompt text:

```bash
PYTHONPATH=. python bench/run_qwen35_apple_baseline.py \
  --prompt-target-chars 1024,4096,8192 \
  --decode-lengths 128,512 \
  --qwen-models '' \
  --rwkv-mlx-models /path/to/rwkv7-g1d-0.4b-hf,/path/to/rwkv7-g1g-1.5b-hf \
  --rwkv-dtype fp16 \
  --rwkv-quantization none \
  --rwkv-wkv-backend metal \
  --rwkv-chunk-size 2048 \
  --results bench/results_qwen35_apple_baseline.jsonl
```

Run RWKV-7 W4/Metal rows:

```bash
RWKV7_MLX_GROUP_RKV_QUANT_PROJECTION=1 \
PYTHONPATH=. python bench/run_qwen35_apple_baseline.py \
  --prompt-target-chars 1024,4096,8192 \
  --decode-lengths 128,512 \
  --qwen-models '' \
  --rwkv-mlx-models /path/to/rwkv7-g1d-0.4b-hf,/path/to/rwkv7-g1g-1.5b-hf \
  --rwkv-dtype fp16 \
  --rwkv-quantization mm4 \
  --rwkv-quant-min-params 4000000 \
  --rwkv-quant-backend auto \
  --rwkv-wkv-backend metal \
  --rwkv-chunk-size 2048 \
  --results bench/results_qwen35_apple_baseline.jsonl
```

Run CoreML runtime rows from an export manifest:

```bash
# Plan rows only; safe without CoreMLTools or an .mlpackage.
PYTHONPATH=. python bench/run_coreml_apple_baseline.py \
  --manifest exports/rwkv7-g1g-1.5b-coreml/coreml_export_manifest.json \
  --dry-run \
  --prompt-target-chars 1024,4096 \
  --decode-lengths 128,512 \
  --results bench/results_qwen35_apple_baseline.jsonl

# Live full-logits runtime smoke. Rows are marked partial until stateful
# decode/prefill lands, so comparison gates will not treat them as wins.
PYTHONPATH=. python bench/run_coreml_apple_baseline.py \
  --manifest exports/rwkv7-g1g-1.5b-coreml/coreml_export_manifest.json \
  --compute-units cpu-and-ne \
  --results bench/results_qwen35_apple_baseline.jsonl
```

Plan and smoke a CoreML package export:

```bash
# Import-safe plan: no CoreMLTools required.
PYTHONPATH=. python scripts/export_rwkv7_coreml.py \
  /path/to/rwkv7-g1g-1.5b-hf \
  exports/rwkv7-g1g-1.5b-coreml \
  --dry-run \
  --chunks 4 \
  --prefill-seq-length 2048 \
  --sample-seq-length 128 \
  --state-mode wkv-coreml \
  --quantization lut4 \
  --results bench/results_qwen35_apple_baseline.jsonl

# Live first-step export when CoreMLTools is installed.
PYTHONPATH=. python scripts/export_rwkv7_coreml.py \
  /path/to/rwkv7-g1g-1.5b-hf \
  exports/rwkv7-g1g-1.5b-coreml \
  --sample-seq-length 128 \
  --deployment-target iOS18 \
  --compute-units cpu-and-ne \
  --quantization lut4 \
  --results bench/results_qwen35_apple_baseline.jsonl
```

The export row uses `axis=rwkv7_coreml_export`.  `status=plan` only records the
manifest/contract; `status=pass` means a `.mlpackage` was produced.  A CoreML
export row is **not** a Qwen3.5 performance win until a later CoreML runner adds
TTFT, prefill/decode tok/s, memory, and correctness fields to the
`qwen35_apple_baseline` matrix.

Summarize an existing result file:

```bash
PYTHONPATH=. python bench/run_qwen35_apple_baseline.py \
  --summarize bench/results_qwen35_apple_baseline.jsonl
```

Score quality rows after collecting responses:

```bash
# Collect full response text/token ids first.
STORE_RESPONSES=1 \
QUALITY_RUBRIC=docs/hardware/qwen35_quality_rubric.example.json \
scripts/run_qwen35_apple_acceptance.sh

# Or score an existing JSONL file directly.
PYTHONPATH=. python bench/score_qwen35_quality.py \
  --results bench/results_qwen35_apple_baseline.jsonl \
  --rubric docs/hardware/qwen35_quality_rubric.example.json \
  --pair qwen3.5:0.8b-mlx=rwkv7-g1d-0.4b-hf \
  --append bench/results_qwen35_apple_baseline.jsonl
```

Quality rows use `axis=qwen35_apple_quality`; pairwise quality comparisons use
`axis=qwen35_apple_quality_comparison`.  Missing full `response_text` is
reported as `unknown`, so quality parity cannot be claimed from truncated
previews.

Compare RWKV rows against Qwen3.5 rows and emit explicit gate results:

```bash
PYTHONPATH=. python bench/compare_qwen35_apple_baseline.py   --results bench/results_qwen35_apple_baseline.jsonl   --pair qwen3.5:0.8b-mlx=rwkv7-g1d-0.4b-hf   --pair qwen3.5:2b-mlx=rwkv7-g1g-1.5b-hf   --min-decode-ratio 1.0   --require-prefill   --require-ttft   --max-ttft-ratio 1.1   --diagnostics   --append bench/results_qwen35_apple_baseline.jsonl
```

The comparison rows use `axis=qwen35_apple_baseline_comparison`; optional
`--diagnostics` rows use `axis=qwen35_apple_baseline_gap_diagnostic`; the
summary row uses `axis=qwen35_apple_baseline_comparison_summary`.  Missing
required metrics produce `status=unknown`, not `pass`, so a PR cannot claim a
Qwen3.5 win from an incomplete row.  Diagnostic rows translate missing/failing
gates into concrete actions such as `collect_qwen_baseline_rows`,
`collect_memory_telemetry`, `optimize_decode_kernel_or_batching`, or
`reduce_peak_memory_or_quantize_more`; `scripts/run_qwen35_apple_acceptance.sh`
enables these rows by default with `COMPARE_DIAGNOSTICS=1`.

## Initial acceptance matrix

| RWKV target | Qwen3.5 comparator | Runtime gate | Current status |
|---|---|---|---|
| RWKV-7 0.4B W4/MLX | `qwen3.5:0.8b-mlx` | lower memory and higher decode tok/s at prompt 1k/4k/8k, decode 128/512 | needs same-device rows |
| RWKV-7 1.5B W4/MLX | `qwen3.5:2b-mlx` | lower memory and higher or equal decode tok/s; TTFT no worse by >10% | needs same-device rows |
| RWKV-7 2.9B W4/MLX/CoreML | `qwen3.5:4b-mlx` | lower memory and higher decode tok/s | CoreML export prototype exists; ANE runtime rows not landed |
| RWKV-7 larger / distilled mobile | `qwen3.5:9b-mlx` | mobile-useful memory envelope plus quality eval | requires model/quality work |

## CoreML / ANE follow-up

`rwkv-mobile` shows the right production direction for mobile Apple devices:

- separate `decode` and `prefill` CoreML functions
- chunked model export
- CoreML state / tensor state / WKV-CoreML state variants
- async prefill loading
- int8 / int4 / LUT quantization

The repository now has `scripts/export_rwkv7_coreml.py` as the first
CoreML/ANE bridge.  It records the intended chunking, state mode, quantization,
and deployment target, supports a first `full-logits` export, and deliberately
keeps stateful decode/prefill marked as unimplemented in the manifest.

The next repository lane should add:

1. HF RWKV-7 -> Torch traced decode/prefill -> CoreML multifunction export.
2. CoreML correctness checks against HF/MLX logits and state.
3. CoreML W4/LUT/INT4 export rows in the same `qwen35_apple_baseline` schema.
4. A CoreML runtime benchmark runner that emits TTFT, prefill/decode tok/s,
   memory, chunked-prefill correctness, and state-cache reuse rows.
5. iPhone/iPad rows once device access is available.

## Non-goals for the first baseline PR

- It does not claim final quality superiority over Qwen3.5.
- It does not implement final stateful CoreML decode/prefill yet; the current CoreML path is an export prototype and manifest contract.
- It does not mark W8/W4 as fp16-beating until JSONL evidence proves it.
- It does not replace the existing Apple MLX session and quant regression tests.
