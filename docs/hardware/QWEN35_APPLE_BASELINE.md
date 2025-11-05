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

For reproducible prefill rows, the wrapper defaults to `OLLAMA_THINK=0`,
`OLLAMA_KEEP_ALIVE=0`, and `OLLAMA_CACHE_PROMPT=0`. This keeps short thinking
traces out of `response_text` and unloads Ollama after each row so a completed
prompt cannot be reported as near-zero prefill. The runner records both steady
`ttft_s` (load duration removed) and load-inclusive `cold_ttft_s`. Override
these defaults only when deliberately measuring a shared prompt-cache service.
With the default isolated policy it temporarily keeps the model alive long
enough to query official `/api/ps`, records `ollama_loaded_memory_bytes`, and
then explicitly unloads it. This is loaded-runtime memory, not peak memory, so
the strict peak-to-peak gate remains unknown.

The wrapper also defaults `RWKV_PREFILL_EVAL_INTERVAL=2`. This batches two
lazy MLX recurrent prompt steps between graph evaluations. The reusable model
API keeps the safer interval-1 default. Before changing this value on a new
model/device, run `scripts/mlx_prefill_eval_interval_bench.py`; it treats
logits, all recurrent/cache tensors, seen-token count, and next-token parity as
a hard gate rather than inferring correctness from throughput alone.

The DPLR model prefill path is deliberately opt-in. Set
`RWKV_PREFILL_BACKEND=auto`, `RWKV_DPLR_MIN_TOKENS=128`, and
`RWKV_DPLR_CHUNK_SIZE=64` to use recurrent prefill for short prompts and the
layer-major Metal DPLR route for longer prompts. Keep the default
`RWKV_PREFILL_BACKEND=recurrent` for production claims until long-context,
peak-memory, decode-after-prefill, quality, and cross-device gates pass.

It emits rows with `axis=qwen35_apple_baseline` and can run:

1. Qwen3.5 through a local Ollama server using the streaming `/api/generate`
   endpoint.
2. Qwen3.5 through Hugging Face `mlx-community/*-MLX-4bit` models using the
   optional `mlx-vlm` runtime when Ollama is unavailable.
3. RWKV-7 through this repository's optional MLX recurrent backend.
4. RWKV-7 through the stateful CoreML multifunction runner; confirmed ANE
   placement remains a separate gate.

The companion export entry point is `scripts/export_rwkv7_coreml.py`; the companion runtime row generator is `bench/run_coreml_apple_baseline.py`.  It writes
a reproducible CoreML export manifest in `--dry-run` mode on any machine. With
`--export-kind stateful-multifunction` it exports masked `prefill` and one-token
`decode` functions with packed RWKV Core ML state. The runtime records state
transfer, chunk-boundary drift, HF greedy parity, TTFT, throughput, package
bytes, and peak process memory.

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
  --repeat 1 \
  --warmup-repeats 1 \
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
`mlx-community/Qwen3.5-2B-MLX-4bit`,
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

The 2B-size token-only row is recorded in
[`../../bench/apple_qwen35_2b_tokenonly_m5_20260707/`](../../bench/apple_qwen35_2b_tokenonly_m5_20260707/).
The current goal-level audit over the 0.8B and 2B token-only evidence is
recorded in
[`../../bench/apple_qwen35_goal_audit_m5_20260707/`](../../bench/apple_qwen35_goal_audit_m5_20260707/).
It marks the overall Apple/Qwen3.5 goal as non-passing: the available rows cover
same-prompt Qwen/RWKV MLX, W4 quantization, and chunked/state-cache checks for
0.8B/2B, but still require speed/latency closure, response-quality rows,
long-context rows, stateful CoreML decode/prefill runtime rows, and 4B/9B tier
coverage.

The first explicit 0.8B long-context row is recorded in
[`../../bench/apple_qwen35_08b_longctx_m5_20260707/`](../../bench/apple_qwen35_08b_longctx_m5_20260707/).
On the same Apple M5 `4096 chars / 128 tokens` token-only shape, RWKV-7 0.4B/mm4
+ grouped R/K/V quant + fused FFN key/relu² passes the memory gate
(`memory_ratio_rwkv_over_qwen=0.180759`) and fills the long-context audit slot,
but it remains far behind the Qwen3.5 0.8B MLX-4bit token-only baseline on
speed/latency (`decode_ratio_rwkv_over_qwen=0.474667`,
`prefill_ratio_rwkv_over_qwen=0.034954`, `ttft_ratio_rwkv_over_qwen=30.184709`).
This makes the next Apple performance target concrete: roughly `2.11x` decode
and `28.61x` prefill speedup are needed on this long-context 0.8B tier before a
Qwen3.5-over-Apple claim can pass.

The chunked-prefill state-only follow-up is recorded in
[`../../bench/apple_mlx_chunked_state_only_m5_20260707/`](../../bench/apple_mlx_chunked_state_only_m5_20260707/).
It adds a production-shaped MLX seam where non-final chunks update only the RWKV
recurrent state and skip final logits/lm_head; the same long-context row records
`chunked_state_only_prefill_calls=2`, `chunked_state_only_prefill_tokens=1024`,
and `chunked_prefill_max_abs=0.0`.  This removes two unnecessary chunk-boundary
logits projections at `chunk_size=512`; it is a correctness/dispatch cleanup,
not the main prefill-gap solution.  The remaining bottleneck is still the
per-token/per-layer recurrent WKV and projection launch count.

The decode synchronization cleanup and attention-mix probe are recorded in
[`../../bench/apple_mlx_decode_sync_m5_20260707/`](../../bench/apple_mlx_decode_sync_m5_20260707/).
The baseline harness no longer adds an extra `mx.eval(logits)` after MLX
`prefill()` / `decode_step()` / `chunked_prefill()` because those paths already
synchronize returned logits and recurrent state; decode timing now waits for the
streaming `next_token` sync instead.  The optional `RWKV7_MLX_FUSED_ATTN_MIX=1`
seam fuses the six attention mix tensors into one Metal kernel and records
`fused_attn_mix_counts`, but the 512/64 AB row keeps it disabled by default
because decode regressed while prefill improved only modestly.

The first multi-token WKV scan prototype is recorded in
[`../../bench/apple_mlx_wkv_scan_m5_20260707/`](../../bench/apple_mlx_wkv_scan_m5_20260707/).
It adds `rwkv7_hf.mlx_scan.wkv_scan()`, a Metal recurrent scan over
`r/w/v/k/kk/a [B,T,H,N]` for one layer.  In the isolated Apple M5 WKV microbench,
the scan kernel is `2.49x` faster than a per-token Metal WKV loop at `T=32` and
`4.09x` faster at `T=128`.  This is the first kernel with the right shape to
attack the long-context prefill gap, but it is not yet wired into full MLX
prefill; the next step is a layer-major prefill path with full-model parity and
Qwen3.5 end-to-end evidence.
The component-profile follow-up is recorded in
[`../../bench/apple_mlx_component_profile_m5_20260707/`](../../bench/apple_mlx_component_profile_m5_20260707/).
It uses synchronized component boundaries on the same Apple M5 1.5B/mm4 path and
shows the top profiled buckets as FFN step ≈39.46%, attention/WKV step ≈33.15%,
attention layernorm ≈18.54%, and FFN layernorm ≈6.83%.  Treat this as a fusion
ranking, not an end-to-end speed row: the next Apple MLX kernel work should
prioritize FFN step fusion plus attention/norm fusion before spending effort on
final logits.

The first positive FFN fusion seam is recorded in
[`../../bench/apple_mlx_fused_ffn_relu2_m5_20260707/`](../../bench/apple_mlx_fused_ffn_relu2_m5_20260707/).
`RWKV7_MLX_FUSED_FFN_KEY_RELU2=1` fuses the MM4 FFN key projection and `relu²`
activation into one Metal kernel.  On the same 1.5B/mm4 `512 chars / 64 tokens`
smoke it keeps the generated preview identical, keeps chunked prefill exact, and
improves prefill/decode/TTFT by about `1.05x` / `1.026x` / `1.05x`.  This is a
real speed seam but not yet enough to close the Qwen3.5 2B gap; the one-command
Apple acceptance wrapper enables it by default while the base model keeps the
feature opt-in through the environment variable.

The Qwen3.5 2B MLX-4bit snapshot can stall during the large Xet-backed weight
file after small metadata files have downloaded; `scripts/hf_parallel_download.py`
provides a bounded, resumable HTTP Range fallback for the single
`model.safetensors` shard.  On the same `512 chars / 64 tokens` row, RWKV-7
1.5B/mm4 + RKV quant is runnable and passes the memory gate
(`memory_ratio_rwkv_over_qwen=0.606417`) but remains below the Qwen3.5 2B token
baseline on speed (`decode_ratio_rwkv_over_qwen=0.242215`,
`prefill_ratio_rwkv_over_qwen=0.036051`, `ttft_ratio_rwkv_over_qwen=29.082024`).
The same harness now supports `--warmup-repeats` so Apple rows can separate
MLX/Metal compile cold-start from steady-state generation.  With
`--warmup-repeats 1`, Qwen3.5 2B reaches ≈1205.58 prefill tok/s and ≈110.63
decode tok/s, while RWKV-7 1.5B/mm4 + RKV quant reaches ≈42.84 prefill tok/s
and ≈31.70 decode tok/s with `wkv_backend_counts={"metal":4728}` and grouped
R/K/V quant `fallback=0`.  The warmed row improves RWKV absolute decode but still
needs about `3.49x` decode and `28.14x` prefill speedup to match the warmed
Qwen3.5 2B baseline.

Run RWKV-7 MLX rows against the same prompt text:

```bash
PYTHONPATH=. python bench/run_qwen35_apple_baseline.py \
  --prompt-target-chars 1024,4096,8192 \
  --decode-lengths 128,512 \
  --repeat 1 \
  --warmup-repeats 1 \
  --qwen-models '' \
  --rwkv-mlx-models /path/to/rwkv7-g1d-0.4b-hf,/path/to/rwkv7-g1g-1.5b-hf \
  --rwkv-dtype fp16 \
  --rwkv-quantization none \
  --rwkv-wkv-backend metal \
  --rwkv-chunk-size 2048 \
  --rwkv-prefill-eval-interval 2 \
  --results bench/results_qwen35_apple_baseline.jsonl
```

Run RWKV-7 W4/Metal rows:

```bash
RWKV7_MLX_GROUP_RKV_QUANT_PROJECTION=1 \
PYTHONPATH=. python bench/run_qwen35_apple_baseline.py \
  --prompt-target-chars 1024,4096,8192 \
  --decode-lengths 128,512 \
  --repeat 1 \
  --warmup-repeats 1 \
  --qwen-models '' \
  --rwkv-mlx-models /path/to/rwkv7-g1d-0.4b-hf,/path/to/rwkv7-g1g-1.5b-hf \
  --rwkv-dtype fp16 \
  --rwkv-quantization mm4 \
  --rwkv-quant-min-params 4000000 \
  --rwkv-quant-rkv-min-params 0 \
  --rwkv-quant-backend auto \
  --rwkv-wkv-backend metal \
  --rwkv-chunk-size 2048 \
  --rwkv-prefill-eval-interval 2 \
  --results bench/results_qwen35_apple_baseline.jsonl
```

`--rwkv-quant-rkv-min-params 0` is the Apple grouped-projection knob: it keeps
the general quantization threshold for FFN/lm_head policy, but additionally
quantizes attention `r_proj`/`k_proj`/`v_proj` so
`RWKV7_MLX_GROUP_RKV_QUANT_PROJECTION=1` can hit the fused MLX/Metal R/K/V path
instead of recording grouped fallbacks.  The wrapper defaults
`RWKV_QUANT_RKV_MIN_PARAMS=0`; set it to `-1` to preserve the historical single
`RWKV_QUANT_MIN_PARAMS` threshold.

The first M5 evidence for this activation is recorded in
[`../../bench/apple_rkv_quant_min_m5_20260707/`](../../bench/apple_rkv_quant_min_m5_20260707/).
On the `512 chars / 64 tokens` 0.4B/mm4 row, the direct R/K/V quant path records
`group_rkv_quant_projection_counts={"metal":7920,"fallback":0}`, lowers peak
memory from the earlier ≈514.8MB row to ≈402.2MB, and improves prefill to
≈69.06 tok/s.  It does not close the Qwen3.5 speed gap yet; decode remains
≈0.23x of the Qwen3.5 0.8B MLX token-only row, so the next work remains deeper
decode/WKV/projection fusion.

`RWKV7_MLX_STEP_EVAL_INTERVAL` controls how often the MLX recurrent loop forces
state evaluation.  The model default is `1` for historical behavior; the
Qwen3.5 Apple acceptance wrapper now defaults to `8`.  The first M5 smoke in
[`../../bench/apple_step_eval_interval_m5_20260707/`](../../bench/apple_step_eval_interval_m5_20260707/)
improved the `512 chars / 64 tokens` 0.4B/mm4 direct R/K/V row from
≈69.06/50.51 prefill/decode tok/s to ≈76.91/58.36 tok/s at essentially the same
peak memory and with chunked/full prefill `max_abs=0.0`.  The follow-up 1.5B/mm4
fused-FFN sweep in
[`../../bench/apple_step_eval_interval_15b_m5_20260707/`](../../bench/apple_step_eval_interval_15b_m5_20260707/)
showed interval 8 as the best prefill/TTFT point (`29.72` prefill tok/s,
`4.48s` TTFT) while still improving decode over interval 2.

The CoreML stateful-contract follow-up is recorded in
[`../../bench/apple_coreml_state_contract_m5_20260707/`](../../bench/apple_coreml_state_contract_m5_20260707/).
The export manifest now contains `state_contract.version=rwkv7_coreml_state_contract_v1`
with explicit per-layer `wkv_state`, `attn_x_prev`, `ffn_x_prev`, and global
`v_first` / `seen_tokens` tensor shapes.  The runtime plan row surfaces
`stateful_contract_present=true`, while keeping `decode_implemented=false` and
`prefill_implemented=false`; the goal audit therefore reports CoreML as
`prototype`, not `pass`, until a real stateful `.mlpackage` emits TTFT,
prefill/decode throughput, memory, quantization, and correctness rows.

Run CoreML runtime rows from an export manifest:

```bash
# Plan rows only; safe without CoreMLTools or an .mlpackage.
PYTHONPATH=. python bench/run_coreml_apple_baseline.py \
  --manifest exports/rwkv7-g1g-1.5b-coreml/coreml_export_manifest.json \
  --dry-run \
  --prompt-target-chars 1024,4096 \
  --decode-lengths 128,512 \
  --results bench/results_qwen35_apple_baseline.jsonl

# Live stateful runtime + correctness gates.
PYTHONPATH=. python bench/run_coreml_apple_baseline.py \
  --manifest exports/rwkv7-g1g-1.5b-coreml/coreml_export_manifest.json \
  --compute-units cpu-and-ne \
  --verify-chunked-prefill \
  --verify-hf-parity \
  --require-hf-greedy-match \
  --results bench/results_qwen35_apple_baseline.jsonl
```

Plan and smoke a CoreML package export:

```bash
# Import-safe plan: no CoreMLTools required.
PYTHONPATH=. python scripts/export_rwkv7_coreml.py \
  /path/to/rwkv7-g1g-1.5b-hf \
  exports/rwkv7-g1g-1.5b-coreml \
  --dry-run \
  --export-kind stateful-multifunction \
  --chunks 4 \
  --prefill-seq-length 16 \
  --sample-seq-length 128 \
  --state-mode wkv-coreml \
  --quantization none \
  --results bench/results_qwen35_apple_baseline.jsonl

# Live correctness-first stateful export when CoreMLTools is installed.
PYTHONPATH=. python scripts/export_rwkv7_coreml.py \
  /path/to/rwkv7-g1d-0.1b-hf \
  exports/rwkv7-g1d-0.1b-coreml \
  --export-kind stateful-multifunction \
  --prefill-seq-length 16 \
  --deployment-target iOS18 \
  --compute-units cpu-and-ne \
  --coreml-compute-precision auto \
  --quantization none \
  --results bench/results_qwen35_apple_baseline.jsonl
```

Stateful TorchScript prefill is statically unrolled, so the exported chunk is
intentionally small (default `16`, maximum `128`). Longer prompts do not require
a larger package: the runtime streams them through repeated masked chunks.

The export row uses `axis=rwkv7_coreml_export`.  `status=plan` only records the
manifest/contract; `status=pass` means a `.mlpackage` was produced.  A CoreML
export row alone is **not** a Qwen3.5 performance win. Only live stateful runtime
rows with TTFT, prefill/decode tok/s, memory, and correctness fields enter the
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

Audit the full Apple/mobile goal coverage:

```bash
PYTHONPATH=. python bench/audit_qwen35_apple_goal.py \
  --results bench/results_qwen35_apple_baseline.jsonl \
  --results bench/apple_qwen35_2b_tokenonly_m5_20260707 \
  --required-shape chars1024:128 \
  --required-shape chars4096:512 \
  --require-quality \
  --require-coreml \
  --append bench/results_qwen35_apple_baseline.jsonl
```

The audit accepts one or more JSONL files or evidence directories, and emits
`axis=qwen35_apple_goal_audit` rows plus a
`qwen35_apple_goal_audit_summary`.  It is intentionally broader than the speed
comparator: for every configured Qwen3.5/RWKV tier it checks same-prompt Qwen
coverage, RWKV MLX coverage, decode/prefill/TTFT/memory fields, W8/W4 quantized
rows, chunked-prefill/state-cache correctness, comparison-gate rows, quality
comparison rows, long-context rows, and stateful CoreML runtime evidence.  The
one-command wrapper runs this audit after comparison gates by default; tune it
with `GOAL_AUDIT_TIERS`, `GOAL_AUDIT_SHAPES`,
`GOAL_AUDIT_REQUIRE_QUALITY`, `GOAL_AUDIT_REQUIRE_COREML`, and
`GOAL_AUDIT_FAIL_ON_GATE`.

## Initial acceptance matrix

The first M5/16GB live 0.8B-vs-0.4B matrix is now present. At 128/512 prompt
characters and 32 generated tokens, the retained conservative RWKV fp16 decode
rows reach about `0.82x/0.92x` Qwen, while prefill reaches only `0.090x/0.049x`. RWKV W4 lowers
its own peak memory from about `929MB` to `528MB`, but decode falls to about
`0.62x/0.60x` Qwen and prefill to `0.064x/0.030x`. Qwen `/api/ps` loaded memory
is about `1.09-1.11GB`, but peak memory is not yet captured. W4 does not
preserve fp16 tokens on every prompt, so neither the peak-memory nor quality
gate is complete. See the two
`bench/results_qwen35_apple_m5_20260710_*.jsonl` files.

| RWKV target | Qwen3.5 comparator | Runtime gate | Current status |
|---|---|---|---|
| RWKV-7 0.4B fp16/W4 MLX | `qwen3.5:0.8b-mlx` | lower memory and higher decode tok/s at prompt 1k/4k/8k, decode 128/512 | first short same-device rows landed; decode/prefill/TTFT gates fail, Qwen loaded memory is recorded, and Qwen peak memory is unknown |
| RWKV-7 1.5B W4/MLX | `qwen3.5:2b-mlx` / `mlx-community/Qwen3.5-2B-MLX-4bit` | lower memory and higher or equal decode tok/s; TTFT no worse by >10% | same-device 512/64 token-only row collected: memory pass, speed/TTFT fail |
| RWKV-7 2.9B W4/MLX/CoreML | `qwen3.5:4b-mlx` | lower memory and higher decode tok/s | 0.1B stateful CoreML correctness passes; 2.9B quantized/ANE rows not landed |
| RWKV-7 larger / distilled mobile | `qwen3.5:9b-mlx` | mobile-useful memory envelope plus quality eval | requires model/quality work |

## CoreML / ANE follow-up

`rwkv-mobile` shows the right production direction for mobile Apple devices:

- separate `decode` and `prefill` CoreML functions
- chunked model export
- CoreML state / tensor state / WKV-CoreML state variants
- async prefill loading
- int8 / int4 / LUT quantization

The repository now has a live CoreML bridge. It records chunking, state mode,
quantization, deployment target, and compute precision, and exports deduplicated
stateful prefill/decode functions. On M5, the 0.1B fp32-compute short row passes
MLState transfer, alternate chunk split, and HF greedy-token parity. fp16
stateful compute remains opt-in because its first live row mismatched HF tokens.

The next repository lane should add:

1. Extend live correctness rows to prefill chunks 16/64 and long prompts/decode.
2. Add 0.4B/1.5B and CoreML W4/LUT/INT4 rows in the same schema.
3. Fix/selectively preserve recurrent precision in the fp16/ANE lane.
4. Record confirmed runtime placement rather than treating `CPU_AND_NE` as
   proof of ANE use.
5. Add iPhone/iPad rows once device access is available.

## Non-goals for the first baseline PR

- It does not claim final quality superiority over Qwen3.5.
- It does not claim the short 0.1B CoreML correctness row as production ANE performance.
- It does not mark W8/W4 as fp16-beating until JSONL evidence proves it.
- It does not replace the existing Apple MLX session and quant regression tests.

## 2026-07-07 MLX WKV scan prefill end-to-end row

The first opt-in end-to-end MLX multi-token WKV scan path is now wired behind
`RWKV_WKV_SCAN_PREFILL=1` / `RWKV7_MLX_WKV_SCAN_PREFILL=1`.
Evidence lives in `bench/apple_e2e_scan_prefill_m5_20260707/`.

Short-prompt Apple M5 smoke rows show the prefill path is active via
`wkv_scan_prefill=true` and `wkv_scan_prefill_counts={"metal": 24}`:

| Model | Previous token-major prefill tok/s | Scan prefill tok/s | Speedup | Notes |
|---|---:|---:|---:|---|
| RWKV-7 0.4B mm4 | 53.62 | 178.51 | 3.33x | decode remains single-token path |
| RWKV-7 1.5B mm4 | 21.49 | 38.11 | 1.77x | decode remains single-token path |

This is not yet the final Qwen3.5 acceptance win: the flag remains opt-in while
longer prompts, larger batches, quality drift, and same-device Qwen comparison
matrices are expanded.  It is the first real end-to-end replacement of the
prefill token-major WKV loop with a layer-major multi-token scan.

## 2026-07-08 scan-prefill comparison gate

A second Apple M5 evidence batch lives in
`bench/apple_e2e_scan_prefill_m5_20260708/`.  It adds a reusable correctness and
speed gate, `scripts/mlx_scan_prefill_compare.py`, which compares the previous
MLX token-major prefill path against the opt-in scan-prefill path on the same
HF model, prompt, quantization, and WKV backend.

Recorded real-model rows:

| Model | Token-major prefill tok/s | Scan prefill tok/s | Speedup | Generated ids |
|---|---:|---:|---:|---|
| RWKV-7 0.4B mm4 | 57.00 | 221.10 | 3.88x | identical |
| RWKV-7 1.5B mm4 | 21.93 | 53.52 | 2.44x | identical |

The Apple acceptance wrapper can run this gate with:

```bash
SCAN_PREFILL_COMPARE_MODELS=/path/to/rwkv7-hf-model \
SCAN_PREFILL_COMPARE_FAIL_ON_GATE=1 \
bash scripts/run_qwen35_apple_acceptance.sh
```

The gate appends `axis=mlx_scan_prefill_compare` rows and records logit drift,
state drift, generated-id equality, token-major/scan prefill timing, and WKV
kernel count reduction.

## 2026-07-08 scan-prefill auto policy

The MLX scan-prefill path now supports a production-shaped auto policy:

```bash
RWKV_WKV_SCAN_PREFILL=auto
RWKV_WKV_SCAN_PREFILL_MIN_TOKENS=32
```

`auto` enables scan prefill for multi-token chunks above the threshold and keeps
single-token decode on the existing decode path.  Telemetry records
`wkv_scan_prefill_mode`, `wkv_scan_prefill_min_tokens`, and
`wkv_scan_prefill_reason_counts`.

Apple M5 evidence in `bench/apple_scan_prefill_auto_m5_20260708/`:

| Model | Shape | Prefill tok/s | Decode tok/s | TTFT s | Peak memory |
|---|---|---:|---:|---:|---:|
| RWKV-7 0.4B mm4 | 1024 chars / 128 decode | 254.27 | 61.29 | 1.285 | 602 MB |
| RWKV-7 1.5B mm4 | 1024 chars / 128 decode | 61.37 | 28.54 | 5.316 | 1.47 GB |
| RWKV-7 0.4B mm4 | 4096 chars / 128 decode | 247.42 | 60.14 | 5.295 | 1.24 GB |
| RWKV-7 1.5B mm4 | 4096 chars / 128 decode | 53.60 | 25.40 | 24.443 | 2.08 GB |

The 4096-char rows also validate chunked prefill with three scan chunks
(`chunked_wkv_scan_prefill_counts.metal=72`) and two state-only intermediate
chunks.
