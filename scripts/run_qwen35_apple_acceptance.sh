#!/usr/bin/env bash
# One-command Apple/Qwen3.5 evidence collection gate.
#
# This wrapper runs the shared same-prompt Qwen3.5-vs-RWKV Apple baseline
# harness, optionally pulls Ollama models, optionally emits CoreML export
# manifests, then appends comparison-gate and goal-audit rows.  It is intentionally controlled
# through environment variables so contributors can run small local smoke rows
# or full 0.8B/2B/4B/9B acceptance matrices without editing the script.

set -euo pipefail

source "$(dirname "$0")/_hf_script_common.sh"

RESULTS="${RESULTS:-bench/results_qwen35_apple_baseline.jsonl}"
rwkv7_prepare_results

PROMPT_TARGET_CHARS="${PROMPT_TARGET_CHARS:-1024,4096}"
DECODE_LENGTHS="${DECODE_LENGTHS:-128,512}"
PROMPT_SEED="${PROMPT_SEED:-User: Compare RWKV-7 and Qwen3.5 on Apple Silicon. Report TTFT, prefill, decode, memory, state cache, W4/LUT/INT4, and quality gaps. Assistant: }"
REPEAT="${REPEAT:-1}"
WARMUP_REPEATS="${WARMUP_REPEATS:-0}"
DRY_RUN="${DRY_RUN:-0}"
STORE_RESPONSES="${STORE_RESPONSES:-0}"

QWEN_MODELS="${QWEN_MODELS:-qwen3.5:0.8b-mlx,qwen3.5:2b-mlx,qwen3.5:4b-mlx,qwen3.5:9b-mlx}"
RUN_QWEN="${RUN_QWEN:-auto}"
QWEN_MLX_VLM_MODELS="${QWEN_MLX_VLM_MODELS:-}"
RUN_QWEN_MLX_VLM="${RUN_QWEN_MLX_VLM:-auto}"
QWEN_MLX_VLM_TOKEN_ONLY="${QWEN_MLX_VLM_TOKEN_ONLY:-0}"
PULL_QWEN="${PULL_QWEN:-0}"
OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
OLLAMA_TIMEOUT_S="${OLLAMA_TIMEOUT_S:-600}"
OLLAMA_PULL_TIMEOUT_S="${OLLAMA_PULL_TIMEOUT_S:-7200}"
OLLAMA_PULL_IDLE_TIMEOUT_S="${OLLAMA_PULL_IDLE_TIMEOUT_S:-120}"
OLLAMA_PULL_FAIL_ON_TIMEOUT="${OLLAMA_PULL_FAIL_ON_TIMEOUT:-1}"
OLLAMA_PULL_RESULTS="${OLLAMA_PULL_RESULTS:-${RESULTS}}"
OLLAMA_THINK="${OLLAMA_THINK:-0}"
OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-0}"
OLLAMA_CACHE_PROMPT="${OLLAMA_CACHE_PROMPT:-0}"
OLLAMA_CAPTURE_MEMORY="${OLLAMA_CAPTURE_MEMORY:-1}"
TEMPERATURE="${TEMPERATURE:-0.0}"

RWKV_MLX_MODELS="${RWKV_MLX_MODELS:-}"
RUN_RWKV="${RUN_RWKV:-auto}"
RWKV_DTYPE="${RWKV_DTYPE:-fp16}"
RWKV_QUANTIZATION="${RWKV_QUANTIZATION:-mm4}"
RWKV_QUANT_MIN_PARAMS="${RWKV_QUANT_MIN_PARAMS:-4000000}"
RWKV_QUANT_RKV_MIN_PARAMS="${RWKV_QUANT_RKV_MIN_PARAMS:-0}"
RWKV_QUANT_BACKEND="${RWKV_QUANT_BACKEND:-auto}"
RWKV_WKV_BACKEND="${RWKV_WKV_BACKEND:-metal}"
RWKV_CHUNK_SIZE="${RWKV_CHUNK_SIZE:-2048}"
RWKV_FUSED_FFN_KEY_RELU2="${RWKV_FUSED_FFN_KEY_RELU2:-1}"
RWKV_FUSED_ATTN_MIX="${RWKV_FUSED_ATTN_MIX:-0}"
RWKV_WKV_SCAN_PREFILL="${RWKV_WKV_SCAN_PREFILL:-0}"
RWKV_WKV_SCAN_PREFILL_MIN_TOKENS="${RWKV_WKV_SCAN_PREFILL_MIN_TOKENS:-32}"
SCAN_PREFILL_COMPARE_MODELS="${SCAN_PREFILL_COMPARE_MODELS:-}"
SCAN_PREFILL_COMPARE_APPEND="${SCAN_PREFILL_COMPARE_APPEND:-${RESULTS}}"
SCAN_PREFILL_COMPARE_PROMPT_TARGET_CHARS="${SCAN_PREFILL_COMPARE_PROMPT_TARGET_CHARS:-512}"
SCAN_PREFILL_COMPARE_MAX_NEW_TOKENS="${SCAN_PREFILL_COMPARE_MAX_NEW_TOKENS:-16}"
SCAN_PREFILL_COMPARE_FAIL_ON_GATE="${SCAN_PREFILL_COMPARE_FAIL_ON_GATE:-0}"
# Interval 2 is exact on the current 0.1B/0.4B/1.5B fp16 and 0.4B/1.5B
# W4 Apple matrix, while avoiding one host/GPU synchronization per prompt
# token. The model API itself retains interval 1 as its conservative default.
RWKV_PREFILL_EVAL_INTERVAL="${RWKV_PREFILL_EVAL_INTERVAL:-2}"
RWKV_PREFILL_BACKEND="${RWKV_PREFILL_BACKEND:-recurrent}"
RWKV_DPLR_CHUNK_SIZE="${RWKV_DPLR_CHUNK_SIZE:-64}"
RWKV_DPLR_MIN_TOKENS="${RWKV_DPLR_MIN_TOKENS:-8}"
RWKV_DPLR_SUMMARY_IMPLEMENTATION="${RWKV_DPLR_SUMMARY_IMPLEMENTATION:-tiled}"
RWKV_DPLR_LAYER_EVAL_INTERVAL="${RWKV_DPLR_LAYER_EVAL_INTERVAL:-4}"
RWKV_DPLR_LAYER_EVAL_MIN_TOKENS="${RWKV_DPLR_LAYER_EVAL_MIN_TOKENS:-64}"
RWKV_DPLR_WINDOW_TOKENS="${RWKV_DPLR_WINDOW_TOKENS:-512}"
RWKV_DECODE_BACKEND="${RWKV_DECODE_BACKEND:-auto}"
RWKV_DECODE_NORM_BACKEND="${RWKV_DECODE_NORM_BACKEND:-reference}"
RWKV_PREPARE_COMPILED_DECODE="${RWKV_PREPARE_COMPILED_DECODE:-0}"
RWKV_COMPILED_DECODE_VALIDATION_TOKENS="${RWKV_COMPILED_DECODE_VALIDATION_TOKENS:-32}"
RWKV_COMPILED_DECODE_REFERENCE_LOGITS_ATOL="${RWKV_COMPILED_DECODE_REFERENCE_LOGITS_ATOL:-0.25}"
RWKV_COMPILED_DECODE_REFERENCE_STATE_ATOL="${RWKV_COMPILED_DECODE_REFERENCE_STATE_ATOL:-0.5}"

# Comparison defaults cover the current local/public model classes.  Add the
# 4B/9B pairs once matching RWKV 2.9B/larger or distilled mobile exports exist.
PAIRS="${PAIRS:-qwen3.5:0.8b-mlx=rwkv7-g1d-0.4b-hf,qwen3.5:2b-mlx=rwkv7-g1g-1.5b-hf}"
SKIP_COMPARE="${SKIP_COMPARE:-0}"
FAIL_ON_GATE="${FAIL_ON_GATE:-0}"
REQUIRE_PREFILL="${REQUIRE_PREFILL:-1}"
REQUIRE_TTFT="${REQUIRE_TTFT:-1}"
REQUIRE_MEMORY="${REQUIRE_MEMORY:-1}"
MIN_DECODE_RATIO="${MIN_DECODE_RATIO:-1.0}"
MIN_PREFILL_RATIO="${MIN_PREFILL_RATIO:-1.0}"
MAX_TTFT_RATIO="${MAX_TTFT_RATIO:-1.1}"
MAX_MEMORY_RATIO="${MAX_MEMORY_RATIO:-1.0}"
COMPARE_APPEND="${COMPARE_APPEND:-${RESULTS}}"
COMPARE_DIAGNOSTICS="${COMPARE_DIAGNOSTICS:-1}"
QUALITY_RUBRIC="${QUALITY_RUBRIC:-}"
QUALITY_PAIRS="${QUALITY_PAIRS:-${PAIRS}}"
QUALITY_APPEND="${QUALITY_APPEND:-${RESULTS}}"
QUALITY_ALLOW_PREVIEW="${QUALITY_ALLOW_PREVIEW:-0}"
QUALITY_FAIL_ON_GATE="${QUALITY_FAIL_ON_GATE:-0}"

# Goal-level audit across public Qwen3.5 0.8B/2B/4B/9B tiers.  This is stricter
# than pairwise comparison rows: it records missing MLX/CoreML/quant/state-cache/
# quality/long-context evidence as explicit JSONL rows.
SKIP_GOAL_AUDIT="${SKIP_GOAL_AUDIT:-0}"
GOAL_AUDIT_APPEND="${GOAL_AUDIT_APPEND:-${RESULTS}}"
GOAL_AUDIT_TIERS="${GOAL_AUDIT_TIERS:-}"
GOAL_AUDIT_SHAPES="${GOAL_AUDIT_SHAPES:-auto}"
GOAL_AUDIT_LONG_CONTEXT_CHARS="${GOAL_AUDIT_LONG_CONTEXT_CHARS:-4096}"
GOAL_AUDIT_STATE_CACHE_TOLERANCE="${GOAL_AUDIT_STATE_CACHE_TOLERANCE:-0.0001}"
GOAL_AUDIT_REQUIRE_QUALITY="${GOAL_AUDIT_REQUIRE_QUALITY:-1}"
GOAL_AUDIT_REQUIRE_COREML="${GOAL_AUDIT_REQUIRE_COREML:-1}"
GOAL_AUDIT_FAIL_ON_GATE="${GOAL_AUDIT_FAIL_ON_GATE:-0}"

# Optional CoreML export-manifest lane.  This does not claim ANE performance;
# it records the export/state/quant contract next to the baseline rows.
COREML_EXPORT_MODELS="${COREML_EXPORT_MODELS:-}"
COREML_OUTPUT_ROOT="${COREML_OUTPUT_ROOT:-exports/coreml}"
COREML_DRY_RUN="${COREML_DRY_RUN:-1}"
COREML_REQUIRE_TOOLS="${COREML_REQUIRE_TOOLS:-0}"
COREML_CHUNKS="${COREML_CHUNKS:-4}"
COREML_PREFILL_SEQ_LENGTH="${COREML_PREFILL_SEQ_LENGTH:-16}"
COREML_SAMPLE_SEQ_LENGTH="${COREML_SAMPLE_SEQ_LENGTH:-128}"
COREML_EXPORT_KIND="${COREML_EXPORT_KIND:-stateful-multifunction}"
COREML_STATE_MODE="${COREML_STATE_MODE:-wkv-coreml}"
COREML_QUANTIZATION="${COREML_QUANTIZATION:-none}"
COREML_DEPLOYMENT_TARGET="${COREML_DEPLOYMENT_TARGET:-iOS18}"
COREML_COMPUTE_UNITS="${COREML_COMPUTE_UNITS:-cpu-and-ne}"
COREML_COMPUTE_PRECISION="${COREML_COMPUTE_PRECISION:-auto}"
COREML_RUNTIME_MANIFESTS="${COREML_RUNTIME_MANIFESTS:-}"
COREML_RUN_EXPORTED="${COREML_RUN_EXPORTED:-1}"
COREML_RUNTIME_DRY_RUN="${COREML_RUNTIME_DRY_RUN:-${COREML_DRY_RUN}}"
COREML_RUNTIME_REQUIRE_TOOLS="${COREML_RUNTIME_REQUIRE_TOOLS:-0}"
COREML_RUNTIME_PROMPT_TARGET_CHARS="${COREML_RUNTIME_PROMPT_TARGET_CHARS:-${PROMPT_TARGET_CHARS}}"
COREML_RUNTIME_DECODE_LENGTHS="${COREML_RUNTIME_DECODE_LENGTHS:-${DECODE_LENGTHS}}"
COREML_RUNTIME_REPEAT="${COREML_RUNTIME_REPEAT:-${REPEAT}}"
COREML_RUNTIME_WARMUP="${COREML_RUNTIME_WARMUP:-1}"
COREML_RUNTIME_COMPUTE_UNITS="${COREML_RUNTIME_COMPUTE_UNITS:-${COREML_COMPUTE_UNITS}}"
COREML_RUNTIME_VERIFY_CHUNKED="${COREML_RUNTIME_VERIFY_CHUNKED:-1}"
COREML_RUNTIME_VERIFY_CHUNK_SIZE="${COREML_RUNTIME_VERIFY_CHUNK_SIZE:-1}"
COREML_RUNTIME_REQUIRE_CHUNKED="${COREML_RUNTIME_REQUIRE_CHUNKED:-0}"
COREML_RUNTIME_VERIFY_HF="${COREML_RUNTIME_VERIFY_HF:-1}"
COREML_RUNTIME_HF_DTYPE="${COREML_RUNTIME_HF_DTYPE:-fp32}"
COREML_RUNTIME_REQUIRE_HF_MATCH="${COREML_RUNTIME_REQUIRE_HF_MATCH:-1}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export RWKV7_NATIVE_MODEL="${RWKV7_NATIVE_MODEL:-1}"
export RWKV7_MLX_GROUP_RKV_QUANT_PROJECTION="${RWKV7_MLX_GROUP_RKV_QUANT_PROJECTION:-1}"
export RWKV7_MLX_STEP_EVAL_INTERVAL="${RWKV7_MLX_STEP_EVAL_INTERVAL:-8}"
export RWKV7_MLX_FUSED_FFN_KEY_RELU2="${RWKV7_MLX_FUSED_FFN_KEY_RELU2:-${RWKV_FUSED_FFN_KEY_RELU2}}"
export RWKV7_MLX_FUSED_ATTN_MIX="${RWKV7_MLX_FUSED_ATTN_MIX:-${RWKV_FUSED_ATTN_MIX}}"
export RWKV7_MLX_WKV_SCAN_PREFILL="${RWKV7_MLX_WKV_SCAN_PREFILL:-${RWKV_WKV_SCAN_PREFILL}}"
export RWKV7_MLX_WKV_SCAN_PREFILL_MIN_TOKENS="${RWKV7_MLX_WKV_SCAN_PREFILL_MIN_TOKENS:-${RWKV_WKV_SCAN_PREFILL_MIN_TOKENS}}"

rwkv7_csv_items() {
  local raw="$1"
  local item
  local -a _rwkv7_items=()
  IFS=',' read -r -a _rwkv7_items <<< "${raw}"
  for item in "${_rwkv7_items[@]-}"; do
    item="${item#${item%%[![:space:]]*}}"
    item="${item%${item##*[![:space:]]}}"
    if [[ -n "${item}" ]]; then
      printf '%s\n' "${item}"
    fi
  done
}

rwkv7_bool_auto() {
  local mode="$1"
  local values="$2"
  if [[ "${mode}" == "auto" ]]; then
    if [[ -n "${values}" ]]; then
      printf '1'
    else
      printf '0'
    fi
  else
    printf '%s' "${mode}"
  fi
}

RUN_QWEN="$(rwkv7_bool_auto "${RUN_QWEN}" "${QWEN_MODELS}")"
RUN_QWEN_MLX_VLM="$(rwkv7_bool_auto "${RUN_QWEN_MLX_VLM}" "${QWEN_MLX_VLM_MODELS}")"
RUN_RWKV="$(rwkv7_bool_auto "${RUN_RWKV}" "${RWKV_MLX_MODELS}")"

if [[ "${PULL_QWEN}" == "1" && "${RUN_QWEN}" == "1" ]]; then
  if ! command -v ollama >/dev/null 2>&1; then
    echo "PULL_QWEN=1 requested, but ollama is not installed or not on PATH." >&2
    exit 2
  fi
  while IFS= read -r model; do
    rwkv7_log "Pulling ${model} through Ollama"
    pull_args=(
      "${model}"
      --host "${OLLAMA_HOST}"
      --timeout-s "${OLLAMA_PULL_TIMEOUT_S}"
      --idle-timeout-s "${OLLAMA_PULL_IDLE_TIMEOUT_S}"
      --results "${OLLAMA_PULL_RESULTS}"
    )
    if [[ "${OLLAMA_PULL_FAIL_ON_TIMEOUT}" != "1" ]]; then
      pull_args+=(--no-fail-on-timeout)
    fi
    rwkv7_run "${PYTHON_BIN}" scripts/ollama_pull_with_timeout.py "${pull_args[@]}"
  done < <(rwkv7_csv_items "${QWEN_MODELS}")
fi

coreml_runtime_manifests=()
if [[ -n "${COREML_EXPORT_MODELS}" ]]; then
  while IFS= read -r model; do
    out_dir="${COREML_OUTPUT_ROOT}/$(basename "${model}")"
    args=(
      "${model}"
      "${out_dir}"
      --chunks "${COREML_CHUNKS}"
      --prefill-seq-length "${COREML_PREFILL_SEQ_LENGTH}"
      --sample-seq-length "${COREML_SAMPLE_SEQ_LENGTH}"
      --export-kind "${COREML_EXPORT_KIND}"
      --state-mode "${COREML_STATE_MODE}"
      --quantization "${COREML_QUANTIZATION}"
      --deployment-target "${COREML_DEPLOYMENT_TARGET}"
      --compute-units "${COREML_COMPUTE_UNITS}"
      --coreml-compute-precision "${COREML_COMPUTE_PRECISION}"
      --results "${RESULTS}"
    )
    if [[ "${COREML_DRY_RUN}" == "1" ]]; then
      args+=(--dry-run)
    fi
    if [[ "${COREML_REQUIRE_TOOLS}" == "1" ]]; then
      args+=(--require-coremltools)
    fi
    rwkv7_log "CoreML export manifest/prototype for ${model}"
    rwkv7_run "${PYTHON_BIN}" scripts/export_rwkv7_coreml.py "${args[@]}"
    if [[ "${COREML_RUN_EXPORTED}" == "1" ]]; then
      coreml_runtime_manifests+=("${out_dir}/coreml_export_manifest.json")
    fi
  done < <(rwkv7_csv_items "${COREML_EXPORT_MODELS}")
fi
while IFS= read -r manifest; do
  coreml_runtime_manifests+=("${manifest}")
done < <(rwkv7_csv_items "${COREML_RUNTIME_MANIFESTS}")

if (( ${#coreml_runtime_manifests[@]} > 0 )); then
  for manifest in "${coreml_runtime_manifests[@]}"; do
    args=(
      --manifest "${manifest}"
      --prompt-target-chars "${COREML_RUNTIME_PROMPT_TARGET_CHARS}"
      --decode-lengths "${COREML_RUNTIME_DECODE_LENGTHS}"
      --repeat "${COREML_RUNTIME_REPEAT}"
      --warmup "${COREML_RUNTIME_WARMUP}"
      --compute-units "${COREML_RUNTIME_COMPUTE_UNITS}"
      --prompt-seed "${PROMPT_SEED}"
      --results "${RESULTS}"
    )
    if [[ "${COREML_RUNTIME_VERIFY_CHUNKED}" == "1" ]]; then
      args+=(--verify-chunked-prefill --verify-chunk-size "${COREML_RUNTIME_VERIFY_CHUNK_SIZE}")
    fi
    if [[ "${COREML_RUNTIME_REQUIRE_CHUNKED}" == "1" ]]; then
      args+=(--require-chunked-prefill-match)
    fi
    if [[ "${COREML_RUNTIME_VERIFY_HF}" == "1" ]]; then
      args+=(--verify-hf-parity --hf-parity-dtype "${COREML_RUNTIME_HF_DTYPE}")
    fi
    if [[ "${COREML_RUNTIME_REQUIRE_HF_MATCH}" == "1" ]]; then
      args+=(--require-hf-greedy-match)
    fi
    if [[ "${STORE_RESPONSES}" == "1" ]]; then
      args+=(--store-responses)
    fi
    if [[ "${COREML_RUNTIME_DRY_RUN}" == "1" ]]; then
      args+=(--dry-run)
    fi
    if [[ "${COREML_RUNTIME_REQUIRE_TOOLS}" == "1" ]]; then
      args+=(--require-coremltools)
    fi
    rwkv7_log "CoreML runtime baseline rows for ${manifest}"
    rwkv7_run "${PYTHON_BIN}" bench/run_coreml_apple_baseline.py "${args[@]}"
  done
fi

baseline_args=(
  --results "${RESULTS}"
  --prompt-target-chars "${PROMPT_TARGET_CHARS}"
  --prompt-seed "${PROMPT_SEED}"
  --decode-lengths "${DECODE_LENGTHS}"
  --repeat "${REPEAT}"
  --warmup-repeats "${WARMUP_REPEATS}"
  --ollama-host "${OLLAMA_HOST}"
  --ollama-timeout-s "${OLLAMA_TIMEOUT_S}"
  --ollama-keep-alive "${OLLAMA_KEEP_ALIVE}"
  --temperature "${TEMPERATURE}"
  --rwkv-dtype "${RWKV_DTYPE}"
  --rwkv-quantization "${RWKV_QUANTIZATION}"
  --rwkv-quant-min-params "${RWKV_QUANT_MIN_PARAMS}"
  --rwkv-quant-rkv-min-params "${RWKV_QUANT_RKV_MIN_PARAMS}"
  --rwkv-quant-backend "${RWKV_QUANT_BACKEND}"
  --rwkv-wkv-backend "${RWKV_WKV_BACKEND}"
  --rwkv-chunk-size "${RWKV_CHUNK_SIZE}"
  --rwkv-prefill-eval-interval "${RWKV_PREFILL_EVAL_INTERVAL}"
  --rwkv-prefill-backend "${RWKV_PREFILL_BACKEND}"
  --rwkv-dplr-chunk-size "${RWKV_DPLR_CHUNK_SIZE}"
  --rwkv-dplr-min-tokens "${RWKV_DPLR_MIN_TOKENS}"
  --rwkv-dplr-summary-implementation "${RWKV_DPLR_SUMMARY_IMPLEMENTATION}"
  --rwkv-dplr-layer-eval-interval "${RWKV_DPLR_LAYER_EVAL_INTERVAL}"
  --rwkv-dplr-layer-eval-min-tokens "${RWKV_DPLR_LAYER_EVAL_MIN_TOKENS}"
  --rwkv-dplr-window-tokens "${RWKV_DPLR_WINDOW_TOKENS}"
  --rwkv-decode-backend "${RWKV_DECODE_BACKEND}"
  --rwkv-decode-norm-backend "${RWKV_DECODE_NORM_BACKEND}"
  --rwkv-compiled-decode-validation-tokens "${RWKV_COMPILED_DECODE_VALIDATION_TOKENS}"
  --rwkv-compiled-decode-reference-logits-atol "${RWKV_COMPILED_DECODE_REFERENCE_LOGITS_ATOL}"
  --rwkv-compiled-decode-reference-state-atol "${RWKV_COMPILED_DECODE_REFERENCE_STATE_ATOL}"
)
if [[ "${RWKV_PREPARE_COMPILED_DECODE}" == "1" ]]; then
  baseline_args+=(--rwkv-prepare-compiled-decode)
fi
if [[ "${OLLAMA_THINK}" == "1" ]]; then
  baseline_args+=(--ollama-think)
fi
if [[ "${OLLAMA_CACHE_PROMPT}" == "1" ]]; then
  baseline_args+=(--ollama-cache-prompt)
fi
if [[ "${OLLAMA_CAPTURE_MEMORY}" != "1" ]]; then
  baseline_args+=(--no-ollama-memory)
fi
if [[ "${STORE_RESPONSES}" == "1" ]]; then
  baseline_args+=(--store-responses)
fi
if [[ "${RUN_QWEN}" == "1" ]]; then
  baseline_args+=(--qwen-models "${QWEN_MODELS}")
else
  baseline_args+=(--qwen-models "")
fi
if [[ "${RUN_QWEN_MLX_VLM}" == "1" ]]; then
  baseline_args+=(--qwen-mlx-vlm-models "${QWEN_MLX_VLM_MODELS}")
else
  baseline_args+=(--qwen-mlx-vlm-models "")
fi
if [[ "${QWEN_MLX_VLM_TOKEN_ONLY}" == "1" ]]; then
  baseline_args+=(--qwen-mlx-vlm-token-only)
fi
if [[ "${RUN_RWKV}" == "1" ]]; then
  baseline_args+=(--rwkv-mlx-models "${RWKV_MLX_MODELS}")
else
  baseline_args+=(--rwkv-mlx-models "")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  baseline_args+=(--dry-run)
fi

rwkv7_print_env
rwkv7_log "Apple/Qwen3.5 same-prompt baseline matrix"
rwkv7_run "${PYTHON_BIN}" bench/run_qwen35_apple_baseline.py "${baseline_args[@]}"

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

if [[ -n "${SCAN_PREFILL_COMPARE_MODELS}" ]]; then
  while IFS= read -r model; do
    scan_compare_args=(
      "${model}"
      --prompt-target-chars "${SCAN_PREFILL_COMPARE_PROMPT_TARGET_CHARS}"
      --max-new-tokens "${SCAN_PREFILL_COMPARE_MAX_NEW_TOKENS}"
      --dtype "${RWKV_DTYPE}"
      --quantization "${RWKV_QUANTIZATION}"
      --quant-min-params "${RWKV_QUANT_MIN_PARAMS}"
      --quant-backend "${RWKV_QUANT_BACKEND}"
      --wkv-backend "${RWKV_WKV_BACKEND}"
    )
    if [[ "${RWKV_QUANT_RKV_MIN_PARAMS}" != "" ]]; then
      scan_compare_args+=(--quant-rkv-min-params "${RWKV_QUANT_RKV_MIN_PARAMS}")
    fi
    if [[ -n "${SCAN_PREFILL_COMPARE_APPEND}" ]]; then
      scan_compare_args+=(--results "${SCAN_PREFILL_COMPARE_APPEND}")
    fi
    rwkv7_log "MLX WKV scan prefill vs token-major comparison for ${model}"
    if [[ "${SCAN_PREFILL_COMPARE_FAIL_ON_GATE}" == "1" ]]; then
      rwkv7_run "${PYTHON_BIN}" scripts/mlx_scan_prefill_compare.py "${scan_compare_args[@]}"
    else
      rwkv7_run "${PYTHON_BIN}" scripts/mlx_scan_prefill_compare.py "${scan_compare_args[@]}" || true
    fi
  done < <(rwkv7_csv_items "${SCAN_PREFILL_COMPARE_MODELS}")
fi

if [[ -n "${QUALITY_RUBRIC}" ]]; then
  quality_args=(
    --results "${RESULTS}"
    --rubric "${QUALITY_RUBRIC}"
    --append "${QUALITY_APPEND}"
  )
  while IFS= read -r pair; do
    quality_args+=(--pair "${pair}")
  done < <(rwkv7_csv_items "${QUALITY_PAIRS}")
  if [[ "${QUALITY_ALLOW_PREVIEW}" == "1" ]]; then
    quality_args+=(--allow-preview)
  fi
  if [[ "${QUALITY_FAIL_ON_GATE}" == "1" ]]; then
    quality_args+=(--fail-on-gate)
  fi
  rwkv7_log "Apple/Qwen3.5 quality matrix"
  rwkv7_run "${PYTHON_BIN}" bench/score_qwen35_quality.py "${quality_args[@]}"
fi

if [[ "${SKIP_COMPARE}" != "1" && -n "${PAIRS}" ]]; then
  compare_args=(
    --results "${RESULTS}"
    --min-decode-ratio "${MIN_DECODE_RATIO}"
    --min-prefill-ratio "${MIN_PREFILL_RATIO}"
    --max-ttft-ratio "${MAX_TTFT_RATIO}"
    --max-memory-ratio "${MAX_MEMORY_RATIO}"
  )
  while IFS= read -r pair; do
    compare_args+=(--pair "${pair}")
  done < <(rwkv7_csv_items "${PAIRS}")
  if [[ "${REQUIRE_PREFILL}" == "1" ]]; then
    compare_args+=(--require-prefill)
  fi
  if [[ "${REQUIRE_TTFT}" == "1" ]]; then
    compare_args+=(--require-ttft)
  fi
  if [[ "${REQUIRE_MEMORY}" == "1" ]]; then
    compare_args+=(--require-memory)
  fi
  if [[ -n "${COMPARE_APPEND}" ]]; then
    compare_args+=(--append "${COMPARE_APPEND}")
  fi
  if [[ "${COMPARE_DIAGNOSTICS}" == "1" ]]; then
    compare_args+=(--diagnostics)
  fi
  if [[ "${FAIL_ON_GATE}" == "1" ]]; then
    compare_args+=(--fail-on-gate)
  fi

  rwkv7_log "Apple/Qwen3.5 comparison gates"
  rwkv7_run "${PYTHON_BIN}" bench/compare_qwen35_apple_baseline.py "${compare_args[@]}"
fi

if [[ "${SKIP_GOAL_AUDIT}" != "1" ]]; then
  audit_args=(
    --results "${RESULTS}"
    --long-context-chars "${GOAL_AUDIT_LONG_CONTEXT_CHARS}"
    --state-cache-tolerance "${GOAL_AUDIT_STATE_CACHE_TOLERANCE}"
  )
  if [[ -n "${GOAL_AUDIT_APPEND}" ]]; then
    audit_args+=(--append "${GOAL_AUDIT_APPEND}")
  fi
  while IFS= read -r tier; do
    audit_args+=(--tier "${tier}")
  done < <(rwkv7_csv_items "${GOAL_AUDIT_TIERS}")
  if [[ "${GOAL_AUDIT_SHAPES}" == "auto" ]]; then
    while IFS= read -r prompt_chars; do
      while IFS= read -r decode_tokens; do
        if [[ "${prompt_chars}" == chars* ]]; then
          audit_args+=(--required-shape "${prompt_chars}:${decode_tokens}")
        else
          audit_args+=(--required-shape "chars${prompt_chars}:${decode_tokens}")
        fi
      done < <(rwkv7_csv_items "${DECODE_LENGTHS}")
    done < <(rwkv7_csv_items "${PROMPT_TARGET_CHARS}")
  else
    while IFS= read -r shape; do
      audit_args+=(--required-shape "${shape}")
    done < <(rwkv7_csv_items "${GOAL_AUDIT_SHAPES}")
  fi
  if [[ "${GOAL_AUDIT_REQUIRE_QUALITY}" == "1" ]]; then
    audit_args+=(--require-quality)
  fi
  if [[ "${GOAL_AUDIT_REQUIRE_COREML}" == "1" ]]; then
    audit_args+=(--require-coreml)
  fi
  if [[ "${GOAL_AUDIT_FAIL_ON_GATE}" == "1" ]]; then
    audit_args+=(--fail-on-gate)
  fi
  rwkv7_log "Apple/Qwen3.5 goal coverage audit"
  rwkv7_run "${PYTHON_BIN}" bench/audit_qwen35_apple_goal.py "${audit_args[@]}"
fi
