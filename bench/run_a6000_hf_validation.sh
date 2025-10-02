#!/usr/bin/env bash
# RTX A6000 / Ampere sm_86 HF adapter validation for issue #115.
#
# Defaults target the current local validation host. Override any VAR=value on
# the command line when reusing this on a different machine.
#
# Examples:
#   bash bench/run_a6000_hf_validation.sh
#   RUN_CONVERT=0 RUN_TRAINING=0 RUN_ZERO=0 bash bench/run_a6000_hf_validation.sh
set -euo pipefail

for arg in "$@"; do
  case "${arg}" in
    *=*) export "${arg}" ;;
    *) echo "unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-/home/zhiyuanzhou/draft/venv/bin/python}"
source "${REPO_ROOT}/scripts/_hf_script_common.sh"

MODEL_ROOT="${MODEL_ROOT:-/home/zhiyuanzhou/rwkv_models}"
HF_ROOT="${HF_ROOT:-${MODEL_ROOT}/hf}"
VOCAB_FILE="${VOCAB_FILE:-${MODEL_ROOT}/rwkv_vocab_v20230424.txt}"
RESULTS="${RESULTS:-bench/results.jsonl}"

A6000_SINGLE_VISIBLE_DEVICES="${A6000_SINGLE_VISIBLE_DEVICES:-2}"
A6000_MULTI_VISIBLE_DEVICES="${A6000_MULTI_VISIBLE_DEVICES:-2,3}"
EXPECTED_GPU_NAME="${EXPECTED_GPU_NAME:-RTX A6000}"
EXPECTED_CAPABILITY="${EXPECTED_CAPABILITY:-8.6}"

RUN_CONVERT="${RUN_CONVERT:-1}"
RUN_A6000_GUARD="${RUN_A6000_GUARD:-1}"
RUN_ENV="${RUN_ENV:-1}"
RUN_01B_BASELINE="${RUN_01B_BASELINE:-1}"
RUN_LARGER_SMOKE="${RUN_LARGER_SMOKE:-1}"
RUN_BATCH_SWEEP="${RUN_BATCH_SWEEP:-1}"
RUN_QUANT="${RUN_QUANT:-1}"
RUN_NATIVE_MM_QUANT="${RUN_NATIVE_MM_QUANT:-1}"
RUN_TRAINING="${RUN_TRAINING:-1}"
RUN_HF_RESUME="${RUN_HF_RESUME:-1}"
RUN_ZERO="${RUN_ZERO:-1}"
RUN_ZERO_RESUME="${RUN_ZERO_RESUME:-1}"
RUN_ANALYZE="${RUN_ANALYZE:-1}"

ALL_MODEL_LABELS="${ALL_MODEL_LABELS:-0.1b 0.4b 1.5b 2.9b 7.2b}"
VALIDATION_MODEL_LABELS="${VALIDATION_MODEL_LABELS:-0.4b 1.5b 2.9b 7.2b}"
TRAIN_MODEL_LABELS="${TRAIN_MODEL_LABELS:-0.4b 1.5b 2.9b}"
ZERO_MODEL_LABELS="${ZERO_MODEL_LABELS:-0.4b 1.5b 2.9b}"
INFER_DTYPES="${INFER_DTYPES:-fp16 bf16}"
QUANT_DTYPE="${QUANT_DTYPE:-fp16}"
NATIVE_MM_QUANT_DTYPE="${NATIVE_MM_QUANT_DTYPE:-fp16}"
TRAIN_DTYPE="${TRAIN_DTYPE:-bf16}"

PROMPT_TOKENS="${PROMPT_TOKENS:-128}"
DECODE_TOKENS="${DECODE_TOKENS:-16}"
SMOKE_MAX_NEW_TOKENS="${SMOKE_MAX_NEW_TOKENS:-2}"
WARMUP="${WARMUP:-1}"
RUNS="${RUNS:-1}"
NATIVE_MM_MIN_PARAMS="${NATIVE_MM_MIN_PARAMS:-8000000}"

TRAIN_MAX_LENGTH="${TRAIN_MAX_LENGTH:-32}"
TRAIN_MAX_STEPS="${TRAIN_MAX_STEPS:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
TRAIN_RL_BATCH_SIZE="${TRAIN_RL_BATCH_SIZE:-2}"
TRAIN_DATASET_REPEATS="${TRAIN_DATASET_REPEATS:-4}"
RESUME_FIRST_STEPS="${RESUME_FIRST_STEPS:-1}"
RESUME_STEPS="${RESUME_STEPS:-2}"

ZERO_STAGE="${ZERO_STAGE:-both}"
ZERO_MAX_LENGTH="${ZERO_MAX_LENGTH:-16}"
ZERO_MAX_STEPS="${ZERO_MAX_STEPS:-1}"
ZERO_BATCH_SIZE="${ZERO_BATCH_SIZE:-1}"
ZERO_DATASET_REPEATS="${ZERO_DATASET_REPEATS:-4}"
ZERO_RESUME_FIRST_STEPS="${ZERO_RESUME_FIRST_STEPS:-1}"
ZERO_RESUME_STEPS="${ZERO_RESUME_STEPS:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

export RWKV_V7_ON="${RWKV_V7_ON:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export DS_IGNORE_CUDA_DETECTION="${DS_IGNORE_CUDA_DETECTION:-1}"
export RWKV7_FAST_TOKEN_BACKEND="${RWKV7_FAST_TOKEN_BACKEND:-auto}"

PYTHON_BIN_DIR="$(cd "$(dirname "${PYTHON_BIN}")" && pwd)"

run() {
  rwkv7_run "$@"
}

run_single_gpu() {
  rwkv7_run env CUDA_VISIBLE_DEVICES="${A6000_SINGLE_VISIBLE_DEVICES}" "$@"
}

run_multi_gpu() {
  rwkv7_run env CUDA_VISIBLE_DEVICES="${A6000_MULTI_VISIBLE_DEVICES}" PATH="${PYTHON_BIN_DIR}:${PATH}" "$@"
}

pth_for_label() {
  case "$1" in
    0.1b) echo "${MODEL_ROOT}/rwkv7-g1d-0.1b-20260129-ctx8192.pth" ;;
    0.4b) echo "${MODEL_ROOT}/rwkv7-g1d-0.4b-20260210-ctx8192.pth" ;;
    1.5b) echo "${MODEL_ROOT}/rwkv7-g1g-1.5b-20260526-ctx8192.pth" ;;
    2.9b) echo "${MODEL_ROOT}/rwkv7-g1g-2.9b-20260526-ctx8192.pth" ;;
    7.2b) echo "${MODEL_ROOT}/rwkv7-g1g-7.2b-20260523-ctx8192.pth" ;;
    *) echo "unknown model label: $1" >&2; exit 2 ;;
  esac
}

hf_dir_for_label() {
  case "$1" in
    0.1b) echo "${HF_ROOT}/rwkv7-g1d-0.1b-hf" ;;
    0.4b) echo "${HF_ROOT}/rwkv7-g1d-0.4b-hf" ;;
    1.5b) echo "${HF_ROOT}/rwkv7-g1g-1.5b-hf" ;;
    2.9b) echo "${HF_ROOT}/rwkv7-g1g-2.9b-hf" ;;
    7.2b) echo "${HF_ROOT}/rwkv7-g1g-7.2b-hf" ;;
    *) echo "unknown model label: $1" >&2; exit 2 ;;
  esac
}

batch_sizes_for_label() {
  case "$1" in
    0.1b|0.4b) echo "${BATCH_SIZES_04B:-1 2 4 8}" ;;
    1.5b) echo "${BATCH_SIZES_15B:-1 2 4}" ;;
    2.9b|7.2b) echo "${BATCH_SIZES_LARGE:-1 2}" ;;
    *) echo "1" ;;
  esac
}

require_inputs() {
  rwkv7_require_model "${VOCAB_FILE}"
  for label in ${ALL_MODEL_LABELS}; do
    rwkv7_require_model "$(pth_for_label "${label}")"
  done
}

require_hf_dir() {
  local label="$1"
  local hf_dir
  hf_dir="$(hf_dir_for_label "${label}")"
  rwkv7_require_model "${hf_dir}/config.json"
  rwkv7_require_model "${hf_dir}/tokenizer_config.json"
  if ! compgen -G "${hf_dir}/*.safetensors" >/dev/null; then
    echo "MODEL does not contain safetensors: ${hf_dir}" >&2
    exit 2
  fi
}

guard_a6000() {
  if [[ "${RUN_A6000_GUARD}" != "1" ]]; then
    return
  fi
  rwkv7_log "A6000 guard single_visible=${A6000_SINGLE_VISIBLE_DEVICES}"
  run_single_gpu "${PYTHON_BIN}" - <<PY
import torch

expected_name = ${EXPECTED_GPU_NAME@Q}
expected_cap = tuple(int(v) for v in ${EXPECTED_CAPABILITY@Q}.split("."))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print(f"visible_device_0={name} sm_{cap[0]}{cap[1]}")
if expected_name.lower() not in name.lower():
    raise SystemExit(f"expected {expected_name!r} in GPU name, got {name!r}")
if cap != expected_cap:
    raise SystemExit(f"expected capability {expected_cap}, got {cap}")
PY
}

print_environment() {
  if [[ "${RUN_ENV}" != "1" ]]; then
    return
  fi
  rwkv7_log "single-GPU environment"
  run_single_gpu PYTHON_BIN="${PYTHON_BIN}" bash scripts/print_env.sh
  rwkv7_log "multi-GPU environment"
  run_multi_gpu PYTHON_BIN="${PYTHON_BIN}" bash scripts/print_env.sh
}

convert_one() {
  local label="$1"
  local pth
  local hf_dir
  pth="$(pth_for_label "${label}")"
  hf_dir="$(hf_dir_for_label "${label}")"
  if [[ "${RUN_CONVERT}" != "1" ]]; then
    return
  fi
  if [[ -f "${hf_dir}/config.json" && -f "${hf_dir}/tokenizer_config.json" && "${FORCE_CONVERT:-0}" != "1" ]]; then
    if compgen -G "${hf_dir}/*.safetensors" >/dev/null; then
      rwkv7_log "skip convert ${label}: ${hf_dir} already exists"
      return
    fi
  fi
  rwkv7_log "convert ${label}"
  run "${PYTHON_BIN}" scripts/convert_rwkv7_to_hf.py \
    --input "${pth}" \
    --output "${hf_dir}" \
    --vocab-file "${VOCAB_FILE}" \
    --precision fp16 \
    --attn-mode fused_recurrent \
    --no-fuse-norm
}

run_batch_sweep_one() {
  local label="$1"
  local dtype="$2"
  local model
  local batches
  model="$(hf_dir_for_label "${label}")"
  batches="$(batch_sizes_for_label "${label}")"
  rwkv7_log "batch sweep label=${label} dtype=${dtype} batches=${batches}"
  # shellcheck disable=SC2086 # batch list is intentionally split.
  run_single_gpu "${PYTHON_BIN}" bench/bench_batch_sweep.py \
    --hf-dir "${model}" \
    --model-size-label "${label}" \
    --dtype "${dtype}" \
    --device cuda \
    --attn-mode fused_recurrent \
    --fuse-norm false \
    --fast-cache true \
    --fast-decode-api true \
    --fast-token-backend native_graph \
    --batch-sizes ${batches} \
    --prompt-tokens "${PROMPT_TOKENS}" \
    --decode-tokens "${DECODE_TOKENS}" \
    --warmup "${WARMUP}" \
    --runs "${RUNS}" \
    --results "${RESULTS}"
}

run_native_mm_quant_one() {
  local label="$1"
  local model
  model="$(hf_dir_for_label "${label}")"
  rwkv7_log "native mm quant label=${label}"
  run_single_gpu "${PYTHON_BIN}" bench/bench_native_mm_quant_decode.py \
    --hf-dir "${model}" \
    --model-size-label "${label}" \
    --device cuda \
    --dtype "${NATIVE_MM_QUANT_DTYPE}" \
    --quantizations none mm8 mm4 \
    --min-params "${NATIVE_MM_MIN_PARAMS}" \
    --prompt-tokens "${PROMPT_TOKENS}" \
    --decode-tokens "${DECODE_TOKENS}" \
    --warmup "${WARMUP}" \
    --runs "${RUNS}" \
    --optional \
    --results "${RESULTS}"
}

run_quant_one() {
  local label="$1"
  local model
  model="$(hf_dir_for_label "${label}")"
  rwkv7_log "bnb quant label=${label}"
  run_single_gpu "${PYTHON_BIN}" bench/bench_quantization.py \
    --hf-dir "${model}" \
    --model-size-label "${label}" \
    --device cuda \
    --dtype "${QUANT_DTYPE}" \
    --attn-mode fused_recurrent \
    --quantizations none 8bit 4bit \
    --prompt-tokens "${PROMPT_TOKENS}" \
    --decode-tokens "${DECODE_TOKENS}" \
    --warmup "${WARMUP}" \
    --runs "${RUNS}" \
    --results "${RESULTS}"
}

run_01b_baseline() {
  if [[ "${RUN_01B_BASELINE}" != "1" ]]; then
    return
  fi
  local label="0.1b"
  local model
  local pth
  model="$(hf_dir_for_label "${label}")"
  pth="$(pth_for_label "${label}")"
  rwkv7_log "0.1B core smoke"
  run_single_gpu "${PYTHON_BIN}" tests/smoke_hf_generate.py \
    --model "${model}" \
    --device cuda \
    --max-new-tokens "${SMOKE_MAX_NEW_TOKENS}"
  run_single_gpu "${PYTHON_BIN}" tests/test_hf_api_contract.py \
    --model "${model}" \
    --device cuda \
    --dtype "${QUANT_DTYPE}" \
    --attn-mode fused_recurrent \
    --fuse-norm false
  run_single_gpu "${PYTHON_BIN}" tests/test_quantized_inference.py \
    --model "${model}" \
    --device cuda \
    --dtype "${QUANT_DTYPE}" \
    --attn-mode fused_recurrent \
    --quantization 8bit \
    --optional
  run_single_gpu "${PYTHON_BIN}" tests/test_quantized_inference.py \
    --model "${model}" \
    --device cuda \
    --dtype "${QUANT_DTYPE}" \
    --attn-mode fused_recurrent \
    --quantization 4bit \
    --optional
  run_single_gpu "${PYTHON_BIN}" bench/bench_speed.py \
    --hf-dir "${model}" \
    --model-size-label "${label}" \
    --pth "${pth}" \
    --backend hf \
    --dtype "${QUANT_DTYPE}" \
    --device cuda \
    --prompt-tokens "${PROMPT_TOKENS}" \
    --decode-tokens "${DECODE_TOKENS}" \
    --warmup "${WARMUP}" \
    --runs "${RUNS}" \
    --attn-mode fused_recurrent \
    --fuse-norm false \
    --fast-cache true \
    --fast-token-backend native_graph \
    --results "${RESULTS}"
  run_batch_sweep_one "${label}" "${QUANT_DTYPE}"
  run_native_mm_quant_one "${label}"
  if [[ "${RUN_TRAINING}" == "1" ]]; then
    rwkv7_log "0.1B training smoke"
    run_single_gpu PYTHON_BIN="${PYTHON_BIN}" RESULTS="${RESULTS}" DEVICE=cuda TRAIN_DTYPE="${TRAIN_DTYPE}" \
      ATTN_MODE=fused_recurrent RUN_PEFT=1 RUN_TRAINER=1 RUN_RL=1 RUN_RESUME="${RUN_HF_RESUME}" \
      RL_BACKEND=dpo MAX_LENGTH="${TRAIN_MAX_LENGTH}" MAX_STEPS="${TRAIN_MAX_STEPS}" \
      BATCH_SIZE="${TRAIN_BATCH_SIZE}" RL_BATCH_SIZE="${TRAIN_RL_BATCH_SIZE}" \
      DATASET_REPEATS="${TRAIN_DATASET_REPEATS}" RESUME_FIRST_STEPS="${RESUME_FIRST_STEPS}" \
      RESUME_STEPS="${RESUME_STEPS}" bash scripts/run_hf_training_matrix.sh "${model}"
  fi
}

run_larger_smoke_one() {
  local label="$1"
  local dtype="$2"
  local model
  local pth
  model="$(hf_dir_for_label "${label}")"
  pth="$(pth_for_label "${label}")"
  rwkv7_log "larger model smoke label=${label} dtype=${dtype}"
  run_single_gpu "${PYTHON_BIN}" bench/bench_larger_model_smoke.py \
    --hf-dir "${model}" \
    --model-size-label "${label}" \
    --checkpoint-path "${pth}" \
    --device cuda \
    --dtype "${dtype}" \
    --attn-mode fused_recurrent \
    --fast-token-backend auto \
    --max-new-tokens "${SMOKE_MAX_NEW_TOKENS}" \
    --results "${RESULTS}"
}

run_training_one() {
  local label="$1"
  local model
  model="$(hf_dir_for_label "${label}")"
  rwkv7_log "single-GPU training label=${label}"
  run_single_gpu PYTHON_BIN="${PYTHON_BIN}" RESULTS="${RESULTS}" DEVICE=cuda TRAIN_DTYPE="${TRAIN_DTYPE}" \
    ATTN_MODE=fused_recurrent RUN_PEFT=1 RUN_TRAINER=1 RUN_RL=1 RUN_RESUME="${RUN_HF_RESUME}" \
    RL_BACKEND=dpo MAX_LENGTH="${TRAIN_MAX_LENGTH}" MAX_STEPS="${TRAIN_MAX_STEPS}" \
    BATCH_SIZE="${TRAIN_BATCH_SIZE}" RL_BATCH_SIZE="${TRAIN_RL_BATCH_SIZE}" \
    DATASET_REPEATS="${TRAIN_DATASET_REPEATS}" RESUME_FIRST_STEPS="${RESUME_FIRST_STEPS}" \
    RESUME_STEPS="${RESUME_STEPS}" bash scripts/run_hf_training_matrix.sh "${model}"
}

run_zero_one() {
  local label="$1"
  local model
  model="$(hf_dir_for_label "${label}")"
  if [[ "${RUN_ZERO}" == "1" ]]; then
    rwkv7_log "DeepSpeed ZeRO base label=${label}"
    run_multi_gpu PYTHON_BIN="${PYTHON_BIN}" RESULTS="${RESULTS}" DEVICE=cuda TRAIN_DTYPE="${TRAIN_DTYPE}" \
      ATTN_MODE=fused_recurrent ZERO_STAGE="${ZERO_STAGE}" NPROC_PER_NODE="${NPROC_PER_NODE}" \
      MAX_LENGTH="${ZERO_MAX_LENGTH}" MAX_STEPS="${ZERO_MAX_STEPS}" BATCH_SIZE="${ZERO_BATCH_SIZE}" \
      DATASET_REPEATS="${ZERO_DATASET_REPEATS}" bash scripts/run_zero_training_smoke.sh "${model}"
  fi
  if [[ "${RUN_ZERO_RESUME}" == "1" ]]; then
    rwkv7_log "DeepSpeed ZeRO resume label=${label}"
    run_multi_gpu "${PYTHON_BIN}" -m torch.distributed.run \
      --standalone \
      --nproc_per_node="${NPROC_PER_NODE}" \
      tests/test_deepspeed_resume_smoke.py \
      --model "${model}" \
      --model-size-label "${label}" \
      --zero-stage "${ZERO_STAGE}" \
      --attn-mode fused_recurrent \
      --train-dtype "${TRAIN_DTYPE}" \
      --first-steps "${ZERO_RESUME_FIRST_STEPS}" \
      --resume-steps "${ZERO_RESUME_STEPS}" \
      --batch-size "${ZERO_BATCH_SIZE}" \
      --dataset-repeats "${ZERO_DATASET_REPEATS}" \
      --max-length "${ZERO_MAX_LENGTH}" \
      --results "${RESULTS}"
  fi
}

run_analyze() {
  if [[ "${RUN_ANALYZE}" != "1" ]]; then
    return
  fi
  rwkv7_log "analyze A6000 rows"
  run "${PYTHON_BIN}" bench/analyze_results.py \
    --results "${RESULTS}" \
    --device "NVIDIA RTX A6000" \
    --dtype "${QUANT_DTYPE}" \
    --json
}

main() {
  rwkv7_log "A6000 HF validation results=${RESULTS}"
  rwkv7_log "python=${PYTHON_BIN} model_root=${MODEL_ROOT}"
  rwkv7_log "single CUDA_VISIBLE_DEVICES=${A6000_SINGLE_VISIBLE_DEVICES}; multi CUDA_VISIBLE_DEVICES=${A6000_MULTI_VISIBLE_DEVICES}"
  require_inputs
  rwkv7_prepare_results
  guard_a6000
  print_environment
  for label in ${ALL_MODEL_LABELS}; do
    convert_one "${label}"
  done
  for label in ${ALL_MODEL_LABELS}; do
    require_hf_dir "${label}"
  done
  run_01b_baseline
  for label in ${VALIDATION_MODEL_LABELS}; do
    if [[ "${RUN_LARGER_SMOKE}" == "1" ]]; then
      for dtype in ${INFER_DTYPES}; do
        run_larger_smoke_one "${label}" "${dtype}"
      done
    fi
    if [[ "${RUN_BATCH_SWEEP}" == "1" ]]; then
      for dtype in ${INFER_DTYPES}; do
        run_batch_sweep_one "${label}" "${dtype}"
      done
    fi
    if [[ "${RUN_QUANT}" == "1" ]]; then
      run_quant_one "${label}"
    fi
    if [[ "${RUN_NATIVE_MM_QUANT}" == "1" ]]; then
      run_native_mm_quant_one "${label}"
    fi
  done
  if [[ "${RUN_TRAINING}" == "1" ]]; then
    for label in ${TRAIN_MODEL_LABELS}; do
      run_training_one "${label}"
    done
  fi
  for label in ${ZERO_MODEL_LABELS}; do
    run_zero_one "${label}"
  done
  run_analyze
  rwkv7_log "A6000 HF validation complete"
}

main "$@"
