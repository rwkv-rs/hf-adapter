#!/usr/bin/env bash
# Tesla T4 / Turing sm_75 HF adapter validation.
#
# The T4 is an exact-card target.  This runner intentionally starts from the
# centralized Turing policy and does not inherit 3090/4090/5090 tuning flags.
# Pass settings as KEY=VALUE arguments, for example:
#
#   bash bench/run_t4_hf_validation.sh \
#     HF_DIR=/opt/models/rwkv7-g1d-0.1b-hf \
#     PYTHON_BIN=/opt/rwkv7-t4-venv/bin/python
set -euo pipefail

for arg in "$@"; do
  case "${arg}" in
    *=*) export "${arg}" ;;
    *) echo "unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

# Always benchmark the worktree that owns this runner. Invoking a script by
# path puts ``bench/`` (not the repository root) first on ``sys.path``; without
# this guard an older editable install can silently provide ``rwkv7_hf`` while
# the remote-code overlay uses the current checkout. That mixed-code process
# invalidates both performance and compatibility evidence.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_ROOT="${MODEL_ROOT:-/opt/models}"
MATRIX_MODE="${MATRIX_MODE:-short}"
FULL_MODEL_LABELS="${FULL_MODEL_LABELS:-0.1b 0.4b 1.5b 2.9b}"
MODEL_SIZE_LABEL="${MODEL_SIZE_LABEL:-0.1b}"
HF_DIR="${HF_DIR:-}"
OUT_DIR="${OUT_DIR:-bench/t4_turing_validation_$(date +%Y%m%d_%H%M%S)}"
RESULTS="${RESULTS:-${OUT_DIR}/results_t4.jsonl}"
DTYPE="${DTYPE:-fp16}"
DEVICE="${DEVICE:-cuda}"
EXPECTED_GPU_NAME="${EXPECTED_GPU_NAME:-Tesla T4}"
EXPECTED_GPU_CAPABILITY="${EXPECTED_GPU_CAPABILITY:-7.5}"
ALLOW_GPU_MISMATCH="${ALLOW_GPU_MISMATCH:-0}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8}"
PROMPT_TOKENS="${PROMPT_TOKENS:-32}"
DECODE_TOKENS="${DECODE_TOKENS:-32}"
DYNAMIC_BATCH_SIZE="${DYNAMIC_BATCH_SIZE:-4}"
PREFILL_PROMPT_TOKENS="${PREFILL_PROMPT_TOKENS:-512}"
PREFILL_BATCH_SIZES="${PREFILL_BATCH_SIZES:-1,2,4,8}"
CHUNK_SIZES="${CHUNK_SIZES:-64 128 256}"
WARMUP="${WARMUP:-2}"
RUNS="${RUNS:-2}"
STEPS="${STEPS:-32}"
RUN_FUNCTIONAL="${RUN_FUNCTIONAL:-1}"
RUN_TRAINING="${RUN_TRAINING:-1}"
RUN_PERF="${RUN_PERF:-1}"
RUN_PREFILL="${RUN_PREFILL:-1}"
RUN_FUSED_AB="${RUN_FUSED_AB:-1}"
RUN_QUANT="${RUN_QUANT:-1}"
RUN_NATIVE_MM_QUANT="${RUN_NATIVE_MM_QUANT:-1}"
RUN_ANALYZE="${RUN_ANALYZE:-1}"
RUN_TRAINER_RESUME="${RUN_TRAINER_RESUME:-0}"
RUN_TRL="${RUN_TRL:-0}"
RUN_DEEPSPEED="${RUN_DEEPSPEED:-0}"
RUN_LONG_PREFILL="${RUN_LONG_PREFILL:-0}"
T4_CHILD_RUN="${T4_CHILD_RUN:-0}"
TRAIN_DTYPE="${TRAIN_DTYPE:-auto}"
TRAIN_MAX_LENGTH="${TRAIN_MAX_LENGTH:-auto}"
PEFT_MAX_LOGIT_DIFF="${PEFT_MAX_LOGIT_DIFF:-auto}"

model_hf_dir() {
  case "$1" in
    0.1b) echo "${MODEL_ROOT}/rwkv7-g1d-0.1b-hf" ;;
    0.4b) echo "${MODEL_ROOT}/rwkv7-g1d-0.4b-hf" ;;
    1.5b) echo "${MODEL_ROOT}/rwkv7-g1g-1.5b-hf" ;;
    2.9b) echo "${MODEL_ROOT}/rwkv7-g1g-2.9b-hf" ;;
    *) echo "unknown model label: $1" >&2; return 2 ;;
  esac
}

model_batch_sizes() {
  case "$1" in
    0.1b|0.4b) echo "1 2 4 8" ;;
    1.5b) echo "1 2 4" ;;
    2.9b) echo "1 2" ;;
    *) return 2 ;;
  esac
}

model_prefill_batch_sizes() {
  case "$1" in
    0.1b|0.4b) echo "1,2,4,8" ;;
    1.5b) echo "1,2,4" ;;
    2.9b) echo "1,2" ;;
    *) return 2 ;;
  esac
}

model_dynamic_batch_size() {
  case "$1" in
    0.1b|0.4b|1.5b) echo "4" ;;
    2.9b) echo "2" ;;
    *) return 2 ;;
  esac
}

model_train_dtype() {
  case "$1" in
    0.1b|0.4b) echo "fp32" ;;
    1.5b|2.9b) echo "fp16" ;;
    *) return 2 ;;
  esac
}

model_train_max_length() {
  case "$1" in
    0.1b|0.4b) echo "24" ;;
    1.5b|2.9b) echo "16" ;;
    *) return 2 ;;
  esac
}

model_peft_max_logit_diff() {
  case "$1" in
    0.1b|0.4b) echo "0.0001" ;;
    # PEFT save/reload is exact on T4. Merging LoRA into fp16 weights rounds
    # individual logits by up to 0.1875 on the measured g1g checkpoints while
    # preserving greedy output, so the declared fp16 merge gate is 0.2.
    1.5b|2.9b) echo "0.2" ;;
    *) return 2 ;;
  esac
}

if [[ "${MATRIX_MODE}" == "full" && "${T4_CHILD_RUN}" != "1" ]]; then
  mkdir -p "${OUT_DIR}"
  : > "${RESULTS}"
  for model_label in ${FULL_MODEL_LABELS}; do
    MODEL_OUT_DIR="${OUT_DIR}/${model_label}"
    model_dir="$(model_hf_dir "${model_label}")"
    model_batches="$(model_batch_sizes "${model_label}")"
    model_prefill_batches="$(model_prefill_batch_sizes "${model_label}")"
    model_dynamic_batch="$(model_dynamic_batch_size "${model_label}")"
    if [[ "${TRAIN_DTYPE}" == "auto" ]]; then
      model_training_dtype="$(model_train_dtype "${model_label}")"
    else
      model_training_dtype="${TRAIN_DTYPE}"
    fi
    if [[ "${TRAIN_MAX_LENGTH}" == "auto" ]]; then
      model_training_length="$(model_train_max_length "${model_label}")"
    else
      model_training_length="${TRAIN_MAX_LENGTH}"
    fi
    if [[ "${PEFT_MAX_LOGIT_DIFF}" == "auto" ]]; then
      model_peft_diff="$(model_peft_max_logit_diff "${model_label}")"
    else
      model_peft_diff="${PEFT_MAX_LOGIT_DIFF}"
    fi
    if [[ ! -f "${model_dir}/config.json" ]]; then
      echo "missing converted model for ${model_label}: ${model_dir}" >&2
      exit 2
    fi
    T4_CHILD_RUN=1 MATRIX_MODE=short MODEL_SIZE_LABEL="${model_label}" \
      HF_DIR="${model_dir}" OUT_DIR="${MODEL_OUT_DIR}" RESULTS="${RESULTS}" \
      BATCH_SIZES="${model_batches}" PREFILL_BATCH_SIZES="${model_prefill_batches}" \
      DYNAMIC_BATCH_SIZE="${model_dynamic_batch}" \
      TRAIN_DTYPE="${model_training_dtype}" TRAIN_MAX_LENGTH="${model_training_length}" \
      PEFT_MAX_LOGIT_DIFF="${model_peft_diff}" \
      RUN_ANALYZE=0 bash "$0"
  done
  if [[ "${RUN_ANALYZE}" != "0" && -s "${RESULTS}" ]]; then
    "${PYTHON_BIN}" bench/analyze_results.py \
      --results "${RESULTS}" --device "${EXPECTED_GPU_NAME}" --dtype "${DTYPE}" --json \
      > "${RESULTS%.jsonl}.report.json"
  fi
  echo "wrote ${OUT_DIR}"
  echo "wrote ${RESULTS}"
  exit 0
fi

if [[ -z "${HF_DIR}" ]]; then
  HF_DIR="$(model_hf_dir "${MODEL_SIZE_LABEL}")"
fi
if [[ "${TRAIN_DTYPE}" == "auto" ]]; then
  TRAIN_DTYPE="$(model_train_dtype "${MODEL_SIZE_LABEL}")"
fi
if [[ "${TRAIN_MAX_LENGTH}" == "auto" ]]; then
  TRAIN_MAX_LENGTH="$(model_train_max_length "${MODEL_SIZE_LABEL}")"
fi
if [[ "${PEFT_MAX_LOGIT_DIFF}" == "auto" ]]; then
  PEFT_MAX_LOGIT_DIFF="$(model_peft_max_logit_diff "${MODEL_SIZE_LABEL}")"
fi

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RWKV_V7_ON="${RWKV_V7_ON:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5}"
# PyTorch 2.7 / Triton 3.3 Inductor workers import the removed legacy
# AttrsDescriptor path on this T4 stack.  Set the import-time guard explicitly
# for deterministic runner logs; remote-code loading also applies the same
# measured fallback through triton_compat.py.
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"

run() {
  echo "+ $*" >&2
  "$@"
}

mkdir -p "${OUT_DIR}"

"${PYTHON_BIN}" - "${EXPECTED_GPU_NAME}" "${EXPECTED_GPU_CAPABILITY}" "${ALLOW_GPU_MISMATCH}" <<'PY'
import sys
import torch

expected_name, expected_capability, allow_mismatch = sys.argv[1:]
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
if torch.cuda.device_count() != 1:
    raise SystemExit(f"expected one visible GPU, got {torch.cuda.device_count()}")
name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
capability = f"{major}.{minor}"
if allow_mismatch != "1" and expected_name.lower() not in name.lower():
    raise SystemExit(f"expected GPU name containing {expected_name!r}, got {name!r}")
if allow_mismatch != "1" and capability != expected_capability:
    raise SystemExit(f"expected capability {expected_capability}, got {capability}")
PY

{
  echo "# Tesla T4 HF validation"
  echo "date=$(date -Is)"
  echo "commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
  echo "model_size_label=${MODEL_SIZE_LABEL} matrix_mode=${MATRIX_MODE}"
  echo "hf_dir=${HF_DIR}"
  echo "out_dir=${OUT_DIR} results=${RESULTS}"
  echo "dtype=${DTYPE} device=${DEVICE} batch_sizes=${BATCH_SIZES}"
  echo "train_dtype=${TRAIN_DTYPE} train_max_length=${TRAIN_MAX_LENGTH} peft_max_logit_diff=${PEFT_MAX_LOGIT_DIFF}"
  echo "torch_compile_disable=${TORCH_COMPILE_DISABLE}"
  nvidia-smi --query-gpu=name,uuid,compute_cap,driver_version,memory.total,pstate,temperature.gpu \
    --format=csv,noheader
  "${PYTHON_BIN}" - <<'PY'
import importlib
import torch

print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
for module_name in ("transformers", "triton", "bitsandbytes", "peft", "trl", "fla"):
    try:
        module = importlib.import_module(module_name)
        print(module_name, getattr(module, "__version__", "unknown"))
    except Exception as exc:
        print(module_name, "unavailable", type(exc).__name__, str(exc))
PY
} | tee "${OUT_DIR}/env.log"

run "${PYTHON_BIN}" -m py_compile \
  rwkv7_hf/kernel_policy.py \
  rwkv7_hf/modeling_rwkv7.py \
  rwkv7_hf/native_jit.py \
  bench/bench_batch_sweep.py \
  bench/bench_native_graph_overhead.py \
  bench/bench_native_prefill_scan.py \
  bench/bench_chunked_prefill.py \
  bench/bench_native_graph_fused_output.py \
  bench/bench_native_graph_fused_recurrent_output.py \
  bench/bench_quantization.py \
  bench/bench_native_mm_quant_decode.py \
  bench/bench_native_quant_e2e_decode.py \
  tests/test_native_trainer_resume_smoke.py \
  tests/test_native_sft_smoke.py \
  tests/test_native_dpo_smoke.py \
  tests/test_native_grpo_smoke.py \
  tests/test_deepspeed_training_smoke.py \
  tests/test_deepspeed_resume_smoke.py \
  bench/analyze_results.py

if [[ "${RUN_FUNCTIONAL}" != "0" ]]; then
  run "${PYTHON_BIN}" tests/smoke_hf_generate.py \
    --model "${HF_DIR}" --device "${DEVICE}" --max-new-tokens 4 \
    | tee "${OUT_DIR}/smoke_hf_generate.log"
  run "${PYTHON_BIN}" tests/test_hf_api_contract.py \
    --model "${HF_DIR}" --device "${DEVICE}" --dtype "${DTYPE}" --beam-new-tokens 2 \
    | tee "${OUT_DIR}/hf_api_contract.log"
  run "${PYTHON_BIN}" tests/test_batch_cache.py \
    --model "${HF_DIR}" --device "${DEVICE}" --dtype "${DTYPE}" \
    --batch-sizes ${BATCH_SIZES} --prompt-tokens "${PROMPT_TOKENS}" --decode-steps 4 \
    | tee "${OUT_DIR}/batch_cache.log"
  run "${PYTHON_BIN}" tests/test_dynamic_batch_cache.py \
    --model "${HF_DIR}" --device "${DEVICE}" --dtype "${DTYPE}" \
    --batch-size "${DYNAMIC_BATCH_SIZE}" --prompt-tokens "${PROMPT_TOKENS}" --decode-steps 4 \
    | tee "${OUT_DIR}/dynamic_batch_cache.log"
  run "${PYTHON_BIN}" tests/test_chunked_prefill.py \
    --model "${HF_DIR}" --device "${DEVICE}" --dtype "${DTYPE}" \
    --batch-size 2 --chunk-sizes 1 8 32 64 \
    | tee "${OUT_DIR}/chunked_prefill_correctness.log"
fi

if [[ "${RUN_TRAINING}" != "0" ]]; then
  run "${PYTHON_BIN}" tests/test_native_trainer_smoke.py \
    --model "${HF_DIR}" --dtype "${TRAIN_DTYPE}" --max-steps 6 --batch-size 1 --length "${TRAIN_MAX_LENGTH}" \
    | tee "${OUT_DIR}/native_trainer_smoke.log"
  run "${PYTHON_BIN}" tests/test_native_peft_save_load_merge.py \
    --model "${HF_DIR}" --device "${DEVICE}" --dtype "${TRAIN_DTYPE}" --steps 1 \
    --max-logit-diff "${PEFT_MAX_LOGIT_DIFF}" \
    | tee "${OUT_DIR}/native_peft_save_load_merge.log"
fi

if [[ "${RUN_TRAINER_RESUME}" != "0" ]]; then
  run "${PYTHON_BIN}" tests/test_native_trainer_resume_smoke.py \
    --model "${HF_DIR}" --device "${DEVICE}" --dtype "${TRAIN_DTYPE}" \
    --first-steps 1 --resume-steps 2 --batch-size 1 --length "${TRAIN_MAX_LENGTH}" \
    | tee "${OUT_DIR}/native_trainer_resume.log"
fi

if [[ "${RUN_TRL}" != "0" ]]; then
  run "${PYTHON_BIN}" tests/test_native_sft_smoke.py \
    --model "${HF_DIR}" --device "${DEVICE}" --dtype "${TRAIN_DTYPE}" \
    --max-steps 1 --batch-size 1 --max-length "${TRAIN_MAX_LENGTH}" \
    | tee "${OUT_DIR}/native_sft_smoke.log"
  run "${PYTHON_BIN}" tests/test_native_dpo_smoke.py \
    --model "${HF_DIR}" --dtype "${TRAIN_DTYPE}" \
    --max-steps 1 --batch-size 1 --max-length "${TRAIN_MAX_LENGTH}" \
    | tee "${OUT_DIR}/native_dpo_smoke.log"
  run "${PYTHON_BIN}" tests/test_native_grpo_smoke.py \
    --model "${HF_DIR}" --dtype "${TRAIN_DTYPE}" \
    --max-steps 1 --batch-size 1 --max-completion-length 4 \
    | tee "${OUT_DIR}/native_grpo_smoke.log"
fi

if [[ "${RUN_DEEPSPEED}" != "0" ]]; then
  run "${PYTHON_BIN}" tests/test_deepspeed_training_smoke.py \
    --model "${HF_DIR}" --model-size-label "${MODEL_SIZE_LABEL}" \
    --zero-stage both --max-length "${TRAIN_MAX_LENGTH}" --train-dtype "${TRAIN_DTYPE}" \
    --max-steps 1 --batch-size 1 --gradient-accumulation-steps 1 \
    --optional --results "${RESULTS}" \
    | tee "${OUT_DIR}/deepspeed_training_smoke.log"
  run "${PYTHON_BIN}" tests/test_deepspeed_resume_smoke.py \
    --model "${HF_DIR}" --model-size-label "${MODEL_SIZE_LABEL}" \
    --zero-stage both --max-length "${TRAIN_MAX_LENGTH}" --train-dtype "${TRAIN_DTYPE}" \
    --first-steps 1 --resume-steps 2 --batch-size 1 \
    --gradient-accumulation-steps 1 --results "${RESULTS}" \
    | tee "${OUT_DIR}/deepspeed_resume_smoke.log"
fi

if [[ "${RUN_PERF}" != "0" ]]; then
  run "${PYTHON_BIN}" bench/bench_batch_sweep.py \
    --hf-dir "${HF_DIR}" --model-size-label "${MODEL_SIZE_LABEL}" \
    --dtype "${DTYPE}" --device "${DEVICE}" \
    --attn-mode fused_recurrent --fuse-norm false --fast-cache true \
    --fast-decode-api true --fast-token-backend native_graph \
    --batch-sizes ${BATCH_SIZES} \
    --prompt-tokens "${PROMPT_TOKENS}" --decode-tokens "${DECODE_TOKENS}" \
    --warmup "${WARMUP}" --runs "${RUNS}" --results "${RESULTS}" \
    | tee "${OUT_DIR}/batch_sweep.log"
  run "${PYTHON_BIN}" bench/bench_native_graph_overhead.py \
    --hf-dir "${HF_DIR}" --dtype "${DTYPE}" --device "${DEVICE}" \
    --attn-mode fused_recurrent --fuse-norm false --fast-cache true \
    --batch-sizes ${BATCH_SIZES} --prompt-tokens "${PROMPT_TOKENS}" \
    --warmup "${WARMUP}" --steps "${STEPS}" --results "${RESULTS}" \
    | tee "${OUT_DIR}/native_graph_overhead.log"
fi

if [[ "${RUN_LONG_PREFILL}" != "0" ]]; then
  run "${PYTHON_BIN}" bench/bench_chunked_prefill.py \
    --hf-dir "${HF_DIR}" --dtype "${DTYPE}" --device "${DEVICE}" \
    --attn-mode fused_recurrent --fuse-norm false \
    --batch-size 1 --prompt-tokens 2048 --chunk-sizes 128 256 512 \
    --warmup 1 --runs 1 --results "${RESULTS}" \
    | tee "${OUT_DIR}/chunked_prefill_long.log"
fi

if [[ "${RUN_PREFILL}" != "0" ]]; then
  run "${PYTHON_BIN}" bench/bench_chunked_prefill.py \
    --hf-dir "${HF_DIR}" --dtype "${DTYPE}" --device "${DEVICE}" \
    --attn-mode fused_recurrent --fuse-norm false \
    --batch-size 1 --prompt-tokens "${PREFILL_PROMPT_TOKENS}" \
    --chunk-sizes ${CHUNK_SIZES} --warmup "${WARMUP}" --runs "${RUNS}" \
    --results "${RESULTS}" \
    | tee "${OUT_DIR}/chunked_prefill.log"
  for fused_scan in false true; do
    run "${PYTHON_BIN}" bench/bench_native_prefill_scan.py \
      --model "${HF_DIR}" --device "${DEVICE}" --dtype "${DTYPE}" \
      --batch-sizes "${PREFILL_BATCH_SIZES}" --prompt-tokens "${PREFILL_PROMPT_TOKENS}" \
      --fused-scan "${fused_scan}" --code-source repo \
      --warmup "${WARMUP}" --steps "${RUNS}" --results "${RESULTS}" \
      | tee "${OUT_DIR}/native_prefill_scan_${fused_scan}.log"
  done
fi

if [[ "${RUN_FUSED_AB}" != "0" ]]; then
  for batch_size in ${BATCH_SIZES}; do
    run "${PYTHON_BIN}" bench/bench_native_graph_fused_output.py \
      --hf-dir "${HF_DIR}" --dtype "${DTYPE}" --device "${DEVICE}" \
      --attn-mode fused_recurrent --fuse-norm false --fast-cache true \
      --batch-size "${batch_size}" --prompt-tokens "${PROMPT_TOKENS}" \
      --warmup "${WARMUP}" --steps "${STEPS}" --results "${RESULTS}" \
      | tee "${OUT_DIR}/fused_output_bsz${batch_size}.log"
    run "${PYTHON_BIN}" bench/bench_native_graph_fused_recurrent_output.py \
      --hf-dir "${HF_DIR}" --dtype "${DTYPE}" --device "${DEVICE}" \
      --attn-mode fused_recurrent --fuse-norm false --fast-cache true \
      --batch-size "${batch_size}" --prompt-tokens "${PROMPT_TOKENS}" \
      --warmup "${WARMUP}" --steps "${STEPS}" --results "${RESULTS}" \
      | tee "${OUT_DIR}/fused_recurrent_output_bsz${batch_size}.log"
  done
fi

if [[ "${RUN_QUANT}" != "0" ]]; then
  # T4's measured FLA 0.5.0 / Triton 3.3.1 stack cannot lower the long
  # sequence WY kernel.  Exercise bnb through the native/no-FLA HF model,
  # which still loads through AutoModelForCausalLM and BitsAndBytesConfig.
  run env RWKV7_NATIVE_MODEL=1 "${PYTHON_BIN}" bench/bench_quantization.py \
    --hf-dir "${HF_DIR}" --model-size-label "${MODEL_SIZE_LABEL}" \
    --dtype "${DTYPE}" --device "${DEVICE}" --attn-mode fused_recurrent \
    --quantizations none 8bit 4bit --quant-skip-policy memory \
    --prompt-tokens 128 --decode-tokens 16 --warmup "${WARMUP}" --runs "${RUNS}" \
    --optional --results "${RESULTS}" \
    | tee "${OUT_DIR}/bnb_quant_memory.log"
fi

if [[ "${RUN_NATIVE_MM_QUANT}" != "0" ]]; then
  for policy in memory speed; do
    run "${PYTHON_BIN}" bench/bench_native_mm_quant_decode.py \
      --hf-dir "${HF_DIR}" --model-size-label "${MODEL_SIZE_LABEL}" \
      --dtype "${DTYPE}" --device "${DEVICE}" \
      --quantizations none mm8 mm4 --policy "${policy}" \
      --prompt-tokens 128 --decode-tokens 16 --warmup "${WARMUP}" --runs "${RUNS}" \
      --optional --results "${RESULTS}" \
      | tee "${OUT_DIR}/native_mm_quant_${policy}.log"
  done
  run "${PYTHON_BIN}" bench/bench_native_quant_e2e_decode.py \
    --hf-dir "${HF_DIR}" --model-size-label "${MODEL_SIZE_LABEL}" \
    --dtype "${DTYPE}" --device "${DEVICE}" --attn-mode fused_recurrent \
    --fuse-norm false --fast-cache true --fast-token-backend native_graph \
    --quantizations none mm8 mm4 --policy speed \
    --prompt-tokens 64 --decode-tokens 32 --warmup "${WARMUP}" \
    --results "${RESULTS}" \
    | tee "${OUT_DIR}/native_quant_e2e_decode.log"
fi

if [[ "${RUN_ANALYZE}" != "0" && -s "${RESULTS}" ]]; then
  run "${PYTHON_BIN}" bench/analyze_results.py \
    --results "${RESULTS}" --device "${EXPECTED_GPU_NAME}" --dtype "${DTYPE}" --json \
    > "${RESULTS%.jsonl}.report.json"
else
  echo "SKIP analyzer: RUN_ANALYZE=${RUN_ANALYZE} results_missing_or_empty=${RESULTS}" \
    | tee "${OUT_DIR}/analyzer_skip.log"
fi

echo "wrote ${OUT_DIR}"
echo "wrote ${RESULTS}"
if [[ -f "${RESULTS%.jsonl}.report.json" ]]; then
  echo "wrote ${RESULTS%.jsonl}.report.json"
fi
