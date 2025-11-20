#!/usr/bin/env bash
# Run one bsz=8 RTX 3090 pair with separate dense and RWKV-quant gates.
#
# The public W8/W4 contract has two independent implementation concerns:
#   * BnB memory lanes demonstrate material end-to-end footprint reduction.
#   * Vendored A8W8/MM4 speed lanes demonstrate no regression versus fp16.
#   * The hybrid W8 lane keeps BnB-compressed blocks and replaces the skipped
#     dense lm_head with the vendored A8W8 kernel.
# Dense RWKV is compared with dense Qwen3.5. Quantized RWKV is compared only
# with the matching RWKV dense row for speed, footprint, peak VRAM and quality;
# Qwen quantization is not an acceptance dependency. The fail-closed route
# composer retains every implementation and selects one measured W8/W4 route
# per exact shape only when all RWKV-local gates pass.
set -uo pipefail

PAIR_LABEL="${PAIR_LABEL:-${1:-}}"
RWKV_MODEL="${RWKV_MODEL:-${2:-}}"
QWEN_MODEL="${QWEN_MODEL:-${3:-}}"
OUT_DIR="${OUT_DIR:-${4:-}}"
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
WARMUP="${WARMUP:-1}"
RUNS="${RUNS:-3}"
PREFILL_CHUNK_SIZE="${PREFILL_CHUNK_SIZE:-0}"
BATCH_SIZES="${BATCH_SIZES:-8}"

if [[ -z "${PAIR_LABEL}" || -z "${RWKV_MODEL}" || -z "${QWEN_MODEL}" || -z "${OUT_DIR}" ]]; then
  echo "usage: $0 PAIR_LABEL RWKV_MODEL QWEN_MODEL OUT_DIR" >&2
  exit 2
fi
if [[ ! -d "${RWKV_MODEL}" || ! -d "${QWEN_MODEL}" ]]; then
  echo "both RWKV_MODEL and QWEN_MODEL must be local model directories" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"
rm -f \
  "${OUT_DIR}"/{dense,memory,reference_quant,native_speed,hybrid_speed,combined_memory,combined_speed,combined_auto}.jsonl \
  "${OUT_DIR}"/{progress.log,matrix_failures.txt,compose_exit_code.txt,compare_memory_exit_code.txt,compare_speed_exit_code.txt}

export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

rwkv_size="${PAIR_LABEL#rwkv-}"
rwkv_size="${rwkv_size%%__*}"
qwen_size="${PAIR_LABEL##*qwen3.5-}"
failures=0
case "${PAIR_LABEL}" in
  rwkv-1.5b__qwen3.5-2b) default_dense_gate="1.65" ;;
  rwkv-2.9b__qwen3.5-4b) default_dense_gate="1.75" ;;
  rwkv-7.2b__qwen3.5-9b) default_dense_gate="1.50" ;;
  *) default_dense_gate="1.05" ;;
esac
DENSE_SPEEDUP_GATE="${DENSE_SPEEDUP_GATE:-${default_dense_gate}}"
read -r -a batch_size_args <<< "${BATCH_SIZES}"
expected_base_cells=$(( ${#batch_size_args[@]} * 3 * 2 ))
expected_total_cells=$(( expected_base_cells * 3 ))

run_sweep() {
  local label="$1" role="$2" kind="$3" model="$4" size="$5" quant="$6" output="$7"
  shift 7
  local extra=("$@")
  printf 'START %s\n' "${label}" | tee -a "${OUT_DIR}/progress.log"
  "${PYTHON_BIN}" bench/bench_cross_model_speed_resident.py \
    --model "${model}" --model-kind "${kind}" --model-role "${role}" \
    --model-pair "${PAIR_LABEL}" --model-size-label "${size}" \
    --benchmark-matrix qwen35_3090_hf_final --dtype fp16 --quantization "${quant}" \
    --device cuda --batch-sizes "${batch_size_args[@]}" --prompt-tokens 128 512 2048 \
    --decode-tokens 128 512 --prefill-chunk-size "${PREFILL_CHUNK_SIZE}" \
    --warmup "${WARMUP}" --runs "${RUNS}" --rwkv-attn-mode fused_recurrent \
    --rwkv-code-source repo --qwen-backend auto "${extra[@]}" --results "${output}" \
    > "${OUT_DIR}/${label}.log" 2>&1
  local rc=$?
  printf 'DONE %s rc=%s\n' "${label}" "${rc}" | tee -a "${OUT_DIR}/progress.log"
  [[ ${rc} -eq 0 ]] || failures=$((failures + 1))
}

export RWKV7_FAST_TOKEN_BACKEND=native_graph
export RWKV7_NATIVE_PREFILL_GRAPH=1
run_sweep dense_candidate candidate rwkv "${RWKV_MODEL}" "${rwkv_size}" none "${OUT_DIR}/dense.jsonl"

unset RWKV7_FAST_TOKEN_BACKEND RWKV7_NATIVE_PREFILL_GRAPH
run_sweep dense_reference reference qwen35 "${QWEN_MODEL}" "${qwen_size}" none \
  "${OUT_DIR}/dense.jsonl" --require-qwen-fast-path

export RWKV7_FAST_TOKEN_BACKEND=native_graph
export RWKV7_NATIVE_PREFILL_GRAPH=1
export RWKV7_BNB_SKIP_POLICY=memory
export RWKV7_BNB_INT8_THRESHOLD=0
for quant in bnb8 bnb4; do
  run_sweep "memory_candidate_${quant}" candidate rwkv "${RWKV_MODEL}" "${rwkv_size}" \
    "${quant}" "${OUT_DIR}/memory.jsonl"
done

export RWKV7_FAST_TOKEN_BACKEND=native_graph
export RWKV7_NATIVE_PREFILL_GRAPH=1
export RWKV7_NATIVE_PREFILL_EXTERNAL_QUANT_GRAPH=1
export RWKV7_BNB_SKIP_POLICY=memory
export RWKV7_BNB_INT8_THRESHOLD=0
run_sweep hybrid_speed_bnb8_a8w8_head candidate rwkv "${RWKV_MODEL}" "${rwkv_size}" \
  bnb8_a8w8_head "${OUT_DIR}/hybrid_speed.jsonl" \
  --native-quant-policy speed --native-quant-min-params 1

export RWKV7_FAST_TOKEN_BACKEND=native_graph
export RWKV7_NATIVE_PREFILL_GRAPH=1
run_sweep native_speed_a8w8 candidate rwkv "${RWKV_MODEL}" "${rwkv_size}" a8w8 \
  "${OUT_DIR}/native_speed.jsonl" --native-quant-policy speed --native-quant-min-params 1
run_sweep native_speed_mm4 candidate rwkv "${RWKV_MODEL}" "${rwkv_size}" mm4 \
  "${OUT_DIR}/native_speed.jsonl" --native-quant-policy speed --native-quant-min-params 1
run_sweep native_speed_torchao_w4 candidate rwkv "${RWKV_MODEL}" "${rwkv_size}" torchao_w4 \
  "${OUT_DIR}/native_speed.jsonl" --native-quant-policy speed \
  --native-quant-min-params 1 --torchao-group-size 128

cat "${OUT_DIR}/dense.jsonl" "${OUT_DIR}/memory.jsonl" > "${OUT_DIR}/combined_memory.jsonl"
cat "${OUT_DIR}/combined_memory.jsonl" "${OUT_DIR}/native_speed.jsonl" \
  "${OUT_DIR}/hybrid_speed.jsonl" \
  > "${OUT_DIR}/combined_speed.jsonl"

"${PYTHON_BIN}" bench/compose_qwen35_quant_routes.py \
  --results "${OUT_DIR}/dense.jsonl" \
  --results "${OUT_DIR}/memory.jsonl" \
  --results "${OUT_DIR}/native_speed.jsonl" \
  --results "${OUT_DIR}/hybrid_speed.jsonl" \
  --output "${OUT_DIR}/combined_auto.jsonl" \
  --manifest "${OUT_DIR}/route_manifest.json" \
  --no-quant-qwen-gate --fail-on-gate
compose_rc=$?
printf '%s\n' "${compose_rc}" > "${OUT_DIR}/compose_exit_code.txt"

common_compare=(
  --expected-cells "${expected_total_cells}"
  --min-prefill-speedup "${DENSE_SPEEDUP_GATE}" --min-decode-speedup "${DENSE_SPEEDUP_GATE}"
  --require-native-candidate --require-qwen-fast-path
  --require-quant-memory-reduction --require-prefill-mode-match
  --fail-on-gate
)

"${PYTHON_BIN}" bench/compare_qwen35_speed_matrix.py \
  --results "${OUT_DIR}/combined_auto.jsonl" "${common_compare[@]}" \
  --min-quant-prefill-speedup 0.00 --min-quant-decode-speedup 0.00 \
  --json-output "${OUT_DIR}/summary_memory.json" \
  --markdown-output "${OUT_DIR}/summary_memory.md"
memory_rc=$?
printf '%s\n' "${memory_rc}" > "${OUT_DIR}/compare_memory_exit_code.txt"

"${PYTHON_BIN}" bench/compare_qwen35_speed_matrix.py \
  --results "${OUT_DIR}/combined_auto.jsonl" "${common_compare[@]}" \
  --min-quant-prefill-speedup 0.00 --min-quant-decode-speedup 0.00 \
  --require-quant-not-slower-than-dense \
  --json-output "${OUT_DIR}/summary_speed.json" \
  --markdown-output "${OUT_DIR}/summary_speed.md"
speed_rc=$?
printf '%s\n' "${speed_rc}" > "${OUT_DIR}/compare_speed_exit_code.txt"

printf '%s\n' "${failures}" > "${OUT_DIR}/matrix_failures.txt"
[[ ${failures} -eq 0 && ${compose_rc} -eq 0 && ${memory_rc} -eq 0 && ${speed_rc} -eq 0 ]]
