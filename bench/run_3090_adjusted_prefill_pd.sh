#!/usr/bin/env bash
# Reproduce the RTX 3090 latest-checkpoint parameter-adjusted prefill gate.
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT_DIR="${OUT_DIR:-${1:-}}"
RWKV_PYTHON_BIN="${RWKV_PYTHON_BIN:-python}"
QWEN_PYTHON_BIN="${QWEN_PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
WARMUP="${WARMUP:-2}"
RUNS="${RUNS:-5}"
SOURCE_COMMIT="${SOURCE_COMMIT:-}"
BENCHMARK_MATRIX="${BENCHMARK_MATRIX:-qwen35_3090_g1i_pd_20260812}"

required=(
  RWKV_04_MODEL RWKV_15_MODEL RWKV_29_MODEL RWKV_72_MODEL
  QWEN_08_MODEL QWEN_2_MODEL QWEN_4_MODEL QWEN_9_MODEL
)
if [[ -z "${OUT_DIR}" ]]; then
  echo "OUT_DIR and all eight *_MODEL variables are required" >&2
  exit 2
fi
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" || ! -d "${!name}" ]]; then
    echo "${name} must name a local model directory" >&2
    exit 2
  fi
done

mkdir -p "${OUT_DIR}/logs"
nvidia-smi \
  --query-gpu=name,driver_version,memory.total,power.limit,clocks.current.sm,clocks.max.sm \
  --format=csv > "${OUT_DIR}/system.csv"
if [[ -n "${SOURCE_COMMIT}" ]]; then
  printf '%s\n' "${SOURCE_COMMIT}" > "${OUT_DIR}/source_commit.txt"
else
  git -C "${ROOT}" rev-parse HEAD > "${OUT_DIR}/source_commit.txt"
fi
{
  for name in "${required[@]}"; do
    printf '[%s]\n' "${name}"
    (cd "${!name}" && sha256sum config.json)
    find "${!name}" -maxdepth 1 -type f -name '*.safetensors' \
      ! -name 'SHA256SUMS.safetensors' -print0 \
      | sort -z \
      | xargs -0 --no-run-if-empty sha256sum \
      | sed "s#${!name}/#<model>/#g"
  done
} > "${OUT_DIR}/model_hashes.txt"

export CUDA_VISIBLE_DEVICES
for python_bin in "${RWKV_PYTHON_BIN}" "${QWEN_PYTHON_BIN}"; do
  gpu_name="$(${python_bin} - <<'PY'
import torch
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY
)"
  "${python_bin}" "${ROOT}/bench/check_exact_gpu.py" --model 3090 --name "${gpu_name}"
done

"${RWKV_PYTHON_BIN}" - > "${OUT_DIR}/rwkv_runtime.json" <<'PY'
import json
from importlib.metadata import version
import torch, transformers, triton
actual = (
    str(torch.__version__), str(torch.version.cuda), str(triton.__version__),
    str(transformers.__version__), version("accelerate"),
)
expected = ("2.7.1+cu126", "12.6", "3.3.1", "5.12.1", "1.14.0")
assert actual == expected, f"RWKV runtime {actual} != validated {expected}"
assert hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation")
print(json.dumps({
    "torch": actual[0], "cuda": actual[1], "triton": actual[2],
    "transformers": actual[3], "accelerate": actual[4],
}))
PY
"${QWEN_PYTHON_BIN}" - > "${OUT_DIR}/qwen_runtime.json" <<'PY'
import json
from importlib.metadata import version
import torch, transformers, triton
actual = (
    str(torch.__version__), str(torch.version.cuda), str(triton.__version__),
    str(transformers.__version__), version("flash-linear-attention"),
    version("bitsandbytes"), version("accelerate"),
)
expected = (
    "2.6.0+cu124", "12.4", "3.2.0", "5.12.1", "0.5.1", "0.49.2",
    "1.14.0",
)
assert actual == expected, f"Qwen runtime {actual} != validated {expected}"
print(json.dumps({
    "torch": actual[0], "cuda": actual[1], "triton": actual[2],
    "transformers": actual[3], "fla": actual[4],
    "bitsandbytes": actual[5], "accelerate": actual[6],
}))
PY

rm -f "${OUT_DIR}/results.jsonl" "${OUT_DIR}/summary.json" "${OUT_DIR}/summary.md"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "${ROOT}"

run_matrix() {
  local python_bin="$1" role="$2" kind="$3" model="$4" pair="$5"
  local size="$6" label="$7"
  shift 7
  "${python_bin}" bench/bench_cross_model_speed_resident.py \
    --model "${model}" --model-kind "${kind}" --model-role "${role}" \
    --model-pair "${pair}" --model-size-label "${size}" \
    --benchmark-matrix "${BENCHMARK_MATRIX}" \
    --dtype fp16 --quantization none --device cuda \
    --batch-sizes 1 8 --prompt-tokens 128 512 2048 \
    --decode-tokens 128 --prefill-chunk-size 512 \
    --warmup "${WARMUP}" --runs "${RUNS}" "$@" \
    --results "${OUT_DIR}/results.jsonl" \
    > "${OUT_DIR}/logs/${label}.log" 2>&1
}

export RWKV7_FAST_TOKEN_BACKEND=native_graph
run_matrix "${RWKV_PYTHON_BIN}" candidate rwkv "${RWKV_04_MODEL}" \
  rwkv-0.4b__qwen3.5-0.8b 0.4b candidate_0p4 \
  --rwkv-attn-mode fused_recurrent --rwkv-code-source repo
run_matrix "${RWKV_PYTHON_BIN}" candidate rwkv "${RWKV_15_MODEL}" \
  rwkv-1.5b__qwen3.5-2b 1.5b candidate_1p5 \
  --rwkv-attn-mode fused_recurrent --rwkv-code-source repo
run_matrix "${RWKV_PYTHON_BIN}" candidate rwkv "${RWKV_29_MODEL}" \
  rwkv-2.9b__qwen3.5-4b 2.9b candidate_2p9 \
  --rwkv-attn-mode fused_recurrent --rwkv-code-source repo
run_matrix "${RWKV_PYTHON_BIN}" candidate rwkv "${RWKV_72_MODEL}" \
  rwkv-7.2b__qwen3.5-9b 7.2b candidate_7p2 \
  --rwkv-attn-mode fused_recurrent --rwkv-code-source repo
unset RWKV7_FAST_TOKEN_BACKEND

run_matrix "${QWEN_PYTHON_BIN}" reference qwen35 "${QWEN_08_MODEL}" \
  rwkv-0.4b__qwen3.5-0.8b 0.8b reference_0p8 \
  --qwen-backend fla --qwen-conv-backend fla_triton --require-qwen-fast-path
run_matrix "${QWEN_PYTHON_BIN}" reference qwen35 "${QWEN_2_MODEL}" \
  rwkv-1.5b__qwen3.5-2b 2b reference_2b \
  --qwen-backend fla --qwen-conv-backend fla_triton --require-qwen-fast-path
run_matrix "${QWEN_PYTHON_BIN}" reference qwen35 "${QWEN_4_MODEL}" \
  rwkv-2.9b__qwen3.5-4b 4b reference_4b \
  --qwen-backend fla --qwen-conv-backend fla_triton --require-qwen-fast-path
run_matrix "${QWEN_PYTHON_BIN}" reference qwen35 "${QWEN_9_MODEL}" \
  rwkv-7.2b__qwen3.5-9b 9b reference_9b \
  --qwen-backend fla --qwen-conv-backend fla_triton --require-qwen-fast-path

"${RWKV_PYTHON_BIN}" bench/compare_qwen35_speed_matrix.py \
  --results "${OUT_DIR}/results.jsonl" --expected-cells 24 \
  --require-qwen-fast-path --require-qwen-full-fused \
  --required-reference-backend fla \
  --min-prefill-active-parameter-throughput-ratio 1.0 \
  --fail-on-gate --json-output "${OUT_DIR}/summary.json" \
  --markdown-output "${OUT_DIR}/summary.md"

CORRECTNESS_RESULTS="${OUT_DIR}/correctness.jsonl"
rm -f "${CORRECTNESS_RESULTS}"
run_correctness() {
  local model="$1" batch_text="$2" prompt_text="$3" chunk="${4:-0}"
  local batches=() prompts=()
  read -r -a batches <<< "${batch_text}"
  read -r -a prompts <<< "${prompt_text}"
  local chunk_args=()
  if [[ "${chunk}" -gt 0 ]]; then
    chunk_args=(--chunk-size "${chunk}")
  fi
  "${RWKV_PYTHON_BIN}" bench/bench_native_prefill_accum_correctness.py \
    --model "${model}" --device cuda --dtype fp16 --batch-size "${batches[@]}" \
    --prompt-tokens "${prompts[@]}" "${chunk_args[@]}" --code-source repo \
    --min-cosine 0.9999 --results "${CORRECTNESS_RESULTS}" \
    >> "${OUT_DIR}/logs/accum_correctness.log" 2>&1
}

# Every direct or chunk-carried shape that opts into global FP16 GEMM
# accumulation gets prompt-logit, cache-handoff, and greedy-token coverage.
run_correctness "${RWKV_04_MODEL}" 1 "512 2048"
run_correctness "${RWKV_04_MODEL}" 8 "128 512"
run_correctness "${RWKV_04_MODEL}" "1 8" 2048 512
run_correctness "${RWKV_15_MODEL}" 1 "128 512 2048"
run_correctness "${RWKV_15_MODEL}" 8 "128 512"
run_correctness "${RWKV_15_MODEL}" "1 8" 2048 512
run_correctness "${RWKV_29_MODEL}" "1 8" "128 512"
run_correctness "${RWKV_29_MODEL}" "1 8" 2048 512
run_correctness "${RWKV_72_MODEL}" "1 8" "128 512"
run_correctness "${RWKV_72_MODEL}" "1 8" 2048 512

"${RWKV_PYTHON_BIN}" - "${CORRECTNESS_RESULTS}" \
  "${OUT_DIR}/correctness_summary.json" <<'PY'
import json
import sys
from pathlib import Path

source, target = map(Path, sys.argv[1:])
rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
assert len(rows) == 25, f"expected 25 correctness rows, got {len(rows)}"
assert all(row.get("status") == "pass" for row in rows), "correctness failure"
summary = {
    "status": "pass",
    "rows": len(rows),
    "min_prompt_cosine": min(row["min_cosine"] for row in rows),
    "min_decode_after_prefill_cosine": min(
        row["decode_after_prefill_min_cosine"] for row in rows
    ),
    "prompt_greedy_matches": sum(bool(row["greedy_match"]) for row in rows),
    "decode_after_prefill_greedy_matches": sum(
        bool(row["decode_after_prefill_greedy_match"]) for row in rows
    ),
}
target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
PY
