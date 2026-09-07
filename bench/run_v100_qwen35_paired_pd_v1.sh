#!/usr/bin/env bash
# Run and authenticate the complete strict Tesla V100 RWKV/Qwen P+D matrix.
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-${1:-}}"
CACHE_ROOT="${CACHE_ROOT:-}"
CUDA_TOOLKIT_VIEW="${CUDA_TOOLKIT_VIEW:-}"
CUDA_COMPONENT_INCLUDE="${CUDA_COMPONENT_INCLUDE:-}"
REPOSITORY_COMMIT="${REPOSITORY_COMMIT:-}"
RWKV_04_MODEL="${RWKV_04_MODEL:-}"
RWKV_15_MODEL="${RWKV_15_MODEL:-}"
RWKV_29_MODEL="${RWKV_29_MODEL:-}"
RWKV_72_MODEL="${RWKV_72_MODEL:-}"
QWEN_08_MODEL="${QWEN_08_MODEL:-}"
QWEN_2_MODEL="${QWEN_2_MODEL:-}"
QWEN_4_MODEL="${QWEN_4_MODEL:-}"
QWEN_9_MODEL="${QWEN_9_MODEL:-}"
FLA_TARGET="${FLA_TARGET:-}"
TRITON_TARGET="${TRITON_TARGET:-}"
FROZEN_QWEN_DIR="${FROZEN_QWEN_DIR:-}"
FROZEN_QWEN_REFERENCE_SHA256="${FROZEN_QWEN_REFERENCE_SHA256:-}"

if [[ -z "${OUT_DIR}" || -z "${CACHE_ROOT}" || -z "${CUDA_TOOLKIT_VIEW}" || -z "${CUDA_COMPONENT_INCLUDE}" || -z "${REPOSITORY_COMMIT}" || -z "${FLA_TARGET}" || -z "${TRITON_TARGET}" ]]; then
  echo "OUT_DIR, CACHE_ROOT, CUDA_TOOLKIT_VIEW, CUDA_COMPONENT_INCLUDE and REPOSITORY_COMMIT are required" >&2
  exit 2
fi
for name in RWKV_04_MODEL RWKV_15_MODEL RWKV_29_MODEL RWKV_72_MODEL; do
  [[ -n "${!name}" ]] || { echo "${name} is required" >&2; exit 2; }
done
if [[ -z "${FROZEN_QWEN_DIR}" ]]; then
  for name in QWEN_08_MODEL QWEN_2_MODEL QWEN_4_MODEL QWEN_9_MODEL; do
    [[ -n "${!name}" ]] || { echo "${name} is required without FROZEN_QWEN_DIR" >&2; exit 2; }
  done
elif [[ ! "${FROZEN_QWEN_REFERENCE_SHA256}" =~ ^[0-9a-fA-F]{64}$ ]]; then
  echo "FROZEN_QWEN_REFERENCE_SHA256 must be 64 hex in frozen-reference mode" >&2
  exit 2
fi
if [[ ! "${REPOSITORY_COMMIT}" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "REPOSITORY_COMMIT must be 40 hexadecimal characters" >&2
  exit 2
fi

ROOT="$(realpath -e -- "${ROOT}")"
OUT_DIR="$(realpath -m -- "${OUT_DIR}")"
CACHE_ROOT="$(realpath -m -- "${CACHE_ROOT}")"
CUDA_TOOLKIT_VIEW="$(realpath -e -- "${CUDA_TOOLKIT_VIEW}")"
CUDA_COMPONENT_INCLUDE="$(realpath -e -- "${CUDA_COMPONENT_INCLUDE}")"
FLA_TARGET="$(realpath -e -- "${FLA_TARGET}")"
TRITON_TARGET="$(realpath -e -- "${TRITON_TARGET}")"
if [[ "${PYTHON_BIN}" == */* ]]; then
  python_dir="$(realpath -e -- "$(dirname -- "${PYTHON_BIN}")")"
  PYTHON_BIN="${python_dir}/$(basename -- "${PYTHON_BIN}")"
else
  PYTHON_BIN="$(command -v -- "${PYTHON_BIN}")"
fi
for name in RWKV_04_MODEL RWKV_15_MODEL RWKV_29_MODEL RWKV_72_MODEL; do
  printf -v "${name}" '%s' "$(realpath -e -- "${!name}")"
done
if [[ -n "${FROZEN_QWEN_DIR}" ]]; then
  FROZEN_QWEN_DIR="$(realpath -e -- "${FROZEN_QWEN_DIR}")"
else
  for name in QWEN_08_MODEL QWEN_2_MODEL QWEN_4_MODEL QWEN_9_MODEL; do
    printf -v "${name}" '%s' "$(realpath -e -- "${!name}")"
  done
fi
if [[ -e "${OUT_DIR}" || -e "${CACHE_ROOT}" ]]; then
  echo "formal OUT_DIR and CACHE_ROOT must both be absent" >&2
  exit 2
fi
protected_inputs=("${ROOT}" "${RWKV_04_MODEL}" "${RWKV_15_MODEL}" "${RWKV_29_MODEL}" "${RWKV_72_MODEL}")
if [[ -n "${FROZEN_QWEN_DIR}" ]]; then
  protected_inputs+=("${FROZEN_QWEN_DIR}")
else
  protected_inputs+=("${QWEN_08_MODEL}" "${QWEN_2_MODEL}" "${QWEN_4_MODEL}" "${QWEN_9_MODEL}")
fi
for protected in "${protected_inputs[@]}"; do
  case "${OUT_DIR}/" in "${protected}/"*) echo "OUT_DIR is inside protected input ${protected}" >&2; exit 2;; esac
  case "${protected}/" in "${OUT_DIR}/"*) echo "OUT_DIR contains protected input ${protected}" >&2; exit 2;; esac
  case "${CACHE_ROOT}/" in "${protected}/"*) echo "CACHE_ROOT is inside protected input ${protected}" >&2; exit 2;; esac
  case "${protected}/" in "${CACHE_ROOT}/"*) echo "CACHE_ROOT contains protected input ${protected}" >&2; exit 2;; esac
done

export ROOT PYTHON_BIN REPOSITORY_COMMIT CUDA_TOOLKIT_VIEW CUDA_COMPONENT_INCLUDE FLA_TARGET TRITON_TARGET
export RWKV_04_MODEL RWKV_15_MODEL RWKV_29_MODEL RWKV_72_MODEL
OUT_DIR="${OUT_DIR}" CACHE_ROOT="${CACHE_ROOT}/rwkv" \
  bash "${ROOT}/bench/run_v100_rwkv_paired_pd_v1.sh"

run_qwen() {
  local tag="$1" model="$2" pair="$3" size="$4" route="$5" sdpa_policy="$6"
  OUT_DIR="${OUT_DIR}" MODEL="${model}" MODEL_PAIR="${pair}" MODEL_SIZE_LABEL="${size}" \
    RESULT_NAME="qwen_${tag}.jsonl" CACHE_ROOT="${CACHE_ROOT}/qwen_${tag}" \
    QWEN_DECODE_OPTIMIZATION="${route}" QWEN_COMPILE_MODE=max-autotune \
    QWEN_SDPA_POLICY="${sdpa_policy}" \
    bash "${ROOT}/bench/run_v100_qwen35_best_optimized_hf_v1.sh"
}
if [[ -z "${FROZEN_QWEN_DIR}" ]]; then
  run_qwen 0p8 "${QWEN_08_MODEL}" rwkv-0.4b__qwen3.5-0.8b 0.8b static_cache_raw_cudagraph auto
  run_qwen 2b "${QWEN_2_MODEL}" rwkv-1.5b__qwen3.5-2b 2b static_cache_raw_cudagraph auto
  run_qwen 4b "${QWEN_4_MODEL}" rwkv-2.9b__qwen3.5-4b 4b static_cache_raw_cudagraph auto
  run_qwen 9b "${QWEN_9_MODEL}" rwkv-7.2b__qwen3.5-9b 9b static_cache_raw_cudagraph math_only
else
  "${PYTHON_BIN}" - "${FROZEN_QWEN_DIR}" "${OUT_DIR}" "${FROZEN_QWEN_REFERENCE_SHA256}" <<'PY'
import hashlib, json, shutil, sys
from pathlib import Path
source, target = map(Path, sys.argv[1:3])
expected_reference_sha = sys.argv[3].lower()
copied = []
for tag in ("0p8", "2b", "4b", "9b"):
    manifest_source = source / f"qwen_{tag}_route.json"
    manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
    for record in (
        manifest.get("result"),
        manifest.get("model_hash_contract", {}).get("before"),
        manifest.get("model_hash_contract", {}).get("after"),
    ):
        if type(record) is not dict:
            raise SystemExit(f"malformed frozen Qwen artifact in {manifest_source}")
        item = source / Path(record["path"]).name
        digest = hashlib.sha256(item.read_bytes()).hexdigest()
        if digest != record.get("sha256"):
            raise SystemExit(f"frozen Qwen SHA mismatch: {item}")
        destination = target / item.name
        if not destination.exists():
            shutil.copyfile(item, destination)
        copied.append({"path": destination.name, "sha256": digest})
    destination = target / manifest_source.name
    shutil.copyfile(manifest_source, destination)
    copied.append({
        "path": destination.name,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    })
provenance = {
    "schema_version": 1,
    "source_directory": str(source.resolve()),
    "expected_reference_sha256": expected_reference_sha,
    "artifacts": copied,
}
(target / "frozen_qwen_provenance.json").write_text(
    json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
)
PY
fi

reference="${OUT_DIR}/qwen_reference.jsonl"
[[ ! -e "${reference}" ]] || { echo "refusing to overwrite ${reference}" >&2; exit 2; }
cat "${OUT_DIR}/qwen_0p8.jsonl" "${OUT_DIR}/qwen_2b.jsonl" "${OUT_DIR}/qwen_4b.jsonl" "${OUT_DIR}/qwen_9b.jsonl" > "${reference}"
if [[ -n "${FROZEN_QWEN_DIR}" ]]; then
  actual_reference_sha="$("${PYTHON_BIN}" - "${reference}" <<'PY'
import hashlib, sys
print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())
PY
)"
  [[ "${actual_reference_sha,,}" == "${FROZEN_QWEN_REFERENCE_SHA256,,}" ]] || {
    echo "frozen Qwen reference SHA mismatch" >&2
    exit 2
  }
fi

"${PYTHON_BIN}" "${ROOT}/bench/validators/validate_qwen35_v100_paired_pd_v1.py" \
  --candidate "${OUT_DIR}/rwkv_candidate.jsonl" \
  --reference "${reference}" \
  --correctness-manifest "${OUT_DIR}/rwkv_native_graph_fla_correctness.json" \
  --candidate-route-manifest "${OUT_DIR}/rwkv_candidate_routes.json" \
  --runtime-lock "${OUT_DIR}/runtime-lock.json" \
  --candidate-model-hashes "${OUT_DIR}/model_hashes.sha256" \
  --qwen-result "${OUT_DIR}/qwen_0p8.jsonl" \
  --qwen-result "${OUT_DIR}/qwen_2b.jsonl" \
  --qwen-result "${OUT_DIR}/qwen_4b.jsonl" \
  --qwen-result "${OUT_DIR}/qwen_9b.jsonl" \
  --qwen-route-manifest "${OUT_DIR}/qwen_0p8_route.json" \
  --qwen-route-manifest "${OUT_DIR}/qwen_2b_route.json" \
  --qwen-route-manifest "${OUT_DIR}/qwen_4b_route.json" \
  --qwen-route-manifest "${OUT_DIR}/qwen_9b_route.json" \
  --summary "${OUT_DIR}/paired_validation.json" \
  --paired-table "${OUT_DIR}/paired_pd_table.jsonl" \
  --markdown "${OUT_DIR}/paired_pd.md"

printf '0\n' > "${OUT_DIR}/exit_code.txt"
echo "strict Tesla V100 paired P+D bundle passed: ${OUT_DIR}"
