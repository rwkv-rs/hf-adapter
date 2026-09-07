#!/usr/bin/env bash
# Capture the strict Tesla V100 RWKV candidate matrix and FLA correctness oracle.
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-${1:-}}"
REPOSITORY_COMMIT="${REPOSITORY_COMMIT:-}"
RWKV_04_MODEL="${RWKV_04_MODEL:-}"
RWKV_15_MODEL="${RWKV_15_MODEL:-}"
RWKV_29_MODEL="${RWKV_29_MODEL:-}"
RWKV_72_MODEL="${RWKV_72_MODEL:-}"
FLA_TARGET="${FLA_TARGET:-}"
TRITON_TARGET="${TRITON_TARGET:-}"
CUDA_TOOLKIT_VIEW="${CUDA_TOOLKIT_VIEW:-}"
CUDA_COMPONENT_INCLUDE="${CUDA_COMPONENT_INCLUDE:-}"
CACHE_ROOT="${CACHE_ROOT:-}"
PROTOCOL="qwen35_v100_paired_pd_v1"
CORRECTNESS_PROTOCOL="rwkv_native_graph_fla_correctness_v100_v1"

if [[ -z "${OUT_DIR}" || -z "${CACHE_ROOT}" || -z "${CUDA_TOOLKIT_VIEW}" || -z "${CUDA_COMPONENT_INCLUDE}" || -z "${RWKV_04_MODEL}" || -z "${RWKV_15_MODEL}" || -z "${RWKV_29_MODEL}" || -z "${RWKV_72_MODEL}" || -z "${FLA_TARGET}" || -z "${TRITON_TARGET}" ]]; then
  echo "OUT_DIR, CACHE_ROOT, CUDA_TOOLKIT_VIEW, CUDA_COMPONENT_INCLUDE and all four RWKV model paths are required" >&2
  exit 2
fi
if [[ ! "${REPOSITORY_COMMIT}" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "REPOSITORY_COMMIT must be the explicit 40-hex commit under test" >&2
  exit 2
fi

ROOT="$(realpath -e -- "${ROOT}")"
OUT_DIR="$(realpath -m -- "${OUT_DIR}")"
RWKV_04_MODEL="$(realpath -e -- "${RWKV_04_MODEL}")"
RWKV_15_MODEL="$(realpath -e -- "${RWKV_15_MODEL}")"
RWKV_29_MODEL="$(realpath -e -- "${RWKV_29_MODEL}")"
RWKV_72_MODEL="$(realpath -e -- "${RWKV_72_MODEL}")"
FLA_TARGET="$(realpath -e -- "${FLA_TARGET}")"
TRITON_TARGET="$(realpath -e -- "${TRITON_TARGET}")"
CUDA_TOOLKIT_VIEW="$(realpath -e -- "${CUDA_TOOLKIT_VIEW}")"
CUDA_COMPONENT_INCLUDE="$(realpath -e -- "${CUDA_COMPONENT_INCLUDE}")"
[[ -f "${CUDA_COMPONENT_INCLUDE}/cusparse.h" ]] || {
  echo "CUDA_COMPONENT_INCLUDE lacks cusparse.h: ${CUDA_COMPONENT_INCLUDE}" >&2
  exit 2
}
CACHE_ROOT="$(realpath -m -- "${CACHE_ROOT}")"
if [[ "${PYTHON_BIN}" == */* ]]; then
  python_dir="$(realpath -e -- "$(dirname -- "${PYTHON_BIN}")")"
  PYTHON_BIN="${python_dir}/$(basename -- "${PYTHON_BIN}")"
else
  PYTHON_BIN="$(command -v -- "${PYTHON_BIN}")"
fi
for model in "${RWKV_04_MODEL}" "${RWKV_15_MODEL}" "${RWKV_29_MODEL}" "${RWKV_72_MODEL}"; do
  [[ -f "${model}/config.json" ]] || { echo "missing ${model}/config.json" >&2; exit 2; }
  compgen -G "${model}/*.safetensors" >/dev/null || { echo "missing weights in ${model}" >&2; exit 2; }
done
if [[ -e "${OUT_DIR}" ]]; then
  echo "formal OUT_DIR must not already exist: ${OUT_DIR}" >&2
  exit 2
fi
for protected in "${ROOT}" "${RWKV_04_MODEL}" "${RWKV_15_MODEL}" "${RWKV_29_MODEL}" "${RWKV_72_MODEL}"; do
  case "${OUT_DIR}/" in "${protected}/"*) echo "OUT_DIR is inside protected input ${protected}" >&2; exit 2;; esac
  case "${protected}/" in "${OUT_DIR}/"*) echo "OUT_DIR contains protected input ${protected}" >&2; exit 2;; esac
  case "${CACHE_ROOT}/" in "${protected}/"*) echo "CACHE_ROOT is inside protected input ${protected}" >&2; exit 2;; esac
  case "${protected}/" in "${CACHE_ROOT}/"*) echo "CACHE_ROOT contains protected input ${protected}" >&2; exit 2;; esac
done

validate_repository() {
  local top head
  top="$(realpath -e -- "$(git -C "${ROOT}" rev-parse --show-toplevel)")"
  head="$(git -C "${ROOT}" rev-parse HEAD)"
  [[ "${top}" == "${ROOT}" && "${head,,}" == "${REPOSITORY_COMMIT,,}" ]] || {
    echo "repository root/HEAD does not match the frozen contract" >&2; exit 2;
  }
  [[ -z "$(git -C "${ROOT}" status --porcelain --untracked-files=all)" ]] || {
    echo "formal capture requires a completely clean repository" >&2; exit 2;
  }
}
validate_repository
cd "${ROOT}"
mkdir -p "${OUT_DIR}/logs"
if [[ -e "${CACHE_ROOT}" ]]; then
  [[ -d "${CACHE_ROOT}" && -z "$(find "${CACHE_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]] || {
    echo "CACHE_ROOT must be absent or empty: ${CACHE_ROOT}" >&2; exit 2;
  }
fi
mkdir -p "${CACHE_ROOT}/torchinductor" "${CACHE_ROOT}/triton" "${CACHE_ROOT}/huggingface"

PIP_FREEZE="${OUT_DIR}/pip-freeze.txt"
RUNTIME_LOCK="${OUT_DIR}/runtime-lock.json"
SYSTEM_CSV="${OUT_DIR}/system.csv"
MODEL_HASHES="${OUT_DIR}/model_hashes.sha256"
MODEL_HASHES_AFTER="${OUT_DIR}/model_hashes.after.sha256"
CANDIDATE="${OUT_DIR}/rwkv_candidate.jsonl"
CANDIDATE_SHA="${OUT_DIR}/rwkv_candidate.sha256"
CORRECTNESS_MANIFEST="${OUT_DIR}/rwkv_native_graph_fla_correctness.json"
ROUTE_MANIFEST="${OUT_DIR}/rwkv_candidate_routes.json"

COMMON_ENV=(
  "HOME=${HOME}" "LANG=C.UTF-8" "PATH=$(dirname "${PYTHON_BIN}"):${CUDA_TOOLKIT_VIEW}/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
  "CUDA_VISIBLE_DEVICES=0" "CUDA_DEVICE_ORDER=PCI_BUS_ID" "PYTHONPATH=${ROOT}:${TRITON_TARGET}:${FLA_TARGET}"
  "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" "TORCH_CUDA_ARCH_LIST=7.0"
  "HF_HUB_OFFLINE=1" "TRANSFORMERS_OFFLINE=1" "TOKENIZERS_PARALLELISM=false"
  "CUDA_HOME=${CUDA_TOOLKIT_VIEW}" "LD_LIBRARY_PATH=${CUDA_TOOLKIT_VIEW}/lib64"
  "CPATH=${CUDA_COMPONENT_INCLUDE}"
  "XDG_CACHE_HOME=${CACHE_ROOT}" "HF_HOME=${CACHE_ROOT}/huggingface"
  "TORCHINDUCTOR_CACHE_DIR=${CACHE_ROOT}/torchinductor" "TRITON_CACHE_DIR=${CACHE_ROOT}/triton"
  "REPOSITORY_COMMIT=${REPOSITORY_COMMIT}"
)

gpu_name="$(env -i "${COMMON_ENV[@]}" "${PYTHON_BIN}" -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")')"
"${PYTHON_BIN}" "${ROOT}/bench/validators/check_exact_gpu.py" --exact-name "Tesla V100-PCIE-32GB" --name "${gpu_name}"

env -i "${COMMON_ENV[@]}" "${PYTHON_BIN}" -m pip freeze --all | LC_ALL=C sort > "${PIP_FREEZE}"

env -i "${COMMON_ENV[@]}" "${PYTHON_BIN}" - "${RUNTIME_LOCK}" "${PIP_FREEZE}" "${PROTOCOL}" "${REPOSITORY_COMMIT}" <<'PY'
import hashlib, json, platform, sys
from importlib.metadata import version
import torch, transformers, triton
runtime = {
    "python": platform.python_version(), "torch": str(torch.__version__),
    "torch_cuda": str(torch.version.cuda), "triton": str(triton.__version__),
    "transformers": str(transformers.__version__),
    "fla": version("flash-linear-attention"),
    "causal_conv1d": None,
}
expected = {"python":"3.11.15","torch":"2.5.1+cu124","torch_cuda":"12.4","triton":"3.4.0","transformers":"5.12.1","fla":"0.5.1","causal_conv1d":None}
if runtime != expected:
    raise SystemExit(f"runtime mismatch: {runtime!r} != {expected!r}")
pip = open(sys.argv[2], "rb").read()
doc = {"schema_version":1,"protocol":sys.argv[3],"repository_commit":sys.argv[4],"runtime":runtime,"pip_freeze_sha256":hashlib.sha256(pip).hexdigest(),"torch_cuda_arch_list":"7.0"}
open(sys.argv[1], "w", encoding="utf-8").write(json.dumps(doc, indent=2)+"\n")
PY

nvidia-smi --query-gpu=name,uuid,pci.bus_id,compute_cap,driver_version,memory.total,power.limit,clocks.current.sm,clocks.max.sm,clocks.current.memory,clocks.max.memory --format=csv > "${SYSTEM_CSV}"

hash_models() {
  local output="$1"
  env -i "${COMMON_ENV[@]}" "${PYTHON_BIN}" - "${output}" "${RWKV_04_MODEL}" "${RWKV_15_MODEL}" "${RWKV_29_MODEL}" "${RWKV_72_MODEL}" <<'PY'
import hashlib, os, sys
from pathlib import Path
out=Path(sys.argv[1]); lines=[]
for raw in sys.argv[2:]:
    root=Path(raw).resolve(strict=True); lines.append(f"[{root.as_posix()}]")
    files=sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p:p.relative_to(root).as_posix())
    if not files: raise SystemExit(f"empty model directory: {root}")
    for path in files:
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(root).as_posix()}")
out.write_text("\n".join(lines)+"\n", encoding="utf-8")
PY
}
hash_models "${MODEL_HASHES}"

run_native_lane() {
  local tag="$1" pair="$2" size="$3" model="$4" batch="$5"
  local ada_require_extension=0 sm70_require_extension=0 rkv_policy=manual
  local norm_mix_warps=4
  local fused_wavg_lora=1
  if [[ "${batch}" == "1" ]]; then
    sm70_require_extension=1
  fi
  if [[ "${tag}" == "7p2" && "${batch}" == "8" ]]; then
    rkv_policy=vkwr_auto
    norm_mix_warps=8
    fused_wavg_lora=0
  fi
  local result="${OUT_DIR}/rwkv_${tag}_b${batch}.jsonl"
  local probe="${OUT_DIR}/decode_correctness_${tag}_b${batch}_native.pt"
  local log="${OUT_DIR}/logs/rwkv_${tag}_b${batch}.log"
  env -i "${COMMON_ENV[@]}" \
    "RWKV7_FAST_TOKEN_BACKEND=native_graph" "RWKV7_NATIVE_MODEL_BACKEND=native_graph" \
    "RWKV7_NATIVE_GRAPH_ADA_WAGV_LORA_REQUIRE_EXTENSION=${ada_require_extension}" \
    "RWKV7_NATIVE_GRAPH_SM70_WAGV_LORA_REQUIRE_EXTENSION=${sm70_require_extension}" \
    "RWKV7_NATIVE_GRAPH_RKV_POLICY=${rkv_policy}" \
    "RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX_NUM_WARPS=${norm_mix_warps}" \
    "RWKV7_NATIVE_GRAPH_FUSED_WAVG_LORA=${fused_wavg_lora}" \
    "RWKV7_NATIVE_GRAPH_ADA_WAGV_BMM=0" "RWKV7_NATIVE_GRAPH_SM120_WAGV_BMM_G=0" \
    "RWKV7_NATIVE_GRAPH_SM120_COMPILED_FFN=0" \
    "${PYTHON_BIN}" "${ROOT}/bench/runners/bench_cross_model_speed_resident.py" \
      --model "${model}" --model-kind rwkv --model-role candidate \
      --model-pair "${pair}" --model-size-label "${size}" \
      --benchmark-matrix "${PROTOCOL}" --optimization-lane best_optimized_hf \
      --dtype fp16 --quantization none --device cuda \
      --cells "${batch}x128x128" "${batch}x128x512" "${batch}x512x128" \
              "${batch}x512x512" "${batch}x2048x128" "${batch}x2048x512" \
      --prefill-chunk-size 512 --warmup 3 --runs 7 \
      --rwkv-attn-mode fused_recurrent --rwkv-code-source repo \
      --rwkv-implementation auto --probe-output "${probe}" \
      --probe-cell "${batch}x2048x512" --probe-tokens 512 --probe-batch-size "${batch}" \
      --fail-fast --results "${result}" > "${log}" 2>&1
}

run_fla_reference() {
  local tag="$1" pair="$2" size="$3" model="$4" batch="$5"
  local result="${OUT_DIR}/decode_correctness_${tag}_b${batch}_fla.jsonl"
  local probe="${OUT_DIR}/decode_correctness_${tag}_b${batch}_fla.pt"
  local log="${OUT_DIR}/logs/decode_correctness_${tag}_b${batch}_fla.log"
  env -i "${COMMON_ENV[@]}" \
    "RWKV7_FAST_TOKEN_BACKEND=fla" "RWKV7_NATIVE_MODEL_BACKEND=eager" "RWKV7_NATIVE_MODEL=0" \
    "RWKV7_FAST_PREFILL=0" "RWKV7_NATIVE_PREFILL_GRAPH=0" "RWKV7_NATIVE_GRAPH_ADA_WAGV_BMM=0" \
    "RWKV7_NATIVE_GRAPH_ADA_WAGV_LORA_REQUIRE_EXTENSION=0" "RWKV7_NATIVE_GRAPH_RKV_POLICY=manual" \
    "RWKV7_NATIVE_GRAPH_SM70_WAGV_LORA_REQUIRE_EXTENSION=0" \
    "RWKV7_NATIVE_GRAPH_SM120_WAGV_BMM_G=0" "RWKV7_NATIVE_GRAPH_SM120_COMPILED_FFN=0" \
    "TORCHDYNAMO_DISABLE=1" "TORCH_COMPILE_DISABLE=1" \
    "${PYTHON_BIN}" "${ROOT}/bench/runners/bench_cross_model_speed_resident.py" \
      --model "${model}" --model-kind rwkv --model-role candidate \
      --model-pair "${pair}" --model-size-label "${size}" \
      --benchmark-matrix "${CORRECTNESS_PROTOCOL}" --optimization-lane fla_reference \
      --dtype fp16 --quantization none --device cuda --cells "${batch}x2048x512" \
      --prefill-chunk-size 512 --warmup 1 --runs 1 --rwkv-attn-mode fused_recurrent \
      --rwkv-code-source repo --rwkv-implementation wrapper_repo \
      --probe-output "${probe}" --probe-cell "${batch}x2048x512" \
      --probe-tokens 512 --probe-batch-size "${batch}" --fail-fast --results "${result}" \
      > "${log}" 2>&1
}

run_native_eager_closure_reference() {
  local result="${OUT_DIR}/decode_correctness_7p2_b8_p128_native_eager.jsonl"
  local probe="${OUT_DIR}/decode_correctness_7p2_b8_p128_native_eager.pt"
  local log="${OUT_DIR}/logs/decode_correctness_7p2_b8_p128_native_eager.log"
  env -i "${COMMON_ENV[@]}" \
    "RWKV7_FAST_TOKEN_BACKEND=module_call" "RWKV7_NATIVE_MODEL_BACKEND=eager" \
    "RWKV7_FAST_PREFILL=0" "RWKV7_NATIVE_PREFILL_GRAPH=0" \
    "RWKV7_NATIVE_GRAPH_ADA_WAGV_LORA_REQUIRE_EXTENSION=0" \
    "RWKV7_NATIVE_GRAPH_SM70_WAGV_LORA_REQUIRE_EXTENSION=0" \
    "RWKV7_NATIVE_GRAPH_RKV_POLICY=manual" \
    "RWKV7_NATIVE_GRAPH_ADA_WAGV_BMM=0" "RWKV7_NATIVE_GRAPH_SM120_WAGV_BMM_G=0" \
    "RWKV7_NATIVE_GRAPH_SM120_COMPILED_FFN=0" \
    "TORCHDYNAMO_DISABLE=1" "TORCH_COMPILE_DISABLE=1" \
    "${PYTHON_BIN}" "${ROOT}/bench/runners/bench_cross_model_speed_resident.py" \
      --model "${RWKV_72_MODEL}" --model-kind rwkv --model-role candidate \
      --model-pair rwkv-7.2b__qwen3.5-9b --model-size-label 7.2b \
      --benchmark-matrix "${CORRECTNESS_PROTOCOL}" --optimization-lane native_eager_reference \
      --dtype fp16 --quantization none --device cuda --cells 8x128x128 \
      --prefill-chunk-size 512 --warmup 1 --runs 1 --rwkv-attn-mode fused_recurrent \
      --rwkv-code-source repo --rwkv-implementation auto \
      --probe-output "${probe}" --probe-cell 8x128x128 \
      --probe-tokens 512 --probe-batch-size 8 --fail-fast --results "${result}" \
      > "${log}" 2>&1
}

run_native_graph_closure_candidate() {
  local result="${OUT_DIR}/decode_correctness_7p2_b8_p128_native_graph.jsonl"
  local probe="${OUT_DIR}/decode_correctness_7p2_b8_p128_native_graph.pt"
  local log="${OUT_DIR}/logs/decode_correctness_7p2_b8_p128_native_graph.log"
  env -i "${COMMON_ENV[@]}" \
    "RWKV7_FAST_TOKEN_BACKEND=native_graph" "RWKV7_NATIVE_MODEL_BACKEND=native_graph" \
    "RWKV7_NATIVE_GRAPH_ADA_WAGV_LORA_REQUIRE_EXTENSION=0" \
    "RWKV7_NATIVE_GRAPH_SM70_WAGV_LORA_REQUIRE_EXTENSION=0" \
    "RWKV7_NATIVE_GRAPH_RKV_POLICY=vkwr_auto" \
    "RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX_NUM_WARPS=8" \
    "RWKV7_NATIVE_GRAPH_FUSED_WAVG_LORA=0" \
    "RWKV7_NATIVE_GRAPH_ADA_WAGV_BMM=0" "RWKV7_NATIVE_GRAPH_SM120_WAGV_BMM_G=0" \
    "RWKV7_NATIVE_GRAPH_SM120_COMPILED_FFN=0" \
    "${PYTHON_BIN}" "${ROOT}/bench/runners/bench_cross_model_speed_resident.py" \
      --model "${RWKV_72_MODEL}" --model-kind rwkv --model-role candidate \
      --model-pair rwkv-7.2b__qwen3.5-9b --model-size-label 7.2b \
      --benchmark-matrix "${CORRECTNESS_PROTOCOL}" --optimization-lane native_graph_closure \
      --dtype fp16 --quantization none --device cuda --cells 8x128x128 \
      --prefill-chunk-size 512 --warmup 3 --runs 7 --rwkv-attn-mode fused_recurrent \
      --rwkv-code-source repo --rwkv-implementation auto \
      --probe-output "${probe}" --probe-cell 8x128x128 \
      --probe-tokens 512 --probe-batch-size 8 --fail-fast --results "${result}" \
      > "${log}" 2>&1
}

for spec in \
  "0p4|rwkv-0.4b__qwen3.5-0.8b|0.4b|${RWKV_04_MODEL}" \
  "1p5|rwkv-1.5b__qwen3.5-2b|1.5b|${RWKV_15_MODEL}" \
  "2p9|rwkv-2.9b__qwen3.5-4b|2.9b|${RWKV_29_MODEL}" \
  "7p2|rwkv-7.2b__qwen3.5-9b|7.2b|${RWKV_72_MODEL}"; do
  IFS='|' read -r tag pair size model <<< "${spec}"
  for batch in 1 8; do
    run_native_lane "${tag}" "${pair}" "${size}" "${model}" "${batch}"
    run_fla_reference "${tag}" "${pair}" "${size}" "${model}" "${batch}"
    compare_contract=()
    if [[ "${batch}" == 8 ]]; then compare_contract+=(--require-distinct-batch-prompts); fi
    env -i "${COMMON_ENV[@]}" "${PYTHON_BIN}" "${ROOT}/bench/analyzers/compare_rwkv_prefill_probe.py" \
      --reference-probe "${OUT_DIR}/decode_correctness_${tag}_b${batch}_fla.pt" \
      --native-probe "${OUT_DIR}/decode_correctness_${tag}_b${batch}_native.pt" \
      --min-cosine 0.9999 --required-batch-size "${batch}" --required-probe-tokens 512 \
      "${compare_contract[@]}" \
      --output "${OUT_DIR}/decode_correctness_${tag}_b${batch}_compare.json" --fail-on-gate
  done
done

run_native_eager_closure_reference
run_native_graph_closure_candidate
env -i "${COMMON_ENV[@]}" "${PYTHON_BIN}" "${ROOT}/bench/analyzers/compare_rwkv_prefill_probe.py" \
  --reference-probe "${OUT_DIR}/decode_correctness_7p2_b8_p128_native_eager.pt" \
  --native-probe "${OUT_DIR}/decode_correctness_7p2_b8_p128_native_graph.pt" \
  --min-cosine 0.9999 --required-batch-size 8 --required-probe-tokens 512 \
  --require-distinct-batch-prompts \
  --output "${OUT_DIR}/decode_correctness_7p2_b8_p128_native_eager_compare.json" \
  --fail-on-gate

env -i "${COMMON_ENV[@]}" "${PYTHON_BIN}" - "${OUT_DIR}" "${CANDIDATE}" "${CANDIDATE_SHA}" "${CORRECTNESS_MANIFEST}" "${ROUTE_MANIFEST}" "${MODEL_HASHES}" "${RUNTIME_LOCK}" "${SYSTEM_CSV}" "${REPOSITORY_COMMIT}" "${CACHE_ROOT}" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
out, candidate, sidecar, correctness, routes, model_hashes, runtime, system = map(Path, sys.argv[1:9])
commit=sys.argv[9]; cache_root=str(Path(sys.argv[10]).resolve())
specs=[("0p4","rwkv-0.4b__qwen3.5-0.8b","0.4b"),("1p5","rwkv-1.5b__qwen3.5-2b","1.5b"),("2p9","rwkv-2.9b__qwen3.5-4b","2.9b"),("7p2","rwkv-7.2b__qwen3.5-9b","7.2b")]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def artifact(path): return {"path":str(path.resolve()),"sha256":sha(path)}
rows=[]; entries=[]; lanes=[]
for tag,pair,size in specs:
  for batch in (1,8):
    lane=out/f"rwkv_{tag}_b{batch}.jsonl"; lane_rows=[json.loads(x) for x in lane.read_text().splitlines() if x.strip()]
    if len(lane_rows)!=6: raise SystemExit(f"{lane}: expected six rows")
    rows.extend(lane_rows)
    target=[r for r in lane_rows if r["prompt_tokens"]==2048 and r["decode_tokens"]==512]
    if len(target)!=1: raise SystemExit(f"{lane}: missing target cell")
    native_row=out/f"decode_correctness_{tag}_b{batch}_native.jsonl"
    native_row.write_text(json.dumps(target[0])+"\n")
    fla_row=out/f"decode_correctness_{tag}_b{batch}_fla.jsonl"
    native_probe=out/f"decode_correctness_{tag}_b{batch}_native.pt"
    fla_probe=out/f"decode_correctness_{tag}_b{batch}_fla.pt"
    comparison=out/f"decode_correctness_{tag}_b{batch}_compare.json"
    entries.append({"model_pair":pair,"model_size_label":size,"model_path":target[0]["model_id_or_path"],"batch_size":batch,"prompt_tokens":2048,"decode_tokens":512,"probe_tokens":512,"fla_reference":{"row":artifact(fla_row),"probe":artifact(fla_probe)},"native_candidate":{"row":artifact(native_row),"probe":artifact(native_probe),"source_lane":artifact(lane),"source_cell":{"batch_size":batch,"prompt_tokens":2048,"decode_tokens":512}},"comparison":artifact(comparison)})
    closure_lane=tag=="7p2" and batch==8
    rkv_policy="vkwr_auto" if closure_lane else "manual"
    lanes.append({"model_pair":pair,"batch_size":batch,"artifact":artifact(lane),"rows":6,"probe_cell":[batch,2048,512],"ada_wagv_lora_require_extension":False,"sm70_wagv_lora_require_extension":batch==1,"rkv_policy":rkv_policy,"fused_norm_mix_num_warps":8 if closure_lane else 4,"fused_wavg_lora":False if closure_lane else True})
candidate.write_text("".join(json.dumps(row)+"\n" for row in rows))
candidate_digest=sha(candidate); sidecar.write_text(f"{candidate_digest}  {candidate.name}\n")
closure={"model_pair":"rwkv-7.2b__qwen3.5-9b","model_size_label":"7.2b","model_path":str(Path(rows[-1]["model_id_or_path"]).resolve()),"batch_size":8,"prompt_tokens":128,"decode_tokens":128,"probe_tokens":512,"reference_contract":{"rwkv_implementation":"native_model","effective_backend":"eager","RWKV7_FAST_TOKEN_BACKEND":"module_call","RWKV7_NATIVE_MODEL_BACKEND":"eager","RWKV7_FAST_PREFILL":"0","RWKV7_NATIVE_PREFILL_GRAPH":"0"},"candidate_contract":{"rwkv_implementation":"native_model","effective_backend":"native_graph","rkv_policy":"vkwr_auto","state_dtype":"torch.float16","triton_fp16_state":True,"fused_norm_mix_num_warps":8,"fused_wavg_lora":False},"native_eager_reference":{"row":artifact(out/"decode_correctness_7p2_b8_p128_native_eager.jsonl"),"probe":artifact(out/"decode_correctness_7p2_b8_p128_native_eager.pt")},"native_graph_candidate":{"row":artifact(out/"decode_correctness_7p2_b8_p128_native_graph.jsonl"),"probe":artifact(out/"decode_correctness_7p2_b8_p128_native_graph.pt")},"comparison":artifact(out/"decode_correctness_7p2_b8_p128_native_eager_compare.json")}
correctness.write_text(json.dumps({"schema_version":1,"protocol":"rwkv_native_graph_fla_correctness_v100_v1","benchmark_repository_commit":commit,"model_hashes_sha256":sha(model_hashes),"runtime":artifact(runtime),"coverage":{"models":4,"batch_sizes":[1,8],"entries":8,"baseline_fresh_gpu_processes":8,"candidate_additional_gpu_processes":1,"candidate_formal_lane_processes":8,"targeted_native_eager_fresh_gpu_processes":1,"prompt_tokens":2048,"decode_tokens":512,"probe_tokens":512,"targeted_closure_entries":1},"reference_contract":{"rwkv_implementation":"wrapper_repo","RWKV7_FAST_TOKEN_BACKEND":"fla","RWKV7_NATIVE_MODEL_BACKEND":"eager","RWKV7_FAST_PREFILL":"0","RWKV7_NATIVE_PREFILL_GRAPH":"0","TORCHDYNAMO_DISABLE":"1","TORCH_COMPILE_DISABLE":"1","performance_role":False},"gates":{"greedy_tokens":"exact_all_512","prompt_logits_min_row_cosine":0.9999,"final_logits_min_row_cosine":0.9999,"decode_logits_all_finite":True,"b8_distinct_prompts":True},"targeted_same_implementation_closure":closure,"entries":entries},indent=2)+"\n")
routes.write_text(json.dumps({"schema_version":1,"protocol":"qwen35_v100_paired_pd_v1","benchmark_repository_commit":commit,"repository_clean_pre_and_post":True,"candidate_rows":48,"candidate_result":artifact(candidate),"candidate_sha256_sidecar":artifact(sidecar),"model_hash_contract":{"algorithm":"sha256","scope":"every recursive regular file","before":artifact(model_hashes)},"native_graph_fla_correctness_manifest":artifact(correctness),"runtime_lock":artifact(runtime),"pip_freeze":artifact(out/"pip-freeze.txt"),"system_identity":artifact(system),"forced_environment":{"CUDA_VISIBLE_DEVICES":"0","CUDA_DEVICE_ORDER":"PCI_BUS_ID","PYTHONPATH":os.environ["PYTHONPATH"],"PYTORCH_CUDA_ALLOC_CONF":"expandable_segments:True","TORCH_CUDA_ARCH_LIST":"7.0","CPATH":str(Path(os.environ["CPATH"]).resolve()),"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","TOKENIZERS_PARALLELISM":"false","RWKV7_FAST_TOKEN_BACKEND":"native_graph","RWKV7_NATIVE_MODEL_BACKEND":"native_graph","RWKV7_NATIVE_PREFILL_GRAPH":"unset_exact_card_policy","RWKV7_NATIVE_GRAPH_ADA_WAGV_LORA_REQUIRE_EXTENSION":"0","RWKV7_NATIVE_GRAPH_SM70_WAGV_LORA_REQUIRE_EXTENSION":"per_lane_exact_v100_policy","RWKV7_NATIVE_GRAPH_RKV_POLICY":"per_lane_exact_v100_policy","RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX_NUM_WARPS":"per_lane_exact_v100_policy","RWKV7_NATIVE_GRAPH_FUSED_WAVG_LORA":"per_lane_exact_v100_policy","RWKV7_NATIVE_GRAPH_ADA_WAGV_BMM":"0","RWKV7_NATIVE_GRAPH_SM120_WAGV_BMM_G":"0","RWKV7_NATIVE_GRAPH_SM120_COMPILED_FFN":"0","CACHE_ROOT":cache_root},"lanes":lanes},indent=2)+"\n")
PY

hash_models "${MODEL_HASHES_AFTER}"
cmp --silent "${MODEL_HASHES}" "${MODEL_HASHES_AFTER}" || { echo "model inputs changed during formal capture" >&2; exit 2; }
validate_repository
env -i "${COMMON_ENV[@]}" "${PYTHON_BIN}" - "${ROUTE_MANIFEST}" "${MODEL_HASHES_AFTER}" <<'PY'
import hashlib,json,sys
from pathlib import Path
path=Path(sys.argv[1]); after=Path(sys.argv[2]); doc=json.loads(path.read_text())
doc["model_hash_contract"]["after"]={"path":str(after.resolve()),"sha256":hashlib.sha256(after.read_bytes()).hexdigest()}
doc["model_hash_contract"]["byte_identical"]=True
path.write_text(json.dumps(doc,indent=2)+"\n")
PY
echo "wrote ${CANDIDATE}"
