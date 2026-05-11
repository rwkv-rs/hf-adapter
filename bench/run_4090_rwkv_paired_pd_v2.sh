#!/usr/bin/env bash
# Capture the strict RTX 4090 RWKV matrix and independent FLA correctness probes.
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-${1:-}}"
CACHE_ROOT="${CACHE_ROOT:-}"
REPOSITORY_COMMIT="${REPOSITORY_COMMIT:-}"
RWKV_04_MODEL="${RWKV_04_MODEL:-}"
RWKV_15_MODEL="${RWKV_15_MODEL:-}"
RWKV_29_MODEL="${RWKV_29_MODEL:-}"
RWKV_72_MODEL="${RWKV_72_MODEL:-}"
PROTOCOL="${PROTOCOL:-qwen35_4090_paired_pd_v2}"
CORRECTNESS_PROTOCOL="${CORRECTNESS_PROTOCOL:-rwkv_native_graph_fla_correctness_4090_v2}"
EXPECTED_GPU_NAME="${EXPECTED_GPU_NAME:-NVIDIA GeForce RTX 4090}"
EXPECTED_PYTHON="${EXPECTED_PYTHON:-3.12.8}"
TORCH_CUDA_ARCH="${TORCH_CUDA_ARCH:-8.9}"
SMALL_B8_MODE="${SMALL_B8_MODE:-sm89_bundle}"
SPLIT_7P2_B8="${SPLIT_7P2_B8:-1}"
ADA_WAGV_BMM_OVERRIDE="${ADA_WAGV_BMM_OVERRIDE:-}"
ROUTE_PROFILE="${ROUTE_PROFILE:-default}"

if [[ -z "${OUT_DIR}" || -z "${CACHE_ROOT}" || -z "${REPOSITORY_COMMIT}" ]]; then
  echo "OUT_DIR, CACHE_ROOT and REPOSITORY_COMMIT are required" >&2
  exit 2
fi
if [[ ! "${REPOSITORY_COMMIT}" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "REPOSITORY_COMMIT must be 40 hexadecimal characters" >&2
  exit 2
fi
for name in RWKV_04_MODEL RWKV_15_MODEL RWKV_29_MODEL RWKV_72_MODEL; do
  [[ -n "${!name}" ]] || { echo "${name} is required" >&2; exit 2; }
done

ROOT="$(realpath -e -- "${ROOT}")"
OUT_DIR="$(realpath -m -- "${OUT_DIR}")"
CACHE_ROOT="$(realpath -m -- "${CACHE_ROOT}")"
RWKV_04_MODEL="$(realpath -e -- "${RWKV_04_MODEL}")"
RWKV_15_MODEL="$(realpath -e -- "${RWKV_15_MODEL}")"
RWKV_29_MODEL="$(realpath -e -- "${RWKV_29_MODEL}")"
RWKV_72_MODEL="$(realpath -e -- "${RWKV_72_MODEL}")"
if [[ "${PYTHON_BIN}" == */* ]]; then
  python_dir="$(realpath -e -- "$(dirname -- "${PYTHON_BIN}")")"
  PYTHON_BIN="${python_dir}/$(basename -- "${PYTHON_BIN}")"
else
  PYTHON_BIN="$(command -v -- "${PYTHON_BIN}")"
fi
[[ ! -e "${OUT_DIR}" ]] || { echo "refusing existing OUT_DIR ${OUT_DIR}" >&2; exit 2; }
[[ ! -e "${CACHE_ROOT}" ]] || {
  [[ -d "${CACHE_ROOT}" && -z "$(find "${CACHE_ROOT}" -mindepth 1 -print -quit)" ]] || {
    echo "CACHE_ROOT must be absent or empty" >&2; exit 2;
  }
}
for model in "${RWKV_04_MODEL}" "${RWKV_15_MODEL}" "${RWKV_29_MODEL}" "${RWKV_72_MODEL}"; do
  [[ -f "${model}/config.json" ]] || { echo "missing config in ${model}" >&2; exit 2; }
  compgen -G "${model}/*.safetensors" >/dev/null || { echo "missing weights in ${model}" >&2; exit 2; }
done

validate_repository() {
  local top head dirty
  top="$(realpath -e -- "$(git -C "${ROOT}" rev-parse --show-toplevel)")"
  head="$(git -C "${ROOT}" rev-parse HEAD)"
  dirty="$(git -C "${ROOT}" status --porcelain --untracked-files=all)"
  [[ "${top}" == "${ROOT}" && "${head,,}" == "${REPOSITORY_COMMIT,,}" && -z "${dirty}" ]] || {
    echo "formal capture requires the exact clean repository commit" >&2
    exit 2
  }
}
validate_repository
mkdir -p "${OUT_DIR}/logs" "${CACHE_ROOT}"
cd "${ROOT}"

COMMON_ENV=(
  "HOME=${HOME}" "LANG=C.UTF-8"
  "PATH=$(dirname "${PYTHON_BIN}"):/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
  "CUDA_VISIBLE_DEVICES=0" "CUDA_DEVICE_ORDER=PCI_BUS_ID"
  "PYTHONPATH=${ROOT}" "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
  "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH}" "HF_HUB_OFFLINE=1" "TRANSFORMERS_OFFLINE=1"
  "TOKENIZERS_PARALLELISM=false" "REPOSITORY_COMMIT=${REPOSITORY_COMMIT}"
)

gpu_name="$(env -i "${COMMON_ENV[@]}" "${PYTHON_BIN}" -c 'import torch; print(torch.cuda.get_device_name(0))')"
[[ "${gpu_name}" == "${EXPECTED_GPU_NAME}" ]] || { echo "unexpected GPU ${gpu_name}" >&2; exit 2; }

env -i "${COMMON_ENV[@]}" "${PYTHON_BIN}" -m pip freeze --all | LC_ALL=C sort > "${OUT_DIR}/pip-freeze.txt"
nvidia-smi --query-gpu=name,uuid,pci.bus_id,compute_cap,driver_version,memory.total --format=csv > "${OUT_DIR}/system.csv"
env -i "${COMMON_ENV[@]}" "${PYTHON_BIN}" - "${OUT_DIR}/runtime-lock.json" "${OUT_DIR}/pip-freeze.txt" "${PROTOCOL}" "${REPOSITORY_COMMIT}" "${EXPECTED_PYTHON}" "${TORCH_CUDA_ARCH}" <<'PY'
import hashlib, json, platform, sys
from importlib.metadata import version
import torch, transformers, triton
runtime={
 "python":platform.python_version(),"torch":str(torch.__version__),
 "torch_cuda":str(torch.version.cuda),"triton":str(triton.__version__),
 "transformers":str(transformers.__version__),
 "fla":version("flash-linear-attention"),"causal_conv1d":version("causal-conv1d"),
}
expected={"python":sys.argv[5],"torch":"2.7.1+cu126","torch_cuda":"12.6","triton":"3.3.1","transformers":"5.12.1","fla":"0.5.1","causal_conv1d":"1.6.2.post1"}
if runtime != expected: raise SystemExit(f"runtime mismatch: {runtime!r} != {expected!r}")
pip=open(sys.argv[2],"rb").read()
doc={"schema_version":1,"protocol":sys.argv[3],"repository_commit":sys.argv[4],"runtime":runtime,"pip_freeze_sha256":hashlib.sha256(pip).hexdigest(),"torch_cuda_arch_list":sys.argv[6]}
open(sys.argv[1],"w",encoding="utf-8").write(json.dumps(doc,indent=2)+"\n")
PY

hash_models() {
  env -i "${COMMON_ENV[@]}" "${PYTHON_BIN}" - "$1" "${RWKV_04_MODEL}" "${RWKV_15_MODEL}" "${RWKV_29_MODEL}" "${RWKV_72_MODEL}" <<'PY'
import hashlib,sys
from pathlib import Path
lines=[]
for raw in sys.argv[2:]:
 root=Path(raw).resolve(strict=True); lines.append(f"[{root.as_posix()}]")
 for path in sorted((p for p in root.rglob("*") if p.is_file()),key=lambda p:p.relative_to(root).as_posix()):
  lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}")
Path(sys.argv[1]).write_text("\n".join(lines)+"\n",encoding="utf-8")
PY
}
hash_models "${OUT_DIR}/model_hashes.sha256"

lane_results=()
lane_probes=()
build_route_env() {
  local tag="$1" batch="$2" mode="$3"
  route_env=(
    "RWKV7_FAST_TOKEN_BACKEND=native_graph" "RWKV7_NATIVE_MODEL_BACKEND=native_graph"
    "RWKV7_BENCHMARK_ROUTE_PROFILE=${ROUTE_PROFILE}"
    "RWKV7_NATIVE_GRAPH_SM120_WAGV_BMM_G=0"
    "RWKV7_NATIVE_GRAPH_SM120_COMPILED_FFN=0"
  )
  if [[ -n "${ADA_WAGV_BMM_OVERRIDE}" ]]; then
    route_env+=("RWKV7_NATIVE_GRAPH_ADA_WAGV_BMM=${ADA_WAGV_BMM_OVERRIDE}")
  fi
  if [[ "${mode}" == sm89_bundle ]]; then
    route_env=(
      "RWKV7_FAST_TOKEN_BACKEND=native_graph" "RWKV7_NATIVE_MODEL_BACKEND=native_graph"
      "RWKV7_BENCHMARK_ROUTE_PROFILE=${ROUTE_PROFILE}"
      "RWKV7_NATIVE_GRAPH_ADA_WAGV_BMM=1" "RWKV7_NATIVE_GRAPH_SM120_WAGV_BMM_G=1"
      "RWKV7_NATIVE_GRAPH_SM120_COMPILED_FFN=1" "RWKV7_NATIVE_GRAPH_RKV_POLICY=vkwr_auto"
      "RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN=0" "RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_LOW_MEMORY_PACK=0"
      "RWKV7_BLACKWELL_TORCH_COMPILE=1"
    )
  fi
  if [[ "${ROUTE_PROFILE}" != sm86_qwen_alignment ]]; then
    return
  fi

  local rkv_policy=vkwr_auto state_dtype=fp16 fp16_recurrent=1
  local require_wagv_extension=0 ada_linear=0 ada_linear_required=0
  local base_bmm=0 fused_g=0 compiled_ffn=0
  [[ "${tag}" == 7p2 ]] && rkv_policy=manual
  if [[ "${batch}" == 1 ]]; then
    require_wagv_extension=1
    if [[ "${tag}" == 0p4 || "${tag}" == 7p2 ]]; then
      ada_linear=1
      ada_linear_required=1
    fi
  elif [[ "${tag}" == 0p4 || "${tag}" == 1p5 ]]; then
    state_dtype=fp32
    fp16_recurrent=0
    base_bmm=1
    fused_g=1
    compiled_ffn=1
  elif [[ "${tag}" == 2p9 ]]; then
    base_bmm=1
  fi
  route_env=(
    "RWKV7_FAST_TOKEN_BACKEND=native_graph" "RWKV7_NATIVE_MODEL_BACKEND=native_graph"
    "RWKV7_BENCHMARK_ROUTE_PROFILE=${ROUTE_PROFILE}"
    "RWKV7_NATIVE_GRAPH_RKV_POLICY=${rkv_policy}"
    "RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX=1"
    "RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX_NUM_WARPS=8"
    "RWKV7_NATIVE_GRAPH_FUSED_RECURRENT_RAW=1"
    "RWKV7_NATIVE_GRAPH_STATE_DTYPE=${state_dtype}"
    "RWKV7_NATIVE_GRAPH_FP16_RECURRENT=${fp16_recurrent}"
    "RWKV7_NATIVE_PREFILL_FP16_RECURRENT=${fp16_recurrent}"
    "RWKV7_NATIVE_GRAPH_PRECOMPUTE_EMB_LN0=1"
    "RWKV7_NATIVE_GRAPH_FUSED_OUTPUT_PROJECT=0"
    "RWKV7_NATIVE_GRAPH_ADA_WAG_LORA=0"
    "RWKV7_NATIVE_GRAPH_ADA_WAGV_LORA=1"
    "RWKV7_NATIVE_GRAPH_ADA_WAGV_LORA_REQUIRE_EXTENSION=${require_wagv_extension}"
    "RWKV7_NATIVE_GRAPH_ADA_LINEAR=${ada_linear}"
    "RWKV7_NATIVE_GRAPH_ADA_LINEAR_REQUIRE_EXTENSION=${ada_linear_required}"
    "RWKV7_NATIVE_GRAPH_ADA_LINEAR_ROWS=1"
    "RWKV7_NATIVE_GRAPH_ADA_LINEAR_ROLES=hidden,ffn_up,ffn_down"
    "RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN=0"
    "RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_LOW_MEMORY_PACK=0"
    "RWKV7_NATIVE_GRAPH_ADA_WAGV_BMM=${base_bmm}"
    "RWKV7_NATIVE_GRAPH_SM120_WAGV_BMM_G=${fused_g}"
    "RWKV7_NATIVE_GRAPH_SM120_COMPILED_FFN=${compiled_ffn}"
  )
  if [[ "${compiled_ffn}" == 1 ]]; then
    route_env+=("RWKV7_BLACKWELL_TORCH_COMPILE=1")
  fi
}

run_lane() {
  local tag="$1" model="$2" pair="$3" size="$4" batch="$5" mode="$6"
  local result="${OUT_DIR}/rwkv_${tag}_b${batch}.jsonl"
  local probe="${OUT_DIR}/decode_correctness_${tag}_b${batch}_native.pt"
  local cache="${CACHE_ROOT}/${tag}_b${batch}"
  mkdir -p "${cache}/inductor" "${cache}/triton" "${cache}/xdg"
  build_route_env "${tag}" "${batch}" "${mode}"
  env -i "${COMMON_ENV[@]}" "${route_env[@]}" \
    "XDG_CACHE_HOME=${cache}/xdg" "TORCHINDUCTOR_CACHE_DIR=${cache}/inductor" "TRITON_CACHE_DIR=${cache}/triton" \
    "${PYTHON_BIN}" bench/bench_cross_model_speed_resident.py \
      --model "${model}" --model-kind rwkv --model-role candidate \
      --model-pair "${pair}" --model-size-label "${size}" \
      --benchmark-matrix "${PROTOCOL}" --optimization-lane best_optimized_hf \
      --dtype fp16 --quantization none --device cuda --batch-sizes "${batch}" \
      --prompt-tokens 128 512 2048 --decode-tokens 128 512 \
      --prefill-chunk-size 512 --warmup 3 --runs 7 --rwkv-attn-mode fused_recurrent \
      --rwkv-code-source repo --rwkv-implementation auto \
      --probe-output "${probe}" --probe-cell "${batch}x2048x512" \
      --probe-tokens 512 --probe-batch-size "${batch}" --fail-fast --results "${result}" \
      > "${OUT_DIR}/logs/rwkv_${tag}_b${batch}.log" 2>&1
  lane_results+=("${result}"); lane_probes+=("${probe}")
}

run_7p2_b8() {
  local result="${OUT_DIR}/rwkv_7p2_b8.jsonl"
  local short="${OUT_DIR}/rwkv_7p2_b8_short.jsonl"
  local long="${OUT_DIR}/rwkv_7p2_b8_long.jsonl"
  local probe="${OUT_DIR}/decode_correctness_7p2_b8_native.pt"
  local common=(
    "RWKV7_FAST_TOKEN_BACKEND=native_graph" "RWKV7_NATIVE_MODEL_BACKEND=native_graph"
    "RWKV7_NATIVE_GRAPH_SM120_WAGV_BMM_G=0" "RWKV7_NATIVE_GRAPH_SM120_COMPILED_FFN=0"
  )
  if [[ -n "${ADA_WAGV_BMM_OVERRIDE}" ]]; then
    common+=("RWKV7_NATIVE_GRAPH_ADA_WAGV_BMM=${ADA_WAGV_BMM_OVERRIDE}")
  fi
  mkdir -p "${CACHE_ROOT}/7p2_b8_short" "${CACHE_ROOT}/7p2_b8_long"
  env -i "${COMMON_ENV[@]}" "${common[@]}" \
    "XDG_CACHE_HOME=${CACHE_ROOT}/7p2_b8_short" \
    "${PYTHON_BIN}" bench/bench_cross_model_speed_resident.py \
      --model "${RWKV_72_MODEL}" --model-kind rwkv --model-role candidate \
      --model-pair rwkv-7.2b__qwen3.5-9b --model-size-label 7.2b \
      --benchmark-matrix "${PROTOCOL}" --optimization-lane best_optimized_hf \
      --dtype fp16 --quantization none --device cuda \
      --cells 8x128x128 8x128x512 8x512x128 8x512x512 \
      --prefill-chunk-size 512 --warmup 3 --runs 7 --rwkv-attn-mode fused_recurrent \
      --rwkv-code-source repo --rwkv-implementation auto --fail-fast --results "${short}" \
      > "${OUT_DIR}/logs/rwkv_7p2_b8_short.log" 2>&1
  env -i "${COMMON_ENV[@]}" "${common[@]}" "RWKV7_NATIVE_PREFILL_GRAPH=0" \
    "XDG_CACHE_HOME=${CACHE_ROOT}/7p2_b8_long" \
    "${PYTHON_BIN}" bench/bench_cross_model_speed_resident.py \
      --model "${RWKV_72_MODEL}" --model-kind rwkv --model-role candidate \
      --model-pair rwkv-7.2b__qwen3.5-9b --model-size-label 7.2b \
      --benchmark-matrix "${PROTOCOL}" --optimization-lane best_optimized_hf \
      --dtype fp16 --quantization none --device cuda --cells 8x2048x128 8x2048x512 \
      --prefill-chunk-size 512 --warmup 3 --runs 7 --rwkv-attn-mode fused_recurrent \
      --rwkv-code-source repo --rwkv-implementation auto \
      --probe-output "${probe}" --probe-cell 8x2048x512 --probe-tokens 512 --probe-batch-size 8 \
      --fail-fast --results "${long}" > "${OUT_DIR}/logs/rwkv_7p2_b8_long.log" 2>&1
  env -i "${COMMON_ENV[@]}" "${PYTHON_BIN}" - "${short}" "${long}" "${result}" <<'PY'
from pathlib import Path
import sys
with Path(sys.argv[3]).open("x",encoding="utf-8",newline="\n") as out:
 for raw in sys.argv[1:3]: out.write(Path(raw).read_text(encoding="utf-8"))
PY
  lane_results+=("${result}"); lane_probes+=("${probe}")
}

run_lane 0p4 "${RWKV_04_MODEL}" rwkv-0.4b__qwen3.5-0.8b 0.4b 1 base
run_lane 0p4 "${RWKV_04_MODEL}" rwkv-0.4b__qwen3.5-0.8b 0.4b 8 "${SMALL_B8_MODE}"
run_lane 1p5 "${RWKV_15_MODEL}" rwkv-1.5b__qwen3.5-2b 1.5b 1 base
run_lane 1p5 "${RWKV_15_MODEL}" rwkv-1.5b__qwen3.5-2b 1.5b 8 "${SMALL_B8_MODE}"
run_lane 2p9 "${RWKV_29_MODEL}" rwkv-2.9b__qwen3.5-4b 2.9b 1 base
run_lane 2p9 "${RWKV_29_MODEL}" rwkv-2.9b__qwen3.5-4b 2.9b 8 base
run_lane 7p2 "${RWKV_72_MODEL}" rwkv-7.2b__qwen3.5-9b 7.2b 1 base
if [[ "${SPLIT_7P2_B8}" == 1 ]]; then
  run_7p2_b8
else
  run_lane 7p2 "${RWKV_72_MODEL}" rwkv-7.2b__qwen3.5-9b 7.2b 8 base
fi

env -i "${COMMON_ENV[@]}" "${PYTHON_BIN}" - "${OUT_DIR}/rwkv_candidate.jsonl" "${lane_results[@]}" <<'PY'
import json,sys
from pathlib import Path
rank={"rwkv-0.4b__qwen3.5-0.8b":0,"rwkv-1.5b__qwen3.5-2b":1,"rwkv-2.9b__qwen3.5-4b":2,"rwkv-7.2b__qwen3.5-9b":3}
rows=[]
for raw in sys.argv[2:]: rows += [json.loads(x) for x in Path(raw).read_text().splitlines() if x.strip()]
rows.sort(key=lambda r:(rank[r["model_pair"]],r["batch_size"],r["prompt_tokens"],r["decode_tokens"]))
keys=[(r["model_pair"],r["batch_size"],r["prompt_tokens"],r["decode_tokens"]) for r in rows]
if len(rows)!=48 or len(set(keys))!=48 or any(r.get("status")!="pass" for r in rows): raise SystemExit("candidate matrix is incomplete")
with Path(sys.argv[1]).open("x",encoding="utf-8",newline="\n") as f:
 for row in rows: f.write(json.dumps(row,separators=(",",":"))+"\n")
PY
sha256sum "${OUT_DIR}/rwkv_candidate.jsonl" > "${OUT_DIR}/rwkv_candidate.sha256"

run_fla() {
  local tag="$1" model="$2" pair="$3" size="$4" batch="$5"
  local row="${OUT_DIR}/decode_correctness_${tag}_b${batch}_fla.jsonl"
  local probe="${OUT_DIR}/decode_correctness_${tag}_b${batch}_fla.pt"
  local cache="${CACHE_ROOT}/fla_${tag}_b${batch}"
  mkdir -p "${cache}/inductor" "${cache}/triton" "${cache}/xdg"
  env -i "${COMMON_ENV[@]}" "RWKV7_FAST_TOKEN_BACKEND=fla" "RWKV7_NATIVE_MODEL=0" \
    "RWKV7_NATIVE_MODEL_BACKEND=eager" "RWKV7_FAST_PREFILL=0" "RWKV7_NATIVE_PREFILL_GRAPH=0" \
    "TORCH_COMPILE_DISABLE=1" "TORCHDYNAMO_DISABLE=1" \
    "XDG_CACHE_HOME=${cache}/xdg" "TORCHINDUCTOR_CACHE_DIR=${cache}/inductor" "TRITON_CACHE_DIR=${cache}/triton" \
    "${PYTHON_BIN}" bench/bench_cross_model_speed_resident.py \
      --model "${model}" --model-kind rwkv --model-role candidate --model-pair "${pair}" \
      --model-size-label "${size}" --benchmark-matrix "${CORRECTNESS_PROTOCOL}" \
      --optimization-lane fla_reference --dtype fp16 --quantization none --device cuda \
      --cells "${batch}x2048x512" --prefill-chunk-size 512 --warmup 1 --runs 1 \
      --rwkv-attn-mode fused_recurrent --rwkv-code-source repo --rwkv-implementation wrapper_repo \
      --probe-output "${probe}" --probe-cell "${batch}x2048x512" \
      --probe-tokens 512 --probe-batch-size "${batch}" --fail-fast --results "${row}" \
      > "${OUT_DIR}/logs/decode_correctness_${tag}_b${batch}_fla.log" 2>&1
}
for spec in \
  "0p4|${RWKV_04_MODEL}|rwkv-0.4b__qwen3.5-0.8b|0.4b" \
  "1p5|${RWKV_15_MODEL}|rwkv-1.5b__qwen3.5-2b|1.5b" \
  "2p9|${RWKV_29_MODEL}|rwkv-2.9b__qwen3.5-4b|2.9b" \
  "7p2|${RWKV_72_MODEL}|rwkv-7.2b__qwen3.5-9b|7.2b"; do
  IFS='|' read -r tag model pair size <<<"${spec}"
  run_fla "${tag}" "${model}" "${pair}" "${size}" 1
  run_fla "${tag}" "${model}" "${pair}" "${size}" 8
done

env -i "${COMMON_ENV[@]}" "${PYTHON_BIN}" - "${OUT_DIR}" "${REPOSITORY_COMMIT}" "${OUT_DIR}/model_hashes.sha256" "${CORRECTNESS_PROTOCOL}" <<'PY'
import hashlib,json,sys
from pathlib import Path
import torch
from bench.compare_rwkv_prefill_probe import compare
root=Path(sys.argv[1]); commit=sys.argv[2]; model_hashes=Path(sys.argv[3])
models=(
 ("0p4","rwkv-0.4b__qwen3.5-0.8b"),("1p5","rwkv-1.5b__qwen3.5-2b"),
 ("2p9","rwkv-2.9b__qwen3.5-4b"),("7p2","rwkv-7.2b__qwen3.5-9b"),
)
def evidence(path): return {"path":path.name,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
entries=[]
for tag,pair in models:
 for batch in (1,8):
  fla=root/f"decode_correctness_{tag}_b{batch}_fla.pt"
  native=root/f"decode_correctness_{tag}_b{batch}_native.pt"
  comparison=compare(torch.load(fla,map_location="cpu",weights_only=True),torch.load(native,map_location="cpu",weights_only=True),0.9999)
  comparison["contract_errors"]=[]
  if comparison.get("status")!="pass" or comparison.get("greedy_tokens_match") is not True: raise SystemExit(f"{pair} B{batch} correctness failed")
  comp=root/f"decode_correctness_{tag}_b{batch}_compare.json"
  comp.write_text(json.dumps(comparison,indent=2)+"\n",encoding="utf-8")
  entries.append({"model_pair":pair,"batch_size":batch,"prompt_tokens":2048,"decode_tokens":512,"probe_tokens":512,"fla_probe":evidence(fla),"native_probe":evidence(native),"comparison":evidence(comp),"status":"pass"})
doc={"schema_version":1,"protocol":sys.argv[4],"benchmark_repository_commit":commit,"model_hashes_sha256":hashlib.sha256(model_hashes.read_bytes()).hexdigest(),"entries":entries}
(root/"rwkv_native_graph_fla_correctness.json").write_text(json.dumps(doc,indent=2)+"\n",encoding="utf-8")
PY

hash_models "${OUT_DIR}/model_hashes.after.sha256"
cmp -s "${OUT_DIR}/model_hashes.sha256" "${OUT_DIR}/model_hashes.after.sha256" || { echo "model files changed" >&2; exit 2; }
validate_repository
printf '0\n' > "${OUT_DIR}/exit_code.txt"
