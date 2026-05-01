#!/usr/bin/env bash
# Capture the 48-row RTX 5090 RWKV candidate for qwen35_paired_decode_v1.
#
# This entrypoint never loads or reruns Qwen. Its FLA correctness oracles use
# the same RWKV checkpoints; each production (checkpoint, batch-size) lane is
# a fresh resident-runner process so CUDA Graph and torch.compile state cannot
# leak across promoted route boundaries.
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT_DIR="${OUT_DIR:-${1:-}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="0"
REPOSITORY_COMMIT="${REPOSITORY_COMMIT:-}"

BENCHMARK_MATRIX="qwen35_paired_decode_v1"
OPTIMIZATION_LANE="best_optimized_hf"

required=(RWKV_04_MODEL RWKV_15_MODEL RWKV_29_MODEL RWKV_72_MODEL)
if [[ -z "${OUT_DIR}" ]]; then
  echo "usage: OUT_DIR=... RWKV_04_MODEL=... RWKV_15_MODEL=... RWKV_29_MODEL=... RWKV_72_MODEL=... REPOSITORY_COMMIT=... $0" >&2
  exit 2
fi
if [[ ! "${REPOSITORY_COMMIT}" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "REPOSITORY_COMMIT must be the explicit 40-hex commit under test" >&2
  exit 2
fi

# Resolve every caller-supplied path before changing cwd. This keeps the
# append-never checks and the eventual reads/writes pointed at one directory.
ROOT="$(realpath -e -- "${ROOT}")"
OUT_DIR="$(realpath -m -- "${OUT_DIR}")"
if [[ "${PYTHON_BIN}" == */* ]]; then
  # Preserve the final launcher component: a venv's ``bin/python`` is usually
  # a symlink to the system interpreter, and resolving that symlink would drop
  # the venv's site-packages. Canonicalize only its containing directory.
  python_dir="$(realpath -e -- "$(dirname -- "${PYTHON_BIN}")")"
  PYTHON_BIN="${python_dir}/$(basename -- "${PYTHON_BIN}")"
else
  PYTHON_BIN="$(command -v -- "${PYTHON_BIN}")"
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "PYTHON_BIN must resolve to an executable Python launcher" >&2
  exit 2
fi
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" || ! -d "${!name}" ]]; then
    echo "${name} must name a local RWKV Hugging Face model directory" >&2
    exit 2
  fi
  printf -v "${name}" '%s' "$(realpath -e -- "${!name}")"
  case "${OUT_DIR}/" in
    "${!name}/"*)
      echo "OUT_DIR must not be inside ${name}; evidence would mutate a model input" >&2
      exit 2
      ;;
  esac
done
case "${OUT_DIR}/" in
  "${ROOT}/"*)
    echo "OUT_DIR must be outside the repository so formal artifacts cannot dirty HEAD" >&2
    exit 2
    ;;
esac

# Provenance is fail-closed: repo code must be exactly the committed tree that
# every result row records. Formal evidence may never attribute a dirty overlay
# or a different checkout to REPOSITORY_COMMIT.
validate_repository_provenance() {
  local actual_commit repo_root
  repo_root="$(git -C "${ROOT}" rev-parse --show-toplevel)"
  repo_root="$(realpath -e -- "${repo_root}")"
  if [[ "${repo_root}" != "${ROOT}" ]]; then
    echo "ROOT must be the repository top level: ${repo_root}" >&2
    exit 2
  fi
  actual_commit="$(git -C "${ROOT}" rev-parse HEAD)"
  if [[ "${actual_commit,,}" != "${REPOSITORY_COMMIT,,}" ]]; then
    echo "repository HEAD ${actual_commit} does not match REPOSITORY_COMMIT ${REPOSITORY_COMMIT}" >&2
    exit 2
  fi
  if [[ -n "$(git -C "${ROOT}" status --porcelain --untracked-files=all)" ]]; then
    echo "formal candidate requires a completely clean repository worktree" >&2
    exit 2
  fi
}
validate_repository_provenance

lane_results=(
  "${OUT_DIR}/rwkv_0p4_b1.jsonl"
  "${OUT_DIR}/rwkv_0p4_b8.jsonl"
  "${OUT_DIR}/rwkv_1p5_b1.jsonl"
  "${OUT_DIR}/rwkv_1p5_b8.jsonl"
  "${OUT_DIR}/rwkv_2p9_b1.jsonl"
  "${OUT_DIR}/rwkv_2p9_b8.jsonl"
  "${OUT_DIR}/rwkv_7p2_b1.jsonl"
  "${OUT_DIR}/rwkv_7p2_b8.jsonl"
)
candidate_result="${OUT_DIR}/rwkv_candidate.jsonl"
candidate_sha256="${OUT_DIR}/rwkv_candidate.sha256"
route_manifest="${OUT_DIR}/rwkv_candidate_routes.json"
runtime_lock="${OUT_DIR}/runtime-lock.json"
pip_freeze="${OUT_DIR}/pip-freeze.txt"
system_csv="${OUT_DIR}/system.csv"
model_hashes="${OUT_DIR}/model_hashes.sha256"
model_hashes_after="${OUT_DIR}/model_hashes.after.sha256"
sm120_ab_manifest="${OUT_DIR}/rwkv_sm120_b8_ab.json"
decode_correctness_manifest="${OUT_DIR}/rwkv_native_graph_fla_correctness.json"
sm120_ab_artifacts=(
  "${OUT_DIR}/sm120_0p4_baseline.jsonl"
  "${OUT_DIR}/sm120_0p4_baseline.pt"
  "${OUT_DIR}/sm120_0p4_candidate.jsonl"
  "${OUT_DIR}/sm120_0p4_candidate.pt"
  "${OUT_DIR}/sm120_0p4_compare.json"
  "${OUT_DIR}/sm120_1p5_baseline.jsonl"
  "${OUT_DIR}/sm120_1p5_baseline.pt"
  "${OUT_DIR}/sm120_1p5_candidate.jsonl"
  "${OUT_DIR}/sm120_1p5_candidate.pt"
  "${OUT_DIR}/sm120_1p5_compare.json"
)
decode_correctness_artifacts=()
decode_correctness_logs=()
for tag in 0p4 1p5 2p9 7p2; do
  for batch in 1 8; do
    decode_correctness_artifacts+=(
      "${OUT_DIR}/decode_correctness_${tag}_b${batch}_fla_reference.jsonl"
      "${OUT_DIR}/decode_correctness_${tag}_b${batch}_fla_reference.pt"
      "${OUT_DIR}/decode_correctness_${tag}_b${batch}_native_candidate.jsonl"
      "${OUT_DIR}/decode_correctness_${tag}_b${batch}_native_candidate.pt"
      "${OUT_DIR}/decode_correctness_${tag}_b${batch}_compare.json"
    )
    decode_correctness_logs+=(
      "${OUT_DIR}/logs/decode_correctness_${tag}_b${batch}_fla_reference.log"
    )
  done
done

# Formal result directories are append-never. A partial run remains available
# for diagnosis; a rerun must use a new directory instead of erasing evidence.
for path in \
  "${lane_results[@]}" \
  "${candidate_result}" \
  "${candidate_sha256}" \
  "${route_manifest}" \
  "${runtime_lock}" \
  "${pip_freeze}" \
  "${system_csv}" \
  "${model_hashes}" \
  "${model_hashes_after}" \
  "${sm120_ab_manifest}" \
  "${sm120_ab_artifacts[@]}" \
  "${decode_correctness_manifest}" \
  "${decode_correctness_artifacts[@]}" \
  "${decode_correctness_logs[@]}"; do
  if [[ -e "${path}" ]]; then
    echo "refusing to overwrite existing artifact: ${path}" >&2
    exit 2
  fi
done
mkdir -p "${OUT_DIR}/logs"
for name in \
  rwkv_0p4_b1 rwkv_0p4_b8 rwkv_1p5_b1 rwkv_1p5_b8 \
  rwkv_2p9_b1 rwkv_2p9_b8 rwkv_7p2_b1 rwkv_7p2_b8 \
  sm120_0p4_baseline sm120_0p4_candidate \
  sm120_1p5_baseline sm120_1p5_candidate; do
  if [[ -e "${OUT_DIR}/logs/${name}.log" ]]; then
    echo "refusing to overwrite existing log: ${OUT_DIR}/logs/${name}.log" >&2
    exit 2
  fi
done

export CUDA_VISIBLE_DEVICES REPOSITORY_COMMIT
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export PYTHONPATH="${ROOT}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TORCH_CUDA_ARCH_LIST="12.0"
export TORCHDYNAMO_DISABLE=0
export TORCH_COMPILE_DISABLE=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

gpu_name="$("${PYTHON_BIN}" - <<'PY'
import torch
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY
)"
"${PYTHON_BIN}" "${ROOT}/bench/check_exact_gpu.py" --model 5090 --name "${gpu_name}"

# The frozen reference validator compares these six fields exactly. Refuse a
# candidate runtime that could never join the immutable Qwen artifact.
"${PYTHON_BIN}" - "${runtime_lock}" "${pip_freeze}" <<'PY'
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version

import torch
import transformers
import triton

expected = {
    "python": "3.10.12",
    "torch": "2.8.0+cu128",
    "torch_cuda": "12.8",
    "triton": "3.4.0",
    "transformers": "5.12.1",
    "fla": "0.5.1",
    "causal_conv1d": "1.6.2.post1",
}
try:
    actual = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "triton": str(triton.__version__),
        "transformers": str(transformers.__version__),
        "fla": version("flash-linear-attention"),
        "causal_conv1d": version("causal-conv1d"),
    }
except PackageNotFoundError as exc:
    raise SystemExit(f"missing frozen-reference runtime package: {exc.name}") from exc
mismatches = [
    f"{name}={actual[name]!r} (expected {wanted!r})"
    for name, wanted in expected.items()
    if actual[name] != wanted
]
if mismatches:
    raise SystemExit("frozen-reference runtime mismatch: " + "; ".join(mismatches))

freeze = subprocess.run(
    [__import__("sys").executable, "-m", "pip", "freeze", "--all"],
    check=True,
    capture_output=True,
).stdout.decode("utf-8")
canonical = "\n".join(sorted(line.rstrip() for line in freeze.splitlines())) + "\n"
digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
expected_digest = "f5bf8ef181f2c1b29b79d6fae5c8019fa85008df120569b9e18646bd09eee5cf"
if digest != expected_digest:
    raise SystemExit(
        f"frozen-reference pip freeze mismatch: {digest} (expected {expected_digest})"
    )
with Path(sys.argv[2]).open("x", encoding="utf-8") as handle:
    handle.write(canonical)
with Path(sys.argv[1]).open("x", encoding="utf-8") as handle:
    json.dump(
        {
            "schema_version": 1,
            "protocol": "qwen35_paired_decode_v1",
            "repository_commit": os.environ["REPOSITORY_COMMIT"],
            "runtime": actual,
            "pip_freeze_sha256": digest,
            "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        },
        handle,
        indent=2,
    )
    handle.write("\n")
PY

driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 | tr -d '[:space:]')"
if [[ "${driver_version}" != "595.58.03" ]]; then
  echo "frozen-reference driver mismatch: ${driver_version} (expected 595.58.03)" >&2
  exit 2
fi
nvidia-smi --query-gpu=name,uuid,pci.bus_id,compute_cap,driver_version,memory.total,power.limit,clocks.current.sm,clocks.max.sm,clocks.current.memory,clocks.max.memory --format=csv > "${system_csv}"

hash_models() {
  local output="$1"
  "${PYTHON_BIN}" - "${output}" \
    "${RWKV_04_MODEL}" "${RWKV_15_MODEL}" "${RWKV_29_MODEL}" "${RWKV_72_MODEL}" <<'PY'
import hashlib
import sys
from pathlib import Path

output = Path(sys.argv[1])
models = [Path(value) for value in sys.argv[2:]]
if len(set(models)) != 4:
    raise SystemExit("all four canonical model directories must be distinct")
lines = ["# sha256 of every regular file under each canonical model directory"]
for model in models:
    files = sorted(
        (path for path in model.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(model).as_posix(),
    )
    relative_names = {path.relative_to(model).as_posix() for path in files}
    if "config.json" not in relative_names:
        raise SystemExit(f"model has no config.json: {model}")
    if not any(name.endswith(".safetensors") for name in relative_names):
        raise SystemExit(f"model has no safetensors weights: {model}")
    lines.append(f"[{model}]")
    for path in files:
        relative = path.relative_to(model).as_posix()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        lines.append(f"{digest.hexdigest()}  {relative}")
with output.open("x", encoding="utf-8") as handle:
    handle.write("\n".join(lines) + "\n")
PY
}

hash_models "${model_hashes}"

cd "${ROOT}"

# Remove inherited experiments once, then establish the exact common route.
# Per-lane toggles below are the only RWKV7 variables allowed to vary.
while IFS= read -r name; do
  unset "${name}"
done < <(compgen -e | grep '^RWKV7_' || true)
export RWKV7_FAST_TOKEN_BACKEND=native_graph
export RWKV7_NATIVE_MODEL_BACKEND=native_graph
# Deliberately leave RWKV7_NATIVE_PREFILL_GRAPH unset. The exact-card policy
# selects each supported prefill shape and keeps the known-slower 7.2B B8
# prefill Graph off; the formal Decode backend remains native_graph.

configure_sm120_small_b8() {
  local enabled="$1" size="$2"
  export RWKV7_NATIVE_GRAPH_ADA_WAGV_BMM="${enabled}"
  export RWKV7_NATIVE_GRAPH_SM120_WAGV_BMM_G="${enabled}"
  export RWKV7_NATIVE_GRAPH_SM120_COMPILED_FFN="${enabled}"
  if [[ "${enabled}" == "1" ]]; then
    local cache_root
    export RWKV7_NATIVE_GRAPH_RKV_POLICY=vkwr_auto
    export RWKV7_BLACKWELL_TORCH_COMPILE=1
    cache_root="$(mktemp -d "/home/ubuntu/.cache/rwkv-paired-${size}-XXXXXXXX")"
    export TORCHINDUCTOR_CACHE_DIR="${cache_root}/inductor"
    export TRITON_CACHE_DIR="${cache_root}/triton"
    mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"
  else
    unset RWKV7_NATIVE_GRAPH_RKV_POLICY
    unset RWKV7_BLACKWELL_TORCH_COMPILE
    unset TORCHINDUCTOR_CACHE_DIR
    unset TRITON_CACHE_DIR
  fi
}

run_lane() {
  local model="$1" pair="$2" size="$3" batch="$4" sm120_small_b8="$5"
  local output="$6" log="$7" probe="$8"

  # The paired validator requires all three 24-layer routes for 0.4B/1.5B B8.
  # B1 and every 2.9B/7.2B lane must report all three as unrequested/ineffective.
  export RWKV7_FAST_TOKEN_BACKEND=native_graph
  unset RWKV7_NATIVE_MODEL
  export RWKV7_NATIVE_MODEL_BACKEND=native_graph
  unset RWKV7_FAST_PREFILL
  unset RWKV7_NATIVE_PREFILL_GRAPH
  configure_sm120_small_b8 "${sm120_small_b8}" "${size}-b${batch}"

  "${PYTHON_BIN}" bench/bench_cross_model_speed_resident.py \
    --model "${model}" \
    --model-kind rwkv \
    --model-role candidate \
    --model-pair "${pair}" \
    --model-size-label "${size}" \
    --benchmark-matrix "${BENCHMARK_MATRIX}" \
    --optimization-lane "${OPTIMIZATION_LANE}" \
    --dtype fp16 \
    --quantization none \
    --device cuda \
    --batch-sizes "${batch}" \
    --prompt-tokens 128 512 2048 \
    --decode-tokens 128 512 \
    --prefill-chunk-size 512 \
    --warmup 3 \
    --runs 7 \
    --rwkv-attn-mode fused_recurrent \
    --rwkv-code-source repo \
    --rwkv-implementation auto \
    --probe-output "${probe}" \
    --probe-cell "${batch}x2048x512" \
    --probe-tokens 512 \
    --probe-batch-size "${batch}" \
    --fail-fast \
    --results "${output}" > "${log}" 2>&1
}

run_sm120_ab_variant() {
  local model="$1" pair="$2" size="$3" tag="$4" variant="$5" enabled="$6"
  local row="${OUT_DIR}/sm120_${tag}_${variant}.jsonl"
  local probe="${OUT_DIR}/sm120_${tag}_${variant}.pt"
  local log="${OUT_DIR}/logs/sm120_${tag}_${variant}.log"

  configure_sm120_small_b8 "${enabled}" "ab-${tag}-${variant}"
  "${PYTHON_BIN}" bench/bench_cross_model_speed_resident.py \
    --model "${model}" \
    --model-kind rwkv \
    --model-role candidate \
    --model-pair "${pair}" \
    --model-size-label "${size}" \
    --benchmark-matrix sm120_b8_decode_ab_v1 \
    --optimization-lane "${variant}" \
    --dtype fp16 \
    --quantization none \
    --device cuda \
    --cells 8x2048x512 \
    --prefill-chunk-size 512 \
    --warmup 3 \
    --runs 7 \
    --rwkv-attn-mode fused_recurrent \
    --rwkv-code-source repo \
    --probe-output "${probe}" \
    --probe-tokens 512 \
    --probe-batch-size 8 \
    --results "${row}" \
    --fail-fast > "${log}" 2>&1
}

run_sm120_ab() {
  local model="$1" pair="$2" size="$3" tag="$4"
  run_sm120_ab_variant "${model}" "${pair}" "${size}" "${tag}" baseline 0
  run_sm120_ab_variant "${model}" "${pair}" "${size}" "${tag}" candidate 1
  "${PYTHON_BIN}" bench/compare_rwkv_prefill_probe.py \
    --reference-probe "${OUT_DIR}/sm120_${tag}_baseline.pt" \
    --native-probe "${OUT_DIR}/sm120_${tag}_candidate.pt" \
    --min-cosine 0.9999 \
    --required-batch-size 8 \
    --required-probe-tokens 512 \
    --require-distinct-batch-prompts \
    --output "${OUT_DIR}/sm120_${tag}_compare.json" \
    --fail-on-gate
}

# Full 512-step, B8, distinct-prompt correctness is an independent hard gate.
# It validates the complete promotion bundle against the exact-card baseline;
# the performance matrix below cannot substitute for this token/logits oracle.
run_sm120_ab "${RWKV_04_MODEL}" rwkv-0.4b__qwen3.5-0.8b 0.4b 0p4
run_sm120_ab "${RWKV_15_MODEL}" rwkv-1.5b__qwen3.5-2b 1.5b 1p5

"${PYTHON_BIN}" - "${sm120_ab_manifest}" "${model_hashes}" \
  "${REPOSITORY_COMMIT}" "${OUT_DIR}" \
  "${RWKV_04_MODEL}" "${RWKV_15_MODEL}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
model_hashes = Path(sys.argv[2])
commit = sys.argv[3]
root = Path(sys.argv[4])
model_paths = [Path(value) for value in sys.argv[5:]]

def evidence(path: Path) -> dict[str, str]:
    payload = path.read_bytes()
    return {"path": path.name, "sha256": hashlib.sha256(payload).hexdigest()}

def one_row(path: Path) -> dict[str, object]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 1 or not isinstance(rows[0], dict):
        raise SystemExit(f"{path}: expected exactly one JSON object")
    return rows[0]

entries = []
for (tag, pair, size), model_path in zip(
    (
        ("0p4", "rwkv-0.4b__qwen3.5-0.8b", "0.4b"),
        ("1p5", "rwkv-1.5b__qwen3.5-2b", "1.5b"),
    ),
    model_paths,
    strict=True,
):
    baseline_row = root / f"sm120_{tag}_baseline.jsonl"
    baseline_probe = root / f"sm120_{tag}_baseline.pt"
    candidate_row = root / f"sm120_{tag}_candidate.jsonl"
    candidate_probe = root / f"sm120_{tag}_candidate.pt"
    comparison = root / f"sm120_{tag}_compare.json"
    comparison_data = json.loads(comparison.read_text(encoding="utf-8"))
    if comparison_data.get("status") != "pass":
        raise SystemExit(f"{comparison}: comparison did not pass")
    baseline_data = one_row(baseline_row)
    candidate_data = one_row(candidate_row)
    for lane, row in (("baseline", baseline_data), ("candidate", candidate_data)):
        expected = {
            "benchmark_matrix": "sm120_b8_decode_ab_v1",
            "optimization_lane": lane,
            "benchmark_repository_commit": commit,
            "model_pair": pair,
            "model_size_label": size,
            "model_id_or_path": str(model_path),
            "status": "pass",
        }
        mismatches = [
            f"{field}={row.get(field)!r}, expected {wanted!r}"
            for field, wanted in expected.items()
            if row.get(field) != wanted
        ]
        if mismatches:
            raise SystemExit(f"{baseline_row if lane == 'baseline' else candidate_row}: " + "; ".join(mismatches))
    baseline_rate = float(baseline_data["decode_tokps_total_raw"])
    candidate_rate = float(candidate_data["decode_tokps_total_raw"])
    if candidate_rate <= baseline_rate:
        raise SystemExit(
            f"{pair}: promoted SM120 bundle is not faster: "
            f"{candidate_rate} <= {baseline_rate} tok/s"
        )
    entries.append(
        {
            "model_pair": pair,
            "model_size_label": size,
            "model_path": str(model_path),
            "fresh_process_per_variant": True,
            "baseline_environment": {
                "RWKV7_NATIVE_GRAPH_ADA_WAGV_BMM": 0,
                "RWKV7_NATIVE_GRAPH_SM120_WAGV_BMM_G": 0,
                "RWKV7_NATIVE_GRAPH_SM120_COMPILED_FFN": 0,
                "RWKV7_NATIVE_GRAPH_RKV_POLICY": None,
                "RWKV7_BLACKWELL_TORCH_COMPILE": None,
            },
            "candidate_environment": {
                "RWKV7_NATIVE_GRAPH_ADA_WAGV_BMM": 1,
                "RWKV7_NATIVE_GRAPH_SM120_WAGV_BMM_G": 1,
                "RWKV7_NATIVE_GRAPH_SM120_COMPILED_FFN": 1,
                "RWKV7_NATIVE_GRAPH_RKV_POLICY": "vkwr_auto",
                "RWKV7_BLACKWELL_TORCH_COMPILE": 1,
                "compile_cache": "fresh_unique_directory",
            },
            "baseline_decode_tokps_total_raw": baseline_rate,
            "candidate_decode_tokps_total_raw": candidate_rate,
            "raw_decode_speedup": candidate_rate / baseline_rate,
            "baseline": {
                "row": evidence(baseline_row),
                "probe": evidence(baseline_probe),
            },
            "candidate": {
                "row": evidence(candidate_row),
                "probe": evidence(candidate_probe),
            },
            "comparison": evidence(comparison),
        }
    )

document = {
    "schema_version": 1,
    "protocol": "sm120_b8_decode_ab_v1",
    "benchmark_repository_commit": commit,
    "model_hashes_sha256": hashlib.sha256(model_hashes.read_bytes()).hexdigest(),
    "cell": {
        "batch_size": 8,
        "prompt_tokens": 2048,
        "decode_tokens": 512,
        "probe_tokens": 512,
        "distinct_batch_prompts": True,
    },
    "entries": entries,
}
with output.open("x", encoding="utf-8") as handle:
    json.dump(document, handle, indent=2)
    handle.write("\n")
PY

run_fla_decode_correctness_probe() {
  local model="$1" pair="$2" size="$3" tag="$4" batch="$5"
  local stem="decode_correctness_${tag}_b${batch}_fla_reference"
  local row="${OUT_DIR}/${stem}.jsonl"
  local probe="${OUT_DIR}/${stem}.pt"
  local log="${OUT_DIR}/logs/${stem}.log"

  # This is the only additional GPU load for this pair/B correctness entry.
  # Disable both native-prefill routes and every promoted Decode experiment so
  # effective telemetry must identify the independent FLA oracle.
  configure_sm120_small_b8 0 "correctness-${tag}-b${batch}-fla"
  export RWKV7_FAST_TOKEN_BACKEND=fla
  export RWKV7_NATIVE_MODEL=0
  export RWKV7_NATIVE_MODEL_BACKEND=eager
  export RWKV7_FAST_PREFILL=0
  export RWKV7_NATIVE_PREFILL_GRAPH=0
  "${PYTHON_BIN}" bench/bench_cross_model_speed_resident.py \
    --model "${model}" \
    --model-kind rwkv \
    --model-role candidate \
    --model-pair "${pair}" \
    --model-size-label "${size}" \
    --benchmark-matrix rwkv_native_graph_fla_correctness_v1 \
    --optimization-lane fla_reference \
    --dtype fp16 \
    --quantization none \
    --device cuda \
    --cells "${batch}x2048x512" \
    --prefill-chunk-size 512 \
    --warmup 1 \
    --runs 1 \
    --rwkv-attn-mode fused_recurrent \
    --rwkv-code-source repo \
    --rwkv-implementation wrapper_repo \
    --probe-output "${probe}" \
    --probe-cell "${batch}x2048x512" \
    --probe-tokens 512 \
    --probe-batch-size "${batch}" \
    --fail-fast \
    --results "${row}" > "${log}" 2>&1
}

run_fla_decode_correctness_model() {
  local model="$1" pair="$2" size="$3" tag="$4"
  local batch
  for batch in 1 8; do
    run_fla_decode_correctness_probe \
      "${model}" "${pair}" "${size}" "${tag}" "${batch}"
  done
}

# Collect only eight fresh FLA processes. Native correctness probes are emitted
# in-place by the eight production lanes at their P2048/D512 cell below.
run_fla_decode_correctness_model \
  "${RWKV_04_MODEL}" rwkv-0.4b__qwen3.5-0.8b 0.4b 0p4
run_fla_decode_correctness_model \
  "${RWKV_15_MODEL}" rwkv-1.5b__qwen3.5-2b 1.5b 1p5
run_fla_decode_correctness_model \
  "${RWKV_29_MODEL}" rwkv-2.9b__qwen3.5-4b 2.9b 2p9
run_fla_decode_correctness_model \
  "${RWKV_72_MODEL}" rwkv-7.2b__qwen3.5-9b 7.2b 7p2

assemble_decode_correctness_manifest() {
"${PYTHON_BIN}" - "${decode_correctness_manifest}" "${model_hashes}" \
  "${REPOSITORY_COMMIT}" "${OUT_DIR}" "${runtime_lock}" \
  "${RWKV_04_MODEL}" "${RWKV_15_MODEL}" "${RWKV_29_MODEL}" "${RWKV_72_MODEL}" \
  "${lane_results[@]}" <<'PY'
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import torch

from bench.compare_rwkv_prefill_probe import compare

output = Path(sys.argv[1])
model_hashes = Path(sys.argv[2])
commit = sys.argv[3]
root = Path(sys.argv[4])
runtime_lock = Path(sys.argv[5])
model_paths = [Path(value) for value in sys.argv[6:10]]
lane_paths = [Path(value) for value in sys.argv[10:]]
models = tuple(
    zip(
        ("0p4", "1p5", "2p9", "7p2"),
        (
            "rwkv-0.4b__qwen3.5-0.8b",
            "rwkv-1.5b__qwen3.5-2b",
            "rwkv-2.9b__qwen3.5-4b",
            "rwkv-7.2b__qwen3.5-9b",
        ),
        ("0.4b", "1.5b", "2.9b", "7.2b"),
        model_paths,
        strict=True,
    )
)
lane_path_by_key = {
    (tag, batch): path
    for (tag, batch), path in zip(
        (
            (tag, batch)
            for tag in ("0p4", "1p5", "2p9", "7p2")
            for batch in (1, 8)
        ),
        lane_paths,
        strict=True,
    )
}

def evidence(path: Path) -> dict[str, str]:
    payload = path.read_bytes()
    return {"path": path.name, "sha256": hashlib.sha256(payload).hexdigest()}

def one_row(path: Path) -> dict[str, object]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 1 or not isinstance(rows[0], dict):
        raise SystemExit(f"{path}: expected exactly one JSON object")
    return rows[0]


def production_row(path: Path, batch: int) -> dict[str, object]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 6 or any(not isinstance(row, dict) for row in rows):
        raise SystemExit(f"{path}: expected exactly six production JSON objects")
    matches = [
        row
        for row in rows
        if row.get("batch_size") == batch
        and row.get("prompt_tokens") == 2048
        and row.get("decode_tokens") == 512
    ]
    if len(matches) != 1:
        raise SystemExit(f"{path}: expected one B{batch}/P2048/D512 production cell")
    return matches[0]

def require(row: dict[str, object], field: str, expected: object, source: Path) -> None:
    actual = row.get(field)
    if type(actual) is not type(expected) or actual != expected:
        raise SystemExit(
            f"{source}: {field}={actual!r}, expected {expected!r}"
        )

def load_probe(
    path: Path,
    *,
    pair: str,
    size: str,
    model_path: Path,
    batch: int,
) -> dict[str, object]:
    probe = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(probe, dict):
        raise SystemExit(f"{path}: probe must be a dictionary")
    expected = {
        "probe_schema_version": 2,
        "benchmark_repository_commit": commit,
        "model_pair": pair,
        "model_size_label": size,
        "model_id_or_path": str(model_path),
        "decode_logits_all_finite": True,
    }
    for field, wanted in expected.items():
        require(probe, field, wanted, path)
    input_ids = probe.get("input_ids")
    greedy = probe.get("greedy_tokens")
    finite = probe.get("decode_logits_finite_by_batch")
    expected_greedy_shape = (512,) if batch == 1 else (512, batch)
    if not isinstance(input_ids, torch.Tensor) or tuple(input_ids.shape) != (batch, 2048):
        raise SystemExit(f"{path}: input_ids must have shape [{batch}, 2048]")
    if batch > 1 and int(torch.unique(input_ids, dim=0).shape[0]) != batch:
        raise SystemExit(f"{path}: all B8 prompt rows must be distinct")
    if not isinstance(greedy, torch.Tensor) or tuple(greedy.shape) != expected_greedy_shape:
        raise SystemExit(f"{path}: greedy_tokens must have shape {expected_greedy_shape}")
    if (
        not isinstance(finite, torch.Tensor)
        or tuple(finite.shape) != (batch,)
        or not bool(finite.bool().all())
    ):
        raise SystemExit(f"{path}: every Decode step must be finite for every batch row")
    return probe

entries = []
for tag, pair, size, model_path in models:
    for batch in (1, 8):
        lane_rows: dict[str, dict[str, object]] = {}
        lane_probes: dict[str, dict[str, object]] = {}
        lane_evidence: dict[str, dict[str, object]] = {}
        for lane in ("fla_reference", "native_candidate"):
            stem = f"decode_correctness_{tag}_b{batch}_{lane}"
            row_path = root / f"{stem}.jsonl"
            probe_path = root / f"{stem}.pt"
            if lane == "fla_reference":
                source_lane_path = None
                row = one_row(row_path)
                benchmark_matrix = "rwkv_native_graph_fla_correctness_v1"
                optimization_lane = "fla_reference"
                warmup, runs = 1, 1
            else:
                source_lane_path = lane_path_by_key[(tag, batch)]
                row = production_row(source_lane_path, batch)
                benchmark_matrix = "qwen35_paired_decode_v1"
                optimization_lane = "best_optimized_hf"
                warmup, runs = 3, 7
                with row_path.open("x", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n")
                if one_row(row_path) != row:
                    raise SystemExit(
                        f"{row_path}: extracted native row differs from its production source"
                    )
            expected = {
                "benchmark_matrix": benchmark_matrix,
                "optimization_lane": optimization_lane,
                "benchmark_repository_commit": commit,
                "model_pair": pair,
                "model_size_label": size,
                "model_id_or_path": str(model_path),
                "model_role": "candidate",
                "model_kind": "rwkv",
                "dtype": "fp16",
                "quantization": "none",
                "batch_size": batch,
                "prompt_tokens": 2048,
                "decode_tokens": 512,
                "prefill_chunk_size": 512,
                "warmup": warmup,
                "runs": runs,
                "resident_sweep": True,
                "status": "pass",
                "logits_finite": True,
                "probe_tokens": 512,
                "probe_batch_size": batch,
                "probe_distinct_batch_prompts": batch > 1,
                "probe_decode_logits_all_finite": True,
                "resident_cell_index": 1 if lane == "fla_reference" else 6,
                "resident_cells_total": 1 if lane == "fla_reference" else 6,
                "resident_probe_cell": [batch, 2048, 512],
                "resident_probe_cell_selected": True,
            }
            for field, wanted in expected.items():
                require(row, field, wanted, row_path)
            actual_probe_output = row.get("probe_output")
            if not isinstance(actual_probe_output, str) or Path(actual_probe_output).resolve() != probe_path.resolve():
                raise SystemExit(
                    f"{row_path}: probe_output={actual_probe_output!r} is not bound to {probe_path}"
                )
            if lane == "fla_reference":
                route = {
                    "rwkv_implementation_requested": "wrapper_repo",
                    "rwkv_implementation_effective": "wrapper_repo",
                    "rwkv_fast_token_backend_requested": "fla",
                    "rwkv_native_model_backend_requested": "eager",
                    "rwkv_fast_prefill_requested": "0",
                    "rwkv_prefill_graph_requested": "0",
                    "effective_backend": "fla",
                    "step_backend": "rwkv_fast_token",
                    "prefill_backend_effective": None,
                }
            else:
                route = {
                    "rwkv_implementation_requested": "auto",
                    "rwkv_implementation_effective": "native_model",
                    "rwkv_fast_token_backend_requested": "native_graph",
                    "rwkv_native_model_backend_requested": "native_graph",
                    "rwkv_prefill_graph_requested": None,
                    "effective_backend": "native_graph",
                    "step_backend": "rwkv_fast_token",
                }
                promoted = batch == 8 and size in {"0.4b", "1.5b"}
                for route_name in ("sm120_wagv_bmm_g", "sm120_compiled_ffn"):
                    require(
                        row,
                        f"rwkv_native_graph_{route_name}_full_model_effective",
                        promoted,
                        row_path,
                    )
                require(
                    row,
                    "rwkv_native_graph_ada_wagv_bmm_full_model_effective",
                    promoted,
                    row_path,
                )
            for field, wanted in route.items():
                require(row, field, wanted, row_path)
            probe = load_probe(
                probe_path,
                pair=pair,
                size=size,
                model_path=model_path,
                batch=batch,
            )
            lane_rows[lane] = row
            lane_probes[lane] = probe
            lane_evidence[lane] = {
                "row": evidence(row_path),
                "probe": evidence(probe_path),
            }
            if source_lane_path is not None:
                lane_evidence[lane].update(
                    {
                        "source_lane": evidence(source_lane_path),
                        "source_cell": {
                            "batch_size": batch,
                            "prompt_tokens": 2048,
                            "decode_tokens": 512,
                        },
                    }
                )

        comparison = compare(
            lane_probes["fla_reference"],
            lane_probes["native_candidate"],
            0.9999,
        )
        contract_errors = []
        if comparison.get("probe_batch_size") != batch:
            contract_errors.append("probe_batch_size mismatch")
        if comparison.get("probe_tokens") != 512:
            contract_errors.append("probe_tokens mismatch")
        if batch == 8 and comparison.get("distinct_batch_prompts") is not True:
            contract_errors.append("B8 prompts are not distinct")
        comparison["contract_errors"] = contract_errors
        if contract_errors:
            comparison["status"] = "fail"
        if comparison.get("status") != "pass":
            raise SystemExit(
                f"{pair} B{batch}: native_graph vs FLA correctness failed: {comparison}"
            )
        for field in ("prompt_logits_cosine", "final_logits_cosine"):
            value = comparison.get(field)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0.9999:
                raise SystemExit(f"{pair} B{batch}: {field}={value!r}, expected >= 0.9999")
        comparison_path = root / f"decode_correctness_{tag}_b{batch}_compare.json"
        with comparison_path.open("x", encoding="utf-8") as handle:
            json.dump(comparison, handle, indent=2)
            handle.write("\n")
        entries.append(
            {
                "model_pair": pair,
                "model_size_label": size,
                "model_path": str(model_path),
                "batch_size": batch,
                "prompt_tokens": 2048,
                "decode_tokens": 512,
                "probe_tokens": 512,
                **lane_evidence,
                "comparison": evidence(comparison_path),
            }
        )

document = {
    "schema_version": 1,
    "protocol": "rwkv_native_graph_fla_correctness_v1",
    "benchmark_repository_commit": commit,
    "model_hashes_sha256": hashlib.sha256(model_hashes.read_bytes()).hexdigest(),
    "runtime": evidence(runtime_lock),
    "coverage": {
        "models": 4,
        "batch_sizes": [1, 8],
        "entries": 8,
        "baseline_fresh_gpu_processes": 8,
        "candidate_additional_gpu_processes": 0,
        "candidate_formal_lane_processes": 8,
        "prompt_tokens": 2048,
        "decode_tokens": 512,
        "probe_tokens": 512,
    },
    "reference_contract": {
        "rwkv_implementation": "wrapper_repo",
        "RWKV7_FAST_TOKEN_BACKEND": "fla",
        "RWKV7_NATIVE_MODEL_BACKEND": "eager",
        "RWKV7_FAST_PREFILL": 0,
        "RWKV7_NATIVE_PREFILL_GRAPH": 0,
    },
    "candidate_contract": {
        "rwkv_implementation": "auto",
        "RWKV7_FAST_TOKEN_BACKEND": "native_graph",
        "RWKV7_NATIVE_MODEL_BACKEND": "native_graph",
        "RWKV7_FAST_PREFILL": "unset_exact_card_policy",
        "RWKV7_NATIVE_PREFILL_GRAPH": "unset_exact_card_policy",
        "small_model_b8_promoted_bundle": True,
    },
    "gates": {
        "greedy_tokens": "exact_all_512",
        "prompt_logits_min_row_cosine": 0.9999,
        "final_logits_min_row_cosine": 0.9999,
        "decode_logits_all_finite": True,
        "b8_distinct_prompts": True,
    },
    "entries": entries,
}
with output.open("x", encoding="utf-8") as handle:
    json.dump(document, handle, indent=2)
    handle.write("\n")
PY
}

# Stable production order: model size, B1/B8, prompt, decode. Every call is a
# separate Python process even though each six-cell lane keeps one model load.
run_lane "${RWKV_04_MODEL}" rwkv-0.4b__qwen3.5-0.8b 0.4b 1 0 \
  "${lane_results[0]}" "${OUT_DIR}/logs/rwkv_0p4_b1.log" \
  "${OUT_DIR}/decode_correctness_0p4_b1_native_candidate.pt"
run_lane "${RWKV_04_MODEL}" rwkv-0.4b__qwen3.5-0.8b 0.4b 8 1 \
  "${lane_results[1]}" "${OUT_DIR}/logs/rwkv_0p4_b8.log" \
  "${OUT_DIR}/decode_correctness_0p4_b8_native_candidate.pt"
run_lane "${RWKV_15_MODEL}" rwkv-1.5b__qwen3.5-2b 1.5b 1 0 \
  "${lane_results[2]}" "${OUT_DIR}/logs/rwkv_1p5_b1.log" \
  "${OUT_DIR}/decode_correctness_1p5_b1_native_candidate.pt"
run_lane "${RWKV_15_MODEL}" rwkv-1.5b__qwen3.5-2b 1.5b 8 1 \
  "${lane_results[3]}" "${OUT_DIR}/logs/rwkv_1p5_b8.log" \
  "${OUT_DIR}/decode_correctness_1p5_b8_native_candidate.pt"
run_lane "${RWKV_29_MODEL}" rwkv-2.9b__qwen3.5-4b 2.9b 1 0 \
  "${lane_results[4]}" "${OUT_DIR}/logs/rwkv_2p9_b1.log" \
  "${OUT_DIR}/decode_correctness_2p9_b1_native_candidate.pt"
run_lane "${RWKV_29_MODEL}" rwkv-2.9b__qwen3.5-4b 2.9b 8 0 \
  "${lane_results[5]}" "${OUT_DIR}/logs/rwkv_2p9_b8.log" \
  "${OUT_DIR}/decode_correctness_2p9_b8_native_candidate.pt"
run_lane "${RWKV_72_MODEL}" rwkv-7.2b__qwen3.5-9b 7.2b 1 0 \
  "${lane_results[6]}" "${OUT_DIR}/logs/rwkv_7p2_b1.log" \
  "${OUT_DIR}/decode_correctness_7p2_b1_native_candidate.pt"
run_lane "${RWKV_72_MODEL}" rwkv-7.2b__qwen3.5-9b 7.2b 8 0 \
  "${lane_results[7]}" "${OUT_DIR}/logs/rwkv_7p2_b8.log" \
  "${OUT_DIR}/decode_correctness_7p2_b8_native_candidate.pt"

# Bind each embedded native probe to its exact production row, compare it to
# the fresh FLA oracle, and hash every referenced artifact before final merge.
assemble_decode_correctness_manifest

# Detect any source mutation while the eight fresh processes were running.
# Such a run cannot truthfully attribute every lane to one immutable commit.
validate_repository_provenance

# Re-hash every model input after all GPU work; checkpoint/tokenizer mutation
# during the run invalidates the complete evidence bundle.
hash_models "${model_hashes_after}"
cmp --silent "${model_hashes}" "${model_hashes_after}" || {
  echo "one or more complete model directories changed while formal evidence was running" >&2
  exit 2
}

# Assemble only after all eight fail-fast processes succeed. This structural
# check intentionally does not consume Qwen; the paired validator remains the
# authority for effective routes, runtime equality, and the strict speed gate.
"${PYTHON_BIN}" - "${candidate_result}" "${candidate_sha256}" \
  "${route_manifest}" "${REPOSITORY_COMMIT}" \
  "${model_hashes}" "${model_hashes_after}" "${sm120_ab_manifest}" \
  "${decode_correctness_manifest}" "${runtime_lock}" "${pip_freeze}" \
  "${system_csv}" "${ROOT}" \
  "${RWKV_04_MODEL}" "${RWKV_15_MODEL}" "${RWKV_29_MODEL}" "${RWKV_72_MODEL}" \
  "${lane_results[@]}" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

output = Path(sys.argv[1])
checksum_output = Path(sys.argv[2])
manifest_output = Path(sys.argv[3])
commit = sys.argv[4]
model_hashes = Path(sys.argv[5])
model_hashes_after = Path(sys.argv[6])
sm120_ab_manifest = Path(sys.argv[7])
decode_correctness_manifest = Path(sys.argv[8])
runtime_lock = Path(sys.argv[9])
pip_freeze = Path(sys.argv[10])
system_csv = Path(sys.argv[11])
repository_root = Path(sys.argv[12])
model_paths = [Path(value) for value in sys.argv[13:17]]
inputs = [Path(value) for value in sys.argv[17:]]
pairs = (
    ("rwkv-0.4b__qwen3.5-0.8b", "0.4b"),
    ("rwkv-1.5b__qwen3.5-2b", "1.5b"),
    ("rwkv-2.9b__qwen3.5-4b", "2.9b"),
    ("rwkv-7.2b__qwen3.5-9b", "7.2b"),
)
expected_lanes = [
    (pair, size, model_path, batch)
    for (pair, size), model_path in zip(pairs, model_paths, strict=True)
    for batch in (1, 8)
]
expected_cells = [
    (prompt, decode)
    for prompt in (128, 512, 2048)
    for decode in (128, 512)
]
if len(inputs) != len(expected_lanes):
    raise SystemExit(f"expected 8 lane files, got {len(inputs)}")

encoded_lines: list[bytes] = []
manifest_lanes: list[dict[str, object]] = []
for path, (pair, size, model_path, batch) in zip(inputs, expected_lanes, strict=True):
    lines = [line for line in path.read_bytes().splitlines() if line.strip()]
    if len(lines) != 6:
        raise SystemExit(f"{path}: expected 6 rows, got {len(lines)}")
    actual_cells: list[tuple[int, int]] = []
    for line_number, line in enumerate(lines, 1):
        row = json.loads(line)
        expected = {
            "benchmark_matrix": "qwen35_paired_decode_v1",
            "optimization_lane": "best_optimized_hf",
            "benchmark_repository_commit": commit,
            "model_pair": pair,
            "model_size_label": size,
            "model_id_or_path": str(model_path),
            "model_role": "candidate",
            "model_kind": "rwkv",
            "batch_size": batch,
            "status": "pass",
        }
        mismatches = [
            f"{field}={row.get(field)!r}, expected {wanted!r}"
            for field, wanted in expected.items()
            if row.get(field) != wanted
        ]
        if mismatches:
            raise SystemExit(
                f"{path}:{line_number}: " + "; ".join(mismatches)
            )
        actual_cells.append((int(row["prompt_tokens"]), int(row["decode_tokens"])))
        encoded_lines.append(line + b"\n")
    if actual_cells != expected_cells:
        raise SystemExit(
            f"{path}: cells={actual_cells!r}, expected stable order {expected_cells!r}"
        )
    promoted = batch == 8 and size in {"0.4b", "1.5b"}
    manifest_lanes.append(
        {
            "model_pair": pair,
            "model_size_label": size,
            "model_path": str(model_path),
            "batch_size": batch,
            "cells": 6,
            "fresh_process": True,
            "rwkv_implementation_requested": "auto",
            "rwkv_implementation_effective": "native_model",
            "RWKV7_NATIVE_PREFILL_GRAPH": "exact_card_policy",
            "RWKV7_NATIVE_GRAPH_ADA_WAGV_BMM": int(promoted),
            "RWKV7_NATIVE_GRAPH_SM120_WAGV_BMM_G": int(promoted),
            "RWKV7_NATIVE_GRAPH_SM120_COMPILED_FFN": int(promoted),
            "RWKV7_NATIVE_GRAPH_RKV_POLICY": "vkwr_auto" if promoted else None,
            "RWKV7_BLACKWELL_TORCH_COMPILE": 1 if promoted else None,
            "compile_cache": "fresh_unique_directory" if promoted else None,
        }
    )

payload = b"".join(encoded_lines)
if len(encoded_lines) != 48:
    raise SystemExit(f"candidate matrix expected 48 rows, got {len(encoded_lines)}")
with output.open("xb") as handle:
    handle.write(payload)
digest = hashlib.sha256(payload).hexdigest()

def artifact(path: Path) -> dict[str, str]:
    payload = path.read_bytes()
    return {
        "path": str(path.resolve(strict=True)),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }

before_hashes = model_hashes.read_bytes()
after_hashes = model_hashes_after.read_bytes()
if before_hashes != after_hashes:
    raise SystemExit("model hash snapshots differ during final assembly")
with checksum_output.open("x", encoding="utf-8") as handle:
    handle.write(f"{digest}  {output.name}\n")
with manifest_output.open("x", encoding="utf-8") as handle:
    json.dump(
        {
            "schema_version": 1,
            "protocol": "qwen35_paired_decode_v1",
            "benchmark_repository_commit": commit,
            "repository_root": str(repository_root),
            "repository_clean_pre_and_post": True,
            "candidate_rows": 48,
            "candidate_result": artifact(output),
            "candidate_sha256_sidecar": artifact(checksum_output),
            "qwen_rerun": False,
            "rwkv_implementation_requested": "auto",
            "rwkv_implementation_effective": "native_model",
            "forced_environment": {
                "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
                "CUDA_DEVICE_ORDER": os.environ["CUDA_DEVICE_ORDER"],
                "PYTHONPATH": os.environ["PYTHONPATH"],
                "PYTORCH_CUDA_ALLOC_CONF": os.environ["PYTORCH_CUDA_ALLOC_CONF"],
                "TORCH_CUDA_ARCH_LIST": os.environ["TORCH_CUDA_ARCH_LIST"],
                "TORCHDYNAMO_DISABLE": os.environ["TORCHDYNAMO_DISABLE"],
                "TORCH_COMPILE_DISABLE": os.environ["TORCH_COMPILE_DISABLE"],
                "HF_HUB_OFFLINE": os.environ["HF_HUB_OFFLINE"],
                "TRANSFORMERS_OFFLINE": os.environ["TRANSFORMERS_OFFLINE"],
                "TOKENIZERS_PARALLELISM": os.environ["TOKENIZERS_PARALLELISM"],
                "RWKV7_FAST_TOKEN_BACKEND": "native_graph",
                "RWKV7_NATIVE_MODEL_BACKEND": "native_graph",
                "RWKV7_NATIVE_PREFILL_GRAPH": "unset_exact_card_policy",
            },
            "model_hash_contract": {
                "algorithm": "sha256",
                "scope": "every recursive regular file",
                "before": artifact(model_hashes),
                "after": artifact(model_hashes_after),
                "byte_identical": True,
            },
            "sm120_b8_ab_manifest": artifact(sm120_ab_manifest),
            "native_graph_fla_correctness_manifest": artifact(
                decode_correctness_manifest
            ),
            "runtime_lock": artifact(runtime_lock),
            "pip_freeze": artifact(pip_freeze),
            "system_identity": artifact(system_csv),
            "lanes": manifest_lanes,
        },
        handle,
        indent=2,
    )
    handle.write("\n")
print(f"wrote {output} ({digest})")
PY

echo "candidate-only matrix complete: ${candidate_result}"
