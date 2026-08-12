#!/usr/bin/env bash
# Build the exact official FLA/causal-conv sources for all comparison cards.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
FLA_SOURCE="${FLA_SOURCE:-}"
CAUSAL_CONV1D_SOURCE="${CAUSAL_CONV1D_SOURCE:-}"
OUT_DIR="${OUT_DIR:-}"
FLA_SOURCE_COMMIT="${FLA_SOURCE_COMMIT:-2e38c1fab332174d056928feaf29f8c5fd5ac550}"
CAUSAL_CONV1D_SOURCE_COMMIT="${CAUSAL_CONV1D_SOURCE_COMMIT:-82867a9d2e6907cc0f637ac6aff318f696838548}"
TORCH_CUDA_ARCH_LIST="8.6;8.9;12.0"

if [[ -z "${FLA_SOURCE}" || -z "${CAUSAL_CONV1D_SOURCE}" || -z "${OUT_DIR}" ]]; then
  echo "FLA_SOURCE, CAUSAL_CONV1D_SOURCE and OUT_DIR are required" >&2
  exit 2
fi
for source in "${FLA_SOURCE}" "${CAUSAL_CONV1D_SOURCE}"; do
  if [[ ! -d "${source}/.git" ]]; then
    echo "${source} must be a git checkout" >&2
    exit 2
  fi
  if [[ -n "$(git -C "${source}" status --porcelain)" ]]; then
    echo "${source} must be a clean source checkout" >&2
    exit 2
  fi
done

actual_fla="$(git -C "${FLA_SOURCE}" rev-parse HEAD)"
actual_causal="$(git -C "${CAUSAL_CONV1D_SOURCE}" rev-parse HEAD)"
if [[ "${actual_fla}" != "${FLA_SOURCE_COMMIT}" ]]; then
  echo "FLA source ${actual_fla} != required ${FLA_SOURCE_COMMIT}" >&2
  exit 2
fi
if [[ "${actual_causal}" != "${CAUSAL_CONV1D_SOURCE_COMMIT}" ]]; then
  echo "causal-conv source ${actual_causal} != required ${CAUSAL_CONV1D_SOURCE_COMMIT}" >&2
  exit 2
fi
if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc is required; use the same CUDA developer image on all cards" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"
export TORCH_CUDA_ARCH_LIST
export CAUSAL_CONV1D_FORCE_BUILD=TRUE

"${PYTHON_BIN}" - <<'PY'
import platform
if platform.python_version_tuple()[:2] != ("3", "10"):
    raise SystemExit(f"Python 3.10 required, got {platform.python_version()}")
PY

nvcc --version > "${OUT_DIR}/nvcc.txt"
"${PYTHON_BIN}" -m pip install --no-build-isolation --no-deps --force-reinstall \
  "${CAUSAL_CONV1D_SOURCE}"
"${PYTHON_BIN}" -m pip install --no-build-isolation --no-deps --force-reinstall \
  "${FLA_SOURCE}"

"${PYTHON_BIN}" - "${OUT_DIR}/extension_build_manifest.json" \
  "${actual_fla}" "${actual_causal}" <<'PY'
import json
import os
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import torch

target, fla_commit, causal_commit = sys.argv[1:]
manifest = {
    "schema_version": 1,
    "protocol": "hf_fast_path_v1",
    "python_version": platform.python_version(),
    "torch_version": str(torch.__version__),
    "torch_cuda_version": str(torch.version.cuda),
    "torch_cuda_arch_list": os.environ["TORCH_CUDA_ARCH_LIST"],
    "fla_version": version("flash-linear-attention"),
    "fla_source_commit": fla_commit,
    "causal_conv1d_version": version("causal-conv1d"),
    "causal_conv1d_source_commit": causal_commit,
}
Path(target).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps(manifest))
PY
