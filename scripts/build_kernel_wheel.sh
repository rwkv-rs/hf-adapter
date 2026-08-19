#!/usr/bin/env bash
# Build and structurally validate one exact-runtime RWKV-7 CUDA kernel wheel.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
ARCH_LIST="${RWKV7_KERNEL_ARCH_LIST:-}"
OUT_DIR="${OUT_DIR:-${ROOT}/kernel_wheel/dist}"

if [[ -z "${ARCH_LIST}" ]]; then
  echo "RWKV7_KERNEL_ARCH_LIST is required (for example 7.0 or 8.9)" >&2
  exit 2
fi
if [[ "$(uname -s)" != "Linux" ]]; then
  echo "binary CUDA kernel wheels must be built on Linux" >&2
  exit 2
fi
for command in nvcc ninja; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "${command} is required" >&2
    exit 2
  fi
done

"${PYTHON_BIN}" - <<'PY'
import torch
if not torch.version.cuda:
    raise SystemExit("CUDA-enabled PyTorch is required")
print(f"torch={torch.__version__} cuda={torch.version.cuda} cxx11_abi={torch._C._GLIBCXX_USE_CXX11_ABI}")
PY
nvcc --version

mkdir -p "${OUT_DIR}"
export RWKV7_KERNEL_ARCH_LIST="${ARCH_LIST}"
if "${PYTHON_BIN}" -c 'import build.__main__' >/dev/null 2>&1; then
  "${PYTHON_BIN}" -m build \
    --wheel \
    --no-isolation \
    --outdir "${OUT_DIR}" \
    "${ROOT}/kernel_wheel"
else
  # Offline hardware labs commonly retain setuptools/wheel but not the small
  # PEP 517 frontend. The produced bdist_wheel is identical in content and is
  # still subjected to the manifest/binary closure inspection below.
  (
    cd "${ROOT}/kernel_wheel"
    "${PYTHON_BIN}" setup.py bdist_wheel --dist-dir "${OUT_DIR}"
  )
fi

mapfile -t wheels < <(find "${OUT_DIR}" -maxdepth 1 -type f -name '*.whl' -print | sort)
if [[ "${#wheels[@]}" -eq 0 ]]; then
  echo "no wheel was produced in ${OUT_DIR}" >&2
  exit 1
fi
"${PYTHON_BIN}" "${ROOT}/scripts/inspect_kernel_wheel.py" "${wheels[@]}"
