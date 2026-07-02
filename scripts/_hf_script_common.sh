#!/usr/bin/env bash
# Shared helpers for RWKV-7 HF adapter validation scripts.

set -euo pipefail

RWKV7_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RWKV7_REPO_ROOT="$(cd "${RWKV7_SCRIPT_DIR}/.." && pwd)"
cd "${RWKV7_REPO_ROOT}"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export RWKV_V7_ON="${RWKV_V7_ON:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export DS_IGNORE_CUDA_DETECTION="${DS_IGNORE_CUDA_DETECTION:-1}"
export PYTHONPATH="${RWKV7_REPO_ROOT}:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS="${RESULTS:-bench/results.jsonl}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-fp16}"
TRAIN_DTYPE="${TRAIN_DTYPE:-fp32}"
ATTN_MODE="${ATTN_MODE:-fused_recurrent}"
FUSE_NORM="${FUSE_NORM:-auto}"
FAST_TOKEN_BACKEND="${FAST_TOKEN_BACKEND:-auto}"
FAST_CACHE="${FAST_CACHE:-auto}"

rwkv7_log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

rwkv7_run() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  "$@"
}

rwkv7_require_model() {
  local model_path="$1"
  if [[ -z "${model_path}" ]]; then
    echo "MODEL is required. Pass it as the first argument or set MODEL=/path/to/hf-model." >&2
    exit 2
  fi
  if [[ ! -e "${model_path}" ]]; then
    echo "MODEL does not exist: ${model_path}" >&2
    exit 2
  fi
}

rwkv7_prepare_results() {
  if [[ -n "${RESULTS}" ]]; then
    mkdir -p "$(dirname "${RESULTS}")"
  fi
}

rwkv7_print_env() {
  rwkv7_log "environment"
  "${PYTHON_BIN}" - <<'PY'
import importlib.util
import os
import platform
import sys

print(f"python={platform.python_version()} executable={sys.executable}")
print(f"platform={platform.platform()}")
for name in ["torch", "transformers", "peft", "trl", "deepspeed", "bitsandbytes", "fla"]:
    if importlib.util.find_spec(name) is None:
        print(f"{name}=missing")
        continue
    try:
        mod = __import__(name)
        print(f"{name}={getattr(mod, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"{name}=import-error:{type(exc).__name__}:{exc}")
try:
    import torch
    print(f"torch_cuda_available={torch.cuda.is_available()}")
    print(f"torch_cuda_device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            cap = torch.cuda.get_device_capability(idx)
            print(f"cuda_device_{idx}={torch.cuda.get_device_name(idx)} sm_{cap[0]}{cap[1]}")
except Exception as exc:
    print(f"torch_cuda_probe_error={type(exc).__name__}:{exc}")
for key in [
    "CUDA_VISIBLE_DEVICES",
    "PYTHONNOUSERSITE",
    "RWKV_V7_ON",
    "TORCHDYNAMO_DISABLE",
    "DS_IGNORE_CUDA_DETECTION",
    "RWKV7_NATIVE_MODEL",
    "RWKV7_FAST_FORWARD",
    "RWKV7_FAST_TOKEN_BACKEND",
]:
    print(f"env_{key}={os.environ.get(key, '')}")
PY
}
