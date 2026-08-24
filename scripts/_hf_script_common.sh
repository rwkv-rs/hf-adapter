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
# Windows (MSYS / Git-Bash / Cygwin) Python uses ';' as the PYTHONPATH
# separator, not ':'. With ':' the entries get misparsed (the ':' in drive
# paths like D:/...) and `from bench.xxx` / `import rwkv7_hf` fail with
# ModuleNotFoundError -- which broke run_hf_acceptance.sh on RTX 5070/Windows.
if [[ "${OSTYPE:-}" == msys* || "${OSTYPE:-}" == cygwin* || -n "${MSYSTEM:-}" ]]; then
    export PYTHONPATH="${RWKV7_REPO_ROOT};${PYTHONPATH:-}"
else
    export PYTHONPATH="${RWKV7_REPO_ROOT}:${PYTHONPATH:-}"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
RESULTS="${RESULTS:-bench/_runs/results.jsonl}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-fp16}"
TRAIN_DTYPE="${TRAIN_DTYPE:-bf16}"
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
from importlib import metadata

print(f"python={platform.python_version()} executable={sys.executable}")
print(f"platform={platform.platform()}")
for name in ["torch", "torch_npu", "torch_musa", "transformers", "peft", "trl", "deepspeed", "bitsandbytes", "fla", "mlx"]:
    if importlib.util.find_spec(name) is None:
        print(f"{name}=missing")
        continue
    try:
        mod = __import__(name)
        version = getattr(mod, "__version__", None)
        if version is None:
            try:
                version = metadata.version(name)
            except metadata.PackageNotFoundError:
                version = "unknown"
        print(f"{name}={version}")
    except Exception as exc:
        print(f"{name}=import-error:{type(exc).__name__}:{exc}")
try:
    import torch
    try:
        from rwkv7_hf.ascend_runtime import import_torch_npu
        import_torch_npu(required=False)
    except Exception:
        pass
    npu = getattr(torch, "npu", None)
    npu_available = False
    if npu is not None:
        npu_is_available = getattr(npu, "is_available", None)
        npu_available = bool(callable(npu_is_available) and npu_is_available())
    print(f"torch_npu_available={npu_available}")
    if npu_available:
        npu_count = int(npu.device_count())
        print(f"torch_npu_device_count={npu_count}")
        for idx in range(npu_count):
            print(f"npu_device_{idx}={npu.get_device_name(idx)}")
    musa = getattr(torch, "musa", None)
    musa_available = False
    if musa is not None:
        musa_is_available = getattr(musa, "is_available", None)
        musa_available = bool(callable(musa_is_available) and musa_is_available())
    print(f"torch_musa_available={musa_available}")
    if musa_available:
        musa_count = int(musa.device_count())
        print(f"torch_musa_device_count={musa_count}")
        for idx in range(musa_count):
            props = musa.get_device_properties(idx)
            name = getattr(props, "name", None) or musa.get_device_name(idx)
            architecture = getattr(props, "architecture", None)
            suffix = f" arch={architecture}" if architecture is not None else ""
            print(f"musa_device_{idx}={name}{suffix}")
    try:
        from rwkv7_hf.biren_runtime import import_torch_br
        import_torch_br(required=False)
    except Exception:
        pass
    supa = getattr(torch, "supa", None)
    supa_available = False
    if supa is not None:
        supa_is_available = getattr(supa, "is_available", None)
        supa_available = bool(callable(supa_is_available) and supa_is_available())
    print(f"torch_supa_available={supa_available}")
    if supa_available:
        supa_count = int(supa.device_count())
        print(f"torch_supa_device_count={supa_count}")
        for idx in range(supa_count):
            print(f"supa_device_{idx}={supa.get_device_name(idx)}")
    print(f"torch_cuda_available={torch.cuda.is_available()}")
    print(f"torch_cuda_device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    metax_available = False
    if torch.cuda.is_available():
        try:
            from rwkv7_hf.metax_runtime import is_metax_c500_name
        except Exception:
            is_metax_c500_name = lambda _name: False
        for idx in range(torch.cuda.device_count()):
            cap = torch.cuda.get_device_capability(idx)
            name = torch.cuda.get_device_name(idx)
            metax_available = metax_available or bool(is_metax_c500_name(name))
            print(f"cuda_device_{idx}={name} sm_{cap[0]}{cap[1]}")
    print(f"metax_c500_available={metax_available}")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None:
        print(f"torch_mps_built={mps.is_built()}")
        print(f"torch_mps_available={mps.is_available()}")
except Exception as exc:
    print(f"torch_device_probe_error={type(exc).__name__}:{exc}")
for key in [
    "CUDA_VISIBLE_DEVICES",
    "ASCEND_RT_VISIBLE_DEVICES",
    "ASCEND_TOOLKIT_VERSION",
    "BIRENSUPA_SDK_VERSION",
    "BIREN_DRIVER_VERSION",
    "BIREN_SUPA_VERSION",
    "ACCELERATE_TORCH_DEVICE",
    "RWKV7_ALLOW_UNVALIDATED_BIREN",
    "MACA_PATH",
    "CUCC_PATH",
    "MXMACA_VERSION",
    "RWKV7_ALLOW_UNVALIDATED_METAX",
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
