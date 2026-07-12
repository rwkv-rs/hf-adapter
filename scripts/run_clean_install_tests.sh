#!/usr/bin/env bash
set -euo pipefail

# Build an isolated environment from this checkout, verify that the installed
# wheel imports outside the source tree, and then run a reproducible test
# profile. No pre-existing site packages or inherited PYTHONPATH entries are
# used; legacy script-style tests receive only the explicit checkout root.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-smoke}"
CPU_ONLY="${RWKV7_CPU_ONLY:-auto}"
KEEP_VENV="${RWKV7_KEEP_TEST_VENV:-0}"
REQUIRE_APPLE="${RWKV7_REQUIRE_APPLE:-0}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  for candidate in python3.11 python3.12 python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 \
      && "$candidate" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi
if [[ -z "${PYTHON_BIN:-}" ]]; then
  echo "Python >=3.10 is required (set PYTHON_BIN to an explicit interpreter)" >&2
  exit 2
fi

case "$PROFILE" in
  smoke|full|apple) ;;
  *)
    echo "usage: $0 [smoke|full|apple]" >&2
    exit 2
    ;;
esac

if [[ -n "${RWKV7_TEST_VENV:-}" ]]; then
  VENV="$RWKV7_TEST_VENV"
  rm -rf "$VENV"
  mkdir -p "$(dirname "$VENV")"
  OWN_VENV=0
else
  VENV="$(mktemp -d "${TMPDIR:-/tmp}/rwkv7-clean-test.XXXXXX")"
  OWN_VENV=1
fi

HAD_BUILD=0
[[ -e "$ROOT/build" ]] && HAD_BUILD=1

cleanup() {
  if [[ "$HAD_BUILD" == 0 ]]; then
    rm -rf "$ROOT/build"
  fi
  if [[ "$OWN_VENV" == 1 && "$KEEP_VENV" != 1 ]]; then
    rm -rf "$VENV"
  elif [[ "$KEEP_VENV" == 1 ]]; then
    echo "kept test environment: $VENV"
  fi
}
trap cleanup EXIT

"$PYTHON_BIN" -m venv "$VENV"
if [[ -x "$VENV/bin/python" ]]; then
  PY="$VENV/bin/python"
else
  PY="$VENV/Scripts/python.exe"
fi

"$PY" -m pip install --upgrade pip setuptools wheel

if [[ "$CPU_ONLY" == auto ]]; then
  if [[ "$PROFILE" != apple && "$(uname -s)" == Linux ]]; then
    CPU_ONLY=1
  else
    CPU_ONLY=0
  fi
fi
if [[ "$CPU_ONLY" == 1 && "$(uname -s)" == Linux ]]; then
  "$PY" -m pip install torch \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple
fi

# This is intentionally non-editable: it exercises PEP 517 metadata, wheel
# contents, dependency resolution, and importability as a user would install it.
"$PY" -m pip install "${ROOT}[test]"
"$PY" -m pip check

(
  cd "$VENV"
  ROOT_FOR_PYTHON="$ROOT" "$PY" - <<'PY'
import importlib.metadata
import os
from pathlib import Path

import rwkv7_hf

root = Path(os.environ["ROOT_FOR_PYTHON"]).resolve()
module_path = Path(rwkv7_hf.__file__).resolve()
try:
    module_path.relative_to(root)
except ValueError:
    pass
else:
    raise SystemExit(f"clean-install import leaked into source tree: {module_path}")

version = importlib.metadata.version("rwkv7-hf-adapter")
print(f"installed rwkv7-hf-adapter={version} from {module_path}")
PY
)

export PYTHONNOUSERSITE=1
export PYTHONPATH=""
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export DS_IGNORE_CUDA_DETECTION="${DS_IGNORE_CUDA_DETECTION:-1}"
if [[ "$PROFILE" != apple ]]; then
  export CUDA_VISIBLE_DEVICES=""
fi

cd "$ROOT"
"$PY" -m compileall -q rwkv7_hf bench scripts tests
while IFS= read -r -d '' script; do
  bash -n "$script"
done < <(find scripts -maxdepth 1 -name '*.sh' -print0)

# Collection is a first-class gate: optional backends must never fail import
# just because CUDA, MLX, CoreML, DeepSpeed, a model directory, or a driver is
# absent. The full run below then reports runtime skips with `-ra`.
"$PY" -m pytest --collect-only -q

if [[ "$PROFILE" == smoke ]]; then
  "$PY" -m pytest -q -ra \
    tests/test_clean_install_packaging.py \
    tests/test_acceptance_scripts.py \
    tests/test_apple_silicon_packaging.py \
    tests/test_backend_boundaries.py \
    tests/test_kernel_policy.py \
    tests/test_tokenizer_fast_trie.py

  # Preserve the historical executable-script PR coverage. Several of these
  # files expose a main() smoke rather than pytest-collected test functions.
  for test_script in \
    tests/test_convert_config.py \
    tests/test_batch_convert_manifest.py \
    tests/test_deepspeed_configs.py \
    tests/test_sync_hf_adapter_code.py \
    tests/test_larger_model_results.py \
    tests/test_result_tools.py \
    tests/test_markdown_links.py \
    tests/test_native_fla_free_import.py \
    tests/test_native_model_training_unit.py \
    tests/test_native_model_generate_unit.py \
    tests/test_bnb_skip_policy.py \
    tests/test_native_quant_mm8_cpu.py \
    tests/test_train_spec_draft_unit.py; do
    PYTHONPATH="$ROOT" "$PY" "$test_script"
  done
  exit 0
fi

if [[ "$PROFILE" == apple ]]; then
  APPLE_REASON="$("$PY" - <<'PY'
import platform
import sys

if platform.system() != "Darwin" or platform.machine() != "arm64":
    print(f"requires Darwin arm64, got {platform.system()} {platform.machine()}")
    sys.exit(0)

import torch
if not torch.backends.mps.is_available():
    print("torch MPS backend is unavailable")
    sys.exit(0)

try:
    import mlx.core  # noqa: F401
except Exception as exc:
    print(f"MLX import failed: {exc!r}")
PY
)"
  if [[ -n "$APPLE_REASON" ]]; then
    if [[ "$REQUIRE_APPLE" == 1 ]]; then
      echo "APPLE HARDWARE REQUIREMENT FAILED: $APPLE_REASON" >&2
      exit 1
    fi
    echo "SKIPPED Apple hardware profile: $APPLE_REASON"
    exit 0
  fi
fi

"$PY" -m pytest -q -ra

if [[ "$PROFILE" == full ]]; then
  echo "SKIPPED Apple executable hardware profile: run '$0 apple' on Darwin arm64"
  if [[ -z "${RWKV7_TEST_MODEL:-}" ]]; then
    echo "SKIPPED model-backed acceptance: RWKV7_TEST_MODEL is unset"
  fi
fi

if [[ "$PROFILE" == apple ]]; then
  # These are executable acceptance smokes rather than pytest test functions.
  # They use tiny random models and do not need downloaded checkpoints.
  "$PY" tests/test_apple_silicon_smoke.py
  "$PY" tests/test_apple_silicon_training_smoke.py
  "$PY" tests/test_apple_silicon_trainer_smoke.py
  "$PY" tests/test_apple_silicon_quant_smoke.py
  "$PY" tests/test_apple_silicon_mlx_smoke.py
  "$PY" tests/test_apple_silicon_mlx_model_smoke.py

  if [[ -z "${RWKV7_TEST_MODEL:-}" ]]; then
    echo "SKIPPED model-backed Apple acceptance: RWKV7_TEST_MODEL is unset"
  else
    "$PY" tests/test_apple_silicon_model_sweep.py --model "$RWKV7_TEST_MODEL"
    "$PY" tests/test_apple_silicon_model_training_smoke.py --model "$RWKV7_TEST_MODEL"
  fi
fi
