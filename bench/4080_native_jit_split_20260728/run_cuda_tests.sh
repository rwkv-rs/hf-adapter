#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/wzu/.private/rwkv7-hf-4080
RUN=$ROOT/native-jit-split-final-20260728
CAND=$RUN/candidate
PY=$ROOT/venv/bin/python
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=$ROOT/cuda-12.4
export PYTHONPATH=$CAND
export HF_HOME=$ROOT/cache
cd "$CAND"
{
  echo "started_at=$(date -Iseconds)"
  "$PY" - <<'PYENV'
import torch, transformers, platform
print('python='+platform.python_version())
print('torch='+torch.__version__)
print('transformers='+transformers.__version__)
print('cuda='+str(torch.version.cuda))
print('gpu='+torch.cuda.get_device_name(0))
PYENV
  "$PY" -m pytest -q tests/test_native_jit_*_split.py tests/test_native_quant_bnb8.py
  rc=$?
  echo "finished_at=$(date -Iseconds)"
  echo "__EXIT__=$rc"
  exit "$rc"
} > "$RUN/results/candidate_cuda_split_tests.log" 2>&1
