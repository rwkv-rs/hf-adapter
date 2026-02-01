#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/wzu/.private/rwkv7-hf-4080
RUN=$ROOT/native-jit-split-final-20260728
CAND=$RUN/candidate
PY=$ROOT/venv/bin/python
export CUDA_VISIBLE_DEVICES=0 CUDA_HOME=$ROOT/cuda-12.4 PYTHONPATH=$CAND HF_HOME=$ROOT/cache
cd "$CAND"
{
 echo "started_at=$(date -Iseconds)"
 RWKV7_NATIVE_MODEL=1 "$PY" tests/test_native_bnb_quant_smoke.py --model "$ROOT/models/rwkv7-g1d-0.4b-hf" --device cuda --dtype fp16 --quantization both --max-new-tokens 4
 rc=$?
 echo "finished_at=$(date -Iseconds)"
 echo "__EXIT__=$rc"
 exit "$rc"
} > "$RUN/results/candidate_bnb_smoke.log" 2>&1
