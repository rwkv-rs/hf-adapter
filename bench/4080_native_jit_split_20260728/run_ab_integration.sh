#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/wzu/.private/rwkv7-hf-4080
RUN=$ROOT/native-jit-split-final-20260728
PY=$ROOT/venv/bin/python
MODEL=$ROOT/models/rwkv7-g1g-1.5b-hf
export CUDA_VISIBLE_DEVICES=0 CUDA_HOME=$ROOT/cuda-12.4 HF_HOME=$ROOT/cache
export RWKV7_FAST_PREFILL=1
export TOKENIZERS_PARALLELISM=false
rm -f "$RUN/results/candidate_decode.jsonl" "$RUN/results/baseline_decode.jsonl" \
      "$RUN/results/candidate_prefill.jsonl" "$RUN/results/baseline_prefill.jsonl"
run_decode() {
  local role=$1 dir=$2 out=$3
  echo "=== decode $role $(date -Iseconds) ==="
  cd "$dir"
  PYTHONPATH="$dir" "$PY" bench/bench_native_model_decode.py \
    --hf-dir "$MODEL" --dtype fp16 --device cuda \
    --prompt-tokens 128 --decode-steps 128 --warmup 3 --repetitions 2 \
    --batch-sizes 1 8 --backends native_jit native_graph --fast-token-api \
    --results "$out"
}
run_prefill() {
  local role=$1 dir=$2 out=$3
  echo "=== prefill $role $(date -Iseconds) ==="
  cd "$dir"
  PYTHONPATH="$dir" "$PY" bench/bench_native_prefill_scan.py \
    --model "$MODEL" --device cuda --dtype fp16 --batch-sizes 1,8 \
    --prompt-tokens 128 --fused-scan auto --reference-backend hf \
    --quantization none --code-source repo --warmup 5 --steps 25 \
    --timing cuda-event --results "$out"
}
{
 echo "started_at=$(date -Iseconds)"
 echo "gpu=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader)"
 "$PY" - <<'PYENV'
import torch, transformers
try:
 import triton
 tv=triton.__version__
except Exception as e: tv=repr(e)
try:
 import bitsandbytes as bnb
 bv=bnb.__version__
except Exception as e: bv=repr(e)
print(f'torch={torch.__version__} cuda={torch.version.cuda} transformers={transformers.__version__} triton={tv} bitsandbytes={bv}')
PYENV
 run_decode candidate_a "$RUN/candidate" "$RUN/results/candidate_decode.jsonl" || exit $?
 run_decode baseline "$RUN/baseline" "$RUN/results/baseline_decode.jsonl" || exit $?
 run_decode candidate_b "$RUN/candidate" "$RUN/results/candidate_decode.jsonl" || exit $?
 run_prefill candidate_a "$RUN/candidate" "$RUN/results/candidate_prefill.jsonl" || exit $?
 run_prefill baseline "$RUN/baseline" "$RUN/results/baseline_prefill.jsonl" || exit $?
 run_prefill candidate_b "$RUN/candidate" "$RUN/results/candidate_prefill.jsonl" || exit $?
 echo "finished_at=$(date -Iseconds)"
 echo "__EXIT__=0"
} > "$RUN/results/ab_integration.log" 2>&1
