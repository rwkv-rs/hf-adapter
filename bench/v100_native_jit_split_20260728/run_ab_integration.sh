#!/usr/bin/env bash
set -uo pipefail
D=/home/wzu/.private/rwkv7-native-jit-split-v100-20260728
PY=/home/wzu/venvs/v100-active/bin/python
MODEL=/home/wzu/models/rwkv7/rwkv7-g1d-0.1b-hf
export CUDA_VISIBLE_DEVICES=0 HF_HOME=/home/wzu/.cache/huggingface RWKV7_FAST_PREFILL=1 TOKENIZERS_PARALLELISM=false
rm -f "$D/results/baseline_decode.jsonl" "$D/results/candidate_decode.jsonl" "$D/results/baseline_prefill.jsonl" "$D/results/candidate_prefill.jsonl"
run_decode() {
 local role=$1 dir=$2 out=$3
 echo "=== decode $role $(date -Iseconds) ==="
 cd "$dir"
 PYTHONPATH="$dir" "$PY" bench/bench_native_model_decode.py \
   --hf-dir "$MODEL" --dtype fp16 --device cuda --prompt-tokens 128 \
   --decode-steps 256 --warmup 5 --repetitions 2 --batch-sizes 1 8 \
   --backends native_jit native_graph --fast-token-api --results "$out"
}
run_prefill() {
 local role=$1 dir=$2 out=$3
 echo "=== prefill $role $(date -Iseconds) ==="
 cd "$dir"
 PYTHONPATH="$dir" "$PY" bench/bench_native_prefill_scan.py \
   --model "$MODEL" --device cuda --dtype fp16 --batch-sizes 1,8 \
   --prompt-tokens 128 --fused-scan auto --reference-backend hf \
   --quantization none --code-source repo --warmup 10 --steps 50 \
   --timing cuda-event --results "$out"
}
{
 echo "started_at=$(date -Iseconds)"
 echo "gpu=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader -i 0)"
 run_decode candidate_a "$D/candidate" "$D/results/candidate_decode.jsonl" || exit $?
 run_decode baseline "$D/baseline" "$D/results/baseline_decode.jsonl" || exit $?
 run_decode candidate_b "$D/candidate" "$D/results/candidate_decode.jsonl" || exit $?
 run_prefill candidate_a "$D/candidate" "$D/results/candidate_prefill.jsonl" || exit $?
 run_prefill baseline "$D/baseline" "$D/results/baseline_prefill.jsonl" || exit $?
 run_prefill candidate_b "$D/candidate" "$D/results/candidate_prefill.jsonl" || exit $?
 echo "finished_at=$(date -Iseconds)"
 echo "__EXIT__=0"
} > "$D/results/ab_integration.log" 2>&1
