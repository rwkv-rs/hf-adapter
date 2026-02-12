#!/usr/bin/env bash
set -uo pipefail
D=/home/wzu/.private/rwkv7-native-jit-split-v100-20260728
PY=/home/wzu/venvs/v100-active/bin/python
MODEL=/home/wzu/models/rwkv7/rwkv7-g1d-0.1b-hf
export CUDA_VISIBLE_DEVICES=0 HF_HOME=/home/wzu/.cache/huggingface RWKV7_FAST_PREFILL=1 TOKENIZERS_PARALLELISM=false
rm -f "$D/results/baseline_graph_b1_long.jsonl" "$D/results/candidate_graph_b1_long.jsonl"
run_one() {
 local role=$1 dir=$2 out=$3
 echo "=== $role $(date -Iseconds) ==="
 cd "$dir"
 PYTHONPATH="$dir" "$PY" bench/bench_native_model_decode.py \
   --hf-dir "$MODEL" --dtype fp16 --device cuda --prompt-tokens 128 \
   --decode-steps 1024 --warmup 20 --repetitions 5 --batch-sizes 1 \
   --backends native_graph --fast-token-api --results "$out"
}
{
 echo "started_at=$(date -Iseconds)"
 nvidia-smi --query-gpu=name,clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv,noheader -i 0
 run_one candidate_a "$D/candidate" "$D/results/candidate_graph_b1_long.jsonl" || exit $?
 run_one baseline "$D/baseline" "$D/results/baseline_graph_b1_long.jsonl" || exit $?
 run_one candidate_b "$D/candidate" "$D/results/candidate_graph_b1_long.jsonl" || exit $?
 echo "finished_at=$(date -Iseconds)"
 echo "__EXIT__=0"
} > "$D/results/graph_b1_long.log" 2>&1
