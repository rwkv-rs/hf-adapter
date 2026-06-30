#!/usr/bin/env bash
set -euo pipefail

# Reproducible two-GPU launcher for the formal 3-model x 2-batch x 8-task
# lm-evaluation-harness matrix.  The split keeps HellaSwag and the smaller
# multiple-choice tasks together so both V100s finish at roughly the same time.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-python}"
MODEL_ROOT="${MODEL_ROOT:?set MODEL_ROOT to the directory containing rwkv7_{01b,04b,15b}_hf}"
OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR for the formal result bundle}"
CODE_SHA="${CODE_SHA:-$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || printf unknown)}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT_DIR"

models=(
  --model "0.1b=$MODEL_ROOT/rwkv7_01b_hf"
  --model "0.4b=$MODEL_ROOT/rwkv7_04b_hf"
  --model "1.5b=$MODEL_ROOT/rwkv7_15b_hf"
)

common=(
  --batch-size 1
  --batch-size 8
  --code-sha "$CODE_SHA"
)

CUDA_VISIBLE_DEVICES="$GPU_A" "$PYTHON" "$REPO_ROOT/evaluation/run_lm_eval_matrix.py" \
  --output-dir "$OUTPUT_DIR/shard-a" \
  "${models[@]}" "${common[@]}" \
  --task hellaswag --task winogrande --task openbookqa \
  >"$OUTPUT_DIR/shard-a-launch.log" 2>&1 &
pid_a=$!

CUDA_VISIBLE_DEVICES="$GPU_B" "$PYTHON" "$REPO_ROOT/evaluation/run_lm_eval_matrix.py" \
  --output-dir "$OUTPUT_DIR/shard-b" \
  "${models[@]}" "${common[@]}" \
  --task wikitext --task lambada_openai --task piqa \
  --task arc_easy --task arc_challenge \
  >"$OUTPUT_DIR/shard-b-launch.log" 2>&1 &
pid_b=$!

set +e
wait "$pid_a"
status_a=$?
wait "$pid_b"
status_b=$?
set -e

printf 'shard-a exit=%s\nshard-b exit=%s\n' "$status_a" "$status_b" \
  | tee "$OUTPUT_DIR/shard-status.txt"
if (( status_a != 0 || status_b != 0 )); then
  exit 1
fi

rm -rf "$OUTPUT_DIR/merged"
"$PYTHON" "$REPO_ROOT/evaluation/merge_lm_eval_shards.py" \
  --output-dir "$OUTPUT_DIR/merged" \
  --shard "$OUTPUT_DIR/shard-a" \
  --shard "$OUTPUT_DIR/shard-b"
"$PYTHON" "$REPO_ROOT/evaluation/validate_lm_eval_matrix.py" \
  --result-dir "$OUTPUT_DIR/merged"
