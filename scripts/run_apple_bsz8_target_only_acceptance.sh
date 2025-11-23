#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_ROOT="${MODEL_ROOT:-$ROOT/../models}"
RWKV_15="${RWKV_15:-$MODEL_ROOT/rwkv7-g1g-1.5b-hf}"
QWEN_2="${QWEN_2:-$MODEL_ROOT/qwen35-2b-mlx-4bit}"
OUT="${1:-$ROOT/bench/apple_bsz8_active_m5_20260714}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-30}"
INITIAL_COOLDOWN_SECONDS="${INITIAL_COOLDOWN_SECONDS:-60}"

for path in "$RWKV_15" "$QWEN_2"; do
  if [[ ! -e "$path" ]]; then
    echo "missing model: $path" >&2
    exit 2
  fi
done

mkdir -p "$OUT"
RESULTS="$OUT/active_1p5b_target_only_vs_qwen_2b.jsonl"
STDOUT="$OUT/active_1p5b_target_only_vs_qwen_2b.stdout"
rm -f "$RESULTS" "$STDOUT"

echo "[apple-b8-target-only] cool down ${INITIAL_COOLDOWN_SECONDS}s"
sleep "$INITIAL_COOLDOWN_SECONDS"

# This command deliberately omits a draft model. The Python benchmark exits
# non-zero unless target-only RWKV passes both active-normalized speed gates,
# the raw peak-memory gate, and every measured row.
"$PYTHON_BIN" "$ROOT/bench/run_apple_bsz8_active_compare.py" \
  --rwkv-model "$RWKV_15" --qwen-model "$QWEN_2" --rwkv-draft-model "" \
  --batch-size 8 --prompt-chars 512 --decode-tokens 64 \
  --warmup 1 --repeat 3 --order balanced --cooldown-seconds "$COOLDOWN_SECONDS" \
  --rwkv-quant-min-params 1000000 --rwkv-draft-quant-min-params 100000 \
  --rwkv-quant-group-size 128 --rwkv-proposal-tokens 32 \
  --rwkv-fused-lora-down --rwkv-fused-scan-prep-post --rwkv-fused-sequence-mix \
  --rwkv-fused-add-layer-norm --rwkv-fused-square-qmm \
  --no-rwkv-prefix-cache-dedup \
  --results "$RESULTS" \
  | tee "$STDOUT"

echo "[apple-b8-target-only] PASS"
