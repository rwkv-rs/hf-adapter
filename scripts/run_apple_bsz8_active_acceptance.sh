#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_ROOT="${MODEL_ROOT:-$ROOT/../models}"
RWKV_01="${RWKV_01:-$MODEL_ROOT/rwkv7-g1d-0.1b-hf}"
RWKV_04="${RWKV_04:-$MODEL_ROOT/rwkv7-g1d-0.4b-hf}"
RWKV_15="${RWKV_15:-$MODEL_ROOT/rwkv7-g1g-1.5b-hf}"
QWEN_08="${QWEN_08:-$MODEL_ROOT/qwen35-0.8b-mlx-4bit}"
QWEN_2="${QWEN_2:-$MODEL_ROOT/qwen35-2b-mlx-4bit}"
OUT="${1:-$ROOT/bench/apple_bsz8_active_m5_20260714}"
COOLDOWN_SECONDS="${COOLDOWN_SECONDS:-30}"
INITIAL_COOLDOWN_SECONDS="${INITIAL_COOLDOWN_SECONDS:-60}"

for path in "$RWKV_01" "$RWKV_04" "$RWKV_15" "$QWEN_08" "$QWEN_2"; do
  if [[ ! -e "$path" ]]; then
    echo "missing model: $path" >&2
    exit 2
  fi
done

mkdir -p "$OUT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

# The Python harness appends JSONL rows so an interrupted run is inspectable.
# A fresh acceptance invocation must not mix rows from an older candidate.
rm -f \
  "$OUT/fidelity_0p4b.jsonl" \
  "$OUT/active_0p4b_vs_qwen_0p8b.jsonl" \
  "$OUT/fidelity_1p5b_cache.jsonl" \
  "$OUT/active_1p5b_cache_vs_qwen_2b.jsonl" \
  "$OUT/active_1p5b_cold_vs_qwen_2b.jsonl"

echo "[apple-b8] output=$OUT"
echo "[apple-b8] cool down ${INITIAL_COOLDOWN_SECONDS}s before isolated engine rows"
sleep "$INITIAL_COOLDOWN_SECONDS"

"$PYTHON_BIN" "$ROOT/bench/validate_apple_bsz8_fidelity.py" \
  --model "$RWKV_04" \
  --batch-size 8 --prompt-chars 512 --decode-tokens 64 \
  --quant-group-size 128 --fused-lora-down --fused-ffn-key-relu2 \
  --fused-scan-prep-post --fused-sequence-mix --fused-add-layer-norm --fused-square-qmm \
  --results "$OUT/fidelity_0p4b.jsonl" \
  | tee "$OUT/fidelity_0p4b.stdout"

"$PYTHON_BIN" "$ROOT/bench/run_apple_bsz8_active_compare.py" \
  --rwkv-model "$RWKV_04" --qwen-model "$QWEN_08" --rwkv-draft-model "$RWKV_01" \
  --batch-size 8 --prompt-chars 512 --decode-tokens 64 \
  --warmup 1 --repeat 3 --order qwen-first --cooldown-seconds "$COOLDOWN_SECONDS" \
  --rwkv-quant-min-params 1000000 --rwkv-draft-quant-min-params 100000 \
  --rwkv-quant-group-size 128 --rwkv-proposal-tokens 32 \
  --rwkv-fused-scan-prep-post --rwkv-fused-sequence-mix \
  --rwkv-fused-add-layer-norm --rwkv-fused-square-qmm \
  --results "$OUT/active_0p4b_vs_qwen_0p8b.jsonl" \
  | tee "$OUT/active_0p4b_vs_qwen_0p8b.stdout"

sleep "$COOLDOWN_SECONDS"

"$PYTHON_BIN" "$ROOT/bench/validate_apple_bsz8_fidelity.py" \
  --model "$RWKV_15" \
  --draft-model "$RWKV_01" \
  --batch-size 8 --prompt-chars 512 --decode-tokens 64 \
  --quant-group-size 128 --fused-lora-down --fused-ffn-key-relu2 \
  --fused-scan-prep-post --fused-sequence-mix --fused-add-layer-norm --fused-square-qmm \
  --compare-fp16 --compare-fused-post --compare-prefix-cache --prefix-unique-prompts 2 \
  --compare-speculative-mismatch \
  --results "$OUT/fidelity_1p5b_cache.jsonl" \
  | tee "$OUT/fidelity_1p5b_cache.stdout"

"$PYTHON_BIN" "$ROOT/bench/run_apple_bsz8_active_compare.py" \
  --rwkv-model "$RWKV_15" --qwen-model "$QWEN_2" --rwkv-draft-model "$RWKV_01" \
  --batch-size 8 --prompt-chars 512 --decode-tokens 64 \
  --warmup 1 --repeat 3 --order qwen-first --cooldown-seconds "$COOLDOWN_SECONDS" \
  --rwkv-quant-min-params 1000000 --rwkv-draft-quant-min-params 100000 \
  --rwkv-quant-group-size 128 --rwkv-proposal-tokens 32 \
  --rwkv-fused-lora-down --rwkv-fused-scan-prep-post --rwkv-fused-sequence-mix \
  --rwkv-fused-add-layer-norm --rwkv-fused-square-qmm --rwkv-prefix-cache-dedup \
  --results "$OUT/active_1p5b_cache_vs_qwen_2b.jsonl" \
  | tee "$OUT/active_1p5b_cache_vs_qwen_2b.stdout"

# The independent cold row is order-balanced (ABBA) because a single
# qwen-first pass materially heat-biases the fanless M5.  No prefix/state
# cache coalescing is enabled in this gate.
sleep "$INITIAL_COOLDOWN_SECONDS"
"$PYTHON_BIN" "$ROOT/bench/run_apple_bsz8_active_compare.py" \
  --rwkv-model "$RWKV_15" --qwen-model "$QWEN_2" --rwkv-draft-model "$RWKV_01" \
  --batch-size 8 --prompt-chars 512 --decode-tokens 64 \
  --warmup 1 --repeat 3 --order balanced --cooldown-seconds "$COOLDOWN_SECONDS" \
  --rwkv-quant-min-params 1000000 --rwkv-draft-quant-min-params 100000 \
  --rwkv-quant-group-size 128 --rwkv-proposal-tokens 32 \
  --rwkv-fused-lora-down --rwkv-fused-scan-prep-post --rwkv-fused-sequence-mix \
  --rwkv-fused-add-layer-norm --rwkv-fused-square-qmm --no-rwkv-prefix-cache-dedup \
  --results "$OUT/active_1p5b_cold_vs_qwen_2b.jsonl" \
  | tee "$OUT/active_1p5b_cold_vs_qwen_2b.stdout"

echo "[apple-b8] PASS"
