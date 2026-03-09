#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL="${1:-${MODEL:-}}"
RESULTS="${RESULTS:-$ROOT/bench/ascend_910b3_$(date +%Y%m%d)/results.jsonl}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ -z "$MODEL" ]]; then
  MODEL="$(mktemp -d /tmp/rwkv7-tiny-hf.XXXXXX)"
  "$PYTHON_BIN" "$ROOT/scripts/create_tiny_native_hf_model.py" --output "$MODEL"
  trap 'rm -rf "$MODEL"' EXIT
fi
"$PYTHON_BIN" "$ROOT/tests/test_huawei_ascend_smoke.py" \
  --model "$MODEL" --device "${DEVICE:-npu:0}" --dtype "${DTYPE:-bf16}" \
  --backend "${BACKEND:-eager}" --results "$RESULTS" \
  ${TRAINING_SMOKE:+--training-smoke}
echo "PASS: $RESULTS"
