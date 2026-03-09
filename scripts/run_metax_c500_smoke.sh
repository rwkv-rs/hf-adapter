#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RESULTS="${RESULTS:-$ROOT/bench/metax_c500_$(date +%Y%m%d)/results.json}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
ARGS=(--output "$RESULTS")
if [[ -n "${MODEL:-}" ]]; then
  ARGS+=(--model "$MODEL")
fi
"$PYTHON_BIN" "$ROOT/tests/test_metax_c500_smoke.py" "${ARGS[@]}"
echo "PASS: $RESULTS"
