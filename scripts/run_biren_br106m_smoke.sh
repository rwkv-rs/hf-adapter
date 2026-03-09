#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RESULTS="${RESULTS:-$ROOT/bench/biren_br106m_$(date +%Y%m%d)/results.json}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
ARGS=(--output "$RESULTS")
if [[ -n "${MODEL:-}" ]]; then
  ARGS+=(--model "$MODEL")
fi
[[ "${CPU_ORACLE:-0}" == "1" ]] && ARGS+=(--cpu-oracle)
[[ "${SAVE_RELOAD:-0}" == "1" ]] && ARGS+=(--save-reload)
"$PYTHON_BIN" "$ROOT/tests/test_biren_br106m_smoke.py" "${ARGS[@]}"
echo "PASS: $RESULTS"
