#!/usr/bin/env bash
# Strict, evidence-driven Apple production acceptance audit.
#
# This does not turn smoke rows into production claims. It audits the committed
# manifest and exits non-zero while any required gate is failed, missing, or
# unknown. Evidence is appended rather than overwritten.

set -euo pipefail

source "$(dirname "$0")/_hf_script_common.sh"

MANIFEST="${MANIFEST:-bench/apple_production_gates.json}"
RESULTS="${RESULTS:-bench/results_apple_production_acceptance.jsonl}"
CATEGORY="${CATEGORY:-}"
STRICT="${STRICT:-1}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"

rwkv7_prepare_results

args=(
  bench/check_apple_production_acceptance.py
  --manifest "${MANIFEST}"
  --results "${RESULTS}"
)
if [[ -n "${CATEGORY}" ]]; then
  args+=(--category "${CATEGORY}")
fi
if [[ "${SUMMARY_ONLY}" == "1" ]]; then
  args+=(--summary-only)
fi
if [[ "${STRICT}" == "1" ]]; then
  args+=(--strict)
fi

rwkv7_log "Apple production acceptance audit (strict=${STRICT}, category=${CATEGORY:-all})"
rwkv7_run "${PYTHON_BIN}" "${args[@]}"
