#!/usr/bin/env bash
# Hardware/card validation wrapper: model API + quant + short speed/batch rows.
# This intentionally skips static tests and training unless overridden.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_STATIC_TESTS="${RUN_STATIC_TESTS:-0}"
export RUN_MODEL_SMOKE="${RUN_MODEL_SMOKE:-1}"
export RUN_QUANT="${RUN_QUANT:-1}"
export RUN_BENCH="${RUN_BENCH:-1}"
export RUN_BATCH_SWEEP="${RUN_BATCH_SWEEP:-1}"
export RUN_TRAINING="${RUN_TRAINING:-0}"
exec "${SCRIPT_DIR}/run_hf_acceptance.sh" "$@"
