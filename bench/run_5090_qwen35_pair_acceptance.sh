#!/usr/bin/env bash
# Run the shared configured-batch Qwen3.5 acceptance contract on an exact RTX 5090.
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

export REQUIRED_GPU_MODEL="5090"
export BENCHMARK_MATRIX="${BENCHMARK_MATRIX:-qwen35_5090_hf_final}"
# Blackwell validation must not depend on causal-conv1d extension availability.
# The repository bridge binds FLA's compiled Triton prefill and cached update.
export QWEN_CONV_BACKEND="fla_triton"
export RUN_NATIVE_MM8=1
export REQUIRE_QWEN_FULL_FUSED=1
unset ALLOW_NON_4090

# Blackwell keeps native prefill opt-in globally. The exact-card acceptance
# entrypoint enables the fused route validated on sm_120 without changing the
# repository defaults for other 50-series cards or workloads.
export RWKV7_NATIVE_MODEL=0
export RWKV7_FAST_PREFILL=1
export RWKV7_FAST_PREFILL_QUANT=1
export RWKV7_NATIVE_PREFILL_GRAPH=1
export RWKV7_NATIVE_PREFILL_EXTERNAL_QUANT_GRAPH=1
export RWKV7_NATIVE_PREFILL_FUSED_SCAN=1
export RWKV7_NATIVE_PREFILL_FUSED_SHIFT_MIX=1
export RWKV7_NATIVE_PREFILL_FUSED_STATE_PREP=1
export RWKV7_NATIVE_PREFILL_FUSED_STATE_SCAN=0
export RWKV7_NATIVE_PREFILL_FUSED_OUTPUT=1
export RWKV7_NATIVE_PREFILL_FUSED_SCAN_OUTPUT=0
export RWKV7_NATIVE_PREFILL_DPLR_SCAN=0
# Exact-card clampw/stacked routes come from KernelPolicy shape gates. Clear
# inherited overrides so this runner measures the checked-in policy.
unset RWKV7_NATIVE_PREFILL_FUSED_CLAMPW_SCAN
unset RWKV7_NATIVE_PREFILL_STACKED_RKV
unset RWKV7_NATIVE_PREFILL_FUSED_RESIDUAL_GEMM
unset RWKV7_NATIVE_PREFILL_FUSED_SEQUENCE_FFN

exec "${ROOT}/bench/run_4090_qwen35_pair_acceptance.sh" "$@"
