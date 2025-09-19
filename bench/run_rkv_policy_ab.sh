#!/usr/bin/env bash
# One-click A/B for the VKWR-style stacked R/K/V projection policy
# (RWKV7_NATIVE_GRAPH_RKV_POLICY=vkwr_auto) vs the default three-F.linear
# path (manual). GPU-agnostic -- run on any CUDA card to measure how much
# the stacked path helps THERE.
#
# Verified on RTX 5070 Laptop (sm_120): +7.4% @0.1B bsz1, +5.2% bsz8,
# +2.4% @1.5B bsz1 (greedy-identical). The win is a generic "fewer launches
# + batched matmul" effect, NOT Blackwell-specific -- expect >= this on V100
# (launch overhead is relatively costlier on older arch).
#
# Usage:
#   bench/run_rkv_policy_ab.sh <hf_dir> [dtype=fp16] [batch=1]
# Example (V100):
#   bench/run_rkv_policy_ab.sh /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf fp16 1
set -euo pipefail

HF_DIR=${1:?"usage: $0 <hf_dir> [dtype] [batch]"}
DTYPE=${2:-fp16}
BATCH=${3:-1}
PY=${PYTHON:-python}

echo "=== VKWR stacked-RKV policy A/B ==="
"$PY" - <<'PY'
import torch
print(f"GPU: {torch.cuda.get_device_name(0)} | cap {torch.cuda.get_device_capability()}")
PY

export RWKV_V7_ON=1
"$PY" bench/bench_native_graph_vkwr_rkv_policy.py \
    --hf-dir "$HF_DIR" --dtype "$DTYPE" --batch-size "$BATCH" --steps 64 --warmup 8
