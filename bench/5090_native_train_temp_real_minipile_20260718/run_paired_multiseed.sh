#!/usr/bin/env bash
set -uo pipefail
BASE=/home/ubuntu/bench/native_train_temp_real_minipile_20260718
REPO=/home/ubuntu/rwkv7-hf-native-width
source /home/ubuntu/venv-rwkv5090/bin/activate
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_EXTENSIONS_DIR=/home/ubuntu/bench/train_temp_alignment_20260717/integrated_backend/torch_extensions
export MAX_JOBS=2
export RWKV7_TRAIN_TEMP_CHECKPOINT_BACKEND=deepspeed
cd "$REPO"
COMMON=(--precision bf16 --device cuda --learning-rate 0.0006 --learning-rate-final 0.00006 --schedule-total-steps 182888 --warmup-steps 10 --weight-decay 0.001 --beta1 0.9 --beta2 0.99 --adam-eps 1e-18 --grad-clip 1.0 --eval-interval 50 --optimizer fused_adam)
for EPOCH in 1 2; do
  python bench/bench_train_temp_alignment.py make-dataset-sequence \
    --output "$BASE/train_epoch${EPOCH}_s1000.safetensors" \
    --metadata "$BASE/train_epoch${EPOCH}_s1000.json" \
    --data-prefix /home/ubuntu/datasets/rwkv-minipile-e6f74b6/minipile \
    --batch-size 16 --seq-len 512 --steps 1000 --epoch "$EPOCH" \
    --magic-prime 2926181 > "$BASE/make_train_epoch${EPOCH}.log" 2>&1
done
run_pair() {
  local EPOCH="$1" SEED="$2"
  python bench/bench_train_temp_alignment.py converge-official \
    --sequence "$BASE/train_epoch${EPOCH}_s1000.safetensors" \
    --validation-batch "$BASE/validation_epoch100.safetensors" \
    --output-json "$BASE/official_paired_epoch${EPOCH}_seed${SEED}_s1000.json" \
    --seed "$SEED" "${COMMON[@]}" \
    --official-checkout /home/ubuntu/RWKV-LM-train-temp \
    --official-config "$REPO/configs/train_temp_official_x070_12x768_b16.json" \
    --checkpoint /home/ubuntu/bench/train_temp_alignment_20260717/rwkv-init-12x768-bf16.pth \
    > "$BASE/official_paired_epoch${EPOCH}_seed${SEED}.log" 2>&1
  echo "official epoch=$EPOCH seed=$SEED done $(date -Is)"
  python bench/bench_train_temp_alignment.py converge-hf \
    --sequence "$BASE/train_epoch${EPOCH}_s1000.safetensors" \
    --validation-batch "$BASE/validation_epoch100.safetensors" \
    --output-json "$BASE/native_paired_epoch${EPOCH}_seed${SEED}_s1000.json" \
    --seed "$SEED" "${COMMON[@]}" \
    --model /home/ubuntu/bench/train_temp_alignment_20260717/rwkv-init-12x768-hf \
    --checkpoint-sha256 5fcb1f16231626f0fde51c30c2d51994ef1ec80e6f737735afe83093c253b943 \
    --native --train-temp-cuda --gradient-checkpointing \
    > "$BASE/native_paired_epoch${EPOCH}_seed${SEED}.log" 2>&1
  echo "native epoch=$EPOCH seed=$SEED done $(date -Is)"
  python bench/bench_train_temp_alignment.py compare-convergence \
    --reference-json "$BASE/official_paired_epoch${EPOCH}_seed${SEED}_s1000.json" \
    --candidate-json "$BASE/native_paired_epoch${EPOCH}_seed${SEED}_s1000.json" \
    --output "$BASE/compare_paired_epoch${EPOCH}_seed${SEED}_s1000.json" \
    > "$BASE/compare_paired_epoch${EPOCH}_seed${SEED}.log" 2>&1
  echo "compare epoch=$EPOCH seed=$SEED done $(date -Is)"
}
run_pair 0 131
run_pair 1 232
run_pair 2 333
python bench/bench_train_temp_alignment.py compare-convergence-cohort \
  --reference-json "$BASE/official_paired_epoch0_seed131_s1000.json" \
  --reference-json "$BASE/official_paired_epoch1_seed232_s1000.json" \
  --reference-json "$BASE/official_paired_epoch2_seed333_s1000.json" \
  --candidate-json "$BASE/native_paired_epoch0_seed131_s1000.json" \
  --candidate-json "$BASE/native_paired_epoch1_seed232_s1000.json" \
  --candidate-json "$BASE/native_paired_epoch2_seed333_s1000.json" \
  --success-threshold 5.0 --deep-success-threshold 4.8 \
  --min-candidate-over-reference-throughput-ratio 1.0 \
  --output "$BASE/cohort_paired_s1000.json" > "$BASE/cohort_paired_s1000.log" 2>&1
rc=$?
echo "$rc" > "$BASE/paired_exit_code.txt"
echo "all done rc=$rc $(date -Is)"
