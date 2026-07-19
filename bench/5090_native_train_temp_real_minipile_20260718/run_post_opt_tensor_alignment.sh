#!/usr/bin/env bash
set -euo pipefail
source /home/ubuntu/venv-rwkv5090/bin/activate
export PYTHONPATH=/home/ubuntu/rwkv7-hf-native-width:${PYTHONPATH:-}
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_EXTENSIONS_DIR=/home/ubuntu/bench/train_temp_alignment_20260717/integrated_backend/torch_extensions
export MAX_JOBS=2
export RWKV7_TRAIN_TEMP_CHECKPOINT_BACKEND=deepspeed
cd /home/ubuntu/rwkv7-hf-native-width
OUT=/home/ubuntu/bench/native_train_temp_real_minipile_20260718/post_opt_tensor_alignment; mkdir -p "$OUT"
BATCH="$OUT/batch_seed131_b16_t512.safetensors"
python bench/bench_train_temp_alignment.py make-batch --output "$BATCH" --metadata "$OUT/batch_seed131_b16_t512.json" --vocab-size 65536 --batch-size 16 --seq-len 512 --seed 131 --pattern increment --active-vocab-size 256
COMMON=(--batch "$BATCH" --precision bf16 --device cuda --seed 131 --learning-rate 0.0006 --weight-decay 0.001 --beta1 0.9 --beta2 0.99 --adam-eps 1e-18 --grad-clip 1.0 --optimizer fused_adam --omit-logits)
for PHASE in backward step; do
 python bench/bench_train_temp_alignment.py capture-official "${COMMON[@]}" --phase "$PHASE" --output-json "$OUT/official_${PHASE}.json" --snapshot "$OUT/official_${PHASE}.safetensors" --official-checkout /home/ubuntu/RWKV-LM-train-temp --official-config configs/train_temp_official_x070_12x768_b16.json --checkpoint /home/ubuntu/bench/train_temp_alignment_20260717/rwkv-init-12x768-bf16.pth > "$OUT/official_${PHASE}.log" 2>&1
 python bench/bench_train_temp_alignment.py capture-hf "${COMMON[@]}" --phase "$PHASE" --output-json "$OUT/native_${PHASE}.json" --snapshot "$OUT/native_${PHASE}.safetensors" --model /home/ubuntu/bench/train_temp_alignment_20260717/rwkv-init-12x768-hf --checkpoint-sha256 5fcb1f16231626f0fde51c30c2d51994ef1ec80e6f737735afe83093c253b943 --native --train-temp-cuda --gradient-checkpointing > "$OUT/native_${PHASE}.log" 2>&1
 python bench/bench_train_temp_alignment.py compare --reference-json "$OUT/official_${PHASE}.json" --candidate-json "$OUT/native_${PHASE}.json" --output "$OUT/compare_${PHASE}.json" > "$OUT/compare_${PHASE}.log" 2>&1
done
rm -f "$OUT"/*.safetensors
echo 0 > "$OUT/exit_code.txt"
