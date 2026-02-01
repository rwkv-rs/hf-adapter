#!/usr/bin/env bash
set -uo pipefail
D=/home/wzu/.private/rwkv7-native-jit-split-v100-20260728
C=$D/candidate
PY=/home/wzu/venvs/v100-active/bin/python
MODEL_SRC=/home/wzu/models/rwkv7/rwkv7-g1d-0.4b-hf
TOK_SRC=/home/wzu/models/rwkv7/rwkv7-g1d-0.1b-hf
MODEL=$D/model-0.4b-hf
mkdir -p "$MODEL"
for f in "$MODEL_SRC"/*; do ln -sfn "$f" "$MODEL/$(basename "$f")"; done
for name in rwkv_vocab_v20230424.txt special_tokens_map.json tokenizer_config.json tokenization_rwkv7.py; do
  ln -sfn "$TOK_SRC/$name" "$MODEL/$name"
done
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$C HF_HOME=/home/wzu/.cache/huggingface
cd "$C"
{
 echo "started_at=$(date -Iseconds)"
 "$PY" - <<'PYENV'
import platform,torch,transformers
import triton,bitsandbytes
print('python='+platform.python_version())
print('torch='+torch.__version__)
print('cuda='+str(torch.version.cuda))
print('transformers='+transformers.__version__)
print('triton='+triton.__version__)
print('bitsandbytes='+bitsandbytes.__version__)
print('gpu='+torch.cuda.get_device_name(0))
print('cc='+str(torch.cuda.get_device_capability(0)))
PYENV
 "$PY" -m pytest -q tests/test_native_jit_*_split.py tests/test_native_quant_bnb8.py
 rc=$?
 echo "finished_at=$(date -Iseconds)"
 echo "__EXIT__=$rc"
 exit "$rc"
} > "$D/results/candidate_cuda_split_tests.log" 2>&1
