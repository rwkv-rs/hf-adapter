# RWKV-7 vs Qwen3.5：GPU 速度复现教程

[English](QWEN35_SPEED_REPRODUCTION.md) · [结果总表](QWEN35_LATEST_P_D_TOKPS.md)

本教程直接在单张 NVIDIA GPU 上重跑 RWKV-7 与 Qwen3.5，分别输出 Prefill、
Decode、活跃参数和两种速度比。最快的验证路线是 RTX 4090 上复现
`RWKV-7 1.5B / Qwen3.5-2B / B8`；完整正式矩阵入口在文末。

## 1. 固定测试口径

| 项目 | 设置 |
|---|---|
| 精度 | Dense FP16；不开量化、MTP、speculative decode |
| Batch | 快速教程用 B8；正式矩阵用 B1、B8 |
| Prompt | 128、512、2048 token |
| Decode | 128、512 token |
| 统计 | warmup 3 次，正式运行 7 次，取中位值 |
| RWKV | Native Prefill + Native Graph cached Decode |
| Qwen | Transformers + FLA + StaticCache CUDA Graph；RTX 4090 快速路线使用官方 `causal_conv1d` |
| 吞吐 | B8 为 8 条序列合计的 aggregate `tok/s` |

只比较 Prefill 和 Decode，不计算 E2E。两种速度比为：

```text
原始速度比 = RWKV tok/s / Qwen tok/s
参数规模校正速度比 = 原始速度比 × RWKV 活跃参数 / Qwen 活跃参数
```

## 2. 准备机器与环境

以下快速命令按 RTX 4090 的已验证运行时编写。使用 Linux、CUDA developer
环境，并让 `nvcc`、C++ 编译器和目标 GPU 可见。

```bash
nvidia-smi

git clone https://github.com/rwkv-rs/hf-adapter.git
cd hf-adapter
git checkout main
git pull --ff-only

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools packaging ninja
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install transformers==5.12.1 triton==3.3.1
python -m pip install --no-build-isolation causal-conv1d==1.6.2.post1
python -m pip install --no-build-isolation flash-linear-attention==0.5.1
python -m pip install -e ".[cuda,fla-reference]"
```

验证 GPU 和版本：

```bash
python - <<'PY'
import importlib.metadata as md
import torch, transformers, triton

print("GPU:", torch.cuda.get_device_name(0))
print("compute capability:", torch.cuda.get_device_capability(0))
print("torch/CUDA:", torch.__version__, torch.version.cuda)
print("transformers/triton:", transformers.__version__, triton.__version__)
print("FLA:", md.version("flash-linear-attention"))
print("causal-conv1d:", md.version("causal-conv1d"))
PY
```

正式产物使用的精确运行时如下；逐包锁见各证据目录的 `pip-freeze.txt`。

| GPU | Python | PyTorch / CUDA | Triton | Transformers | FLA / causal-conv1d |
|---|---|---|---|---|---|
| V100 | 3.11.15 | 2.5.1+cu124 / 12.4 | 3.4.0 | 5.12.1 | 0.5.1 / 仓库 FLA-Triton conv |
| RTX 3090 | 3.10.12 | 2.7.1+cu126 / 12.6 | 3.3.1 | 5.12.1 | 0.5.1 / 1.6.2.post1 |
| RTX 4080 | 3.12.2 | 2.11.0+cu130 / 13.0 | 3.6.0 | 5.12.1 | 0.5.1 / 1.6.2.post1 |
| RTX 4090 | 3.12.8 | 2.7.1+cu126 / 12.6 | 3.3.1 | 5.12.1 | 0.5.1 / 1.6.2.post1 |
| RTX 5090 | 3.10.12 | 2.8.0+cu128 / 12.8 | 3.4.0 | 5.12.1 | 0.5.1 / 1.6.2.post1 |

## 3. 准备模型

Qwen 直接使用官方 HF 模型目录，例如：

```bash
python -m pip install -U "huggingface_hub[cli]"
hf download Qwen/Qwen3.5-2B --local-dir /models/Qwen3.5-2B
export QWEN_MODEL=/models/Qwen3.5-2B
```

RWKV 使用官方 `.pth`、官方词表和本仓库转换脚本：

```bash
rwkv7-hf convert \
  --input /models/rwkv7-g1i-1.5b.pth \
  --output /models/rwkv7-g1i-1.5b-hf \
  --vocab-file /models/rwkv_vocab_v20230424.txt \
  --precision fp16 \
  --attn-mode chunk \
  --no-fuse-norm

export RWKV_MODEL=/models/rwkv7-g1i-1.5b-hf
python examples/check_environment.py --model "$RWKV_MODEL"
```

检查结果应包含 `RESULT: READY` 和 `[PASS] Model directory`。

四组正式配对为：

| RWKV-7 | Qwen3.5 | RWKV 活跃参数 | Qwen 活跃参数 |
|---|---|---:|---:|
| 0.4B | 0.8B | 0.451B | 0.752B |
| 1.5B | 2B | 1.527B | 1.882B |
| 2.9B | 4B | 2.948B | 4.206B |
| 7.2B | 9B | 7.199B | 8.954B |

## 4. RTX 4090 快速复现：1.5B 对 2B，B8

先建立互相独立的输出和编译缓存。旧结果必须删除，避免 JSONL 追加：

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OUT=/tmp/rwkv-qwen35-repro-4090
rm -rf "$OUT"
mkdir -p "$OUT"/{rwkv-cache,qwen-cache}
```

### 4.1 跑 RWKV

```bash
export RWKV7_FAST_TOKEN_BACKEND=native_graph
export RWKV7_NATIVE_MODEL_BACKEND=native_graph
export RWKV7_NATIVE_PREFILL_GRAPH=1
export RWKV7_NATIVE_GRAPH_ADA_WAGV_BMM=1
export RWKV7_NATIVE_GRAPH_SM120_WAGV_BMM_G=1
export RWKV7_NATIVE_GRAPH_SM120_COMPILED_FFN=1
export RWKV7_NATIVE_GRAPH_RKV_POLICY=vkwr_auto
export RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN=0
export RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_LOW_MEMORY_PACK=0
export RWKV7_BLACKWELL_TORCH_COMPILE=1

TORCHINDUCTOR_CACHE_DIR="$OUT/rwkv-cache" \
TRITON_CACHE_DIR="$OUT/rwkv-cache" \
python bench/bench_cross_model_speed_resident.py \
  --model "$RWKV_MODEL" \
  --model-kind rwkv --model-role candidate \
  --model-pair rwkv-1.5b__qwen3.5-2b --model-size-label 1.5b \
  --benchmark-matrix external_repro_v1 --optimization-lane best_optimized_hf \
  --dtype fp16 --quantization none --device cuda --batch-sizes 8 \
  --prompt-tokens 128 512 2048 --decode-tokens 128 512 \
  --prefill-chunk-size 512 --warmup 3 --runs 7 \
  --rwkv-attn-mode fused_recurrent --rwkv-code-source repo \
  --rwkv-implementation auto --fail-fast \
  --results "$OUT/rwkv.jsonl"
```

### 4.2 跑 Qwen

Qwen 必须使用同一张卡和同一运行时。先清除 RWKV 路线变量，再启用完整 FLA
和 CUDA Graph 路线：

```bash
while IFS= read -r name; do unset "$name"; done \
  < <(compgen -e | grep '^RWKV7_' || true)

TORCHINDUCTOR_CACHE_DIR="$OUT/qwen-cache" \
TRITON_CACHE_DIR="$OUT/qwen-cache" \
python bench/bench_cross_model_speed_resident.py \
  --model "$QWEN_MODEL" \
  --model-kind qwen35 --model-role reference \
  --model-pair rwkv-1.5b__qwen3.5-2b --model-size-label 2b \
  --benchmark-matrix external_repro_v1 --optimization-lane qwen_best_optimized_hf \
  --dtype fp16 --quantization none --device cuda --batch-sizes 8 \
  --prompt-tokens 128 512 2048 --decode-tokens 128 512 \
  --prefill-chunk-size 512 --warmup 3 --runs 7 \
  --qwen-backend fla --qwen-conv-backend causal_conv1d \
  --require-qwen-fast-path \
  --qwen-decode-optimization static_cache_inductor_cudagraph \
  --qwen-compile-mode max-autotune --qwen-graph-probe-tokens 16 \
  --fail-fast --results "$OUT/qwen.jsonl"
```

首次 Qwen 运行会先编译和捕获 Graph，等待命令自然结束；正式计时不包含模型
加载和 Graph 建立时间。

## 5. 汇总结果并验收后端

```bash
python - "$OUT/rwkv.jsonl" "$OUT/qwen.jsonl" <<'PY'
import json, statistics, sys

def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

rwkv, qwen = map(load, sys.argv[1:])
key = lambda r: (r["batch_size"], r["prompt_tokens"], r["decode_tokens"])
rmap, qmap = ({key(r): r for r in rows} for rows in (rwkv, qwen))
assert rmap.keys() == qmap.keys() and len(rmap) == 6

raw_p, raw_d, adj_p, adj_d = [], [], [], []
print(" B  P    D  | RWKV P/D tok/s       | Qwen P/D tok/s       | raw P/D     | adjusted P/D")
for cell in sorted(rmap):
    r, q = rmap[cell], qmap[cell]
    assert r["status"] == q["status"] == "pass"
    assert r["effective_backend"] == "native_graph"
    assert q["qwen_fast_path_verified"] is True
    assert q["qwen_full_fused_contract_pass"] is True
    assert q["qwen_conv_backend_effective"] == "causal_conv1d"
    assert q["qwen_decode_cuda_graph_verified"] is True
    scale = r["active_parameter_count"] / q["active_parameter_count"]
    rp = r["prefill_tokps_total_raw"] / q["prefill_tokps_total_raw"]
    rd = r["decode_tokps_total_raw"] / q["decode_tokps_total_raw"]
    raw_p.append(rp); raw_d.append(rd); adj_p.append(rp * scale); adj_d.append(rd * scale)
    print(f"{cell[0]:2d} {cell[1]:4d} {cell[2]:4d} | "
          f"{r['prefill_tokps_total_raw']:9.3f}/{r['decode_tokps_total_raw']:8.3f} | "
          f"{q['prefill_tokps_total_raw']:9.3f}/{q['decode_tokps_total_raw']:8.3f} | "
          f"{rp:5.3f}x/{rd:5.3f}x | {rp*scale:5.3f}x/{rd*scale:5.3f}x")

print("median raw P/D:", f"{statistics.median(raw_p):.3f}x", f"{statistics.median(raw_d):.3f}x")
print("median adjusted P/D:", f"{statistics.median(adj_p):.3f}x", f"{statistics.median(adj_d):.3f}x")
PY
```

RTX 4090 正式产物中，这一组 B8 六格的中位结果约为：

| | RWKV | Qwen | 原始速度比 | 参数规模校正速度比 |
|---|---:|---:|---:|---:|
| Prefill | 57,929 tok/s | 36,953 tok/s | 1.568x | 1.272x |
| Decode | 1,909 tok/s | 1,302 tok/s | 1.468x | 1.191x |

不同驱动、时钟、散热和系统负载会带来少量波动；判断时使用六格原始 JSONL，
不要用文档中已经舍入的数值反推。

## 6. 跑完整正式矩阵

完整矩阵为四组模型（4080 为前三组）、B1/B8、P128/512/2048、D128/512。
各正式 runner 会锁定 GPU、运行时、模型哈希、后端遥测和正确性结果。

| GPU | 正式入口 | 结果与精确环境 |
|---|---|---|
| V100 | [`run_v100_qwen35_paired_pd_v1.sh`](../bench/run_v100_qwen35_paired_pd_v1.sh) | [P+D v1](../bench/v100_qwen35_paired_pd_v1_20260814/README.md) |
| RTX 3090 | [`run_3090_rwkv_paired_pd_v2.sh`](../bench/run_3090_rwkv_paired_pd_v2.sh)、[`run_3090_qwen35_best_optimized_hf.sh`](../bench/run_3090_qwen35_best_optimized_hf.sh)、[`validate_qwen35_3090_paired_pd_v2.py`](../bench/validate_qwen35_3090_paired_pd_v2.py) | [P+D v2](../bench/3090_qwen35_paired_pd_v2_20260816/README.md) |
| RTX 4080 | [`run_4080_qwen35_paired_pd_v1.sh`](../bench/run_4080_qwen35_paired_pd_v1.sh) | [P+D v1](../bench/4080_qwen35_paired_pd_v1_20260814/README.md) |
| RTX 4090 | [`run_4090_rwkv_paired_pd_v2.sh`](../bench/run_4090_rwkv_paired_pd_v2.sh)、[`run_5090_qwen35_best_optimized_hf.sh`](../bench/run_5090_qwen35_best_optimized_hf.sh)、[`validate_qwen35_4090_paired_pd_v2.py`](../bench/validate_qwen35_4090_paired_pd_v2.py) | [P+D v2](../bench/4090_qwen35_paired_pd_v2_20260815/README.md) |
| RTX 5090 | [`run_5090_rwkv_paired_decode_v1.sh`](../bench/run_5090_rwkv_paired_decode_v1.sh)、[`run_5090_qwen35_best_optimized_hf.sh`](../bench/run_5090_qwen35_best_optimized_hf.sh)、[`validate_qwen35_paired_decode_v1.py`](../bench/validate_qwen35_paired_decode_v1.py) | [Decode v1](../bench/5090_qwen35_paired_decode_v1_20260813/README.md) |

以 RTX 4080 为例，准备全新输出目录和模型变量后即可一次跑完：

```bash
export OUT_DIR=/data/bench/4080-paired-pd-v1
export CACHE_ROOT=/data/cache/4080-paired-pd-v1
export PYTHON_BIN=/path/to/4080-runtime/bin/python
export REPOSITORY_COMMIT=$(git rev-parse HEAD)
export CUDA_TOOLKIT_VIEW=/usr/local/cuda
export CUDA_COMPONENT_INCLUDE=/usr/local/cuda/include
export RWKV_04_MODEL=/models/rwkv7-0.4b-hf
export RWKV_15_MODEL=/models/rwkv7-1.5b-hf
export RWKV_29_MODEL=/models/rwkv7-2.9b-hf
export QWEN_08_MODEL=/models/Qwen3.5-0.8B
export QWEN_2_MODEL=/models/Qwen3.5-2B
export QWEN_4_MODEL=/models/Qwen3.5-4B

# OUT_DIR 和 CACHE_ROOT 在运行前都不能存在。
bash bench/run_4080_qwen35_paired_pd_v1.sh
cat "$OUT_DIR/exit_code.txt"
cat "$OUT_DIR/paired_validation.json"
```

其他显卡先打开上表的结果 README，复制其中运行时锁和模型变量；每个模型使用
独立的 Inductor/Triton 缓存。需要精确扩展来源时使用：

```bash
export FLA_SOURCE_COMMIT=2e38c1fab332174d056928feaf29f8c5fd5ac550
export CAUSAL_CONV1D_SOURCE_COMMIT=4f6ae4e26ae5fe8af9372f8d312ab25cc4595223
bash bench/build_hf_fast_path_v1_extensions.sh
```

## 7. 常见问题

- **Qwen 快速路径未通过：** 检查输出行中的
  `qwen_fast_path_verified`、`qwen_full_fused_contract_pass`、
  `qwen_conv_backend_effective` 和 `qwen_decode_cuda_graph_verified`。四项必须分别为
  `true`、`true`、`causal_conv1d`、`true`。
- **结果文件重复：** 删除旧 JSONL 和缓存后重跑。正式 runner 会直接拒绝已有的
  `OUT_DIR`；不要把两次测量追加到同一个文件。
- **显存不足：** 关闭桌面、训练任务和其他 GPU 进程；为每个模型保留独立缓存，
  并设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
- **编译或换卡后变慢：** 删除该卡对应的 `TORCHINDUCTOR_CACHE_DIR` 和
  `TRITON_CACHE_DIR`。不同 GPU、PyTorch 或 Triton 版本不要共用编译缓存。
- **结果抖动：** 保持 GPU 空闲和散热稳定，使用 3 次 warmup、7 次正式运行，比较
  中位数与逐格 JSONL。

最终公开复现材料至少保留：`git rev-parse HEAD`、`nvidia-smi`、`pip freeze`、
模型哈希、两个原始 JSONL、完整日志和汇总输出。
