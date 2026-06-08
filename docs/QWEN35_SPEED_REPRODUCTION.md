# RWKV-7 vs Qwen3.5: GPU Speed Reproduction

[中文版](QWEN35_SPEED_REPRODUCTION_ZH.md) · [Result tables](QWEN35_LATEST_P_D_TOKPS_EN.md)

This guide reruns RWKV-7 and Qwen3.5 on one NVIDIA GPU and reports Prefill,
Decode, active parameters, raw speed ratio, and parameter-adjusted speed ratio.
The shortest verified route is RWKV-7 1.5B versus Qwen3.5-2B at B8 on an RTX
4090. Full-card entry points are listed below.

## 1. Fixed protocol

| Item | Setting |
|---|---|
| Precision | Dense FP16; no quantization, MTP, or speculative decode |
| Batch | B8 in the quick run; B1 and B8 in the formal matrix |
| Prompt | 128, 512, and 2048 tokens |
| Decode | 128 and 512 tokens |
| Statistic | 3 warmups, 7 measured runs, median |
| RWKV | Native Prefill + Native Graph cached Decode |
| Qwen | Transformers + FLA + StaticCache CUDA Graph; the RTX 4090 quick route uses official `causal_conv1d` |
| Throughput | B8 is aggregate `tok/s` across eight sequences |

Prefill and Decode are measured separately; no E2E value is derived.

```text
raw ratio = RWKV tok/s / Qwen tok/s
parameter-adjusted ratio = raw ratio × RWKV active parameters / Qwen active parameters
```

## 2. Environment

The quick commands use the validated RTX 4090 runtime. Run them in a Linux
CUDA developer environment with `nvcc`, a C++ compiler, and the target GPU.

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

Check the device and runtime:

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

Validated formal runtimes are:

| GPU | Python | PyTorch / CUDA | Triton | Transformers | FLA / causal-conv1d |
|---|---|---|---|---|---|
| V100 | 3.11.15 | 2.5.1+cu124 / 12.4 | 3.4.0 | 5.12.1 | 0.5.1 / repository FLA-Triton conv |
| RTX 3090 | 3.10.12 | 2.7.1+cu126 / 12.6 | 3.3.1 | 5.12.1 | 0.5.1 / 1.6.2.post1 |
| RTX 4080 | 3.12.2 | 2.11.0+cu130 / 13.0 | 3.6.0 | 5.12.1 | 0.5.1 / 1.6.2.post1 |
| RTX 4090 | 3.12.8 | 2.7.1+cu126 / 12.6 | 3.3.1 | 5.12.1 | 0.5.1 / 1.6.2.post1 |
| RTX 5090 | 3.10.12 | 2.8.0+cu128 / 12.8 | 3.4.0 | 5.12.1 | 0.5.1 / 1.6.2.post1 |

Use the evidence directory's `pip-freeze.txt` for the complete package lock.

## 3. Models

Download the official Qwen model directly:

```bash
python -m pip install -U "huggingface_hub[cli]"
hf download Qwen/Qwen3.5-2B --local-dir /models/Qwen3.5-2B
export QWEN_MODEL=/models/Qwen3.5-2B
```

Convert the official RWKV `.pth` checkpoint with the official vocabulary:

```bash
rwkv7-hf convert \
  --input /models/rwkv7-g1i-1.5b.pth \
  --output /models/rwkv7-g1i-1.5b-hf \
  --vocab-file /models/rwkv_vocab_v20230424.txt \
  --precision fp16 --attn-mode chunk --no-fuse-norm

export RWKV_MODEL=/models/rwkv7-g1i-1.5b-hf
python examples/check_environment.py --model "$RWKV_MODEL"
```

The check should print `RESULT: READY` and `[PASS] Model directory`.

| RWKV-7 | Qwen3.5 | RWKV active params | Qwen active params |
|---|---|---:|---:|
| 0.4B | 0.8B | 0.451B | 0.752B |
| 1.5B | 2B | 1.527B | 1.882B |
| 2.9B | 4B | 2.948B | 4.206B |
| 7.2B | 9B | 7.199B | 8.954B |

## 4. Quick RTX 4090 run: 1.5B versus 2B, B8

Create isolated output and compilation caches. Remove prior output so JSONL
rows are never appended across runs.

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OUT=/tmp/rwkv-qwen35-repro-4090
rm -rf "$OUT"
mkdir -p "$OUT"/{rwkv-cache,qwen-cache}
```

Run RWKV:

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
  --model "$RWKV_MODEL" --model-kind rwkv --model-role candidate \
  --model-pair rwkv-1.5b__qwen3.5-2b --model-size-label 1.5b \
  --benchmark-matrix external_repro_v1 --optimization-lane best_optimized_hf \
  --dtype fp16 --quantization none --device cuda --batch-sizes 8 \
  --prompt-tokens 128 512 2048 --decode-tokens 128 512 \
  --prefill-chunk-size 512 --warmup 3 --runs 7 \
  --rwkv-attn-mode fused_recurrent --rwkv-code-source repo \
  --rwkv-implementation auto --fail-fast --results "$OUT/rwkv.jsonl"
```

Clear RWKV route variables, then run Qwen on the same GPU and runtime:

```bash
while IFS= read -r name; do unset "$name"; done \
  < <(compgen -e | grep '^RWKV7_' || true)

TORCHINDUCTOR_CACHE_DIR="$OUT/qwen-cache" \
TRITON_CACHE_DIR="$OUT/qwen-cache" \
python bench/bench_cross_model_speed_resident.py \
  --model "$QWEN_MODEL" --model-kind qwen35 --model-role reference \
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

The first Qwen run compiles and captures its graph before timed measurements.

## 5. Summarize and verify

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

The formal RTX 4090 B8 medians for this pair are approximately:

| | RWKV | Qwen | Raw ratio | Parameter-adjusted ratio |
|---|---:|---:|---:|---:|
| Prefill | 57,929 tok/s | 36,953 tok/s | 1.568x | 1.272x |
| Decode | 1,909 tok/s | 1,302 tok/s | 1.468x | 1.191x |

Use the six raw JSONL cells when comparing runs rather than reverse-calculating
from rounded documentation values.

## 6. Full formal matrices

The full matrix covers all four pairs (the RTX 4080 covers the first three),
B1/B8, P128/512/2048, and D128/512. Formal runners lock the exact GPU,
runtime, model hashes, backend telemetry, and correctness evidence.

| GPU | Formal entry point | Result and exact environment |
|---|---|---|
| V100 | [`run_v100_qwen35_paired_pd_v1.sh`](../bench/run_v100_qwen35_paired_pd_v1.sh) | [P+D v1](../bench/v100_qwen35_paired_pd_v1_20260814/README.md) |
| RTX 3090 | [`run_3090_rwkv_paired_pd_v2.sh`](../bench/run_3090_rwkv_paired_pd_v2.sh), [`run_3090_qwen35_best_optimized_hf.sh`](../bench/run_3090_qwen35_best_optimized_hf.sh), and [`validate_qwen35_3090_paired_pd_v2.py`](../bench/validate_qwen35_3090_paired_pd_v2.py) | [P+D v2](../bench/3090_qwen35_paired_pd_v2_20260816/README.md) |
| RTX 4080 | [`run_4080_qwen35_paired_pd_v1.sh`](../bench/run_4080_qwen35_paired_pd_v1.sh) | [P+D v1](../bench/4080_qwen35_paired_pd_v1_20260814/README.md) |
| RTX 4090 | [`run_4090_rwkv_paired_pd_v2.sh`](../bench/run_4090_rwkv_paired_pd_v2.sh), [`run_5090_qwen35_best_optimized_hf.sh`](../bench/run_5090_qwen35_best_optimized_hf.sh), and [`validate_qwen35_4090_paired_pd_v2.py`](../bench/validate_qwen35_4090_paired_pd_v2.py) | [P+D v2](../bench/4090_qwen35_paired_pd_v2_20260815/README.md) |
| RTX 5090 | [`run_5090_rwkv_paired_decode_v1.sh`](../bench/run_5090_rwkv_paired_decode_v1.sh), [`run_5090_qwen35_best_optimized_hf.sh`](../bench/run_5090_qwen35_best_optimized_hf.sh), and [`validate_qwen35_paired_decode_v1.py`](../bench/validate_qwen35_paired_decode_v1.py) | [Decode v1](../bench/5090_qwen35_paired_decode_v1_20260813/README.md) |

Example full RTX 4080 run:

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

# OUT_DIR and CACHE_ROOT must not exist before the run.
bash bench/run_4080_qwen35_paired_pd_v1.sh
cat "$OUT_DIR/exit_code.txt"
cat "$OUT_DIR/paired_validation.json"
```

Open the result README for the target card first and match its runtime lock and
model variables. Use a separate Inductor/Triton cache per model. The exact
extension sources can be built with:

```bash
export FLA_SOURCE_COMMIT=2e38c1fab332174d056928feaf29f8c5fd5ac550
export CAUSAL_CONV1D_SOURCE_COMMIT=4f6ae4e26ae5fe8af9372f8d312ab25cc4595223
bash bench/build_hf_fast_path_v1_extensions.sh
```

## 7. Troubleshooting

- **Qwen fast path fails:** every row must have
  `qwen_fast_path_verified=true`, `qwen_full_fused_contract_pass=true`,
  `qwen_conv_backend_effective=causal_conv1d`, and
  `qwen_decode_cuda_graph_verified=true`.
- **Duplicate rows:** remove old JSONL files and caches before rerunning. Formal
  runners refuse an existing `OUT_DIR`.
- **Out of memory:** stop other GPU processes and set
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Compilation failure or a slowdown after changing cards:** remove that card's
  `TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR`. Never share compiled caches
  across GPU, PyTorch, or Triton versions.
- **Noisy measurements:** keep the GPU idle and thermally stable, retain 3
  warmups and 7 measured runs, and compare medians plus raw per-cell JSONL.

For a public reproduction, retain `git rev-parse HEAD`, `nvidia-smi`,
`pip freeze`, model hashes, both raw JSONL files, complete logs, and the summary.
