# RWKV-7 HF Adapter — Benchmark Target

This file is the persistent benchmark contract for the RWKV-7 HF adapter work.
The goal is to iterate until the HF path approaches the official `rwkv` package
and Albatross-style paths in correctness, speed, and memory.

## V100 production-close milestone (2026-07-11)

The exact-sm70 native path now passes the canonical 0.1B/0.4B/1.5B ×
bsz1/2/4/8 matrix. Dense decode and prompt-512 prefill reach
`0.908x–1.248x` and `0.930x–1.047x` of the same-host Albatross references.
Native W8/W4 speed policy reduces payload to `0.803x–0.956x`, reaches
`1.006x–1.128x` fp16 decode, and stays within `0.996x–1.007x` of fp16
prefill under paired same-process CUDA-event timing. All correctness, cache
handoff, generation, and focused regression gates pass.

Canonical rows, method, commands and the fail-closed gate are in
[`bench/v100_production_close_20260711/README.md`](bench/v100_production_close_20260711/README.md).
For benchmark directory layout, evidence naming, and the current script/directory inventory, see [`bench/README.md`](bench/README.md) and [`bench/INDEX.md`](bench/INDEX.md).

## Hardware currently measured

- Development server: **Tesla V100-PCIE-32GB**, CUDA fp16.
- A100 validation server: **NVIDIA A100-PCIE-40GB**, fp16/bf16.
- Local dev box baseline from earlier PR: **NVIDIA RTX 5070 Laptop GPU**, fp16/bf16/fp32.
- Ada validation server: **NVIDIA GeForce RTX 4090 24GB (sm_89)**, CUDA 12.8.
- Pascal validation box: **4 x NVIDIA GeForce GTX 1080 Ti (sm_61)**, fp16/quant smoke on one GPU.
- Baseline model: **rwkv7-g1d-0.1b-20260129-ctx8192**.
- A100 large-model validation: **0.4B / 1.5B / 2.9B / 7.2B**.
- Apple/Qwen3.5 comparison lane: see [QWEN35_APPLE_BASELINE.md](docs/hardware/QWEN35_APPLE_BASELINE.md) for the same-prompt Ollama Qwen3.5 vs RWKV-7 MLX/CoreML JSONL schema.
- **Ascend 910B2C 64GB** (华为昇腾, CANN 8.5.1, torch_npu 2.9.0rc1) — fla-free native 后端,详见 [rwkv7-hf-adapter-ascend](https://github.com/123123213weqw/rwkv7-hf-adapter-ascend) 仓库及下文 § Ascend 910B。

## Current GTX 1080 Ti / Pascal Status

Pascal validation was run on 2026-07-03 on one GTX 1080 Ti (`CUDA_VISIBLE_DEVICES=0`)
using the converted 0.1B HF model at `/tmp/rwkv7-g1d-0.1b-hf-pascal.current`.
The source checkpoint was
`/data/zhiyuanzhou/rwkv7-g1d-0.1b-20260129-ctx8192.pth`. An optional 0.4B
fp16 speed row was also run from
`/data/zhiyuanzhou/rwkv7-g1d-0.4b-20260210-ctx8192.pth`.

Environment:

- GPU: 4 x NVIDIA GeForce GTX 1080 Ti; validation used 1 GPU, `sm_61`.
- Driver / CUDA: NVIDIA driver `550.127.05`, `nvidia-smi` CUDA `12.4`.
- Runtime: Python `3.10.12`, PyTorch `2.7.1+cu118` (`torch.version.cuda=11.8`), Transformers `5.12.1`, bitsandbytes `0.49.2`, FLA `0.5.1`.
- Model: 0.1B, `hidden_size=768`, `num_hidden_layers=12`, `num_heads=12`, `head_dim=64`, dtype `fp16`; optional 0.4B fp16 speed row, `hidden_size=1024`, `num_hidden_layers=24`, `num_heads=16`, `head_dim=64`.
- Policy: Pascal defaults to compatibility-first native/no-FLA loading. The unpatched default FLA wrapper failed on this card because Triton emitted PTX using `.evict_last`, which requires `sm_70+`. The current default route selects `NativeRWKV7ForCausalLM` for Pascal unless `RWKV7_NATIVE_MODEL` explicitly overrides it.

Command:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=0 \
  PYTHON_BIN=/tmp/rwkv7-pascal-venv/bin/python \
  MODEL=/tmp/rwkv7-g1d-0.1b-hf-pascal.current DEVICE=cuda DTYPE=fp16 \
  bash scripts/run_hardware_smoke.sh
```

Additional benchmark commands:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. PYTHONNOUSERSITE=0 \
  /tmp/rwkv7-pascal-venv/bin/python bench/bench_quantization.py \
  --hf-dir /tmp/rwkv7-g1d-0.1b-hf-pascal.current \
  --model-size-label 0.1b \
  --dtype fp16 --device cuda --attn-mode fused_recurrent \
  --quantizations none 8bit 4bit \
  --prompt-tokens 128 --decode-tokens 16 \
  --warmup 1 --runs 1 --decode-mode compare \
  --results bench/results.jsonl

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. PYTHONNOUSERSITE=0 \
  /tmp/rwkv7-pascal-venv/bin/python bench/bench_native_mm_quant_decode.py \
  --hf-dir /tmp/rwkv7-g1d-0.1b-hf-pascal.current \
  --model-size-label 0.1b \
  --dtype fp16 --device cuda \
  --quantizations mm8 mm4 \
  --min-params 8000000 \
  --prompt-tokens 128 --decode-tokens 16 \
  --warmup 1 --runs 1 --optional \
  --results bench/results.jsonl

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. PYTHONNOUSERSITE=0 \
  /tmp/rwkv7-pascal-venv/bin/python bench/bench_batch_sweep.py \
  --hf-dir /tmp/rwkv7-g1d-0.1b-hf-pascal.current \
  --model-size-label 0.1b \
  --dtype fp16 --device cuda --attn-mode fused_recurrent \
  --fuse-norm auto --fast-cache auto --fast-token-backend auto \
  --batch-sizes 4 \
  --prompt-tokens 128 --decode-tokens 16 \
  --warmup 1 --runs 1 \
  --results bench/results.jsonl

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. PYTHONNOUSERSITE=0 \
  /tmp/rwkv7-pascal-venv/bin/python scripts/convert_rwkv7_to_hf.py \
  --input /data/zhiyuanzhou/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
  --output /tmp/rwkv7-g1d-0.4b-hf-pascal.current \
  --vocab-file /tmp/rwkv_vocab_v20230424.txt \
  --precision fp16 --attn-mode chunk --no-fuse-norm

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. PYTHONNOUSERSITE=0 \
  /tmp/rwkv7-pascal-venv/bin/python bench/bench_speed.py \
  --hf-dir /tmp/rwkv7-g1d-0.4b-hf-pascal.current \
  --model-size-label 0.4b \
  --backend hf --dtype fp16 --device cuda \
  --prompt-tokens 128 --decode-tokens 16 \
  --warmup 1 --runs 1 \
  --attn-mode fused_recurrent --fuse-norm auto \
  --fast-cache auto --fast-token-backend auto \
  --results bench/results.jsonl
```

Smoke status:

| Check | Result |
|---|---|
| `smoke_hf_generate` | PASS; logits shape `(1, 8, 65536)`, generated `User: Hello!\n\nAssistant: Hello! I'm` |
| `test_hf_api_contract --dtype fp16` | PASS |
| `test_quantized_inference` 8-bit / 4-bit | PASS; footprint `283.4 MB` / `242.9 MB`, peak `314.1 MB` / `278.3 MB` |
| `bench_quantization.py --quantizations none 8bit 4bit` | PASS; appended W8/W4 decode speed + footprint rows to `bench/results.jsonl` |
| `bench_native_mm_quant_decode.py --quantizations mm8 mm4` | PASS; native mm8/mm4 each replace 1 module (`lm_head`) and append decode speed + footprint rows |
| `bench_speed.py` | PASS; 0.1B fp16 and optional 0.4B fp16 rows appended to `bench/results.jsonl` |
| `bench_batch_sweep.py --batch-sizes 1 2 4` | PASS; appended to `bench/results.jsonl` |
| Training | Not run for this Pascal smoke |
| Turing | Still TODO |

Single-model speed rows in `bench/results.jsonl`, `prompt_tokens=128`,
`decode_tokens=16`, `fast_cache=true`, `cache_type=NativeRWKV7Cache`:

| Model | Dtype | Prefill tok/s | Forward decode tok/s | Decode ms/tok | Peak VRAM |
|---|---|---:|---:|---:|---:|
| 0.1B | fp16 | 46.4 | 95.2 | 10.50 | 406.8 MB |
| 0.4B | fp16 | 23.3 | 48.8 | 20.48 | 906.0 MB |

Quantization speed rows from `bench_quantization.py`, 0.1B, `prompt_tokens=128`,
`decode_tokens=16`, `decode_mode=compare`, `quant_skip_policy=memory`:

| Load | Decode tok/s | Model footprint | Peak VRAM | Notes |
|---|---:|---:|---:|---|
| fp16 / none | 70.4 | 364.4 MB | 478.8 MB | Same-script baseline |
| W8 / bitsandbytes | 12.7 | 283.4 MB | 399.1 MB | `0.18x` same-script fp16; `0.13x` 0.1B `bench_speed.py` fp16 |
| W4 / bitsandbytes | 27.2 | 242.9 MB | 362.9 MB | `0.39x` same-script fp16; `0.29x` 0.1B `bench_speed.py` fp16 |

Repository-native mm quantization rows from `bench_native_mm_quant_decode.py`,
0.1B, `prompt_tokens=128`, `decode_tokens=16`, `min_params=8_000_000`:

| Load | Replaced modules | Decode tok/s | Model footprint | Peak VRAM | Notes |
|---|---:|---:|---:|---:|---|
| native mm8 | 1 (`lm_head`) | 88.2 | 316.6 MB | 1036.9 MB | `0.93x` 0.1B `bench_speed.py` fp16; model path uses the naive mm8 GEMV, while the split-K mm8 microbench hits a Pascal PTX `.acq_rel` / `sm_70+` requirement |
| native mm4 | 1 (`lm_head`) | 89.3 | 292.6 MB | 1045.5 MB | `0.94x` 0.1B `bench_speed.py` fp16 |

Batch sweep rows:

| Batch | Prefill tok/s total | Prefill tok/s per seq | Decode tok/s total | Decode tok/s per seq | Peak VRAM |
|---:|---:|---:|---:|---:|---:|
| 1 | 29.9 | 29.9 | 93.1 | 93.1 | 1245.2 MB |
| 2 | 52.1 | 26.0 | 169.0 | 84.5 | 2108.0 MB |
| 4 | 99.6 | 24.9 | 351.2 | 87.8 | 3834.2 MB |

Pascal bnb quantization remains a memory/compatibility fallback. GTX 1080 Ti has
no newer tensor-core path, and the current W8/W4 bnb rows are slower than fp16.
The repository-native mm8/mm4 path is usable for this 0.1B shape and preserves
near-fp16 decode while reducing model footprint, but only `lm_head` crosses the
default `8_000_000` parameter gate. Broader quant promotion still needs
card-local rows on larger shapes where more projections are actually quantized.

## Current RTX 4090 / Ada status

Issue #66 (`RTX 4090 / Ada — HF 适配验证`) is validated on the 0.4B HF model
using repo remote code over `/workspace/models/rwkv7/rwkv7-g1d-0.4b-hf`.
Results are recorded in:

- `bench/results_4090_issue66_final_20260702_113804.jsonl`
- appended rows in `bench/results.jsonl`
- detailed summary in `bench/4090_validation_summary.md`

Environment: PyTorch `2.11.0+cu128`, CUDA `12.8`, Transformers `5.12.1`,
PEFT `0.19.1`, TRL `1.7.0`, bitsandbytes `0.49.2`, DeepSpeed `0.19.2`.

Issue #66 checklist status:

| Area | Result |
|---|---|
| HF generate smoke | PASS (`native_graph` fast-token backend) |
| HF API contract | PASS, fp16 + bf16 |
| Quantized inference | PASS, W8 + W4; quantized fast-forward resolves to FLA |
| Speed benchmark | PASS, fp16 prefill `22,222.6 tok/s`, decode `376.7 tok/s` |
| Batch sweep | PASS, bsz 1/2/4 decode `377.0` / `549.8` / `1,138.0 tok/s` |
| PEFT LoRA | PASS, non-zero LoRA grads |
| HF Trainer / TRL SFT | PASS, trainable delta ≈ `1e-4` |
| TRL DPO | PASS, trainable delta ≈ `1e-4` |

The exact-4090 policy now promotes fixed-shape native prefill CUDA Graph replay
with split recurrent scan, fused state prep/output, a no-`cat` sequence
attention/FFN shift-mix, and a one-launch ReLU² kernel. On 0.4B fp16 / prompt512
it reaches `64,511.2 tok/s` at bsz1 and `107,870.1 tok/s` at bsz4. A same-host,
same-session Albatross rerun reaches `107,149.6 tok/s` at bsz4, so the current
ratio is **`1.007x`**. Against the strongest older recorded Albatross bsz4 row
(`117,789.0 tok/s`) the conservative ratio is **`0.916x`**; both references are
retained rather than silently replacing the historical high-water mark. All HF
rows pass logit greedy equality and prefill-to-decode cache handoff. The same
public path previously reached `32,357.8 tok/s` on 1.5B / bsz1 / prompt512.
Other Ada cards retain the compatible fallback until card-local rows exist.

The serving API caches fixed `(batch, prompt_tokens)` graphs and exposes
`rwkv7_warmup_fast_prefill()` for cold-start preparation. `rwkv7_prefill_chunks`
can replay a captured chunk shape while carrying `RWKV7StateCache`; full-vs-two
chunk prompt1024 and the following decode token preserve greedy equality.

### RTX 4090 prefill sequence-fusion update (2026-07-11)

Profiling showed that the two per-layer sequence `torch.cat` operations and
separate FFN ReLU/power launches consumed a material part of the remaining B4
gap. The promoted graph path now computes previous-token addressing directly
inside Triton, writes the next attention/FFN shift state in the same launch, and
computes ReLU² in one graph-safe kernel. This removes 48 `CatArrayBatchedCopy`
launches per prefill replay without changing the public cache layout.

| bsz | prompt | latency | throughput | greedy/cache handoff |
|---:|---:|---:|---:|:---:|
| 1 | 128 | 3.5970 ms | 35,585.6 tok/s | PASS |
| 1 | 512 | 7.9366 ms | **64,511.2 tok/s** | PASS |
| 1 | 1024 | 16.1203 ms | 63,522.3 tok/s | PASS |
| 4 | 128 | 6.0730 ms | 84,308.0 tok/s | PASS |
| 4 | 512 | 18.9858 ms | **107,870.1 tok/s** | PASS |
| 4 | 1024 | 41.8619 ms | 97,845.6 tok/s | PASS |

The B4/prompt512 path improved from `20.0067 ms` / `102,365.8 tok/s` to
`18.9858 ms` / `107,870.1 tok/s` (**`1.054x`**). Under Nsight Systems, the
captured HF graph's summed kernel time is now about `18.8 ms`, slightly below
the same-session Albatross graph's approximately `18.9 ms`. Full HF forward,
`generate`, full-vs-chunked prefill (chunk sizes 1/4/16/32), following-token
decode, and dynamic cache select/reorder/compact checks pass.

Evidence on the validation host:

- `/data/rwkv4090/results/prefill_sequence_mix_matrix.jsonl`
- `/data/rwkv4090/results/prefill_sequence_fusion_final.jsonl`
- `/data/rwkv4090/results/prefill_relu2_block512.jsonl`
- `/tmp/hfprof20_seq.nsys-rep` and `/tmp/hfprof20_seq_stats.csv`
- `/tmp/albprof20.nsys-rep` and `/tmp/albprof20_stats.csv`

### RTX 4090 dense fp16 decode parity update (2026-07-11)

The exact-4090 dense `native_graph` policy now closes the recorded 0.4B
Albatross decode rows at every required active batch size. The promoted path
combines 8-warp norm/mix, copy-free stacked R/K/V input storage, grouped Ada
W/A/G/V low-rank kernels, and the sparse FFN route for bsz1/2. Sparse value
weights are packed outside graph capture and keyed by batch shape; this is
required because sharing one packed allocation across independently replayable
CUDA graphs corrupts later graph runners. Bsz4/8 retain the safe dense FFN
contraction selected by the exact-card policy.

| bsz | HF dense fp16 tok/s | Albatross fp16 tok/s | HF / Albatross | cumulative graph-cache peak VRAM |
|---:|---:|---:|---:|---:|
| 1 | **795.7** | 790.1 | **1.007x** | 1,414.0 MB |
| 2 | **1,469.5** | 1,445.8 | **1.016x** | 1,684.5 MB |
| 4 | **2,585.7** | 2,564.0 | **1.008x** | 1,807.1 MB |
| 8 | **3,185.3** | 2,246.0 | **1.418x** | 2,047.9 MB |

The peak column is intentionally cumulative: one process retains the B1/B2/B4/B8
graph runners to validate production dynamic-batch residency. It is not the
model payload size. A 32-step test with all four runners resident passes
`32/32` greedy equality for every batch, maximum logits absolute difference
`<=0.1875`, standard-HF fallback difference `<=0.09375`, and graph-cache
warmup/retention checks. Evidence on the validation host:

- `/data/rwkv4090/results/dense_albatross_parity_final_memory_fixed.jsonl`
- `/data/rwkv4090/results/dense_albatross_parity_b1_confirm.jsonl`
- `/data/rwkv4090/results/dense_multigraph_correctness_final.log`

Post-change quant regression remains non-negative: the W8 speed lane is
`1.057x` fp16 with `0.9258x` payload and cosine `0.99999398`; the W4 speed lane
is `1.046x` bf16 with `0.8907x` payload and cosine `0.99990374`. Both preserve
the next token after the prefill sequence-fusion change. Evidence:
`quant_w8_post_prefill_fusion.jsonl` and `quant_w4_post_prefill_fusion.jsonl`
in the same validation-host results directory.

```bash
PYTHONPATH=. python bench/bench_native_graph_overhead.py \
  --hf-dir /path/to/rwkv7-g1d-0.4b-fp16 --dtype fp16 --device cuda \
  --attn-mode fused_recurrent --batch-sizes 1 2 4 8 \
  --warmup 100 --steps 1000 --fixed-token --results bench/results.jsonl
```

```bash
PYTHONPATH=. python bench/bench_native_prefill_scan.py \
  --model /path/to/rwkv7-g1d-0.4b-hf --code-source repo \
  --device cuda --dtype fp16 --batch-sizes 1,4 --prompt-tokens 512 \
  --fused-scan auto --warmup 20 --steps 50 --results bench/results.jsonl
```

### RTX 4090 native-graph + TorchAO W4 update (2026-07-10)

The 0.4B model now has an optional tensor-core W4 lane through
`rwkv7_hf.native_quant_torchao`. It combines TorchAO group-128 packed W4
projections with the adapter's CUDA-graph decode and the fp16/bf16 Ada fused
W/A/G/V low-rank kernel. The run used bf16 activations because the current
CUDA `aten::_weight_int4pack_mm` contract requires bf16. All 145 large
projection/FFN/head modules selected by `min_params=1_000_000` were quantized.

| bsz | bf16 tok/s | TorchAO W4 tok/s | W4 / bf16 | Albatross fp16 tok/s | W4 / Albatross |
|---:|---:|---:|---:|---:|---:|
| 1 | 692.7 | **927.0** | **1.338x** | 790.1 | **1.173x** |
| 2 | 1,056.9 | **1,712.6** | **1.620x** | 1,445.8 | **1.185x** |
| 4 | 2,003.0 | **3,093.2** | **1.544x** | 2,564.0 | **1.206x** |
| 8 | 2,746.4 | **3,407.0** | **1.241x** | 2,246.0 | **1.517x** |

Packed model payload falls from `859.8 MB` to `342.8 MB` (`0.3987x`). Across
bsz 1/2/4/8, prompt-logit cosine is `0.999239-0.999344`, final-logit cosine is
`0.999460-0.999550`, and the next token matches the bf16 baseline. Evidence:
`/data/rwkv4090/results/torchao_w4_0.4b_b{1,2,4,8}_bf16ada.jsonl` on the
validation host.

The same lane generalizes to the 1.5B checkpoint rather than relying on the
0.4B shape: bsz1 improves from `267.2` to `580.5 tok/s` (`2.173x`) and bsz2
from `461.6` to `1,094.5 tok/s` (`2.371x`). Packed payload is `1,033.3 MB`
versus `2,913.3 MB` (`0.3547x`), final-logit cosine is `>=0.999400`, and both
rows preserve the next token. A matching 4090 Albatross 1.5B baseline has not
yet been produced, so these two rows are only W4-vs-bf16 evidence.

Reproduce with:

```bash
python bench/bench_native_quant_e2e_decode.py \
  --hf-dir /path/to/rwkv7-g1d-0.4b-hf \
  --dtype bf16 --device cuda --attn-mode chunk \
  --fast-token-backend native_graph \
  --quantizations none torchao_w4 \
  --min-params 1000000 --policy memory \
  --batch-size 1 --prompt-tokens 32 --decode-tokens 128 --warmup 8
```

This closes the maximum-memory-saving W4 **decode** lane for bsz 1/2/4/8. The
full-model W4 prefill graph reaches `26,652.9 tok/s`, `0.454x` the dense bf16
prefill graph, although it is `3.58x` faster than the same quantized HF/FLA
path. For applications
that require every inference phase to stay non-negative, the `speed` policy
quantizes `lm_head` only and uses the new prefill graph:

- native A8/W8 speed lane: payload `0.9258x`, prompt512 prefill `1.011x` fp16,
  decode ratios `1.001x/1.008x/1.020x/1.015x` at bsz1/2/4/8, prefill cosine
  `0.999995`, and greedy equality;
- TorchAO W4 speed lane: payload `0.8907x`, prompt512 prefill `1.010x` bf16,
  and decode `1.043x/1.058x` bf16 at measured bsz1/4. Real-prompt prefill and
  decode preserve the next token.

Thus `speed` is the all-phase non-regression lane with moderate memory saving;
`memory` is the large-footprint-reduction lane (`0.399x` payload) with much
faster decode but a remaining quantized-prefill kernel gap.

```bash
# Run once with --quantization none, then repeat with a8w8 or torchao_w4.
python bench/bench_native_prefill_scan.py \
  --model /path/to/rwkv7-g1d-0.4b-hf --code-source repo \
  --device cuda --dtype fp16 --batch-sizes 1 --prompt-tokens 512 \
  --quantization a8w8 --quant-policy speed --quant-min-params 1 \
  --warmup 50 --steps 50 --results bench/results.jsonl
```

Use `--dtype bf16 --quantization torchao_w4` for the W4 speed row.

## Ascend 910B status (华为昇腾 NPU)

独立仓库 [rwkv7-hf-adapter-ascend](https://github.com/123123213weqw/rwkv7-hf-adapter-ascend)(PR #2)。fla-free native 后端(`NativeRWKV7ForCausalLM`,纯 PyTorch + `torch_npu`),无需 CUDA/Triton/FLA。

- **硬件**: Ascend 910B2C, 64GB HBM, CANN 8.5.1, torch_npu 2.9.0rc1
- **C++ op-coalesced forward**(单序列基线): 12 层 TMix+CMix 收进一次 C++ 调用,消除 Python dispatch → **323 tok/s (B=1), cos=1.0**(bit-exact 对齐 Python)
- **batch decode 2× Albatross(aggregate 吞吐)**: launch 开销被 batch 摊薄 + 投影 GEMM 化,forward 时间 1→128 batch 只涨 ~3×。全规模 aggregate tok/s(B>1 已验证 cos=1.0):

| model | B=1 | B=16 | B=64 | B=128 |
|---|---|---|---|---|
| 0.1B | 323 | 3433 | 9446 | **13504** |
| 1.5B | 87 | 953 | 2121 | — |
| 2.9B | 69 | 680 | 1748 | — |
| 7B | 33 | 313 | 793 | — |
| 13B | 25 | 235 | 585 | — |

- 0.1B B=8/16/64/128 全部进/超 2× Albatross 区间(~1500–3000 aggregate);13B B=64 = 585(31GB/64GB)
- 复现 + 完整结果见 [rwkv7-hf-adapter-ascend/ASCEND_RESULTS.md](https://github.com/123123213weqw/rwkv7-hf-adapter-ascend/blob/wangyue/ascend-2x-albatross/ASCEND_RESULTS.md)
- 注:速度用**随机权重**(910B 服务器被墙下不到真实模型),速度数字有效、输出质量未验。单序列延迟(B=1, 25–323 tok/s)未达 2×(需 GEMV-Cube 融合,多月工程)。

## Acceptance targets for 0.1B smoke baseline

### 1. Precision

| Metric | Target |
|---|---:|
| top-5 token IDs match | 100% for fp32, high stability for fp16/bf16 |
| cosine similarity | >= 0.9999 |
| max abs logit diff | <= 0.05 for fp32 reference; dtype-aware for fp16/bf16 |
| greedy decode equality window | identical for >= 64 tokens |

### 2. Speed

| Metric | Target |
|---|---:|
| prefill tok/s | HF >= 0.9 x official comparable path |
| decode tok/s | HF >= 0.9 x official comparable path |

### 3. Memory

| Metric | Target |
|---|---:|
| peak VRAM | HF <= 1.1 x official comparable path |

## Current A100 / Ampere status

The initial issue #68 A100 0.1B baseline was run on 2026-07-02 on `gpu03`
and merged in #82.

Environment:

- GPU: 8 x NVIDIA A100-PCIE-40GB; inference used 1 GPU, ZeRO used 2 GPUs.
- Driver / CUDA: NVIDIA driver `570.133.20`, `nvidia-smi` CUDA `12.8`.
- Runtime: Python `3.12.8`, PyTorch `2.8.0+cu126` (`torch.version.cuda=12.6`), Transformers `4.57.1`, PEFT `0.19.1`, TRL `1.7.0`, DeepSpeed `0.19.2`, bitsandbytes `0.49.2`, FLA `0.5.1`.
- Model: converted HF `rwkv7-g1d-0.1b-20260129-ctx8192`, `fused_recurrent`, `fuse_norm=false`, `RWKV7StateCache`, `RWKV7_FAST_TOKEN_BACKEND=auto`.
- Note: the first FLA backward pass compiled slowly on the 4.18 kernel; later steps reused the compiled path.

Representative commands:

```bash
DTYPE=fp16 DEVICE=cuda FUSE_NORM=false FAST_CACHE=true FAST_TOKEN_BACKEND=auto \
  BATCH_SIZES="1 2 4 8 16 32" PROMPT_TOKENS=512 DECODE_TOKENS=128 WARMUP=2 RUNS=3 \
  bash scripts/run_hardware_smoke.sh "$MODEL"

DTYPE=bf16 DEVICE=cuda RUN_QUANT=0 FUSE_NORM=false FAST_CACHE=true FAST_TOKEN_BACKEND=auto \
  BATCH_SIZES="1 2 4 8 16 32" PROMPT_TOKENS=512 DECODE_TOKENS=128 WARMUP=2 RUNS=3 \
  bash scripts/run_hardware_smoke.sh "$MODEL"

TRAIN_DTYPE=bf16 DEVICE=cuda MAX_LENGTH=32 MAX_STEPS=1 DATASET_REPEATS=2 \
  RUN_PEFT=0 RUN_TRAINER=1 RUN_RL=1 RL_BACKEND=both \
  bash scripts/run_hf_training_matrix.sh "$MODEL"

CUDA_VISIBLE_DEVICES=0,1 TRAIN_DTYPE=bf16 NPROC_PER_NODE=2 ZERO_STAGE=both \
  MAX_LENGTH=32 MAX_STEPS=1 DATASET_REPEATS=2 \
  bash scripts/run_zero_training_smoke.sh "$MODEL"
```

Smoke status:

| Check | Result |
|---|---|
| `smoke_hf_generate` | PASS; `generate_fast_token_backend native_graph` |
| `test_hf_api_contract --dtype bf16` | PASS |
| `test_quantized_inference` 8-bit / 4-bit | PASS on fp16; footprint `283.4 MB` / `242.9 MB`, peak `310.6 MB` / `273.3 MB` |
| `test_peft_lora` | PASS; `663552` trainable parameters, finite loss, `72` non-zero LoRA gradients |

Single-model speed rows in `bench/results.jsonl`:

| Dtype | Prefill tok/s | Forward decode tok/s | Decode ms/tok | Peak VRAM |
|---|---:|---:|---:|---:|
| fp16 | 19538.7 | 52.9 | 18.90 | 660.3 MB |
| bf16 | 19562.5 | 59.5 | 16.82 | 631.1 MB |

A100 serving-style batch sweep, `rwkv7_forward_token`, `fast_token_backend_effective=native_graph`:

| Batch | fp16 decode tok/s | bf16 decode tok/s | bf16 prefill tok/s | Peak VRAM |
|---:|---:|---:|---:|---:|
| 1 | 368.5 | 372.8 | 13914.3 | 727.1 MB |
| 2 | 618.7 | 691.6 | 28281.7 | 1114.0 MB |
| 4 | 1282.3 | 1333.8 | 54674.3 | 1819.8 MB |
| 8 | 2591.1 | 2500.8 | 103898.8 | 3263.0 MB |
| 16 | 5694.9 | 4974.8 | 121949.7 | 6112.5 MB |
| 32 | 10376.9 | 9966.4 | 124579.8 | 11818.9 MB |

Training and RL rows:

| Backend | Dtype | Status | Loss | Runtime | Trainable delta |
|---|---|---|---:|---:|---:|
| HF Trainer | bf16 | PASS | 1.7299 | 311.6833 s | `1.0e-4` |
| TRL SFT | bf16 | PASS | 1.6520 | 0.2667 s | `1.0e-4` |
| TRL DPO | bf16 | PASS | 0.6931 | 1.5805 s | `1.0e-4` |
| TRL GRPO | bf16 | PASS | 0.0000 | 43.2822 s | `1.0e-4` |

DeepSpeed ZeRO rows, 2 x A100, `world_size=2`:

| ZeRO stage | Dtype | Status | Loss | Rank-0 runtime | Rank-0 trainable delta |
|---:|---|---|---:|---:|---:|
| 2 | bf16 | PASS | 4.8672 | 66.3160 s | `1.001e-4` |
| 3 | bf16 | PASS | 4.8672 | 0.7241 s | `1.0e-4` |

### A100 extended large-model validation

The follow-up issue #68 A100 40GB pass added 0.4B / 1.5B / 2.9B / 7.2B
evidence for smoke generation, fp16/bf16 batch sweeps, quantized speed/memory,
single-GPU Trainer/SFT/DPO, HF Trainer checkpoint resume, 2 x A100 ZeRO-2/3
base smoke, and 2 x A100 ZeRO-2 checkpoint resume. Detailed environment,
commands, model hashes, and tables are recorded in
[`docs/validation/A100_HF_VALIDATION.md`](docs/validation/A100_HF_VALIDATION.md).

Large-model smoke rows:

| Model | Layers | Hidden | Footprint | Peak VRAM | Status |
|---|---:|---:|---:|---:|---|
| 0.4B | 24 | 1024 | 859.8 MB | 1124.5 MB | PASS |
| 1.5B | 24 | 2048 | 2913.3 MB | 3178.6 MB | PASS |
| 2.9B | 32 | 2560 | 5622.4 MB | 5888.0 MB | PASS |
| 7.2B | 32 | 4096 | 13731.3 MB | 13997.8 MB | PASS |

A100 fast-token batch sweep, `rwkv7_forward_token`, `native_graph`,
`prompt_tokens=128`, `decode_tokens=16`:

| Model | Dtype | Batches | Batch-1 decode tok/s | Max-batch decode tok/s | Max-batch peak VRAM |
|---|---|---|---:|---:|---:|
| 0.4B | fp16 | 1,2,4,8 | 147.0 | 1539.8 | 2867.4 MB |
| 0.4B | bf16 | 1,2,4,8 | 146.5 | 1530.8 | 2867.4 MB |
| 1.5B | fp16 | 1,2,4 | 164.5 | 578.2 | 4904.1 MB |
| 1.5B | bf16 | 1,2,4 | 164.9 | 552.6 | 4904.1 MB |
| 2.9B | fp16 | 1,2 | 101.8 | 189.1 | 7261.5 MB |
| 2.9B | bf16 | 1,2 | 78.5 | 166.2 | 7261.5 MB |
| 7.2B | fp16 | 1,2 | 59.2 | 117.2 | 16336.1 MB |
| 7.2B | bf16 | 1,2 | 58.5 | 117.1 | 16336.1 MB |

Quantized A100 memory and interim decode telemetry:

| Model | fp16 footprint / decode | 8bit footprint / decode (interim) | 4bit footprint / decode (interim) |
|---|---:|---:|---:|
| 0.4B | 859.8 MB / 144.8 tok/s | 571.8 MB / 12.3 tok/s | 427.8 MB / 25.3 tok/s |
| 1.5B | 2913.3 MB / 119.5 tok/s | 1761.3 MB / 11.5 tok/s | 1185.3 MB / 25.0 tok/s |
| 2.9B | 5622.4 MB / 73.5 tok/s | 3222.4 MB / 8.9 tok/s | 2022.4 MB / 19.2 tok/s |
| 7.2B | 13731.3 MB / 61.4 tok/s | 7587.3 MB / 7.0 tok/s | 4515.3 MB / 15.3 tok/s |

All W8/W4 rows reduce memory. Their decode-speed fields are marked
`quant_speed_status=interim` in `bench/results.jsonl` because the native-fused
packed-quant / tensor-core-aware kernel work is expected to replace these
generic bitsandbytes speed numbers; they are still slower than
fp16/native-graph and therefore remain part of the fused/native quantization
performance gap.

Training and resume coverage:

| Model | Single-GPU Trainer/SFT/DPO | HF checkpoint resume | ZeRO-2 base | ZeRO-2 resume | ZeRO-3 base | ZeRO-3 resume |
|---|---|---|---|---|---|---|
| 0.1B | smoke | smoke | smoke | smoke | smoke | V100 2-GPU PASS |
| 0.4B | PASS | PASS | PASS | PASS | PASS | pending scale-up |
| 1.5B | PASS | PASS | PASS | PASS | PASS | pending scale-up |
| 2.9B | PASS | PASS | PASS | PASS | PASS | pending scale-up |
| 7.2B | PASS | PASS | PASS | PASS | PASS | pending scale-up |

The A100 40GB validation block brings `bench/results.jsonl` to 134 A100 rows:
68 batch-sweep rows, 20 DeepSpeed base rows, 16 single-GPU training rows, 12
quantization rows including 8 W8/W4 rows with interim speed status, 8
DeepSpeed resume rows, 4 large-model smoke rows, 4 HF checkpoint-resume rows,
and the 2 legacy 0.1B speed rows from #82. A100 80GB was not available in the
current cluster. A follow-up V100 run closed the initial ZeRO3 checkpoint
resume smoke on 0.1B native/HF (`bench/results_v100_zero3_resume_2gpu_20260703.jsonl`);
the remaining work is scaling that same ZeRO3-resume proof to 0.4B+ and
rechecking the A100 large-model dtype-mismatch path.

## Current V100 status

Latest V100 runs are appended in `bench/results.jsonl`. The HF training /
quant / ZeRO matrix from 2026-07-02 is summarized in
[`docs/validation/V100_HF_VALIDATION.md`](docs/validation/V100_HF_VALIDATION.md): 0.4B/1.5B pass the
Trainer/SFT/DPO/GRPO/PEFT/ZeRO/quant smoke matrix, 2.9B passes the
native TRL/PEFT/ZeRO2-resume/ZeRO3-base/quant matrix, and 7.2B passes
PEFT plus 8/4-bit quantized inference within V100 memory limits.

### Correctness / precision

Command:

```bash
python tests/test_official_alignment.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --pth /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-20260129-ctx8192.pth \
  --dtype fp16 \
  --device cuda \
  --official-strategy 'cpu fp32' \
  --greedy-window 64 \
  --fuse-norm false \
  --results bench/results.jsonl
```

Result on Tesla V100:

| Metric | Result | Status |
|---|---:|---|
| top5_match | 1.0000 | PASS |
| argmax_match | 1.0000 | PASS |
| cosine | 0.9999977 | PASS |
| max_abs_diff | 0.0718 | PASS for fp16 smoke; fp32 reference remains ≈0.030 |
| greedy window | 64 / 64 tokens | PASS |

Earlier fp32 reference on the 5070 Laptop produced `max_abs_diff≈0.030`, proving
that the adapter math and weight mapping are correct when dtype noise is removed.
The V100 optimized path uses `fuse_norm=false`; it preserves top-k/greedy behavior
and improves fp16 max-abs error versus the FLA fused-norm path.

### Save/reload roundtrip

Command:

```bash
python tests/test_reload_roundtrip.py \
  --model /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --device cuda \
  --dtype fp16
```

Result:

| Metric | Result | Status |
|---|---:|---|
| reloaded logits max_abs_diff | 0.0 | PASS |

### High-level speed/memory, serving-style HF prefill

`bench/bench_speed.py` now measures HF prefill with `use_cache=True` and
`logits_to_keep=1`, which matches serving needs and avoids retaining full prompt
logits. The HF path now uses the adapter remote-code class and the lightweight
`RWKV7StateCache` hot path by default (`RWKV7_FAST_CACHE=1`).

Command:

```bash
python bench/bench_speed.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --pth /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-20260129-ctx8192.pth \
  --backend both \
  --dtype fp16 \
  --prompt-tokens 512 \
  --decode-tokens 128 \
  --device cuda \
  --warmup 2 \
  --runs 3 \
  --hf-logits-to-keep 1 \
  --fuse-norm false \
  --fast-cache true
```

Result on Tesla V100:

| Backend | Prefill tok/s | Decode tok/s | Decode ms/tok | Peak VRAM |
|---|---:|---:|---:|---:|
| HF adapter, `fuse_norm=true` | 11852.0 | 31.5 | 31.70 | 406.4 MB |
| HF adapter, `fuse_norm=false` | 14247.7 | 41.3 | 24.24 | 406.4 MB |
| HF adapter, `fuse_norm=false`, `RWKV7StateCache` | 13801.4 | 41.2 | 24.28 | 406.4 MB |
| HF adapter, `rwkv7_forward_token` | 14055.1 | 59.2 | 16.89 | 406.4 MB |
| HF adapter, `rwkv7_forward_token`, `native_jit` backend | 13755.4 | 92.1 | 10.86 | 406.4 MB |
| HF adapter, `rwkv7_forward_token`, `native_graph` backend | 18386.6 | 255.5 | 3.91 | 643.7 MB |
| official `rwkv` | 225.6 | 92.1 | 10.86 | 406.2 MB |

Interpretation:

- **Memory target is met** for the 0.1B V100 serving-style path: HF is roughly equal to official.
- HF prefill is much faster than the official pure-torch reference path measured here.
- Disabling FLA fused norm for inference improved HF decode from `31.5` to about `41` tok/s (`+31%`).
- The lightweight `RWKV7StateCache` preserves exact logits/cache behavior and keeps the real remote-code `AutoModelForCausalLM` path at the same ~41 tok/s level while avoiding FLA CacheLayer bookkeeping.
- `RWKV7StateCache.select_batch` / `batch_select` now gives serving stacks a
  direct dynamic-batch compact/drop API; `reorder_cache` remains as the HF beam
  compatibility hook. `RWKV7StateCache.rwkv7_cache_metrics()` exposes
  update/select/reorder/offload counters and current cache shape telemetry.
- `RWKV7StateCache.detach()` and `to(device, dtype=None)` cover serving state
  offload/restore. V100 dynamic cache tests now compact active rows, detach the
  cache, move it to CPU, restore it to CUDA, and verify the next logits.
- **bsz=1 decode target is met** with the opt-in `native_jit` fast-token backend:
  standard optimized HF decode is about `0.45x` official, FLA fast-token reaches
  about `0.64x` official, and `RWKV7_FAST_TOKEN_BACKEND=native_jit` reaches
  `1.00x` official on this V100 run.
- `RWKV7_FAST_TOKEN_BACKEND=native_graph` moves the standalone CUDA-graph
  prototype into the HF `rwkv7_forward_token` API for fixed bsz and dynamic
  active-batch serving: bsz=1 reaches `255.5 tok/s` (`2.77x` official), with
  bsz=1/2/4/8 batch sweep rows shown below. Captured graph runners are kept in a
  per-model LRU controlled by `RWKV7_NATIVE_GRAPH_CACHE_SIZE`; serving code can
  call `rwkv7_clear_native_graph_cache()` to release retained graph buffers. The
  formal memory target remains anchored to the lower-memory native-JIT row.
  Native-graph replay overhead rows also record cache requests, hits, misses,
  evictions, retained batch sizes, and hit rate so serving cache reuse is a
  gated metric rather than an undocumented implementation detail.
- `RWKV7_FAST_TOKEN_BACKEND=auto` now resolves the effective fast-token backend
  at runtime as `native_graph` -> `native_jit` -> FLA, gated by CUDA/model
  placement, available native helpers, active batch size, and dense
  non-bitsandbytes weights. Benchmark scripts set the env var even when
  `--fast-token-backend auto` is used and write
  `fast_token_backend_effective` for regression analysis.
- `RWKV7_FAST_FORWARD=1` (default) routes ordinary eval/no-grad HF cached
  one-token `forward()` calls through `rwkv7_forward_token`, so
  `model.generate(..., use_cache=True)` gets the same auto-selected backend.
  Benchmark baseline loops explicitly set `RWKV7_FAST_FORWARD=0` around
  reference forward timing so historical forward-vs-fast comparisons stay
  comparable.

### Decode breakdown

Command:

```bash
python bench/bench_decode_breakdown.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --pth /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-20260129-ctx8192.pth \
  --dtype fp16 \
  --device cuda \
  --prompt-tokens 512 \
  --decode-tokens 128 \
  --warmup 2 \
  --runs 3 \
  --attn-modes chunk fused_recurrent \
  --fuse-norm false \
  --fast-cache true \
  --results bench/results.jsonl
```

Result on Tesla V100:

| Path | Prefill tok/s | Greedy decode tok/s | Fixed-token decode tok/s | Sampling overhead | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| HF `chunk`, `fuse_norm=true` | 11536.2 | 30.4 | 30.4 | 0.05 ms/tok | 439.7 MB |
| HF `chunk`, `fuse_norm=false` | 13343.7 | 38.2 | 38.0 | ≈0 ms/tok | 439.7 MB |
| HF `chunk`, `fuse_norm=false`, `RWKV7StateCache` | 13510.3 | 36.7 | 37.4 | 0.51 ms/tok | 439.7 MB |
| HF `fused_recurrent`, `fuse_norm=false` | 17192.8 | 38.3 | 38.2 | ≈0 ms/tok | 440.2 MB |
| HF `fused_recurrent`, `fuse_norm=false`, `RWKV7StateCache` | 17198.9 | 38.4 | 38.5 | 0.09 ms/tok | 440.2 MB |
| HF `fused_recurrent`, `rwkv7_forward_token` | 16571.8 | 52.9 | 53.0 | ≈0 ms/tok | 440.2 MB |
| official `rwkv` | 222.1 | 91.5 | n/a | n/a | 470.0 MB |

Interpretation:

- Greedy argmax/sampling overhead is negligible.
- `chunk` vs `fused_recurrent` does not materially change single-token decode.
- `fuse_norm=false` removes the expensive FLA `LayerNormFunction` path and improves decode, but does not remove the main gap.
- The fast token API reduces standard HF one-token decode from about `26 ms/token`
  to about `19 ms/token`, but the remaining gap is still inside the HF/FLA model
  + recurrent cache + per-token layer path, not in Python sampling.


### Decode profiler findings

Profiler commands were added via `bench/profile_decode.py`. On V100 fixed-token
decode, the original HF path spent most wall time in CPU dispatch/custom-function
overhead, not GPU math. The most important finding was:

- `fuse_norm=true`: FLA `LayerNormFunction` showed about `54.8 ms` CPU total over 6 active decode tokens.
- `fuse_norm=false`: native `aten::native_layer_norm` path reduced norm overhead to about `6.6 ms` CPU total over 6 active decode tokens.
- Result: high-level HF decode improved from `31.5` tok/s to `41.3` tok/s on V100.

The profile still shows thousands of tiny kernel launches per handful of decode
tokens, so the next optimization has to reduce/fuse the one-token layer path
rather than tune sampling.

## Reproducible V100 fast-decode validation

When the V100 server is reachable, run the committed bundle from the repository root:

```bash
./bench/run_v100_fast_decode_validation.sh
```

It runs `test_fast_decode_api.py`, `bench_speed.py --hf-decode-api rwkv7_forward_token`,
`test_batch_cache.py`, `test_dynamic_batch_cache.py`, `bench_batch_sweep.py`, `bench_dynamic_batch.py`, `bench_decode_breakdown.py --fast-decode-api true`, `bench_decode_micro.py`, `bench_forward_fast_path.py`, `bench_generate_fast_path.py`, `tests/test_device_map_generate.py` when at least two CUDA devices are visible, `bench_fast_token_warmup.py`, `bench_native_graph_overhead.py`, `bench_decode_components.py`, `bench_projection_lora.py`, `bench_fused_projection.py`, `bench_fused_wa_lora.py`, `bench_fused_wag_lora.py`, `bench_fused_rkv_wag_projection.py`, `bench_fused_attn_output.py`, `bench_fused_ffn.py`, `bench_fused_shift_mix.py`, `bench_fused_recurrent.py`, `bench_native_graph_fused_recurrent.py`, `bench_native_graph_fused_output.py`, `bench_native_quant_gemv.py`, `bench_native_quant_w4_gemv.py`, `bench_native_quant_rkv.py`, `bench_native_quant_w4_rkv.py`, `bench_larger_model_smoke.py` when the 0.4B/1.5B/2.9B/7.2B/13.3B paths exist, `bench_speculative_decode.py` when the target/draft HF dirs exist, `profile_decode.py --hf-decode-api rwkv7_forward_token`, `bench/analyze_results.py`, and `bench/check_results.py`,
then writes logs under `bench/logs/`. The bundle now also validates the
`native_jit` backend plus fixed-batch and dynamic `native_graph` fast-token
backends, and appends native HF speed rows before running the target gate. Use
`python bench/summarize_results.py --device V100
--last 12` for a compact view of the latest JSONL rows.

## Fast-token layout A/B harness

The validated fast-token path remains the default `3d` layout.  For candidate
one-token hot-path changes, the repository also includes an opt-in layout switch
and a V100 A/B bundle:

```bash
# Default baseline behavior.
RWKV7_FAST_TOKEN_LAYOUT=3d python bench/bench_speed.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --pth /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-20260129-ctx8192.pth \
  --backend both \
  --dtype fp16 \
  --hf-decode-api rwkv7_forward_token \
  --fast-token-layout 3d

# Run 3d vs experimental 2d correctness + speed + microbench rows.
./bench/run_v100_fast_token_layout_ab.sh

# Resume only the missing candidate side after an interrupted/flaky-SSH run.
LAYOUTS=2d SPEED_BACKEND=hf ./bench/run_v100_fast_token_layout_ab.sh

python bench/compare_fast_token_layouts.py --results bench/results.jsonl --device V100 --dtype fp16 --require-candidate --min-speedup 1.0
```

Rows without `fast_token_layout` are treated as `3d` by
`bench/compare_fast_token_layouts.py`, so older V100 results remain the baseline
until new A/B rows are appended. Candidate rows are not accepted as an
optimization until `tests/test_fast_decode_api.py --fast-token-layouts 2d` passes
and the layout comparison command with `--require-candidate --min-speedup 1.0`
passes on V100.

## Batch-size coverage

The serving path now has a dedicated repeated-prompt batch smoke test:

```bash
python tests/test_batch_cache.py \
  --model /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --fuse-norm false \
  --batch-sizes 1 2 4
```

The benchmark sweep records both aggregate and per-sequence throughput:

```bash
python bench/bench_batch_sweep.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --fast-cache true \
  --fast-decode-api auto \
  --batch-sizes 1 2 4 8 \
  --results bench/results.jsonl
```

Latest V100 batch sweep:

| Batch | Forward total tok/s | Fast-token total tok/s | Fast-token per-seq tok/s |
|---:|---:|---:|---:|
| 1 | 40.0 | 56.4 | 56.4 |
| 2 | 79.1 | 111.3 | 55.7 |
| 4 | 156.6 | 221.0 | 55.3 |
| 8 | 312.9 | 441.3 | 55.2 |

## Dynamic-batch coverage

The dynamic-batch smoke test uses heterogeneous prompts, advances both batched
and per-row states, reorders the batched cache, then verifies the reordered next
logits against independently decoded rows:

```bash
python tests/test_dynamic_batch_cache.py \
  --model /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --fuse-norm false \
  --batch-size 3 \
  --prompt-tokens 64
```

The benchmark simulation repeatedly reorders active rows and drops completed
rows from the recurrent state cache:

```bash
python bench/bench_dynamic_batch.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --fast-cache true \
  --fast-token-backend auto \
  --decode-apis forward rwkv7_forward_token \
  --batch-size 8 \
  --min-batch-size 2 \
  --results bench/results.jsonl
```

This is not a full scheduler, but it gives a reproducible `axis=dynamic_batch`
signal for the cache operations needed by dynamic batching.

Latest V100 dynamic-batch simulation with native-JIT fast-token enabled:

| Decode API | Fast backend | Initial -> final batch | Reorders | Drops | Total tok/s | ms/token |
|---|---|---:|---:|---:|---:|---:|
| `forward` | n/a | 8 -> 4 | 32 | 4 | 214.8 | 4.6555 |
| `rwkv7_forward_token` | native-JIT | 8 -> 4 | 32 | 4 | 417.9 | 2.3931 |

## Decode microbench coverage

`bench_decode_micro.py` appends `axis=decode_micro` rows with stable per-component timings:

```bash
python bench/bench_decode_micro.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --fast-cache true \
  --fast-decode-api auto \
  --steps 128 \
  --results bench/results.jsonl
```

The row records reference HF fixed/greedy one-token decode, ordinary HF
fixed/greedy decode with `RWKV7_FAST_FORWARD=1`, optional direct fast-token API
fixed/greedy decode, and isolated `lm_head`, `norm+lm_head`, `argmax`,
embedding, and empty-loop costs. This gives an easier regression signal than
profiler tables while keeping the profiler for operator-level investigation.

Latest V100 microbench:

| Component | ms/token | tok/s |
|---|---:|---:|
| Reference HF `forward` fixed-token (`RWKV7_FAST_FORWARD=0`) | 25.1180 | 39.8 |
| Ordinary HF `forward` fixed-token (`RWKV7_FAST_FORWARD=1`, auto->native_graph) | 3.9643 | 252.3 |
| Direct `rwkv7_forward_token` fixed-token (auto->native_graph) | 3.9494 | 253.2 |
| `lm_head` only | 0.1388 | 7205.2 |
| argmax only | 0.0249 | 40233.1 |

`bench_forward_fast_path.py` emits a smaller `axis=forward_fast_path` gate row
for the production-facing path. It compares `RWKV7_FAST_FORWARD=0` reference HF
forward, ordinary HF forward with fast-forward enabled, and direct
`rwkv7_forward_token`; `check_results.py` requires the ordinary HF fast path to
be at least `3.0x` faster than reference forward, at least `0.9x` of direct
fast-token speed, and within fp16 diff tolerance.

`bench_generate_fast_path.py` emits `axis=generate_fast_path` for the top-level
HF API. It compares greedy `model.generate(..., use_cache=True)` with
`RWKV7_FAST_FORWARD=0` and `1`; `check_results.py` requires identical generated
tokens, bsz>=2 coverage, a valid effective backend, and at least `2.0x`
end-to-end new-token throughput improvement. The recorded V100 prompt=8/new=16 bsz=2 row is `75.3 tok/s`
aggregate for reference generate vs `303.5 tok/s` aggregate with fast-forward
(`4.03x`), with `generated_equal=true`, `32/32` generated tokens matched,
and effective backend `native_graph`.

`rwkv7_warmup_fast_token()` exposes a public serving preflight API for native
fast-token resources. With `backend="auto"` it follows the same native-graph ->
native-JIT -> FLA resolution as `rwkv7_forward_token`; with
`backend="native_graph"` it raises if graph replay is unavailable. The paired
`rwkv7_native_graph_cache_batch_sizes()` API reports which active batch sizes
are currently retained in the per-model graph-runner LRU, and
`rwkv7_native_graph_cache_stats()` reports requests/hits/misses/evictions plus
hit rate for cache-reuse dashboards.

`bench_fast_token_warmup.py` emits `axis=fast_token_warmup`:

```bash
python bench/bench_fast_token_warmup.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --fast-cache true \
  --fast-token-backend auto \
  --batch-sizes 1 2 4 8 \
  --native-graph-cache-size 8 \
  --results bench/results.jsonl
```

`check_results.py` now requires the warmup row to prove bsz=1/2/4/8 resolve to
`native_graph`, fit inside the configured graph cache, and are visible through
the cache-size inspection API before production traffic starts.

The native-graph runner now skips cache copies when the cache is already bound
to the graph runner's own buffers, which is the steady state for continuous
decode. `bench_native_graph_overhead.py` emits
`axis=native_graph_replay_overhead` to keep that wrapper overhead visible:

```bash
python bench/bench_native_graph_overhead.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --fast-cache true \
  --batch-sizes 1 2 4 8 \
  --prompt-tokens 64 \
  --steps 32 \
  --fixed-token \
  --results bench/results.jsonl
```

Latest V100 rows for bsz=1/2/4/8: public API `254.9` / `449.8` / `858.5` /
`1546.9` aggregate tok/s, runner-vs-API max diff `0.0` for all rows, graph
replay `3.9375` / `4.4620` / `4.6760` / `5.1876ms`, and cache-copy share
`0.0703` / `0.0376` / `0.0361` / `0.0329`. `check_results.py` gates every
required batch size with a minimum API throughput, runner/API equality
tolerance, and maximum cache-copy share.

## Decode component benchmark

`bench_decode_components.py` instruments the fast-token path itself:

```bash
python bench/bench_decode_components.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --fast-cache true \
  --fixed-token \
  --results bench/results.jsonl
```

It appends `axis=decode_components` rows with `component_ms`, `top_components`,
and `top_layers`. This bridges the gap between stable microbench rows and raw
profiler tables, and should be used to decide which per-layer operations to fuse
next.

Latest V100 component timing (instrumented, so use relative component weights
rather than the instrumented wall tok/s):

| Component group | ms/token |
|---|---:|
| attention linears + LoRA projections | 9.8695 |
| attention norm/correction/output projection | 4.5735 |
| recurrent kernel | 3.9276 |
| attention key mix/norm | 3.2613 |
| FFN key + ReLU square | 1.8493 |
| attention shift/mix | 1.7954 |

This makes the next optimization target concrete: reduce/fuse the many
one-token attention projection/LoRA calls first, then revisit output projection
and recurrent/norm groups.

## Projection/LoRA benchmark

`bench_projection_lora.py` drills into the largest component group:

```bash
python bench/bench_projection_lora.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --fast-cache true \
  --layers 0 1 11 \
  --results bench/results.jsonl
```

Latest V100 projection/LoRA timing for sampled layers:

| Item | ms/layer |
|---|---:|
| R/K/V current separate projections | 0.0896 |
| R/K/V PyTorch bmm candidate | 0.0836 |
| W/A LoRA current | 0.1424 |
| W/A LoRA PyTorch bmm candidate | 0.2658 |
| Avg current linears+LoRA sum | 0.3502 |
| Avg PyTorch candidate sum | 0.4679 |

Interpretation: simple PyTorch bmm grouping is not enough (`0.75x` of current
overall for this group). R/K/V batched matmul is only a small win, while W/A
LoRA bmm is slower and can introduce larger fp16 numerical differences. The
next real optimization should be a custom fused projection/LoRA path or a
deeper rewrite that reduces launches without adding stack/bmm overhead.

Newer rows also emit `sample_matrix_profile`, `sample_matrix_profile_summary`,
and `fused_kernel_plan`. These fields turn the profiler into the first concrete
step of `docs/performance/FUSED_BACKEND.md`: they record matrix shapes, per-token FLOPs,
fp16/int8/int4 weight sizes, timed members, the first fp16 fusion target, and
the native-quant candidates that should later replace generic bnb kernels.

## Fused projection prototype

`rwkv7_hf/fused_projection.py` contains the first optional fp16 fused projection
prototype. `bench_fused_projection.py` times a single Triton R/K/V GEMV launch
against the current three separate projection linears:

```bash
python bench/bench_fused_projection.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --batch-size 1 \
  --layers 0 1 11 \
  --results bench/results.jsonl
```

Latest V100 prototype row: `triton_rkv_gemv` matches the current projections
with max abs diff `0.001953125` and min cosine `0.9999997`, but it is still
slower (`0.8429x` current linears, `0.11798ms` vs `0.09945ms`). This is useful
negative evidence: the first integration target should be a more optimized
shape-specialized/tensor-core-aware projection or a deeper fused time-mix path,
not this initial GEMV kernel.

## Fused W/A LoRA prototype

`rwkv7_hf/fused_lora.py` contains the first custom LoRA fusion probe for the
attention W/A pair. `bench_fused_wa_lora.py` times a grouped Triton
down/activation kernel plus a grouped up/bias kernel against the current W/A
LoRA modules:

```bash
python bench/bench_fused_wa_lora.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --batch-size 1 \
  --layers 0 1 11 \
  --block-m 64 \
  --block-r 64 \
  --block-k 64 \
  --results bench/results.jsonl
```

Latest V100 prototype row: `triton_fused_wa_lora` is correctness-clean
(`max_abs_diff=0.015625`, `min_cosine=0.9999999`) but still slower than the
current W/A LoRA modules (`0.8601x`, `0.16883ms` vs `0.14521ms`). This is
negative evidence for standalone two-kernel LoRA grouping; the next LoRA attempt
should fuse deeper with R/K/V and other attention projection work.

## Fused W/A/G LoRA prototype

`bench_fused_wag_lora.py` extends the LoRA grouping probe from W/A to W/A/G.
This covers the larger attention LoRA bucket and supports mixed ranks (`w=64`,
`a=64`, `g=128` on the 0.1B V100 checkpoint):

```bash
python bench/bench_fused_wag_lora.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --batch-size 1 \
  --layers 0 1 11 \
  --block-m 64 \
  --block-r 64 \
  --block-k 64 \
  --steps 512 \
  --results bench/results.jsonl
```

Latest stable V100 row: `triton_fused_wag_lora` is correctness-clean
(`max_abs_diff=0.0078125`, `min_cosine=0.99999994`) and is faster than the
current W/A/G LoRA modules (`1.0985x`, `0.26336ms` vs `0.28931ms`). This is the
first profitable LoRA grouping row, but it is still only a sub-kernel win; the
next performance step is to combine W/A/G with R/K/V projection and state/update
work so the full token path can move toward the Albatross ratios.

## Fused R/K/V + W/A/G projection prototype

`rwkv7_hf/fused_attention_projection.py` contains the first combined attention
projection probe. It computes R/K/V dense projections and W/A/G LoRA down
activations in one Triton launch, then computes the W/A/G LoRA up projections in
a second launch:

```bash
python bench/bench_fused_rkv_wag_projection.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --batch-size 1 \
  --layers 0 1 11 \
  --block-m 64 \
  --block-r 64 \
  --block-k 64 \
  --steps 512 \
  --results bench/results.jsonl
```

Latest stable V100 row: `triton_rkv_wag_down_plus_wag_up` is correctness-clean
(`max_abs_diff=0.015625`, `min_cosine=0.99999988`) and is a small positive step
against the current R/K/V + W/A/G modules (`1.0103x`, `0.31102ms` vs
`0.31422ms`). This shows launch grouping can work across dense projection and
LoRA, but the gain is too small for the Albatross gap; next work should fold in
more of the attention state/update/output path or improve the dense projection
math path.

`RWKV7_NATIVE_GRAPH_FUSED_PROJECTION=1` wires this prototype into native-graph
decode as an opt-in integration guard. The V100 bsz=1/2/4/8 fixed-token matrix
is correctness-clean, but currently slower than the default output-fused graph:

| bsz | default ms/step | projection-fused ms/step | speedup | greedy |
|---:|---:|---:|---:|---:|
| 1 | 3.9060 | 4.5563 | 0.8573x | 32/32 |
| 2 | 4.3770 | 4.7904 | 0.9137x | 64/64 |
| 4 | 4.5721 | 5.0759 | 0.9008x | 128/128 |
| 8 | 5.0785 | 5.4719 | 0.9281x | 256/256 |

Therefore this two-kernel projection path must stay opt-in; it is useful
telemetry, but the next projection attempt needs fewer launches, better
tensor-core occupancy, or fusion across output/recurrent work before default
native-graph integration.

## Fused attention output prototype

`rwkv7_hf/fused_output.py` targets the `attn_norm_out_proj` bucket without
replacing the final dense `o_proj`. The prototype fuses attention output prep:
group norm over recurrent output, recurrent correction, and gate multiply. The
final `o_proj` remains cuBLAS-backed so the probe measures whether the
non-GEMM output work is worth folding into the fused attention path:

```bash
python bench/bench_fused_attn_output.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --batch-size 1 \
  --layers 0 1 11 \
  --input-scale 0.3 \
  --steps 512 \
  --results bench/results.jsonl
```

Latest stable V100 row: `triton_attn_output_prepare_plus_cublas_o` is
correctness-clean (`max_abs_diff=0.00390625`, `output_max_abs_diff=0.0009765625`,
`min_cosine=0.99999970`) and is faster than the current group-norm/correction
prep plus cuBLAS output path (`1.2225x`, `0.19117ms` vs `0.23370ms`). This is
the strongest current fp16 sub-kernel win after recurrent-state fusion, but it
still needs integration into a larger attention fusion path before it can close
the end-to-end Albatross gap.

## Fused FFN prototype

`rwkv7_hf/fused_ffn.py` provides the first FFN path probe. It fuses FFN
shift-mix, key projection, and relu² into one Triton launch, followed by a value
projection launch:

```bash
python bench/bench_fused_ffn.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --batch-size 1 \
  --layers 0 1 11 \
  --block-m 128 \
  --block-k 128 \
  --steps 512 \
  --results bench/results.jsonl
```

Latest stable V100 row: `triton_ffn_shift_key_relu_value` is correctness-clean
(`max_abs_diff=0.0009765625`, `min_cosine=0.99999964`) but slower than the
current cuBLAS-backed FFN path (`0.8949x`, `0.13080ms` vs `0.11705ms`). This is
negative evidence for replacing the FFN key/value GEMMs with a naive two-kernel
Triton path; FFN should either stay cuBLAS-backed or be fused into a larger graph
where launch reduction outweighs the GEMM loss.

## Fused shift-mix prototype

`rwkv7_hf/fused_time_mix.py` contains an optional Triton prototype for the six
attention time-mix inputs used before RWKV-7 R/W/K/V/A/G projections. It is
measured separately because the native-graph decode path is launch-sensitive:

```bash
python bench/bench_fused_shift_mix.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --batch-size 1 \
  --input-rank 2 \
  --layers 0 1 11 \
  --results bench/results.jsonl
```

Latest V100 prototype row: `triton_attn_shift_mix` is exact (`max_abs_diff=0`)
with min cosine `0.9999999`, but it is slower than current torch pointwise ops
(`0.7715x`, `0.13416ms` vs `0.10351ms`). This rules out integrating a standalone
shift-mix kernel; the next fused fp16 attempt should combine shift-mix with the
following projection/LoRA/state-update work so one launch does more useful math.

## Fused recurrent prototype

`rwkv7_hf/fused_recurrent_update.py` contains an optional Triton prototype for
the one-token recurrent state update. It avoids materializing the rank-1
transition matrix and fuses state update plus readout:

```bash
python bench/bench_fused_recurrent.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --batch-size 1 \
  --layers 0 1 11 \
  --results bench/results.jsonl
```

Latest V100 prototype row: `triton_rank1_recurrent` is profitable versus the
current torch expression (`2.7931x`, `0.07841ms` vs `0.21901ms`) with
`out_max_abs_diff=0.0234375`, `state_max_abs_diff=0.0037985`, and
`out_min_cosine=0.9999998`. This is the first fused fp16 prototype worth
integrating behind the HF native-graph fast-token path, subject to full
end-to-end greedy/cache correctness gates.

## Native-graph fused recurrent integration

Set `RWKV7_NATIVE_GRAPH_FUSED_RECURRENT=1` to capture native-graph decode with
the recurrent Triton prototype. The graph-runner cache key includes this flag so
serving can switch the experiment on/off without accidentally reusing a graph
captured under the other mode.

```bash
python bench/bench_native_graph_fused_recurrent.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --batch-size 1 \
  --prompt-tokens 64 \
  --fixed-token \
  --results bench/results.jsonl
```

Latest V100 integration row: first-step logits are identical
(`max_abs_diff_first_step=0`, `min_cosine_first_step=1.0000002`) and greedy
tokens match `32/32`. End-to-end graph replay is currently neutral
(`1.0033x`, `4.2878ms` fused vs `4.3018ms` baseline), so this remains opt-in;
the isolated recurrent kernel is fast, but the captured full-token graph still
needs deeper fusion around the larger projection/LoRA bottleneck.

## Native-graph fused output integration

Native-graph decode now enables fused attention output-prep by default. Set
`RWKV7_NATIVE_GRAPH_FUSED_OUTPUT=0` to disable it for A/B or fallback testing.
The graph-runner cache key includes both `RWKV7_NATIVE_GRAPH_FUSED_RECURRENT`
and `RWKV7_NATIVE_GRAPH_FUSED_OUTPUT`, while keeping the active batch size
visible in cache telemetry.

```bash
python bench/bench_native_graph_fused_output.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --batch-size 1 \
  --prompt-tokens 64 \
  --fixed-token \
  --results bench/results.jsonl
```

Latest V100 integration row: first-step logits remain aligned
(`max_abs_diff_first_step=0.0625`, `min_cosine_first_step=1.0000001`) and greedy
tokens match `32/32`. End-to-end native-graph replay improves to `1.0997x`
(`4.0205ms` fused vs `4.4214ms` baseline, `248.7` vs `226.2` tok/s). This makes
fused output prep the first default Triton kernel to move full native-graph
token latency on V100.

The V100 fixed-token batch matrix now covers the same active batch sizes used by
the native-graph serving cache:

| bsz | baseline ms/step | fused ms/step | speedup | greedy |
|---:|---:|---:|---:|---:|
| 1 | 4.4214 | 4.0205 | 1.0997x | 32/32 |
| 2 | 12.1951 | 9.2125 | 1.3238x | 64/64 |
| 4 | 12.6281 | 12.2061 | 1.0346x | 128/128 |
| 8 | 12.9932 | 12.4389 | 1.0446x | 256/256 |

The minimum V100 speedup across bsz=1/2/4/8 is therefore `1.0346x`; next
validation should cover 5070/newer GPUs and combining this with
recurrent/projection/LoRA fusion.

Greedy `bench_batch_sweep.py` on an otherwise idle second V100 confirms that
making output-prep fusion the native-graph default improves the normal
`rwkv7_forward_token` serving path too. With
`RWKV7_NATIVE_GRAPH_FUSED_OUTPUT=0`, the no-output-fusion baseline was
`252.3/451.2/852.9/1542.3` aggregate tok/s for bsz=1/2/4/8. With the default
output fusion enabled, the same sweep reached:

| bsz | default fused tok/s | default fused ms/step | speedup vs no-output |
|---:|---:|---:|---:|
| 1 | 274.7 | 3.64 | 1.0888x |
| 2 | 492.3 | 4.06 | 1.0911x |
| 4 | 934.2 | 4.28 | 1.0953x |
| 8 | 1673.1 | 4.78 | 1.0848x |

## Fused recurrent + output-prep native-graph probe

The next profitable fp16 step is deeper than standalone recurrent or standalone
output-prep fusion. `fused_recurrent_output_prepare()` combines recurrent state
update/readout, group norm, recurrent correction, and gate multiply into one
Triton kernel while keeping the final `o_proj` on cuBLAS. This is now the
native-graph default; set `RWKV7_NATIVE_GRAPH_FUSED_RECURRENT_OUTPUT=0` to
disable it for A/B or fallback testing.

Isolated kernel benchmark:

```bash
python bench/bench_fused_recurrent_output.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --batch-size 1 \
  --layers 0 1 11 \
  --results bench/results.jsonl
```

Latest V100 isolated row: `triton_recurrent_output_prepare` averages
`0.10580ms`, beating split fused recurrent/output kernels by `1.7956x` and the
torch current path by `4.1916x`. Correctness is aligned with
`split_out_max_abs_diff=0.00390625`, `split_state_max_abs_diff=1.19e-7`, and
`split_out_min_cosine=0.99999994`.

Native-graph A/B:

```bash
python bench/bench_native_graph_fused_recurrent_output.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --fast-cache true \
  --batch-size 1 \
  --prompt-tokens 64 \
  --fixed-token \
  --results bench/results.jsonl
```

V100 bsz=1/2/4/8 native-graph matrix:

| bsz | baseline ms/step | fused ms/step | speedup | baseline tok/s | fused tok/s | greedy |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.9255 | 3.2364 | 1.2129x | 254.7 | 309.0 | 32/32 |
| 2 | 4.4472 | 3.7672 | 1.1805x | 449.7 | 530.9 | 64/64 |
| 4 | 4.5937 | 3.6998 | 1.2416x | 870.8 | 1081.1 | 128/128 |
| 8 | 5.1479 | 4.1170 | 1.2504x | 1554.0 | 1943.2 | 256/256 |

A normal `bench_batch_sweep.py` run with the new default
`RWKV7_NATIVE_GRAPH_FUSED_RECURRENT_OUTPUT=1` plus default fused output enabled
reaches `332.2`/`589.5`/`1177.9`/`2136.7` aggregate tok/s for bsz=1/2/4/8.
That raises the current Albatross decode comparison to min `0.4352x`, max
`0.6474x`: bsz=8 is now above the P1 decode line, but the overall P1 gate is
still GAP because the minimum batch ratio is below `0.55x`.

Additional flag sweeps under the recurrent+output default show that the current
projection-side opt-ins are not the next P1 route: `RWKV7_NATIVE_GRAPH_FUSED_WAG_LORA=1`
with `block_m=16, block_r=64, block_k=64` reaches only `0.94x`-`0.99x` of
default, and `RWKV7_NATIVE_GRAPH_FUSED_PROJECTION=1` reaches only `0.84x`-`0.91x`.
Analyzer Albatross gates now use the default native-graph batch rows even if
later experimental flag rows are appended.

## Fused output-prep + `o_proj` prototype

The next attention-output probe folds the final dense `o_proj` into the Triton
output-prep kernel. It is not enabled by default because the full native-graph
integration is slower than the current fused-prep+cuBLAS default, but it is a
useful occupancy/deeper-fusion target.

Isolated kernel sweep:

```bash
python bench/bench_fused_attn_output_project.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --batch-size 1 \
  --layers 0 1 11 \
  --block-m 8 16 32 64 \
  --results bench/results.jsonl
```

Latest isolated V100 row: `triton_attn_output_prepare_o_proj` averages
`0.14649ms`, which is `1.5965x` faster than the old output path and `1.2931x`
faster than fused output-prep plus cuBLAS `o_proj`. Correctness remains aligned
(`max_abs_diff=0.001953125`, `min_cosine=0.99999976`).

Native-graph A/B:

```bash
python bench/bench_native_graph_fused_output_project.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --fast-cache true \
  --batch-size 1 \
  --prompt-tokens 64 \
  --fixed-token \
  --block-m 16 \
  --results bench/results.jsonl
```

Set `RWKV7_NATIVE_GRAPH_FUSED_OUTPUT_PROJECT=1` to opt into this path manually;
`RWKV7_NATIVE_GRAPH_FUSED_OUTPUT_PROJECT_BLOCK_M` selects the row tile and is
included in the native-graph runner cache key. The V100 bsz=1/2/4/8 matrix is
greedy-exact but slower than the default output-fused graph:

| bsz | baseline ms/step | fused project ms/step | speedup | greedy |
|---:|---:|---:|---:|---:|
| 1 | 3.9019 | 4.0968 | 0.9524x | 32/32 |
| 2 | 4.4253 | 4.5679 | 0.9688x | 64/64 |
| 4 | 4.6334 | 4.8186 | 0.9616x | 128/128 |
| 8 | 5.0789 | 5.2805 | 0.9618x | 256/256 |

Conclusion: the isolated one-launch project kernel is promising, but the
captured full-token graph does not yet preserve the win. Keep it opt-in and use
the telemetry to guide a better `o_proj` fusion instead of making it default.

## Native-graph fused W/A/G LoRA integration

`bench_fused_wag_lora.py` showed isolated W/A/G LoRA grouping can beat the three
separate LoRA modules. The native-graph integration keeps R/K/V projections on
cuBLAS and tests only the LoRA grouping behind
`RWKV7_NATIVE_GRAPH_FUSED_WAG_LORA=1`.

```bash
python bench/bench_native_graph_fused_wag_lora.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --fast-cache true \
  --batch-size 1 \
  --prompt-tokens 64 \
  --fixed-token \
  --block-m 16 \
  --block-r 64 \
  --block-k 64 \
  --results bench/results.jsonl
```

The graph-runner cache key includes `RWKV7_NATIVE_GRAPH_FUSED_WAG_LORA` and
`RWKV7_NATIVE_GRAPH_FUSED_WAG_LORA_BLOCK_{M,R,K}`. V100 correctness is clean,
but the batch matrix is not defaultable:

| bsz | baseline ms/step | fused W/A/G LoRA ms/step | speedup | greedy |
|---:|---:|---:|---:|---:|
| 1 | 4.0200 | 4.2737 | 0.9406x | 32/32 |
| 2 | 4.3872 | 4.5523 | 0.9637x | 64/64 |
| 4 | 4.5884 | 4.7723 | 0.9615x | 128/128 |
| 8 | 5.0921 | 5.0624 | 1.0059x | 256/256 |

Conclusion: LoRA-only fusion is only marginally positive at bsz=8 and slower at
smaller active batches, so it remains opt-in telemetry. The next useful kernel
needs deeper projection/LoRA/state/output fusion rather than a standalone LoRA
replacement.

## Native W8 dequant-GEMV prototype

`rwkv7_hf/native_quant.py` contains the first RWKV-native W8 serving prototype:
row-wise int8 weight packing plus a fused dequant GEMV/GEMM. This is separate
from bitsandbytes; it is intended to become the native quant fast path after the
kernel is fast enough.

```bash
python bench/bench_native_quant_gemv.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --batch-size 1 \
  --layers 0 1 11 \
  --modules attn.r_proj attn.k_proj attn.v_proj attn.o_proj ffn.key ffn.value \
  --results bench/results.jsonl
```

Latest V100 prototype row: `triton_int8_rowwise_gemv` reduces the sampled
projection/FFN weight footprint to `0.502x` fp16 and keeps output cosine high
(`min_cosine=0.9999172`, `max_abs_diff=0.044921875`), but the first Triton
kernel is still slower (`0.3816x`, `0.05409ms` vs `0.02064ms`). This confirms
the native W8 packing direction while showing the kernel still needs a more
optimized/tensor-core-aware implementation before it can replace bnb or fp16.

## Native W4 dequant-GEMV prototype

`rwkv7_hf/native_quant.py` also contains the first RWKV-native W4 serving
prototype: row-wise signed int4 weight packing with two values per byte plus a
fused nibble-unpack/dequant GEMV/GEMM. This is telemetry-first and separate
from bitsandbytes.

```bash
python bench/bench_native_quant_w4_gemv.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --batch-size 1 \
  --layers 0 1 11 \
  --modules attn.r_proj attn.k_proj attn.v_proj attn.o_proj ffn.key ffn.value \
  --results bench/results.jsonl
```

Latest V100 prototype row: `triton_int4_rowwise_gemv` reduces the sampled
projection/FFN weight footprint to `0.252x` fp16. Correctness is usable as a
first W4 probe but visibly looser than W8 (`min_cosine=0.9802878`,
`max_abs_diff=0.9287109`). The first Triton kernel remains slower than fp16
cuBLAS (`0.359x`, `0.05773ms` vs `0.02072ms`), so W4 now has working pack,
fallback, fused-kernel telemetry, and analyzer visibility, but still needs a
faster packed reduction / fusion with projection groups before it can satisfy
the final `decode >= fp16` target.

## Native W8 fused R/K/V quant projection prototype

`bench/bench_native_quant_rkv.py` measures the next native quant step: the three
decode-hot attention R/K/V projections are computed from row-wise W8 weights in
one Triton launch, then compared with both fp16 linears and three separate
native W8 GEMVs.

```bash
python bench/bench_native_quant_rkv.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --batch-size 1 \
  --layers 0 1 11 \
  --results bench/results.jsonl
```

Latest V100 prototype row: `triton_int8_fused_rkv_gemv` keeps the R/K/V sampled
weight footprint at `0.5026x` fp16 and is bit-identical to the three separate
native W8 GEMVs (`max_abs_diff_separate_vs_fused=0`). It improves the separate
W8 GEMV path by `1.7628x` (`0.08878ms` fused vs `0.1565ms` separate) while
remaining below fp16 cuBLAS at `0.7847x` (`0.08878ms` fused vs `0.06967ms`
fp16). This shows launch/group fusion is the right direction for native quant,
but the next step must fuse more projection/LoRA work or specialize the packed
reduction further to clear the `>=1.0x fp16` target.

## Native W4 fused R/K/V quant projection prototype

`bench/bench_native_quant_w4_rkv.py` mirrors the W8 R/K/V fusion for packed W4
weights, using one Triton launch to unpack/dequantize and compute R/K/V.

```bash
python bench/bench_native_quant_w4_rkv.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --batch-size 1 \
  --layers 0 1 11 \
  --results bench/results.jsonl
```

Latest V100 prototype row: `triton_int4_fused_rkv_gemv` keeps the R/K/V sampled
weight footprint at `0.2526x` fp16 and is bit-identical to the three separate
native W4 GEMVs (`max_abs_diff_separate_vs_fused=0`). It improves separate W4
GEMVs by `1.7958x` (`0.0912ms` fused vs `0.16378ms` separate), with
`min_cosine_fp16_vs_fused=0.9750665`. Like the W8 fused R/K/V row, it is still
below fp16 cuBLAS (`0.7795x`), so launch fusion works but the final quant
target needs deeper fusion with LoRA/projection groups or a faster packed
reduction.

## Native W8/W4 fused R/K/V block sweep

`bench/bench_native_quant_rkv_sweep.py` loads the model once, measures a shared
fp16 R/K/V baseline, then sweeps the W8/W4 fused R/K/V kernels across
`block_m`/`block_k`. This avoids the per-config model-load drift that made the
standalone prototype rows overstate some speedup ratios.

```bash
python bench/bench_native_quant_rkv_sweep.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --batch-size 1 \
  --layers 0 1 11 \
  --quantizations w8 w4 \
  --block-m 8 16 32 64 \
  --block-k 32 64 128 \
  --results bench/results.jsonl
```

Latest single-load V100 sweep: W8 best latency is `block_m=64, block_k=128`,
`0.08965ms`, `0.7873x` fp16 and `1.7561x` separate W8, with footprint
`0.5026x`. W4 best latency is `block_m=8, block_k=64`, `0.09203ms`, `0.7675x`
fp16 and `1.7931x` separate W4, with footprint `0.2526x`. The sweep confirms
the gap is not just a block-size choice; the next quant step needs a
tensor-core-aware packed kernel and/or deeper fusion beyond R/K/V.

## Larger converted-model smoke

`bench_larger_model_smoke.py` proves the shape-inferred converter on real
checkpoints beyond the 0.1B development model. It loads each generated HF
directory with AutoConfig/AutoTokenizer/AutoModelForCausalLM, runs cached
forward, runs greedy generation, records config dimensions, checkpoint
provenance, backend selection, and memory.

```bash
python bench/bench_larger_model_smoke.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.4b-hf \
  --model-size-label 0.4b \
  --checkpoint-path /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fast-token-backend auto \
  --max-new-tokens 4 \
  --results bench/results.jsonl

python bench/bench_larger_model_smoke.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1g-1.5b-hf \
  --model-size-label 1.5b \
  --checkpoint-path /home/data/wangyue/models/rwkv7/rwkv7-g1g-1.5b-20260526-ctx8192.pth \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fast-token-backend auto \
  --max-new-tokens 2 \
  --results bench/results.jsonl

python bench/bench_larger_model_smoke.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1g-2.9b-hf \
  --model-size-label 2.9b \
  --checkpoint-path /home/data/wangyue/models/rwkv7/rwkv7-g1g-2.9b-20260526-ctx8192.pth \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fast-token-backend auto \
  --max-new-tokens 2 \
  --results bench/results.jsonl

python bench/bench_larger_model_smoke.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1g-7.2b-hf \
  --model-size-label 7.2b \
  --checkpoint-path /home/data/wangyue/models/rwkv7/rwkv7-g1g-7.2b-20260523-ctx8192.pth \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fast-token-backend auto \
  --max-new-tokens 2 \
  --results bench/results.jsonl

python bench/bench_larger_model_smoke.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1g-13.3b-hf \
  --model-size-label 13.3b \
  --checkpoint-path /home/data/wangyue/models/rwkv7/rwkv7-g1g-13.3b-20260523-ctx8192.pth \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --fast-token-backend native_jit \
  --max-new-tokens 2 \
  --results bench/results.jsonl
```

Latest V100 larger-model rows:

| Model | hidden | layers | head_dim | value_dim | generated | backend | load s | generate s | footprint | peak VRAM |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| rwkv7-g1d-0.4b-hf | 1024 | 24 | 64 | 1024 | 4 | native_graph | 15.095 | 0.6751 | 859.8 MB | 1124.5 MB |
| rwkv7-g1g-1.5b-hf | 2048 | 24 | 64 | 2048 | 2 | native_graph | 27.991 | 0.6307 | 2913.3 MB | 3178.6 MB |
| rwkv7-g1g-2.9b-hf | 2560 | 32 | 64 | 2560 | 2 | native_graph | 35.589 | 0.7148 | 5622.4 MB | 5888.0 MB |
| rwkv7-g1g-7.2b-hf | 4096 | 32 | 64 | 4096 | 2 | native_graph | 66.292 | 0.7564 | 13731.3 MB | 13997.8 MB |
| rwkv7-g1g-13.3b-hf | 4096 | 61 | 64 | 4096 | 2 | native_jit | 99.107 | 0.7428 | 25309.1 MB | 25575.6 MB |

Checkpoint provenance is recorded in the rows: 0.4B SHA256
`947cb9b8013224e06b112b72204256bec65096cc935a7767ce63d8e3ddef83bb`, size
`901776749` bytes; 1.5B SHA256
`441f70b096ad62442b5c33128bfe717c5d8529915c45a9709d4482016e8a0482`, size
`3055444605` bytes; 2.9B SHA256
`3d118ed77fe94e63e6fc0a6afd5a4fac49fe70da4e3d9d91b628951bb55dd798`, size
`5896273469` bytes; 7.2B SHA256
`425fc9bda2d12d4ce3b6bfe5c3b3f355be8b14d85960cf40fcca58a19d632630`, size
`14400007869` bytes; 13.3B SHA256
`0aa686d3ca4bb486e83e3071f4798a210f960e1fc1f5042e6cb418cc463814d6`, size
`26540868485` bytes. The regression gate now requires all five smoke rows so
the converter cannot silently regress to 0.1B-only shape assumptions.

### 13.3B official alignment + decode speed

Beyond the 2-token smoke above, 13.3B is official-alignment and decode-speed
validated on a single V100-32GB (full detail in
[`docs/validation/V100_HF_VALIDATION.md#133b-inference-validation`](docs/validation/V100_HF_VALIDATION.md#133b-inference-validation)).
HF fp16 vs official `rwkv` `cpu fp32`: cosine `0.9999976`, top5 `1.0`,
argmax `1.0`, max_abs `0.0813`, greedy `16/16` matched. Decode (prompt=128,
decode=64, fp16): fla `11.6`, `native_jit` `18.4` (1.58x fla), `native_graph`
`17.1` tok/s at `25594 MB` peak. `native_jit` is the recommended 13.3B backend;
`native_graph` fits 32GB but is slower than `native_jit` because 13.3B decode is
memory-bound and graph-replay overhead inverts the usual small-model graph win.

## Quantized inference coverage

`tests/test_quantized_inference.py` checks that the adapter loads and generates
through standard HF `BitsAndBytesConfig` paths:

```bash
python tests/test_quantized_inference.py \
  --model /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --quantization 8bit

python tests/test_quantized_inference.py \
  --model /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --quantization 4bit
```

`bench/bench_quantization.py` records comparable fp16 / 8-bit / 4-bit rows and
can compare the slower cached-HF reference decode against the HF fast-forward
path:

```bash
python bench/bench_quantization.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --quantizations none 8bit 4bit \
  --prompt-tokens 128 \
  --decode-tokens 8 \
  --warmup 1 \
  --runs 1 \
  --results bench/results.jsonl
```

Latest short V100 rows:

| Quantization | Model footprint | Peak VRAM | Prefill tok/s | Reference decode tok/s | Fast-forward decode tok/s | Status |
|---|---:|---:|---:|---:|---:|---|
| none/fp16 | 364.4 MB | 636.2 MB | 8370.4 | 41.0 | 217.2 | PASS |
| 8-bit bnb + dense LoRA rank | 283.4 MB | 321.6 MB | 3226.6 | 15.9 | 16.3 | PASS smoke, speed gap |
| 4-bit bnb + dense LoRA rank | 242.9 MB | 286.4 MB | 6075.9 | 32.6 | 32.1 | PASS smoke, speed gap |
| 8-bit bnb `decode_hot` | 310.4 MB | 582.4 MB | 5406.3 | 25.6 | 27.0 | faster hybrid, speed gap |
| 4-bit bnb `decode_hot` | 283.4 MB | 310.0 MB | 7527.1 | 37.5 | 39.1 | faster hybrid, speed gap |

The adapter appends `lm_head` and `.*_lora\.lora\.[02]` to HF/bnb
`llm_int8_skip_modules` so tiny RWKV LoRA rank projections are not replaced
with inefficient quantized kernels, while the large projection/FFN weights
remain W8/W4. `bench_quantization.py` now records `quant_skip_modules`,
`module_counts`, and `selected_decode_path`; the latest row selects fast-forward
for 8-bit and reference cached decode for 4-bit because that path is slightly
faster on V100. The memory direction is correct, but selected W8/W4 decode is
still slower than fp16 native-graph decode, so production quantized serving
still needs a custom fused/native quantized projection path before it can meet
the original "not slower than fp16" target.

`RWKV7_BNB_SKIP_POLICY` / `--quant-skip-policy` adds explicit quantization
speed-memory policies:

- `memory` (default): keep only `lm_head` and tiny LoRA rank projections dense;
  this is the canonical memory-target row used by result gates.
- `decode_hot`: additionally keep attention `r_proj/k_proj/v_proj/o_proj`
  dense while FFN key/value remain quantized. Latest V100 rows improve selected
  decode to `27.0 tok/s` for 8-bit and `39.1 tok/s` for 4-bit while keeping
  footprint below fp16 (`310.4 MB` / `283.4 MB`). It is useful as a hybrid speed
  probe but still far below fp16 native-graph.
- `dense`: keep attention and FFN projections dense; diagnostic upper bound,
  effectively fp16 footprint.

Analyzer/check gates keep canonical quantization status anchored to `memory`
policy rows so hybrid probes do not accidentally overwrite W4 memory-target
evidence. The analyzer now also reports `quantization_best_variants`, selecting
the fastest passing policy per W8/W4 mode and comparing its decode and footprint
ratios against fp16.

### RTX 5090 native MM8/MM4 speed-policy quantization

`bench/5090_blackwell_quant_policy_20260705/` records the repository-native
MM8/MM4 policy split on RTX 5090. `native_mm_policy=memory` keeps the historical
size-gated behavior and is the maximum-footprint-reduction lane; it is not a
speed claim when many FFN/projection modules are quantized. `native_mm_policy=speed`
quantizes only `lm_head` after the same size gate, which gives a smaller but
real footprint reduction while keeping cached decode near fp16 speed.

Key RTX 5090 rows:

| model | quantization | policy | replaced modules | decode ratio vs fp16 | footprint ratio vs fp16 | prompt/final logits cosine | same next token |
|---|---|---|---:|---:|---:|---:|---|
| 1.5B | mm8 | speed | 1 | 0.9841 | 0.9562 | 0.99999499 / 0.99999452 | true |
| 1.5B | mm4 | speed | 1 | 0.9860 | 0.9342 | 0.99982727 / 0.99983704 | true |
| 2.9B | mm8 | speed | 1 | 0.9975 | 0.9716 | 0.99999589 / 0.99999553 | true |
| 2.9B | mm4 | speed | 1 | 0.9706 | 0.9573 | 0.99983841 / 0.99976456 | true |
| 7.2B | mm8 | speed | 1 | 1.0074 | 0.9814 | 0.99999332 / 0.99999321 | true |
| 7.2B | mm4 | speed | 1 | 0.9988 | 0.9720 | 0.99946028 / 0.99944884 | true |

The same artifact demonstrates why `memory` must stay separate from the speed
acceptance lane: 1.5B `memory` replaces 49 modules and falls to about `0.47x`
fp16, while 2.9B `memory` replaces 65 modules and also falls to about `0.47x`
fp16. The fused quantized block-kernel roadmap is still required for large
footprint reductions at fp16-or-better speed, but the speed-policy lane matches
the acceptance wording of "footprint decreases and W8/W4 is not slower" much
more closely than the full-memory policy. 7.2B W8/W4 rows were run in fresh
processes on the 32GB 5090 to avoid peak-memory fragmentation from consecutive
large-model loads.

`bench/5090_blackwell_quant_matrix_20260705/` extends the same speed-policy
route into the full fresh-process Blackwell matrix requested for acceptance:
1.5B / 2.9B / 7.2B × fp16 / native MM8 / native MM4 × prompt 128 / 512 / 2048
× decode 128 / 512 × batch-size 1 / 2 / 4 / 8. All 216 rows pass. Each
quantized row reloads a cached fp16 baseline from the matching fresh process,
so the ratios are not affected by the earlier consecutive-load 7B fragmentation
problem.

Summary of the 144 quantized rows:

| model | quantization | rows | min speed ratio | median speed ratio | max speed ratio | rows >= fp16 | footprint ratio | same next token | min prompt/final logits cosine |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.5B | mm8 | 24 | 0.9614 | 0.9941 | 1.0646 | 7/24 | 0.9562 | 24/24 | 0.99996233 / 0.99996215 |
| 1.5B | mm4 | 24 | 0.9516 | 0.9845 | 1.0691 | 6/24 | 0.9342 | 24/24 | 0.99980509 / 0.99976385 |
| 2.9B | mm8 | 24 | 0.9315 | 0.9908 | 1.0759 | 10/24 | 0.9716 | 24/24 | 0.99995863 / 0.99995917 |
| 2.9B | mm4 | 24 | 0.9413 | 0.9908 | 1.0352 | 8/24 | 0.9573 | 24/24 | 0.99981976 / 0.99979115 |
| 7.2B | mm8 | 24 | 0.7619 | 0.9679 | 1.0405 | 3/24 | 0.9814 | 24/24 | 0.99996132 / 0.99996006 |
| 7.2B | mm4 | 24 | 0.6695 | 0.9610 | 1.0379 | 3/24 | 0.9720 | 24/24 | 0.99944627 / 0.99946213 |

Interpretation: correctness is stable across the full matrix, footprint always
drops, and many Blackwell rows are already fp16-or-better. The remaining gap is
shape-dependent rather than a load/correctness blocker: 1.5B/2.9B medians are
near fp16, while the 7.2B `bsz=8`, prompt-2048, decode-512 pressure rows expose
the fused/head-dispatch work still needed for a strict all-shapes speed claim.

13.3B boundary probe: the official ModelScope LFS checkpoint was pulled on the
same 5090 host (`rwkv7-g1g-13.3b-20260523-ctx8192.pth`, about 25GB on disk; LFS
declares 26,540,868,485 bytes). The current converter holds the full official
checkpoint and a full HF template at once; on the 48GB-RAM rental this did not
produce an HF model directory. No 13.3B fp16/MM8/MM4 speed row is claimed here;
the next step is a low-memory/streaming converter or a larger-RAM host, then the
same fresh-process boundary probe can be reused.

### RTX 5090 MATH500 final acceptance artifact

`bench/math500_final_acceptance_5090_1p5b_20260705/` records a full MATH500
avg@64 run using the final acceptance runner. It includes best-bsz sweep,
full `500 × 64 = 32000` generation summary, HF-vs-Albatross comparison, and
the uncheatable teacher-forced compression/logits-alignment report. The large
`generations.jsonl` byproduct is not committed; summaries and logs are.

Best-bsz sweep selected `bsz=128`:

| requested bsz | generation tok/s | rank |
|---:|---:|---:|
| 128 | 4855.721 | 1 |
| 96 | 4412.250 | 2 |
| 64 | 3970.580 | 3 |
| 192 | 3731.191 | 4 |
| 32 | 2463.453 | 5 |

Full avg@64 result against the current Albatross full reference:

| metric | HF 1.5B bsz128 | Albatross reference | delta / ratio |
|---|---:|---:|---:|
| correct generations | 12756 / 32000 | 4670 / 32000 | +8086 |
| rollout accuracy | 0.398625 | 0.1459375 | +0.2526875 |
| pass@64 | 0.662000 | 0.370000 | +0.292000 |
| summary token/s | 5918.906 | 3903.633 | 1.516x |
| wall token/s | 5854.033 | 3903.633 | 1.500x |
| steady decode token/s | 7410.107 | 3970.135 | 1.866x |

Interpretation: MATH500 avg@64 accuracy passes strongly and the compression
identity check is exact (`candidate/reference bits ratio = 1.00000000` over
43865 external MATH500 tokens). The strict `>=2x` Albatross speed gates do not
yet pass on this RTX 5090 full artifact, so this result is a high-accuracy
Blackwell acceptance record plus a clear remaining full-eval speed-gap marker.

### 0.4B V100 quantization sweep

Before refreshing older converted model dirs, run the code-only sync helper so
their remote-code wrappers include the latest quantization skip-policy support:

```bash
python scripts/sync_hf_adapter_code.py \
  /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.4b-hf
```

Then benchmark the 0.4B model:

```bash
python bench/bench_quantization.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.4b-hf \
  --model-size-label 0.4b \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --quantizations none 8bit 4bit \
  --quant-skip-policy memory \
  --prompt-tokens 128 \
  --decode-tokens 8 \
  --decode-mode compare \
  --warmup 1 \
  --runs 1 \
  --results bench/results.jsonl
```

Latest V100 0.4B rows:

| Quantization | Policy | Model footprint | Peak VRAM | Prefill tok/s | Selected decode tok/s | Fast backend | Status |
|---|---|---:|---:|---:|---:|---|---|
| none/fp16 | memory | 859.8 MB | 1136.7 MB | 2117.8 | 107.0 | native_graph | PASS |
| 8-bit bnb | memory | 571.8 MB | 629.5 MB | 817.6 | 8.4 | FLA | PASS, speed gap |
| 4-bit bnb | memory | 427.8 MB | 502.6 MB | 1517.3 | 16.3 | FLA | PASS, speed gap |
| 8-bit bnb | `decode_hot` | 667.8 MB | 945.3 MB | 1362.4 | 13.7 | FLA | faster hybrid, speed gap |
| 4-bit bnb | `decode_hot` | 571.8 MB | 624.3 MB | 1882.3 | 19.6 | FLA | faster hybrid, speed gap |

`analyze_results.py` keeps the canonical quantization gate anchored to the 0.1B
baseline, and reports larger-model rows separately under
`quantization_model_sweep`. The 0.4B rows confirm memory decreases
substantially, but V100 decode speed is still far below fp16 native-graph; the
next quantization task remains a fused/native W8/W4 serving path.

### A800 validation sweep

A800 rows use `NVIDIA A800-SXM4-80GB`, fp16, `attn_mode=fused_recurrent`, and
prompt128/decode8. The 0.4B / 1.5B / 2.9B converted HF directories are recorded
with placeholder paths under `/path/to/...`; use local converted checkpoint
paths when reproducing.

Latest A800 `bench_batch_sweep.py` native_graph decode rows:

| Model | Batch 1 tok/s | Batch 2 tok/s | Batch 4 tok/s | Batch 4 peak VRAM |
|---|---:|---:|---:|---:|
| 0.4B | 233.1 | 325.6 | 737.6 | 1875.7 MiB |
| 1.5B | 168.0 | 270.0 | 599.2 | 4907.1 MiB |
| 2.9B | 93.6 | 199.1 | 388.5 | 8906.6 MiB |

Latest A800 2.9B detailed `bench_batch_sweep.py` rows:

| Batch | Prefill tok/s | Forward decode tok/s | `rwkv7_forward_token` tok/s | Peak VRAM |
|---:|---:|---:|---:|---:|
| 1 | 848.4 | 19.9 | 93.6 | 6428.9 MiB |
| 2 | 2313.5 | 39.3 | 199.1 | 7262.5 MiB |
| 4 | 4261.5 | 77.8 | 388.5 | 8906.6 MiB |

Latest A800 `bench_quantization.py --quant-skip-policy memory` rows:

| Model | fp16 footprint | 8-bit footprint | 4-bit footprint | fp16 decode tok/s | 8-bit decode tok/s | 4-bit decode tok/s |
|---|---:|---:|---:|---:|---:|---:|
| 0.4B | 859.8 MB | 571.8 MB | 427.8 MB | 171.2 | 11.5 | 23.4 |
| 1.5B | 2913.3 MB | 1761.3 MB | 1185.3 MB | 139.7 | 10.9 | 22.7 |
| 2.9B | 5622.4 MB | 3222.4 MB | 2022.4 MB | 91.3 | 8.0 | 16.7 |

Latest A800 2.9B detailed quantization rows:

| Quantization | Model footprint | Peak VRAM | Prefill tok/s | Selected decode tok/s | Fast backend | Status |
|---|---:|---:|---:|---:|---|---|
| none/fp16 | 5622.4 MB | 5771.4 MiB | 1676.4 | 91.3 | native_graph | PASS |
| 8-bit bnb | 3222.4 MB | 4624.9 MiB | 705.4 | 8.0 | FLA | PASS, speed gap |
| 4-bit bnb | 2022.4 MB | 4250.6 MiB | 1273.3 | 16.7 | FLA | PASS, speed gap |

The 0.4B A800 Trainer and TRL SFT smoke rows also pass with nonzero trainable
parameter deltas. These rows validate the conservative Ampere policy on A800.
They do not promote native prefill-scan or quantized-speed kernels: W8/W4 reduce
footprint, but fp16 native_graph remains much faster end to end.

Issue #98 adds the missing A800 rows for 0.1B smoke/alignment/RL training, 7.2B
large-model smoke, 13.3B bnb W8/W4 80GB quantized smoke, single-GPU and 2-GPU
ZeRO-2/3 base/resume, and repository-native mm8/mm4 decode through 13.3B. Full details are in
[`docs/validation/A800_HF_VALIDATION.md`](docs/validation/A800_HF_VALIDATION.md).

0.1B A800 compatibility rows:

| Check | Result |
|---|---|
| `smoke_hf_generate` | PASS; fast token backend `native_graph` |
| `test_hf_api_contract` | PASS |
| `test_quantized_inference` 8-bit / 4-bit | PASS; footprint `283.4 MB` / `242.9 MB` |
| `test_peft_lora` | PASS; nonzero LoRA gradients |
| official alignment | PASS; top5 `1.0`, argmax `1.0`, cosine `0.9999957`, greedy `64/64` |
| Trainer / TRL SFT / TRL DPO / TRL GRPO | PASS; nonzero trainable deltas |

A800 native mm quantization rows from `bench_native_mm_quant_decode.py`,
prompt128/decode64, fp16 load, `min_params=8_000_000`:

| Model | Quantization | Replaced modules | Model footprint | Decode tok/s | vs fp16 |
|---|---|---:|---:|---:|---:|
| 0.4B | none | 0 | 859.8 MB | 185.2 | 1.00x |
| 0.4B | native mm8 | 1 | 796.0 MB | 187.9 | 1.01x |
| 0.4B | native mm4 | 1 | 764.0 MB | 185.8 | 1.00x |
| 1.5B | none | 0 | 2913.3 MB | 172.7 | 1.00x |
| 1.5B | native mm8 | 49 | 2019.4 MB | 27.5 | 0.16x |
| 1.5B | native mm4 | 49 | 1571.4 MB | 27.1 | 0.16x |
| 2.9B | none | 0 | 5622.4 MB | 110.7 | 1.00x |
| 2.9B | native mm8 | 65 | 3865.7 MB | 20.5 | 0.19x |
| 2.9B | native mm4 | 65 | 2985.7 MB | 19.5 | 0.18x |
| 7.2B | none | 0 | 13731.3 MB | 36.1 | 1.00x |
| 7.2B | native mm8 | 193 | 7340.5 MB | 17.0 | 0.47x |
| 7.2B | native mm4 | 193 | 4140.5 MB | 15.9 | 0.44x |
| 13.3B | none | 0 | 25309.1 MB | 10.2 | 1.00x |
| 13.3B | native mm8 | 367 | 13358.5 MB | 7.7 | 0.75x |
| 13.3B | native mm4 | 367 | 7374.5 MB | 8.6 | 0.84x |

Native mm8/mm4 works and reduces model footprint on A800 through 13.3B, but the
current 1.5B+ rows are slower than fp16. The regression comes from the default
`8_000_000` parameter gate replacing every per-layer FFN `key`/`value` matrix
plus `lm_head`; the current Triton dequant-GEMV kernels do not beat A800 fp16
cuBLAS on those decode shapes. A `50_000_000` gate leaves only `lm_head`
quantized and is roughly neutral for 1.5B/2.9B decode, but saves far less
footprint. Quantized speed therefore remains open until a native fused quant
kernel beats fp16 end to end.

Additional A800 issue #98 rows:

| Area | Model | Result |
|---|---|---|
| 7.2B larger smoke | 7.2B fp16 | PASS; footprint `13731.3 MB`, peak `13998.8 MiB`, generate `6.52 tok/s` |
| 13.3B quantized smoke | 13.3B bnb 8bit | PASS; footprint `13597.1 MB`, peak `20108.6 MiB`, decode `3.9 tok/s` |
| 13.3B quantized smoke | 13.3B bnb 4bit | PASS; footprint `7741.1 MB`, peak `18998.6 MiB`, decode `8.4 tok/s` |
| DeepSpeed ZeRO-2 | 0.4B bf16, 1 GPU | PASS; loss `1.9297`, trainable delta `0.000100` |
| DeepSpeed ZeRO-3 | 0.4B bf16, 1 GPU | PASS; loss `1.9297`, trainable delta `0.000100` |
| ZeRO-2 checkpoint resume | 0.4B bf16, 1 GPU | PASS; resumed to global step `2`, resume loss `1.5781` |
| ZeRO-3 checkpoint resume | 0.4B bf16, 1 GPU | PASS; resumed to global step `2`, resume loss `1.5938` |
| DeepSpeed ZeRO-2 | 0.4B bf16, 2 GPU | PASS; loss `5.1328`, trainable delta `0.000100` |
| DeepSpeed ZeRO-3 | 0.4B bf16, 2 GPU | PASS; loss `5.1328`, trainable delta `0.000100` |
| ZeRO-2 checkpoint resume | 0.4B bf16, 2 GPU | PASS; resumed to global step `2`, resume loss `2.4336` |
| ZeRO-3 checkpoint resume | 0.4B bf16, 2 GPU | PASS; resumed to global step `2`, resume loss `2.4453` |

A800 80GB VRAM coverage now includes fp16 decode/smoke rows for 0.4B / 1.5B /
2.9B / 7.2B / 13.3B, bnb 8bit/4bit rows through 13.3B, and native mm8/mm4 rows
through 13.3B. The bnb and current native-mm quant rows are memory evidence,
not quantized-speed wins.

### RTX A6000 validation sweep

Issue #115 was validated on 2026-07-04 with 1x and 2x `NVIDIA RTX A6000`
(`sm_86`, 48GB-class visible memory on GPUs 2/3). Rows were appended to
`bench/results.jsonl` lines 659-786. The card follows the conservative Ampere
policy: stable output/recurrent-output fusions are allowed, while prefill-scan,
projection/LoRA fusions, and quantized-speed promotion remain opt-in.
The quantization rows below are functional, memory-footprint, and decode
telemetry evidence; they are not a quantized-throughput pass.

Environment captured with `scripts/print_env.sh`:

- GPU: `NVIDIA RTX A6000 sm_86`; ZeRO used `CUDA_VISIBLE_DEVICES=2,3`.
- Driver / CUDA: NVIDIA driver `580.82.07`, `nvidia-smi` CUDA `13.0`.
- Runtime: Python `3.10.12` from the draft venv, PyTorch `2.12.1+cu130`,
  Transformers `5.13.0`, PEFT `0.19.1`, TRL `1.7.1`, DeepSpeed `0.19.2`,
  bitsandbytes `0.49.2`, FLA `0.5.1`.

Representative commands:

```bash
PYTHON_BIN=/path/to/python \
MODEL_ROOT=/path/to/rwkv_models \
A6000_SINGLE_VISIBLE_DEVICES=2 \
A6000_MULTI_VISIBLE_DEVICES=2,3 \
bash bench/run_a6000_hf_validation.sh

CUDA_VISIBLE_DEVICES=2 PYTHON_BIN=/path/to/python \
  bash scripts/print_env.sh

CUDA_VISIBLE_DEVICES=2,3 PYTHON_BIN=/path/to/python \
  bash scripts/print_env.sh
```

Core smoke status:

| Check | Result |
|---|---|
| 0.1B `smoke_hf_generate` | PASS |
| 0.1B `test_hf_api_contract` | PASS |
| 0.1B `test_quantized_inference` 8-bit / 4-bit | PASS / PASS |
| 0.1B `bench_speed.py` | PASS; fp16 prefill `6759.0` tok/s, forward decode `90.4` tok/s, peak `636.4` MiB |
| 0.1B `bench_batch_sweep.py` | PASS; native_graph `rwkv7_forward_token` bsz1/8 `582.5` / `3872.2` tok/s |
| 0.1B native mm8/mm4 decode telemetry | PASS; none/mm8/mm4 `532.8` / `536.2` / `508.5` tok/s |

Single-GPU large-model smoke, `attn_mode=fused_recurrent`:

| Model | Dtype | Actual param dtype | Footprint | Peak VRAM |
|---|---|---|---:|---:|
| 0.4B | fp16 | `torch.float16` | 859.8 MB | 1124.5 MiB |
| 0.4B | bf16 | `torch.bfloat16` | 859.8 MB | 903.6 MiB |
| 1.5B | fp16 | `torch.float16` | 2913.3 MB | 3178.6 MiB |
| 1.5B | bf16 | `torch.bfloat16` | 2913.3 MB | 3178.6 MiB |
| 2.9B | fp16 | `torch.float16` | 5622.4 MB | 5888.0 MiB |
| 2.9B | bf16 | `torch.bfloat16` | 5622.4 MB | 5888.0 MiB |
| 7.2B | fp16 | `torch.float16` | 13731.3 MB | 13997.8 MiB |
| 7.2B | bf16 | `torch.bfloat16` | 13731.3 MB | 13997.8 MiB |

Latest A6000 `bench_batch_sweep.py` native_graph decode rows
(`rwkv7_forward_token`):

| Model | Dtype | Batch 1 tok/s | Batch 2 tok/s | Batch 4 tok/s | Batch 8 tok/s | Largest batch peak VRAM |
|---|---|---:|---:|---:|---:|---:|
| 0.4B | fp16 | 286.3 | 453.9 | 894.8 | 1750.2 | 2867.4 MiB |
| 0.4B | bf16 | 284.9 | 437.1 | 861.4 | 1675.7 | 2867.4 MiB |
| 1.5B | fp16 | 149.9 | 260.4 | 504.1 | - | 4904.1 MiB |
| 1.5B | bf16 | 149.7 | 245.6 | 476.2 | - | 4904.1 MiB |
| 2.9B | fp16 | 81.7 | 148.1 | - | - | 7261.5 MiB |
| 2.9B | bf16 | 80.5 | 143.3 | - | - | 7261.5 MiB |
| 7.2B | fp16 | 41.4 | 78.7 | - | - | 16336.1 MiB |
| 7.2B | bf16 | 41.4 | 77.8 | - | - | 16336.1 MiB |

Latest A6000 bnb quantization rows, `--quant-skip-policy memory`
(footprint improves; decode is slower than fp16):

| Model | fp16 footprint | 8-bit footprint | 4-bit footprint | fp16 decode tok/s | 8-bit decode tok/s | 4-bit decode tok/s |
|---|---:|---:|---:|---:|---:|---:|
| 0.4B | 859.8 MB | 571.8 MB | 427.8 MB | 261.9 | 18.5 | 36.2 |
| 1.5B | 2913.3 MB | 1761.3 MB | 1185.3 MB | 142.9 | 17.2 | 36.1 |
| 2.9B | 5622.4 MB | 3222.4 MB | 2022.4 MB | 80.1 | 13.1 | 26.8 |
| 7.2B | 13731.3 MB | 7587.3 MB | 4515.3 MB | 40.8 | 13.0 | 27.2 |

Latest A6000 repository-native mm quant decode telemetry rows:

| Model | Native mm mode | Replaced modules | Footprint | Decode tok/s | Status |
|---|---|---:|---:|---:|---|
| 0.1B | none / mm8 / mm4 | 0 / 1 / 1 | 364.4 / 316.6 / 292.6 MB | 532.8 / 536.2 / 508.5 | PASS |
| 0.4B | none / mm8 / mm4 | 0 / 1 / 1 | 859.8 / 796.0 / 764.0 MB | 261.7 / 265.8 / 264.6 | PASS |
| 1.5B | none / mm8 / mm4 | 0 / 49 / 49 | 2913.3 / 2019.4 / 1571.4 MB | 142.8 / 38.1 / 35.2 | PASS |
| 2.9B | none / mm8 / mm4 | 0 / 65 / 65 | 5622.4 / 3865.7 / 2985.7 MB | 78.3 / 25.7 / 31.9 | PASS |
| 7.2B | none / mm8 / mm4 | 0 / 193 / 193 | 13731.3 / 7340.5 / 4140.5 MB | 41.0 / 26.4 / 23.0 | PASS |

The quantization rows validate functional W8/W4 loading, footprint reduction,
and real native mm8/mm4 decode coverage. They do not close the quantized-speed
gate: generic bnb W8/W4 are slower than fp16 on every A6000 model measured, and
native mm8/mm4 only matches or slightly beats fp16 on the small 0.1B/0.4B cases
under the current replacement policy. Larger native-mm rows remain slower than
fp16, so quantized speed stays open pending fused/native quant kernels.

Training rows:

| Model | Single-GPU Trainer/SFT/DPO | HF Trainer resume | 2x A6000 ZeRO-2/3 | 2x A6000 ZeRO-2/3 resume |
|---|---|---|---|---|
| 0.1B | PASS | PASS | - | - |
| 0.4B | PASS | PASS | PASS / PASS | PASS / PASS |
| 1.5B | PASS | PASS | PASS / PASS | PASS / PASS |
| 2.9B | PASS | PASS | PASS / PASS | PASS / PASS |

Cross-card summary: A6000 (`sm_86`, 48GB) behaves like a conservative Ampere
workstation card. It extends the A800/A100 Ampere coverage with a 48GB
single-card matrix and confirms 7.2B fp16/bf16 inference fits comfortably under
48GB. Decode throughput is below the 80GB A800 rows for 2.9B bsz1/2
(`81.7`/`148.1` tok/s on A6000 vs `93.6`/`199.1` tok/s on A800), but A6000 now
has broader single-card training/resume and 2-card ZeRO-2/3 resume evidence
than the earlier A800 block. As on V100/A800, bnb quantization is a memory
fallback, not a speed win; native fused quant kernels remain the performance
work item before any quantized-speed promotion.

## HF speculative decoding smoke

`rwkv7_speculative_generate()` is the initial HF-only speculative decoding
helper. It keeps the target and draft as ordinary HF models, proposes greedy
draft spans, verifies them with block target forwards, and reports acceptance
telemetry:

```bash
python tests/test_speculative_decode.py \
  --model /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --max-new-tokens 8 \
  --draft-tokens 4
```

The default smoke uses the same model as target and draft, so every proposed
token should be accepted and the sequence must match greedy `generate()`.
Passing `--draft-model /path/to/smaller-hf-rwkv` exercises the same API with a
real draft model. The real-draft benchmark records the production gate row:

```bash
python bench/bench_speculative_decode.py \
  --target-model /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.4b-hf \
  --draft-model /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --attn-mode fused_recurrent \
  --max-new-tokens 8 \
  --draft-tokens 4 \
  --results bench/results.jsonl
```

Latest V100 row: target greedy and speculative outputs match for 8/8 new
tokens, the 0.1B draft proposes 9 tokens, accepts 7, corrects 1, resyncs once,
replays 3 cache-resync tokens instead of 11 full-prefix tokens, and reports
acceptance `0.7777777777777778`. The short V100 row now reaches `2.1079x`
speedup over target greedy; next work is validating longer prompts and better
draft/block-size choices.

## HF RL / ZeRO training smoke

`tests/test_hf_rl_training_smoke.py` covers one-step LoRA preference/RL training
through common TRL trainers:

```bash
python tests/test_hf_rl_training_smoke.py \
  --model /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --device cuda \
  --backend dpo

python tests/test_hf_rl_training_smoke.py \
  --model /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --device cuda \
  --backend grpo \
  --grpo-max-completion-length 2
```

`configs/deepspeed/zero2.json` and `configs/deepspeed/zero3.json` are
HF Trainer-compatible ZeRO presets with auto micro-batch, gradient accumulation,
fp16/bf16, and bucket sizing. Validate them with:

```bash
python tests/test_deepspeed_configs.py
```

`tests/test_deepspeed_training_smoke.py` is the executable ZeRO training
harness. It loads the HF adapter through `AutoModelForCausalLM`, attaches PEFT
LoRA adapters, runs one or more HF `Trainer` steps with `deepspeed=zero2/zero3`,
checks that loss is finite, checks that trainable parameters changed, and emits
`deepspeed_training_smoke` rows for the analyzer:

```bash
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH=/home/data/wangyue/projects/flash-linear-attention:$PYTHONPATH

python tests/test_deepspeed_training_smoke.py \
  --model /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --zero-stage both \
  --train-dtype fp32 \
  --max-steps 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --results bench/results.jsonl
```

On machines without DeepSpeed or live GPUs, use `--optional --results
/tmp/rwkv7_zero_optional.jsonl` to record explicit skip rows while keeping local
analyzer/report tests green.

Latest V100 validation: the server environment exposes torch CUDA through the
pip/conda CUDA runtime without a system CUDA toolkit, so the harness defaults
`DS_IGNORE_CUDA_DETECTION=1` for ZeRO-only smoke runs and seeds a one-process
`RANK/WORLD_SIZE/LOCAL_RANK` environment when no launcher is present. Full
2-process validation was also run with `torchrun --standalone --nproc_per_node=2`
on `CUDA_VISIBLE_DEVICES=0,1`. The committed rank-0 rows pass on 2 x Tesla V100
fp32, `max_steps=1`, `batch_size=1`, `max_length=16`: ZeRO-2 reports finite
loss `4.857278823852539` and `max_trainable_delta=9.999999747378752e-05`;
ZeRO-3 reports the same finite loss and trainable delta after gathering full
ZeRO-3 parameter shards for the update check.

## Benchmark gap report

`bench/analyze_results.py` turns accumulated JSONL rows into a target/gap report:

```bash
python bench/analyze_results.py \
  --results bench/results.jsonl \
  --device V100 \
  --dtype fp16
```

It reports HF-vs-official prefill/decode/memory ratios, best decode-breakdown
rows, fast-token API status, latest correctness row, batch/dynamic rows, decode
microbench rows, fast-token warmup and native-graph overhead rows, larger-model
smoke rows, quantization rows, `fused_backend_targets`, and a short next-focus
list. Current committed
V100 rows show:

| Metric | Current | Target | Status |
|---|---:|---:|---|
| speed_mem fast-token decode ratio (`native_jit`, bsz=1) | 1.00x official | >=0.90x | PASS |
| fast_decode best ratio (`native_graph`, bsz=1) | 2.77x official | >=0.90x | PASS |
| decode_breakdown fast-token ratio | ~0.57x official | >=0.90x | GAP |
| native_graph prototype decode ratio | ~2.76x official | >=0.90x | PASS prototype |
| native_graph warmup bsz=1/2/4/8 | cache contains 1/2/4/8 in 1.389s | preflight complete | PASS |
| native_graph replay overhead bsz=1/2/4/8 (0.1B fused) | API `637.9` / `1114.0` / `1852.8` / `3531.7` tok/s, max copy share `0.014`, hit rate `0.9906` | >=150 tok/s, <=0.15 copy share, >=0.80 hit rate | PASS |
| HF device_map generate smoke | 2 x V100, split layer 6, greedy tail matches single-device, fast backend skipped | >=2 CUDA devices, finite logits, greedy equality | PASS |
| speed_mem memory ratio | ~1.00x official | <=1.10x | PASS |
| 8-bit / 4-bit footprint ratio | 0.76x / 0.65x fp16 | lower is better | PASS smoke |
| 8-bit / 4-bit decode ratio | 0.24x / 0.67x fp16 | >=1.00x | GAP |
| Albatross V100 decode ratio | HF fused native-graph `0.629x`-`1.185x` over 0.1B/0.4B/1.5B × bsz1/2/4/8 | approach Albatross | P1 PASS; universal P2/P3 GAP |
| Albatross V100 prefill ratio | HF `0.787x`-`0.890x` on matching 0.4B/1.5B prompt512 rows | approach Albatross | P1 PASS; universal P2 GAP |
| Fused backend P1 decode ladder | analyzer target min ratio `>=0.55x` Albatross | `docs/performance/FUSED_BACKEND.md` P1 | PASS |
| Fused backend quant ladder | W8/W4 decode `>=1.0x` fp16 reference with W8 footprint `<=0.75x`, W4 footprint `<=0.55x` | native W8/W4 fused path | GAP |
| 0.4B converted-model smoke | hidden=1024, layers=24, generated=4, backend=native_graph | load + generate | PASS |
| 1.5B converted-model smoke | hidden=2048, layers=24, generated=2, backend=native_graph | load + generate | PASS |
| 2.9B converted-model smoke | hidden=2560, layers=32, generated=2, backend=native_graph | load + generate | PASS |
| 7.2B converted-model smoke | hidden=4096, layers=32, generated=2, backend=native_graph | load + generate | PASS |
| 13.3B converted-model smoke | hidden=4096, layers=61, generated=2, backend=native_jit | load + generate | PASS |

Apple MLX/Metal quant ratio evidence is recorded separately in
`bench/results_apple_silicon_m5_20260704.jsonl` and
`docs/hardware/APPLE_SILICON.md`. On the local M5 / 16GB prompt512/1024 decode16
matrix, same-shape fp16 Metal baselines show 0.4B W8/W4 decode at
`0.79x` / `0.81x` fp16 with peak memory `0.71x` / `0.57x`, and 1.5B W8/W4 decode
at `0.75x` / `0.84x` fp16 with peak memory `0.70x` / `0.55x`. The longer
prompt2048/decode128 ratio row reaches 0.4B W8/W4 decode `0.88x` /
`1.04x` fp16 with peak memory `0.71x` / `0.56x`; the same 1.5B row remains
below fp16 at W8/W4 decode `0.68x` / `0.73x` with peak memory `0.70x` /
`0.54x`. The W4 `--quant-backend auto` row now caches the auto decision and
favors Metal for normal prefill/decode rows (`metal=202885`): 0.4B W4 auto
prompt2048/decode128 reaches prefill/decode `60.61` / `59.73 tok/s`
(`0.88x` / `1.25x` fp16, peak `0.56x`), while 1.5B W4 auto reaches
`27.64` / `20.42 tok/s` (`0.93x` / `0.75x` fp16, peak `0.54x`). A newer
prompt4096/decode256 gate with chunk1024 still passes chunked/full prefill
(`max_abs=0.0`) and records 0.4B fp16 `94.08` / `75.38 tok/s` versus W4 auto
`62.01` / `55.29 tok/s` (peak `515 MB`, `0.56x` fp16), and 1.5B fp16
`35.34` / `33.21 tok/s` versus W8/Metal `22.52` / `20.54 tok/s` (peak
`2147 MB`, `0.70x` fp16) and W4 auto `27.40` / `25.46 tok/s` (peak
`1677 MB`, `0.54x` fp16). The new 1.5B prompt8192/decode512 chunk2048 row
also passes chunked/full prefill (`max_abs=0.0`): fp16 reaches `27.97` /
`26.02 tok/s`, while W4 auto reaches `22.77` / `21.20 tok/s` with peak
`1677 MB` (`0.54x` fp16) and `metal=811525`, or about `0.81x` fp16 for both
prefill and decode. A direct grouped R/K/V W4 row extends 1.5B to
prompt8192/decode1024 with chunk2048 and `quant_min_params=4000000`: it keeps
chunked/full prefill `max_abs=0.0`, records `21.09` / `20.48 tok/s`, peak
`1074.6 MB`, quantized-linear `metal=2507781`, grouped `metal=417792`, and
grouped fallback `0`. This strengthens long-decode and memory evidence but also
shows W4 does not yet stably beat fp16 at longer prompt/decode sizes; stable
W8/W4 speed `>=1.0x` fp16 across sizes and modes still requires deeper fused
kernels. The isolated MLX quant projection microbench
(`axis=mlx_quant_projection_bench`, 1.5B-sized 2048x2048 projection, rows=1/4)
now makes the bottleneck explicit: W4/Metal reaches `0.39x` dense for rows=1
and `1.11x` for rows=4 (auto `1.38x`), while W8/Metal reaches `0.87x` /
`0.67x` dense and W8 auto still routes to affine because the W8 session
exactness guard remains open. The new grouped-projection prototype
(`axis=mlx_quant_group_projection_bench`, groups=3) pre-packs grouped weights and
uses one Metal launch for R/K/V-style projection groups. It preserves exactness
versus separate Metal (`max_abs_vs_separate_metal=0.0`); W8 rows=1 improves to
`1.12x` dense and `1.10x` separate-Metal, while W8 rows=4 is still only `0.58x`
dense (`1.08x` separate). W4 grouped launch does not help yet (`0.75x` dense /
`0.93x` separate for rows=1; `0.62x` dense / `0.98x` separate for rows=4), so
W4 still needs a better packed reduction rather than launch fusion alone.
The model-level opt-in seam now exists behind
`RWKV7_MLX_GROUP_RKV_QUANT_PROJECTION=1`: when R/K/V are quantized, share
bit-width, and resolve to the Metal backend, MLX routes the three distinct R/K/V
inputs and their three existing packed weights through one Metal launch. The
new default `RWKV7_MLX_GROUP_RKV_QUANT_PROJECTION_MODE=direct` avoids the old
extra grouped packed-weight cache; `packed` remains available for A/B with the
prepacked microbench path. The default inference path remains unchanged, and
this is a correctness-gated integration point rather than a claim that W8/W4 now
stably beats fp16.

Initial packed-cache A/B rows showed the seam executes end-to-end with
`fallback=0`: 0.4B W4 auto prompt128/decode8 improved from `39.33` /
`38.68 tok/s` to `44.33` / `41.38 tok/s` (`metal=12672`), 1.5B W4 auto
improved from `19.03` / `18.37` to `20.18` / `19.03 tok/s` (`metal=6336`),
0.4B W8/Metal improved from `40.32` / `38.93` to `43.01` / `42.80 tok/s`
(`metal=6336`), and a shorter 1.5B W8/Metal prompt64/decode4 row improved from
`17.62` / `17.22` to `19.02` / `17.53 tok/s` (`metal=3168`). The direct
no-copy path keeps the useful W4 signal while removing the duplicated-weight
memory penalty: 0.4B W4 auto prompt128/decode8 records `43.30` / `42.54 tok/s`,
peak `364.7 MB`, `metal=6336`, and chunked/full prefill `max_abs=0.0`; 1.5B
W4 auto records `20.95` / `19.52 tok/s`, peak `1074.6 MB`, `metal=6336`, and
`max_abs=0.0`. These peaks are back near the separate-path baselines and well
below the old packed-cache grouped peaks (`415.9 MB` / `1226.8 MB`). Longer
direct rows also pass: 0.4B W4 auto prompt512/decode16 reaches `45.83` /
`45.17 tok/s`, peak `364.7 MB`, `metal=24960`; 1.5B W4 auto prompt512/decode16
reaches `20.69` / `19.28 tok/s`, peak `1074.6 MB`, `metal=24960`; 0.4B W8/Metal
prompt512/decode16 reaches `44.50` / `41.50 tok/s`, peak `549.3 MB`,
`metal=24960`; and 1.5B W8/Metal prompt512/decode16 reaches `19.81` /
`19.27 tok/s`, peak `1745.6 MB`, `metal=24960`. The broader-threshold
prompt2048/decode128 direct rows, which include R/K/V in the quantized set, also
pass with grouped fallback `0` and chunked/full prefill `max_abs=0.0`: 0.4B W4
records `46.50` / `43.70 tok/s`, peak `364.7 MB`, `metal=101376`; 0.4B W8
records `42.52` / `41.36 tok/s`, peak `549.3 MB`, `metal=101376`; 1.5B W4
records `21.31` / `19.63 tok/s`, peak `1074.6 MB`, `metal=101376`; and 1.5B
W8 records `20.47` / `19.78 tok/s`, peak `1745.6 MB`, `metal=101376`. Against
the same-shape fp16 baselines (`47.97 tok/s` decode at 0.4B and `27.20 tok/s`
at 1.5B), these direct broad-threshold rows improve memory (`~0.40x/0.60x` peak
for 0.4B W4/W8 and `~0.35x/0.57x` for 1.5B W4/W8) but still do not close the
stable fp16-beating speed gate. New direct W4 prompt4096/decode256 rows extend
that matrix while keeping chunked/full prefill `max_abs=0.0`: 0.4B with the
broader `quant_min_params=500000` threshold records `52.05` / `45.05 tok/s`,
peak `364.7 MB`, grouped `metal=202752`, fallback `0`; 1.5B with
`quant_min_params=4000000` records `21.14` / `19.98 tok/s`, peak `1074.6 MB`,
grouped `metal=202752`, fallback `0`. The same 1.5B direct grouped W4 route now
also passes prompt8192/decode1024 with chunked/full prefill `max_abs=0.0`,
`21.09` / `20.48 tok/s`, peak `1074.6 MB`, grouped `metal=417792`, fallback
`0`, and quantized-linear `metal=2507781`. A 0.4B `quant_min_params=4000000`
control row records `52.24` / `47.81 tok/s`, peak `514.9 MB`, and grouped
fallback `202752`, confirming that the lower threshold is needed to include
0.4B R/K/V in the direct grouped path. Direct grouped session pressure now covers
0.4B 4-session rounds4,4, 0.4B 6-session rounds8,8 repeat=2, 0.4B 8-session
rounds8,8 repeat=2, and 1.5B 5-session rounds4,4 / rounds8,8 repeat=2 probes.
The 0.4B rows keep one-shot token/text/seen-token checks passing with grouped
fallback `0`: W4 6-session aggregate round min `75.31 tok/s`, peak `466.5 MB`,
`metal=10176`; W8 6-session aggregate round min `91.71 tok/s`, peak
`651.1 MB`, `metal=15552`; W4 8-session aggregate round min `97.85 tok/s`, peak
`505.4 MB`, `metal=17472`; and W8 8-session aggregate round min `91.08 tok/s`,
peak `690.0 MB`, `metal=17472`. For 1.5B, W4/W8 rounds4,4 direct rows pass
with aggregate round mins `27.33` / `23.45 tok/s` and peaks `1239.0` /
`1910.1 MB`. The longer 1.5B rounds8,8 repeat=2 direct path is currently safe
under sequential scheduling (`19.49` / `18.35 tok/s` aggregate round min for
W4/W8, peaks `1126.1` / `1797.2 MB`, grouped fallback `0`). A strict compare
shows 1.5B W8 direct batched still matches sequential and one-shot tokens
(`26.02` / `25.04 tok/s` aggregate round decode), but 1.5B W4 direct batched
mismatches one-shot on two synthetic sessions (first mismatch indices `6` and
`9`, grouped fallback `0`), so the long 1.5B W4 batched direct route remains a
correctness gap rather than a production path. `SESSION_BACKEND=auto` now guards
W4/Metal the same way as W8/Metal and falls back with
`auto_mm4_metal_batch_exactness_guard`; the new 1.5B W4 direct auto row keeps
one-shot token/text/seen checks passing for 5 sessions, rounds8,8, repeat=2,
with aggregate round min `20.67 tok/s`, peak `1126.1 MB`, grouped fallback `0`,
and `metal=13632`. Follow-up direct grouped pressure rows extend the same seam:
0.4B broad-threshold W4 direct grouped `SESSION_BACKEND=batched` now passes
12-session rounds8,8 repeat=3 with one-shot token/text/seen-token checks,
aggregate round min `93.92 tok/s`, peak `583.6 MB`, grouped `metal=50112`,
fallback `0`, and quantized-linear `metal=301368`; 1.5B direct W4
`SESSION_BACKEND=auto` now passes 5-session rounds8,8 repeat=4 under
`auto_mm4_metal_batch_exactness_guard` with aggregate round min `12.77 tok/s`,
peak `1126.1 MB`, grouped `metal=31296`, fallback `0`, and quantized-linear
`metal=188456`. The latter is intentionally safe sequential scheduling, not a
claim that true 1.5B W4 batched direct is production-ready. Longer end-to-end
ratio gates are still required before enabling true W4 batched direct by
default. An opt-in 1.5B W4 direct grouped strict compare now closes this same
matrix with `SESSION_BACKEND=batched_stable`,
`RWKV7_MLX_SESSION_STABLE_ARGMAX_MODE=repair`, and tolerance `0.0625`:
sequential vs batched_stable matches backend tokens/text, both sides match
one-shot, seen-token checks pass for 5 sessions and rounds8,8, the structured
repair counts are `[2, 3]`, aggregate round min is `25.32 tok/s`, peak is
`1434.0 MB`, grouped `metal=10320`, fallback `0`, and quantized-linear
`metal=62116`. This is a correctness bring-up path that selectively replays
low-margin rows; it is not yet the default production W4 batched route.
Quant+Metal session-batch pressure rows also pass: 0.4B W8/W4 4-session
repeat=2 reaches min decode `40.18` / `41.17 tok/s` with peak `669` /
`534 MB`, and the higher-concurrency 6-session repeat=3 row reaches min decode
`34.33` / `27.14 tok/s` with peak `682` / `547 MB`. 1.5B W8/W4
4-session repeat=1 reaches min decode `19.58` / `20.38 tok/s` with peak
`2185` / `1716 MB`, and the 5-session repeat=2 row reaches min decode
`15.60` / `18.87 tok/s` with peak `2198` / `1728 MB`. Longer session
pressure now also covers 0.4B W4 8-session rounds8,8 repeat=2
(aggregate round min `103.91 tok/s`, peak `656 MB`) plus 1.5B W4/W8
5-session rounds8,8 repeat=2 (`29.63 tok/s` batched W4 aggregate round min,
`18.38 tok/s` safe-auto W8 aggregate round min, peaks `1841` / `2198 MB`).
The opt-in equal-round
`SESSION_BACKEND=batched` path also has initial W4 correctness rows: 0.4B
6-session repeat=2 passes with per-session min decode `19.00 tok/s`,
aggregate round min decode `105.44 tok/s`, and peak `617 MB`; 1.5B 5-session
repeat=1 passes with per-session min decode `6.61 tok/s`, aggregate round min
decode `32.38 tok/s`, and peak `1841 MB`. During W8/Metal strict batched
decode bring-up, larger 0.4B multi-round exactness diverged from one-shot greedy
tokens, so `SESSION_BACKEND=auto` now records an
`auto_mm8_metal_batch_exactness_guard` reason and falls back to sequential for
W8/Metal while W4 uses the batched path. Safe W8/Metal auto rows pass for 0.4B
6-session repeat=2 (min decode `39.80 tok/s`, peak `682 MB`) and 1.5B
5-session repeat=1 (min decode `17.43 tok/s`, peak `2198 MB`). A dedicated
`axis=mlx_session_batch_backend_compare` row now compares sequential vs batched
without hiding mismatches: 0.4B W4 and 1.5B W4 both match each other and
one-shot tokens (`all_backend_token_match=true`,
`all_right_one_shot_token_match=true`) with batched aggregate round mins
`145.89` and `38.31 tok/s`; 1.5B W8 also matches in this matrix with
batched aggregate round min `34.67 tok/s`; 0.4B W8 reproduces the exactness gap
(`backend_compare_status=mismatch`, first mismatch at token index `6` for the
short prompt). A new optional `--trace-mismatch-logits` row localizes that gap:
at step 6 the sequential path has an exact tie between tokens `11` and `261`
(logits `8.476562` / `8.476562`, `mx.argmax` selects `11`), while the batched
Metal path shifts token `11` down to `8.46875` and selects `261`; the traced
left/right max-abs logit delta at that step is only `0.03125`. An explicit
`SESSION_BACKEND=batched_stable` low-margin argmax policy closes this traced
0.4B W8/Metal compare gate: 3-session and 6-session strict rows both match
sequential and one-shot tokens, with the 6-session row reaching batched
aggregate round mins `162.12` / `163.72 tok/s` (`metal=20378`, peak `790 MB`).
The stable policy now has longer/repeat coverage too: 0.4B W8/Metal
`batched_stable` 8-session rounds8,8 repeat=2 matches one-shot tokens with
aggregate round min `184.62 tok/s` (peak `790 MB`). 1.5B W8/Metal 5-session
rounds8,8 repeat=2 matches one-shot tokens with aggregate round min
`53.66 tok/s` (peak `2311 MB`), and the stronger repeat=4 pressure row still
matches one-shot tokens/text/seen-tokens with aggregate round min `26.11 tok/s`,
peak `2311 MB`, and `metal=50728`. The same 1.5B W4 auto 5-session
rounds8,8 repeat=4 pressure row also matches one-shot with aggregate round min
`30.94 tok/s`, peak `1841 MB`, and `metal=50728`. These repeat=4 rows are
stability evidence and also show throughput degradation under sustained local
M5/16GB pressure. A 1.5B strict sequential-vs-batched-stable compare for
rounds8,8 also passes. The default W8/Metal auto path remains
guarded, but `RWKV7_MLX_SESSION_AUTO_W8_STABLE=1` now opts `SESSION_BACKEND=auto`
into this stable policy; a 0.4B W8/Metal auto row selects `batched_stable` and
passes with aggregate round min `90.73 tok/s` (`metal=5126`). The MLX quant
backend now also has a conservative `--quant-backend auto` policy with
backend-count telemetry: W4 auto selects the Metal fused dequant-projection path
for normal row counts and the 0.4B 3-session sequential-vs-batched gate passes
with `quantized_linear_last_backend_counts` showing `metal=4913` and batched
aggregate round mins `78.68` / `69.17 tok/s`; W8 auto stays on the affine path
by default unless W8 Metal is explicitly enabled, and the 0.4B W8 auto
`SESSION_BACKEND=auto` row batches safely with `affine=5126` and aggregate round
min `49.76 tok/s`. These rows validate the batching seam, safe backend routing,
and telemetry, not the final fp16-beating quant speed gate.

The current next-focus list is: 13.3B official-alignment/speed sweeps are now
done (cos~1.0, `native_jit` 18.4 tok/s on V100; see
[13.3B official alignment + decode speed](#133b-official-alignment--decode-speed));
remaining: validate newer GPUs, and solve the generic bnb quantized decode speed gap. The bsz=1 HF fast-token target is exceeded by `native_graph`;
bsz=2/4/8 native-graph serving now reaches `434.3` / `852.6` / `1539.1`
aggregate tok/s, and preflight warmup confirms graph runners are captured for
bsz=1/2/4/8 before the first serving request. The native-graph overhead rows
confirm the fused 0.1B public API scales to `3531.7` aggregate tok/s at bsz=8
while cache-copy overhead stays below `1.4%` of measured manual replay wall
time and graph-runner cache hit rate stays at `0.9906` for all required batch sizes. The
HF `device_map` row validates the multi-GPU pipeline-parallel direction on
2 x V100 by splitting the 0.1B model at layer 6; normal cached `generate()`
keeps finite logits, bypasses the single-device fast-token backend, and matches
the single-device greedy tail.

### Albatross A/B baseline

`bench/bench_albatross.py` ingests Albatross `RESULT B=... T=...` rows into the
same JSONL report used by the HF benchmarks:

```bash
python bench/bench_albatross.py \
  --engine faster3a \
  --engine-config wkv=fp32io16 \
  --model /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-20260129-ctx8192.pth \
  --model-size-label 0.1b \
  --device-name 'Tesla V100-PCIE-32GB' \
  --cases '1x1,1x2,1x4,1x8,1x16,1x32,1x64,1x128,1x256,1x512,2x1,4x1,8x1,16x1,32x1,2x2,4x4,8x8,16x16' \
  --warmup 1 \
  --iters 3 \
  --results bench/results.jsonl \
  -- --wkv fp32io16
```

On V100, Albatross faster3a's default fp16 WKV kernel uses `cp.async` and does
not compile for sm70, so the recorded V100 baseline uses `--wkv fp32io16`.
Latest 0.1B decode rows show Albatross `788.19` / `1504.72` / `2612.88` /
`3611.88` tok/s for bsz=1/2/4/8. The fused HF API reaches
`637.9` / `1114.0` / `1852.8` / `3531.7`, or
`0.809x` / `0.740x` / `0.709x` / `0.978x`. Matching 0.4B and 1.5B rows are
also retained; across all 12 rows the ratio spans `0.629x-1.185x` and the P1
floor passes. Full commands, VRAM, raw JSONL, and excluded-contention notes are
in [`bench/v100_sm70_decode_gap_20260710/README.md`](bench/v100_sm70_decode_gap_20260710/README.md).

## Benchmark regression and target gates

`bench/check_results.py` turns the report into an executable gate:

```bash
# Passing regression gate for the current PR/V100 baseline.
python bench/check_results.py \
  --results bench/results.jsonl \
  --device V100 \
  --dtype fp16

# Final acceptance gate for the current V100 0.1B HF fast-token target.
python bench/check_results.py \
  --results bench/results.jsonl \
  --device V100 \
  --dtype fp16 \
  --target
```

Current committed V100 rows pass both the regression gate and the target gate.
The gate now uses the native-JIT HF fast-token speed row (`92.1 tok/s` vs
official `92.1 tok/s`) for the low-memory 0.1B bsz=1 target, while the
`fast_decode` section reports the optional native-graph row at `255.5 tok/s`.
It also requires passing 0.4B, 1.5B, 2.9B, 7.2B, and 13.3B `larger_model_smoke` rows with checkpoint
SHA256 and generated-token evidence.

## Current optimization target

The next optimization work should focus on **HF recurrent decode**:

1. Continue beyond the first cache optimization: `RWKV7StateCache` removes generic
   FLA CacheLayer bookkeeping, but the remaining gap requires reducing per-layer
   tiny kernels and Python dispatch in the one-token path.
2. Inspect FLA `Cache.update`, per-layer state gather/update, token shift, group norm,
   and output projection overhead in the single-token path.
3. Profile one-token decode with `torch.profiler` / Nsight and compare against official
   `rwkv` package layer-by-layer. `profile_decode.py --hf-decode-api rwkv7_forward_token` profiles the fast token API directly.
4. Benchmark the new batched `rwkv7_forward_token` API with `bench_speed.py --hf-decode-api rwkv7_forward_token`, `bench_batch_sweep.py --fast-decode-api true`, and `bench_decode_breakdown.py --fast-decode-api true`; the V100 result is now stable enough that ordinary eval/no-grad HF `forward`/`generate` use the same path by default, while benchmarks can still disable it with `RWKV7_FAST_FORWARD=0` for reference timing.
5. Use `bench_batch_sweep.py` to keep bsz=1/2/4/8 regressions visible while optimizing the batched fast decode path.
6. Use `tests/test_dynamic_batch_cache.py` and `bench_dynamic_batch.py` to keep heterogeneous-row cache reorder/drop behavior correct while approaching serving-style dynamic batching.
7. Use `tests/test_chunked_prefill.py` and `bench_chunked_prefill.py` to keep long-prompt chunked prefill logits/cache compatible with full prefill while measuring the memory/throughput tradeoff.
8. Use `bench_decode_micro.py` to separate recurrent model cost from `lm_head`, argmax, and Python loop overhead before changing the decode implementation.
9. Use `bench_decode_components.py` to choose the next fusion target inside the fast-token layer path.
10. Use `bench_projection_lora.py` to verify projection/LoRA fusion candidates before changing model code.
11. Use `bench/analyze_results.py` after every V100 run to verify target ratios and missing axes before choosing the next optimization.
12. Use `bench/check_results.py` as the regression gate, and `bench/check_results.py --target` as the final performance gate.
13. Use `rwkv7_warmup_fast_token()` and `bench_fast_token_warmup.py` to remove
   first-request native-graph capture from serving latency before measuring
   production traffic.
14. Use `bench_native_graph_overhead.py` to keep cache-copy/bind overhead around
   the captured graph below the gate while optimizing dynamic serving paths.
15. Use `bench_speculative_decode.py` to keep real-draft greedy equality and
   acceptance telemetry gated while optimizing HF speculative decoding.
16. Keep `logits_to_keep=1` as the default serving benchmark path because it already
   fixes the earlier excess-memory measurement.
17. After V100 decode approaches official `rwkv`, rerun on newer GPUs and larger models.

## Loop state

- Correctness tests are now strong enough for 0.1B smoke: prompt logits, greedy 64,
  and save/reload roundtrip.
- Memory for the serving-style HF path is now at parity with official on V100.
- First V100 decode optimizations landed: `fuse_norm=false` plus the exact-match
  `RWKV7StateCache` keep the real remote-code HF path at ~41 tok/s vs official
  ~92 tok/s on V100.
- Batch correctness and sweep harnesses are in place; V100 native-JIT bsz=1/2/4/8
  fast-token decode runs at `91.5` / `195.3` / `374.5` / `647.3` aggregate tok/s.
- HF native-graph fast-token is now integrated for fixed bsz=1/2/4/8; V100
  speed_mem reaches `255.5 tok/s`, batch sweep reaches `253.9` / `434.3` /
  `852.6` / `1539.1` aggregate tok/s, and dynamic reorder/drop reaches
  `1209.3` total tok/s through the explicit cache select API while using the
  normal HF prefill/cache handoff. The graph runner cache is now an LRU over
  active batch sizes instead of a single most recent runner, so dynamic serving
  does not recapture when a retained size reappears.
- Dynamic-batch cache reorder/drop correctness and benchmark harnesses are in
  place; V100 tests now cover non-inplace reorder plus compact/drop through
  `select_batch` / `batch_select`, plus detach and CPU offload/restore before
  continuing decode.
- Chunked prefill helper, correctness test, benchmark, analyzer section, and
  regression gate are in place. V100 bsz=2 prompt=512 chunked prefill matches
  full prefill/decode within fp16 tolerance; chunk sizes 64/128/256 reduce peak
  VRAM to `0.598x` / `0.616x` / `0.633x` of full prefill while reaching
  `0.125x` / `0.252x` / `0.499x` of full-prefill throughput.
- Decode microbench harness is in place; V100 shows `rwkv7_forward_token` at
  `16.8 ms/token` vs HF `forward` at `24.5 ms/token`, while `lm_head` and argmax
  are tiny.
- Decode component harness is in place; V100 shows `attn_linears_lora` is the
  largest remaining fast-token component at about `9.87 ms/token`.
- Projection/LoRA harness is in place; V100 shows naive PyTorch bmm grouping is
  slower overall, so custom fusion is needed.
- Quantization smoke and benchmark harnesses are in place; V100 bnb 8-bit/4-bit
  loads pass and reduce model footprint, but current generic bnb decode is
  slower than fp16.
- Real-draft HF speculative benchmark is in place; V100 0.1B draft -> 0.4B
  target matches target greedy for 8/8 new tokens with 7/9 accepted proposals
  and one correction/resync; cached-prefix resync saves 8 token replays and the
  short V100 row reaches `2.1079x` speedup over target greedy.
- Benchmark gap analysis is in place and currently identifies decode throughput
  as the active optimization gap.
- Benchmark check gate is in place: current regression gate passes, target gate
  now passes after the opt-in HF `native_jit` fast-token backend reached
  `1.00x` official for the bsz=1 V100 speed row.
- Latest `main` added a native RWKV-7 decode experiment for 50-series / Blackwell:
  `rwkv7_hf/native.py`, `rwkv7_hf/native_jit.py`, and `bench/bench_batch.py`.
  This is valuable as a next V100 experiment because it attacks the same tiny
  kernel / dispatch bottleneck with a TorchScript block step and CUDA graph.
- Formal V100 native-decode row is now recorded: native JIT reaches `103.52 tok/s`
  and native CUDA graph reaches `254.33 tok/s` on the 0.1B V100 smoke model, with
  graph-vs-JIT greedy equality `16/16`.
- The active V100 blocker has moved from decode parity to additional
  larger-model/newer-GPU and quantized serving validation: bsz=1 native-graph HF
  is at `255.5 tok/s` vs official `92.1`, bsz=2/4/8 native-graph reaches
  `434.3`, `852.6`, `1539.1` aggregate tok/s in the latest sweep, and the real
  0.4B, 1.5B, 2.9B, 7.2B, and 13.3B converted HF directories now pass load/forward/generate smoke
  on V100; the 13.3B row uses native-JIT to avoid native-graph memory overhead on 32GB V100.

### Batched native-JIT fast-token results

Latest V100 `bench_batch_sweep.py --fast-token-backend native_jit` rows:

| bsz | HF forward total tok/s | native-JIT fast-token total tok/s | per-seq fast tok/s | step ms |
|---:|---:|---:|---:|---:|
| 1 | 41.4 | 91.5 | 91.5 | 10.92 |
| 2 | 84.0 | 195.3 | 97.7 | 10.24 |
| 4 | 167.0 | 374.5 | 93.6 | 10.68 |
| 8 | 331.5 | 647.3 | 80.9 | 12.36 |

Latest V100 `bench_batch_sweep.py --fast-token-backend native_graph` rows:

| bsz | HF forward total tok/s | native-graph fast-token total tok/s | per-seq fast tok/s | step ms |
|---:|---:|---:|---:|---:|
| 1 | 40.5 | 253.9 | 253.9 | 3.94 |
| 2 | 80.8 | 434.3 | 217.1 | 4.61 |
| 4 | 159.3 | 852.6 | 213.2 | 4.69 |
| 8 | 317.7 | 1539.1 | 192.4 | 5.20 |

Dynamic-batch reorder/drop with `RWKV7_FAST_TOKEN_BACKEND=native_graph` now
reaches `1209.3` total tok/s for `832` decoded tokens with active batch dropping
from 8 to 4, compared with the latest forward row at `211.7` total tok/s and
the previous native-graph row at `524.7` total tok/s. Both latest rows report
`cache_select_api=true` and `final_cache_batch_size=4`, so the result is using
the production-facing cache compact/select path rather than only the beam
reorder hook.

### Chunked prefill results

Latest V100 `bench_chunked_prefill.py --batch-size 2 --prompt-tokens 512` rows:

| mode | chunk | prefill tok/s | speed vs full | peak VRAM | VRAM vs full | max diff | decode diff |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | - | 36447.0 | 1.0000 | 658.9 MB | 1.0000 | - | - |
| chunked | 64 | 4566.4 | 0.1253 | 394.0 MB | 0.5980 | 0.09375 | 0.09375 |
| chunked | 128 | 9185.5 | 0.2520 | 405.8 MB | 0.6159 | 0.046875 | 0.0625 |
| chunked | 256 | 18178.9 | 0.4988 | 417.1 MB | 0.6330 | 0.125 | 0.03125 |

## Latest main native-decode context (50-series / Blackwell)

`rwkv7_hf/native_jit.py` ports the official `RWKV_x070_TMix_one`/`CMix_one`
per-token math natively (no FLA backend at decode time) and captures the whole
fixed-shape decode step in a CUDA graph. On the latest `main` branch, this path
was validated on RTX 5070 Laptop / Blackwell sm_120 and larger smoke models.

Decode speed (0.1B, RTX 5070 Laptop, fp16, single batch):

| path | tok/s | note |
|---|---:|---|
| FLA HF adapter (`generate`) | 37 | original wrapper path |
| native eager | 40 | direct Python native math |
| native + `torch.jit.script` | ~78 | full-block fused |
| native + CUDA graph | ~395 | about 4x official `rwkv` at 99 tok/s |

Correctness claims from the latest `main` branch:

- forward logits vs FLA: cosine 1.000000, max_abs approximately 0 at fp32.
- CUDA-graph greedy decode: 40/40 tokens identical to the JIT path.
- end-to-end vs `model.generate()` greedy: 32/32 generated tokens identical.

### Production TTFT/TPOT + batch generate (RTX 5070 Laptop, sm_120, 0.1B fp16)

`bench/bench_ttft_tpot.py`, native_graph fast-token backend, `RWKV7_FAST_FORWARD=1`,
`attn_mode=fused_recurrent`, `fuse_norm=false`.

TTFT (time-to-first-token, bsz=1, p50):

| input len | TTFT p50 | TTFT p99 | prefill tok/s |
|---|---:|---:|---:|
| 32 | 19.1 ms | 20.4 ms | 1,676 |
| 128 | 23.6 ms | 24.1 ms | 5,430 |
| 512 | 24.0 ms | 26.9 ms | 21,318 |

TPOT (per-output-token, bsz=1, decode 32): p50 **3.77 ms** (decode 265 tok/s),
p99 4.34 ms -- tight tail.

Batch-generate throughput (32 new tokens, prompt 128):

| batch | total tok/s | peak VRAM |
|---|---:|---:|
| 1 | 212 | 413 MB |
| 4 | 784 | 562 MB |
| 8 | 1,581 | 590 MB |

The 265 tok/s single-stream TPOT number is the realistic `model.generate()`
figure via the standard HF path; the ~395 tok/s in the table above is a
tighter isolated native-graph bench. The real serving lever on Blackwell is
**batch scaling** (bsz 1 -> 8 gives ~7.5x aggregate throughput, since RWKV
has no KV cache), not single-stream speedup.

Usage:

```python
from rwkv7_hf.native_jit import fast_generate
print(fast_generate(model, tokenizer, "User: Hello!\n\nAssistant:", max_new_tokens=48))
```

Caveats for the HF adaptation target: the imported CUDA-graph path is currently
single-batch / fixed-shape greedy decode. Dynamic batching, PEFT/RL integration,
state-cache serving semantics, and V100 performance still need separate
validation before it can replace or augment the HF `forward` / `generate` path.

### V100 native JIT / CUDA graph validation

Command:

```bash
python bench/bench_native_decode.py \
  --hf-dir /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --prompt-tokens 32 \
  --decode-tokens 64 \
  --greedy-check-tokens 16 \
  --results bench/results.jsonl
```

Result on Tesla V100-PCIE-32GB:

| Path | Decode tok/s | ms/token | Status |
|---|---:|---:|---|
| native JIT block step | 103.52 | 9.6596 | 1.12x official V100 baseline |
| native CUDA graph | 254.33 | 3.9319 | 2.76x official V100 baseline |

Multi-size native-graph decode on V100-32GB (fp16, `rwkv7_forward_token`, prompt 512 / decode 128):

| Model | Decode tok/s | ms/token | Prefill tok/s | Peak VRAM |
|---|---:|---:|---:|---:|
| 0.1B | 254.33 | 3.93 | — | ~640 MB |
| 2.9B | 57.8 | 17.3 | 6238.8 | 5937 MB |
| 7.2B | 32.1 | 31.12 | 3452.4 | 14076 MB |

Correctness checks in the same row:

- native logits vs HF logits: cosine `1.00000024`, max_abs `0.03125`, argmax match.
- native CUDA graph greedy tokens vs native JIT greedy tokens: `16/16` identical.
- peak VRAM: `400.3 MB`, comparable to the official/HF 0.1B smoke rows.

Interpretation: this does not finish the full HF serving target because it is a
single-batch fixed-shape greedy path, but it gives a concrete implementation
direction: move the TorchScript block-step packing / graph-capture idea into the
HF fast-token API while preserving batched state-cache semantics.

### V100 sm70 native-prefill policy (fp16, prompt 512)

The V100 path now uses a split-row Triton recurrent scan instead of keeping the
complete 64x64 state tile live in one program. Runtime policy selects tile 16
for bsz 1/2, tile 32 for bsz >=4, and limits the larger fused state-prep + scan
kernel to bsz 1. Explicit environment settings remain available for A/B and
fallback.

Same-card matching-checkpoint comparison against Albatross `faster3a` with
`--wkv fp32io16`:

| Model | bsz | HF native prefill tok/s | Albatross tok/s | Ratio | Stage |
|---|---:|---:|---:|---:|---|
| 0.1B | 1 | 32058.9 | 39323.63 | 0.8153x | P2 |
| 0.1B | 2 | 56598.4 | 71382.68 | 0.7929x | P1, near P2 |
| 0.1B | 4 | 94135.9 | 109051.25 | 0.8632x | P2 |
| 0.1B | 8 | 122043.7 | 153368.36 | 0.7958x | P1, near P2 |
| 0.4B | 1/2/4/8 | 16439.1 / 27492.5 / 38753.8 / 46475.0 | 18462.45 / 31264.66 / 45953.77 / 59046.69 | 0.8904x / 0.8793x / 0.8433x / 0.7871x | P2/P2/P2/P1 |
| 1.5B | 1/2/4/8 | 10305.4 / 14419.5 / 17108.3 / 17752.3 | 11911.85 / 16332.13 / 20141.39 / 21807.28 | 0.8651x / 0.8829x / 0.8494x / 0.8141x | P2/P2/P2/P2 |

The 0.1B figures are medians across three fresh processes and are
`1.162x`-`1.234x` faster than the old full-head fused state-scan route. A
separate no-env run verifies default dispatch at bsz 1/2/4/8. Independent
unfused native token-loop alignment passes prefill and cached-next-token greedy
checks for 0.1B/0.4B/1.5B at bsz 1/4; ordinary HF `model.generate()` reports
`native_prefill` + `native_graph` and passes for the same six cases.

Raw evidence and exact caveats:
[`bench/v100_sm70_prefill_policy_20260710/README.md`](bench/v100_sm70_prefill_policy_20260710/README.md).

### V100 experimental native-model telemetry

The FLA-free `NativeRWKV7ForCausalLM` remains an experimental fallback, not the
production wrapper replacement. Its smoke row is nevertheless tracked because it
is the long-term base for upstream Transformers, AMD/CPU, and small-shared-memory
training fallback work.

Command:

```bash
python tests/test_native_model.py \
  --model /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --expect-jit-decode \
  --results bench/results.jsonl
```

Latest V100 row:

| Check | Result |
|---|---:|
| forward min cosine vs FLA wrapper | `0.99999976` |
| forward max abs diff | `0.00003815` |
| prompt argmax match | `3/3` |
| batched forward min cosine | `0.9999994` |
| batched cached-decode argmax match | `3/3` |
| greedy generate match | `16/16` |
| incremental cache exercised | `true` |
| cached decode backend | `native_jit` |

### Larger-model 50-series native results from latest `main`

| model | metric | FLA HF | official | native path |
|---|---|---:|---:|---:|
| 0.4B | decode tok/s | 11.5 | 26.0 | 174.7 CUDA graph, 6.7x official |
| 1.5B | decode tok/s | 13.3 | 30.7 | 26.6 JIT, 87% official |

Interpretation from latest `main`: the native CUDA-graph path wins strongly on
small launch-bound models, while larger models become compute/bandwidth-bound and
need a different serving-oriented fusion strategy. For the V100 branch, the next
useful step is to validate this native JIT/CUDA-graph path on the V100 0.1B model
and then decide whether to integrate its block-step packing into the HF fast-token
API.

## Apple M5 stateful CoreML multifunction bring-up (2026-07-10)

The CoreML lane now has a live recurrent-state implementation rather than only
`full_logits` planning. The export combines fixed masked `prefill` and one-token
`decode` functions into one deduplicated `.mlpackage`. Core ML 8/9 state is
fp16-only, so the fp32 WKV cache is stored as fp16 high + fp16 residual tensors;
attention/FFN previous inputs and `v_first` are separate states. The runtime
transfers these states between function handles with `MLState.read_state` /
`write_state`, then performs exact shared-prompt greedy decode.

Live environment: MacBook Air / Apple M5 / 16GB / macOS 26.5, Python 3.11.15,
PyTorch 2.13.0, Transformers 5.13.0, CoreMLTools 9.0. PyTorch 2.13 is newer than
the latest version advertised as tested by CoreMLTools 9, so the explicit
runtime gates below are required.

```bash
PYTHONPATH=. RWKV7_NATIVE_MODEL=1 python scripts/export_rwkv7_coreml.py \
  /path/to/rwkv7-g1d-0.1b-hf /tmp/rwkv7-coreml-0.1b \
  --export-kind stateful-multifunction \
  --prefill-seq-length 2 \
  --compute-units cpu-only \
  --coreml-compute-precision auto \
  --deployment-target macOS15 \
  --require-coremltools

PYTHONPATH=. python bench/run_coreml_apple_baseline.py \
  --manifest /tmp/rwkv7-coreml-0.1b/coreml_export_manifest.json \
  --prompt-target-chars 16 --decode-lengths 2 \
  --verify-chunked-prefill --verify-chunk-size 1 \
  --verify-hf-parity --hf-parity-dtype fp32 \
  --require-hf-greedy-match \
  --compute-units cpu-and-ne --require-coremltools
```

| Compute units | CoreML compute | Prefill tok/s | Decode tok/s | State transfer max abs | Chunk logits/state max abs | HF greedy |
|---|---|---:|---:|---:|---:|---|
| CPU only | fp32 | 101.78 | 72.19 | 0.0 | 0.0 / 0.0 | 2/2 |
| CPU + Neural Engine eligible | fp32 | 99.78 | 70.74 | 0.0 | 0.0 / 0.0 | 2/2 |

The exact short prompt has four RWKV tokens and the generated ids are
`[1184, 460]`, identical to the source HF native fp32 path. The package is
`765,660,573` bytes and the transferred recurrent state is `4,795,392` bytes.
`CPU_AND_NE` only constrains eligible compute units; it is **not** proof that the
Neural Engine executed the graph. These are correctness bring-up rows, not
production throughput claims.

An explicit stateful fp16-compute experiment was smaller (`383,414,809` bytes)
and faster on this short CPU-only shape, but generated `[47, 11]` instead of the
HF `[1184, 460]`. Therefore `--coreml-compute-precision auto` resolves to fp32
for stateful exports and fp16 remains opt-in until selective recurrent precision
or an equivalent numerically stable ANE layout closes that mismatch. The next
CoreML matrix is prefill chunks 16/64, longer prompts/decode, 0.4B/1.5B,
LUT4/INT4, and measured runtime placement.

The same stateful package was also exported through the initial CoreMLTools
weight-compression modes. All rows keep fp32 recurrent compute, verify exact
state transfer and alternate chunk splitting, and use the same four-token prompt
plus two-token decode:

| Weight mode | Package bytes | vs fp32 package | Prefill tok/s | Decode tok/s | HF greedy |
|---|---:|---:|---:|---:|---|
| fp32 weights | 765,660,573 | 1.000x | 99.78 | 70.74 | 2/2 |
| INT8 per-channel | 344,882,067 | 0.450x | 104.80 | 67.49 | 2/2 |
| INT4 per-block | 287,366,795 | 0.375x | 92.25 | 71.40 | 0/2 |
| INT4 per-block, `lm_head` kept | 461,954,961 | 0.603x | 107.64 | 71.46 | 0/2 |
| LUT4 grouped-channel | 98,213,860 | 0.128x | 85.36 | 67.06 | 0/2 |

This closes a first **functional W8 CoreML stateful lane**: package footprint
falls by about 55%, chunk/state gates remain exact, and short greedy tokens still
match. It does not close the production speed target because decode is about
`0.95x` the fp32-compute row. The W4/LUT4 lanes prove large package reduction
and valid stateful execution, but fail the current HF greedy gate; they are
quality experiments only until calibration/mixed-precision policies and broader
quality scoring pass. Quantized rows may record the mismatch without failing the
runtime harness when `--require-hf-greedy-match` is omitted, but acceptance
runs for unquantized/W8 exactness should keep that flag enabled.

### Apple M5 CoreML 0.4B extension

The same live path also exports and runs `rwkv7-g1d-0.4b-hf` on the 16GB M5.
The shape remains intentionally short (`prefill_seq_length=2`, shared prompt
four tokens, decode two) so this row validates model-size generality before the
long-context sweep:

| 0.4B mode | Package bytes | Prefill tok/s | Decode tok/s | State bytes | Chunk max abs | HF greedy |
|---|---:|---:|---:|---:|---:|---|
| fp32 compute / uncompressed | 1,805,544,379 | 29.23 | 20.87 | 12,783,616 | 0.0 | 2/2 |
| fp32 compute / INT8 per-channel | 657,633,909 | 32.94 | 20.47 | 12,783,616 | 0.0 | 2/2 |

The 0.4B INT8 package is about `0.364x` the uncompressed package and improves
short prefill to about `1.13x`, while decode is about `0.98x`; it therefore
still misses the strict "quant decode no slower" production gate despite exact
short greedy/state/chunk correctness. CoreMLTools emitted zero-scale division
warnings while annotating a few zero-valued tensors, so larger prompt/quality
rows must stay mandatory even though this runtime row is finite and passes.

A longer 0.4B correctness row uses a 256-character shared prompt (`67` RWKV
tokens), `32` generated tokens, ten-plus recurrent state crossings, and an
alternate one-token chunk verification. Both uncompressed and INT8 keep
`chunk logits/state max_abs=0.0` and HF greedy `32/32`. On one `CPU_AND_NE`
run, uncompressed vs INT8 prefill/decode were `50.11/46.31` vs
`55.96/46.30 tok/s`. Repeated warmed measurements remain variable: INT8 decode
is near parity rather than a stable win (roughly `0.99x` median in the sampled
runs), and first model warmup/compilation is much longer for the compressed
package. The runtime now exposes `--warmup` and records `warmup_s` so cold
CoreML compilation cannot silently contaminate steady-state throughput rows.

## Apple M5 live Qwen3.5 0.8B comparison (2026-07-10)

The first real same-device Qwen3.5 row is recorded in
`bench/results_qwen35_apple_m5_20260710_fp16.jsonl` and
`bench/results_qwen35_apple_m5_20260710_w4.jsonl`. Environment: MacBook Air,
Apple M5, 16GB, macOS 26.5, Ollama 0.31.1 with
`qwen3.5:0.8b-mlx` (1.2GB public package), and MLX 0.32.0. Both engines receive
the same prompt text; tokenizer token counts differ naturally.

The Ollama runner disables thinking for response comparability and uses
`keep_alive=0` per row. This prevents an already-completed prompt from turning
`prompt_eval_duration` into a cache-hit number. `ttft_s` excludes the reported
model load duration on both sides; `cold_ttft_s` retains load-inclusive latency.

Conservative comparison values use minimum throughput and maximum steady TTFT
across two repeats:

| RWKV mode | Prompt chars | Qwen/RWKV tokens | Qwen/RWKV decode tok/s | Decode ratio | Qwen/RWKV prefill tok/s | Prefill ratio | Qwen/RWKV TTFT | RWKV peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fp16 + Metal WKV | 128 | 54 / 43 | 112.15 / 92.14 | 0.822x | 1041.27 / 93.43 | 0.090x | 0.071 / 0.460 s | 929.0MB |
| fp16 + Metal WKV | 512 | 173 / 164 | 111.41 / 102.13 | 0.917x | 2403.92 / 116.99 | 0.049x | 0.087 / 1.402 s | 929.1MB |
| W4 auto + Metal WKV | 128 | 54 / 43 | 109.14 / 68.11 | 0.624x | 1126.44 / 72.62 | 0.064x | 0.064 / 0.593 s | 527.6MB |
| W4 auto + Metal WKV | 512 | 173 / 164 | 113.73 / 68.62 | 0.603x | 2460.76 / 73.40 | 0.030x | 0.087 / 2.235 s | 527.7MB |

This is a clear **gap**, not a win. fp16 decode is relatively close but variable
(about `0.82x-0.92x` Qwen in the retained repeat), while the sequential MLX
prefill path is only about `0.05x-0.09x`. Current W4 lowers RWKV peak memory to
about `0.568x` fp16 but also lowers decode to about `0.67x-0.74x` fp16. W4 matched fp16 greedy tokens
for the 512-character sample and diverged at token zero for the 128-character
sample, so no quality-parity claim is made. Ollama runtime memory is still
missing, so the cross-engine peak-memory gate remains unknown. The runner now
also records Ollama's official `/api/ps` loaded-memory value separately
(`1.09-1.11GB` here); it is not mislabeled as peak memory.

### MLX prefill graph-evaluation batching

The recurrent MLX reference historically called `mx.eval` after every prompt
token. `RWKV7_MLX_PREFILL_EVAL_INTERVAL` and
`--rwkv-prefill-eval-interval` now expose a conservative graph-batching seam:
the model default stays `1`, while the Apple acceptance wrapper defaults to
`2`. `scripts/mlx_prefill_eval_interval_bench.py` rotates interval order in one
loaded process and compares logits, all WKV/attention/FFN/v-first cache arrays,
seen-token count, and next token against interval `1`.

M5, 512 prompt characters (107 RWKV tokens), four interleaved repeats:

| Mode | Model | interval=1 median | interval=2 median | Speedup | Parity |
|---|---|---:|---:|---:|---|
| fp16 | 0.1B | 243.03 tok/s | 255.37 tok/s | 1.05x | exact |
| fp16 | 0.4B | 115.47 tok/s | 148.08 tok/s | 1.28x | exact |
| fp16 | 1.5B | 42.52 tok/s | 46.38 tok/s | 1.09x | exact |
| W4/Metal | 0.4B | 72.49 tok/s | 99.77 tok/s | 1.38x | exact |
| W4/Metal | 1.5B | 26.80 tok/s | 35.39 tok/s | 1.32x | exact |

Raw rows are in `bench/results_mlx_prefill_eval_m5_20260710_fp16.jsonl`
and `bench/results_mlx_prefill_eval_m5_20260710_w4.jsonl`. The corresponding
isolated Qwen rerun is in
`bench/results_qwen35_apple_m5_20260710_eval2_{fp16,w4}.jsonl`. Conservative
eval2 comparisons remain gaps: fp16 prefill is `0.120x/0.050x` Qwen for
128/512 characters; W4 is `0.088x/0.042x`. Removing synchronization overhead
therefore helps but cannot close the sequential-recurrence gap. The next Apple
prefill milestone is a native MLX/Metal port of DPLR/WY chunk summary, prefix
combine, and chunk apply/output—not a larger eval interval.

Reproduce the interval sweep:

```bash
PYTHONPATH=. python scripts/mlx_prefill_eval_interval_bench.py \
  --models /path/to/rwkv7-g1d-0.4b-hf,/path/to/rwkv7-g1g-1.5b-hf \
  --intervals 1,2,4 --prompt-target-chars 512 --repeat 4 --warmup 1 \
  --dtype fp16 --quantization none --wkv-backend metal --atol 0 \
  --results bench/results_mlx_prefill_eval.jsonl
```
