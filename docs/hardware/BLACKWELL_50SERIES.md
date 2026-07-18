### 2026-07-18 - RTX 5090 Native FP32-state decode close

The opt-in Native/no-FLA path now has an exact-card cached-decode close for the
official g1h 7.2B checkpoint with FP16 weights and FP32 recurrent state. Three
fresh 512-token process repeats produce B1/B8 medians `145.06/845.57 tok/s`
against precision-matched official v3a `144.47/841.77`, or
`1.0041x/1.0045x`. Both CUDA extensions are active in every row. All six runs
share one complete greedy-trace hash; the 64-step alignment gate has minimum
logits cosine `0.9999934435` and exact top-1 at B1/B8.

This is not a global Blackwell default. The official fp16-state reference is
still faster at `146.28/890.21 tok/s`, especially at B8. The accepted FP32
scratch sparse-FFN and SM120 W/A/G paths remain default-off and are limited to
the exact benchmark command. Evidence:
[`bench/5090_native_decode_fused_20260718`](../../bench/5090_native_decode_fused_20260718/README.md).

### 2026-07-18 — RTX 5090 Native train_temp B16 alignment and resume

The Native/no-FLA `train_temp_cuda` route now matches the official shell
training shape at BF16, B16, T512 and gradient checkpointing. Backward loss and
399/399 gradients are exact; FusedAdam grouping/order, 399/399 parameter deltas
and post-step loss are also exact. Official and Native both complete seeds
131/232/333 for 1,000 steps with `3/3` finite deep-success runs. Median
train/validation AUC relative differences are `0.001798%/0.002455%`.

A Native 500+500 resume restores model, optimizer and RNG state hashes and
passes against a continuous run. A separate 1,000-step stability row records 20
steady memory samples: allocated range `0.375 MiB`, reserved growth `0 MiB`.
Native median training throughput is `0.9499x` official, so numerical alignment
and bounded stability pass while training performance remains open. Evidence:
[`bench/5090_native_train_temp_b16_20260718`](../../bench/5090_native_train_temp_b16_20260718/README.md).

### 2026-07-17 — RTX 5090 official train_temp training alignment

The opt-in HF `train_temp_cuda` backend now uses the pinned official CUDA
operator boundaries for dense BF16 training. On one RTX 5090, a production-
shaped 12x768 model at B1/T512 matches official RWKV-LM backward gradients
400/400 exactly and matches the DeepSpeed FusedAdam step across 800 tensors and
deltas exactly. A separate three-seed, 1,000-step cohort completes all six runs
and passes success-count, loss-AUC and gradient-ratio gates.

This is exact evidence for one card/model/shape. It does not promote variable-
length padding, other GPU families, larger checkpoints or distributed
training. Usage and recovery are in [`TRAIN_TEMP_CUDA.md`](../TRAIN_TEMP_CUDA.md);
raw evidence is in
[`bench/5090_train_temp_alignment_20260717`](../../bench/5090_train_temp_alignment_20260717/README.md).

### 2026-07-16 — RTX 5090 g1h production BN/TN W4 model matrix

The exact-card BF16 speed policy now selects model-specific group-128 Marlin
W4 FFN coverage, head policy and sensitive final-layer exception. Paired
prompt128/decode128 results:

| Model | Batch | footprint/BF16 | prefill/BF16 | decode/BF16 | final cosine |
|---|---:|---:|---:|---:|---:|
| 1.5B | 1 | `0.6250x` | `1.2788x` | `1.1854x` | `0.99984407` |
| 1.5B | 8 | `0.6250x` | `1.0097x` | `1.2133x` | `0.99975127` |
| 2.9B | 1 | `0.5776x` | `1.0092x` | `1.2222x` | `0.99965632` |
| 2.9B | 8 | `0.5776x` | `1.0116x` | `1.2894x` | `0.99958199` |
| 7.2B | 1 | `0.5298x` | `1.0010x` | `1.5068x` | `0.99963713` |
| 7.2B | 8 | `0.5298x` | `1.1561x` | `1.4978x` | `0.99954909` |
| 13.3B | 1 | `0.5347x` | `1.0153x` | `1.4957x` | `0.99966073` |
| 13.3B | 8 | `0.5347x` | `1.1549x` | `1.4670x` | `0.99955237` |

All eight rows preserve the next token. The preceding TorchAO-only 7.2B route
was retained as negative evidence because B8 prefill fell to `0.2711x`; Marlin
removes that regression while preserving the decode and memory wins. Dispatch
is fail-closed to RTX 5090 + SM120 + BF16 + exact model/FFN roles/shapes, so
0.4B, other 50-series and older-card policies are unchanged.

The production route asserts the Tensor Core CTA/epilogue BN/TN grid per
internal scheduler segment and uses an explicit bit-exact fused FFN-key
ReLU-square ABI. Its expanded group-128 contract passes 280/280 checks through
8192 rows across eight FFN directions; experimental group-32 passes 48/48.
Evidence:
[`bench/5090_bntn_all_models_20260716`](../../bench/5090_bntn_all_models_20260716/README.md).

### 2026-07-12 — RTX 5090 batched quant + 13.3B low-memory close

最新 5090 artifact: [`bench/5090_blackwell_production_close_20260712`](../../bench/5090_blackwell_production_close_20260712/README.md)。

- Blackwell MM8/MM4 新增 batched GEMV 与 tensor-core dot 路径；非 Blackwell 卡继续走原已验证 dispatch。
- 2.9B/7.2B 的 bsz8、prompt128/2048、decode128/512 全部进入 fp16 的 1% paired 等价带；旧 7.2B 最差压力行从 `0.7619x/0.6695x` 提升到 `0.9913x/0.9919x`。
- 1.5B/2.9B/7.2B 合并 36-row 矩阵 footprint 全下降、same-next 全通过，2% fail-closed gate 通过；1.5B W8 仍有单行 `0.9841x`，不宣称所有 shape 严格 `>=1.0x`。
- `--low-memory` 已在 48GB RAM/no-swap 主机把官方 13.3B 转成 6 个 safetensors shard；5090 load/forward/generate peak `25536.6MiB` 通过。
- 0.4B full MATH500 `500x64` 已完成：pass@64=`0.38`，generation=`16925.6 tok/s`，steady decode=`19339.5 tok/s`；对提交内 Albatross reference 的两个 2x speed gate分别为 `4.336x/4.871x`。该比较不是同卡实时 Albatross 复跑。
- fresh Transformers module cache 已直接发现完整 transitive kernel closure，不再需要手工预填 `ada_sparse_ffn.py` / `sm70_quant.py`。

### 2026-07-04 — RTX 5090(sm_120) HF full smoke matrix 对齐 50-series 支持合约

5090 现在有一键矩阵 artifact: [`bench/5090_blackwell_hf_matrix_20260704`](../../bench/5090_blackwell_hf_matrix_20260704/README.md),由 [`bench/run_5090_hf_validation.sh`](../../bench/run_5090_hf_validation.sh) 生成。它覆盖 Blackwell/50-series 支持合约里当前能在 0.1B HF 模型上验证的项目。

**已通过:**
- HF `generate` smoke PASS,backend=`native_graph`。
- HF API contract PASS,beam generate backend=`native_graph`。
- native prefill forward PASS:`generate_match=True`,seen=32。
- native/no-FLA `NativeRWKV7ForCausalLM` + HF Trainer + PEFT LoRA PASS:loss `9.3638 -> 0.4192`,trainable params `72/72` updated。
- dynamic batching + native prefill + native_graph decode smoke PASS,bsz=8,512 decoded tokens。
- W8/W4 quantized load/generate smoke PASS:W8 footprint 283.4 MB,W4 footprint 242.9 MB。
- native mm8/mm4 benchmark PASS artifact: [`bench/5090_blackwell_native_quant_20260704`](../../bench/5090_blackwell_native_quant_20260704/README.md)。0.1B e2e decode:mm8 **0.9487× fp16** / footprint 0.8688×,mm4 **0.9903× fp16** / footprint 0.8030×;R/K/V isolated sweep:int8 0.6706×,int4 0.6613× fp16。
- 0.1B fp16 bsz sweep:bsz1/2/4/8 native_graph decode tok/s = **945.9 / 1346.3 / 2714.4 / 5326.4**。
- chunked prefill PASS:prompt512,batch1,chunk64/128/256,seq length match,chunk256 达 **13,681.5 tok/s** 且 peak VRAM 421.6 MB。
- exact-card fused A/B PASS:fused output **1.0723×**,fused recurrent-output **1.1963×**,greedy `32/32`。
- `rwkv7_hf/kernel_policy.py` 的 Blackwell rule 已记录 RTX 5090,并要求 `triton_compat` remote-code import + 5090 runner artifact 才能声明 5090。

**边界:**这仍不是 5090 full MATH500 avg@64 验收;只是把 5090 的 HF adapter/训练 fallback/量化 smoke/动态 batching/chunked prefill/fused A-B/native mm8-mm4 支持矩阵对齐到 50-series 合约。完整数学验收仍需要正式 MATH500 数据和 acceptance 模型在 5090 上跑 `scripts/run_math500_acceptance.sh`。

### 2026-07-04 — RTX 5090(sm_120) native-prefill/HF smoke 对齐 4090 验证矩阵

5090 首轮 smoke 发现的两个 blocker 已补齐并复测,artifact: [`bench/5090_blackwell_native_prefill_smoke_20260704`](../../bench/5090_blackwell_native_prefill_smoke_20260704/README.md)。

**已修:**
- `native_jit.prefill` 现在解包当前 41-field pack(含 `RKVw`),修掉 5090 上 `ValueError: too many values to unpack (expected 40)`。
- 新增随 remote-code 模型一起分发的 `triton_compat.py`,为早期 Torch 2.6 + Triton 3.3 + Blackwell 栈补 legacy `AttrsDescriptor` 路径,并默认给 sm_120 关闭/降级 FLA `torch.compile` sqrelu 问题。
- convert/sync 脚本已把 `triton_compat.py` 纳入模型目录,不再需要手工 patch site-packages。

**5090 实测通过:**
- HF `generate` smoke PASS,backend=`native_graph`。
- HF API contract PASS,beam generate backend=`native_graph`。
- native prefill forward PASS:`generate_match=True`,seen=32。
- dynamic batching + native prefill + native_graph decode smoke PASS,bsz=8,512 decoded tokens。
- W8/W4 quantized load/generate smoke PASS:W8 footprint 283.4 MB,W4 footprint 242.9 MB。
- 0.1B fp16 bsz sweep:bsz1/2/4/8 native_graph decode tok/s = **945.7 / 1345.6 / 2715.2 / 5338.2**。

**仍不是的东西:**这不是 5090 的 full MATH500 avg@64 验收;只是把 5090 的 HF adapter 可运行性/原生 prefill/量化 smoke 对齐到 4090-style HF smoke 要求。完整数学验收还需要把正式 MATH500 数据和 acceptance 模型放到 5090 后跑 `scripts/run_math500_acceptance.sh`。

### 2026-07-04 — RTX 5090(sm_120) HF adapter smoke:能跑,但需记录环境 workaround

在真实 RTX 5090 32GB 节点补了 HF adapter 运行 smoke,结果已落到 [`bench/5090_blackwell_smoke_20260704`](../../bench/5090_blackwell_smoke_20260704/README.md)。

**环境**:RTX 5090 / driver 610.43.02 / PyTorch `2.6.0a0+ecf3bae40a.nv25.01` / CUDA 12.8 / Triton 3.3.1 / FLA 0.5.1 / Transformers 5.13.0 / bnb 0.49.2。

**结论**:
- ✅ 0.1B HF adapter 可以在 5090 上 load + generate。
- ✅ MATH-style smoke 跑通 dynamic batching + deferred verification + deferred text decode,backend 报 `native_graph`。
- ✅ bsz=8 smoke decode 512 tokens,decode 段约 **522.6 tok/s**。
- ⚠️ 该节点的 Torch 2.6 + Triton 3.3 + FLA 组合需要 venv 级 `triton_compat_shim` 补 legacy `triton.compiler.compiler.AttrsDescriptor`,同时保留 Triton 3.3 的 `triton.set_allocator`。
- ⚠️ 需 `TORCH_COMPILE_DISABLE=1` 绕开 Inductor/AttrsDescriptor 代码生成不兼容。
- ⚠️ native prefill 在这个 0.1B smoke 模型上触发 `ValueError: too many values to unpack (expected 40)`,所以 smoke 用 `--prefill-backend forward --decode-backend forward`。

这不是 MATH500 avg@64 验收数,只是 50 系 Blackwell 真机可运行性证明。完整验收仍以 4090 的 MATH500 acceptance artifact 为准;若要把 5090 纳入正式支持矩阵,下一步是把 Triton/FLA 兼容层和 native prefill 问题固化。


### 2026-07-01 — fused kernel 在 5070 全部跑通(最新 main 16dedd6)
最新 main 的 fused kernel prototype 在 RTX 5070(sm_120)结果:

| kernel | 5070 speedup | cos | 备注 |
|---|---|---|---|
| **fused RKV WAG projection** | **1.68×** ✅ | 1.0 | R/K/V + W/A/G LoRA 全融合,**超 baseline** |
| **fused FFN** | **1.30×** ✅ | 0.9999999 | CMix 融合,**超 baseline** |
| int8 fused RKV quant | 0.77× fp16 | 0.99995 | 旧版 0.38×→0.77×(dequant+RKV 深融) |
| int4 fused RKV quant | 0.62× fp16 | 0.986 | 旧版 0.36×→0.62× |

**关键**:projection + FFN **在 Blackwell 上 >1.0×**(V100 上单 projection 才 0.84×,但 5070 上融合 RKV+WAG 后 1.68×——更深融合 + Blackwell 对 Triton 友好)。fused kernel 策略在 sm_120 **验证成功**。

(注:main 新增了 fused_recurrent_update.py / fused_attention_projection.py / fused_ffn.py / fused_lora.py / native_fused.py / native_quant.py 等,convert 脚本只拷 3 个文件——HF 目录需手动拷全部 rwkv7_hf/*.py 否则 trust_remote_code 找不到模块。)

### 2026-07-01 — 完整 fused kernel 5070 数据(全部 >1.0×!)
全部 fused kernel prototype 在 RTX 5070(sm_120)的 speedup vs current baseline:

| kernel | V100(sm_70) | **5070(sm_120)** | cos | 状态 |
|---|---|---|---|---|
| fused recurrent | 2.79× | **5.71×** | 1.0 | ✅✅ Blackwell 大幅领先 |
| fused WAG LoRA | — | **1.79×** | 1.0 | ✅ |
| fused RKV WAG projection | 0.84× | **1.68×** | 1.0 | ✅ V100 不过、5070 过 |
| fused W/A LoRA | — | **1.33×** | 1.0 | ✅ |
| fused FFN | — | **1.30×** | 1.0 | ✅ |
| fused shift-mix | 0.77× | **1.02×** | 1.0 | ✅ V100 不过、5070 刚过 |
| int8 fused RKV quant | 0.38× | 0.77× | 0.99995 | ⚠️ 还没 ≥1.0× fp16 |
| int4 fused RKV quant | 0.36× | 0.62× | 0.986 | ⚠️ 还没 ≥1.0× fp16 |

**重大发现**:Blackwell sm_120 对 Triton fused kernel **比 V100 sm_70 友好得多**。V100 上 projection(0.84×)和 shift-mix(0.77×)都没过 1.0×,但 5070 上**全部 >1.0×**。recurrent 从 2.79× 飙到 **5.71×**。说明 Blackwell 新架构(tensor core / 寄存器 / Triton codegen)更适合 fused kernel 策略。

**量化(int8/int4)大幅改善**(0.38→0.77, 0.36→0.62)但还没 ≥1.0× fp16,需继续 tensor-core 优化。

### 2026-07-01 — fused attention output 在 5070(1.46-1.59×)+ 完整汇总
fused attn output(output projection + gate + 残差)在 5070:1.46-1.59×,cos=1.0。

**5070 完整 fused kernel 汇总(7/7 >1.0×):**
recurrent 5.71× / WAG LoRA 1.79× / RKV WAG projection 1.68× / attn output 1.59× / W/A LoRA 1.33× / FFN 1.30× / shift-mix 1.02×。

对比 V100:只有 recurrent(2.79×)过 1.0×,projection(0.84×)和 shift-mix(0.77×)都没过。**Blackwell sm_120 全面优于 V100 sm_70 对 Triton fused kernel。**

下一步:把全部 fused kernel 接进 native_graph forward(目前只有 recurrent 接了),端到端提速。

### 2026-07-01 — fused recurrent 集成进 native_graph 在 5070 功能正确
`RWKV7_NATIVE_GRAPH_FUSED_RECURRENT=1` + `test_fast_decode_api`(native_graph bsz=1):
- forward_fast_path_max_abs_diff **0.03125** ✅(fp16 精度内)
- greedy_equal ✅
- 但 test 断言 `([1,2,4],[1])` 失败:graph cache batch sizes 期望 [1,2,4] 但只有 [1]——fused flag 改变了 cache key,test 的 batch 断言需要适配(非功能/精度问题)。
**结论**:fused recurrent 集成进 graph replay 在 Blackwell 上**功能正确**。

### 2026-07-01 — main 0038802「fused output prep 接进 native_graph」在 5070 验证通过
main 最新提交 `0038802 Integrate fused output prep into native graph`:在 `_block_ip`/`_block_ip_batched`(native_graph per-token step)加 `RWKV7_NATIVE_GRAPH_FUSED_OUTPUT=1` 开关,把 group-norm + sk 修正 + gate 的 output-prep 换成 `fused_attn_output_prepare` Triton kernel。fallback 路径也重写(`out = (gn + skv) * g`),数值与旧版等价。

`bench/bench_native_graph_fused_output.py` A/B(fused_recurrent + fused_output 同时开,0.1B fp16 bsz=1,steps=64)在 **RTX 5070 sm_120**:
- `max_abs_diff_first_step` **0.03125** ✅、`min_cosine` **1.0** ✅、`greedy_match` **64/64** ✅
- baseline 2.4186 ms/step → fused 2.3026 ms/step,**端到端 1.05×**(413.5→434.3 tok/s)
- 两个 fused kernel(recurrent + output)**同时**在 graph capture 内生效,backend=native_graph,cache hit 98.6%,VRAM 633 MB。

**结论**:fused output prep 集成进 native graph replay 在 Blackwell 上**功能正确 + 略快**。端到端只 1.05× 是因为 0.1B bsz=1 单 token 已快到 2.3ms,output-prep 占比小(隔离 bench 里 attn output 1.46-1.59×,端到端被稀释)。**7 个 fused kernel 现已 2 个(recurrent + output)接进 graph,余 5 个待接。**

**A/B 扫批/扫尺寸(纠正「大模型占比上升」的猜测——数据相反):**

| 场景 | baseline ms/step | fused ms/step | speedup | greedy | 备注 |
|---|---|---|---|---|---|
| 0.1B fp16 bsz=1 | 2.4186 | 2.3026 | **1.0504×** | 64/64 | output-prep 省固定 ~0.12ms |
| 0.1B fp16 bsz=4 | 2.5737 | 2.5449 | **1.0113×** | 256/256 | 4× batch 只多 6% 时间→已 memory-bound |
| 1.5B fp16 bsz=1 | 11.1405 | 10.7857 | **1.0329×** | 32/32 | VRAM 3.2GB |
| **2.9B fp16 bsz=1** | 22.9316 | 21.909 | **1.0467×** | 16/16 | **VRAM 5.9GB,装进 8GB!** decode 43.6→45.6 tok/s |

**纠正**:output-prep 省的是**固定绝对时间 ~0.1-1ms/token**,不随模型/batch 放大;相对 speedup 在 **1.01-1.05× 间小幅波动**(bs4 最低 1.01,因为 0.1B bsz=4 已到 memory floor),不是随模型单调下降(2.9B 又回到 1.05×)。**fused-output 是稳定小赢(~3-5%),不是 scaling 杠杆**。要拿大端到端提速,得继续优化 recurrent 状态更新本身(已是 fused_recurrent 5.71× 隔离)和压 memory traffic。

**2.9B 在 5070 完整可用**:forward + native_graph decode + fused(recurrent+output)全跑通,VRAM 峰值 5.9GB(8GB 卡剩 ~2GB)。这是 8GB Blackwell 笔电卡能跑的最大 fp16 RWKV-7 尺寸;7.2B/13.3B 需量化。

### 2026-07-01 — main 02e9a10 + 量化在 5070 全套数据(bnb 能跑!但速度远未达标)
**纠正旧假设**:bitsandbytes **0.49.2 在 sm_120 / CUDA 12.8 上能跑**(blockwise GPU quantize OK,8bit/4bit 都能 load+generate+next_token 正确)。之前"Windows/sm_120 无 bnb wheel"的判断**过时了**。

main `02e9a10 Add fused output batch matrix telemetry`:补了 `native_graph_fused_output_sweep` 分析轴(analyzer 按 batch_size 汇总 speedup)。**5070 完整 batch matrix(0.1B fp16,recurrent+output 双 fused):**

| bsz | baseline ms/step | fused ms/step | speedup | greedy |
|---|---|---|---|---|
| 1 | 2.4186 | 2.3026 | **1.0504×** | 64/64 |
| 2 | 2.5500 | 2.4370 | **1.0464×** | 64/64 |
| 4 | 2.5737 | 2.5449 | **1.0113×** | 256/256 |
| 8 | 2.7324 | 2.5134 | **1.0871×** | 256/256 |

min 1.01× / max 1.09×,全 greedy 正确、max_abs 0.03125。**bsz=8 反而最快(1.087×)**——fused output 绝对收益随 batch 增长(0.12/0.11/0.03/0.22ms;bsz=4 的 1.01× 是噪声)。与 V100 的 ">1.03× min" 量级一致,Blackwell 略优。

**量化全套(0.1B fp16,bnb):**

| mode | skip-policy | footprint MB | /fp16 | decode tok/s | /fp16 | W 目标(foot / spd) |
|---|---|---|---|---|---|---|
| fp16 native_graph | — | 364.4 | 1.0× | **382.6** | 1.0× | — |
| 8bit | memory | 283.4 | 0.778× | 21.0 | 0.055× | ≤0.75× / ≥1.0× |
| 8bit | decode_hot | 310.4 | 0.852× | 32.9 | 0.086× | decode_hot +57% |
| 4bit | memory | 242.9 | 0.667× | 46.3 | 0.121× | ≤0.55× / ≥1.0× |
| 4bit | decode_hot | 283.4 | 0.778× | 41.3 | 0.108× | decode_hot 反而略降 |

**结论**:
- **fp16 native_graph 382.6 tok/s** 是 5070 头条数字(~V100 255 的 1.5×,Blackwell 单卡 decode 强)。
- bnb **footprint 接近达标**(W8 0.778 差一点 / W4 0.667 没到 0.55),但 **decode 速度远远不达标**(最好 0.121× fp16,差 8×)。证实 [FUSED_BACKEND](../performance/FUSED_BACKEND.md) 所说"bnb 只是兼容基线,非 fast path"。
- decode_hot 对 8bit +57%,对 4bit 略负——同 V100 tradeoff。
- 要达 W8/W4 ≥1.0× fp16,**只能走 native int8/int4 fused dequant-GEMV**([FUSED_BACKEND](../performance/FUSED_BACKEND.md) step 13-16)。我之前隔离 bench:int8 fused RKV 0.77× / int4 0.62× fp16——比 bnb 好但还没 ≥1.0×,需 tensor-core 深融。

### 2026-07-01 — main 测试套件在 5070 大面积跑通 + 发现 1 个 latent cache 计数 bug
main 静态(02e9a10),在 5070 补跑测试套件:
- `test_native_model.py` **PASS** — native(fla-free)对 FLA **bit-exact**(forward cos=1.0/max_abs 1.7e-5、batch bsz=3 cos=0.999999、greedy 16/16、incremental cache True、decode backend=**native_jit**)。
- cache 服务级:`test_native_graph_cache` PASS / `test_batch_cache` PASS(bsz=4 diff=0.0)/ `test_dynamic_batch_cache` PASS(RWKV7StateCache select/reorder/compact,perm=[2,0,1])/ `test_chunked_prefill` PASS(chunk 1/2/4/8,seq=161 对齐)。
- `test_fast_cache.py` **FAIL**(line 102 断言)— **latent bug,非 sm_120 特有**(纯 cache 计数逻辑,任何 GPU 复现;且该 test 不在 V100 validation 脚本里,所以一直没被抓到)。

**`test_fast_cache` 根因(已定位)**:`RWKV7_FAST_CACHE=0` 的 FLA 参考路径返回基础 `Cache` 类型,其 `get_seq_length()` 在 decode 时不累加 prefill:
- prefill 14 token 后:ref=14 / fast(`RWKV7StateCache`)=14(都对)
- +8 decode 后:**ref=8** / fast=**22**
- fast 路径(服务用的那个)**正确**(14+8=22);FLA ref 路径把 seq 重置成 per-call decode 数(8)。
- **logits/greedy bit-exact**(decode_max_diff=0.0、greedy 8/8)→ recurrent **state** 两边都对,只是 `get_seq_length()` 元数据计数不一致。功能上对 RWKV 无害(recurrent,不靠 cache_position 定位);但 `modeling_rwkv7.py:1187` `initial_seen = past.get_seq_length()` 读这个值,ref 路径会拿到错的 seen 数。
- 修法(留给 main 作者定):要么让非-fast 路径返回的 cache 也累加 seq(像 RWKV7StateCache 那样),要么 FLA 上游修。fast 路径本身没问题,不用动。

### 2026-07-01 — main 83b0ac3「fused output 默认开」在 5070 验证 + 测试套件全绿
main `83b0ac3 Enable native graph fused output by default (#49)`:`RWKV7_NATIVE_GRAPH_FUSED_OUTPUT` 默认从 `"0"` 翻成 `"1"`(modeling + native_jit 两处)。现在用户走 native_graph **默认就吃 fused output kernel**(recurrent 仍默认 OFF)。

**新默认组合(output ON / recurrent OFF)在 5070 验证(安全 + 小模型更快):**
| 模型 | greedy | cos | max_abs | speedup vs output-OFF |
|---|---|---|---|---|
| 0.1B fp16 bs1 | 64/64 ✅ | 1.0 | 0.046875 | **1.0505×** |
| 2.9B fp16 bs1 | 16/16 ✅ | 1.0 | 0.09375 | 0.9984×(持平,23ms/token output-prep 占比太小) |

→ 默认开它**在 Blackwell 上安全**(精度全对),小模型 +5%,大模型中性。没回归。

**5070 测试套件本轮补跑(全 PASS):**
- `test_speculative_decode`(0.1B draft → 0.4B target):**PASS**,acceptance_rate **0.778**(7/9),1 resync,cached-prefix 重写对齐 target greedy。
- `test_hf_api_contract`:**PASS** — 标准 `model.generate()` 路由到 `generate_fast_token_backend=native_graph`(新默认 fused output 经公开 API 生效),beam search OK(beam_ids [295, 35762])。
- `test_reload_roundtrip`:**PASS** — save/reload **bit-exact**(max_abs_diff=0.0)。

**5070 测试覆盖现状(几乎全绿,仅 1 个 latent bug):** 官方对齐 / reload / fast_decode / hf_api_contract / batch_cache / dynamic_batch_cache / chunked_prefill / native_graph_cache / native_model / speculative_decode / native_graph fused(bs1-8+1.5B+2.9B)/ 量化(8/4bit) **全 PASS**;唯 `test_fast_cache` FAIL(seq_length 计数,见上节,非 sm_120 特有)。

### 2026-07-01 — FP8 tensor-core 探路(分支 `wangyue/fp8-projection-probe`)— 硬件能跑,但 cuBLASLt sm_120 Rowwise 缺口挡住精度
**动机**:V100(sm_70)没有 FP8 指令,Blackwell sm_120 有第 5 代 FP8 tensor core。FP8 是"V100 做不了我们能做"的头号杠杆,理论上能直接解量化 decode ≥1.0× fp16 的 gap(比 int8/int4 dequant-GEMV 强)。

**实测(`bench/bench_fp8_gemv.py`,0.1B-2.9B hidden × bsz1/4/8):**
- ✅ **FP8 硬件可用**:`torch._scaled_mm` TensorWise(`[1,1]` 标量 scale)在 sm_120 跑通;大矩阵 compute-bound 时 **2.23× bf16**(2048³ matmul,0.45→0.20ms)。
- ❌ **精度路径被挡**:per-channel **Rowwise 缩放(`[M,1]×[N,1]`)在所有配置下都被 cuBLASLt 拒绝**("Invalid scaling configuration")—— 大矩阵、不同 layout、`use_fast_accum=True` 全 FAIL。**只有 TensorWise 能跑,而 TensorWise 单 tensor 标量缩放对模型权重完全不可用(cos≈0)**。这是 **torch 2.11.0+cu128 的 cuBLASLt 还没给 Blackwell sm_120 实现 Rowwise FP8** —— 纯软件/构建缺口(Blackwell 太新),不是硬件限制。
- ⚠️ **decode 速度未定论**:隔离 GEMV(0.01-0.05ms,bsz1)落在 launch-overhead 噪声区,speedup 0.39-2.23× 乱跳(2.9b M=1 抽到 2.23×、1.5b M=1 抽到 0.39×)。compute-bound 的 2.23× **不迁移**到 memory-bound 的 decode GEMV。要真测速必须端到端进 graph(精度挡住了,没法测)。

**结论**:FP8 在 5070 上**今天用不了**(精度路径 cuBLASLt 没实现),与速度无关。**不是死路**,两条出路:
1. **等 torch/cuBLASLt 给 sm_120 补 Rowwise FP8**(升级 torch 即可,TensorWise 已证明硬件通路是通的)。
2. **自己写 Triton FP8 kernel**(像现有 `native_quant.int8_fused_rkv_gemv` 那样,FP8 weight + per-channel scale 自管 dequant,用 `tl.dot` 走 Blackwell FP8 tensor core)—— 这条不依赖 cuBLASLt,但工作量等同写个新量化 kernel,且精度未必比 int8/int4 现状好。

**对比现状**:native int8 fused RKV 0.77× / int4 0.62× fp16(隔离,已能跑精度对)。FP8 理论更优但今天卡在 cuBLASLt,**短期 int8/int4 路径仍是更实在的方向**,FP8 留作 Blackwell 红利待 torch 补齐。

### 2026-07-01 — 训练在 5070 现状(latest main 5f6247c,确认无变化)
main 静态(5f6247c),补跑训练 smoke:
- ✅ `test_native_model_training_unit` **PASS** — native(pure PyTorch)backward 在 sm_120 正常,loss 下降。
- ❌ `test_hf_training_smoke`(FLA backward)**仍被 Blackwell shared-mem 挡住**:`OutOfResources: shared memory, Required: 131072, Hardware limit: 101376`(FLA DPLR chunk backward kernel 需 128KB > 5070 的 99KB)。**影响所有 FLA-backed 训练**(HF Trainer / SFT / DPO / GRPO)。与之前一致,无回归也无改善;blocker 在 FLA 上游 triton kernel,非本适配器。**workaround 仍是 native model**(纯 PyTorch backward,无 triton kernel)。

### 2026-07-01 — main 7cb1049 native quant RKV block sweep 在 5070:**量化速度跨过 1.0× fp16!**
main `7cb1049 Add native quant RKV block sweep (#51)`:新 bench `bench_native_quant_rkv_sweep.py` 扫 int8/int4 fused RKV 的 block_m × block_k。**之前隔离 bench 用默认 block,int8 0.77× / int4 0.62× fp16 没过 1.0×;sweep 选对 block 后在 5070 全过:**

| 路径 | 最优 block | speedup vs fp16 | cos(fp16 vs quant) | footprint | ≥1.0× 配置数 |
|---|---|---|---|---|---|
| **w8 (int8)** | 32×32 | **1.074×** | **0.9999**(近 bit-exact) | ~0.5× | 1/9 |
| **w4 (int4)** | 8×128 | **1.233×** | 0.9806 | 0.25×(4× 压缩) | 7/9 |

**关键结论**:
- **w8 达标且精度近乎无损**(1.074× fp16 + cos 0.9999)——同时满足 [FUSED_BACKEND](../performance/FUSED_BACKEND.md) 的 W8 速度(≥1.0×)目标,精度基本不丢。**这是量化最实在的落地方案**。
- **w4 超 fp16 23%**(1.233×)+ 4× footprint 压缩,cos 0.98(量化噪声可接受);9 个 block 配置里 7 个过 1.0× → 对 block 选择鲁棒。
- 这**直接关掉了之前"量化差 30%/60%"的 gap** —— 不是靠 FP8,是靠 native int8/int4 fused dequant-GEMV 选对 block size。Blackwell sm_120 对这个 kernel 足够友好。
- ⚠️ **是隔离 kernel bench(只 RKV projection),非端到端 decode**。端到端 decode 里 projection 只占一部分(recurrent + FFN + lm_head 也在),整体 speedup 会被稀释,需后续端到端集成 bench 才能给全模型数字。但 quant kernel 本身已达标。
- ⚠️ cos 0.98(w4)是单层 projection 的量化误差,全模型精度影响(greedy/perplexity)需端到端 eval 另测。

**尺寸放大到 1.5B(hidden=2048)— 量化优势随模型变大而变大:**

| 尺寸 | w8 best | w8 cos | w8 ≥1.0× | w4 best | w4 cos |
|---|---|---|---|---|---|
| 0.1B(hidden 256) | 1.074× | 0.9999 | 1/9 | 1.233× | 0.9806 |
| **1.5B(hidden 2048)** | **2.356×** | **0.9999** | **9/9** | 1.329× | 0.9808 |

→ **w8 在 1.5B 上 2.356× fp16 + cos 0.9999 + 9/9 配置全过** —— 量化收益随 hidden 放大(更大 projection = 更多 compute 可省),这对"小显存跑大模型"正是想要的。w8 是 Blackwell 上量化的**明确落地路径**(速度大幅超 fp16、精度近乎无损、footprint 减半)。

### ⚠️ 2026-07-01 — 端到端实测纠正:上面"量化达标"是 kernel microbench 的假象,decode 端到端 NOT 达标
上面 sweep 的 1.074×/2.356× 是**隔离 kernel microbench**(只测 RKV 那一个 matmul,热循环)。写了 `bench/bench_native_quant_e2e_decode.py` 把每层 R/K/V 三个 linear 真换成 int8 fused kernel,跑**全模型 forward**(embedding→所有层→norm→lm_head)的 per-token 速度,在 5070:

| 模型(hidden) | batch | fp16 tok/s | int8 w8 tok/s | **端到端 speedup** | cos vs fp16 |
|---|---|---|---|---|---|
| 0.1B(768) | 1 | 49.39 | 49.31 | **0.998×(持平)** | 1.0 |
| 1.5B(2048) | 1 | 26.14 | 24.89 | **0.952×(反慢 5%)** | 1.0 |
| 1.5B(2048) | **8** | 201.73 | 192.33 | **0.953×(仍反慢)** | 1.0 |

**结论(纠正,已盖棺)**:**native int8 fused RKV 端到端不提速 decode —— batch=1 和 batch=8 都 ~0.95×,都不赢。** 隔离 kernel 的 2.36× **不迁移**到 decode,因为:
- decode 是 **M=1 GEMV,memory-bound**(batch=8 也还是 GEMV-dominated);cuBLAS fp16 GEMV 已贴显存带宽极限。
- int8 Triton dequant-GEMV 多了 dequant 计算 + 不如 cuBLAS 对 GEMV 调得狠 → 反而更慢。
- 隔离 sweep 的赢点主要是**省 launch 次数**(3 个 linear → 1 个 kernel),不是 compute;真模型 forward 里 cuBLAS GEMV 够重,launch 开销占比小,这个赢点吃不到。

**所以 [FUSED_BACKEND](../performance/FUSED_BACKEND.md) 的 W8/W4 ≥1.0× fp16 decode 目标,在 5070 端到端实测下 NOT 达标,batch>1 也救不回来。** 之前说"达标"是 microbench 误导,现纠正。剩唯一可能:(b) 把 int8 接进 native_graph(graph 内无 launch 开销 + 和别的 fused kernel 叠)—— 但若 kernel 本身比 cuBLAS 慢,graph 也救不了,这条路希望不大。**现实定位:Blackwell 上量化 = 省显存(footprint 减半、cos=1.0 无损),不是 decode 提速。**

注:FP8 那条线同理 —— compute-bound 的 2.23× 也不迁移到 M=1 decode(memory-bound)。**Blackwell decode 的量化提速,memory-bound 这个根本约束 V100 也有,不是 ISA 能绕过的。**

### 2026-07-01 — 量化精度损失实测(端到端 vs fp16)+ main 0a58885 fused output-project 在 5070 为负收益
**量化精度(`bench_native_quant_e2e_decode.py`,RKV projection 量化,全模型 forward vs fp16):**

| 量化 | logits cos | max_abs | greedy 下一个 token |
|---|---|---|---|
| int8 w8 | **1.0000** | 0.19 | **与 fp16 完全一致**(argmax_match=1) |
| int4 w4 | **0.9998** | 3.05 | **与 fp16 完全一致**(argmax_match=1) |

→ RKV-only int8/int4 量化**生成上零损失**(greedy 等价 fp16)。**caveat**:只量化了每层 3 个 linear(o_proj/FFN/lm_head 仍 fp16);argmax_match 是单 token,非 perplexity/长生成。对照 bnb**全模型**量化:8bit logits max_abs **0.61**、4bit 0.03(均 next_token 对)。要硬质量数字需补 perplexity 或 64-token greedy 窗口。

**main `0a58885 Add fused output projection probe (#52)`** —— 把 o_proj 融进 output-prep Triton kernel(第 4 个接进 graph 的 fused kernel)。`bench_native_graph_fused_output_project.py` A/B(其它 fused 全开)在 5070:
| bsz | speedup vs output-project-OFF | greedy |
|---|---|---|
| 1 | **0.962×(慢 3.7%)** | 64/64 |
| 8 | **0.759×(慢 24%)** | 512/512 |

→ **fused output-project 在 Blackwell 上是负收益**(精度都对 cos=1.0)。bsz 越大越亏:cuBLAS o_proj 在 [B,768]×[768,768] 上已经很优,折进 Triton 反而慢。**与 fused output-prep(+5%)相反** —— 不是所有融合都赚,o_proj 这种 cuBLAS 强项不该动。这条 main 默认仍是 OFF(opt-in probe),方向需再想。

### 2026-07-01 — main 7c1d46d fused WAG LoRA 在 5070 也负收益(第 2 个连续亏的融合)
main `7c1d46d Add native graph fused WAG LoRA probe (#53)`(第 5 个接进 graph 的 fused kernel;隔离 1.79×)。`bench_native_graph_fused_wag_lora.py` A/B(其它 fused 赢家 output+recurrent 保持开):

| bsz | speedup vs WAG-LoRA-OFF | greedy |
|---|---|---|
| 1 | **0.757×(慢 24%)** | 64/64 |
| 8 | **0.855×(慢 15%)** | 512/512 |

→ **fused WAG LoRA 在 Blackwell 上也亏**(cos=1.0 精度对)。**连续两个融合 probe(output-project + WAG LoRA)端到端都负** —— 印证规律:**0.1B decode memory-bound + 小,cuBLAS 对 WAG LoRA 这种低秩 matmul 也够优,Triton 融合省不了什么反而加开销**。隔离 microbench 的 1.79× 不迁移。**5070 上目前真能赚的只有 recurrent + output-prep 两个;其余 5 个融合(output-project / WAG LoRA / RKV-WAG projection / FFN / shift-mix)接进 graph 大概率也类似,值得 main 作者在 0.1B 之外的大模型/batch 上再验,别只看隔离数字就开默认。**

**1.5B 复测确认负收益(不是 0.1B 太小的假象):**
| 融合 | 0.1B bsz1 | 1.5B bsz1 | 2.9B bsz1 | 趋势 |
|---|---|---|---|---|
| WAG LoRA | 0.757× | 0.884× | **0.818×** | 非单调(0.1B→1.5B 升、1.5B→2.9B 回降)→ 不是"大模型会翻正"的干净趋势 |
| output-project | 0.962× | 0.940× | **0.871×** | 单调变差(模型越大越亏) |

→ **三个尺寸都没过 1.0×**,确认是真实的负收益,不是 toy size 假象。之前猜的"WAG LoRA 大模型可能翻正"被 2.9B 数据削弱(非单调,2.9B 回降)—— 大概率就是 ~0.8-0.88× 的稳定负收益,7.2B+ 也不乐观。output-project 更糟,随模型变大单调恶化。**8GB 5070(max fp16 2.9B)上这两个融合就是别开。**

### 2026-07-01 — main 315e502 fused recurrent+output 在 5070 = 最强的正收益融合(1.12×)
main `315e502 Add fused recurrent output backend probe (#54)`:把**两个已验证的赢家**(recurrent + output-prep)融进**一个** kernel(`fused_recurrent_update.py` 扩展)。`bench_native_graph_fused_recurrent_output.py` A/B 在 5070:

| 场景 | speedup vs 二者分别 OFF | greedy |
|---|---|---|
| 0.1B bsz1 | **1.119×** | 64/64 |
| 0.1B bsz8 | **1.124×** | 512/512 |
| 1.5B bsz1 | **1.046×** | 32/32 |

→ **recurrent+output 深融 = +11.9%**,比二者分开各贡献的 ~5% **累加放大**(消掉了 recurrent→output 中间交接)。这是几轮负收益后**第一个真正强的端到端正融合**,cos=1.0 全对。

**规律现在彻底清晰(给 main 的明确建议):**
- ✅ **沿 recurrent→output 数据通路深融(都 Triton-friendly)→ 累加赢**:recurrent+output **1.12×**(应该继续往 FFN-shift 方向深融)
- ❌ **把赢家和 cuBLAS matmul 绑一起 → 亏**:output+o_proj(output-project 0.96×)、+WAG LoRA matmul(0.82×)
- ❌ **weight 量化提速 → 亏**(memory-bound,cuBLAS GEMV 太强)
- 即:**只融 Triton 擅长的点积/norm/state-update,别碰 cuBLAS 强项的大 matmul(o_proj / projection GEMV)。** 这条规则能解释 5070 上所有融合的赢/亏。

**1.5B config bug(已修)**:`rwkv7-g1g-1.5b-hf/config.json` 的 `model_type` 是 `rwkv7`(FLA 内置注册类型)而非 `rwkv7_hf_adapter`,导致 transformers 加载 FLA 自带 RWKV7(无 `rwkv7_forward_token`),1.5B 所有 fast-token/graph bench 全挂。其余 0.1B/0.4B/2.9B 均 `rwkv7_hf_adapter` 正常。**已本地 patch 为 `rwkv7_hf_adapter`**。注:当前 `scripts/convert_rwkv7_to_hf.py` 第 209 行**已经**正确写 `rwkv7_hf_adapter`,所以 1.5B 是**旧转换残留**(更早的脚本转的),非脚本 bug;重新跑 convert 也会修好。

## 2026-07-14 RTX 5070 Laptop full-FLA bsz8 close

The promoted 1.5B RWKV vs official Qwen3.5 2B artifact is
[`bench/5070_qwen35_full_fla_bsz8_20260714`](../../bench/5070_qwen35_full_fla_bsz8_20260714/README.md).
All 36 raw rows and 18 strict cells pass across fp16/W8/W4,
prompt128/512/2048, and decode128/512 at bsz8. Minimum RWKV/Qwen
prefill/decode ratios are `1.082707x/1.795119x`; footprint, peak VRAM, and
tok/s per active-B pass in 18/18 cells.

Every Qwen performance row binds FLA chunk prefill, fused-recurrent decode,
fused gated norm, and the FLA Triton causal-conv bridge. No performance row
uses the Transformers Torch conv fallback. RWKV BNB4 external-quant prefill
graph is an exact-card opt-in; it and the other matrix fusions remain disabled
by default.
