# RWKV-7 HF 适配下一步

## 已完成：阶段 1 / wrapper 可用版

- 官方 `.pth` -> HF `model.safetensors`
- HF `config.json` + `generation_config.json`
- remote-code wrapper：`AutoConfig` / `AutoModelForCausalLM`
- RWKV trie slow tokenizer：`AutoTokenizer`
- 转换脚本已从权重 shape 推断 layer count / hidden size / head_dim / value_dim / rank dims，并新增离线 shape 测试，避免后续 0.4B+ 尺寸继续沿用 0.1B hardcode。
- `generate(use_cache=True)` 跑通
- PEFT LoRA forward/loss/backward 跑通
- HF Trainer / TRL SFTTrainer 1-step LoRA smoke 跑通
- 官方 `rwkv` pip logits 对齐：top5 一致，fp16 cosine≈0.999996

模型目录：

```text
/home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf
```

适配项目：

```text
/home/data/wangyue/projects/rwkv7-hf-adapter
```

## 阶段 2：把 wrapper 做到更完整

已新增并在 V100 跑通：

- `tests/test_official_alignment.py`：官方 `rwkv` vs HF logits + greedy 64 token 对齐。
- `tests/test_reload_roundtrip.py`：`save_pretrained` / reload roundtrip。
- `tests/test_fast_cache.py`：轻量 `RWKV7StateCache` 与 FLA 默认 cache 的 prefill/decode 等价测试。
- `tests/test_fast_decode_api.py`：`rwkv7_forward_token` batched one-token decode API 和 `rwkv7_forward_one` bsz=1 兼容入口与 HF recurrent forward 的等价测试。
- `tests/test_batch_cache.py`：bsz=1/2/4 repeated prompt cache/layout smoke，覆盖批量 recurrent state。
- `tests/test_dynamic_batch_cache.py`：heterogeneous prompts + cache `select_batch` reorder/drop/compact、`detach`、CPU offload/restore 后继续 decode，对比逐条 independent states，覆盖 dynamic batching state 管理风险。
- `tests/test_chunked_prefill.py`：full prefill vs `rwkv7_prefill_chunks` logits/cache/decode 一致性，覆盖 serving chunked prefill 风险。
- `tests/test_hf_training_smoke.py`：HF Trainer / TRL SFTTrainer 1-step LoRA smoke。
- `bench/bench_decode_breakdown.py`：decode 瓶颈拆分。
- `bench/bench_batch_sweep.py`：bsz=1/2/4/8 serving-style prefill/decode sweep，记录 total/per-seq throughput。
- `bench/bench_dynamic_batch.py`：模拟 active batch reorder/drop，记录 dynamic batching 相关 total decoded tok/s。
- `bench/bench_chunked_prefill.py`：记录 full vs chunked prefill 的 logits/cache correctness、throughput 和 peak VRAM tradeoff。
- `bench/bench_decode_micro.py`：稳定记录 HF forward decode、fast token API、`lm_head`、argmax、embedding、empty loop 等 micro timing。
- `bench/bench_forward_fast_path.py`：正式记录普通 HF cached one-token `forward()` 在 `RWKV7_FAST_FORWARD=1` 下的 production-facing 性能，并和 reference forward / direct fast-token 做速度与正确性 gate。
- `bench/bench_generate_fast_path.py`：正式记录顶层 `model.generate(..., use_cache=True)` 在 `RWKV7_FAST_FORWARD=1` 下的 production-facing 性能，并 gate bsz>=2、greedy 输出一致性和端到端生成速度提升。
- `bench/bench_fast_token_warmup.py`：正式记录 `rwkv7_warmup_fast_token()` serving preflight，gate bsz=1/2/4/8 的 native-graph capture 是否提前完成，并通过 `rwkv7_native_graph_cache_batch_sizes()` 验证 graph runner LRU。
- `bench/bench_native_graph_overhead.py`：正式记录 native-graph replay 周边的 cache copy / token copy / graph replay / cache bind 耗时，并 gate public API tok/s、runner/API diff 和 cache-copy 占比。
- `bench/bench_decode_components.py`：细分 fast-token layer path 的 projection/recurrent/norm/FFN/top layer 耗时，用于决定下一步 fusion 目标。
- `bench/bench_projection_lora.py`：专项测 attention projection/LoRA 子模块和简单 PyTorch bmm 候选，确认下一步需要 custom fusion 而不是简单拼 bmm；新增 matrix-level profile / summary / fused_kernel_plan，记录矩阵 shape、FLOPs、fp16/int8/int4 权重体积、第一 fp16 fusion target 和后续 native quant 候选。
- `bench/bench_native_decode.py`：正式记录 `rwkv7_hf.native_jit` 的 native JIT / CUDA graph decode 结果，用作下一轮 fast-token integration 的性能上限参考。
- `bench/analyze_results.py`：从 `bench/results.jsonl` 输出 target/gap report，直接列出 decode/memory ratio、缺失 benchmark axis 和下一步优化焦点。
- `bench/check_results.py`：把 JSONL 结果变成可执行 gate；默认 regression gate 当前通过，`--target` gate 在 decode 达到 0.9x official 前预期失败。
- `bench/bench_speed.py` 已改成 serving-style prefill：`use_cache=True + logits_to_keep=1`，并可用 `--hf-decode-api rwkv7_forward_token` 测快 decode API。
- `bench/profile_decode.py`：单 token decode profiler。
- `scripts/convert_rwkv7_to_hf.py` 新增 `--no-fuse-norm`，作为当前 V100 推理推荐配置。
- remote config 改为唯一 `rwkv7_hf_adapter` model_type，避免 Transformers 环境中已注册的 FLA `rwkv7` 本地类绕过本仓库 wrapper。

当前 V100 结论：

- correctness：`fuse_norm=false` 下 top5/argmax/cosine/greedy64 均通过，fp16 max_abs 约 0.072。
- memory：HF 406.4 MB vs official 406.2 MB，0.1B serving path 已基本持平。
- speed：`fuse_norm=false` + `RWKV7StateCache` 下标准 remote-code HF decode 约 41.2 tok/s；FLA `rwkv7_forward_token` V100 bsz=1 约 59.2 tok/s；`RWKV7_FAST_TOKEN_BACKEND=native_jit` 的 HF fast-token bsz=1 已到 92.1 tok/s，和 official 92.1 tok/s 持平，target gate 已通过；`RWKV7_FAST_TOKEN_BACKEND=native_graph` 已把 CUDA graph 接入 HF `rwkv7_forward_token` 的固定 bsz=1/2/4/8，speed_mem bsz=1 达到 255.5 tok/s，batch sweep 达到 253.9/434.3/852.6/1539.1 aggregate tok/s，并可用 `rwkv7_warmup_fast_token()` 在 1.389s 内提前捕获 bsz=1/2/4/8 graph runner；native-graph steady decode 已跳过 graph-buffer 自拷贝，overhead rows 覆盖 bsz=1/2/4/8，public API 分别 255.1/449.8/857.2/1548.1 aggregate tok/s、runner/API diff 0.0、cache-copy 占比最高 0.052、graph-runner cache hit rate 0.9737；`RWKV7_FAST_TOKEN_BACKEND=auto` 现在会按 active batch 自动选择 native_graph/native_jit/FLA 并在 benchmark 里记录 effective backend；`RWKV7_FAST_FORWARD=1` 默认让普通 eval/no-grad HF cached one-token `forward`/`generate` 也自动走 fast-token path，短 V100 microbench 里普通 HF forward 从约 40 tok/s 到约 251 tok/s，正式 generate gate 里 bsz=2 `model.generate` 从 75.3 tok/s 到 303.5 tok/s 且 32/32 greedy tokens 全一致，benchmark baseline 用 `RWKV7_FAST_FORWARD=0` 保持可比；dynamic batch native_graph 通过显式 `select_batch` reorder/drop 达到 1209.3 total tok/s；chunked prefill bsz=2 prompt=512 在 chunk 64/128/256 下显存约为 full prefill 的 0.598x/0.616x/0.633x，速度约为 0.125x/0.252x/0.499x；component bench 显示 `attn_linears_lora` 最大，约 9.87ms/token。
- quant：已新增 bitsandbytes 8bit/4bit smoke 和 benchmark，并让 HF/bnb 默认 `memory` policy 跳过 RWKV 小 rank LoRA projection（`.*_lora.lora.[02]` 保持 dense model dtype），避免 V100 上低效小矩阵量化 kernel；V100 0.1B 上 model footprint 从 fp16 `364.4MB` 降到 8bit `283.4MB`、4bit `242.9MB`，selected decode 为 8bit `16.3 tok/s`、4bit `32.6 tok/s`，greedy next token 保持一致；新增 `decode_hot` policy 额外保持 attention r/k/v/o projection dense，4bit cached decode smoke 约 `36.8 tok/s`、footprint 约 `283MB`，是速度/显存折中探针；量化 analyzer 已增加 `quantization_model_sweep`，避免 0.4B/更大模型量化 rows 覆盖 0.1B canonical gate；V100 0.4B fp16 footprint/decode 为 `859.8MB`/`107.0 tok/s`，8bit memory 为 `571.8MB`/`8.4 tok/s`，4bit memory 为 `427.8MB`/`16.3 tok/s`，0.4B `decode_hot` 提升到 8bit `13.7 tok/s`、4bit `19.6 tok/s`；显存下降成立但仍低于 fp16 native-graph，所以仍需自定义/融合量化 serving path 才能满足“不比 16bit 慢”的目标；`bench_native_quant_gemv.py` 已新增 RWKV-native row-wise W8 pack + fused dequant-GEMV prototype，V100 sampled attn/FFN 权重 footprint 为 `0.502x` fp16，`min_cosine=0.9999172`，但首版 Triton kernel 速度只有 `0.3816x` fp16 cuBLAS（`0.05409ms` vs `0.02064ms`），说明 native W8 方向正确但 kernel 还需要 tensor-core-aware / 更深融合优化；`bench_native_quant_w4_gemv.py` 已新增 RWKV-native row-wise W4 pack + fused nibble-unpack/dequant-GEMV prototype，V100 sampled footprint 为 `0.252x` fp16，`min_cosine=0.9802878`、`max_abs_diff=0.9287109`，首版 Triton kernel 速度 `0.359x` fp16 cuBLAS（`0.05773ms` vs `0.02072ms`），因此 W4 功能/telemetry 已补齐但仍未达到速度目标；`bench_native_quant_rkv.py` 进一步新增 W8 fused R/K/V quant projection prototype，V100 上相对三次 separate W8 GEMV 提升 `1.7628x`（`0.08878ms` vs `0.1565ms`），footprint `0.5026x` fp16，且 fused/separate 输出 exact，但仍只有 `0.7847x` fp16 cuBLAS，说明 launch/group fusion 能明显缩小 native quant gap，下一步需要把 LoRA/更多 projection 一起融合；`bench_native_quant_w4_rkv.py` 同步新增 W4 fused R/K/V quant projection prototype，V100 上相对三次 separate W4 GEMV 提升 `1.7958x`（`0.0912ms` vs `0.16378ms`），footprint `0.2526x` fp16，fused/separate 输出 exact，当前为 `0.7795x` fp16、`min_cosine_fp16_vs_fused=0.9750665`。
- fused backend：当前新增 `FUSED_BACKEND.md` 和 analyzer 的 `fused_backend_targets`，把 Albatross 追赶目标固定为 P1 `>=0.55x`、P2 `>=0.75x`、P3 `>=0.90x` decode ratio，并把 W8/W4 目标固定为 footprint 下降且 decode `>=1.0x` fp16 reference；`bench_fused_projection.py` 已提供第一个 Triton R/K/V GEMV prototype，V100 上 correctness 通过但速度仅 `0.8429x` 当前三路 linear；`bench_fused_wa_lora.py` 新增 W/A LoRA 两 kernel fusion probe，V100 上 correctness 通过但速度 `0.8601x` 当前 W/A LoRA（`0.16883ms` vs `0.14521ms`），说明只做 W/A 两段 grouped kernel 不够；`bench_fused_wag_lora.py` 进一步把 W/A/G LoRA grouped 到两 kernel，支持 W/A rank 64 + G rank 128，V100 稳定 row correctness 通过且达到 `1.0985x` 当前 W/A/G LoRA（`0.26336ms` vs `0.28931ms`），这是第一个 LoRA grouping 正收益子 kernel；`bench_fused_rkv_wag_projection.py` 又把 R/K/V dense projection 与 W/A/G LoRA down 合进一个 launch、W/A/G up 一个 launch，V100 稳定 row correctness 通过且小幅正收益 `1.0103x` 当前 R/K/V+W/A/G（`0.31102ms` vs `0.31422ms`），说明跨 dense+LoRA 的 launch grouping 可行但收益还太小，仍需继续融合 state/update/output 或优化 dense math；`bench_fused_shift_mix.py` 又验证了单独融合六路 attention shift-mix correctness exact，但 V100 只有 `0.7715x` 当前 torch pointwise ops；`bench_fused_recurrent.py` 现在验证了 rank-1 recurrent state update + readout fused kernel，V100 上达到 `2.7931x` 当前 torch recurrent expression，`out_min_cosine=0.9999998`、`out_max_abs_diff=0.0234375`。`RWKV7_NATIVE_GRAPH_FUSED_RECURRENT=1` 已接入 native_graph capture，cache key 会区分开关，V100 integration row 首步 logits exact、greedy `32/32`，但 end-to-end 只有 `1.0033x`（`4.2878ms` vs `4.3018ms`），说明 isolated recurrent 已快、完整 token 仍被 projection/LoRA 等更大部分主导。结论更新为：recurrent fused path 保持 opt-in，下一步继续做 projection/LoRA/recurrent 更深融合和 native W8/W4 pack+dequant-GEMV。
- larger model：真实 0.4B / 1.5B / 2.9B / 7.2B / 13.3B `.pth` 已下载、SHA256 校验并转换到 HF；V100 load/forward/generate smoke 已入 `bench/results.jsonl`。0.4B 配置 hidden=1024、layers=24、head_dim=64、value_dim=1024，生成 4 个 token，峰值 VRAM `1124.5MB`；1.5B 配置 hidden=2048、layers=24、head_dim=64、value_dim=2048，生成 2 个 token，峰值 VRAM `3178.6MB`；2.9B 配置 hidden=2560、layers=32、head_dim=64、value_dim=2560，生成 2 个 token，峰值 VRAM `5888.0MB`；7.2B 配置 hidden=4096、layers=32、head_dim=64、value_dim=4096，生成 2 个 token，峰值 VRAM `13997.8MB`；13.3B 配置 hidden=4096、layers=61、head_dim=64、value_dim=4096，生成 2 个 token，峰值 VRAM `25575.6MB`，backend 为 `native_jit`；其余四者 generation fast path 均解析为 native_graph。
- multi-GPU：HF `device_map` 在 2 x V100 上手动按 layer 6 切分 0.1B，`RWKV7_FAST_FORWARD=1` 下会自动跳过单卡 fast-token shortcut，cached `generate()` 输出和单卡 greedy tail 一致，作为 PP 方向 smoke。
- native decode prototype：`rwkv7_hf.native_jit` 在 V100 0.1B 上已验证，logit cosine≈1.00000024、graph-vs-JIT greedy 16/16 一致；native JIT 约 103.5 tok/s，native CUDA graph 约 254.3 tok/s（2.76x official）。这个 reduced-launch 路径已接进 HF 固定 batch 和 dynamic active-batch serving API，graph runner 已改成按 active batch size 的 per-model LRU cache，并提供 `rwkv7_clear_native_graph_cache()` 释放缓存；`rwkv7_hf.native_model` 仍标为 experimental PyTorch/FLA-free 底座，已补 bsz>1 forward、batched incremental cache 对齐测试，并把 cached decode 接到 optional native_jit（可用 `RWKV7_NATIVE_MODEL_JIT=0` 关闭）；同时 native CausalLM 已补 `labels` 序列 loss、`get_input_embeddings()` 和 `get_output_embeddings()`，用 CPU tiny-model 单测覆盖 PEFT/Trainer 需要的基础训练接口；V100 0.1B fp32 telemetry 已入 `bench/results.jsonl`，forward min cosine `0.99999976`、batch-forward min cosine `0.9999994`、cached decode `3/3`、greedy generate `16/16`、incremental cache=True、backend=`native_jit`；V100 fp32 smoke 从 eager `61.7 tok/s` 提到 native_jit `115.5 tok/s`；下一步是更多模型、更多 GPU 和量化 fast path 验证。
- profiler：`fuse_norm=true` 的 FLA `LayerNormFunction` CPU 开销很大，native norm 把 norm CPU total 从约 54.8ms/6tok 降到约 6.6ms/6tok。
- breakdown：argmax 开销约等于 0，`chunk` 和 `fused_recurrent` 单 token decode 基本一样，剩余瓶颈在 HF/FLA model+state/cache+小 kernel launch 路径。

下一步继续补：

1. 支持全部已发布尺寸的配置推断和转换：0.4B / 1.5B / 2.9B / 7.2B / 13.3B。
   - 已把 converter 的 head_dim/value_dim/rank dims 改成权重 shape 推断，并用 `tests/test_convert_config.py` 覆盖非 64 head_dim、value_dim 列表和错误 shape。
   - 真实 0.4B / 1.5B / 2.9B / 7.2B / 13.3B `.pth` 已完成转换、load、forward、generate smoke，并加入 regression gate；下一步继续 official alignment、speed smoke 和更多 GPU。
2. 增加批量转换脚本和 SHA256 manifest。
   - 已新增 `scripts/batch_convert_rwkv7_to_hf.py`，支持 `--input-dir` / `--inputs`、dry-run、跳过已存在输出、追加 manifest，并记录 size / sha256 / 转换选项 / command / status。
   - 已新增 `tests/test_batch_convert_manifest.py` 覆盖 dry-run manifest、append manifest、missing input error。
3. 补 HF behavior：
   - 已补 `resize_token_embeddings` 固定词表保护
   - 已新增 `tests/test_hf_api_contract.py` 覆盖 `prepare_inputs_for_generation`、beam cache reorder、`gradient_checkpointing_enable`
   - 继续补更完整 Transformers 原生 test suite
4. 训练路径：
   - 已跑通 PEFT LoRA backward、HF Trainer 1-step、TRL `SFTTrainer` 1-step
   - 已新增 TRL `DPOTrainer` / `GRPOTrainer` LoRA 1-step smoke 脚本，并补 `configs/deepspeed/zero2.json` / `zero3.json` 作为 HF Trainer ZeRO-2/3 预设
   - 已把 HF Trainer / TRL SFT / DPO / GRPO smoke 扩大到可配置 batch size、gradient accumulation，并校验 LoRA/trainable 参数更新；V100 fp32 已通过 Trainer/SFT batch=2 grad_accum=2 和 DPO/GRPO batch=2。fp16 batch/gradaccum 曾暴露 grad_norm NaN 且参数未更新，因此 smoke 默认 fp32、保留 `--train-dtype fp16` 显式检查路径；训练 smoke 现可写入 `training_smoke` JSONL telemetry，`bench/analyze_results.py` / `bench/check_results.py` 会汇总并校验 trainable delta；继续扩大到真实 SFT 小数据和训练吞吐 benchmark
   - 当前新增 `tests/test_deepspeed_training_smoke.py`，把 ZeRO-2/ZeRO-3 从“配置存在”推进到“HF Trainer + PEFT LoRA 可执行 smoke”：会校验有限 loss、LoRA/trainable 参数实际更新，并写入 `deepspeed_training_smoke` telemetry；无卡/无 DeepSpeed 环境可用 `--optional` 记录 skip，真实 V100 双卡 pass rows 等 GPU 恢复后补入。
   - 当前 smoke 明确 `TORCHDYNAMO_DISABLE=1`，并关闭 `use_l2warp` 避免 Trainer loss 原地缩放与 L2Wrap backward 冲突
5. 性能路径：
   - 继续 profile 单 token decode
   - `RWKV7StateCache` 已减少 generic CacheLayer 开销，并提供 `select_batch` / `batch_select` / `clone` / `detach` / `to` / `get_batch_size` / `rwkv7_cache_metrics()`，服务动态 batching reorder/drop/compact、CPU offload/restore 和 cache telemetry
   - 当前范围只做 HF adapter：Transformers/PEFT/TRL/Trainer、HF state cache、dynamic batch、chunked prefill、HF 多卡方向、量化和 HF-compatible speculative decoding；vLLM/SGLang 不作为当前交付线。
   - 已新增 `rwkv7_forward_token` batched one-token fast decode entrypoint，并保留 `rwkv7_forward_one` bsz=1 兼容入口
   - 已新增 batch cache/sweep、dynamic-batch reorder/drop/compact harness、chunked prefill harness、decode microbench、decode component bench、projection/LoRA bench、larger-model smoke、gap analyzer 和 result gate；V100 bundle 已跑通，native_jit fast-token 已支持 bsz=1/2/4/8 和 dynamic batching，native_graph 已支持固定 bsz=1/2/4/8 和 dynamic active-batch serving，auto backend 已能按可用能力选择 native_graph/native_jit/FLA，并已接入普通 HF `forward`/`generate` 的 one-token 推理路径；chunked prefill 已支持 logits/cache 对齐和显存/速度记录；下一轮重点是更多 GPU、13.3B official-alignment/speed sweep 和量化 fast path
   - native JIT block-step 已接入 `rwkv7_forward_token`，支持 bsz=1/2/4/8 和 dynamic reorder/drop；native graph replay 已接入 `rwkv7_forward_token` 固定 bsz=1/2/4/8 和 dynamic active-batch 场景；graph cache 管理已补 per-model LRU、清理接口、batch-size inspection、hit-rate telemetry、`rwkv7_warmup_fast_token()` 预热接口和 graph-buffer 自拷贝跳过；HF `device_map` 2 卡 generate smoke 已通过；0.4B / 1.5B / 2.9B / 7.2B / 13.3B 已通过 larger-model smoke；下一步做 13.3B official-alignment/speed sweep、更多 GPU 和量化 serving fast path
   - 已新增 HF-only speculative decoding helper 和真实小模 draft benchmark：`rwkv7_speculative_generate()` 用 RWKV/HF draft model 提案，target model 以 block forward 验证并输出 acceptance telemetry；mismatch 后已改成 cached-prefix resync；V100 0.1B draft -> 0.4B target 已匹配 target greedy 8/8 new tokens，acceptance 7/9=0.778，resync 只重放 3 token 而不是 11 token full-prefix，短 benchmark 达到 2.1079x target-greedy speedup；下一步优化 draft 选择和长上下文/多 bsz 验证
   - 已新增 bitsandbytes 8bit/4bit 加载、生成、benchmark，并把量化 cached decode 接入 HF fast-forward 的 FLA fallback；下一步需要把量化权重接到 native/fused int8/int4 projection，解决 generic bnb decode 仍慢的问题

## 阶段 3：Transformers 原生 PR 方向

需要从 FLA wrapper 迁移为 Transformers 原生 backend：

```text
src/transformers/models/rwkv7/
  configuration_rwkv7.py
  modeling_rwkv7.py
  tokenization_rwkv7.py
  convert_rwkv7_original_to_hf.py
  __init__.py
```

关键要求：

- 不依赖 `fla` 包作为必需依赖。
- 先提供 pure PyTorch/reference recurrent 实现保证 CPU/GPU 正确性。
- 再接可选 CUDA/Triton kernels 做性能路径。
- HF tests：modeling common、generation、tokenizer、PEFT/Trainer smoke。
- 文档和 model card：state cache 不同于 KV cache。

## 其他分支审计

- `origin/wangyue/native-transformers` 有另一条 native 适配线，最新包含 H2 JIT decode 设想；但该分支基于旧主线，会回退/删除当前已合并的 HF criteria、TTFT/TPOT、量化 skip policy 等内容，不能整分支合并。可后续单独移植其中有价值的 native JIT decode 思路。
- `origin/wangyue/50series-native-decode` 更旧，会删除大量 bench/test/config，不适合直接合并；只作为历史参考。

## 当前 blocker

- FLA 源对 V100 的 chunk/backward 编译不稳定，已临时用 `fused_recurrent + TORCHDYNAMO_DISABLE=1` 跑 PEFT。
- FLA `chunk` prefill 在 V100 首次编译耗时很长，性能还没达到官方 `rwkv` 路径。
- 真正悬赏级别需要去 FLA 依赖、补 Transformers test suite、以及大模型/多卡/量化验证。
