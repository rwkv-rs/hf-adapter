# Apple Silicon 教学：MPS、MLX 和 CoreML

Apple 用户可以选择三层路线：

- **MPS**：通过原生无 FLA 模型使用标准 Transformers API；
- **MLX**：Apple 原生循环推理、会话、动态批处理和 packed W8/W4；
- **CoreML**：面向 macOS/iOS runtime 的部署导出原型。

![MPS 兼容、MLX serving 和 CoreML 部署路线](assets/tutorials/10-apple-deployment.png)

需要 Apple Silicon Mac、Python 3.10+ 和检查通过的 HF 模型。第一次用 0.1B 或
0.4B。Intel Mac 不是 MLX 目标。

## 1. 使用 MPS 做标准 HF 生成

```bash
python -m pip install -e .
python examples/check_environment.py --model MODEL
python examples/generate.py --model MODEL --prompt "Hello" \
  --device mps --backend native --dtype fp16 --max-new-tokens 8
```

环境检查必须打印 `RESULT: READY`；生成必须退出 0 并打印新文本。如果 fp16 在
某个模型上失败，先用最小模型和 `--dtype fp32` 重试，并同时报告两个结果，不能
静默换精度。

同一路线支持 PEFT、Trainer、SFT、DPO、GRPO 兼容 smoke：

```bash
python -m pip install -e ".[train]"
python tests/test_hf_training_smoke.py --model MODEL \
  --device mps --train-dtype fp32 --max-steps 1 --backend both
python tests/test_hf_rl_training_smoke.py --model MODEL \
  --device mps --train-dtype fp32 --max-steps 1 --backend both
```

两条命令都必须以 `PASS` 结束。Apple 训练行只代表兼容，不能写成高吞吐生产
训练推荐。

## 2. 把 HF safetensors 转成 MLX 目录

安装 MLX，导出全部 tensor 和 tokenizer/config 元数据：

```bash
python -m pip install -e ".[mlx]"
PYTHONPATH=. python scripts/convert_hf_to_mlx.py MODEL MLX_MODEL \
  --dtype fp16 --copy-metadata --require-mlx
```

命令必须退出 0，输出目录必须包含 MLX 权重和模型/tokenizer 元数据。
`--max-tensors` 和 `--include` 只用于诊断，普通生成不能使用不完整导出。

## 3. MLX 文本生成和可复用会话

一次性生成：

```bash
PYTHONPATH=. python scripts/mlx_generate.py MLX_MODEL \
  --prompt "User: Hello. Assistant:" --max-new-tokens 8 \
  --dtype fp16 --wkv-backend auto --require-mlx
```

JSON 必须包含 `status: pass`、有限输出、生成 token 和内存/时间遥测。

prefill 一次、分两段 decode：

```bash
PYTHONPATH=. python scripts/mlx_session_smoke.py MLX_MODEL \
  --prompt "User: Hello. Assistant:" --step-sizes 4,4 \
  --dtype fp16 --wkv-backend auto --require-mlx
```

session token/text 必须与 one-shot greedy 一致，并保持 `seen_tokens`。

交错运行两个独立 session：

```bash
PYTHONPATH=. python scripts/mlx_session_batch_smoke.py MLX_MODEL \
  --prompt "User: Alpha. Assistant:" \
  --prompt "User: Beta. Assistant:" \
  --rounds 2,2 --session-backend auto --dtype fp16 --require-mlx
```

每个 session 都必须匹配各自 one-shot。新 backend 晋升前要和 sequential 比较：

```bash
PYTHONPATH=. python scripts/mlx_session_batch_smoke.py MLX_MODEL \
  --prompt "User: Alpha. Assistant:" --prompt "User: Beta. Assistant:" \
  --rounds 2,2 --session-backend auto \
  --compare-session-backend sequential --require-session-backend-match \
  --dtype fp16 --require-mlx
```

## 4. Packed MLX W8/W4

W8：

```bash
PYTHONPATH=. python scripts/mlx_generate.py MLX_MODEL --prompt "Hello" \
  --max-new-tokens 8 --dtype fp16 --quantization mm8 \
  --quant-backend auto --wkv-backend auto --require-mlx
```

W4：

```bash
PYTHONPATH=. python scripts/mlx_generate.py MLX_MODEL --prompt "Hello" \
  --max-new-tokens 8 --dtype fp16 --quantization mm4 \
  --quant-backend auto --quant-profile uniform --quant-group-size 64 \
  --wkv-backend auto --require-mlx
```

同一 prompt/dtype 再用 `--quantization none` 跑 baseline。只有 quant 峰值更低且
greedy/logit 质量通过，才能写省内存；只有同 shape 配对时间通过，才能写更快。
已发布 M5 行见 [`hardware/APPLE_PRODUCTION_CLOSE.md`](hardware/APPLE_PRODUCTION_CLOSE.md)，
不能外推到全部 M 系列和 shape。

## 5. 动态批处理和 prefix-state cache

运行真实模型验收：

```bash
PYTHONPATH=. python scripts/mlx_dynamic_serving_bench.py \
  --models MLX_MODEL --dtype fp16 --quantization mm4 \
  --quant-backend auto --wkv-backend auto --max-batch-size 4 \
  --results apple-dynamic.jsonl --fail-on-gate
```

命令必须退出 0、写入 pass 行、保持每个请求的 greedy 结果并输出 cache/batch
遥测。它验证测试 workload 的循环状态选择、ragged batching 和有界 prefix cache；
HTTP 服务和请求准入不属于这个脚本。

## 6. MLX 投机验证

MLX runtime 提供单条和 batch greedy 的 target/draft 验证。真实 draft 实验前，
先运行同模型和拒绝回放正确性套件：

```bash
PYTHONPATH=. python -m pytest tests/test_mlx_speculative.py -q
```

在 MLX/Metal 机器上，这些测试必须真实执行，并在 accept/replay 路径保持精确
target greedy token。较小 draft 只有在包含 draft prefill 的配对总时间快于
target-only 时才有实际价值。

## 7. CoreML 计划和导出

不安装 CoreMLTools，先创建部署计划：

```bash
PYTHONPATH=. python scripts/export_rwkv7_coreml.py MODEL coreml-plan \
  --export-kind stateful-plan --state-mode wkv-coreml \
  --deployment-target macOS15 --quantization none --dry-run
```

命令必须退出 0 并写出 manifest/plan。dry run 不是可用 CoreML package。

在兼容 Mac 上安装 CoreMLTools 并请求真实 stateful export：

```bash
python -m pip install coremltools
PYTHONPATH=. python scripts/export_rwkv7_coreml.py MODEL coreml-output \
  --export-kind stateful-multifunction --state-mode wkv-coreml \
  --prefill-seq-length 16 --sample-seq-length 16 \
  --compute-units cpu-and-ne --deployment-target macOS15 \
  --quantization int8 --require-coremltools
```

必须生成 package，并在目标设备验证 runtime parity。INT4 质量和确认 ANE 放置
仍未完成，不能从“导出成功”推导出来。

## 8. 交给 AI 执行

统一使用 [`AI_ASSISTED_SETUP.md`](AI_ASSISTED_SETUP.md) 的完整任务模板，选择
“Apple MPS/MLX”或“CoreML”。本页不再维护第二套 AI 指令。
