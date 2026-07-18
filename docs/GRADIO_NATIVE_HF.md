# 在官方 RWKV-Gradio-3 中使用 Native HF 后端

本教程把 `BlinkDL/RWKV-Gradio-3` 的模型层替换为本仓库的
`NativeRWKV7ForCausalLM`，保留原网页、tokenizer、采样和批量生成逻辑。它适合本机
验证和演示，不是 vLLM/SGLang 或生产服务框架。

## 1. 前置条件和支持环境

- Linux、NVIDIA CUDA GPU，以及已经转换完成的 RWKV-7 HF 模型目录；
- 本仓库已使用 `python -m pip install -e ".[cuda]"` 安装；
- Git、约 2 GB 环境空间，以及模型本身需要的显存和磁盘；
- 当前精确证据是 RTX 5090、FP16、官方 g1h 7.2B、Space commit `cc57df4`。

首次接入请用 0.4B/1.5B 模型确认页面，不要直接以大模型排查环境。模型转换见
[`USER_GUIDE_ZH.md`](USER_GUIDE_ZH.md)。

## 2. 最小安全模型和输入

准备一个可信的本地 HF 模型目录，并使用短 prompt、B1、32 个输出 token：

```bash
export ADAPTER=/absolute/path/to/hf-adapter
export MODEL=/absolute/path/to/rwkv7-g1d-0.4b-hf
export SPACE=/absolute/path/to/RWKV-Gradio-3-native-hf
```

`MODEL` 必须包含 `config.json`、tokenizer 文件和完整权重，不能填单个 `.pth`。

## 3. 可直接复制的安装和启动命令

使用独立 Space clone，不修改原目录：

```bash
git clone https://huggingface.co/spaces/BlinkDL/RWKV-Gradio-3 "$SPACE"
git -C "$SPACE" checkout cc57df4
git -C "$SPACE" apply "$ADAPTER/examples/gradio/rwkv-gradio-3-native-hf.patch"
cp "$ADAPTER/examples/gradio/native_hf_v3a_compat.py" "$SPACE/"

python -m pip install -e "$ADAPTER[cuda]"
python -m pip install -r "$SPACE/requirements.txt"

cd "$SPACE"
APP3_BACKEND=native_hf \
APP3_HF_MODEL_PATH="$MODEL" \
APP3_MODEL_TITLE="RWKV-7 Native HF" \
APP3_DTYPE=float16 \
RWKV7_NATIVE_MODEL_BACKEND=native_graph \
python app.py
```

浏览器打开终端打印的 Gradio 地址。普通用户先不要设置任何
`RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN*` 实验变量；5090 sparse FFN 仍是调优探针。

## 4. 精确且可观察的通过标准

以下条件必须全部满足：

1. 进程没有 import、模型形状、CUDA graph 或 OOM 错误；
2. 页面标题显示 `RWKV-7 Native HF`，提交 prompt 后出现新文本；
3. 性能标签包含 `Output ... token/s @ bsz 1`；
4. 再用页面 Batch Size 选择 B8，八条输出均出现且进程保持存活；
5. 切回 B1 后还能继续生成，证明 B1/B8 cache 和 graph 可以重复使用。

需要记录速度时，预热后每个 batch 至少重复一次，并同时保存页面截图、标签和
`nvidia-smi` 进程显存。真实 5090 截图和原始结果见
[`5090_native_hf_gradio_train_temp_20260718`](../bench/5090_native_hf_gradio_train_temp_20260718/README.md)。

## 5. 失败恢复和当前限制

- `No module named rwkv7_hf`：重新在当前虚拟环境执行
  `python -m pip install -e "$ADAPTER[cuda]"`。
- 模型目录错误：确认 `APP3_HF_MODEL_PATH/config.json` 存在，并先运行普通
  `examples/generate.py`。
- CUDA OOM：停止进程，改用较小模型、B1 和短输出；不要同时保留多个 Gradio
  后端进程。
- 想恢复官方 Space：在独立 clone 中运行
  `git apply -R "$ADAPTER/examples/gradio/rwkv-gradio-3-native-hf.patch"`，删除
  `native_hf_v3a_compat.py`，再按官方方式启动。

5090 7.2B 实测中，Native HF sparse 最好 B1/B8 为 `95.2/651.7 tok/s`，官方 v3a
为 `138.8/841.7 tok/s`；Native 进程显存也更高。最快 sparse 直测的 B8 greedy
完整匹配为 `6/8`，因此 sparse 和 shared-pack 均保持默认关闭。网页生成通过只证明
接口和局部输出，不证明模型质量、长期稳定性或性能领先。

## 6. 让 AI 执行

AI 操作只从唯一入口 [`AI_ASSISTED_SETUP.md`](AI_ASSISTED_SETUP.md) 选择
`TASK_ID=gradio-native-hf`。专题教程不另存提示词，AI 必须按统一模板报告命令、
退出码、页面结果、截图、速度、显存和未通过门槛。
